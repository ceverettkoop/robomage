#!/usr/bin/env python3
"""
LLM-driven test harness for RoboMage card testing.

Runs the game engine with --machine --narrative so that:
  - Game narrative (casts, damage, zone changes) prints alongside binary queries
  - Binary state is decoded into human-readable text for LLM observation

Shuffling: by default libraries are shuffled with the seeded RNG (deterministic
per --seed). Pass --no-shuffle for a STACKED DECK — deck-file order = draw order,
so the first 7 cards become the opening hand. --no-shuffle is implied
automatically when --hand-a/--hand-b are given (those build a stacked temp deck).

Designed for automated testing: an LLM writes a scenario (hands + action script),
runs this harness, and reads the output to verify card behavior.

Usage examples:

    # Scripted action sequence (action indices separated by commas)
    python test_harness.py \\
        --hand-a "Mountain,Lightning Bolt" \\
        --library-a "Mountain,Island,Island,Mountain,Mountain,Mountain,Mountain" \\
        --hand-b "Forest,Grizzly Bears" \\
        --library-b "Forest,Forest,Forest,Forest,Forest,Forest,Forest" \\
        --actions "9,0,7,0,8"

    # Use scripted agent (rule-based auto-play) for both sides
    python test_harness.py \\
        --hand-a "Mountain,Lightning Bolt,Volcanic Island,Delver of Secrets" \\
        --library-a "Mountain,Island,Ponder,Lightning Bolt,Daze,Island,Mountain" \\
        --deck-b delver \\
        --scripted --max-decisions 40

    # Interactive: pause at each decision and prompt for action index
    python test_harness.py \\
        --hand-a "Swamp,Dark Ritual,Doomsday" \\
        --library-a "Swamp,Swamp,Swamp,Swamp,Swamp,Swamp,Swamp" \\
        --hand-b "Island,Island,Island" \\
        --library-b "Island,Island,Island,Island,Island,Island,Island" \\
        --interactive

    # JSON scenario file
    python test_harness.py --scenario scenario.json

Scenario JSON format:
    {
        "name": "bolt_kills_bear",
        "hand_a": ["Mountain", "Lightning Bolt"],
        "library_a": ["Mountain", "Island", "Island", ...],
        "hand_b": ["Forest", "Grizzly Bears"],
        "library_b": ["Forest", "Forest", "Forest", ...],
        "actions": [9, 0, 7, 0, 8],
        "seed": 1
    }
"""

import argparse
import json
import os
import sys
from pathlib import Path

import runner
from opponents import (make_controller, ActionListController,
                       InteractiveController, AutoPassController)

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BIN_DIR = _REPO_ROOT / "bin"
_BINARY = _BIN_DIR / "robomage"
_DECKS_DIR = _BIN_DIR / "resources" / "decks"


# ── Deck file helpers ─────────────────────────────────────────────────────────

def _card_to_deck_name(name):
    """Convert display name to deck-file name (strip apostrophes, keep rest)."""
    return name.replace("'", "").replace("\u2019", "")


_TEMP_DECKS_DIR = _DECKS_DIR / "temp"


def _make_deck_file(hand, library, label):
    """Write a .dk file into decks/temp/ with hand cards first.

    Returns the deck name (relative to decks/) for --deck-a/--deck-b args.
    The engine loads decks from resources/decks/<name>.dk.
    """
    _TEMP_DECKS_DIR.mkdir(exist_ok=True)
    name = f"temp/_test_{label}"
    path = str(_DECKS_DIR / f"{name}.dk")
    with open(path, "w") as f:
        for card in hand:
            f.write(f"1 {_card_to_deck_name(card)}\n")
        for card in library:
            f.write(f"1 {_card_to_deck_name(card)}\n")
        f.write("\nSIDEBOARD:\n")
    return name, path



# ── Main entry point ──────────────────────────────────────────────────────────

def _parse_card_list(s):
    """Parse a comma-separated card list, stripping whitespace."""
    if not s:
        return []
    return [c.strip() for c in s.split(",") if c.strip()]


def _pad_library(hand, library, min_deck_size=15):
    """Ensure library is large enough that the game doesn't immediately deck out.

    Pads with the last card in the library (or a basic land from the hand).
    """
    total = len(hand) + len(library)
    if total >= min_deck_size:
        return library
    # Pick a filler card
    filler = None
    if library:
        filler = library[-1]
    else:
        # Use a basic land from hand, or Mountain as last resort
        for card in hand:
            if card in ("Mountain", "Forest", "Island", "Swamp", "Plains"):
                filler = card
                break
        if filler is None:
            filler = "Mountain"
    padded = list(library)
    while len(hand) + len(padded) < min_deck_size:
        padded.append(filler)
    return padded


def main():
    parser = argparse.ArgumentParser(
        description="RoboMage LLM test harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scenario", help="Path to JSON scenario file")
    parser.add_argument("--hand-a", help="Player A starting hand (comma-separated card names)")
    parser.add_argument("--library-a", help="Player A library after hand (comma-separated)")
    parser.add_argument("--hand-b", help="Player B starting hand (comma-separated card names)")
    parser.add_argument("--library-b", help="Player B library after hand (comma-separated)")
    parser.add_argument("--deck-a", help="Use existing deck file for Player A (name, not path)")
    parser.add_argument("--deck-b", help="Use existing deck file for Player B (name, not path)")
    parser.add_argument("--battlefield-a", help="Cards starting on Player A's battlefield (comma-separated)")
    parser.add_argument("--battlefield-b", help="Cards starting on Player B's battlefield (comma-separated)")
    parser.add_argument("--actions", help="Comma-separated action indices to play")
    parser.add_argument("--interactive", action="store_true", help="Prompt for each action")
    parser.add_argument("--scripted", action="store_true", help="Use scripted agent")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Don't shuffle libraries — deck-file order = draw order "
                             "(first 7 cards = opening hand). Use when feeding a stacked "
                             "deck via --deck-a/--deck-b. Implied automatically when "
                             "--hand-a/--hand-b are given. Without it, libraries are "
                             "shuffled with the seeded RNG (deterministic per --seed).")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: 1, or scenario's seed if given)")
    parser.add_argument("--max-decisions", type=int, default=None,
                        help="Stop after this many decisions (default: 500, "
                             "or scenario's max_decisions if given)")
    parser.add_argument("--binary", default=str(_BINARY), help="Path to robomage binary")
    args = parser.parse_args()

    # Load scenario from JSON if provided
    scenario = {}
    if args.scenario:
        with open(args.scenario) as f:
            scenario = json.load(f)

    hand_a = _parse_card_list(args.hand_a) or scenario.get("hand_a", [])
    library_a = _parse_card_list(args.library_a) or scenario.get("library_a", [])
    hand_b = _parse_card_list(args.hand_b) or scenario.get("hand_b", [])
    library_b = _parse_card_list(args.library_b) or scenario.get("library_b", [])
    battlefield_a = _parse_card_list(args.battlefield_a) or scenario.get("battlefield_a", [])
    battlefield_b = _parse_card_list(args.battlefield_b) or scenario.get("battlefield_b", [])
    seed = args.seed if args.seed is not None else scenario.get("seed", 1)
    # Stacked-deck mode: deck-file order == draw order. Implied by hand sculpting
    # (--hand-a/--hand-b build a stacked temp deck whose first 7 cards must be the
    # opening hand) or requested explicitly via --no-shuffle (e.g. a hand-ordered
    # deck file passed through --deck-a). Otherwise libraries are shuffled.
    no_shuffle = args.no_shuffle or bool(hand_a) or bool(hand_b)
    actions_str = args.actions or scenario.get("actions")
    max_decisions = (args.max_decisions if args.max_decisions is not None
                     else scenario.get("max_decisions", 500))

    # Parse action list
    actions = None
    if actions_str:
        if isinstance(actions_str, str):
            actions = [int(x.strip()) for x in actions_str.split(",")]
        elif isinstance(actions_str, list):
            actions = [int(x) for x in actions_str]

    cleanup_paths = []
    try:
        # Determine deck names
        if args.deck_a and not hand_a:
            deck_a_name = args.deck_a
        elif hand_a:
            library_a = _pad_library(hand_a, library_a)
            deck_a_name, deck_a_file = _make_deck_file(hand_a, library_a, "a")
            cleanup_paths.append(deck_a_file)
        else:
            deck_a_name = "delver"

        if args.deck_b and not hand_b:
            deck_b_name = args.deck_b
        elif hand_b:
            library_b = _pad_library(hand_b, library_b)
            deck_b_name, deck_b_file = _make_deck_file(hand_b, library_b, "b")
            cleanup_paths.append(deck_b_file)
        else:
            deck_b_name = "delver"

        # Print scenario summary
        name = scenario.get("name", "test")
        print(f"=== TEST: {name} ===")
        if hand_a:
            print(f"Player A hand:    {', '.join(hand_a)}")
            print(f"Player A library: {', '.join(library_a)}")
        else:
            print(f"Player A deck: {deck_a_name}")
        if hand_b:
            print(f"Player B hand:    {', '.join(hand_b)}")
            print(f"Player B library: {', '.join(library_b)}")
        else:
            print(f"Player B deck: {deck_b_name}")
        if battlefield_a:
            print(f"Player A battlefield: {', '.join(battlefield_a)}")
        if battlefield_b:
            print(f"Player B battlefield: {', '.join(battlefield_b)}")
        print(f"Seed: {seed}")
        if actions:
            print(f"Actions: {actions}")
        print()

        # Pick the controller for the chosen play mode. The same controller
        # drives both seats (the action list / interactive prompts / auto-pass
        # are global, not per-side), matching the original harness behaviour.
        if actions is not None:
            controller = ActionListController(actions)
            mode_label = "Actions"
        elif args.interactive:
            controller = InteractiveController()
            mode_label = "Human"
        elif args.scripted:
            controller = make_controller("scripted")
            mode_label = "Scripted"
        else:
            controller = AutoPassController()
            mode_label = "Auto"

        # Pre-set battlefields are passed to the engine as comma-joined
        # deck-name strings (apostrophes stripped), same as the deck files.
        bf_a = ",".join(_card_to_deck_name(c) for c in battlefield_a) if battlefield_a else None
        bf_b = ",".join(_card_to_deck_name(c) for c in battlefield_b) if battlefield_b else None

        # The observation/decision loop lives in runner.run_games (shared with
        # train.py observe). test_harness only seeds the state above.
        wins, losses, _ = runner.run_games(
            controller, controller, label_a=mode_label, label_b=mode_label,
            binary_path=args.binary, deck_a=deck_a_name, deck_b=deck_b_name,
            n_games=1, seed=seed, verbose=True,
            battlefield_a=bf_a, battlefield_b=bf_b, no_shuffle=no_shuffle,
            max_decisions=max_decisions)
        winner = bool(wins or losses)
    finally:
        for p in cleanup_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    return 0 if winner else 1


if __name__ == "__main__":
    sys.exit(main())
