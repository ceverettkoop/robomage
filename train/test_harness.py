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

Who decides: the --play / --actions script (global — one entry per decision,
whichever seat is on the clock) makes every decision until it runs out; after
that, and at every decision when there is no script, each seat's --player-a /
--player-b agent decides (any opponents.make_controller spec; default 'auto' =
pass / first choice, so the game auto-advances to its end or --max-decisions).

Usage examples:

    # Semantic action specs (resolved against the live menu each decision —
    # robust to index reordering; the preferred way to script a precise line).
    python test_harness.py --format bo1 \\
        --hand-a "Mountain,Lightning Bolt" \\
        --library-a "Mountain,Island,Island,Mountain,Mountain,Mountain,Mountain" \\
        --battlefield-b "Grizzly Bears" \\
        --play "keep,keep,play:Mountain,cast:Lightning Bolt,target:Grizzly Bears@opp"

    # Seat-keyed specs: when --play drives BOTH seats, prefix a spec with "A:" or
    # "B:" to pin it to a player. When the next spec is keyed to the seat that
    # does NOT have priority, the priority holder auto-passes until the keyed seat
    # is on the clock — so you write each player's intended line and never have to
    # hand-interleave the priority-passes. (Unkeyed specs apply to whoever has
    # priority.) The engine never asks a seat whose only legal action is a pass,
    # so write a keyed pass only where that seat could do something else.
    python test_harness.py --format bo1 \\
        --hand-a "Lightning Bolt" --battlefield-a "Mountain" \\
        --battlefield-b "Grizzly Bears" \\
        --play "A:keep,B:keep,A:cast:Lightning Bolt,A:target:Grizzly Bears@opp"

    # Scripted agent (rule-based play) for both seats
    python test_harness.py --format bo1 \\
        --hand-a "Mountain,Lightning Bolt,Volcanic Island,Delver of Secrets" \\
        --library-a "Mountain,Island,Ponder,Lightning Bolt,Daze,Island,Mountain" \\
        --deck-b delver \\
        --play "keep,keep" --player-a scripted --player-b scripted --max-decisions 40

    # Script an opening, then let the scripted agent play A from there on
    python test_harness.py --format bo1 --deck-a delver --deck-b delver \\
        --play "A:keep,B:keep" --player-a scripted --max-decisions 60

    # Positional action indices (fragile — prefer --play)
    python test_harness.py --format bo1 \\
        --hand-a "Mountain,Lightning Bolt" \\
        --hand-b "Forest,Grizzly Bears" \\
        --actions "0,0,1"

    # A human at the terminal for seat A (needs a TTY — an automated agent
    # driving the harness must use --play instead).
    python test_harness.py --format bo1 \\
        --hand-a "Swamp,Dark Ritual,Doomsday" \\
        --library-a "Swamp,Swamp,Swamp,Swamp,Swamp,Swamp,Swamp" \\
        --player-a human

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
from cli_spec import HARNESS_TOOL, apply_to_parser, is_bo3, is_human_spec
from opponents import (make_controller, ActionListController,
                       HumanController, PlayController)

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BIN_DIR = _REPO_ROOT / "bin"  # resource cwd (not per-config)
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


def _merge_sideboard_deck(deck_name, label):
    """Fold a deck file's SIDEBOARD: section into its mainboard.

    Loads decks/<deck_name>.dk, sums quantities for names that appear in both
    sections, and writes the merged list as a temp deck (empty sideboard) so
    single-game fuzzing can reach sideboard-only cards. Returns (name, path)
    like _make_deck_file; the caller unlinks the temp file on exit.
    """
    src = _DECKS_DIR / f"{deck_name}.dk"
    counts = {}  # card name -> quantity, insertion-ordered
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line or line.upper().startswith("SIDEBOARD"):
                continue
            qty, _, card = line.partition(" ")
            card = card.strip()
            if not qty.isdigit() or not card:
                sys.exit(f"--merge-sideboard: unparseable line in {src}: {line!r}")
            counts[card] = counts.get(card, 0) + int(qty)
    _TEMP_DECKS_DIR.mkdir(exist_ok=True)
    name = f"temp/_merged_{label}"
    path = str(_DECKS_DIR / f"{name}.dk")
    with open(path, "w") as f:
        for card, qty in counts.items():
            f.write(f"{qty} {card}\n")
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


def _player_controller(spec):
    """(controller, label) for one harness seat's --player-a/--player-b spec.

    A 'human' seat skips its own board dump — the harness transcript already
    prints the state and menu at every decision."""
    if is_human_spec(spec):
        return HumanController(label="Human", show_state=False), "Human"
    ctrl = make_controller(spec)
    low = spec.strip().lower()
    if low in ("auto", "autopass"):
        return ctrl, "Auto"
    return ctrl, "Scripted" if low == "scripted" else spec


def _build_controllers(player_a, player_b, play_specs, actions):
    """(ctrl_a, ctrl_b, label_a, label_b) for a harness run.

    The --play / --actions script is global: ONE controller installed as both
    seats makes every decision until it runs out, then hands each decision to
    the priority seat's player. Without a script the players drive their seats
    directly. Identical seat specs share one controller instance."""
    pa, label_pa = _player_controller(player_a)
    if player_b.strip() == player_a.strip():
        pb, label_pb = pa, label_pa
    else:
        pb, label_pb = _player_controller(player_b)
    if play_specs is not None:
        script = PlayController(play_specs, players=(pa, pb))
    elif actions is not None:
        script = ActionListController(actions, players=(pa, pb))
    else:
        return pa, pb, label_pa, label_pb
    labels = tuple(script.label if lbl == "Auto" else f"{script.label}+{lbl}"
                   for lbl in (label_pa, label_pb))
    return script, script, labels[0], labels[1]


def build_parser():
    """The harness CLI, built from cli_spec's HARNESS_TOOL (the single source
    the TUI form is built from too)."""
    parser = argparse.ArgumentParser(
        description="RoboMage LLM test harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    apply_to_parser(parser, HARNESS_TOOL.subs[0])
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    bo3 = is_bo3(args)

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
    graveyard_a = _parse_card_list(args.graveyard_a) or scenario.get("graveyard_a", [])
    graveyard_b = _parse_card_list(args.graveyard_b) or scenario.get("graveyard_b", [])
    exile_a = _parse_card_list(args.exile_a) or scenario.get("exile_a", [])
    exile_b = _parse_card_list(args.exile_b) or scenario.get("exile_b", [])
    sideboard_a = _parse_card_list(args.sideboard_a) or scenario.get("sideboard_a", [])
    sideboard_b = _parse_card_list(args.sideboard_b) or scenario.get("sideboard_b", [])
    life_a = args.life_a if args.life_a is not None else scenario.get("life_a")
    life_b = args.life_b if args.life_b is not None else scenario.get("life_b")
    seed = args.seed if args.seed is not None else scenario.get("seed", 1)
    # Stacked-deck mode: deck-file order == draw order. Implied by hand sculpting
    # (--hand-a/--hand-b build a stacked temp deck whose first 7 cards must be the
    # opening hand) or requested explicitly via --no-shuffle (e.g. a hand-ordered
    # deck file passed through --deck-a). Otherwise libraries are shuffled.
    no_shuffle = args.no_shuffle or bool(hand_a) or bool(hand_b)
    actions_str = args.actions or scenario.get("actions")
    play_specs = args.play or scenario.get("play")
    # A bo3 match is up to 3 games plus sideboarding decisions, so its default
    # decision cap is 3x the single-game one (the engine's own step cap scales
    # the same way: MAX_STEPS_BO3 = 3 * MAX_STEPS in env.py).
    max_decisions = (args.max_decisions if args.max_decisions is not None
                     else scenario.get("max_decisions", 1500 if bo3 else 500))

    if args.merge_sideboard:
        if bo3:
            parser.error("--merge-sideboard requires --format bo1 (the merged "
                         "deck has no sideboard left to board from)")
        if hand_a or hand_b or library_a or library_b:
            parser.error("--merge-sideboard only applies to --deck-a/--deck-b deck "
                         "files; inline --hand/--library seats build stacked temp "
                         "decks with no sideboard to merge")
        if not (args.deck_a and args.deck_b):
            parser.error("--merge-sideboard requires both --deck-a and --deck-b")

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
            if args.merge_sideboard:
                deck_a_name, deck_a_file = _merge_sideboard_deck(args.deck_a, "a")
                cleanup_paths.append(deck_a_file)
        elif hand_a:
            library_a = _pad_library(hand_a, library_a)
            deck_a_name, deck_a_file = _make_deck_file(hand_a, library_a, "a")
            cleanup_paths.append(deck_a_file)
        else:
            deck_a_name = "delver"

        if args.deck_b and not hand_b:
            deck_b_name = args.deck_b
            if args.merge_sideboard:
                deck_b_name, deck_b_file = _merge_sideboard_deck(args.deck_b, "b")
                cleanup_paths.append(deck_b_file)
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
        elif args.merge_sideboard:
            print(f"Player A deck: {deck_a_name} (sideboard merged from {args.deck_a})")
        else:
            print(f"Player A deck: {deck_a_name}")
        if hand_b:
            print(f"Player B hand:    {', '.join(hand_b)}")
            print(f"Player B library: {', '.join(library_b)}")
        elif args.merge_sideboard:
            print(f"Player B deck: {deck_b_name} (sideboard merged from {args.deck_b})")
        else:
            print(f"Player B deck: {deck_b_name}")
        if battlefield_a:
            print(f"Player A battlefield: {', '.join(battlefield_a)}")
        if battlefield_b:
            print(f"Player B battlefield: {', '.join(battlefield_b)}")
        if graveyard_a:
            print(f"Player A graveyard: {', '.join(graveyard_a)}")
        if graveyard_b:
            print(f"Player B graveyard: {', '.join(graveyard_b)}")
        if exile_a:
            print(f"Player A exile: {', '.join(exile_a)}")
        if exile_b:
            print(f"Player B exile: {', '.join(exile_b)}")
        if sideboard_a:
            print(f"Player A sideboard: {', '.join(sideboard_a)}")
        if sideboard_b:
            print(f"Player B sideboard: {', '.join(sideboard_b)}")
        print(f"Seed: {seed}")
        if actions:
            print(f"Actions: {actions}")
        if play_specs:
            print(f"Play: {play_specs}")
        print()

        ctrl_a, ctrl_b, label_a, label_b = _build_controllers(
            args.player_a, args.player_b, play_specs, actions)

        # Pre-set battlefields are passed to the engine as comma-joined
        # deck-name strings (apostrophes stripped), same as the deck files.
        bf_a = ",".join(_card_to_deck_name(c) for c in battlefield_a) if battlefield_a else None
        bf_b = ",".join(_card_to_deck_name(c) for c in battlefield_b) if battlefield_b else None
        gy_a = ",".join(_card_to_deck_name(c) for c in graveyard_a) if graveyard_a else None
        gy_b = ",".join(_card_to_deck_name(c) for c in graveyard_b) if graveyard_b else None
        ex_a = ",".join(_card_to_deck_name(c) for c in exile_a) if exile_a else None
        ex_b = ",".join(_card_to_deck_name(c) for c in exile_b) if exile_b else None
        sb_a = ",".join(_card_to_deck_name(c) for c in sideboard_a) if sideboard_a else None
        sb_b = ",".join(_card_to_deck_name(c) for c in sideboard_b) if sideboard_b else None

        # Coverage accounting (opt-in): resolve each seat's full card list
        # (mainboard + sideboard + zone presets) so the report can flag deck
        # cards that never appeared in any menu. Imported lazily — the feature
        # costs nothing when the flag is off.
        coverage = None
        if args.coverage_json:
            from coverage_report import CoverageAccumulator, read_deck_cards
            seat_cards = []
            for deck_name, hand, library, extras in (
                    (args.deck_a or deck_a_name, hand_a, library_a,
                     (battlefield_a, graveyard_a, exile_a, sideboard_a)),
                    (args.deck_b or deck_b_name, hand_b, library_b,
                     (battlefield_b, graveyard_b, exile_b, sideboard_b))):
                if hand:
                    cards = list(hand) + list(library)
                else:
                    cards = read_deck_cards(str(_DECKS_DIR / f"{deck_name}.dk"))
                for zone in extras:
                    cards.extend(zone)
                seat_cards.append(cards)
            coverage = CoverageAccumulator(
                decks={"a": args.deck_a or deck_a_name,
                       "b": args.deck_b or deck_b_name},
                seeds=[seed], n_games=1,
                deck_cards_a=seat_cards[0], deck_cards_b=seat_cards[1])

        # The observation/decision loop lives in runner.run_games (shared with
        # train.py observe). test_harness only seeds the state above.
        wins, losses, _ = runner.run_games(
            ctrl_a, ctrl_b, label_a=label_a, label_b=label_b,
            binary_path=args.binary, deck_a=deck_a_name, deck_b=deck_b_name,
            n_games=1, bo3=bo3, seed=seed, verbose=True,
            battlefield_a=bf_a, battlefield_b=bf_b,
            graveyard_a=gy_a, graveyard_b=gy_b,
            exile_a=ex_a, exile_b=ex_b,
            sideboard_a=sb_a, sideboard_b=sb_b, no_shuffle=no_shuffle,
            life_a=life_a, life_b=life_b,
            max_decisions=max_decisions, log_decisions=args.log_decisions,
            coverage=coverage)
        winner = bool(wins or losses)
        if coverage is not None:
            coverage.write(args.coverage_json)
            print(f"\ncoverage written to {args.coverage_json}")
        # A --play run resolves specs to concrete indices; print them so the line
        # can be replayed deterministically as a plain --actions integer list.
        if isinstance(ctrl_a, PlayController) and ctrl_a.resolved:
            print(f"\nresolved --actions: {','.join(map(str, ctrl_a.resolved))}")
    finally:
        for p in cleanup_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    return 0 if winner else 1


if __name__ == "__main__":
    sys.exit(main())
