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
import struct
import subprocess
import sys
import tempfile
import numpy as np
from pathlib import Path

from env import STATE_SIZE, MAX_ACTIONS, ACTION_CATEGORY_MAX, _MANDATORY_CATS
from decode import decode_game_state, decode_actions, fmt_mana, fmt_perm

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BIN_DIR = _REPO_ROOT / "bin"
_BINARY = _BIN_DIR / "robomage"
_DECKS_DIR = _BIN_DIR / "resources" / "decks"

# Binary payload sizes
_STATE_BYTES = STATE_SIZE * 4
_CATS_BYTES = MAX_ACTIONS * 4
_IDS_BYTES = MAX_ACTIONS * 4
_CTRL_BYTES = MAX_ACTIONS * 4


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


# ── Formatted output ─────────────────────────────────────────────────────────

def _fmt_stack_entry(e):
    """Format a decoded stack-entry dict."""
    kind = "spell" if e["is_spell"] else "ability"
    return f"{e['name']} ({kind}, {e['controller']})"


def print_state(gs):
    """Print decoded game state in a compact, LLM-readable format."""
    print(f"  Priority: {gs['priority_player']}"
          f" ({'active' if gs['is_active_player'] else 'non-active'})")
    print(f"  Step: {gs['step']}")
    hand_names = [c["name"] for c in gs["self_hand"]]
    print(f"  Self:  {gs['self']['life']} life"
          f" | mana: {fmt_mana(gs['self']['mana'])}"
          f" | hand({gs['self']['hand_count']}): {', '.join(hand_names) or '(empty)'}")
    print(f"  Opp:   {gs['opponent']['life']} life"
          f" | mana: {fmt_mana(gs['opponent']['mana'])}"
          f" | hand count: {gs['opponent']['hand_count']}")

    if gs["self_battlefield"]:
        print(f"  Self BF:  {' | '.join(fmt_perm(p) for p in gs['self_battlefield'])}")
    if gs["opp_battlefield"]:
        print(f"  Opp BF:   {' | '.join(fmt_perm(p) for p in gs['opp_battlefield'])}")
    if gs["stack"]:
        print(f"  Stack: {' -> '.join(_fmt_stack_entry(e) for e in gs['stack'])}")
    if gs["self_graveyard"]:
        print(f"  Self GY: {', '.join(gs['self_graveyard'])}")
    if gs["opp_graveyard"]:
        print(f"  Opp GY:  {', '.join(gs['opp_graveyard'])}")


def print_actions(actions):
    """Print available actions."""
    for a in actions:
        print(f"  {a['index']:>2}: {a['description']}")


# ── Scripted agent (imported from env.py or simple fallback) ──────────────────

def _try_import_scripted():
    """Import scripted_action from env.py if available."""
    try:
        sys.path.insert(0, str(_REPO_ROOT / "train"))
        from env import scripted_action, STATE_SIZE as _ss, MAX_ACTIONS as _ma
        return scripted_action
    except ImportError:
        return None

_imported_scripted = _try_import_scripted()


def simple_scripted_action(state, cats_int, card_ids, ctrl, num_choices):
    """Minimal scripted agent: keeps hand, confirms blockers, passes otherwise."""
    for i in range(num_choices):
        cat = int(cats_int[i])
        # Always keep (don't mulligan)
        if cat == 11:  # MULLIGAN
            for j in range(num_choices):
                if int(cats_int[j]) != 11:
                    return j
            return 0
        # Confirm blockers immediately
        if cat == 5:
            return i
        # Confirm attackers
        if cat == 3:
            return i
    # Pass priority
    return 0


def get_scripted_action(state, cats_int, card_ids, ctrl, num_choices):
    """Use the full scripted agent from env.py, falling back to simple version."""
    if _imported_scripted is not None:
        # Reconstruct the obs format expected by env.py's scripted_action
        cat_arr = (cats_int[:MAX_ACTIONS] / ACTION_CATEGORY_MAX).astype(np.float32)
        id_arr = card_ids[:MAX_ACTIONS].copy()
        ctrl_arr = ctrl[:MAX_ACTIONS].copy()
        obs = np.concatenate([state, cat_arr, id_arr, ctrl_arr])
        # Pad to expected size if needed
        if obs.shape[0] < STATE_SIZE + 3 * MAX_ACTIONS:
            obs = np.pad(obs, (0, STATE_SIZE + 3 * MAX_ACTIONS - obs.shape[0]))
        return _imported_scripted(obs, num_choices)
    return simple_scripted_action(state, cats_int, card_ids, ctrl, num_choices)


# ── Engine driver ─────────────────────────────────────────────────────────────

class TestHarness:
    """Drives the game engine and decodes output for LLM observation."""

    def __init__(self, binary=str(_BINARY)):
        self.binary = binary
        self._proc = None
        self.decision_count = 0
        self.winner = None

    def start(self, deck_a_path, deck_b_path, seed=1,
              battlefield_a=None, battlefield_b=None, no_shuffle=True):
        """Launch the engine process.

        no_shuffle defaults True so deck-file order == draw order for the
        hand-sculpting workflow. Pass no_shuffle=False to let the engine shuffle
        each library with the seeded RNG (deterministic given the seed)."""
        cmd = [
            self.binary, "--machine", "--narrative",
            "--seed", str(seed),
            "--deck-a", deck_a_path,
            "--deck-b", deck_b_path,
        ]
        if no_shuffle:
            cmd.append("--no-shuffle")
        if battlefield_a:
            cmd += ["--battlefield-a", ",".join(_card_to_deck_name(c) for c in battlefield_a)]
        if battlefield_b:
            cmd += ["--battlefield-b", ",".join(_card_to_deck_name(c) for c in battlefield_b)]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=-1,
            cwd=str(_BIN_DIR),
        )

    def _send(self, action):
        self._proc.stdin.write(f"{action}\n".encode())
        self._proc.stdin.flush()

    def _read_exactly(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                raise EOFError("Engine process ended during binary read")
            buf.extend(chunk)
        return bytes(buf)

    def read_until_query(self):
        """Read narrative lines until next BQUERY or game end.

        Returns (narrative_lines, query_data) where query_data is
        (state, cats_int, card_ids, ctrl, num_choices) or None if game ended.
        """
        narrative = []

        while True:
            line = self._proc.stdout.readline()
            if not line:
                # Process ended
                return narrative, None

            line = line.rstrip(b"\n")

            # Check for win/loss
            if b"Player A wins" in line:
                self.winner = "Player A"
                narrative.append(line.decode("ascii", errors="replace"))
                continue
            elif b"Player B wins" in line:
                self.winner = "Player B"
                narrative.append(line.decode("ascii", errors="replace"))
                continue

            # BQUERY: parse binary payload
            if line.startswith(b"BQUERY: "):
                n = int(line[8:])
                num_choices = min(n, MAX_ACTIONS)

                state = np.frombuffer(
                    self._read_exactly(_STATE_BYTES), dtype=np.float32).copy()
                cats_int = np.frombuffer(
                    self._read_exactly(_CATS_BYTES), dtype=np.int32).copy()
                card_ids = np.frombuffer(
                    self._read_exactly(_IDS_BYTES), dtype=np.float32).copy()
                ctrl = np.frombuffer(
                    self._read_exactly(_CTRL_BYTES), dtype=np.float32).copy()

                return narrative, (state, cats_int, card_ids, ctrl, num_choices)

            # Regular narrative line
            text = line.decode("ascii", errors="replace")
            if text.strip():
                narrative.append(text)

    def kill(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.stdout.close()
                self._proc.kill()
                self._proc.wait()
            except Exception:
                pass
            self._proc = None

    def run_game(self, deck_a_path, deck_b_path, seed=1,
                 actions=None, interactive=False, scripted=False,
                 max_decisions=500,
                 battlefield_a=None, battlefield_b=None, no_shuffle=True):
        """Run a full game and print decoded output.

        Args:
            deck_a_path: Path to Player A's deck file
            deck_b_path: Path to Player B's deck file
            seed: RNG seed
            actions: List of action indices to play sequentially
            interactive: If True, prompt for each action
            scripted: If True, use the scripted agent for decisions
            max_decisions: Safety limit on decision points
            battlefield_a: List of card names to start on Player A's battlefield
            battlefield_b: List of card names to start on Player B's battlefield
        """
        self.start(deck_a_path, deck_b_path, seed,
                   battlefield_a=battlefield_a, battlefield_b=battlefield_b,
                   no_shuffle=no_shuffle)
        action_idx = 0
        pending_confirm = False

        try:
            while self.decision_count < max_decisions:
                narrative, query = self.read_until_query()

                # Print narrative
                if narrative:
                    print("--- Narrative ---")
                    for line in narrative:
                        print(f"  {line}")

                # Game ended
                if query is None:
                    if self.winner:
                        print(f"\n=== GAME OVER: {self.winner} wins! ===")
                    else:
                        print("\n=== GAME OVER (no winner detected) ===")
                    break

                state, cats_int, card_ids, ctrl, num_choices = query
                self.decision_count += 1

                # Decode and display
                gs = decode_game_state(state)
                decoded_actions = decode_actions(
                    cats_int, card_ids, ctrl, num_choices)

                # Check for mandatory confirm categories
                has_confirm = any(
                    int(cats_int[i]) in _MANDATORY_CATS
                    for i in range(num_choices))

                print(f"\n--- Decision {self.decision_count} ---")
                print_state(gs)
                print("  Actions:")
                print_actions(decoded_actions)

                # Choose action
                if actions is not None and action_idx < len(actions):
                    choice = actions[action_idx]
                    action_idx += 1
                    print(f"  >> Scripted action: {choice}"
                          f" ({decoded_actions[choice]['description']})")
                elif interactive:
                    while True:
                        try:
                            raw = input("  >> Enter action index: ").strip()
                            choice = int(raw)
                            if 0 <= choice < num_choices:
                                break
                            print(f"     Invalid: must be 0-{num_choices - 1}")
                        except (ValueError, EOFError):
                            print("     Enter a valid integer")
                elif scripted:
                    choice = get_scripted_action(
                        state, cats_int, card_ids, ctrl, num_choices)
                    print(f"  >> Agent: {choice}"
                          f" ({decoded_actions[choice]['description']})")
                else:
                    # No actions provided and not interactive — auto-pass
                    choice = 0
                    print(f"  >> Auto: {choice}"
                          f" ({decoded_actions[choice]['description']})")

                # Remap confirm slot (-1 convention)
                game_action = choice
                if has_confirm and choice == num_choices - 1:
                    game_action = -1

                self._send(game_action)

        finally:
            self.kill()

        return self.winner


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

        harness = TestHarness(binary=args.binary)
        winner = harness.run_game(
            deck_a_name, deck_b_name,
            seed=seed,
            actions=actions,
            interactive=args.interactive,
            scripted=args.scripted,
            max_decisions=max_decisions,
            battlefield_a=battlefield_a or None,
            battlefield_b=battlefield_b or None,
            no_shuffle=no_shuffle,
        )
    finally:
        for p in cleanup_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    return 0 if winner else 1


if __name__ == "__main__":
    sys.exit(main())
