#!/usr/bin/env python3
"""
LLM-driven test harness for RoboMage card testing.

Runs the game engine with --machine --narrative --no-shuffle so that:
  - Deck order = draw order (first 7 cards in deck file become the hand)
  - Game narrative (casts, damage, zone changes) prints alongside binary queries
  - Binary state is decoded into human-readable text for LLM observation

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
        --scripted --max-turns 4

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
        "max_turns": 5,
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

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BIN_DIR = _REPO_ROOT / "bin"
_BINARY = _BIN_DIR / "robomage"
_DECKS_DIR = _BIN_DIR / "resources" / "decks"

# ── Engine constants (mirror src/machine_io.h) ───────────────────────────────
STATE_SIZE = 32551
N_CARD_TYPES = 128
MAX_ACTIONS = 64
ACTION_CATEGORY_MAX = 23
PERM_SLOT_SIZE = 138
STACK_SLOT_SIZE = 130
GY_SLOT_SIZE = 128

# State layout offsets
_SELF_PERM_START = 34
_OPP_PERM_START = 34 + 48 * PERM_SLOT_SIZE       # 6658
_STACK_START = _OPP_PERM_START + 48 * PERM_SLOT_SIZE  # 13282
_GY_START = _STACK_START + 12 * STACK_SLOT_SIZE    # 14842
_OPP_GY_START = _GY_START + 64 * GY_SLOT_SIZE     # 23034
_HAND_START = _OPP_GY_START + 64 * GY_SLOT_SIZE   # 31226

# Permanent slot field offsets
_OFF_POWER = 0
_OFF_TOUGHNESS = 1
_OFF_TAPPED = 2
_OFF_ATTACKING = 3
_OFF_BLOCKING = 4
_OFF_SICKNESS = 5
_OFF_DAMAGE = 6
_OFF_CTRL = 7
_OFF_IS_CREATURE = 8
_OFF_IS_LAND = 9
_OFF_CARD_ONEHOT = 10

# Step one-hot starts at obs[18], 13 steps
_STEP_NAMES = [
    "Untap", "Upkeep", "Draw", "First Main", "Begin Combat",
    "Declare Attackers", "Declare Blockers", "First Strike Damage",
    "Combat Damage", "End of Combat", "Second Main", "End Step", "Cleanup",
]

# ── Card vocab (mirror src/card_vocab.h) ─────────────────────────────────────
_CARD_NAMES = [
    "Mountain", "Forest", "Lightning Bolt", "Grizzly Bears", "Volcanic Island",
    "Scalding Tarn", "Flooded Strand", "Polluted Delta", "Wooded Foothills", "Misty Rainforest",
    "Wasteland", "Ponder", "Force of Will", "Daze", "Soul Warden", "Tundra",
    "Delver of Secrets", "Insectile Aberration", "Flying Men", "Island",
    "Dragon's Rage Channeler", "Air Elemental", "Counterspell", "Lightning Strike",
    "Brainstorm", "Thundering Falls",
    "Murktide Regent", "Mishra's Bauble", "Cori-Steel Cutter", "Unholy Heat",
    "Birds of Paradise", "Collector Ouphe", "Dryad Arbor", "Endurance",
    "Gaea's Cradle", "Green Sun's Zenith", "Horizon Canopy", "Icetill Explorer",
    "Ignoble Hierarch", "Karakas", "Keen-Eyed Curator", "Knight of the Reliquary",
    "Noble Hierarch", "Once Upon a Time", "Plains", "Savannah",
    "Scryb Ranger", "Scythecat Cub", "Swords to Plowshares", "Sylvan Library",
    "Talon Gates of Madara", "Thalia, Guardian of Thraben", "Windswept Heath",
    "Doomsday", "Thoughtseize", "Dark Ritual", "Lotus Petal",
    "Lion's Eye Diamond", "Thassa's Oracle", "Personal Tutor", "Street Wraith",
    "Edge of Autumn", "Swamp", "Undercity Sewers", "Underground Sea",
    "Bloodstained Mire", "Verdant Catacombs", "Cavern of Souls",
    "Consider", "Duress", "Deep Analysis",
]

def card_index_to_name(idx):
    if idx == 127:
        return "Token"
    if 0 <= idx < len(_CARD_NAMES):
        return _CARD_NAMES[idx]
    return f"?({idx})"

# ── Action category names ────────────────────────────────────────────────────
_CAT_NAMES = {
    0: "PASS", 1: "MANA_ABILITY", 2: "SELECT_ATTACKER", 3: "CONFIRM_ATTACKERS",
    4: "SELECT_BLOCKER", 5: "CONFIRM_BLOCKERS", 6: "ACTIVATE", 7: "CAST",
    8: "TARGET", 9: "PLAY_LAND", 10: "CHOICE", 11: "MULLIGAN",
    12: "BOTTOM_CARD", 13: "MANA_W", 14: "MANA_U", 15: "MANA_B",
    16: "MANA_R", 17: "MANA_G", 18: "MANA_C", 19: "SEARCH",
    20: "TOP_LIBRARY", 21: "SHUFFLE", 22: "PAY_COST", 23: "DIG",
}

_MANDATORY_CATS = frozenset({2, 3, 4, 5})

# Binary payload sizes
_STATE_BYTES = STATE_SIZE * 4
_CATS_BYTES = MAX_ACTIONS * 4
_IDS_BYTES = MAX_ACTIONS * 4
_CTRL_BYTES = MAX_ACTIONS * 4
_NULL_SENTINEL = -1.0 / N_CARD_TYPES


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


# ── State decoder ─────────────────────────────────────────────────────────────

def _onehot_to_card(state, base):
    """Decode a one-hot card vector starting at base. Returns card name or None."""
    vec = state[base:base + N_CARD_TYPES]
    idx = int(np.argmax(vec))
    if vec[idx] < 0.5:
        return None
    return card_index_to_name(idx)


def _decode_player(state, offset):
    """Decode a 9-float player block into a dict."""
    return {
        "life": int(round(state[offset] * 20)),
        "hand_count": int(round(state[offset + 1] * 10)),
        "poison": int(round(state[offset + 2] * 10)),
        "mana": {
            "W": int(round(state[offset + 3] * 10)),
            "U": int(round(state[offset + 4] * 10)),
            "B": int(round(state[offset + 5] * 10)),
            "R": int(round(state[offset + 6] * 10)),
            "G": int(round(state[offset + 7] * 10)),
            "C": int(round(state[offset + 8] * 10)),
        },
    }


def _decode_step(state):
    """Decode the current step from one-hot at obs[18:31]."""
    step_vec = state[18:31]
    idx = int(np.argmax(step_vec))
    if step_vec[idx] < 0.5:
        return "Unknown"
    return _STEP_NAMES[idx] if idx < len(_STEP_NAMES) else f"Step({idx})"


def _decode_permanents(state, start, count=48):
    """Decode permanent slots into a list of dicts (non-empty only)."""
    perms = []
    for i in range(count):
        base = start + i * PERM_SLOT_SIZE
        card = _onehot_to_card(state, base + _OFF_CARD_ONEHOT)
        if card is None:
            continue
        p = {"name": card}
        if state[base + _OFF_IS_CREATURE] > 0.5:
            p["power"] = int(round(state[base + _OFF_POWER] * 10))
            p["toughness"] = int(round(state[base + _OFF_TOUGHNESS] * 10))
            dmg = int(round(state[base + _OFF_DAMAGE] * 10))
            if dmg > 0:
                p["damage"] = dmg
        if state[base + _OFF_TAPPED] > 0.5:
            p["tapped"] = True
        if state[base + _OFF_ATTACKING] > 0.5:
            p["attacking"] = True
        if state[base + _OFF_BLOCKING] > 0.5:
            p["blocking"] = True
        if state[base + _OFF_SICKNESS] > 0.5:
            p["summoning_sick"] = True
        if state[base + _OFF_IS_LAND] > 0.5:
            p["is_land"] = True
        perms.append(p)
    return perms


def _decode_zone(state, start, slots, slot_size):
    """Decode a zone (hand/graveyard/stack) into a list of card names."""
    cards = []
    for i in range(slots):
        base = start + i * slot_size
        card = _onehot_to_card(state, base if slot_size == GY_SLOT_SIZE else base + 1)
        if card is not None:
            cards.append(card)
    return cards


def _decode_hand(state):
    """Decode self hand (10 slots x 128 one-hot)."""
    cards = []
    for i in range(10):
        base = _HAND_START + i * N_CARD_TYPES
        card = _onehot_to_card(state, base)
        if card is not None:
            cards.append(card)
    return cards


def _decode_graveyard(state, start):
    """Decode a graveyard zone (64 slots x 128 one-hot)."""
    cards = []
    for i in range(64):
        base = start + i * GY_SLOT_SIZE
        card = _onehot_to_card(state, base)
        if card is not None:
            cards.append(card)
    return cards


def _decode_stack(state):
    """Decode the stack (12 slots x 130: ctrl(1) + card_onehot(128) + is_spell(1))."""
    entries = []
    for i in range(12):
        base = _STACK_START + i * STACK_SLOT_SIZE
        card = _onehot_to_card(state, base + 1)  # skip controller_is_self float
        if card is None:
            continue
        ctrl = "self" if state[base] > 0.5 else "opponent"
        kind = "spell" if state[base + 129] > 0.5 else "ability"
        entries.append(f"{card} ({kind}, {ctrl})")
    return entries


def decode_game_state(state):
    """Decode the full state vector into a human-readable dict."""
    self_player = _decode_player(state, 0)
    opp_player = _decode_player(state, 9)
    step = _decode_step(state)
    is_active = state[31] > 0.5
    is_player_a = state[32] > 0.5
    stack_size = int(round(state[33] * 10))

    return {
        "priority_player": "Player A" if is_player_a else "Player B",
        "is_active_player": is_active,
        "step": step,
        "self": self_player,
        "opponent": opp_player,
        "self_battlefield": _decode_permanents(state, _SELF_PERM_START),
        "opp_battlefield": _decode_permanents(state, _OPP_PERM_START),
        "stack": _decode_stack(state),
        "self_hand": _decode_hand(state),
        "self_graveyard": _decode_graveyard(state, _GY_START),
        "opp_graveyard": _decode_graveyard(state, _OPP_GY_START),
    }


# ── Action decoder ────────────────────────────────────────────────────────────

def decode_actions(cats_int, card_ids, ctrl, num_choices):
    """Decode action arrays into human-readable list."""
    actions = []
    for i in range(num_choices):
        cat = int(cats_int[i])
        cat_name = _CAT_NAMES.get(cat, f"?({cat})")

        card_id_raw = float(card_ids[i])
        card_idx = int(round(card_id_raw * N_CARD_TYPES))
        card_name = card_index_to_name(card_idx) if card_idx >= 0 else None

        ctrl_val = float(ctrl[i])
        if ctrl_val > 0.5:
            ctrl_str = "own"
        elif ctrl_val > _NULL_SENTINEL + 0.01:
            ctrl_str = "opp"
        else:
            ctrl_str = None

        # Build description
        desc = _describe_action(cat, cat_name, card_name, ctrl_str)
        actions.append({"index": i, "category": cat_name, "card": card_name,
                        "controller": ctrl_str, "description": desc})
    return actions


def _describe_action(cat, cat_name, card_name, ctrl_str):
    """Build a concise human-readable action description."""
    owner = f" ({ctrl_str})" if ctrl_str else ""
    name = card_name or ""

    if cat == 0:
        return "Pass priority"
    elif cat == 7:
        return f"Cast {name}"
    elif cat == 9:
        return f"Play land: {name}"
    elif cat == 6:
        return f"Activate {name}{owner}"
    elif cat == 8:
        return f"Target {name}{owner}"
    elif cat == 2:
        return f"Select attacker: {name}"
    elif cat == 3:
        return "Confirm attackers"
    elif cat == 4:
        return f"Select blocker: {name}"
    elif cat == 5:
        return "Confirm blockers"
    elif cat == 11:
        return f"Mulligan ({name})" if name else "Mulligan"
    elif cat == 12:
        return f"Bottom: {name}"
    elif 13 <= cat <= 18:
        color = ["W", "U", "B", "R", "G", "C"][cat - 13]
        return f"Tap {name} for {{{color}}}"
    elif cat == 19:
        return f"Search: {name}" if name else "Fail to find"
    elif cat == 20:
        return f"Put on top: {name}"
    elif cat == 22:
        return f"Pay cost: {name}{owner}"
    elif cat == 23:
        return f"Dig choice: {name}"
    elif cat == 10:
        return f"Choice: {name}{owner}" if name else "Choice"
    else:
        return f"{cat_name}: {name}{owner}" if name else cat_name


# ── Formatted output ─────────────────────────────────────────────────────────

def _fmt_mana(mana):
    """Format mana pool as a compact string."""
    parts = []
    for color in ("W", "U", "B", "R", "G", "C"):
        amt = mana.get(color, 0)
        if amt > 0:
            parts.append(f"{amt}{color}")
    return ", ".join(parts) if parts else "empty"


def _fmt_perm(p):
    """Format a single permanent."""
    s = p["name"]
    if "power" in p:
        s += f" [{p['power']}/{p['toughness']}"
        if "damage" in p:
            s += f", {p['damage']}dmg"
        s += "]"
    flags = []
    if p.get("tapped"):
        flags.append("T")
    if p.get("attacking"):
        flags.append("ATK")
    if p.get("blocking"):
        flags.append("BLK")
    if p.get("summoning_sick"):
        flags.append("SICK")
    if flags:
        s += f" ({','.join(flags)})"
    return s


def print_state(gs):
    """Print decoded game state in a compact, LLM-readable format."""
    print(f"  Priority: {gs['priority_player']}"
          f" ({'active' if gs['is_active_player'] else 'non-active'})")
    print(f"  Step: {gs['step']}")
    print(f"  Self:  {gs['self']['life']} life"
          f" | mana: {_fmt_mana(gs['self']['mana'])}"
          f" | hand({gs['self']['hand_count']}): {', '.join(gs['self_hand']) or '(empty)'}")
    print(f"  Opp:   {gs['opponent']['life']} life"
          f" | mana: {_fmt_mana(gs['opponent']['mana'])}"
          f" | hand count: {gs['opponent']['hand_count']}")

    if gs["self_battlefield"]:
        print(f"  Self BF:  {' | '.join(_fmt_perm(p) for p in gs['self_battlefield'])}")
    if gs["opp_battlefield"]:
        print(f"  Opp BF:   {' | '.join(_fmt_perm(p) for p in gs['opp_battlefield'])}")
    if gs["stack"]:
        print(f"  Stack: {' -> '.join(gs['stack'])}")
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

    def start(self, deck_a_path, deck_b_path, seed=1):
        """Launch the engine process."""
        cmd = [
            self.binary, "--machine", "--narrative", "--no-shuffle",
            "--seed", str(seed),
            "--deck-a", deck_a_path,
            "--deck-b", deck_b_path,
        ]
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
                 max_decisions=500, max_turns=50):
        """Run a full game and print decoded output.

        Args:
            deck_a_path: Path to Player A's deck file
            deck_b_path: Path to Player B's deck file
            seed: RNG seed
            actions: List of action indices to play sequentially
            interactive: If True, prompt for each action
            scripted: If True, use the scripted agent for decisions
            max_decisions: Safety limit on decision points
            max_turns: Not enforced here (engine handles turns)
        """
        self.start(deck_a_path, deck_b_path, seed)
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
    parser.add_argument("--actions", help="Comma-separated action indices to play")
    parser.add_argument("--interactive", action="store_true", help="Prompt for each action")
    parser.add_argument("--scripted", action="store_true", help="Use scripted agent")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed (default: 1)")
    parser.add_argument("--max-decisions", type=int, default=500)
    parser.add_argument("--max-turns", type=int, default=50)
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
    seed = args.seed if args.seed != 1 else scenario.get("seed", 1)
    actions_str = args.actions or scenario.get("actions")
    max_decisions = args.max_decisions or scenario.get("max_decisions", 500)

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
