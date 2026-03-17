"""
Play interactively against a trained RoboMage model.

The model is randomly assigned to Player A or B each game.
The observation is always emitted from the priority player's perspective,
so no manual mirroring is required.

Usage:
    train/.venv/bin/python train/play.py --human-deck delver --model-deck burn
"""

import argparse
import subprocess
import sys
import numpy as np

from env import (RoboMageEnv, STATE_SIZE, ACTION_CATEGORY_MAX, BINARY, MAX_ACTIONS,
                 BIN_DIR, OBS_SIZE, _ACTION_CARD_ID_NULL, _ACTION_CTRL_NULL,
                 _HAND_START, MAX_HAND_SLOTS,
                 _BQUERY_STATE_BYTES, _BQUERY_CATS_BYTES, _BQUERY_IDS_BYTES, _BQUERY_CTRL_BYTES)
import env as _env

try:
    from card_costs import _CARD_COST_MATRIX, _CARD_ABILITY_COST_MATRIX, N_CARD_TYPES, _N_COST_FEATS
except ImportError:
    from train.card_costs import _CARD_COST_MATRIX, _CARD_ABILITY_COST_MATRIX, N_CARD_TYPES, _N_COST_FEATS

try:
    from sb3_contrib import MaskablePPO
    USE_MASKABLE = True
except ImportError:
    from stable_baselines3 import PPO as MaskablePPO
    USE_MASKABLE = False

# ── Card vocab (mirrors src/card_vocab.h) ─────────────────────────────────────
_VOCAB = {
    0: "Mountain", 1: "Forest", 2: "Lightning Bolt", 3: "Grizzly Bears",
    4: "Volcanic Island", 5: "Scalding Tarn", 6: "Flooded Strand",
    7: "Polluted Delta", 8: "Wooded Foothills", 9: "Misty Rainforest",
    10: "Wasteland", 11: "Ponder", 12: "Force of Will", 13: "Daze",
    14: "Soul Warden", 15: "Tundra", 16: "Delver of Secrets",
    17: "Insectile Aberration", 18: "Flying Men", 19: "Island",
    20: "Dragon's Rage Channeler", 21: "Air Elemental", 22: "Counterspell",
    23: "Lightning Strike", 24: "Brainstorm",
}
_N_VOCAB = 32

# ── State layout (mirrors env.py / machine_io.h) ──────────────────────────────
_BF_START        = 33
_BF_SLOT_SIZE    = 42     # 10 status floats + 32 card one-hot
_BF_PERM_SLOTS   = 48     # self occupies slots 0-47; opponent slots 48-95
_BF_CARD_OFF     = 10
_OFF_POWER       = 0
_OFF_TOUGHNESS   = 1
_OFF_TAPPED      = 2
_OFF_ATTACKING   = 3
_OFF_SICKNESS    = 5
_OFF_IS_CREATURE = 8
_OFF_IS_LAND     = 9
_HAND_OBS_START  = 8569   # 4473 + 128 * 32
_HAND_SLOTS      = 10
_STEP_NAMES = [
    "Untap", "Upkeep", "Draw", "First Main",
    "Begin Combat", "Declare Attackers", "Declare Blockers",
    "Combat Damage", "End Combat", "Second Main", "End", "Cleanup",
]
_MANA_CAT_COLOR = {13: "W", 14: "U", 15: "B", 16: "R", 17: "G", 18: "C"}


# ── Decode helpers ────────────────────────────────────────────────────────────

def _card_name(one_hot):
    idx = int(np.argmax(one_hot))
    return _VOCAB.get(idx) if one_hot[idx] > 0.5 else None


def _card_from_id(val: float):
    """Decode a card name from a normalised card-ID float (None for null sentinel)."""
    cid = round(float(val) * _N_VOCAB)
    return _VOCAB.get(cid) if cid >= 0 else None


def _action_label(cat: int, card_id_float: float) -> str:
    card = _card_from_id(card_id_float)
    if cat == 0:  return "pass priority"
    if cat == 2:  return f"attack with {card or '?'}"
    if cat == 3:  return "confirm attackers"
    if cat == 4:  return f"block with {card or '?'}"
    if cat == 5:  return "confirm blockers"
    if cat == 6:  return f"activate {card or '?'}"
    if cat == 7:  return f"cast {card or '?'}"
    if cat == 8:  return "select target"
    if cat == 9:  return f"play {card or '?'}"
    if cat == 10: return f"choose {card}" if card else "choose"
    if cat == 12: return f"bottom {card}" if card else "bottom card"
    if cat in _MANA_CAT_COLOR:
        return f"tap {card or '?'} for {{{_MANA_CAT_COLOR[cat]}}}"
    if cat == 19: return "fail to find" if card is None else f"find {card}"
    return f"action {cat}"


def _decode_step(obs) -> str:
    idx = int(np.argmax(obs[18:31]))
    return _STEP_NAMES[idx] if obs[18 + idx] > 0.5 else "?"


def _perm_str(obs, base):
    """Return a display string for a permanent slot, or None if empty."""
    card = _card_name(obs[base + _BF_CARD_OFF : base + _BF_CARD_OFF + _N_VOCAB])
    if card is None:
        return None
    if obs[base + _OFF_IS_CREATURE] > 0.5:
        pw = int(round(float(obs[base + _OFF_POWER]) * 10))
        tg = int(round(float(obs[base + _OFF_TOUGHNESS]) * 10))
        tags = []
        if obs[base + _OFF_TAPPED]    > 0.5: tags.append("tapped")
        if obs[base + _OFF_ATTACKING] > 0.5: tags.append("atk")
        if obs[base + _OFF_SICKNESS]  > 0.5: tags.append("sick")
        suffix = f"[{','.join(tags)}]" if tags else ""
        return f"{card} {pw}/{tg}{suffix}"
    # land or other permanent
    return f"{card}{'*' if obs[base + _OFF_TAPPED] > 0.5 else ''}"


def _split_bf(obs, slot_offset):
    """Return (creatures, lands) lists for the 48 permanent slots starting at slot_offset."""
    creatures, lands = [], []
    for i in range(_BF_PERM_SLOTS):
        base = _BF_START + (slot_offset + i) * _BF_SLOT_SIZE
        card = _card_name(obs[base + _BF_CARD_OFF : base + _BF_CARD_OFF + _N_VOCAB])
        if card is None:
            continue
        if obs[base + _OFF_IS_CREATURE] > 0.5:
            s = _perm_str(obs, base)
            if s: creatures.append(s)
        elif obs[base + _OFF_IS_LAND] > 0.5:
            s = _perm_str(obs, base)
            if s: lands.append(s)
    return creatures, lands


def _format_state(obs) -> str:
    """Format battlefield from the current priority player's (human's) perspective.

    obs is always perspective-normalised: self occupies permanent slots 0-47,
    opponent slots 48-95. Hand at _HAND_OBS_START is the priority player's hand.
    """
    my_life  = int(round(float(obs[0]) * 20))
    opp_life = int(round(float(obs[9]) * 20))

    my_creatures,  my_lands  = _split_bf(obs, 0)
    opp_creatures, opp_lands = _split_bf(obs, _BF_PERM_SLOTS)
    hand = [s for i in range(_HAND_SLOTS)
            if (s := _card_name(obs[_HAND_OBS_START + i * _N_VOCAB :
                                    _HAND_OBS_START + (i + 1) * _N_VOCAB]))]

    return (
        "--- Battlefield ---\n"
        f"  [You]   {my_life} life | creatures: {', '.join(my_creatures) or '—'}"
        f" | lands: {', '.join(my_lands) or '—'}\n"
        f"  [Model] {opp_life} life | creatures: {', '.join(opp_creatures) or '—'}"
        f" | lands: {', '.join(opp_lands) or '—'}\n"
        f"  Your hand: {', '.join(hand) if hand else '—'}\n"
        "-------------------"
    )


# ── Env subclass with filtered narrative output ───────────────────────────────

class PlayEnv(RoboMageEnv):
    """RoboMageEnv that prints narrative to stdout and suppresses library-search listings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._in_search_block = False

    def _print_narrative_line(self, line: str):
        # Suppress both players' library-search option blocks — play.py shows
        # them more clearly via the QUERY data.
        if "Searching" in line and "library:" in line:
            self._in_search_block = True
            return
        if self._in_search_block:
            if line.startswith("  ") or not line.strip():
                return
            self._in_search_block = False
        print(line, flush=True)


# ── Main play loop ────────────────────────────────────────────────────────────

def play(binary_path: str, model_path: str, human_deck: str = "delver", model_deck: str = "delver"):
    model = MaskablePPO.load(model_path)

    model_is_a = bool(np.random.random() < 0.5)
    deck_a = model_deck if model_is_a else human_deck
    deck_b = human_deck if model_is_a else model_deck
    env = PlayEnv(binary_path=binary_path, render_mode="human", deck_a=deck_a, deck_b=deck_b)
    obs, _ = env.reset()
    done = False

    model_role = "A" if model_is_a else "B"
    human_role = "B" if model_is_a else "A"
    print(f"=== Model (Player {model_role}, {model_deck}) vs You (Player {human_role}, {human_deck}) ===", flush=True)
    print("(type 'quit' to exit)\n", flush=True)

    while not done:
        num_choices = env._num_choices
        # obs[32]=1 means self (priority player) is Player A; obs[31]=1 means priority player is active player
        a_has_priority = obs[32] > 0.5
        model_has_priority = a_has_priority == model_is_a

        cats     = np.round(obs[STATE_SIZE : STATE_SIZE + num_choices] * ACTION_CATEGORY_MAX).astype(int)
        card_ids = obs[STATE_SIZE + MAX_ACTIONS : STATE_SIZE + 2 * MAX_ACTIONS]

        is_mulligan_q = num_choices > 0 and all(c == 11 for c in cats)
        is_bottom_q   = num_choices > 0 and all(c == 12 for c in cats)
        is_search_q   = num_choices > 0 and all(c == 19 for c in cats)

        if model_has_priority:
            masks = env.action_masks() if USE_MASKABLE else None
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            action = int(action)
            if is_mulligan_q:
                label = "keep" if action == 0 else "mulligan"
            else:
                cat = int(cats[action]) if action < len(cats) else -1
                cid = float(card_ids[action]) if action < MAX_ACTIONS else -1.0 / _N_VOCAB
                label = _action_label(cat, cid)
            print(f"[Model/{model_role}] {label}", flush=True)
        else:
            # Human's turn — obs is from human's perspective (priority player)
            print(_format_state(obs), flush=True)
            print()

            if is_mulligan_q:
                print("Mulligan decision:")
                print("  0: keep")
                print("  1: mulligan")
            elif is_bottom_q:
                print(f"Choose a card to put on the bottom ({num_choices} option(s)):")
                for i, c in enumerate(cats):
                    print(f"  {i}: {_action_label(int(c), float(card_ids[i]))}")
            elif is_search_q:
                print("Search your library:")
                for i, c in enumerate(cats):
                    print(f"  {i}: {_action_label(int(c), float(card_ids[i]))}")
            else:
                # obs[31]=1 means priority player is active; obs[32]=1 means priority player is A
                priority_is_a = obs[32] > 0.5
                active_is_a = (obs[31] > 0.5) == priority_is_a
                active = "A" if active_is_a else "B"
                step = _decode_step(obs)
                print(f"[{active}'s turn — {step}]  {num_choices} option(s):")
                for i, c in enumerate(cats):
                    print(f"  {i}: {_action_label(int(c), float(card_ids[i]))}")

            while True:
                try:
                    raw = input("Choose> ").strip()
                    if raw.lower() == "quit":
                        env.close()
                        sys.exit(0)
                    action = int(raw)
                    if 0 <= action < num_choices:
                        break
                    print(f"Enter a number between 0 and {num_choices - 1}.")
                except EOFError:
                    env.close()
                    sys.exit(0)
                except ValueError:
                    print("Enter a number (or 'quit').")

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    print()
    # reward is +1.0 if A wins, -1.0 if B wins
    model_wins = (reward > 0 and model_is_a) or (reward < 0 and not model_is_a)
    human_wins = (reward > 0 and not model_is_a) or (reward < 0 and model_is_a)
    if model_wins:
        print(f"=== Model ({model_role}) wins! ===")
    elif human_wins:
        print("=== You win! ===")
    else:
        print("=== Draw ===")


def play_gui(binary_path: str, model_path: str, human_player: str = None,
             human_deck: str = "delver", model_deck: str = "delver"):
    """Launch the raylib GUI window; model auto-responds on AI turns, human types in GUI text box.

    Architecture:
    - Binary runs with --machine --gui --player <human_player>
    - Model turns: binary emits BQUERY on stdout; Python reads it, predicts, writes to stdin
    - Human turns: binary displays choices in the GUI window and spins waiting for GUI text-box
      input; Python's reader thread collects narrative lines but sees no BQUERY, so the main
      thread simply waits for the next BQUERY to arrive after the human has acted
    - A background reader thread drains stdout continuously so Python never blocks on readline()
      while the binary is waiting for GUI input
    """
    import queue
    import threading

    model = MaskablePPO.load(model_path)

    if human_player is None:
        human_player = "A" if np.random.random() < 0.5 else "B"
    model_player = "B" if human_player == "A" else "A"
    deck_a = human_deck if human_player == "A" else model_deck
    deck_b = model_deck if human_player == "A" else human_deck
    print(f"=== You (Player {human_player}, {human_deck}) vs Model (Player {model_player}, {model_deck}) ===", flush=True)
    print("(Type your choice number into the GUI text box and press Enter)", flush=True)

    cmd = [binary_path, "--machine", "--gui", "--player", human_player,
           "--deck-a", deck_a, "--deck-b", deck_b]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=BIN_DIR,
    )

    # Items put in queue are either:
    #   str  — a decoded narrative line
    #   dict — a parsed BQUERY with keys: num_choices, state_arr, cat_arr, card_id_arr,
    #           ctrl_arr, pending_confirm
    #   None — EOF sentinel
    line_queue: queue.Queue = queue.Queue()

    _MANDATORY = {2, 3, 4, 5}
    _PAYLOAD = _BQUERY_STATE_BYTES + _BQUERY_CATS_BYTES + _BQUERY_IDS_BYTES + _BQUERY_CTRL_BYTES

    def _read_exactly(n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = proc.stdout.read(n - len(buf))
            if not chunk:
                raise EOFError("game process ended mid-payload")
            buf.extend(chunk)
        return bytes(buf)

    def _reader():
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.rstrip(b"\n")
                if line.startswith(b"BQUERY: "):
                    n = min(int(line[8:]), MAX_ACTIONS)
                    payload = _read_exactly(_PAYLOAD)
                    offset = 0
                    state_arr = np.frombuffer(payload[offset:offset + _BQUERY_STATE_BYTES],
                                              dtype=np.float32).copy()
                    offset += _BQUERY_STATE_BYTES
                    cats_int = np.frombuffer(payload[offset:offset + _BQUERY_CATS_BYTES],
                                             dtype=np.int32).copy()
                    offset += _BQUERY_CATS_BYTES
                    id_arr = np.frombuffer(payload[offset:offset + _BQUERY_IDS_BYTES],
                                           dtype=np.float32).copy()
                    offset += _BQUERY_IDS_BYTES
                    ctrl_arr = np.frombuffer(payload[offset:offset + _BQUERY_CTRL_BYTES],
                                             dtype=np.float32).copy()
                    cat_arr = (cats_int / ACTION_CATEGORY_MAX).astype(np.float32)
                    pending_confirm = any(cats_int[i] in _MANDATORY for i in range(n))
                    line_queue.put({"num_choices": n, "state_arr": state_arr,
                                    "cat_arr": cat_arr, "card_id_arr": id_arr,
                                    "ctrl_arr": ctrl_arr, "pending_confirm": pending_confirm})
                else:
                    line_queue.put(line.decode("ascii", errors="replace"))
        except Exception:
            pass
        line_queue.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    reward = 0.0
    try:
        while True:
            item = line_queue.get()
            if item is None:
                break

            if isinstance(item, str):
                if "Player A wins" in item:
                    reward = 1.0
                elif "Player B wins" in item:
                    reward = -1.0
                continue

            # BQUERY dict — model's turn
            num_choices  = item["num_choices"]
            state_arr    = item["state_arr"]
            cat_arr      = item["cat_arr"]
            card_id_arr  = item["card_id_arr"]
            ctrl_arr     = item["ctrl_arr"]
            pending_confirm = item["pending_confirm"]

            hand_onehots = state_arr[_HAND_START : _HAND_START + MAX_HAND_SLOTS * N_CARD_TYPES]
            hand_costs = hand_onehots.reshape(MAX_HAND_SLOTS, N_CARD_TYPES) @ _CARD_COST_MATRIX
            bf_ability_costs = np.zeros((48, _N_COST_FEATS), dtype=np.float32)
            for slot in range(48):
                base = _env._BF_START + slot * _env._BF_SLOT_SIZE + _env._BF_CARD_OFF
                bf_ability_costs[slot] = state_arr[base : base + N_CARD_TYPES] @ _CARD_ABILITY_COST_MATRIX

            obs = np.concatenate([
                state_arr, cat_arr, card_id_arr, ctrl_arr,
                hand_costs.flatten(), bf_ability_costs.flatten(),
            ])

            mask = np.zeros(MAX_ACTIONS, dtype=bool)
            mask[:num_choices] = True
            action, _ = model.predict(obs, action_masks=mask if USE_MASKABLE else None, deterministic=True)
            action = int(action)

            cat = int(round(float(cat_arr[action]) * ACTION_CATEGORY_MAX)) if action < num_choices else -1
            cid = float(card_id_arr[action]) if action < MAX_ACTIONS else float(_ACTION_CARD_ID_NULL)
            print(f"[Model/{model_player}] {_action_label(cat, cid)}", flush=True)

            game_action = -1 if (pending_confirm and action == num_choices - 1) else action
            proc.stdin.write(f"{game_action}\n".encode())
            proc.stdin.flush()
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()

    print()
    model_wins = (reward > 0 and model_player == "A") or (reward < 0 and model_player == "B")
    human_wins = (reward > 0 and human_player == "A") or (reward < 0 and human_player == "B")
    if model_wins:
        print(f"=== Model ({model_player}) wins! ===")
    elif human_wins:
        print("=== You win! ===")
    else:
        print("=== Draw ===")


if __name__ == "__main__":
    import os as _os
    _CHECKPOINT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "checkpoints")

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=BINARY)
    parser.add_argument("--human-deck", required=True,
                        help="Deck the human plays (stem of .dk file)")
    parser.add_argument("--model-deck", required=True,
                        help="Deck the model plays (stem of .dk file). "
                             "Automatically loads checkpoints/<model-deck>_final.zip")
    parser.add_argument("--model", default=None,
                        help="Override: explicit path to trained model .zip "
                             "(default: checkpoints/<model-deck>_final.zip)")
    parser.add_argument("--gui", action="store_true", help="Launch raylib GUI window for human input")
    parser.add_argument("--player", choices=["A", "B"], default=None,
                        help="Which player the human controls (default: random)")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = _os.path.join(_CHECKPOINT_DIR, f"{args.model_deck}_final.zip")
        if not _os.path.exists(model_path):
            parser.error(f"No checkpoint found at {model_path}. "
                         f"Train a model with --deck {args.model_deck} first, or use --model to specify a path.")

    if args.gui:
        play_gui(args.binary, model_path, human_player=args.player,
                 human_deck=args.human_deck, model_deck=args.model_deck)
    else:
        play(args.binary, model_path, human_deck=args.human_deck, model_deck=args.model_deck)
