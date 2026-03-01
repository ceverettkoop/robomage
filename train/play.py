"""
Play interactively against a trained RoboMage model.

The model plays as Player A. You play as Player B.

Usage:
    train/.venv/bin/python train/play.py --model checkpoints/robomage_final.zip
"""

import argparse
import sys
import numpy as np

from env import RoboMageEnv, STATE_SIZE, ACTION_CATEGORY_MAX, BINARY, MAX_ACTIONS

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
_BF_CREATURE_OFF = 33
_BF_LAND_OFF     = 833
_BF_SLOT_SIZE    = 40
_BF_CARD_OFF     = 8
_OFF_POWER       = 0
_OFF_TOUGHNESS   = 1
_OFF_TAPPED      = 2
_OFF_ATTACKING   = 3
_OFF_SICKNESS    = 5
_HAND_OBS_START  = 2438
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
    idx = int(np.argmax(obs[18:30]))
    return _STEP_NAMES[idx] if obs[18 + idx] > 0.5 else "?"


def _creature_str(obs, base):
    card = _card_name(obs[base + _BF_CARD_OFF : base + _BF_CARD_OFF + _N_VOCAB])
    if card is None:
        return None
    pw = int(round(float(obs[base + _OFF_POWER]) * 10))
    tg = int(round(float(obs[base + _OFF_TOUGHNESS]) * 10))
    tags = []
    if obs[base + _OFF_TAPPED]    > 0.5: tags.append("tapped")
    if obs[base + _OFF_ATTACKING] > 0.5: tags.append("atk")
    if obs[base + _OFF_SICKNESS]  > 0.5: tags.append("sick")
    suffix = f"[{','.join(tags)}]" if tags else ""
    return f"{card} {pw}/{tg}{suffix}"


def _land_str(obs, base):
    card = _card_name(obs[base + _BF_CARD_OFF : base + _BF_CARD_OFF + _N_VOCAB])
    if card is None:
        return None
    return f"{card}{'*' if obs[base + _OFF_TAPPED] > 0.5 else ''}"


def _format_state(obs) -> str:
    a_life = int(round(float(obs[0]) * 20))
    b_life = int(round(float(obs[9]) * 20))

    a_creatures = [s for i in range(10)
                   if (s := _creature_str(obs, _BF_CREATURE_OFF + i * _BF_SLOT_SIZE))]
    b_creatures = [s for i in range(10)
                   if (s := _creature_str(obs, _BF_CREATURE_OFF + (i + 10) * _BF_SLOT_SIZE))]
    a_lands     = [s for i in range(10)
                   if (s := _land_str(obs, _BF_LAND_OFF + i * _BF_SLOT_SIZE))]
    b_lands     = [s for i in range(10)
                   if (s := _land_str(obs, _BF_LAND_OFF + (i + 10) * _BF_SLOT_SIZE))]
    hand        = [s for i in range(_HAND_SLOTS)
                   if (s := _card_name(obs[_HAND_OBS_START + i * _N_VOCAB :
                                          _HAND_OBS_START + (i + 1) * _N_VOCAB]))]

    return (
        "--- Battlefield ---\n"
        f"  [A] {a_life} life | creatures: {', '.join(a_creatures) or '—'}"
        f" | lands: {', '.join(a_lands) or '—'}\n"
        f"  [B] {b_life} life | creatures: {', '.join(b_creatures) or '—'}"
        f" | lands: {', '.join(b_lands) or '—'}\n"
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

def play(binary_path: str, model_path: str):
    model = MaskablePPO.load(model_path)
    env = PlayEnv(binary_path=binary_path, render_mode="human")
    obs, _ = env.reset()
    done = False

    print("=== Model (Player A) vs You (Player B) ===", flush=True)
    print("(type 'quit' to exit)\n", flush=True)

    while not done:
        num_choices = env._num_choices
        a_has_priority = obs[31] > 0.5
        cats     = np.round(obs[STATE_SIZE : STATE_SIZE + num_choices] * ACTION_CATEGORY_MAX).astype(int)
        card_ids = obs[STATE_SIZE + MAX_ACTIONS : STATE_SIZE + 2 * MAX_ACTIONS]

        is_mulligan_q = num_choices > 0 and all(c == 11 for c in cats)
        is_bottom_q   = num_choices > 0 and all(c == 12 for c in cats)
        is_search_q   = num_choices > 0 and all(c == 19 for c in cats)

        if a_has_priority:
            masks = env.action_masks() if USE_MASKABLE else None
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            action = int(action)
            if is_mulligan_q:
                label = "keep" if action == 0 else "mulligan"
            else:
                cat = int(cats[action]) if action < len(cats) else -1
                cid = float(card_ids[action]) if action < MAX_ACTIONS else -1.0 / _N_VOCAB
                label = _action_label(cat, cid)
            print(f"[Model/A] {label}", flush=True)
        else:
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
                active = "A" if obs[30] > 0.5 else "B"
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
    if reward > 0:
        print("=== Model (A) wins! ===")
    elif reward < 0:
        print("=== You (B) win! ===")
    else:
        print("=== Draw ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=BINARY)
    parser.add_argument("--model", required=True, help="Path to trained model .zip")
    args = parser.parse_args()
    play(args.binary, args.model)
