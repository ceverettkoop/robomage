"""Shared decoders for the RoboMage observation vector.

Single source of truth for turning the engine's state/action payload into
human-readable structures. Consumed by play.py, test_harness.py and tui_game.py.

Dependency-light on purpose: numpy + the layout constants in env.py + the card
vocabulary in card_costs.py. It deliberately does NOT import train.py (which
pulls in torch / stable-baselines3), so importing this module stays cheap.

Two coordinate systems appear here:
  * `state` — the first STATE_SIZE floats of an observation (the perspective-
    normalised game state). All zone decoders below operate on `state`.
  * `obs`   — the full observation: `state` followed by the per-action metadata
    (categories, card ids, controller flags). The `action_*` helpers read that
    appended block.
"""

import numpy as np

from env import (STATE_SIZE, MAX_ACTIONS, ACTION_CATEGORY_MAX,
                 _SELF_PERM_START, _OPP_PERM_START, _STACK_START,
                 _GY_START, _HAND_START, _PERM_SLOT_SIZE as PERM_SLOT_SIZE,
                 _STACK_SLOT_SIZE, _GY_SLOT_SIZE, _HAND_SLOT_SIZE,
                 _LIBRARY_CTX_START, _CUR_TURN_IDX,
                 _slot_card_idx, _ACTION_CARD_ID_NULL)
from card_costs import N_CARD_TYPES, _VOCAB_NAMES as _CARD_NAMES

# ── Engine constants (card identity is a single normalized id float per slot) ──
STACK_SLOT_SIZE = _STACK_SLOT_SIZE                 # ctrl(1) + card-id(1) + is_spell(1)
GY_SLOT_SIZE = _GY_SLOT_SIZE                       # card-id only
_OPP_GY_START = _GY_START + 64 * GY_SLOT_SIZE      # opp graveyard begins after 64 self slots

# State-vector context indices (derived from env layout)
_IDX_SELF_LIB = _LIBRARY_CTX_START                 # self_library_ct / 60
_IDX_OPP_LIB = _LIBRARY_CTX_START + 1              # opp_library_ct  / 60
_IDX_TURN = _CUR_TURN_IDX                          # turn / 50

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
_OFF_CARD_ID = 10                                  # offset of the card-id float within a permanent slot

_NULL_SENTINEL = _ACTION_CARD_ID_NULL              # -1.0 / N_CARD_TYPES
_TOKEN_IDX = N_CARD_TYPES - 1

# Action-metadata categories that constitute a mandatory attacker/blocker loop.
MANDATORY_CATS = frozenset({2, 3, 4, 5})

_CAT_NAMES = {
    0: "PASS", 1: "MANA", 2: "SEL_ATK", 3: "CONF_ATK",
    4: "SEL_BLK", 5: "CONF_BLK", 6: "ACTIVATE", 7: "CAST",
    8: "TARGET", 9: "LAND", 10: "OTHER", 11: "MULLIGAN", 12: "BOTTOM_CARD",
    13: "MANA_W", 14: "MANA_U", 15: "MANA_B", 16: "MANA_R", 17: "MANA_G",
    18: "MANA_C", 19: "SEARCH", 20: "TOP_LIB", 21: "SHUFFLE", 22: "PAYING",
    23: "DIG", 24: "SB_IN", 25: "SB_OUT", 26: "SB_DONE",
}

_STEP_NAMES = [
    "Untap", "Upkeep", "Draw", "First Main", "Begin Combat",
    "Declare Atk", "Declare Blk", "First Strike Dmg", "Combat Dmg",
    "End Combat", "Second Main", "End Step", "Cleanup",
]

_MANA_COLORS = ("W", "U", "B", "R", "G", "C")


# ── Card-name helpers ─────────────────────────────────────────────────────────

def card_index_to_name(idx):
    if idx == _TOKEN_IDX:
        return "Token"
    if 0 <= idx < len(_CARD_NAMES):
        return _CARD_NAMES[idx]
    return f"?({idx})"


def onehot_to_card(state, base):
    """Decode the card-id float at `base` to a card name, or None if empty."""
    idx = _slot_card_idx(state, base)
    return card_index_to_name(idx) if idx >= 0 else None


def onehot_to_index(state, base):
    """Decode the card-id float at `base` to its vocab index, or -1 if empty."""
    idx = _slot_card_idx(state, base)
    return idx if idx >= 0 else -1


def card_from_id(val):
    """Decode a card name from a normalised action card-ID float (None if null)."""
    idx = int(round(float(val) * N_CARD_TYPES))
    if 0 <= idx < len(_CARD_NAMES) and _CARD_NAMES[idx]:
        return card_index_to_name(idx)
    return None


# ── State decoders (operate on `state` = obs[:STATE_SIZE]) ─────────────────────

def _decode_player(state, offset):
    """Decode a 9-float player block into a dict."""
    return {
        "life": int(round(state[offset] * 20)),
        "hand_count": int(round(state[offset + 1] * 10)),
        "poison": int(round(state[offset + 2] * 10)),
        "mana": {c: int(round(state[offset + 3 + i] * 10))
                 for i, c in enumerate(_MANA_COLORS)},
    }


def decode_step(state):
    """Decode the current step from the one-hot at state[18:31]."""
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
        idx = onehot_to_index(state, base + _OFF_CARD_ID)
        if idx < 0:
            continue
        p = {"name": card_index_to_name(idx), "card_idx": idx}
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


def _decode_hand(state):
    """Decode self hand (10 slots x 128 one-hot) into a list of dicts."""
    cards = []
    for i in range(10):
        idx = onehot_to_index(state, _HAND_START + i * _HAND_SLOT_SIZE)
        if idx >= 0:
            cards.append({"name": card_index_to_name(idx), "card_idx": idx})
    return cards


def _decode_graveyard(state, start):
    """Decode a graveyard zone (64 slots x 128 one-hot) into card names."""
    cards = []
    for i in range(64):
        card = onehot_to_card(state, start + i * GY_SLOT_SIZE)
        if card is not None:
            cards.append(card)
    return cards


def _decode_stack(state):
    """Decode the stack (12 slots x 130: ctrl + card_onehot + is_spell)."""
    entries = []
    for i in range(12):
        base = _STACK_START + i * STACK_SLOT_SIZE
        idx = onehot_to_index(state, base + 1)  # skip controller_is_self float
        if idx < 0:
            continue
        entries.append({
            "name": card_index_to_name(idx),
            "card_idx": idx,
            "controller": "self" if state[base] > 0.5 else "opponent",
            "is_spell": state[base + 2] > 0.5,
        })
    return entries


def decode_game_state(state):
    """Decode the full state vector into a human-readable dict."""
    is_active = state[31] > 0.5
    is_player_a = state[32] > 0.5
    return {
        "priority_player": "Player A" if is_player_a else "Player B",
        "priority_is_a": bool(is_player_a),
        "is_active_player": bool(is_active),
        "active_is_a": (is_active == is_player_a),
        "step": decode_step(state),
        "turn": int(round(state[_IDX_TURN] * 50)),
        "stack_size": int(round(state[33] * 10)),
        "self": _decode_player(state, 0),
        "opponent": _decode_player(state, 9),
        "self_library": int(round(state[_IDX_SELF_LIB] * 60)),
        "opp_library": int(round(state[_IDX_OPP_LIB] * 60)),
        "self_battlefield": _decode_permanents(state, _SELF_PERM_START),
        "opp_battlefield": _decode_permanents(state, _OPP_PERM_START),
        "stack": _decode_stack(state),
        "self_hand": _decode_hand(state),
        "self_graveyard": _decode_graveyard(state, _GY_START),
        "opp_graveyard": _decode_graveyard(state, _OPP_GY_START),
    }


# ── Action decoders ───────────────────────────────────────────────────────────

def action_categories(obs, num_choices=MAX_ACTIONS):
    """Integer ActionCategory per legal action (from the obs metadata block)."""
    raw = obs[STATE_SIZE: STATE_SIZE + num_choices]
    return np.round(raw * ACTION_CATEGORY_MAX).astype(int)


def action_card_ids(obs):
    """Normalised card-ID float per action slot."""
    return obs[STATE_SIZE + MAX_ACTIONS: STATE_SIZE + 2 * MAX_ACTIONS]


def action_ctrls(obs):
    """controller_is_self float per action slot."""
    return obs[STATE_SIZE + 2 * MAX_ACTIONS: STATE_SIZE + 3 * MAX_ACTIONS]


def _ctrl_str(ctrl_val):
    if ctrl_val > 0.5:
        return "own"
    if ctrl_val > _NULL_SENTINEL + 0.01:
        return "opp"
    return None


def describe_action(cat, card_name, ctrl_str):
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
        return f"Tap {name} for {{{_MANA_COLORS[cat - 13]}}}"
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
        cat_name = _CAT_NAMES.get(cat, f"?({cat})")
        return f"{cat_name}: {name}{owner}" if name else cat_name


def decode_actions(cats_int, card_ids, ctrl, num_choices, public_flags=None):
    """Decode the per-action arrays into a list of dicts.

    Each dict: index, category (int), category_name, card (name or None),
    card_idx (vocab index or -1), controller ('own'|'opp'|None), description,
    card_is_public (bool — card identity publicly known, e.g. a revealed tutor).

    `public_flags` is the per-action card_is_public array (env._action_public);
    None means "unknown", treated as not-public.
    """
    actions = []
    for i in range(num_choices):
        cat = int(cats_int[i])
        card_idx = int(round(float(card_ids[i]) * N_CARD_TYPES))
        card_name = card_index_to_name(card_idx) if card_idx >= 0 else None
        ctrl_str = _ctrl_str(float(ctrl[i]))
        is_public = bool(public_flags[i] > 0.5) if public_flags is not None else False
        if cat == 11:
            # Mulligan query: index 0 = keep, index 1 = mulligan.
            desc = "Keep hand" if i == 0 else "Mulligan"
        else:
            desc = describe_action(cat, card_name, ctrl_str)
        actions.append({
            "index": i,
            "category": cat,
            "category_name": _CAT_NAMES.get(cat, f"?({cat})"),
            "card": card_name,
            "card_idx": card_idx if card_idx >= 0 else -1,
            "controller": ctrl_str,
            "description": desc,
            "card_is_public": is_public,
        })
    return actions


def decode_actions_from_obs(obs, num_choices, public_flags=None):
    """Convenience: decode actions straight from a full observation vector.

    `public_flags` (env._action_public) is a side-channel not stored in obs;
    pass it through so revealed-card choices aren't redacted as private.
    """
    return decode_actions(action_categories(obs, num_choices),
                          action_card_ids(obs), action_ctrls(obs), num_choices,
                          public_flags)


# ── Decision-type classification (all read the integer category array) ────────

def is_mulligan(cats):
    return len(cats) > 0 and all(c == 11 for c in cats)


def is_bottom(cats):
    return len(cats) > 0 and all(c == 12 for c in cats)


def is_search(cats):
    return len(cats) > 0 and all(c == 19 for c in cats)


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_mana(mana):
    """Format a mana-pool dict as a compact string."""
    parts = [f"{mana[c]}{c}" for c in _MANA_COLORS if mana.get(c, 0) > 0]
    return ", ".join(parts) if parts else "empty"


def fmt_perm(p):
    """Format a single decoded permanent dict."""
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
