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
                 _STACK_SLOT_SIZE, _STACK_SLOTS, _STACK_MODE_SLOTS,
                 _STACK_TGT_SLOTS, _STACK_TGT_FIELDS,
                 _GY_SLOT_SIZE, _HAND_SLOT_SIZE,
                 _LIBRARY_CTX_START, _CUR_TURN_IDX, MAX_HAND_SLOTS,
                 _PENDING_DECISION_START,
                 _slot_card_idx, _ACTION_CARD_ID_NULL)
from card_costs import (N_CARD_TYPES, _VOCAB_NAMES as _CARD_NAMES,
                        _CARD_COST_MATRIX, _LAND_VOCAB_IDS)

# ── Engine constants (card identity is a single normalized id float per slot) ──
STACK_SLOT_SIZE = _STACK_SLOT_SIZE                 # ctrl + card-id + is_spell + modes + targets (25)
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
_OFF_LOYALTY = 10                                  # planeswalker loyalty (loyalty/10)
_OFF_CARD_ID = 11                                  # offset of the card-id float within a permanent slot

_NULL_SENTINEL = _ACTION_CARD_ID_NULL              # -1.0 / N_CARD_TYPES
_TOKEN_IDX = N_CARD_TYPES - 1

# Action-metadata categories that constitute a mandatory attacker/blocker loop.
MANDATORY_CATS = frozenset({2, 3, 4, 5})

# Action-category / step / zone-ref display names are generated from the C++
# enums by train/gen_enums.py — the single source of truth for both the integer
# values and these names. Re-run that script after changing the C++ enums.
from _enums import _CAT_NAMES, _STEP_NAMES, _REF_NAMES, REF_ZONE_MAX  # noqa: E402

_MANA_COLORS = ("W", "U", "B", "R", "G", "C")

# ── Controller-label vocabulary ───────────────────────────────────────────────
# The action/stack controller flag is RELATIVE TO THE VIEWER (controller_is_self),
# so decode can only speak in perspective-relative words: a permanent/action is
# the viewer's "own", the "opp"-onent's, or (on the stack) "self"/"opponent".
# These maps let callers substitute their own vocabulary (e.g. "You"/"Opp")
# through the one shared code path. Passing nothing keeps today's wording, so
# decode's default output is byte-for-byte unchanged.
#
#   "own" / "opp"      — the suffix appended to action descriptions
#   "self" / "opponent" — the stack-entry controller field in decode_game_state
SELF_OPP_LABELS = {
    "own": "own",
    "opp": "opp",
    "self": "self",
    "opponent": "opponent",
}


# ── Card-name helpers ─────────────────────────────────────────────────────────

def card_index_to_name(idx):
    if idx == _TOKEN_IDX:
        return "Token"
    if 0 <= idx < len(_CARD_NAMES):
        return _CARD_NAMES[idx]
    return f"?({idx})"


# ── Oracle-text lookup (for the TUI card-inspect popup) ────────────────────────

import os  # noqa: E402
import re  # noqa: E402

_CARDS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin", "resources", "cardsfolder")


def _name_to_uid(name):
    """Mirror src/parse.cpp name_to_uid: lowercase, space/hyphen -> '_', drop the rest."""
    return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_").replace("-", "_"))


def _resolve_script_path(uid):
    """Script file the engine would load for `uid` (mirrors src/card_db.cpp):
    the exact `<uid>.txt`, else a double-faced card's combined `<uid>_*.txt`."""
    if not uid:
        return None
    direct = os.path.join(_CARDS_DIR, uid[0], f"{uid}.txt")
    if os.path.exists(direct):
        return direct
    letter_dir = os.path.join(_CARDS_DIR, uid[0])
    if os.path.isdir(letter_dir):
        prefix = uid + "_"
        for fn in sorted(os.listdir(letter_dir)):
            if fn.startswith(prefix) and fn.endswith(".txt"):
                return os.path.join(letter_dir, fn)
    return None


_ORACLE_CACHE = {}


def card_oracle_text(card_idx):
    r"""Oracle text for a vocab card id, or '' when unavailable.

    Reads the card's Forge script `Oracle:` line (with `\n` expanded), resolving
    DFC combined filenames the way the engine does; result cached per id. A token
    (the shared TOKEN_SENTINEL id) has no named script, so returns ''."""
    if card_idx in _ORACLE_CACHE:
        return _ORACLE_CACHE[card_idx]
    text = ""
    if 0 <= card_idx < len(_CARD_NAMES) and card_idx != _TOKEN_IDX:
        path = _resolve_script_path(_name_to_uid(_CARD_NAMES[card_idx]))
        if path:
            try:
                with open(path) as f:
                    for raw in f:
                        if raw.startswith("Oracle:"):
                            text = raw[len("Oracle:"):].strip().replace("\\n", "\n")
                            break
            except OSError:
                pass
    _ORACLE_CACHE[card_idx] = text
    return text


def onehot_to_card(state, base):
    """Decode the card-id float at `base` to a card name, or None if empty."""
    idx = _slot_card_idx(state, base)
    return card_index_to_name(idx) if idx >= 0 else None


def onehot_to_index(state, base):
    """Decode the card-id float at `base` to its vocab index, or -1 if empty."""
    idx = _slot_card_idx(state, base)
    return idx if idx >= 0 else -1


# Border colors by MTG color, for TUI card rendering. Black renders as a muted
# purple-grey (pure black is invisible on a dark terminal); lands are neutral
# grey and colorless (nonland) cards brown, per the color-identity display.
_COLOR_BORDER = {"W": "#efe6c8", "U": "#3f7fd6", "B": "#8a7fa0",
                 "R": "#d64b3b", "G": "#42ae5a"}
_LAND_BORDER = "#8a8a8a"
_COLORLESS_BORDER = "#9a6a38"


def card_border_colors(card_idx):
    """Border color(s) for a card by its color identity (from its cast cost).

    Lands → [grey]; a colorless nonland → [brown]; otherwise one entry per
    colored pip (W,U,B,R,G) present in the mana cost, so a multicolor card
    returns several colors for the caller to split across the border edges.
    """
    if card_idx in _LAND_VOCAB_IDS:
        return [_LAND_BORDER]
    if 0 <= card_idx < len(_CARD_COST_MATRIX):
        cost = _CARD_COST_MATRIX[card_idx]
        colors = [_COLOR_BORDER[c] for i, c in enumerate("WUBRG") if cost[i] > 0]
        if colors:
            return colors
    return [_COLORLESS_BORDER]


def card_from_id(val):
    """Decode a card name from a normalised action card-ID float (None if null)."""
    idx = int(round(float(val) * N_CARD_TYPES))
    if 0 <= idx < len(_CARD_NAMES) and _CARD_NAMES[idx]:
        return card_index_to_name(idx)
    return None


def decode_hand(obs):
    """Return the list of card names in the priority player's hand slots."""
    names = []
    for slot in range(MAX_HAND_SLOTS):
        name = onehot_to_card(obs, _HAND_START + slot * _HAND_SLOT_SIZE)
        if name is not None:
            names.append(name)
    return names


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


def decode_turn(state):
    """Decode the current turn as the sequential 1-based display number.

    The state vector carries the engine's internal ``Game::turn`` (0-based,
    incremented once per player-turn) as ``turn / 50``; the engine's narrative
    ``-------- TURN N --------`` headers display it 1-based (A=1, B=2, A=3, ...),
    so add 1 here to keep every Python-side turn display in agreement.
    During the pregame (mulligans) the internal counter is still 0, so this
    returns 1 — callers that want a pregame marker must detect mulligan/bottom
    decisions themselves (see `is_mulligan` / `is_bottom`).
    """
    return int(round(float(state[_IDX_TURN]) * 50)) + 1


def decode_step(state):
    """Decode the current step from the one-hot at state[18:31]."""
    step_vec = state[18:31]
    idx = int(np.argmax(step_vec))
    if step_vec[idx] < 0.5:
        return "Unknown"
    return _STEP_NAMES[idx] if idx < len(_STEP_NAMES) else f"Step({idx})"


def _decode_permanents(state, start, count=48, counters=None, token_names=None):
    """Decode permanent slots into a list of dicts (non-empty only).

    `counters` is the per-slot typed-counter summary list the engine emits
    under --narrative (env._perm_counters side-channel; slot-aligned with the
    state vector). Loyalty entries are dropped — loyalty has a dedicated
    serialized field and its own display.

    `token_names` is the per-slot token-name list (env._perm_token_names,
    also narrative-only, slot-aligned). Every token shares the generic
    TOKEN_SENTINEL card id in the state vector, so when a slot names a token
    its "name" comes from here instead of the generic "Token"."""
    perms = []
    for i in range(count):
        base = start + i * PERM_SLOT_SIZE
        idx = onehot_to_index(state, base + _OFF_CARD_ID)
        if idx < 0:
            continue
        name = card_index_to_name(idx)
        if token_names is not None and i < len(token_names) and token_names[i]:
            name = token_names[i]
        p = {"name": name, "card_idx": idx}
        if counters is not None and i < len(counters) and counters[i]:
            parts = [c for c in (s.strip() for s in counters[i].split(","))
                     if c and not c.startswith("loyalty:")]
            if parts:
                p["counters"] = ", ".join(parts)
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
        loyalty = int(round(state[base + _OFF_LOYALTY] * 10))
        if loyalty != 0:
            p["loyalty"] = loyalty
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


def _decode_stack(state, labels=SELF_OPP_LABELS):
    """Decode the stack (12 slots x 25: ctrl + card id + is_spell + chosen-mode
    multi-hot(6) + 4 announced-target sub-slots of [present, is_player, ctrl, card id])."""
    entries = []
    for i in range(_STACK_SLOTS):
        base = _STACK_START + i * STACK_SLOT_SIZE
        idx = onehot_to_index(state, base + 1)  # skip controller_is_self float
        if idx < 0:
            continue
        modes = [m for m in range(_STACK_MODE_SLOTS) if state[base + 3 + m] > 0.5]
        targets = []
        target_refs = []
        for t in range(_STACK_TGT_SLOTS):
            tbase = base + 3 + _STACK_MODE_SLOTS + t * _STACK_TGT_FIELDS
            if state[tbase] < 0.5:  # present flag
                continue
            is_self = state[tbase + 2] > 0.5
            is_player = state[tbase + 1] > 0.5
            ctrl = labels["self"] if is_self else labels["opponent"]
            tidx = -1 if is_player else onehot_to_index(state, tbase + 3)
            if is_player:
                targets.append(f"{ctrl} (player)")
            else:
                tname = card_index_to_name(tidx) if tidx >= 0 else "?"
                targets.append(f"{tname} ({ctrl})")
            # Structured form for UIs that highlight the target on the board.
            # `is_self` is viewer-relative (like the rest of the state vector); a
            # mirrored decode must flip it to the human frame at the call site.
            target_refs.append({"is_player": is_player, "is_self": is_self,
                                "card_idx": tidx})
        entries.append({
            "name": card_index_to_name(idx),
            "card_idx": idx,
            "controller": labels["self"] if state[base] > 0.5 else labels["opponent"],
            "is_spell": state[base + 2] > 0.5,
            "modes": modes,       # chosen modal mode indices (empty = not modal)
            "targets": targets,   # announced targets, human-readable
            "target_refs": target_refs,  # structured targets (see note above)
        })
    return entries


def decode_game_state(state, labels=SELF_OPP_LABELS, perm_counters=None,
                      perm_token_names=None):
    """Decode the full state vector into a human-readable dict.

    `labels` substitutes the controller words used in stack entries (and any
    other perspective-relative wording); the default keeps today's output.
    `perm_counters` is the (self_slots, opp_slots) counter-summary side-channel
    (env._perm_counters, narrative mode only); when given, battlefield dicts
    gain a "counters" entry rendered by fmt_perm.
    `perm_token_names` is the (self_slots, opp_slots) token-name side-channel
    (env._perm_token_names, narrative mode only); when given, a token
    permanent's "name" is its real token name instead of the generic "Token".
    """
    is_active = state[31] > 0.5
    is_player_a = state[32] > 0.5
    return {
        "priority_player": "Player A" if is_player_a else "Player B",
        "priority_is_a": bool(is_player_a),
        "is_active_player": bool(is_active),
        "active_is_a": (is_active == is_player_a),
        "step": decode_step(state),
        "turn": decode_turn(state),
        "stack_size": int(round(state[33] * 10)),
        "self": _decode_player(state, 0),
        "opponent": _decode_player(state, 9),
        "self_library": int(round(state[_IDX_SELF_LIB] * 60)),
        "opp_library": int(round(state[_IDX_OPP_LIB] * 60)),
        "self_battlefield": _decode_permanents(
            state, _SELF_PERM_START,
            counters=perm_counters[0] if perm_counters else None,
            token_names=perm_token_names[0] if perm_token_names else None),
        "opp_battlefield": _decode_permanents(
            state, _OPP_PERM_START,
            counters=perm_counters[1] if perm_counters else None,
            token_names=perm_token_names[1] if perm_token_names else None),
        "stack": _decode_stack(state, labels),
        "self_hand": _decode_hand(state),
        "self_graveyard": _decode_graveyard(state, _GY_START),
        "opp_graveyard": _decode_graveyard(state, _OPP_GY_START),
        "pending_decision": _decode_pending_decision(state),
    }


def _decode_pending_decision(state):
    """The spell/ability currently making a mid-resolution choice, or None.

    Returns {"name", "card_idx", "is_self"} for the source of the pending
    target/dig/search/discard/modal choice. The source may not be on the stack
    yet (targets are announced before the spell moves there), so this is the
    only place the observation shows WHAT is asking for the current choice.
    """
    idx = _slot_card_idx(state, _PENDING_DECISION_START)
    if idx < 0:
        return None
    return {"name": card_index_to_name(idx), "card_idx": idx,
            "is_self": float(state[_PENDING_DECISION_START + 1]) > 0.5}


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


def action_zone_refs(obs, num_choices=MAX_ACTIONS):
    """Integer ActionRefZone per legal action (0 = REF_NONE, no referenced entity).

    Display names for the values live in _REF_NAMES.
    """
    raw = obs[STATE_SIZE + 3 * MAX_ACTIONS: STATE_SIZE + 3 * MAX_ACTIONS + num_choices]
    return np.round(raw * REF_ZONE_MAX).astype(int)


def _ctrl_str(ctrl_val, labels=SELF_OPP_LABELS):
    """Resolve a viewer-relative controller flag to a label word, or None.

    The engine emits the null sentinel (a small *negative* value) for actions
    with no entity owner, and 0.0 / 1.0 for opponent / self. So anything below
    the (negative) sentinel midpoint is "no owner"; 1.0 is self; 0.0 is opponent.
    (The old `> _NULL_SENTINEL + 0.01` test mis-classified the 0.0 = opponent
    case as None, so opponent-owned targets never got an "(opp)" tag.)

    `labels` substitutes the words used for the two roles ("own"/"opp" by
    default); pass a custom map to render e.g. "You"/"Opp".
    """
    v = float(ctrl_val)
    if v < _NULL_SENTINEL / 2:        # null sentinel (negative) → no owner info
        return None
    return labels["own"] if v > 0.5 else labels["opp"]


def describe_action(cat, card_name, ctrl_str, labels=SELF_OPP_LABELS):
    """Build a concise human-readable action description.

    `ctrl_str` is the already-resolved controller word (see `_ctrl_str`) and is
    used verbatim; `labels` is accepted for signature symmetry with the other
    formatters so a caller can pass one label map everywhere — it does not
    re-map an already-resolved `ctrl_str`.
    """
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
        if name:
            return f"Target {name}{owner}"
        # No card → a player target. The controller word tells us which player;
        # the exact "Target Player B (N life)" wording comes from the engine's
        # description block when available (this is the metadata-only fallback).
        if ctrl_str == labels["own"]:
            return "Target yourself"
        if ctrl_str == labels["opp"]:
            return "Target opponent"
        return "Target player"
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


def decode_actions(cats_int, card_ids, ctrl, num_choices, public_flags=None,
                   labels=SELF_OPP_LABELS, descriptions=None):
    """Decode the per-action arrays into a list of dicts.

    Each dict: index, category (int), category_name, card (name or None),
    card_idx (vocab index or -1), controller ('own'|'opp'|None), description,
    card_is_public (bool — card identity publicly known, e.g. a revealed tutor).

    `public_flags` is the per-action card_is_public array (env._action_public);
    None means "unknown", treated as not-public. `labels` substitutes the
    controller wording (default keeps today's "own"/"opp").

    `descriptions` is the per-action text the engine emits under --narrative
    (env._action_descriptions). When present and non-empty it is the exact
    CLI/GUI label and is used verbatim — so player targets, Sylvan Library
    pay-vs-return, charm modes, etc. read correctly. When absent, the label is
    reconstructed from the numeric metadata via `describe_action`.
    """
    actions = []
    for i in range(num_choices):
        cat = int(cats_int[i])
        card_idx = int(round(float(card_ids[i]) * N_CARD_TYPES))
        card_name = card_index_to_name(card_idx) if card_idx >= 0 else None
        ctrl_str = _ctrl_str(float(ctrl[i]), labels)
        is_public = bool(public_flags[i] > 0.5) if public_flags is not None else False
        engine_desc = (descriptions[i] if descriptions is not None
                       and i < len(descriptions) and descriptions[i].strip() else None)
        if engine_desc is not None:
            desc = engine_desc
        elif cat == 11:
            # Mulligan query: index 0 = keep, index 1 = mulligan.
            desc = "Keep hand" if i == 0 else "Mulligan"
        else:
            desc = describe_action(cat, card_name, ctrl_str, labels)
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


def decode_actions_from_obs(obs, num_choices, public_flags=None,
                            labels=SELF_OPP_LABELS, descriptions=None):
    """Convenience: decode actions straight from a full observation vector.

    `public_flags` (env._action_public) and `descriptions`
    (env._action_descriptions) are side-channels not stored in obs; pass them
    through so revealed-card choices aren't redacted and so engine-authored
    labels are used when available. `labels` substitutes controller wording.
    """
    return decode_actions(action_categories(obs, num_choices),
                          action_card_ids(obs), action_ctrls(obs), num_choices,
                          public_flags, labels, descriptions)


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


def fmt_stack_entry(e):
    """Format a decoded stack-entry dict (from _decode_stack)."""
    kind = "spell" if e["is_spell"] else "ability"
    s = f"{e['name']} ({kind}, {e['controller']})"
    if e.get("modes"):
        s += f" [modes {','.join(str(m) for m in e['modes'])}]"
    if e.get("targets"):
        s += f" -> {'; '.join(e['targets'])}"
    return s


def format_state_lines(gs):
    """Render a decoded game-state dict as a list of compact display lines.

    Single source for the human-readable board dump shared by the test harness
    (`print_state`) and `train.py observe --verbose`. `gs` is the dict returned
    by `decode_game_state`.
    """
    lines = [
        f"Priority: {gs['priority_player']}"
        f" ({'active' if gs['is_active_player'] else 'non-active'})",
        f"Step: {gs['step']}",
    ]
    hand_names = [c["name"] for c in gs["self_hand"]]
    lines.append(f"Self:  {gs['self']['life']} life"
                 f" | mana: {fmt_mana(gs['self']['mana'])}"
                 f" | hand({gs['self']['hand_count']}): {', '.join(hand_names) or '(empty)'}")
    lines.append(f"Opp:   {gs['opponent']['life']} life"
                 f" | mana: {fmt_mana(gs['opponent']['mana'])}"
                 f" | hand count: {gs['opponent']['hand_count']}")
    if gs["self_battlefield"]:
        lines.append(f"Self BF:  {' | '.join(fmt_perm(p) for p in gs['self_battlefield'])}")
    if gs["opp_battlefield"]:
        lines.append(f"Opp BF:   {' | '.join(fmt_perm(p) for p in gs['opp_battlefield'])}")
    if gs["stack"]:
        lines.append(f"Stack: {' -> '.join(fmt_stack_entry(e) for e in gs['stack'])}")
    if gs["self_graveyard"]:
        lines.append(f"Self GY: {', '.join(gs['self_graveyard'])}")
    if gs["opp_graveyard"]:
        lines.append(f"Opp GY:  {', '.join(gs['opp_graveyard'])}")
    return lines


def format_action_lines(actions):
    """Enumerated legal-action lines (the 'Actions:' menu, shared transcript)."""
    return [f"  {a['index']:>2}: {a['description']}" for a in actions]


def format_decision_block(decision_idx, gs, actions):
    """Shared per-decision transcript block: header + state dump + action menu.

    Returns a list of lines: a leading blank line, '--- Decision N ---', the
    2-space-indented board state, 'Actions:', then the enumerated legal actions.
    Used by both the test harness and `observe --verbose` so the transcript
    format is identical across tools.
    """
    lines = ["", f"--- Decision {decision_idx} ---"]
    lines += [f"  {ln}" for ln in format_state_lines(gs)]
    lines.append("  Actions:")
    lines += format_action_lines(actions)
    return lines


def format_chosen_action(label, choice, actions):
    """The '>> <label>: <i> (<desc>)' line for the action an agent picked."""
    desc = actions[choice]["description"] if 0 <= choice < len(actions) else "?"
    return f"  >> {label}: {choice} ({desc})"


def format_narrative_block(lines):
    """The '--- Narrative ---' block wrapping a list of engine narrative lines."""
    return ["--- Narrative ---"] + [f"  {ln}" for ln in lines]


def fmt_perm(p):
    """Format a single decoded permanent dict."""
    s = p["name"]
    if "power" in p:
        s += f" [{p['power']}/{p['toughness']}"
        if "damage" in p:
            s += f", {p['damage']}dmg"
        s += "]"
    if "loyalty" in p:
        s += f" [loy {p['loyalty']}]"
    if "counters" in p:
        s += f" [{p['counters']}]"
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
