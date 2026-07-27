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
    (categories, card ids, controller flags, zone refs, entity-slot refs). The
    `action_*` helpers read that appended block.
"""

import numpy as np

from env import (STATE_SIZE, MAX_ACTIONS, ACTION_CATEGORY_MAX,
                 ACT_CATS_START, ACT_IDS_START, ACT_CTRL_START,
                 ACT_ZONE_START, ACT_REFS_START, ACT_ORDS_START,
                 _SELF_PERM_START, _OPP_PERM_START, _STACK_START,
                 _GY_START, _EXILE_START, _HAND_START, _PERM_SLOT_SIZE as PERM_SLOT_SIZE,
                 _PERM_SLOTS, _PERM_CARD_OFF, _PERM_CHOSEN_NAME_OFF,
                 _PERM_RETURNABLE_OFF,
                 N_ENTITY_REF_SLOTS,
                 _STACK_SLOT_SIZE, _STACK_SLOTS, _STACK_MODE_SLOTS,
                 _STACK_XAMT_OFF, _STACK_QUAL_START, _STACK_QUALS,
                 _STACK_MODE_START, _STACK_TGT_START,
                 _STACK_TGT_SLOTS, _STACK_TGT_FIELDS,
                 _GY_SLOT_SIZE, _HAND_SLOT_SIZE,
                 _KNOWN_TOP_LIB_START, _KNOWN_TOP_LIB_SLOTS,
                 _KNOWN_TOP_LIB_SLOT_SIZE, _REVEALED_START, _REVEALED_SIZE,
                 _OPP_KNOWN_HAND_START, _OPP_KNOWN_HAND_SLOTS,
                 _OPP_KNOWN_HAND_SLOT_SIZE,
                 _LIBRARY_CTX_START, _CUR_TURN_IDX, MAX_HAND_SLOTS,
                 MAX_GY_SLOTS, MANDATORY_CATS,
                 _PENDING_DECISION_START, _MATCH_CTX_START,
                 _SELF_BLOCK_START, _OPP_BLOCK_START,
                 _PB_LIFE, _PB_HAND_CT, _PB_POISON, _PB_MANA, _PB_ENERGY,
                 _STEP_ONEHOT_START, _STEP_ONEHOT_SIZE,
                 _IS_ACTIVE_IDX, _SELF_IS_A_IDX, _STACK_SIZE_IDX,
                 _OFF_P1P1_NET, _OFF_OTHER_COUNTERS, _OFF_ATTACHED_TO,
                 _OFF_ATTACHED_BY, _OFF_ATTACK_TGT, _OFF_BLOCKING_TGT,
                 _OFF_IS_BLOCKED, _OFF_IS_PHASED_OUT, _OFF_KEYWORDS_START,
                 _EXTRAS_LANDS_SELF, _EXTRAS_LANDS_OPP, _EXTRAS_HAS_PRIORITY,
                 _EXTRAS_MONARCH_SELF, _EXTRAS_MONARCH_OPP,
                 _EXTRAS_BLESSING_SELF, _EXTRAS_BLESSING_OPP,
                 _EXTRAS_REVOLT_SELF, _EXTRAS_REVOLT_OPP,
                 _EXTRAS_EXTRA_TURNS_SELF, _EXTRAS_EXTRA_TURNS_OPP,
                 _EXTRAS_IS_DAY, _EXTRAS_IS_NIGHT, _EXTRAS_MC_ONEHOT_START,
                 _slot_card_idx, _ACTION_CARD_ID_NULL)
from card_costs import (N_CARD_TYPES, _VOCAB_NAMES as _CARD_NAMES,
                        _CARD_COST_MATRIX, _LAND_VOCAB_IDS)

# ── Engine constants (card identity is a single normalized id float per slot) ──
STACK_SLOT_SIZE = _STACK_SLOT_SIZE                 # ctrl + card-id + is_spell + x/quals + modes + targets (37)
GY_SLOT_SIZE = _GY_SLOT_SIZE                       # card-id only
_OPP_GY_START = _GY_START + MAX_GY_SLOTS * GY_SLOT_SIZE  # opp gy begins after the self slots
# Exile blocks mirror the graveyard layout (MAX_GY_SLOTS card-id slots per side,
# recency-ordered); the shared _decode_graveyard decoder handles them too.
_OPP_EXILE_START = _EXILE_START + MAX_GY_SLOTS * GY_SLOT_SIZE  # opp exile after the self slots

# State-vector context indices (derived from env layout)
_IDX_SELF_LIB = _LIBRARY_CTX_START                 # self_library_ct / 60
_IDX_OPP_LIB = _LIBRARY_CTX_START + 1              # opp_library_ct  / 60
_IDX_POST_BOARD = _LIBRARY_CTX_START + 2           # is_post_board (0/1; game 2+ of bo3)
_IDX_TURN = _CUR_TURN_IDX                          # turn / 50

# Bo3 match-context indices (self-perspective; see src/machine_io.h layout).
_IDX_GAME_NUMBER = _MATCH_CTX_START                # game_number / 3
_IDX_SELF_WINS   = _MATCH_CTX_START + 1            # self_match_wins / 2
_IDX_OPP_WINS    = _MATCH_CTX_START + 2            # opp_match_wins  / 2
_IDX_SIDEBOARD   = _MATCH_CTX_START + 3            # is_sideboard_phase (0/1)

# Permanent slot field offsets (src/machine_io.h perm-slot layout; the enriched
# fields at 11-18, the keyword multi-hot at 19-34 and the card id at 35 are
# imported from env.py above — _OFF_P1P1_NET .. _OFF_KEYWORDS_START).
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
_OFF_CHOSEN_NAME = _PERM_CHOSEN_NAME_OFF           # chosen-name card-id float (3rd-last in the slot, 35)
_OFF_RETURNABLE = _PERM_RETURNABLE_OFF             # returnable-exile card-id float (2nd-last in the slot, 36)
_OFF_CARD_ID = _PERM_CARD_OFF                      # card-id float, always LAST in the slot (37)

# Stack-slot cast-qualifier display names, in serialized order (the stack slot's
# [4-10] flags; see src/machine_io.h). All 0.0 for abilities.
_STACK_QUAL_NAMES = ("copy", "kicked", "flashback", "evoke", "escape",
                     "offspring", "impending")

_NULL_SENTINEL = _ACTION_CARD_ID_NULL              # -1.0 / N_CARD_TYPES
_TOKEN_IDX = N_CARD_TYPES - 1

# MANDATORY_CATS (the mandatory attacker/blocker loop categories) is defined
# once in env.py, built from the generated CAT_* names, and imported above so
# there is a single source for it.

# Action-category / step / zone-ref display names AND the name-keyed CAT_*
# integers are generated from the C++ enums by train/gen_enums.py — the single
# source of truth for both the values and these names. Re-run it after changing
# the C++ enums.
from _enums import (_CAT_NAMES, _STEP_NAMES, _REF_NAMES, REF_ZONE_MAX,  # noqa: E402
                    OPTION_ORDINAL_MAX, _MC_NAMES, _OBS_KEYWORDS,
                    CAT_PASS_PRIORITY, CAT_CAST_SPELL, CAT_PLAY_LAND,
                    CAT_ACTIVATE_ABILITY, CAT_SELECT_TARGET, CAT_SELECT_ATTACKER,
                    CAT_CONFIRM_ATTACKERS, CAT_SELECT_BLOCKER,
                    CAT_CONFIRM_BLOCKERS, CAT_MULLIGAN, CAT_KEEP_HAND,
                    CAT_BOTTOM_DECK_CARD,
                    CAT_MANA_W, CAT_MANA_C, CAT_SEARCH_LIBRARY, CAT_TOP_LIBRARY,
                    CAT_PAYING_COSTS, CAT_DIG_CHOICE, CAT_OTHER_CHOICE,
                    CAT_SHUFFLE, CAT_DONT_SHUFFLE, CAT_EXILE_FROM_YARD)

# Categories whose choice references a specific board/stack entity, where an
# "@slotN" suffix disambiguates same-named cards: select attacker/blocker,
# activate, target, pay cost.
_SLOT_REF_CATS = frozenset({CAT_SELECT_ATTACKER, CAT_SELECT_BLOCKER,
                            CAT_ACTIVATE_ABILITY, CAT_SELECT_TARGET,
                            CAT_PAYING_COSTS})

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
    """Mirror src/parse.cpp name_to_uid: lowercase, space/hyphen/slash -> '_', drop the
    rest ('/' is a split-card separator, so "Dead/Gone" -> "dead_gone")."""
    return re.sub(r"[^a-z0-9_]", "",
                  name.lower().replace(" ", "_").replace("-", "_").replace("/", "_"))


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


_SCRIPT_FIELD_CACHE = {}


def _card_script_field(card_idx, field):
    """First `<field>:` line from a vocab card's Forge script, stripped, or ''.

    Shares the DFC-aware script resolution and vocab bounds/token guards with
    card_oracle_text; result cached per (id, field)."""
    key = (card_idx, field)
    if key in _SCRIPT_FIELD_CACHE:
        return _SCRIPT_FIELD_CACHE[key]
    val = ""
    if 0 <= card_idx < len(_CARD_NAMES) and card_idx != _TOKEN_IDX:
        path = _resolve_script_path(_name_to_uid(_CARD_NAMES[card_idx]))
        if path:
            prefix = field + ":"
            try:
                with open(path) as f:
                    for raw in f:
                        if raw.startswith(prefix):
                            val = raw[len(prefix):].strip()
                            break
            except OSError:
                pass
    _SCRIPT_FIELD_CACHE[key] = val
    return val


def card_mana_cost(card_idx):
    """Mana cost string from the card script (e.g. '1 G'); '' for no-cost cards
    (lands carry Forge's 'no cost' sentinel)."""
    cost = _card_script_field(card_idx, "ManaCost")
    return "" if cost.lower() == "no cost" else cost


def card_types(card_idx):
    """Type line from the card script (e.g. 'Creature Bear', 'Basic Land'); ''
    when unavailable."""
    return _card_script_field(card_idx, "Types")


def fmt_mana_cost(cost):
    """Render a raw mana-cost string ('1 G') in MTG brace notation ('{1}{G}');
    an empty cost stays empty."""
    return "".join(f"{{{t}}}" for t in cost.split()) if cost else ""


def onehot_to_card(state, base):
    """Decode the card-id float at `base` to a card name, or None if empty."""
    idx = _slot_card_idx(state, base)
    return card_index_to_name(idx) if idx >= 0 else None


def onehot_to_index(state, base):
    """Decode the card-id float at `base` to its vocab index, or -1 if empty."""
    idx = _slot_card_idx(state, base)
    return idx if idx >= 0 else -1


# ── Entity-slot references ────────────────────────────────────────────────────
# Unified viewer-relative slot space (src/machine_io.h norm_ref): 0-47 self perm
# slots, 48-95 opp perm slots, 96-107 stack slots (0 = top), -1 = none. In the
# float state vector (and the obs action refs block) a ref is normalized
# (idx + 1) / N_ENTITY_REF_SLOTS, so 0.0 is the "none" sentinel.

def decode_slot_ref(val):
    """Decode a normalized entity-slot ref float to its slot index (-1 = none)."""
    return int(round(float(val) * N_ENTITY_REF_SLOTS)) - 1


def slot_ref_card_name(state, ref):
    """Resolve an entity-slot ref to the referenced slot's card name, or None.

    Reads the perm/stack sections of `state`; an out-of-range ref or an empty
    referenced slot yields None."""
    if ref < 0:
        return None
    if ref < _PERM_SLOTS:                      # self permanent slot
        base = _SELF_PERM_START + ref * PERM_SLOT_SIZE + _OFF_CARD_ID
    elif ref < 2 * _PERM_SLOTS:                # opp permanent slot
        base = _OPP_PERM_START + (ref - _PERM_SLOTS) * PERM_SLOT_SIZE + _OFF_CARD_ID
    elif ref < N_ENTITY_REF_SLOTS:             # stack slot (card id at offset 1)
        base = _STACK_START + (ref - 2 * _PERM_SLOTS) * STACK_SLOT_SIZE + 1
    else:
        return None
    return onehot_to_card(state, base)


def _ref_desc(state, ref):
    """Compact display for a slot ref: the referenced card's name when it
    resolves, else "slot N"."""
    name = slot_ref_card_name(state, ref)
    return name if name is not None else f"slot {ref}"


def entity_ref_features(state, ref):
    """The referenced slot's full serialized feature block, or None.

    Perm refs return the PERM_SLOT_SIZE-float slot; stack refs the
    STACK_SLOT_SIZE-float slot. Two refs whose blocks are float-identical are
    indistinguishable to any obs consumer — the basis of
    `menu_is_interchangeable`'s copy-collapse."""
    if ref < 0:
        return None
    if ref < _PERM_SLOTS:                      # self permanent slot
        base = _SELF_PERM_START + ref * PERM_SLOT_SIZE
        return state[base: base + PERM_SLOT_SIZE]
    if ref < 2 * _PERM_SLOTS:                  # opp permanent slot
        base = _OPP_PERM_START + (ref - _PERM_SLOTS) * PERM_SLOT_SIZE
        return state[base: base + PERM_SLOT_SIZE]
    if ref < N_ENTITY_REF_SLOTS:               # stack slot
        base = _STACK_START + (ref - 2 * _PERM_SLOTS) * STACK_SLOT_SIZE
        return state[base: base + STACK_SLOT_SIZE]
    return None


# Every in-state entity-ref field: the four per-perm-slot ref fields (both
# sides) and each stack announced-target sub-slot's ref. Enumerated once so
# state_ref_targets is a single fancy-index read.
_STACK_TGT_REF_OFF = 3  # sub-slot order: present, is_player, ctrl_is_self, slot_ref, card id


def _build_state_ref_field_idx():
    idx = []
    for start in (_SELF_PERM_START, _OPP_PERM_START):
        for s in range(_PERM_SLOTS):
            base = start + s * PERM_SLOT_SIZE
            idx.extend((base + _OFF_ATTACHED_TO, base + _OFF_ATTACHED_BY,
                        base + _OFF_ATTACK_TGT, base + _OFF_BLOCKING_TGT))
    for s in range(_STACK_SLOTS):
        tgt0 = _STACK_START + s * STACK_SLOT_SIZE + _STACK_TGT_START
        idx.extend(tgt0 + t * _STACK_TGT_FIELDS + _STACK_TGT_REF_OFF
                   for t in range(_STACK_TGT_SLOTS))
    return np.asarray(idx, dtype=np.intp)


_STATE_REF_FIELD_IDX = _build_state_ref_field_idx()


def state_ref_targets(state):
    """The set of entity-slot indices some in-state ref field points at
    (attachments, attack/blocking targets, announced stack targets)."""
    refs = np.round(np.asarray(state)[_STATE_REF_FIELD_IDX]
                    * N_ENTITY_REF_SLOTS).astype(int) - 1
    return set(refs[refs >= 0].tolist())


def hidden_info_fingerprint(state):
    """Per-side multisets of the card identities currently visible to the
    viewer — the "what hidden information has been revealed" fingerprint tree
    reuse compares between a search and a later decision.

    Returns ``(self_counts, opp_counts)``, int arrays of length N_CARD_TYPES.
    Self counts cover hand, permanents, graveyard, exile, own stack objects
    and the known-top-of-library ids; opp counts cover their permanents,
    graveyard, exile, stack objects and the known-opponent-hand ids. A card
    moving BETWEEN visible zones conserves its count (cast from hand, die to
    graveyard, draw a known top card), so a count exceeding an earlier
    fingerprint's means an identity became newly visible since — a draw, a
    play from a hidden hand, a mill, a tutor/scry reveal. Token permanents
    are excluded: the shared TOKEN sentinel id carries no hidden identity and
    tokens appear along fully public lines."""
    self_counts = np.zeros(N_CARD_TYPES, dtype=np.int32)
    opp_counts = np.zeros(N_CARD_TYPES, dtype=np.int32)

    def _add(counts, base, n, stride):
        for i in range(n):
            idx = _slot_card_idx(state, base + i * stride)
            if 0 <= idx < N_CARD_TYPES and idx != _TOKEN_IDX:
                counts[idx] += 1

    _add(self_counts, _HAND_START, MAX_HAND_SLOTS, _HAND_SLOT_SIZE)
    _add(self_counts, _SELF_PERM_START + _OFF_CARD_ID, _PERM_SLOTS,
         PERM_SLOT_SIZE)
    _add(self_counts, _GY_START, MAX_GY_SLOTS, GY_SLOT_SIZE)
    _add(self_counts, _EXILE_START, MAX_GY_SLOTS, GY_SLOT_SIZE)
    _add(self_counts, _KNOWN_TOP_LIB_START, _KNOWN_TOP_LIB_SLOTS,
         _KNOWN_TOP_LIB_SLOT_SIZE)
    _add(opp_counts, _OPP_PERM_START + _OFF_CARD_ID, _PERM_SLOTS,
         PERM_SLOT_SIZE)
    _add(opp_counts, _OPP_GY_START, MAX_GY_SLOTS, GY_SLOT_SIZE)
    _add(opp_counts, _OPP_EXILE_START, MAX_GY_SLOTS, GY_SLOT_SIZE)
    _add(opp_counts, _OPP_KNOWN_HAND_START, _OPP_KNOWN_HAND_SLOTS,
         _OPP_KNOWN_HAND_SLOT_SIZE)
    for i in range(_STACK_SLOTS):
        base = _STACK_START + i * STACK_SLOT_SIZE
        idx = _slot_card_idx(state, base + 1)  # card id; sentinel = empty slot
        if 0 <= idx < N_CARD_TYPES and idx != _TOKEN_IDX:
            (self_counts if state[base] > 0.5 else opp_counts)[idx] += 1
    return self_counts, opp_counts


# Border colors by MTG color, for TUI card rendering. Black renders as a muted
# purple-grey (pure black is invisible on a dark terminal); a colorless nonland
# card is brown, per the color-identity display. A mana-producing land is painted
# in the colors it can make (see _LAND_COLOR_BORDER below); a colorless/utility
# land that makes no colored mana falls back to neutral grey.
_COLOR_BORDER = {"W": "#efe6c8", "U": "#3f7fd6", "B": "#8a7fa0",
                 "R": "#d64b3b", "G": "#42ae5a"}
_LAND_BORDER = "#8a8a8a"
_COLORLESS_BORDER = "#9a6a38"

# Brighter, more saturated palette used for a land's produced colors, so a land
# that taps for a color reads in that color yet still stands apart from a
# same-color nonland card (which uses the muted _COLOR_BORDER above).
_LAND_COLOR_BORDER = {"W": "#fff4b0", "U": "#5aa9ff", "B": "#b98fd9",
                      "R": "#ff6a4d", "G": "#5ee06a"}

# Basic land subtype -> the color of mana it taps for.
_LAND_SUBTYPE_COLOR = {"Plains": "W", "Island": "U", "Swamp": "B",
                       "Mountain": "R", "Forest": "G"}

_LAND_COLOR_CACHE = {}


def _mana_spec_to_colors(spec):
    """WUBRG letters from a `Produced$` value (`G`, `Combo G W`, `Any`, `C`).

    `Any` yields all five colors; bare `C`, `Combo`, and numeric tokens
    contribute nothing (colorless mana carries no border color)."""
    out = set()
    for tok in spec.replace(",", " ").split():
        if tok == "Any":
            return set("WUBRG")
        if tok in _LAND_SUBTYPE_COLOR.values():
            out.add(tok)
    return out


def _fetch_spec_to_colors(spec):
    """WUBRG letters a fetch `ChangeType$` can retrieve. Basic subtypes map to
    their color; an any-basic target (`Land.Basic`) contributes all five."""
    out = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if tok.split(".")[0] == "Land" and "Basic" in tok:
            return set("WUBRG")
        if tok in _LAND_SUBTYPE_COLOR:
            out.add(_LAND_SUBTYPE_COLOR[tok])
    return out


def _land_color_letters(card_idx):
    """Colors of mana a land can supply, in WUBRG order (empty if colorless).

    Unions three script signals so basics, duals, and fetchlands all read
    correctly: an `AB$ Mana` `Produced$` clause, the land's basic land subtypes
    on its `Types:` line (duals like Tundra and the surveil lands read from
    their subtypes), and a fetch ability's `ChangeType$` targets (so a fetchland
    shows the colors it can fetch)."""
    if card_idx in _LAND_COLOR_CACHE:
        return _LAND_COLOR_CACHE[card_idx]
    found = set()
    if 0 <= card_idx < len(_CARD_NAMES) and card_idx != _TOKEN_IDX:
        path = _resolve_script_path(_name_to_uid(_CARD_NAMES[card_idx]))
        if path:
            try:
                with open(path) as f:
                    for raw in f:
                        line = raw.rstrip("\n")
                        if line.startswith("Types:"):
                            for tok in line[len("Types:"):].split():
                                if tok in _LAND_SUBTYPE_COLOR:
                                    found.add(_LAND_SUBTYPE_COLOR[tok])
                        m = re.search(r"Produced\$\s*([^|]+)", line)
                        if m:
                            found |= _mana_spec_to_colors(m.group(1))
                        # Only a genuine fetchland ability fixes colors: it puts
                        # the land onto its controller's battlefield. Skip a
                        # ChangeType that fetches for someone else (a
                        # DefinedPlayer, e.g. Ghost Quarter destroying a land and
                        # letting its controller search a basic) or that doesn't
                        # reach the battlefield.
                        if ("Destination$ Battlefield" in line
                                and "DefinedPlayer$" not in line):
                            m = re.search(r"ChangeType\$\s*([^|]+)", line)
                            if m:
                                found |= _fetch_spec_to_colors(m.group(1))
            except OSError:
                pass
    letters = [c for c in "WUBRG" if c in found]
    _LAND_COLOR_CACHE[card_idx] = letters
    return letters


def card_border_colors(card_idx):
    """Border color(s) for a card by its color identity.

    A mana-producing land → the bright colors it can make/fetch (see
    _land_color_letters), one entry per color so the caller can split them
    across the border edges; a colorless/utility land → [grey]; a colorless
    nonland → [brown]; otherwise one entry per colored pip (W,U,B,R,G) present
    in the mana cost.
    """
    if card_idx in _LAND_VOCAB_IDS:
        letters = _land_color_letters(card_idx)
        if letters:
            return [_LAND_COLOR_BORDER[c] for c in letters]
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
    """Decode a player block (env._PLAYER_BLOCK_SIZE floats) into a dict."""
    return {
        "life": int(round(state[offset + _PB_LIFE] * 20)),
        "hand_count": int(round(state[offset + _PB_HAND_CT] * 10)),
        "poison": int(round(state[offset + _PB_POISON] * 10)),
        "mana": {c: int(round(state[offset + _PB_MANA + i] * 10))
                 for i, c in enumerate(_MANA_COLORS)},
        "energy": int(round(state[offset + _PB_ENERGY] * 10)),
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
    """Decode the current step from the step one-hot."""
    step_vec = state[_STEP_ONEHOT_START:_STEP_ONEHOT_START + _STEP_ONEHOT_SIZE]
    idx = int(np.argmax(step_vec))
    if step_vec[idx] < 0.5:
        return "Unknown"
    return _STEP_NAMES[idx] if idx < len(_STEP_NAMES) else f"Step({idx})"


def _decode_permanents(state, start, count=_PERM_SLOTS, counters=None, token_names=None,
                       slot_base=0):
    """Decode permanent slots into a list of dicts (non-empty only).

    `counters` is the per-slot typed-counter summary list the engine emits
    under --narrative (env._perm_counters side-channel; slot-aligned with the
    state vector). Loyalty entries are dropped — loyalty has a dedicated
    serialized field and its own display.

    `token_names` is the per-slot token-name list (env._perm_token_names,
    also narrative-only, slot-aligned). Every token shares the generic
    TOKEN_SENTINEL card id in the state vector, so when a slot names a token
    its "name" comes from here instead of the generic "Token".

    `slot_base` is this side's offset into the unified entity-reference slot
    space (0 = viewer's own perms, _PERM_SLOTS = opponent's); each dict records
    `slot` = slot_base + slot index, directly comparable to an action's
    slot_ref / a stack target's slot, so UIs can match one specific permanent
    among same-named copies."""
    perms = []
    for i in range(count):
        base = start + i * PERM_SLOT_SIZE
        idx = onehot_to_index(state, base + _OFF_CARD_ID)
        if idx < 0:
            continue
        name = card_index_to_name(idx)
        if token_names is not None and i < len(token_names) and token_names[i]:
            name = token_names[i]
        p = {"name": name, "card_idx": idx, "slot": slot_base + i}
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
        # Card name chosen on this permanent (Pithing Needle / Disruptor Flute
        # named card, Petrified Hamlet named land); sentinel = none.
        naming = onehot_to_card(state, base + _OFF_CHOSEN_NAME)
        if naming is not None:
            p["naming"] = naming
        # Card this permanent has exiled that still has a return path (Static Prison
        # holding a real card, Flickerwisp/Phelia EOT blink); sentinel = none.
        holding = onehot_to_card(state, base + _OFF_RETURNABLE)
        if holding is not None:
            p["holding"] = holding
        # Enriched fields (only non-default values are recorded, matching the
        # status-flag style above so fmt_perm stays compact).
        p1p1 = int(round(state[base + _OFF_P1P1_NET] * 10))
        if p1p1 != 0:
            p["p1p1"] = p1p1                      # net +1/+1 counters, signed
        other_ctrs = int(round(state[base + _OFF_OTHER_COUNTERS] * 10))
        if other_ctrs != 0:
            p["other_counters"] = other_ctrs
        ref = decode_slot_ref(state[base + _OFF_ATTACHED_TO])
        if ref >= 0:                              # equipment/aura: its host
            p["attached_to"] = _ref_desc(state, ref)
            p["attached_to_slot"] = ref
        ref = decode_slot_ref(state[base + _OFF_ATTACHED_BY])
        if ref >= 0:                              # host: its equipment/aura
            p["attached_by"] = _ref_desc(state, ref)
            p["attached_by_slot"] = ref
        ref = decode_slot_ref(state[base + _OFF_ATTACK_TGT])
        if ref >= 0:                              # attacked planeswalker (else: the player)
            p["attack_target"] = _ref_desc(state, ref)
            p["attack_target_slot"] = ref
        ref = decode_slot_ref(state[base + _OFF_BLOCKING_TGT])
        if ref >= 0:                              # the attacker this blocker blocks
            p["blocking_target"] = _ref_desc(state, ref)
            p["blocking_target_slot"] = ref
        if state[base + _OFF_IS_BLOCKED] > 0.5:
            p["blocked"] = True                   # attacker was blocked (CR 509.1h)
        if state[base + _OFF_IS_PHASED_OUT] > 0.5:
            p["phased_out"] = True
        kws = [_OBS_KEYWORDS[k] for k in range(len(_OBS_KEYWORDS))
               if state[base + _OFF_KEYWORDS_START + k] > 0.5]
        if kws:
            p["keywords"] = kws
        perms.append(p)
    return perms


def _decode_hand(state):
    """Decode self hand (MAX_HAND_SLOTS slots, 1 normalized card-id float each)
    into a list of dicts."""
    cards = []
    for i in range(MAX_HAND_SLOTS):
        idx = onehot_to_index(state, _HAND_START + i * _HAND_SLOT_SIZE)
        if idx >= 0:
            cards.append({"name": card_index_to_name(idx), "card_idx": idx})
    return cards


def _decode_graveyard(state, start):
    """Decode a graveyard zone (MAX_GY_SLOTS slots, 1 normalized card-id float
    each) into card names."""
    cards = []
    for i in range(MAX_GY_SLOTS):
        card = onehot_to_card(state, start + i * GY_SLOT_SIZE)
        if card is not None:
            cards.append(card)
    return cards


def _decode_known_top_library(state):
    """Decode the viewer's known top-of-library block (belief state).

    Returns a list of one entry per known-window slot (index 0 = top), each a
    card name or "?" for an unknown (sentinel) slot, but ONLY when at least one
    slot is known; otherwise returns [] so callers can hide the block. Set by
    Ponder/Brainstorm/Rearrange/Sylvan, cleared to unknown on shuffle."""
    names = []
    any_known = False
    for i in range(_KNOWN_TOP_LIB_SLOTS):
        idx = onehot_to_index(state, _KNOWN_TOP_LIB_START + i * _KNOWN_TOP_LIB_SLOT_SIZE)
        if idx >= 0:
            any_known = True
            names.append(card_index_to_name(idx))
        else:
            names.append("?")
    return names if any_known else []


def _decode_opp_known_hand(state):
    """Decode the viewer's known opponent-hand cards block (belief state).

    Returns the card names of the non-sentinel slots only (identities of
    opponent-hand cards the viewer has had revealed, e.g. by Thoughtseize);
    empty list when nothing is known."""
    names = []
    for i in range(_OPP_KNOWN_HAND_SLOTS):
        idx = onehot_to_index(state, _OPP_KNOWN_HAND_START + i * _OPP_KNOWN_HAND_SLOT_SIZE)
        if idx >= 0:
            names.append(card_index_to_name(idx))
    return names


def _decode_opp_revealed(state):
    """Decode the opponent revealed-cards multi-hot (belief state).

    Returns the card names of the set bits only — every distinct card the
    opponent-of-viewer has ever revealed this match (entered a public zone or
    was revealed by a tutor); empty list when none seen yet."""
    names = []
    for idx in range(_REVEALED_SIZE):
        if state[_REVEALED_START + idx] > 0.5:
            names.append(card_index_to_name(idx))
    return names


def _decode_stack(state, labels=SELF_OPP_LABELS):
    """Decode the stack (12 slots x 37: ctrl + card id + is_spell + x_or_amount +
    cast-qualifier flags(7) + chosen-mode multi-hot(6) + 4 announced-target
    sub-slots of [present, is_player, ctrl, slot_ref, card id])."""
    entries = []
    for i in range(_STACK_SLOTS):
        base = _STACK_START + i * STACK_SLOT_SIZE
        idx = onehot_to_index(state, base + 1)  # skip controller_is_self float
        if idx < 0:
            continue
        is_spell = state[base + 2] > 0.5
        x_amt = int(round(state[base + _STACK_XAMT_OFF] * 10))
        quals = [_STACK_QUAL_NAMES[q] for q in range(_STACK_QUALS)
                 if state[base + _STACK_QUAL_START + q] > 0.5]
        modes = [m for m in range(_STACK_MODE_SLOTS)
                 if state[base + _STACK_MODE_START + m] > 0.5]
        targets = []
        target_refs = []
        for t in range(_STACK_TGT_SLOTS):
            tbase = base + _STACK_TGT_START + t * _STACK_TGT_FIELDS
            if state[tbase] < 0.5:  # present flag
                continue
            is_self = state[tbase + 2] > 0.5
            is_player = state[tbase + 1] > 0.5
            slot = decode_slot_ref(state[tbase + 3])   # -1 for players / non-serialized
            ctrl = labels["self"] if is_self else labels["opponent"]
            tidx = -1 if is_player else onehot_to_index(state, tbase + 4)
            if is_player:
                targets.append(f"{ctrl} (player)")
            else:
                tname = card_index_to_name(tidx) if tidx >= 0 else "?"
                slot_str = f", slot {slot}" if slot >= 0 else ""
                targets.append(f"{tname} ({ctrl}{slot_str})")
            # Structured form for UIs that highlight the target on the board.
            # `is_self` is viewer-relative (like the rest of the state vector); a
            # mirrored decode must flip it to the human frame at the call site.
            # `slot` is the target's entity-slot ref (-1 = none/player), which
            # disambiguates two same-named permanents.
            target_refs.append({"is_player": is_player, "is_self": is_self,
                                "card_idx": tidx, "slot": slot})
        entries.append({
            "name": card_index_to_name(idx),
            "card_idx": idx,
            "controller": labels["self"] if state[base] > 0.5 else labels["opponent"],
            "is_spell": is_spell,
            "x_or_amount": x_amt,  # spell: X paid at cast; ability: its amount
            "qualifiers": quals,   # set cast-qualifier names (empty = plain cast)
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
    is_active = state[_IS_ACTIVE_IDX] > 0.5
    is_player_a = state[_SELF_IS_A_IDX] > 0.5
    return {
        "priority_player": "Player A" if is_player_a else "Player B",
        "priority_is_a": bool(is_player_a),
        "is_active_player": bool(is_active),
        "active_is_a": (is_active == is_player_a),
        "step": decode_step(state),
        "turn": decode_turn(state),
        "stack_size": int(round(state[_STACK_SIZE_IDX] * 10)),
        "self": _decode_player(state, _SELF_BLOCK_START),
        "opponent": _decode_player(state, _OPP_BLOCK_START),
        "self_library": int(round(state[_IDX_SELF_LIB] * 60)),
        "opp_library": int(round(state[_IDX_OPP_LIB] * 60)),
        "self_battlefield": _decode_permanents(
            state, _SELF_PERM_START,
            counters=perm_counters[0] if perm_counters else None,
            token_names=perm_token_names[0] if perm_token_names else None),
        "opp_battlefield": _decode_permanents(
            state, _OPP_PERM_START,
            counters=perm_counters[1] if perm_counters else None,
            token_names=perm_token_names[1] if perm_token_names else None,
            slot_base=_PERM_SLOTS),
        # NOTE: the action-history ring ([4394-4905] in src/machine_io.h — 128
        # entries x 4 floats) is DELIBERATELY not decoded here. It is bulky, only
        # useful for the policy network's temporal context, and never surfaced in
        # the human-readable board dump; it is the one intentional coverage skip.
        "stack": _decode_stack(state, labels),
        "self_hand": _decode_hand(state),
        "self_graveyard": _decode_graveyard(state, _GY_START),
        "opp_graveyard": _decode_graveyard(state, _OPP_GY_START),
        "self_exile": _decode_graveyard(state, _EXILE_START),
        "opp_exile": _decode_graveyard(state, _OPP_EXILE_START),
        "known_top_library": _decode_known_top_library(state),
        "opp_known_hand": _decode_opp_known_hand(state),
        "opp_revealed": _decode_opp_revealed(state),
        "pending_decision": _decode_pending_decision(state),
        "match": _decode_match_context(state),
        "extras": _decode_extras(state),
    }


def _decode_extras(state):
    """Decode the global-extras block into a dict of NON-DEFAULT values only.

    Keys are present only when the value differs from its default (0 / False /
    priority held), so an empty dict means "nothing notable" and callers can
    render it compactly. Boolean keys hold True when set; `has_priority` is the
    one inverted case — it appears (as False) only when the viewer does NOT
    hold priority. `mandatory_choice` is the pending MandatoryChoice display
    name (_MC_NAMES; absent when NONE)."""
    ex = {}
    lands = int(round(float(state[_EXTRAS_LANDS_SELF]) * 10))
    if lands:
        ex["self_lands_played"] = lands
    lands = int(round(float(state[_EXTRAS_LANDS_OPP]) * 10))
    if lands:
        ex["opp_lands_played"] = lands
    if state[_EXTRAS_HAS_PRIORITY] < 0.5:
        ex["has_priority"] = False
    if state[_EXTRAS_MONARCH_SELF] > 0.5:
        ex["self_monarch"] = True
    if state[_EXTRAS_MONARCH_OPP] > 0.5:
        ex["opp_monarch"] = True
    if state[_EXTRAS_BLESSING_SELF] > 0.5:
        ex["self_citys_blessing"] = True
    if state[_EXTRAS_BLESSING_OPP] > 0.5:
        ex["opp_citys_blessing"] = True
    if state[_EXTRAS_REVOLT_SELF] > 0.5:
        ex["self_revolt"] = True
    if state[_EXTRAS_REVOLT_OPP] > 0.5:
        ex["opp_revolt"] = True
    turns = int(round(float(state[_EXTRAS_EXTRA_TURNS_SELF]) * 3))
    if turns:
        ex["self_extra_turns"] = turns
    turns = int(round(float(state[_EXTRAS_EXTRA_TURNS_OPP]) * 3))
    if turns:
        ex["opp_extra_turns"] = turns
    if state[_EXTRAS_IS_DAY] > 0.5:
        ex["day"] = True
    if state[_EXTRAS_IS_NIGHT] > 0.5:
        ex["night"] = True
    mc_vec = state[_EXTRAS_MC_ONEHOT_START:_EXTRAS_MC_ONEHOT_START + len(_MC_NAMES)]
    mc = int(np.argmax(mc_vec))
    if mc > 0 and mc_vec[mc] > 0.5:               # NONE (index 0) is the default
        ex["mandatory_choice"] = _MC_NAMES[mc]
    return ex


def _decode_match_context(state):
    """Decode the bo3 match-context block (self-perspective).

    Returns {"game_number", "self_wins", "opp_wins", "is_sideboard",
    "is_post_board"}. In a single-game (bo1) match every field is 0 / False, so
    callers can treat a game_number of 0 as "not a bo3 match". is_post_board is
    True in game 2+ of a bo3 (a game played after sideboarding). self_wins/opp_wins
    are viewer-relative (like the rest of the state vector) — a mirrored decode
    must swap them."""
    return {
        "game_number": int(round(float(state[_IDX_GAME_NUMBER]) * 3)),
        "self_wins": int(round(float(state[_IDX_SELF_WINS]) * 2)),
        "opp_wins": int(round(float(state[_IDX_OPP_WINS]) * 2)),
        "is_sideboard": float(state[_IDX_SIDEBOARD]) > 0.5,
        "is_post_board": float(state[_IDX_POST_BOARD]) > 0.5,
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

# Each decoder slices its block at the shared ACT_*_START offset env.py wrote it
# to, rather than counting blocks in as a bare `STATE_SIZE + k * MAX_ACTIONS`.
# Under the old form, reordering the per-action blocks left every decoder here
# silently reading the wrong one.

def action_categories(obs, num_choices=MAX_ACTIONS):
    """Integer ActionCategory per legal action (from the obs metadata block)."""
    raw = obs[ACT_CATS_START: ACT_CATS_START + num_choices]
    return np.round(raw * ACTION_CATEGORY_MAX).astype(int)


def action_card_ids(obs):
    """Normalised card-ID float per action slot."""
    return obs[ACT_IDS_START: ACT_IDS_START + MAX_ACTIONS]


def action_ctrls(obs):
    """controller_is_self float per action slot."""
    return obs[ACT_CTRL_START: ACT_CTRL_START + MAX_ACTIONS]


def action_zone_refs(obs, num_choices=MAX_ACTIONS):
    """Integer ActionRefZone per legal action (0 = REF_NONE, no referenced entity).

    Display names for the values live in _REF_NAMES.
    """
    raw = obs[ACT_ZONE_START: ACT_ZONE_START + num_choices]
    return np.round(raw * REF_ZONE_MAX).astype(int)


def action_slot_refs(obs, num_choices=MAX_ACTIONS):
    """Integer entity-slot ref per legal action (-1 = no referenced entity).

    Fifth action-metadata block (cats|ids|ctrl|zone|refs): the slot in the
    unified entity-reference space (0-47 self perms, 48-95 opp perms, 96-107
    stack) of the choice's source entity, normalized like the in-state ref
    fields ((idx + 1) / N_ENTITY_REF_SLOTS, 0.0 = none).
    """
    raw = obs[ACT_REFS_START: ACT_REFS_START + num_choices]
    return (np.round(raw * N_ENTITY_REF_SLOTS) - 1).astype(int)


def action_ordinals(obs, num_choices=MAX_ACTIONS):
    """Integer option_ordinal per legal action (-1 = not applicable).

    Sixth action-metadata block (cats|ids|ctrl|zone|refs|ords): the per-action
    ordinal/value scalar (mode index, X value, color index, cast variant,
    top-of-library depth, ...), normalized (ord + 1) / (OPTION_ORDINAL_MAX + 1).
    """
    raw = obs[ACT_ORDS_START: ACT_ORDS_START + num_choices]
    return (np.round(raw * (OPTION_ORDINAL_MAX + 1)) - 1).astype(int)


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


def describe_action(cat, card_name, ctrl_str, labels=SELF_OPP_LABELS,
                    slot_ref=-1):
    """Build a concise human-readable action description.

    `ctrl_str` is the already-resolved controller word (see `_ctrl_str`) and is
    used verbatim; `labels` is accepted for signature symmetry with the other
    formatters so a caller can pass one label map everywhere — it does not
    re-map an already-resolved `ctrl_str`.

    `slot_ref` is the action's entity-slot ref (see `action_slot_refs`); when
    it points at a battlefield/stack slot AND the action is one whose entity a
    caller may need to disambiguate (target / attacker / blocker / activate /
    pay-cost), an "@slotN" suffix is appended — two same-named permanents are
    otherwise indistinguishable in the menu.
    """
    suffix = (f" @slot{slot_ref}"
              if slot_ref >= 0 and cat in _SLOT_REF_CATS else "")
    if suffix:
        return describe_action(cat, card_name, ctrl_str, labels) + suffix
    owner = f" ({ctrl_str})" if ctrl_str else ""
    name = card_name or ""
    if cat == CAT_PASS_PRIORITY:
        return "Pass priority"
    elif cat == CAT_CAST_SPELL:
        return f"Cast {name}"
    elif cat == CAT_PLAY_LAND:
        return f"Play land: {name}"
    elif cat == CAT_ACTIVATE_ABILITY:
        return f"Activate {name}{owner}"
    elif cat == CAT_SELECT_TARGET:
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
    elif cat == CAT_SELECT_ATTACKER:
        return f"Select attacker: {name}"
    elif cat == CAT_CONFIRM_ATTACKERS:
        return "Confirm attackers"
    elif cat == CAT_SELECT_BLOCKER:
        return f"Select blocker: {name}"
    elif cat == CAT_CONFIRM_BLOCKERS:
        return "Confirm blockers"
    elif cat == CAT_MULLIGAN:
        return f"Mulligan ({name})" if name else "Mulligan"
    elif cat == CAT_KEEP_HAND:
        return "Keep hand"
    elif cat == CAT_SHUFFLE:
        return "Shuffle"
    elif cat == CAT_DONT_SHUFFLE:
        return "Don't shuffle"
    elif cat == CAT_EXILE_FROM_YARD:
        return f"Exile from graveyard: {name}" if name else "Exile from graveyard"
    elif cat == CAT_BOTTOM_DECK_CARD:
        return f"Bottom: {name}"
    elif CAT_MANA_W <= cat <= CAT_MANA_C:
        return f"Tap {name} for {{{_MANA_COLORS[cat - CAT_MANA_W]}}}"
    elif cat == CAT_SEARCH_LIBRARY:
        return f"Search: {name}" if name else "Fail to find"
    elif cat == CAT_TOP_LIBRARY:
        return f"Put on top: {name}"
    elif cat == CAT_PAYING_COSTS:
        return f"Pay cost: {name}{owner}"
    elif cat == CAT_DIG_CHOICE:
        return f"Dig choice: {name}"
    elif cat == CAT_OTHER_CHOICE:
        return f"Choice: {name}{owner}" if name else "Choice"
    else:
        cat_name = _CAT_NAMES.get(cat, f"?({cat})")
        return f"{cat_name}: {name}{owner}" if name else cat_name


def decode_actions(cats_int, card_ids, ctrl, num_choices, public_flags=None,
                   labels=SELF_OPP_LABELS, descriptions=None, zone_refs=None,
                   slot_refs=None, ordinals=None):
    """Decode the per-action arrays into a list of dicts.

    Each dict: index, category (int), category_name, card (name or None),
    card_idx (vocab index or -1), controller ('own'|'opp'|None), description,
    card_is_public (bool — card identity publicly known, e.g. a revealed tutor),
    zone_ref (int ActionRefZone of the referenced entity, 0 = none),
    slot_ref (int entity-slot ref of the referenced entity, -1 = none),
    option_ordinal (int per-action ordinal/value scalar, -1 = not applicable).

    `zone_refs` is the per-action ActionRefZone array (decode.action_zone_refs /
    env side-channel); it disambiguates a card that appears in two zones at once
    (e.g. a hand card being discarded vs. a same-named battlefield permanent),
    which card_idx + controller alone cannot tell apart.

    `slot_refs` is the per-action entity-slot ref array (decode.action_slot_refs):
    which battlefield/stack SLOT the choice's source entity occupies, so two
    same-named permanents are distinguishable. When given, entity-referencing
    reconstructed descriptions gain an "@slotN" suffix (see `describe_action`).

    `public_flags` is the per-action card_is_public array (env._action_public);
    None means "unknown", treated as not-public. `labels` substitutes the
    controller wording (default keeps today's "own"/"opp").

    `descriptions` is the per-action text the engine emits under --narrative
    (env._action_descriptions). When present and non-empty it is the exact
    CLI label and is used verbatim — so player targets, Sylvan Library
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
        slot_ref = (int(slot_refs[i]) if slot_refs is not None
                    and i < len(slot_refs) else -1)
        ordinal = (int(ordinals[i]) if ordinals is not None
                   and i < len(ordinals) else -1)
        engine_desc = (descriptions[i] if descriptions is not None
                       and i < len(descriptions) and descriptions[i].strip() else None)
        if engine_desc is not None:
            desc = engine_desc
        elif cat in (CAT_MULLIGAN, CAT_KEEP_HAND):
            # Mulligan query: KEEP_HAND (index 0) = keep, MULLIGAN = mulligan.
            desc = "Keep hand" if cat == CAT_KEEP_HAND else "Mulligan"
        else:
            desc = describe_action(cat, card_name, ctrl_str, labels,
                                   slot_ref=slot_ref)
        actions.append({
            "index": i,
            "category": cat,
            "category_name": _CAT_NAMES.get(cat, f"?({cat})"),
            "card": card_name,
            "card_idx": card_idx if card_idx >= 0 else -1,
            "controller": ctrl_str,
            "description": desc,
            "card_is_public": is_public,
            "zone_ref": int(zone_refs[i]) if zone_refs is not None and i < len(zone_refs) else 0,
            "slot_ref": slot_ref,
            "option_ordinal": ordinal,
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
                          public_flags, labels, descriptions,
                          zone_refs=action_zone_refs(obs, num_choices),
                          slot_refs=action_slot_refs(obs, num_choices),
                          ordinals=action_ordinals(obs, num_choices))


# ── Decision-type classification (all read the integer category array) ────────

def is_mulligan(cats):
    # A mulligan menu is the KEEP_HAND / MULLIGAN pair (keep = index 0).
    return len(cats) > 0 and all(c in (CAT_MULLIGAN, CAT_KEEP_HAND) for c in cats)


def is_bottom(cats):
    return len(cats) > 0 and all(c == 12 for c in cats)


def is_search(cats):
    return len(cats) > 0 and all(c == 19 for c in cats)


def menu_is_interchangeable(obs, num_choices):
    """True when every legal action in the menu is the same choice, so any of
    them (take the first) is as good as a searched/evaluated pick.

    The common shapes: four "Play Mountain" from a hand of duplicates, N
    identical untapped basics offered as mana taps, duplicate copies in a
    search/dig/bottom prompt, or targeting one of several serialized-identical
    permanents. A single-choice menu is trivially interchangeable.

    Actions collapse only when everything the obs contract can distinguish is
    equal: category, card id, controller, zone_ref and option ordinal — and
    the card id must be a REAL vocab id (text-only prompts, whose options an
    ordinal or description distinguishes, never collapse; CAT_OTHER_CHOICE is
    excluded outright for the same reason). Entity-referencing actions must
    point at pairwise-distinct slots whose serialized feature blocks are
    float-identical AND at which no other in-state ref field points — an
    announced stack target or an attachment on one of two otherwise identical
    permanents makes the choice real."""
    n = int(num_choices)
    if n <= 1:
        return True
    cats = action_categories(obs, n)
    if not np.all(cats == cats[0]) or int(cats[0]) == CAT_OTHER_CHOICE:
        return False
    ids = action_card_ids(obs)[:n]
    if int(round(float(ids[0]) * N_CARD_TYPES)) < 0 or not np.all(ids == ids[0]):
        return False
    ctrl = action_ctrls(obs)[:n]
    if not np.all(ctrl == ctrl[0]):
        return False
    zones = action_zone_refs(obs, n)
    if not np.all(zones == zones[0]):
        return False
    ords = action_ordinals(obs, n)
    if not np.all(ords == ords[0]):
        return False
    refs = action_slot_refs(obs, n)
    if np.all(refs == -1):
        return True
    if np.any(refs < 0) or len(set(refs.tolist())) != n:
        return False                      # mixed or same-entity actions: real choices
    state = obs[:STATE_SIZE]
    blocks = [entity_ref_features(state, int(r)) for r in refs]
    if any(b is None or not np.array_equal(b, blocks[0]) for b in blocks):
        return False
    return not (state_ref_targets(state) & {int(r) for r in refs})


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_mana(mana):
    """Format a mana-pool dict as a compact string."""
    parts = [f"{mana[c]}{c}" for c in _MANA_COLORS if mana.get(c, 0) > 0]
    return ", ".join(parts) if parts else "empty"


def fmt_resources(player):
    """Format a player's non-mana resource counters (poison, energy) as a
    " | poison N | energy N" suffix, listing only counters that are nonzero.
    Returns an empty string when the player has neither."""
    suffix = ""
    if player.get("poison", 0) > 0:
        suffix += f" | poison {player['poison']}"
    if player.get("energy", 0) > 0:
        suffix += f" | energy {player['energy']}"
    return suffix


def fmt_stack_entry(e):
    """Format a decoded stack-entry dict (from _decode_stack)."""
    kind = "spell" if e["is_spell"] else "ability"
    s = f"{e['name']} ({kind}, {e['controller']})"
    if e.get("x_or_amount"):
        s += (f" [X={e['x_or_amount']}]" if e["is_spell"]
              else f" [amt={e['x_or_amount']}]")
    if e.get("qualifiers"):
        s += f" [{','.join(e['qualifiers'])}]"
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
        f"Step: {gs['step']} (turn {gs['turn']})",
    ]
    hand_names = [c["name"] for c in gs["self_hand"]]
    lines.append(f"Self:  {gs['self']['life']} life"
                 f" | mana: {fmt_mana(gs['self']['mana'])}"
                 f"{fmt_resources(gs['self'])}"
                 f" | hand({gs['self']['hand_count']}): {', '.join(hand_names) or '(empty)'}"
                 f" | lib {gs['self_library']}")
    lines.append(f"Opp:   {gs['opponent']['life']} life"
                 f" | mana: {fmt_mana(gs['opponent']['mana'])}"
                 f"{fmt_resources(gs['opponent'])}"
                 f" | hand count: {gs['opponent']['hand_count']}"
                 f" | lib {gs['opp_library']}")
    if gs["self_battlefield"]:
        lines.append(f"Self BF:  {' | '.join(fmt_perm(p) for p in gs['self_battlefield'])}")
    if gs["opp_battlefield"]:
        lines.append(f"Opp BF:   {' | '.join(fmt_perm(p) for p in gs['opp_battlefield'])}")
    if gs["stack"]:
        lines.append(f"Stack: {' -> '.join(fmt_stack_entry(e) for e in gs['stack'])}")
    # Source of the current mid-resolution choice (may not be on the stack yet,
    # since targets are announced before the spell moves there).
    pend = gs.get("pending_decision")
    if pend:
        lines.append(f"Pending: {pend['name']}"
                     f" ({'self' if pend['is_self'] else 'opp'})")
    if gs["self_graveyard"]:
        lines.append(f"Self GY: {', '.join(gs['self_graveyard'])}")
    if gs["opp_graveyard"]:
        lines.append(f"Opp GY:  {', '.join(gs['opp_graveyard'])}")
    if gs.get("self_exile"):
        lines.append(f"Self exile: {', '.join(gs['self_exile'])}")
    if gs.get("opp_exile"):
        lines.append(f"Opp exile:  {', '.join(gs['opp_exile'])}")
    # Belief-state blocks (shown only when the viewer actually knows something).
    if gs.get("known_top_library"):
        lines.append(f"Known top: {', '.join(gs['known_top_library'])}")
    if gs.get("opp_known_hand"):
        lines.append(f"Known opp hand: {', '.join(gs['opp_known_hand'])}")
    if gs.get("opp_revealed"):
        lines.append(f"Opp revealed: {', '.join(gs['opp_revealed'])}")
    # Bo3 match context (game_number == 0 in bo1, so this line is omitted there).
    match = gs.get("match") or {}
    if match.get("game_number"):
        mstr = (f"Match: game {match['game_number']}"
                f" | wins {match['self_wins']}-{match['opp_wins']}")
        if match.get("is_post_board"):
            mstr += " | post-board"
        if match.get("is_sideboard"):
            mstr += " | sideboarding"
        lines.append(mstr)
    extras = gs.get("extras") or {}
    if extras:
        # _decode_extras records only non-default values: a boolean True renders
        # as its bare key, everything else as key=value.
        parts = [k if v is True else f"{k}={v}" for k, v in extras.items()]
        lines.append(f"Extras: {', '.join(parts)}")
    return lines


def format_action_lines(actions):
    """Enumerated legal-action lines (the 'Actions:' menu, shared transcript)."""
    def _line(a):
        ordv = a.get("option_ordinal", -1)
        suffix = f"  [#{ordv}]" if ordv >= 0 else ""
        return f"  {a['index']:>2}: {a['description']}{suffix}"
    return [_line(a) for a in actions]


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
    if "naming" in p:
        s += f" [naming: {p['naming']}]"
    if "holding" in p:
        s += f" [holding: {p['holding']}]"
    if "counters" in p:
        s += f" [{p['counters']}]"
    elif "p1p1" in p or "other_counters" in p:
        # Serialized counter summary (no narrative side-channel available).
        parts = []
        if "p1p1" in p:
            parts.append(f"p1p1 {p['p1p1']:+d}")
        if "other_counters" in p:
            parts.append(f"ctrs {p['other_counters']}")
        s += f" [{', '.join(parts)}]"
    if "keywords" in p:
        s += f" [{', '.join(p['keywords'])}]"
    if "attached_to" in p:
        s += f" (->{p['attached_to']})"
    if "attached_by" in p:
        s += f" (eq:{p['attached_by']})"
    flags = []
    if p.get("tapped"):
        flags.append("T")
    if p.get("attacking"):
        # Attacking a planeswalker names it; a plain ATK attacks the player.
        flags.append("ATK" + (f"->{p['attack_target']}"
                              if "attack_target" in p else ""))
    if p.get("blocking"):
        flags.append("BLK" + (f"->{p['blocking_target']}"
                              if "blocking_target" in p else ""))
    if p.get("blocked"):
        flags.append("BLOCKED")
    if p.get("summoning_sick"):
        flags.append("SICK")
    if p.get("phased_out"):
        flags.append("PHASED")
    if flags:
        s += f" ({','.join(flags)})"
    return s
