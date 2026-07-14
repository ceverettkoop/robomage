"""
RoboMage gymnasium environment.

The game runs as a subprocess with --machine mode. On each decision point it
emits a BQUERY line to stdout:

    BQUERY: <num_choices> <STATE_SIZE> <MAX_ACTIONS>\n<float32[STATE_SIZE] binary><int32[MAX_ACTIONS] cats><float32[MAX_ACTIONS] ids><float32[MAX_ACTIONS] ctrl><float32[MAX_ACTIONS] pub><int32[MAX_ACTIONS] zone><int32[MAX_ACTIONS] refs>

The environment sends back a single integer on stdin.

Action convention
-----------------
For top-level priority actions (main.cpp):
    action index 0..N-1 maps directly to legal_actions[i].

For mandatory-choice loops (declare attackers / blockers):
    The game uses -1 as "confirm/done". In the gym action space we reserve
    action index = num_choices - 1 as the confirm slot, which the environment
    remaps to -1 before sending to the game.

    e.g. num_choices=3 means actions 0,1 are creatures and action 2 = confirm.

Observation space
-----------------
State is always emitted from the PRIORITY PLAYER'S perspective ("self").

STATE_SIZE-float state vector. Card identity is a single normalized id float per
slot (idx/N_CARD_TYPES, -1/N_CARD_TYPES = empty), NOT a one-hot — the policy
network maps ids through a learned nn.Embedding. The opponent revealed-cards
block is the only vocab-width block (N_CARD_TYPES multi-hot).
State (STATE_SIZE) + 64 action-category floats + 64 action card-ID floats
+ 64 action controller_is_self floats + 64 action zone_ref floats
+ 64 action entity-slot-ref floats
+ 70 hand cost floats + 336 battlefield ability cost floats = OBS_SIZE total.
NOTE: ActionChoice.description is NOT part of the observation — it is for
human-readable display only (CLI) and is never sent to the ML model.
NOTE: Both exile zones (self + opp, 64 recency-ordered card-id slots each) are
serialized right after the graveyard blocks.
See src/machine_io.h for the full state layout.

Reward
------
+1.0 for winning, -1.0 for losing (from Player A's perspective).
"""

import struct
import subprocess
import sys
import os
import re
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

try:
    from card_costs import _CARD_COST_MATRIX, _CARD_ABILITY_COST_MATRIX, N_CARD_TYPES, _N_COST_FEATS
except ImportError:
    from train.card_costs import _CARD_COST_MATRIX, _CARD_ABILITY_COST_MATRIX, N_CARD_TYPES, _N_COST_FEATS

# ACTION_CATEGORY_MAX is generated from the C++ ActionCategory enum (single source
# of truth) by train/gen_enums.py — import it so this module never drifts from the
# engine's category normalization (used for both the action block and history).
# Every value below is GENERATED from the C++ headers by train/gen_enums.py — the
# single source of truth. Import them (never re-type the literals) so this module
# cannot drift from the engine: the layout/size constants (STATE_SIZE, MAX_ACTIONS,
# the per-zone slot widths, ...), the name-keyed CAT_* ActionCategory ints, and the
# derived-block widths this module needs. STATE_SIZE now being generated turns the
# `_EXTRAS_END == STATE_SIZE` assert below into a real C++↔Python cross-check.
try:
    from _enums import (
        ACTION_CATEGORY_MAX, REF_ZONE_MAX, N_OBS_KEYWORDS, N_MANDATORY_CHOICES,
        STATE_SIZE, MAX_ACTIONS, MAX_CHOICE_DESC, PERM_COUNTERS_LEN,
        PERM_TOKEN_NAME_LEN, MAX_BATTLEFIELD_SLOTS, MAX_STACK_DISPLAY,
        MAX_STACK_MODES, MAX_STACK_TGTS, MAX_GY_SLOTS, MAX_HAND_SLOTS,
        KNOWN_TOP_LIBRARY_SIZE, PERM_SLOT_SIZE, ACTION_HISTORY_SIZE,
        DECKLIST_MAIN_SLOTS, DECKLIST_SIDE_SLOTS,
        N_CARD_TYPES as _ENUM_N_CARD_TYPES,
        CAT_PASS_PRIORITY, CAT_MANA_ABILITY, CAT_MANA_W, CAT_MANA_C, CAT_MANA_U,
        CAT_SELECT_ATTACKER, CAT_CONFIRM_ATTACKERS, CAT_SELECT_BLOCKER,
        CAT_CONFIRM_BLOCKERS, CAT_ACTIVATE_ABILITY, CAT_CAST_SPELL,
        CAT_SELECT_TARGET, CAT_PLAY_LAND, CAT_MULLIGAN, CAT_SEARCH_LIBRARY,
        CAT_OTHER_CHOICE, CAT_DISCARD, CAT_PAYING_COSTS, CAT_CHOOSE_X,
        CAT_CHOOSE_CARD, CAT_DIG_CHOICE, CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT,
        CAT_SIDEBOARD_DONE, CAT_COMPANION, CAT_OPTIONAL_YESNO, CAT_TOP_LIBRARY,
        CAT_SHUFFLE)
except ImportError:
    from train._enums import (
        ACTION_CATEGORY_MAX, REF_ZONE_MAX, N_OBS_KEYWORDS, N_MANDATORY_CHOICES,
        STATE_SIZE, MAX_ACTIONS, MAX_CHOICE_DESC, PERM_COUNTERS_LEN,
        PERM_TOKEN_NAME_LEN, MAX_BATTLEFIELD_SLOTS, MAX_STACK_DISPLAY,
        MAX_STACK_MODES, MAX_STACK_TGTS, MAX_GY_SLOTS, MAX_HAND_SLOTS,
        KNOWN_TOP_LIBRARY_SIZE, PERM_SLOT_SIZE, ACTION_HISTORY_SIZE,
        DECKLIST_MAIN_SLOTS, DECKLIST_SIDE_SLOTS,
        N_CARD_TYPES as _ENUM_N_CARD_TYPES,
        CAT_PASS_PRIORITY, CAT_MANA_ABILITY, CAT_MANA_W, CAT_MANA_C, CAT_MANA_U,
        CAT_SELECT_ATTACKER, CAT_CONFIRM_ATTACKERS, CAT_SELECT_BLOCKER,
        CAT_CONFIRM_BLOCKERS, CAT_ACTIVATE_ABILITY, CAT_CAST_SPELL,
        CAT_SELECT_TARGET, CAT_PLAY_LAND, CAT_MULLIGAN, CAT_SEARCH_LIBRARY,
        CAT_OTHER_CHOICE, CAT_DISCARD, CAT_PAYING_COSTS, CAT_CHOOSE_X,
        CAT_CHOOSE_CARD, CAT_DIG_CHOICE, CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT,
        CAT_SIDEBOARD_DONE, CAT_COMPANION, CAT_OPTIONAL_YESNO, CAT_TOP_LIBRARY,
        CAT_SHUFFLE)

# STATE_SIZE / MAX_ACTIONS are imported from _enums (src/machine_io.h,
# src/classes/gamestate.h). Card identity is 1 id float/slot, not a one-hot.
# N_CARD_TYPES comes from card_costs (generated from card_vocab.h); assert it
# agrees with the engine's machine_io.h value so the two generators can't drift.
assert _ENUM_N_CARD_TYPES == N_CARD_TYPES, (_ENUM_N_CARD_TYPES, N_CARD_TYPES)
# NOTE: Both exile zones are serialized right after the graveyard blocks.
# NOTE: ActionChoice.description is never emitted in the BQUERY payload — it is for
#       human-readable display only and is not part of the ML observation.
# Binary BQUERY payload sizes (bytes): state float32s + MAX_ACTIONS each of
# cats(int32)/ids/ctrl(float32)/pub(float32)/zone(int32)/refs(int32)
_BQUERY_STATE_BYTES = STATE_SIZE * 4
_BQUERY_CATS_BYTES  = MAX_ACTIONS * 4  # int32
_BQUERY_IDS_BYTES   = MAX_ACTIONS * 4  # float32
_BQUERY_CTRL_BYTES  = MAX_ACTIONS * 4  # float32
_BQUERY_PUB_BYTES   = MAX_ACTIONS * 4  # float32 — card_is_public per action
_BQUERY_ZONE_BYTES  = MAX_ACTIONS * 4  # int32 — ActionRefZone per action
_BQUERY_REFS_BYTES  = MAX_ACTIONS * 4  # int32 — entity-slot ref of the choice's source (-1 = none)
# Per-action human-readable descriptions, emitted ONLY under --narrative
# (gated on the engine side too). Fixed [MAX_ACTIONS][MAX_CHOICE_DESC] NUL-padded
# char block. MAX_CHOICE_DESC imported from _enums (src/classes/gamestate.h).
_BQUERY_DESC_BYTES  = MAX_ACTIONS * MAX_CHOICE_DESC
# Per-permanent typed-counter summaries ("+1/+1:2, time:3"), also narrative-only.
# Fixed [2*N_PERM_SLOTS][PERM_COUNTERS_LEN] NUL-padded char block (48 self slots
# then 48 opp slots, aligned with the state vector's permanent blocks).
# PERM_COUNTERS_LEN / MAX_BATTLEFIELD_SLOTS imported from _enums (gamestate.h).
N_PERM_SLOTS        = MAX_BATTLEFIELD_SLOTS
_BQUERY_PERM_CTRS_BYTES = 2 * N_PERM_SLOTS * PERM_COUNTERS_LEN
# Per-permanent token names (narrative-only, slot-aligned like the counters block).
# Every token shares the generic TOKEN_SENTINEL card id in the state vector, so this
# is the only channel carrying a token's real name to observers.
# PERM_TOKEN_NAME_LEN imported from _enums (src/classes/gamestate.h).
_BQUERY_PERM_TOKS_BYTES = 2 * N_PERM_SLOTS * PERM_TOKEN_NAME_LEN
# ACTION_CATEGORY_MAX imported from _enums above (mirrors src/classes/action.h).


def _decode_char_block(raw, count, width):
    """Decode a fixed-size NUL-padded char block into `count` strings."""
    out = []
    for k in range(count):
        chunk = raw[k * width:(k + 1) * width]
        end = chunk.find(b"\x00")
        if end >= 0:
            chunk = chunk[:end]
        out.append(chunk.decode("utf-8", errors="replace"))
    return out

# ── Shaping reward magnitudes ─────────────────────────────────────────────────
SHAPING_MANA_WASTED      = -0.00  # per drain event with mana remaining in pool; commented out because we aren't letting it float anymore
SHAPING_MULLIGAN_PENALTY =  0.00  # per mulligan taken beyond the 2nd (C++: >= 3rd)
SHAPING_OPPONENT_BELOW10 =  0.00  # one-time bonus when opponent life first drops < 10
SHAPING_HAND_ADV_PER_CARD = 0.01  # potential weight per card of hand advantage (potential-based)
SHAPING_POWER_ADV_PER_PT  = 0.005 # potential weight per point of power advantage on board
SHAPING_EPISODE_CAP       = 0.3   # max absolute shaping bonus per episode
SHAPING_EPISODE_CAP_DOOMSDAY = 0.6  # higher cap for doomsday deck

# ── Doomsday deck shaping ────────────────────────────────────────────────────
SHAPING_DD_CAST_DOOMSDAY    = 0.2  # reward for casting Doomsday
SHAPING_DD_RITUAL_SETUP     = 0.1  # reward for casting Dark Ritual when Doomsday in hand + main phase
SHAPING_DD_PICK_ORACLE      = 0.2  # reward for picking Thassa's Oracle during Doomsday pile
SHAPING_DD_CAST_DISCARD     = 0.03  # reward for casting Thoughtseize/Duress targeting opponent
SHAPING_DD_STRIP_COUNTER    = 0.00 # reward for selecting Force of Will or Daze with discard spell
SHAPING_DD_TUTOR_DOOMSDAY   = 0.05 # reward for selecting Doomsday with Personal Tutor
SHAPING_DD_KEEP_DOOMSDAY    = 0.00  # reward for not shuffling after placed Doomsday on top (Ponder)
SHAPING_DD_LED_WITH_DRAW    = 0.03 # reward for cracking LED with a cycling/draw ability on the stack
SHAPING_DD_LED_EMPTY_STACK  = -0.02 # penalty for cracking LED with nothing on the stack

# ── Bo3 match rewards ────────────────────────────────────────────────────────
BO3_GAME_WIN_REWARD   =  0.3   # intermediate reward for winning a game in bo3
BO3_GAME_LOSS_REWARD  = -0.3   # intermediate penalty for losing a game in bo3
BO3_MATCH_WIN_REWARD  =  1.0   # terminal reward for winning the match
BO3_MATCH_LOSS_REWARD = -1.0   # terminal penalty for losing the match

# The total shaping an episode can accumulate must stay strictly below the
# smallest decisive terminal reward magnitude, or shaping could flip the SIGN of
# a finished episode's return — a lost game netting positive reward. The bo1
# terminal is ±1.0; in bo3 the per-game budget is ep_cap/3 (see step()), so a
# full match is also bounded by the cap. Keep the caps under 1.0.
assert SHAPING_EPISODE_CAP < 1.0, SHAPING_EPISODE_CAP
assert SHAPING_EPISODE_CAP_DOOMSDAY < 1.0, SHAPING_EPISODE_CAP_DOOMSDAY
_ACTION_CARD_ID_NULL = -1.0 / N_CARD_TYPES  # null sentinel for non-card slots
_ACTION_CTRL_NULL    = -1.0 / N_CARD_TYPES  # null sentinel for non-entity actions
# MAX_HAND_SLOTS imported from _enums (src/classes/gamestate.h).
_HAND_COST_FEATS  = MAX_HAND_SLOTS * _N_COST_FEATS         # 10 * 7 = 70
_BF_ABILITY_FEATS = MAX_BATTLEFIELD_SLOTS * _N_COST_FEATS  # 48 * 7 = 336
# Action metadata in the obs: cats | ids | ctrl | zone_ref | slot_ref (5 blocks
# of MAX_ACTIONS). pub stays a side-channel (self._action_public), not in the obs.
OBS_SIZE = STATE_SIZE + 5 * MAX_ACTIONS + _HAND_COST_FEATS + _BF_ABILITY_FEATS  # 6700

# ── State layout offsets (mirror src/machine_io.h) ───────────────────────────
# Creatures, lands, and other permanents share one unified section (no separate land slots).
# Card identity is a single normalized id float per slot (idx/N_CARD_TYPES, or
# -1/N_CARD_TYPES for empty/unknown). Decode with round(val * N_CARD_TYPES).
# ── Header field offsets (the first _GLOBAL_SIZE floats) ─────────────────────
# The header is two player blocks, the step one-hot, and three scalar flags.
# Define every offset ONCE here (deriving _GLOBAL_SIZE from them) so that adding
# a player-block field or a header flag only requires editing these constants —
# every consumer that reads the header (decode.py, runner.py, analysis.py,
# scripted_agent.py, opponents.py, this module) imports these names instead of
# hardcoding a literal index. Must mirror push_player_block + the header pushes
# in src/machine_io.h / src/machine_io.cpp serialize_state().
_PLAYER_BLOCK_SIZE = 10                        # floats per player block
# Sub-offsets within one player block:
_PB_LIFE    = 0
_PB_HAND_CT = 1
_PB_POISON  = 2
_PB_MANA    = 3                                # 6 floats: W, U, B, R, G, C
_PB_ENERGY  = 9
_SELF_BLOCK_START  = 0
_OPP_BLOCK_START   = _SELF_BLOCK_START + _PLAYER_BLOCK_SIZE          # 10
_STEP_ONEHOT_START = _OPP_BLOCK_START + _PLAYER_BLOCK_SIZE           # 20
_STEP_ONEHOT_SIZE  = 13                        # UNTAP..CLEANUP, incl. FIRST_STRIKE_DAMAGE
# Step one-hot sub-indices (main phases; SECOND_MAIN is +1 vs. tabletop order
# because FIRST_STRIKE_DAMAGE occupies its own one-hot slot):
_STEP_FIRST_MAIN_IDX  = _STEP_ONEHOT_START + 3                       # 23
_STEP_SECOND_MAIN_IDX = _STEP_ONEHOT_START + 10                      # 30
_IS_ACTIVE_IDX  = _STEP_ONEHOT_START + _STEP_ONEHOT_SIZE            # 33: priority player is active
_SELF_IS_A_IDX  = _IS_ACTIVE_IDX + 1                                # 34: "self" is Player A
_STACK_SIZE_IDX = _SELF_IS_A_IDX + 1                                # 35: stack size / 10
_GLOBAL_SIZE    = _STACK_SIZE_IDX + 1                               # 36: full header width
_PERM_SLOTS             = MAX_BATTLEFIELD_SLOTS  # per-player; 96 total (self + opp)
# 11 status (incl. loyalty) + 2 counters + 4 refs + is_blocked + is_phased_out
# + keyword multi-hot + chosen-name id + returnable-exile id + card id (LAST) = 38.
# Keep the derived formula and cross-check it against the engine's PERM_SLOT_SIZE
# (machine_io.h) so a change to either side is caught here. The three id-family
# floats sit last: chosen_name_id then returnable_exile_id then card_id.
_PERM_SLOT_SIZE         = 19 + N_OBS_KEYWORDS + 3
assert _PERM_SLOT_SIZE == PERM_SLOT_SIZE, (_PERM_SLOT_SIZE, PERM_SLOT_SIZE)
_STACK_SLOTS            = MAX_STACK_DISPLAY
_STACK_XAMT_OFF         = 3                    # x_or_amount / 10 within a stack slot
_STACK_QUAL_START       = 4                    # cast qualifiers (is_copy, kicked, flashback, ...)
_STACK_QUALS            = 7
_STACK_MODE_SLOTS       = MAX_STACK_MODES      # chosen-mode multi-hot width per stack slot
_STACK_MODE_START       = _STACK_QUAL_START + _STACK_QUALS                    # 11
_STACK_TGT_SLOTS        = MAX_STACK_TGTS       # announced-target sub-slots per stack slot
_STACK_TGT_FIELDS       = 5                    # present + is_player + ctrl_is_self + slot_ref + card id (LAST)
_STACK_TGT_START        = _STACK_MODE_START + _STACK_MODE_SLOTS               # 17
# ctrl + card id + is_spell, then x_or_amount + qualifiers, then mode multi-hot,
# then target sub-slots (37 total)
_STACK_SLOT_SIZE        = _STACK_TGT_START + _STACK_TGT_SLOTS * _STACK_TGT_FIELDS
_GY_SLOTS_TOTAL         = 2 * MAX_GY_SLOTS     # 64 self + 64 opponent
_GY_SLOT_SIZE           = 1                    # card id only
_EXILE_SLOTS_TOTAL      = 2 * MAX_GY_SLOTS     # 64 self + 64 opponent (same layout as GY)
_EXILE_SLOT_SIZE        = 1                    # card id only
_HAND_SLOTS_TOTAL       = MAX_HAND_SLOTS
_HAND_SLOT_SIZE         = 1
_ACTION_HISTORY_SIZE    = ACTION_HISTORY_SIZE  # entries in the action history ring (src/classes/game.h)
_ACTION_HISTORY_ENTRY   = 4                    # cat_norm, card_id, is_self, turn/50
_MATCH_CTX_SIZE         = 4                    # game_number, self_wins, opp_wins, sideboard_phase
_LIBRARY_CTX_SIZE       = 3                    # self_lib/60, opp_lib/60, is_post_board
_CUR_TURN_SIZE          = 1                    # current turn / 50
_KNOWN_TOP_LIB_SLOTS    = KNOWN_TOP_LIBRARY_SIZE  # serialized known top-of-library cards
_KNOWN_TOP_LIB_SLOT_SIZE = 1                   # card id per slot
_REVEALED_SIZE          = N_CARD_TYPES         # opponent revealed-cards multi-hot (only vocab-width block)
_OPP_KNOWN_HAND_SLOTS   = MAX_HAND_SLOTS       # known opponent-hand card identities
_OPP_KNOWN_HAND_SLOT_SIZE = 1                  # card id per slot

# Offset chain is fully derived from the block-size constants above and pinned by
# the `assert _EXTRAS_END == STATE_SIZE` below, so absolute indices are intentionally
# NOT annotated here (they went stale when the layout last changed). Cross-reference
# the byte ranges in machine_io.h's layout block, not literals in this file.
_SELF_PERM_START     = _GLOBAL_SIZE
_OPP_PERM_START      = _SELF_PERM_START + _PERM_SLOTS * _PERM_SLOT_SIZE
_STACK_START         = _OPP_PERM_START + _PERM_SLOTS * _PERM_SLOT_SIZE
_GY_START            = _STACK_START + _STACK_SLOTS * _STACK_SLOT_SIZE
_EXILE_START         = _GY_START + _GY_SLOTS_TOTAL * _GY_SLOT_SIZE
_HAND_START          = _EXILE_START + _EXILE_SLOTS_TOTAL * _EXILE_SLOT_SIZE
_HIST_START          = _HAND_START + _HAND_SLOTS_TOTAL * _HAND_SLOT_SIZE
_HIST_END            = _HIST_START + _ACTION_HISTORY_SIZE * _ACTION_HISTORY_ENTRY
_MATCH_CTX_START     = _HIST_END
# _MATCH_CTX layout: game_number, self_wins, opp_wins, is_sideboard_phase.
# The sideboard flag is the single source of truth for "this decision is a bo3
# sideboard root" (used by az_selfplay + opponents.SearchController budget selection).
_IS_SIDEBOARD_IDX    = _MATCH_CTX_START + 3
_LIBRARY_CTX_START   = _MATCH_CTX_START + _MATCH_CTX_SIZE
_CUR_TURN_IDX        = _LIBRARY_CTX_START + _LIBRARY_CTX_SIZE
_KNOWN_TOP_LIB_START = _CUR_TURN_IDX + _CUR_TURN_SIZE
_KNOWN_TOP_LIB_END   = _KNOWN_TOP_LIB_START + _KNOWN_TOP_LIB_SLOTS * _KNOWN_TOP_LIB_SLOT_SIZE
_REVEALED_START      = _KNOWN_TOP_LIB_END
_REVEALED_END        = _REVEALED_START + _REVEALED_SIZE
_OPP_KNOWN_HAND_START = _REVEALED_END
_OPP_KNOWN_HAND_END  = _OPP_KNOWN_HAND_START + _OPP_KNOWN_HAND_SLOTS * _OPP_KNOWN_HAND_SLOT_SIZE
# Pending decision context: card id of the spell/ability currently making a
# mid-resolution choice (target select, dig/search/scry pick, discard, modal, ...;
# sentinel = none) + its controller-is-viewer flag. The source may not be on the
# stack yet (targets are announced before the spell moves there), so this is the
# only place the observation shows WHAT is asking for the current choice.
_PENDING_DECISION_START = _OPP_KNOWN_HAND_END
_PENDING_DECISION_SIZE  = 2                    # source card id + ctrl_is_self
_PENDING_DECISION_END   = _PENDING_DECISION_START + _PENDING_DECISION_SIZE
# Global extras (see machine_io.h [5955-5973]): self/opp lands_played/10,
# viewer_has_priority, self/opp is_monarch, self/opp city's blessing, self/opp
# revolt, self/opp pending extra turns/3, is_day, is_night, then the
# MandatoryChoice one-hot (NONE at index 0).
_EXTRAS_START        = _PENDING_DECISION_END
_EXTRAS_LANDS_SELF   = _EXTRAS_START + 0
_EXTRAS_LANDS_OPP    = _EXTRAS_START + 1
_EXTRAS_HAS_PRIORITY = _EXTRAS_START + 2
_EXTRAS_MONARCH_SELF = _EXTRAS_START + 3
_EXTRAS_MONARCH_OPP  = _EXTRAS_START + 4
_EXTRAS_BLESSING_SELF = _EXTRAS_START + 5
_EXTRAS_BLESSING_OPP = _EXTRAS_START + 6
_EXTRAS_REVOLT_SELF  = _EXTRAS_START + 7
_EXTRAS_REVOLT_OPP   = _EXTRAS_START + 8
_EXTRAS_EXTRA_TURNS_SELF = _EXTRAS_START + 9
_EXTRAS_EXTRA_TURNS_OPP  = _EXTRAS_START + 10
_EXTRAS_IS_DAY       = _EXTRAS_START + 11
_EXTRAS_IS_NIGHT     = _EXTRAS_START + 12
_EXTRAS_MC_ONEHOT_START = _EXTRAS_START + 13
_EXTRAS_END          = _EXTRAS_MC_ONEHOT_START + N_MANDATORY_CHOICES

# ── Deck-identity tail blocks (mirror machine_io.h [5974-6195]) ──────────────
# Each slot is (card_id, count): card id via norm_card_id (empty = -1 sentinel),
# count normalized /4.0. Slots packed ascending by vocab id, no holes.
#   SELF_LIVE_LIBRARY : the viewer's LIBRARY zone tallied live (viewer-only).
#   OPP_DECK_MAIN / OPP_DECK_SIDE : the opponent's STATIC decklist (post-board).
_DECKLIST_SLOT_SIZE     = 2                       # card id + count per slot
_SELF_LIVE_LIB_START    = _EXTRAS_END
_SELF_LIVE_LIB_END      = _SELF_LIVE_LIB_START + DECKLIST_MAIN_SLOTS * _DECKLIST_SLOT_SIZE
_OPP_DECK_MAIN_START    = _SELF_LIVE_LIB_END
_OPP_DECK_MAIN_END      = _OPP_DECK_MAIN_START + DECKLIST_MAIN_SLOTS * _DECKLIST_SLOT_SIZE
_OPP_DECK_SIDE_START    = _OPP_DECK_MAIN_END
_OPP_DECK_SIDE_END      = _OPP_DECK_SIDE_START + DECKLIST_SIDE_SLOTS * _DECKLIST_SLOT_SIZE

assert _OPP_DECK_SIDE_END == STATE_SIZE, (_OPP_DECK_SIDE_END, STATE_SIZE)

# Offsets of the three id-family floats within a permanent slot (all LAST): the
# chosen-name id (Permanent::chosen_name — Pithing Needle / Disruptor Flute named
# card, Petrified Hamlet named land), then the returnable-exile id (the card this
# permanent has exiled that still has a return path — Static Prison / Phelia), then
# the card id.
_PERM_CHOSEN_NAME_OFF = _PERM_SLOT_SIZE - 3    # 35
_PERM_RETURNABLE_OFF = _PERM_SLOT_SIZE - 2     # 36
_PERM_CARD_OFF = _PERM_SLOT_SIZE - 1           # 37 (card id is always LAST)

# Unified entity-reference slot space (machine_io.h): 0-47 self perm slots,
# 48-95 opp perm slots, 96-107 stack slots, -1 = none. In the float state
# vector (and the obs action block) a ref is normalized (idx + 1) / 108, so
# 0.0 is the "none" sentinel; decode with round(v * N_ENTITY_REF_SLOTS) - 1.
N_ENTITY_REF_SLOTS = 2 * _PERM_SLOTS + _STACK_SLOTS  # 108


# ── Sideboard observation mask ───────────────────────────────────────────────
# At bo3 sideboard time the engine keeps the ended game's ECS alive, so the
# state vector describes the STALE terminal board of the previous game — noise
# for a sideboarding decision. When is_sideboard_phase (state[_MATCH_CTX_START+3])
# is set we zero every block except the ones that actually inform sideboarding:
# graveyards + exile ("how the game went"), action history (last-game tempo + in-phase
# swap context), match/library/turn context (game number => play/draw), the
# opponent revealed-cards multi-hot (the primary signal), and the pending-decision
# context (which IN card the OUT query is cutting for). The global-extras block
# (lands played, monarch, day/night, MandatoryChoice one-hot, ...) describes the
# stale ended game, so it stays masked; it holds no card-id slots. Card-id slots
# must be filled with the empty sentinel (-1/N_CARD_TYPES), NOT 0.0 — 0.0 decodes
# to a real vocab index 0 and defeats the extractor's empty-slot masking.
def _build_sideboard_mask():
    keep = np.zeros(STATE_SIZE, dtype=bool)
    for lo, hi in (
        (_GY_START, _HAND_START),                   # graveyards + exile (self + opp)
        (_HIST_START, _HIST_END),                   # action history ring
        (_MATCH_CTX_START, _KNOWN_TOP_LIB_START),   # match + library ctx + current turn
        (_REVEALED_START, _REVEALED_END),           # opponent revealed multi-hot
        (_PENDING_DECISION_START, _PENDING_DECISION_END),  # pending-decision context
        # The opponent's STATIC decklist is exactly what informs sideboarding, so
        # both opp-deck blocks stay visible. The SELF_LIVE_LIBRARY block is NOT
        # kept (the library zone is stale during the sideboard phase); its card-id
        # positions are sentinel-filled below, its counts masked to 0.0.
        (_OPP_DECK_MAIN_START, _OPP_DECK_SIDE_END),
    ):
        keep[lo:hi] = True
    # The "self is Player A" flag MUST survive the mask: it is the seat-routing
    # signal for every obs consumer (runner.drive_game's controller pick, the
    # training envs' opponent-turn gate, decode's seat labels). The engine sets
    # it from sideboard_phase_player during sideboarding, so it is live, not
    # stale; masking it to 0 routed BOTH players' sideboard decisions to seat B.
    keep[_SELF_IS_A_IDX] = True
    fill = np.zeros(STATE_SIZE, dtype=np.float32)
    # Card-id positions inside masked blocks must decode to "empty", not vocab 0.
    card_id_idx = []
    for s in range(_PERM_SLOTS):                     # self + opp permanent slots
        # All three id-family floats per slot: chosen-name id, returnable-exile id, card id.
        for off in (_PERM_CHOSEN_NAME_OFF, _PERM_RETURNABLE_OFF, _PERM_CARD_OFF):
            card_id_idx.append(_SELF_PERM_START + s * _PERM_SLOT_SIZE + off)
            card_id_idx.append(_OPP_PERM_START + s * _PERM_SLOT_SIZE + off)
    for s in range(_STACK_SLOTS):                    # stack object id + target sub-slot ids
        base = _STACK_START + s * _STACK_SLOT_SIZE
        card_id_idx.append(base + 1)                 # object card id
        tgt0 = base + _STACK_TGT_START               # first target sub-slot
        for t in range(_STACK_TGT_SLOTS):
            card_id_idx.append(tgt0 + t * _STACK_TGT_FIELDS + (_STACK_TGT_FIELDS - 1))
    for i in range(_HAND_START, _HAND_START + _HAND_SLOTS_TOTAL):      # self hand
        card_id_idx.append(i)
    for i in range(_KNOWN_TOP_LIB_START, _KNOWN_TOP_LIB_END):          # known top-5 library
        card_id_idx.append(i)
    for i in range(_OPP_KNOWN_HAND_START, _OPP_KNOWN_HAND_END):        # known opp hand
        card_id_idx.append(i)
    for s in range(DECKLIST_MAIN_SLOTS):                               # self live library
        # card id is the first float of each (card_id, count) slot; count masks to 0.0
        card_id_idx.append(_SELF_LIVE_LIB_START + s * _DECKLIST_SLOT_SIZE)
    for i in card_id_idx:
        if not keep[i]:
            fill[i] = _ACTION_CARD_ID_NULL
    return keep, fill


_SB_MASK_KEEP, _SB_MASK_FILL = _build_sideboard_mask()


def _slot_card_idx(obs, i):
    """Decode the vocab index from a single normalized id float at obs[i] (-1 = empty)."""
    return int(round(float(obs[i]) * N_CARD_TYPES))


# Status offsets within a permanent slot (mirror the per-slot layout in
# src/machine_io.h; the card id sits LAST, at _PERM_CARD_OFF)
_OFF_POWER        = 0
_OFF_TOUGHNESS    = 1
_OFF_IS_TAPPED    = 2
_OFF_IS_ATTACKING = 3
_OFF_IS_BLOCKING  = 4
_OFF_HAS_SICKNESS = 5
_OFF_DAMAGE       = 6
_OFF_CTRL_IS_SELF = 7
_OFF_IS_CREATURE  = 8    # 1.0 if this slot is a creature
_OFF_IS_LAND      = 9    # 1.0 if this slot is a land
_OFF_LOYALTY      = 10   # loyalty / 10 (planeswalkers; 0 otherwise)
_OFF_P1P1_NET     = 11   # net (+1/+1 minus -1/-1) counters / 10, SIGNED
_OFF_OTHER_COUNTERS = 12 # total counters of every other kind / 10
_OFF_ATTACHED_TO  = 13   # norm_ref: what this equipment/aura is attached to
_OFF_ATTACHED_BY  = 14   # norm_ref: the equipment/aura attached to this
_OFF_ATTACK_TGT   = 15   # norm_ref: attacked walker's slot (0.0 = the player)
_OFF_BLOCKING_TGT = 16   # norm_ref: the attacker this blocker blocks
_OFF_IS_BLOCKED   = 17   # attacker was blocked at declare-blockers (CR 509.1h)
_OFF_IS_PHASED_OUT = 18  # phased-out permanents ARE serialized, with this set
_OFF_KEYWORDS_START = 19 # effective keyword multi-hot (N_OBS_KEYWORDS wide,
                         # _OBS_KEYWORDS order from _enums.py)

_SELF_PERM_POWER_IDX = np.arange(_PERM_SLOTS) * _PERM_SLOT_SIZE + _SELF_PERM_START
_SELF_PERM_CREATURE_IDX = _SELF_PERM_POWER_IDX + _OFF_IS_CREATURE
_SELF_PERM_PHASED_IDX = _SELF_PERM_POWER_IDX + _OFF_IS_PHASED_OUT
_OPP_PERM_POWER_IDX = np.arange(_PERM_SLOTS) * _PERM_SLOT_SIZE + _OPP_PERM_START
_OPP_PERM_CREATURE_IDX = _OPP_PERM_POWER_IDX + _OFF_IS_CREATURE
_OPP_PERM_PHASED_IDX = _OPP_PERM_POWER_IDX + _OFF_IS_PHASED_OUT

def _board_power_advantage(obs):
    """Return self_power - opp_power from the observation vector.

    Phased-out permanents (CR 702.26e) are serialized with real power and
    is_creature=1 but count as nonexistent, so exclude them from the sum."""
    self_mask = (obs[_SELF_PERM_CREATURE_IDX] > 0.5) & (obs[_SELF_PERM_PHASED_IDX] < 0.5)
    self_power = np.sum(obs[_SELF_PERM_POWER_IDX[self_mask]]) * 10.0
    opp_mask = (obs[_OPP_PERM_CREATURE_IDX] > 0.5) & (obs[_OPP_PERM_PHASED_IDX] < 0.5)
    opp_power = np.sum(obs[_OPP_PERM_POWER_IDX[opp_mask]]) * 10.0
    return self_power - opp_power


def _shaping_potential(obs):
    """Shaping potential Φ(s): weighted hand-count + board-power advantage.

    Applied as the delta Φ(s') - Φ(s) between consecutive model decision
    points, with Φ = 0 past the end of a game, so the per-game sum telescopes
    to zero and the shaping cannot change which policies are optimal."""
    hand_adv = max(0.0, obs[_SELF_BLOCK_START + _PB_HAND_CT]
                   - obs[_OPP_BLOCK_START + _PB_HAND_CT]) * 10.0
    return (SHAPING_HAND_ADV_PER_CARD * hand_adv
            + SHAPING_POWER_ADV_PER_PT * _board_power_advantage(obs))


def _deck_named(deck, key):
    """True if `key` occurs anywhere in the deck name (e.g. 'league/car_doomsday').

    The uniform way archetype-specific training behavior is keyed: doomsday
    (combo) and tron (big-mana ramp that hoards its hand by design) both opt
    out of the potential-based shaping this way."""
    return deck is not None and key in deck.lower()

# CLI constants live in cli_spec.py (the single source shared with the TUI).
from cli_spec import REPO_ROOT as _REPO_ROOT, BINARY, BIN_DIR  # noqa: E402


class RoboMageEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # Maximum decision steps per episode before truncation.  Prevents infinite
    # loops (e.g. a model that toggles attackers forever) from hanging training.
    MAX_STEPS = 1000
    MAX_STEPS_BO3 = 3000  # higher limit for best-of-three matches

    def __init__(self, binary_path: str = BINARY, render_mode=None,
                 deck_a: str | None = None, deck_b: str | None = None,
                 bo3: bool = False, auto_sideboard: bool = False,
                 narrative: bool = False, no_shuffle: bool = False,
                 battlefield_a: str | None = None, battlefield_b: str | None = None,
                 graveyard_a: str | None = None, graveyard_b: str | None = None,
                 exile_a: str | None = None, exile_b: str | None = None,
                 sideboard_a: str | None = None, sideboard_b: str | None = None,
                 life_a: int | None = None, life_b: int | None = None,
                 log_viewer: str | None = None, log_decisions: bool = False):
        super().__init__()
        self.binary_path = os.path.realpath(binary_path)
        self.render_mode = render_mode
        self._deck_a = deck_a
        self._deck_b = deck_b
        self._bo3 = bo3
        self._auto_sideboard = auto_sideboard
        # State-seeding / observation flags passed through to the engine. These
        # mirror the test-harness CLI so the shared run loop (runner.run_games)
        # can sculpt a starting state: --narrative emits the full game log,
        # --no-shuffle keeps deck-file order = draw order, and --battlefield-a/-b
        # pre-place permanents (comma-joined deck-name strings).
        self._narrative = narrative
        self._no_shuffle = no_shuffle
        self._battlefield_a = battlefield_a
        self._battlefield_b = battlefield_b
        self._graveyard_a = graveyard_a
        self._graveyard_b = graveyard_b
        # --exile-a/-b and --sideboard-a/-b pre-place cards in those non-battlefield
        # zones so zone-change effects that pull from exile / "outside the game"
        # (e.g. Karn, the Great Creator's -2) can be exercised in isolation.
        self._exile_a = exile_a
        self._exile_b = exile_b
        self._sideboard_a = sideboard_a
        self._sideboard_b = sideboard_b
        # --life-a/-b override the starting life total (default 20) so life-payment
        # costs (fetch lands, Toxic Deluge, phyrexian mana) can be tested at any life.
        self._life_a = life_a
        self._life_b = life_b
        # When set ("A"/"B"), the engine redacts game_log_private narrative to
        # that seat's view (hidden draws, tutored/top-of-library cards) without
        # rerouting input — both seats still respond over the machine protocol.
        self._log_viewer = log_viewer
        # Machine mode writes no decision log by default (training runs millions of
        # episodes); log_decisions=True passes --log-decisions so a harness/observe
        # run can produce a self-contained RMLOG v2 replay log on request.
        self._log_decisions = log_decisions

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        # Discrete action space sized to the max we'd ever see.
        # Invalid actions are masked at each step via `action_masks()`.
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        self._proc = None
        self._num_choices = 1
        self._obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self._action_public = np.zeros(MAX_ACTIONS, dtype=np.float32)  # card_is_public per action
        self._action_cats = np.zeros(MAX_ACTIONS, dtype=np.int32)  # raw ActionCategory per action
        # Under --search-server the engine precedes each query with a
        # "SEARCHINFO safe=<0|1>" marker (whether SNAPSHOT/DETERMINIZE are legal
        # at this decision); None when the flag is off / no query seen yet.
        self.last_search_safe = None
        self._action_descriptions = None  # list[str] per action under --narrative, else None
        self._perm_counters = None        # (self[48], opp[48]) counter summaries under --narrative, else None
        self._perm_token_names = None     # (self[48], opp[48]) token names under --narrative, else None
        self._pending_confirm = False  # True when last query used the -1 convention
        self._step_count = 0
        self.last_engine_seed = None  # engine --seed of the most recent reset()

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._kill_proc()
        # Generate a unique seed for each game so time(nullptr) collisions don't
        # produce repeated games when many resets happen within the same second.
        # A caller can force a specific engine seed (options={"engine_seed": N})
        # to replay a previously-collected game deterministically; the seed
        # actually used is exposed as self.last_engine_seed.
        if options is not None and "engine_seed" in options:
            rng_seed = int(options["engine_seed"])
        else:
            rng_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.last_engine_seed = rng_seed
        cmd = [self.binary_path, "--machine", "--seed", str(rng_seed)]
        if self._narrative:
            cmd += ["--narrative"]
        if self._bo3:
            cmd += ["--bo3"]
        if self._deck_a:
            cmd += ["--deck-a", self._deck_a]
        if self._deck_b:
            cmd += ["--deck-b", self._deck_b]
        if self._no_shuffle:
            cmd += ["--no-shuffle"]
        if self._battlefield_a:
            cmd += ["--battlefield-a", self._battlefield_a]
        if self._battlefield_b:
            cmd += ["--battlefield-b", self._battlefield_b]
        if self._graveyard_a:
            cmd += ["--graveyard-a", self._graveyard_a]
        if self._graveyard_b:
            cmd += ["--graveyard-b", self._graveyard_b]
        if self._exile_a:
            cmd += ["--exile-a", self._exile_a]
        if self._exile_b:
            cmd += ["--exile-b", self._exile_b]
        if self._sideboard_a:
            cmd += ["--sideboard-a", self._sideboard_a]
        if self._sideboard_b:
            cmd += ["--sideboard-b", self._sideboard_b]
        if self._life_a is not None:
            cmd += ["--life-a", str(self._life_a)]
        if self._life_b is not None:
            cmd += ["--life-b", str(self._life_b)]
        if self._log_viewer:
            cmd += ["--log-viewer", self._log_viewer]
        if self._log_decisions:
            cmd += ["--log-decisions"]
        cmd += self._extra_engine_flags()
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit parent stderr so engine errors are visible
            bufsize=-1,  # binary mode, fully buffered
            cwd=BIN_DIR,  # game uses getcwd() to locate resources/
        )
        obs, info = self._read_until_query()
        return obs, info

    def step(self, action: int):
        assert self._proc is not None, "Call reset() first"

        self._step_count += 1
        max_steps = self.MAX_STEPS_BO3 if self._bo3 else self.MAX_STEPS
        if self._step_count >= max_steps:
            self._kill_proc()
            # Return the last real observation, not zeros: SB3 bootstraps
            # V(terminal_observation) on truncation, and an all-zero vector is
            # not even a valid empty encoding (card-id 0.0 decodes to vocab
            # index 0, not the -1/N empty sentinel), so its value estimate
            # would be garbage. The pre-action state is a faithful stand-in.
            return self._obs.copy(), 0.0, False, True, {}

        # Remap: if the last query included a confirm slot (num_choices had +1),
        # and the agent chose the last index, send -1 to the game.
        game_action = action
        if self._pending_confirm and action == self._num_choices - 1:
            game_action = -1

        self._send(game_action)

        obs, info = self._read_until_query()
        reward = info.get("reward", 0.0)
        terminated = info.get("done", False)
        return obs, reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        """Boolean mask of valid actions for the current step (for MaskablePPO)."""
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        mask[: self._num_choices] = True
        return mask

    def close(self):
        self._kill_proc()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extra_engine_flags(self) -> list:
        """Additional engine CLI flags; subclass hook (SearchRoboMageEnv adds
        --search-server) so reset() stays the single command builder."""
        return []

    def _send(self, action: int):
        self._proc.stdin.write(f"{action}\n".encode())
        self._proc.stdin.flush()

    def _read_exactly(self, n: int) -> bytes:
        """Read exactly n bytes from stdout, blocking until available."""
        buf = bytearray()
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                raise EOFError("Game process ended while reading binary payload")
            buf.extend(chunk)
        return bytes(buf)

    def _read_until_query(self):
        """
        Read lines from the game process until a BQUERY line appears or the
        process exits (game over).  After the BQUERY header line, reads the
        binary payload: float32[STATE_SIZE], int32[MAX_ACTIONS] cats,
        float32[MAX_ACTIONS] ids, float32[MAX_ACTIONS] ctrl,
        float32[MAX_ACTIONS] pub, int32[MAX_ACTIONS] zone,
        int32[MAX_ACTIONS] refs.
        Returns (obs, info).
        """
        reward = 0.0
        done = False
        game_result = False
        shaping_a = 0.0
        shaping_b = 0.0

        while True:
            line = self._proc.stdout.readline()

            if not line:
                # Process ended. A clean exit is game over; a nonzero exit means
                # the engine aborted (e.g. fatal_error on a bad decklist) — fail
                # loudly instead of handing the driver a dead env.
                rc = self._proc.wait()
                if rc != 0:
                    self._kill_proc()
                    raise RuntimeError(
                        f"game process exited with code {rc} before the game "
                        "finished — see the engine's FATAL/ERROR output on stderr")
                done = True
                break

            line = line.rstrip(b"\n")

            # Detect win/loss
            if self._bo3:
                # In bo3 mode, individual game results are intermediate rewards
                if line.startswith(b"GAME_RESULT:"):
                    game_result = True
                    if b"Player A wins" in line:
                        reward += BO3_GAME_WIN_REWARD
                    elif b"Player B wins" in line:
                        reward += BO3_GAME_LOSS_REWARD
                elif line.startswith(b"MATCH_RESULT:"):
                    if b"Player A wins" in line:
                        reward += BO3_MATCH_WIN_REWARD
                    elif b"Player B wins" in line:
                        reward += BO3_MATCH_LOSS_REWARD
                    done = True
            else:
                if b"Player A wins" in line:
                    reward = 1.0
                    done = True
                elif b"Player B wins" in line:
                    reward = -1.0
                    done = True

            # Shaping signal: mana wasted at end of phase (pool non-empty on drain)
            if line.startswith(b"MANA_WASTED: "):
                if line.endswith(b"A"):
                    shaping_a += SHAPING_MANA_WASTED
                elif line.endswith(b"B"):
                    shaping_b += SHAPING_MANA_WASTED
                continue

            # Shaping signal: excessive mulligan (3rd and beyond)
            if line.startswith(b"MULLIGAN_PENALTY: "):
                if line.endswith(b"A"):
                    shaping_a += SHAPING_MULLIGAN_PENALTY
                elif line.endswith(b"B"):
                    shaping_b += SHAPING_MULLIGAN_PENALTY
                continue

            # Search-server marker preceding each query under --search-server
            # ("SEARCHINFO safe=<0|1>"); absent otherwise. Captured so a search
            # driver knows whether SNAPSHOT/DETERMINIZE are legal right now.
            if line.startswith(b"SEARCHINFO"):
                self.last_search_safe = line.endswith(b"safe=1")
                continue

            if line.startswith(b"BQUERY: "):
                self._parse_bquery_payload(line)

                # Auto-sideboard: if enabled, automatically pick "done" (action 0)
                # for all sideboard queries so the model never sees them.
                if self._auto_sideboard and any(
                    c == _CAT_SB_DONE for c in self._action_cats[:self._num_choices]
                ):
                    self._send(0)
                    continue

                break

            # Non-BQUERY output: optionally print for human render mode
            if self.render_mode == "human":
                self._print_narrative_line(line.decode("utf-8", errors="replace"))

        info = {"reward": reward, "done": done, "shaping_a": shaping_a, "shaping_b": shaping_b,
                "game_result": game_result}
        if done:
            self._kill_proc()
            # Return a zero obs on terminal step — will be replaced by reset()
            return np.zeros(OBS_SIZE, dtype=np.float32), info

        # self._obs is a reused preallocated buffer mutated in place each step;
        # callers that retain the array across steps must .copy() it themselves.
        return self._obs, info

    def _parse_bquery_payload(self, line: bytes) -> None:
        """Parse one BQUERY header line + binary payload into the per-decision
        state: fills self._obs in place and sets _num_choices, _action_cats,
        _action_public, _pending_confirm, and the narrative side-channels.
        Shared by the training reader (_read_until_query) and the search-server
        simulation reader (SearchRoboMageEnv), which must decode identical
        observations for tree-leaf evaluation.
        """
        # Header: "BQUERY: <num_choices> <STATE_SIZE> <MAX_ACTIONS>".
        # The two trailing sizes are a runtime layout handshake — assert
        # them against our imported constants so a C++ layout change
        # without regenerated Python constants fails loudly here instead
        # of silently misframing the binary payload below.
        fields = line[8:].split()
        if len(fields) != 3:
            raise RuntimeError(
                "malformed BQUERY header "
                f"{line!r}: expected 'BQUERY: <N> <STATE_SIZE> <MAX_ACTIONS>' "
                "(3 fields) — engine and Python driver are out of sync; "
                "regenerate train/_enums.py via `make regen`")
        n = int(fields[0])
        eng_state_size = int(fields[1])
        eng_max_actions = int(fields[2])
        if eng_state_size != STATE_SIZE or eng_max_actions != MAX_ACTIONS:
            raise RuntimeError(
                "BQUERY layout mismatch — "
                f"engine STATE_SIZE={eng_state_size} != python STATE_SIZE={STATE_SIZE}; "
                f"engine MAX_ACTIONS={eng_max_actions} != python MAX_ACTIONS={MAX_ACTIONS} — "
                "regenerate train/_enums.py via `make regen`")
        self._num_choices = min(n, MAX_ACTIONS)

        # Binary reads: state floats, then padded action metadata
        state_arr = np.frombuffer(
            self._read_exactly(_BQUERY_STATE_BYTES), dtype=np.float32).copy()
        cats_int = np.frombuffer(
            self._read_exactly(_BQUERY_CATS_BYTES), dtype=np.int32)
        id_arr = np.frombuffer(
            self._read_exactly(_BQUERY_IDS_BYTES), dtype=np.float32).copy()
        ctrl_arr = np.frombuffer(
            self._read_exactly(_BQUERY_CTRL_BYTES), dtype=np.float32).copy()
        pub_arr = np.frombuffer(
            self._read_exactly(_BQUERY_PUB_BYTES), dtype=np.float32).copy()
        zone_int = np.frombuffer(
            self._read_exactly(_BQUERY_ZONE_BYTES), dtype=np.int32)
        refs_int = np.frombuffer(
            self._read_exactly(_BQUERY_REFS_BYTES), dtype=np.int32)

        # Per-action "card identity is public" flags (revealed tutors). Kept as a
        # side-channel — observers (TUI) read it; not part of the ML observation
        # vector yet, so OBS_SIZE and trained checkpoints are unaffected.
        self._action_public = pub_arr
        self._action_cats = cats_int

        # Under --narrative the engine appends a fixed char block of
        # per-action descriptions (the exact CLI labels). Read and
        # decode it so observers can show "Target Player B", "Pay 4 life",
        # etc. — things the numeric metadata can't express. Off the
        # training path (narrative=False), so OBS is unaffected.
        if self._narrative:
            self._action_descriptions = _decode_char_block(
                self._read_exactly(_BQUERY_DESC_BYTES),
                MAX_ACTIONS, MAX_CHOICE_DESC)
            # Per-permanent counter summaries (side-channel, like the
            # descriptions): (self_slots, opp_slots), aligned with the
            # state vector's permanent blocks.
            ctrs = _decode_char_block(
                self._read_exactly(_BQUERY_PERM_CTRS_BYTES),
                2 * N_PERM_SLOTS, PERM_COUNTERS_LEN)
            self._perm_counters = (ctrs[:N_PERM_SLOTS], ctrs[N_PERM_SLOTS:])
            # Per-permanent token names (side-channel, like the counters):
            # (self_slots, opp_slots), slot-aligned with the permanent blocks.
            # Non-empty only for token permanents.
            toks = _decode_char_block(
                self._read_exactly(_BQUERY_PERM_TOKS_BYTES),
                2 * N_PERM_SLOTS, PERM_TOKEN_NAME_LEN)
            self._perm_token_names = (toks[:N_PERM_SLOTS], toks[N_PERM_SLOTS:])
        else:
            self._action_descriptions = None
            self._perm_counters = None
            self._perm_token_names = None

        # The -1 confirm convention applies to mandatory attacker/blocker queries.
        self._pending_confirm = any(
            c in MANDATORY_CATS for c in cats_int[:self._num_choices])

        # Sideboard decisions observe the stale terminal board of the
        # previous game — mask it down to the sideboard-relevant blocks
        # (see _build_sideboard_mask). Applied before the cost gathers so
        # derived hand/bf cost features zero out consistently.
        if state_arr[_MATCH_CTX_START + 3] > 0.5:
            np.copyto(state_arr, _SB_MASK_FILL, where=~_SB_MASK_KEEP)

        # Hand cast costs: gather cost rows by hand-slot card id
        hand_ids = np.rint(
            state_arr[_HAND_START:_HAND_START + MAX_HAND_SLOTS] * N_CARD_TYPES).astype(np.intp)
        hand_costs = _gather_costs(_CARD_COST_MATRIX, hand_ids)

        # Battlefield activated ability costs (48 self permanent slots)
        bf_ids = np.rint(state_arr[_BF_ID_IDX] * N_CARD_TYPES).astype(np.intp)
        bf_ability_costs = _gather_costs(_CARD_ABILITY_COST_MATRIX, bf_ids)

        # Write sections into preallocated obs buffer (avoids concatenate allocation)
        o = self._obs
        o[:STATE_SIZE] = state_arr
        _act_end = STATE_SIZE + MAX_ACTIONS
        o[STATE_SIZE:_act_end] = cats_int / ACTION_CATEGORY_MAX
        o[_act_end:_act_end + MAX_ACTIONS] = id_arr
        o[_act_end + MAX_ACTIONS:_act_end + 2 * MAX_ACTIONS] = ctrl_arr
        o[_act_end + 2 * MAX_ACTIONS:_act_end + 3 * MAX_ACTIONS] = (
            zone_int / REF_ZONE_MAX)
        # Entity-slot refs, normalized like the in-state ref fields:
        # (idx + 1) / 108, so -1 (none) lands exactly on 0.0.
        o[_act_end + 3 * MAX_ACTIONS:_act_end + 4 * MAX_ACTIONS] = (
            (refs_int + 1) / N_ENTITY_REF_SLOTS)
        _hc_start = _act_end + 4 * MAX_ACTIONS
        o[_hc_start:_hc_start + _HAND_COST_FEATS] = hand_costs.ravel()
        _bf_start = _hc_start + _HAND_COST_FEATS
        o[_bf_start:_bf_start + _BF_ABILITY_FEATS] = bf_ability_costs.ravel()

    def _print_narrative_line(self, line: str):
        print(line, file=sys.stderr)

    def _kill_proc(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.stdout.close()
                self._proc.kill()
                self._proc.wait()
            except Exception:
                pass
            self._proc = None


class NarrativeEnv(RoboMageEnv):
    """RoboMageEnv that collects non-BQUERY game lines into a list instead of printing them.

    Defaults ``narrative=True`` so the engine emits its full game log (casts,
    damage, zone moves, combat); ``flush_lines`` hands those buffered lines to
    the caller. This is what makes the shared observation loop show the same
    narrative the test harness does.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("render_mode", "human")
        kwargs.setdefault("narrative", True)
        super().__init__(**kwargs)
        self.lines: list = []

    def _print_narrative_line(self, line: str):
        self.lines.append(line)

    def flush_lines(self) -> list:
        out, self.lines = self.lines, []
        return out


# ── Action category constants ────────────────────────────────────────────────
# The integer values come from the C++ ActionCategory enum via _enums.py (the
# CAT_* names imported at the top of this module). These short local aliases
# (and the mandatory/mana category sets) are the names the rest of the Python
# side — env, scripted_agent (imports these _CAT_* from env), decode — uses.
_CAT_PASS       = CAT_PASS_PRIORITY
_CAT_MANA       = CAT_MANA_ABILITY  # legacy, no longer emitted by the game (mana
                      # activations arrive as the per-color MANA_W..MANA_C cats — see _MANA_CATS)
_MANA_CATS      = frozenset(range(CAT_MANA_W, CAT_MANA_C + 1))  # MANA_W..MANA_C — mana-source
                      # activations (machine mode: only instant-speed cracks, e.g. LED, at priority)
_CAT_MANA_U     = CAT_MANA_U  # tap for blue mana — the color the scripted Doomsday
                      # agent floats off a Lion's Eye Diamond crack (Oracle's UU)
_CAT_SEL_ATK    = CAT_SELECT_ATTACKER
# Attacker/blocker confirm categories — the ONE canonical definition (decode.py
# imports MANDATORY_CATS from here). Built from the CAT_* names, not raw ints.
MANDATORY_CATS  = frozenset({CAT_SELECT_ATTACKER, CAT_CONFIRM_ATTACKERS,
                             CAT_SELECT_BLOCKER, CAT_CONFIRM_BLOCKERS})
_CAT_CONF_ATK   = CAT_CONFIRM_ATTACKERS
_CAT_SEL_BLK    = CAT_SELECT_BLOCKER
_CAT_CONF_BLK   = CAT_CONFIRM_BLOCKERS
_CAT_ACTIVATE   = CAT_ACTIVATE_ABILITY  # activate a non-mana ability (fetch lands, Wasteland destroy)
_CAT_CAST       = CAT_CAST_SPELL
_CAT_TARGET     = CAT_SELECT_TARGET
_CAT_LAND       = CAT_PLAY_LAND
_CAT_MULLIGAN   = CAT_MULLIGAN
_CAT_SEARCH     = CAT_SEARCH_LIBRARY  # search library (action 0 = fail to find, 1+ = actual cards)
_CAT_OTHER      = CAT_OTHER_CHOICE  # generic/unclassified choice (fallback default)
_CAT_DISCARD    = CAT_DISCARD  # choose a card to discard (cost, effect, or cleanup)
_CAT_PAYING     = CAT_PAYING_COSTS  # paying costs (tap lands for mana, pitch cards)
_CAT_CHOOSE_X   = CAT_CHOOSE_X  # X-value ladder, or delve exile count (delve count actions carry
                      # the spell's card id; an X ladder has the null card-id sentinel)
_CAT_CHOOSE_CARD = CAT_CHOOSE_CARD  # choose a card from a non-library zone (e.g. delve per-card exile)
_CAT_DIG        = CAT_DIG_CHOICE  # dig choice (Once Upon a Time: pick creature/land from top N)
_CAT_SB_IN      = CAT_SIDEBOARD_IN  # sideboard: choose card from sideboard to add
_CAT_SB_OUT     = CAT_SIDEBOARD_OUT  # sideboard: choose card from main deck to remove
_CAT_SB_DONE    = CAT_SIDEBOARD_DONE  # sideboard: finish sideboarding
_CAT_COMPANION  = CAT_COMPANION  # pay {3}: put your companion from the sideboard into your hand
_CAT_YESNO      = CAT_OPTIONAL_YESNO  # optional yes/no confirmation (OPTIONAL_YESNO; 0=Decline, 1=Accept)

# Colored mana requirements per card vocab index (card_vocab.h).
# Keys are color pool indices: W=0, U=1, B=2, R=3, G=4, C=5.
# Generic mana is omitted — any color satisfies it.
_CARD_COLORED_COSTS = {
    2:  {3: 1},   # Lightning Bolt          (R)   — 1 red
    11: {1: 1},   # Ponder                  (U)   — 1 blue
    13: {1: 1},   # Daze                    (1U)  — 1 blue
    16: {1: 1},   # Delver of Secrets       (U)   — 1 blue
    18: {1: 1},   # Flying Men              (U)   — 1 blue
    20: {3: 1},   # Dragon's Rage Channeler (R)   — 1 red
    21: {1: 2},   # Air Elemental           (3UU) — 2 blue
    22: {1: 2},   # Counterspell            (UU)  — 2 blue
    23: {3: 1},   # Lightning Strike        (1R)  — 1 red
    24: {1: 1},   # Brainstorm              (U)   — 1 blue
    26: {1: 2},   # Murktide Regent         (5UU, Delve) — 2 blue
    28: {3: 1},   # Cori-Steel Cutter       (1R)  — 1 red
    29: {3: 1},   # Unholy Heat             (R)   — 1 red
}

# ── Battlefield layout (aliases of the unified state offsets above) ─────────
_BF_START         = _SELF_PERM_START           # 36
_BF_SLOT_SIZE     = _PERM_SLOT_SIZE            # 36
_PERM_A_SLOTS     = _PERM_SLOTS                # 48: self occupies perm slots 0-47, opponent slots 48-95
_BF_CARD_OFF      = _PERM_CARD_OFF             # offset of the card-id float within each permanent slot
# Precomputed indices for gathering the 48 self-permanent card-id floats from the state array
_BF_ID_IDX        = _BF_START + np.arange(_PERM_A_SLOTS) * _BF_SLOT_SIZE + _BF_CARD_OFF


def _gather_costs(matrix, ids):
    """Gather per-slot cost rows by vocab id; empty slots (id < 0) get a zero row.

    matrix : (N_CARD_TYPES, _N_COST_FEATS) cost table
    ids    : (S,) int vocab ids (decoded via round(id_float * N_CARD_TYPES))
    returns: (S, _N_COST_FEATS) float32
    """
    safe = np.clip(ids, 0, N_CARD_TYPES - 1)
    rows = matrix[safe]
    return np.where((ids >= 0)[:, None], rows, 0.0).astype(np.float32)
# Start of bf_ability_costs block in the full obs vector
# (5 per-action blocks: cats | ids | ctrl | zone_ref | slot_ref)
_BF_COST_START    = STATE_SIZE + 5 * MAX_ACTIONS + _HAND_COST_FEATS
# Vocab indices used for targeting decisions (mirror src/card_vocab.h)
_WASTELAND_VOCAB_IDX     = 10
_AETHER_VIAL_VOCAB_IDX   = 121
_BASIC_LAND_IDS          = frozenset({0, 19})  # Mountain(0), Island(19)
_COUNTER_SPELL_VOCAB_IDS = frozenset({12, 13, 22})  # Force of Will(12), Daze(13), Counterspell(22)
_COUNTERSPELL_VOCAB_IDX  = 22
# Death and Taxes: Solitude's ETB exiles a creature, so hard-cast it only with an
# opponent creature to hit. The One Ring's {T} draws a card per burden counter.
_SOLITUDE_VOCAB_IDX      = 141
_THE_ONE_RING_VOCAB_IDX  = 285
_BLUE_POOL_IDX           = _SELF_BLOCK_START + _PB_MANA + 1   # self U pool; mana is W/U/B/R/G/C, /10

# Doomsday deck card vocab indices (mirror src/card_vocab.h)
_DOOMSDAY_VOCAB_IDX      = 53
_THOUGHTSEIZE_VOCAB_IDX  = 54
_DARK_RITUAL_VOCAB_IDX   = 55
_LOTUS_PETAL_VOCAB_IDX   = 56
_LED_VOCAB_IDX           = 57
_THASSAS_ORACLE_VOCAB_IDX = 58
_PERSONAL_TUTOR_VOCAB_IDX = 59
_STREET_WRAITH_VOCAB_IDX = 60
_EDGE_OF_AUTUMN_VOCAB_IDX = 61
_CONSIDER_VOCAB_IDX      = 68
_DEEP_ANALYSIS_VOCAB_IDX = 70
_CAVERN_OF_SOULS_VOCAB_IDX = 67
_DURESS_VOCAB_IDX        = 69
_PONDER_VOCAB_IDX        = 11
_FORCE_OF_WILL_VOCAB_IDX = 12
_DAZE_VOCAB_IDX          = 13
_BRAINSTORM_VOCAB_IDX    = 24
_MISHRAS_BAUBLE_VOCAB_IDX = 27
_ONCE_UPON_A_TIME_VOCAB_IDX = 43

# Maverick (green/white) deck card vocab indices (mirror src/card_vocab.h)
_GREEN_SUNS_ZENITH_VOCAB_IDX = 35
_KEEN_EYED_CURATOR_VOCAB_IDX = 40
# Untapped creatures that tap for mana (count toward X for Green Sun's Zenith).
_MANA_DORK_IDS           = frozenset({30, 38, 42})  # Birds(30), Ignoble(38), Noble Hierarch(42)
# Lands that do NOT reliably produce mana when untapped, so they shouldn't count
# toward an affordable X: fetchlands (sacrifice for a land, no mana) and Gaea's
# Cradle (G per creature — zero with an empty board).
_UNRELIABLE_LAND_IDS     = frozenset({5, 6, 7, 8, 9, 52, 65, 66, 34})
# Cantrips/digging that let us keep a one-land opening hand (they find more lands)
_KEEP_ONE_LANDER_IDS     = frozenset({_PONDER_VOCAB_IDX, _BRAINSTORM_VOCAB_IDX,
                                      _ONCE_UPON_A_TIME_VOCAB_IDX})
_DISCARD_SPELL_IDS       = frozenset({_THOUGHTSEIZE_VOCAB_IDX, _DURESS_VOCAB_IDX})
_COUNTER_STRIP_IDS       = frozenset({_FORCE_OF_WILL_VOCAB_IDX, _DAZE_VOCAB_IDX})
# Self-controlled draw/cycling spells or abilities that, once on the stack, make it
# correct to crack Lion's Eye Diamond in response (float 3 mana that outlives the
# resolving draw, which then refills the discarded hand). Brainstorm is the canonical
# LED-response cantrip — crack LED, discard hand, then Brainstorm draws 3 fresh cards.
# Mishra's Bauble's sac-to-draw activation counts as the on-stack draw marker too.
_LED_DRAW_STACK_IDS      = frozenset({_STREET_WRAITH_VOCAB_IDX, _EDGE_OF_AUTUMN_VOCAB_IDX,
                                      _CONSIDER_VOCAB_IDX, _DEEP_ANALYSIS_VOCAB_IDX,
                                      _PONDER_VOCAB_IDX, _BRAINSTORM_VOCAB_IDX,
                                      _MISHRAS_BAUBLE_VOCAB_IDX})
_DOOMSDAY_DECK_IDS       = frozenset({53, 54, 55, 56, 57, 58, 59, 60, 61, 67, 68, 69, 70})

# Tron deck card vocab indices (mirror src/card_vocab.h). The three Urza lands
# each add extra colorless mana once the other two are also controlled (see
# Count$UrzaLands in src/components/ability.cpp); Planar Nexus is every
# nonbasic land type via a self-CDA (state_manager_statics.cpp), so a single
# Planar Nexus supplies whichever of the three types a real piece is missing.
_URZAS_MINE_VOCAB_IDX        = 274
_URZAS_POWER_PLANT_VOCAB_IDX = 275
_URZAS_TOWER_VOCAB_IDX       = 276
_PLANAR_NEXUS_VOCAB_IDX      = 266
_EXPEDITION_MAP_VOCAB_IDX    = 246
_TRON_LAND_IDS           = frozenset({_URZAS_MINE_VOCAB_IDX, _URZAS_POWER_PLANT_VOCAB_IDX,
                                      _URZAS_TOWER_VOCAB_IDX})
# Karn, the Great Creator wishes an artifact from the sideboard/exile into hand
# (-2). Mycosynth Lattice is the priority wish — with Karn in play it turns every
# opponent permanent into an artifact whose abilities Karn's static shuts off (the
# hard lock, incl. lands not tapping for mana). Cityscape Leveler is the follow-up
# beater once the lock (or a lack of Lattice) leaves nothing better to grab.
_KARN_GREAT_CREATOR_VOCAB_IDX = 280
_CITYSCAPE_LEVELER_VOCAB_IDX  = 287
_MYCOSYNTH_LATTICE_VOCAB_IDX  = 290
# Candelabra of Tawnos untaps X target lands; it's a mana engine only when it
# untaps lands that make MORE THAN ONE mana, so we bias its targets to those.
# Ancient Tomb ({C}{C}) and Urza's Workshop (metalcraft: {C} per Urza's land)
# always/often produce >1; the three Urza's tron lands produce {C}{C}{C} each
# once the set is assembled. Planar Nexus is deliberately excluded — it only
# ever makes a single mana of any type, so untapping it gains nothing.
_CANDELABRA_VOCAB_IDX    = 242
_MULTI_MANA_LAND_IDS     = frozenset({123, 261,  # Ancient Tomb, Urza's Workshop
                                      _URZAS_MINE_VOCAB_IDX, _URZAS_POWER_PLANT_VOCAB_IDX,
                                      _URZAS_TOWER_VOCAB_IDX})

_CAT_TOP_LIBRARY = CAT_TOP_LIBRARY  # choose card to put on top of library (Doomsday pile ordering)
_CAT_SHUFFLE     = CAT_SHUFFLE  # shuffle choice (0 = don't shuffle, 1 = shuffle)


def _hand_has_card(obs: np.ndarray, vocab_idx: int) -> bool:
    """Check if the priority player's hand contains a card with the given vocab index."""
    for slot in range(MAX_HAND_SLOTS):
        if _slot_card_idx(obs, _HAND_START + slot) == vocab_idx:
            return True
    return False


def _obs_action_category(obs: np.ndarray, action: int) -> int:
    """Extract the raw action category int for the given action index from a full obs vector."""
    return int(round(obs[STATE_SIZE + action] * ACTION_CATEGORY_MAX))


def _obs_action_card_id(obs: np.ndarray, action: int) -> int:
    """Extract the card vocab index for the given action index from a full obs vector."""
    return int(round(obs[STATE_SIZE + MAX_ACTIONS + action] * N_CARD_TYPES))


def _obs_is_main_phase(obs: np.ndarray) -> bool:
    """Check if the current step is a main phase (FIRST_MAIN or SECOND_MAIN)."""
    return obs[_STEP_FIRST_MAIN_IDX] > 0.5 or obs[_STEP_SECOND_MAIN_IDX] > 0.5


def _self_has_draw_on_stack(obs: np.ndarray) -> bool:
    """Return True if self controls a cycling or draw ability/spell on the stack."""
    for i in range(_STACK_SLOTS):
        base = _STACK_START + i * _STACK_SLOT_SIZE
        ctrl_is_self = obs[base]
        idx = _slot_card_idx(obs, base + 1)
        if idx >= 0 and ctrl_is_self > 0.5 and idx in _LED_DRAW_STACK_IDS:
            return True
    return False


def _stack_is_empty(obs: np.ndarray) -> bool:
    """Return True if there are no items on the stack."""
    for i in range(_STACK_SLOTS):
        base = _STACK_START + i * _STACK_SLOT_SIZE
        if _slot_card_idx(obs, base + 1) >= 0:
            return False
    return True


class ModelVsScriptedEnv(gym.Env):
    """Wraps RoboMageEnv so the model plays against a scripted agent.

    Each episode the model is randomly assigned to Player A or B; the scripted
    agent takes the other side.  Reward is negated when the model plays as B so
    it is always from the model's perspective (+1 win, -1 loss).
    """

    def __init__(self, model_deck: str | None = None, opp_deck: str | None = None,
                 opponent="scripted", **env_kwargs):
        super().__init__()
        # model_deck/opp_deck override deck_a/deck_b: decks are swapped each episode
        # to match which side the model is assigned to.
        self._model_deck = model_deck
        self._opp_deck = opp_deck
        self._bo3 = env_kwargs.get("bo3", False)
        self._env = RoboMageEnv(**env_kwargs)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = self._env.render_mode
        self._training_is_a = True
        self._opponent_below_10 = False
        self._last_obs = None
        # Scale applied to all shaping signals. Set externally (e.g. via SB3 callback)
        # to anneal shaping toward 0 as training matures.
        self.shaping_scale = 1.0
        # Opponent controller / pool. ``opponent`` may be an OpponentPool, an
        # already-built Controller, or a spec string ("scripted" by default,
        # which is byte-identical to the historical scripted agent).
        from opponents import make_controller
        # A pool exposes per-episode sampling (OpponentPool.sample /
        # LeaguePool.sample_episode); a Controller exposes .choose; anything else
        # is a spec string resolved into a Controller.
        if hasattr(opponent, "sample_episode") or hasattr(opponent, "sample"):
            self._opp_pool = opponent
            self._opp_controller = None
        elif hasattr(opponent, "choose"):
            self._opp_pool = None
            self._opp_controller = opponent
        else:
            self._opp_pool = None
            self._opp_controller = make_controller(opponent)
        self._opp_label = getattr(self._opp_controller, "label", "scripted")

    def reset(self, *, seed=None, options=None):
        self._training_is_a = bool(np.random.random() < 0.5)
        # Sample this episode's opponent (and, in league mode, its deck) BEFORE the
        # deck assignment below so the right opp_deck reaches the game process. A
        # plain OpponentPool keeps the fixed opp_deck and only picks a controller;
        # a LeaguePool also chooses which deck the opponent pilots this episode.
        episode_self_deck = None  # deck the LEARNER pilots this episode (league mixed mode)
        if self._opp_pool is not None:
            if hasattr(self._opp_pool, "sample_episode"):
                sample = self._opp_pool.sample_episode()
                # LeaguePool returns a 4-tuple (self_deck, opp_deck, label, ctrl):
                # in mixed mode self_deck varies per episode; in fixed mode it is
                # the pool's fixed learner deck. Accept a legacy 3-tuple too.
                if len(sample) == 4:
                    episode_self_deck, self._opp_deck, self._opp_label, self._opp_controller = sample
                else:
                    self._opp_deck, self._opp_label, self._opp_controller = sample
            else:
                self._opp_label, self._opp_controller = self._opp_pool.sample()
        # The learner's deck this episode: the pool's per-episode pick when it
        # supplies one (league mixed mode), else the fixed model_deck.
        self_deck = episode_self_deck if episode_self_deck is not None else self._model_deck
        if self_deck is not None:
            self._env._deck_a = self_deck if self._training_is_a else self._opp_deck
            self._env._deck_b = self._opp_deck if self._training_is_a else self_deck
        elif episode_self_deck is None and self._model_deck is None \
                and self._opp_pool is not None and hasattr(self._opp_pool, "sample_episode"):
            # A league pool must always supply a self deck; never run deckless.
            raise RuntimeError(
                "ModelVsScriptedEnv: league pool provided no self deck this episode "
                "and no fixed model_deck is set — refusing to run with default decks.")
        # Push this episode's per-seat deck names to a scripted opponent so its
        # doomsday/tron identification uses the same name rule as the shaping
        # opt-outs above (duck-typed: model/index controllers have no use for it).
        set_names = getattr(self._opp_controller, "set_deck_names", None)
        if set_names is not None:
            set_names(self._env._deck_a, self._env._deck_b)
        self._opponent_below_10 = False
        self._last_obs = None
        self._decision_idx = 0
        self._episode_shaping = 0.0
        # Deck-specific shaping keys off the deck the learner ACTUALLY pilots this
        # episode (self_deck), so the doomsday/tron opt-outs work in league mixed
        # mode where the self deck varies per episode (not just the fixed model_deck).
        self._episode_self_deck = self_deck
        self._is_doomsday = _deck_named(self_deck, "doomsday")
        # Potential-based shaping opt-outs, keyed on the deck name: doomsday
        # (combo gameplan) and tron (hoards its hand and plays few creatures by
        # design, so the potentials would penalize its actual game plan).
        self._skip_potentials = self._is_doomsday or _deck_named(self_deck, "tron")
        self._dd_placed_doomsday = False  # set when agent picks Doomsday in a TOP_LIBRARY choice
        self._dd_fired = set()  # tracks which DD shaping rewards have fired this game
        self._game_meta = {
            "model_is_a": self._training_is_a,
            "opp_deck": self._opp_deck or "unknown",
            "opp_type": self._opp_label,
            # Deck the learner piloted this episode (league win-rate-WITH tracking).
            "self_deck": self_deck or "unknown",
        }
        obs, info = self._env.reset(seed=seed, options=options)
        obs, _reward, terminated, truncated, info, _shaping = self._skip_opponent_turns(
            obs, 0.0, False, False, info
        )
        self._opponent_below_10 = False  # reset after mulligan/setup turns
        self._last_obs = obs.copy()
        return obs, info

    def step(self, action: int):
        self._decision_idx += 1

        # Doomsday deck shaping: inspect the action the model just chose
        # using the pre-step observation (self._last_obs has the action metadata)
        if self._is_doomsday and self._last_obs is not None:
            cat = _obs_action_category(self._last_obs, action)
            card = _obs_action_card_id(self._last_obs, action)
            self._dd_pending_shaping = 0.0
            # Reward casting Doomsday (once per game)
            if cat == _CAT_CAST and card == _DOOMSDAY_VOCAB_IDX and "cast_dd" not in self._dd_fired:
                self._dd_pending_shaping += SHAPING_DD_CAST_DOOMSDAY
                self._dd_fired.add("cast_dd")
            # Reward casting Dark Ritual when Doomsday is in hand and it's a main phase (once per game)
            if (cat == _CAT_CAST and card == _DARK_RITUAL_VOCAB_IDX
                    and _hand_has_card(self._last_obs, _DOOMSDAY_VOCAB_IDX)
                    and _obs_is_main_phase(self._last_obs)
                    and "ritual" not in self._dd_fired):
                self._dd_pending_shaping += SHAPING_DD_RITUAL_SETUP
                self._dd_fired.add("ritual")
            # Reward picking Thassa's Oracle during Doomsday pile building (once per game)
            if (cat == _CAT_TOP_LIBRARY and card == _THASSAS_ORACLE_VOCAB_IDX
                    and "pick_oracle" not in self._dd_fired):
                self._dd_pending_shaping += SHAPING_DD_PICK_ORACLE
                self._dd_fired.add("pick_oracle")
            # Reward selecting Doomsday with Personal Tutor (once per game)
            if cat == _CAT_TOP_LIBRARY and card == _DOOMSDAY_VOCAB_IDX:
                if "tutor_dd" not in self._dd_fired:
                    self._dd_pending_shaping += SHAPING_DD_TUTOR_DOOMSDAY
                    self._dd_fired.add("tutor_dd")
                self._dd_placed_doomsday = True
            # Reward not shuffling after placing Doomsday on top (once per game)
            if cat == _CAT_SHUFFLE and action == 0 and self._dd_placed_doomsday:
                if "keep_dd" not in self._dd_fired:
                    self._dd_pending_shaping += SHAPING_DD_KEEP_DOOMSDAY
                    self._dd_fired.add("keep_dd")
                self._dd_placed_doomsday = False
            elif cat == _CAT_SHUFFLE:
                self._dd_placed_doomsday = False
            # Reward casting Thoughtseize/Duress (once per game)
            if (cat == _CAT_CAST and card in _DISCARD_SPELL_IDS
                    and "cast_discard" not in self._dd_fired):
                self._dd_pending_shaping += SHAPING_DD_CAST_DISCARD
                self._dd_fired.add("cast_discard")
            # Reward selecting Force of Will or Daze with the discard choice (once per game)
            if (cat == _CAT_DISCARD and card in _COUNTER_STRIP_IDS
                    and "strip_counter" not in self._dd_fired):
                self._dd_pending_shaping += SHAPING_DD_STRIP_COUNTER
                self._dd_fired.add("strip_counter")
            # Reward cracking LED with a draw/cycling ability on the stack (once per game);
            # penalize cracking it with nothing on the stack (wastes the discard). LED cracks
            # are mana-ability activations, emitted with the per-color MANA_* categories
            # (13-18) — never ACTIVATE_ABILITY (6).
            if cat in _MANA_CATS and card == _LED_VOCAB_IDX:
                if _self_has_draw_on_stack(self._last_obs) and "led_draw" not in self._dd_fired:
                    self._dd_pending_shaping += SHAPING_DD_LED_WITH_DRAW
                    self._dd_fired.add("led_draw")
                elif _stack_is_empty(self._last_obs) and "led_empty" not in self._dd_fired:
                    self._dd_pending_shaping += SHAPING_DD_LED_EMPTY_STACK
                    self._dd_fired.add("led_empty")
        else:
            self._dd_pending_shaping = 0.0

        obs, reward, terminated, truncated, info = self._env.step(action)

        # Shaping: mana waste and mulligan penalties for the model player
        shaping_key = "shaping_a" if self._training_is_a else "shaping_b"
        shaping = info.get(shaping_key, 0.0)

        # Doomsday deck shaping (computed above from pre-step obs)
        shaping += self._dd_pending_shaping

        # Shaping: one-time bonus when the opponent's life first drops below 10.
        if not (terminated or truncated):
            shaping += self._below10_shaping(obs)

        if not (terminated or truncated):
            obs, reward, terminated, truncated, info, opp_shaping = self._skip_opponent_turns(
                obs, reward, terminated, truncated, info
            )
            shaping += opp_shaping

        # Potential-based hand/power advantage (skipped for doomsday/tron — see
        # _skip_potentials in reset()). Φ = 0 past the end of a game: on the
        # episode-ending step AND at each bo3 game boundary the accumulated
        # potential is settled (-Φ(s_last)) so the per-game sum telescopes to
        # zero — otherwise the episode nets Φ(s_last) - Φ(s_0) and the policy is
        # rewarded for ending games hoarding cards/board instead of spending
        # them to win. Between a bo3 game's end and the next game's first
        # decision the sideboard mask zeroes the hand and permanent blocks, so
        # intermediate deltas read 0 and the next game re-baselines cleanly
        # from its opening state.
        if not self._skip_potentials and self._last_obs is not None:
            game_over = terminated or truncated or info.get("game_result", False)
            phi_curr = 0.0 if game_over else _shaping_potential(obs)
            shaping += phi_curr - _shaping_potential(self._last_obs)

        shaping *= self.shaping_scale
        # In bo3, scale per-game shaping to 1/3 so total across 3 games
        # stays proportional to the match reward (1.0)
        ep_cap = SHAPING_EPISODE_CAP_DOOMSDAY if self._is_doomsday else SHAPING_EPISODE_CAP
        if self._bo3:
            shaping /= 3.0
            # Scale the per-game budget too, so the whole-match shaping total is
            # bounded by ep_cap (< the 1.3 minimum decisive match |reward|) —
            # otherwise 3 full per-game budgets could outweigh a match loss.
            ep_cap /= 3.0
        # Clamp to remaining episode budget
        remaining = ep_cap - self._episode_shaping
        floor = -(ep_cap + self._episode_shaping)
        shaping = max(floor, min(remaining, shaping))
        self._episode_shaping += shaping
        # Reset per-game shaping state at a bo3 game boundary: budget, DD flags,
        # and the opponent-below-10 once-flag (life totals reset with the game).
        if self._bo3 and info.get("game_result", False):
            self._episode_shaping = 0.0
            self._dd_fired.clear()
            self._dd_placed_doomsday = False
            self._opponent_below_10 = False
        self._last_obs = obs.copy() if not (terminated or truncated) else None

        if not self._training_is_a:
            reward = -reward
        if terminated or truncated:
            # Decisive outcome from the model's perspective, taken from the pure
            # game reward BEFORE shaping is added: +1 win, -1 loss, 0 undecided
            # (a step-cap truncation). Win-rate consumers (WinTallyCallback,
            # PFSPCallback, the snapshot promotion gate) classify by this flag,
            # not by the sign of the shaping-contaminated episode return.
            info["outcome"] = 1 if reward > 0 else (-1 if reward < 0 else 0)
        reward += shaping
        info["game_meta"] = self._game_meta
        info["decision_idx"] = self._decision_idx
        info["num_choices"] = self._env._num_choices
        if terminated or truncated:
            info["opp_deck"] = self._opp_deck or "unknown"
        return obs, reward, terminated, truncated, info

    def _below10_shaping(self, obs) -> float:
        """One-time bonus the first time the opponent's life drops below 10.

        obs is always from the priority player's perspective:
          model has priority → opponent life is the opp block's life slot
          opponent has priority → opponent ("self") life is the self block's life slot
        Skipped on the masked bo3 sideboard observation — the player blocks are
        zeroed there, so the life slot reads 0 and would fire spuriously. The
        once-flag resets at each bo3 game boundary (life totals reset with it).
        """
        if self._opponent_below_10 or obs[_MATCH_CTX_START + 3] > 0.5:
            return 0.0
        a_has_priority = obs[_SELF_IS_A_IDX] > 0.5
        model_has_priority = a_has_priority if self._training_is_a else not a_has_priority
        scripted_life = (obs[_OPP_BLOCK_START + _PB_LIFE] if model_has_priority
                         else obs[_SELF_BLOCK_START + _PB_LIFE]) * 20.0
        if scripted_life < 10.0:
            self._opponent_below_10 = True
            return SHAPING_OPPONENT_BELOW10
        return 0.0

    def _skip_opponent_turns(self, obs, reward, terminated, truncated, info):
        """Resolve consecutive opponent turns with the scripted agent.

        Returns the updated (obs, reward, terminated, truncated, info) tuple plus
        the shaping reward accumulated across all opponent steps. Two values are
        ACCUMULATED across the loop rather than taken from the last step, because
        a bo3 game boundary may fall on any step in it:
          * ``reward`` — game results (±0.3) earned on the incoming step or on an
            intermediate opponent step would otherwise be silently overwritten by
            the last step's (usually zero) reward and never reach the learner.
            All engine rewards are Player-A-perspective, so plain summation is
            correct; the caller negates the total for seat B.
          * ``game_result`` — OR-ed into the returned ``info`` so the caller's Φ
            settlement and per-game budget reset can't miss a boundary that
            arrived while the opponent held priority.
        """
        shaping_key = "shaping_a" if self._training_is_a else "shaping_b"
        shaping = 0.0
        game_result = info.get("game_result", False)
        while not (terminated or truncated) and (obs[_SELF_IS_A_IDX] > 0.5) != self._training_is_a:
            action = self._opp_controller.choose(obs, self._env._num_choices)
            obs, step_reward, terminated, truncated, info = self._env.step(action)
            reward += step_reward
            game_result = game_result or info.get("game_result", False)

            shaping += info.get(shaping_key, 0.0)

            if not (terminated or truncated):
                shaping += self._below10_shaping(obs)

        info["game_result"] = game_result
        return obs, reward, terminated, truncated, info, shaping

    def action_masks(self) -> np.ndarray:
        return self._env.action_masks()

    def update_opponent_weights(self, weights):
        """Push fresh PFSP/softmax weights into the league pool (if any).

        Called via ``vec_env.env_method`` from the PFSPCallback. ``env_method``
        resolves through the wrapper chain (get_wrapper_attr), so it reliably
        reaches this inner env — unlike ``set_attr``, which only sets the outer
        wrapper."""
        if self._opp_pool is not None and hasattr(self._opp_pool, "set_weights"):
            self._opp_pool.set_weights(weights)

    def set_shaping_scale(self, value):
        """Set the shaping-reward scale (read in ``step`` as ``self.shaping_scale``).

        MUST be called via ``vec_env.env_method('set_shaping_scale', v)``, not
        ``vec_env.set_attr('shaping_scale', v)``: under SubprocVecEnv/DummyVecEnv
        ``set_attr`` does a plain ``setattr`` on the OUTER Monitor wrapper, which
        gymnasium never forwards inward, so it would never reach this attribute.
        ``env_method`` resolves through the wrapper chain (get_wrapper_attr) to
        this inner env."""
        self.shaping_scale = float(value)

    def close(self):
        self._env.close()


# ── Self-play environment ─────────────────────────────────────────────────────

class SelfPlayEnv(gym.Env):
    """Self-play: the training model plays against a frozen saved checkpoint.

    Each episode one player role (A or B) is randomly assigned to the training
    model; the other is controlled by the frozen opponent.  The game engine emits
    observations from the priority player's perspective, so both sides always
    receive a perspective-normalised view without any mirroring.

    The training model always pilots ``model_deck`` and the frozen opponent always
    pilots ``opp_deck``.  Because there is one generalist that pilots any deck, the
    opponent's checkpoint is sampled from the shared generalist snapshots
    ``gen__v*.zip`` / ``gen__final.zip`` (for a mirror match this is simply this
    run's own past selves).  The checkpoint is resampled every ``RELOAD_EVERY``
    episodes so it tracks the improving policy.  If no compatible checkpoint
    exists yet, the opponent falls back to the scripted agent (with a warning).
    """

    RELOAD_EVERY = 100  # episodes between opponent checkpoint reloads
    _model_cache = {}  # class-level: {path: (mtime, model)}

    def __init__(self, checkpoint_dir: str,
                 model_deck: str | None = None, opp_deck: str | None = None,
                 **env_kwargs):
        super().__init__()
        self._model_deck = model_deck
        self._opp_deck = opp_deck
        self._env = RoboMageEnv(**env_kwargs)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = self._env.render_mode
        self._checkpoint_dir = checkpoint_dir
        self._opponent = None    # loaded model, or None → scripted fallback
        self._episode_count = 0
        self._training_is_a = True
        self._opponent_loaded = False  # defer loading to first reset()
        self._scripted_fallback_warned = False  # warn once when no checkpoint found
        self._opp_mask = np.zeros(MAX_ACTIONS, dtype=bool)  # reusable mask buffer
        # Controller used when no compatible checkpoint exists (default GREEDY).
        from opponents import ScriptedController
        from scripted_agent import make_agent
        self._fallback_controller = ScriptedController(make_agent("scripted"))

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        self._episode_count += 1
        if not self._opponent_loaded or self._episode_count % self.RELOAD_EVERY == 0:
            self._reload_opponent()
            self._opponent_loaded = True

        self._training_is_a = bool(np.random.random() < 0.5)
        if self._model_deck is not None:
            self._env._deck_a = self._model_deck if self._training_is_a else self._opp_deck
            self._env._deck_b = self._opp_deck if self._training_is_a else self._model_deck
        # Per-seat deck names for the scripted fallback's unified deck
        # identification (only consulted when it actually drives the opponent).
        self._fallback_controller.set_deck_names(self._env._deck_a, self._env._deck_b)
        self._decision_idx = 0
        opp_name = "scripted"
        if self._opponent is not None:
            opp_name = getattr(self, "_opp_checkpoint_path", "checkpoint")
        self._game_meta = {
            "model_is_a": self._training_is_a,
            "opp_deck": self._opp_deck or "unknown",
            "opp_type": opp_name,
            "self_deck": self._model_deck or "unknown",  # fixed self deck here
        }
        obs, info = self._env.reset(seed=seed, options=options)

        obs, reward, terminated, truncated, info = self._handle_opponent_turns(
            obs, 0.0, False, False, info
        )
        if terminated or truncated:
            return np.zeros(OBS_SIZE, dtype=np.float32), info
        return self._training_obs(obs), info

    def step(self, action: int):
        self._decision_idx += 1
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = self._handle_opponent_turns(
                obs, reward, terminated, truncated, info
            )
        # Reward is from Player A's perspective; negate if training model plays as B.
        if not self._training_is_a:
            reward = -reward
        if terminated or truncated:
            # Decisive outcome from the model's perspective (see ModelVsScriptedEnv):
            # +1 win, -1 loss, 0 undecided (a step-cap truncation).
            info["outcome"] = 1 if reward > 0 else (-1 if reward < 0 else 0)
        info["game_meta"] = self._game_meta
        info["decision_idx"] = self._decision_idx
        info["num_choices"] = self._env._num_choices
        return self._training_obs(obs), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._env.action_masks()

    def close(self):
        self._env.close()

    def set_shaping_scale(self, value):
        """Accept shaping-scale broadcasts uniformly (self-play applies no shaping)."""
        self.shaping_scale = float(value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _training_has_priority(self, obs: np.ndarray) -> bool:
        """True when it is the training model's turn to act (raw obs)."""
        a_has_priority = obs[_SELF_IS_A_IDX] > 0.5
        return a_has_priority if self._training_is_a else not a_has_priority

    def _training_obs(self, obs: np.ndarray) -> np.ndarray:
        """Return obs from the training model's perspective.

        Perspective normalization is handled by the game engine, so no mirroring
        is needed — the observation is already from the priority player's view.
        """
        return obs

    def _handle_opponent_turns(self, obs, reward, terminated, truncated, info):
        """Step with the frozen opponent until it is the training model's turn.

        ``reward`` is accumulated and ``game_result`` OR-ed across the loop (not
        taken from the last step): in bo3 a game result (±0.3) may arrive on the
        incoming step or on an intermediate opponent step, and would otherwise be
        silently dropped. Rewards are Player-A-perspective, so summation is
        correct; the caller negates the total for seat B.
        """
        game_result = info.get("game_result", False)
        while not (terminated or truncated) and not self._training_has_priority(obs):
            num_choices = self._env._num_choices
            # Opponent receives the state as emitted (already from its perspective)
            opp_obs = obs
            if self._opponent is not None:
                self._opp_mask[:] = False
                self._opp_mask[:num_choices] = True
                action, _ = self._opponent.predict(opp_obs, action_masks=self._opp_mask, deterministic=False)
                action = int(action)
            else:
                # No compatible checkpoint — pilot the opponent with the scripted agent.
                action = self._fallback_controller.choose(opp_obs, num_choices)
            obs, step_reward, terminated, truncated, info = self._env.step(action)
            reward += step_reward
            game_result = game_result or info.get("game_result", False)
        info["game_result"] = game_result
        return obs, reward, terminated, truncated, info

    def _reload_opponent(self):
        """Sample a frozen generalist snapshot to pilot the opponent's deck.

        There is ONE generalist model, so the frozen opponent is any of its
        snapshots (``gen__v*.zip`` / ``gen__final.zip``); the deck it pilots
        (``opp_deck``) is set by this env, not by the checkpoint's filename. If no
        generalist snapshot exists yet, fall back to the scripted agent and warn
        (once)."""
        from opponents import gen_snapshots
        deck = self._opp_deck or self._model_deck
        files = gen_snapshots(self._checkpoint_dir)
        if not files:
            if not self._scripted_fallback_warned:
                print(f"[self-play] WARNING: no generalist snapshot "
                      f"('gen__v*.zip' / 'gen__final.zip') in {self._checkpoint_dir}; "
                      f"opponent (deck {deck}) falling back to the scripted agent.")
                self._scripted_fallback_warned = True
            self._opponent = None
            return
        path = str(np.random.choice(files))
        self._opp_checkpoint_path = path
        try:
            mtime = os.path.getmtime(path)
            cached = SelfPlayEnv._model_cache.get(path)
            if cached is not None and cached[0] == mtime:
                self._opponent = cached[1]
                return
            try:
                from sb3_contrib import MaskablePPO as _PPO
            except ImportError:
                from stable_baselines3 import PPO as _PPO
            model = _PPO.load(path, device="cpu")
            SelfPlayEnv._model_cache[path] = (mtime, model)
            self._opponent = model
        except Exception:
            print(f"[self-play] WARNING: failed to load opponent checkpoint {path}; "
                  f"falling back to the scripted agent.")
            self._opponent = None  # fall back to scripted if load fails


class FixedModelEnv(gym.Env):
    """Train against a fixed opponent model that never changes.

    Like SelfPlayEnv but the opponent is loaded once from a specific path
    and is never reloaded or sampled from checkpoints.
    """

    def __init__(self, opp_model_path: str,
                 model_deck: str | None = None, opp_deck: str | None = None,
                 **env_kwargs):
        super().__init__()
        self._model_deck = model_deck
        self._opp_deck = opp_deck
        self._env = RoboMageEnv(**env_kwargs)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = self._env.render_mode
        self._training_is_a = True
        self._decision_idx = 0

        # Load fixed opponent once
        try:
            from sb3_contrib import MaskablePPO as _PPO
        except ImportError:
            from stable_baselines3 import PPO as _PPO
        self._opponent = _PPO.load(opp_model_path, device="cpu")
        self._opp_model_path = opp_model_path
        self._opp_mask = np.zeros(MAX_ACTIONS, dtype=bool)  # reusable mask buffer

    def reset(self, *, seed=None, options=None):
        self._training_is_a = bool(np.random.random() < 0.5)
        if self._model_deck is not None:
            self._env._deck_a = self._model_deck if self._training_is_a else self._opp_deck
            self._env._deck_b = self._opp_deck if self._training_is_a else self._model_deck
        self._decision_idx = 0
        self._game_meta = {
            "model_is_a": self._training_is_a,
            "opp_deck": self._opp_deck or "unknown",
            "opp_type": self._opp_model_path,
            "self_deck": self._model_deck or "unknown",  # fixed self deck here
        }
        obs, info = self._env.reset(seed=seed, options=options)

        obs, reward, terminated, truncated, info = self._handle_opponent_turns(
            obs, 0.0, False, False, info
        )
        if terminated or truncated:
            return np.zeros(OBS_SIZE, dtype=np.float32), info
        return obs, info

    def step(self, action: int):
        self._decision_idx += 1
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = self._handle_opponent_turns(
                obs, reward, terminated, truncated, info
            )
        if not self._training_is_a:
            reward = -reward
        if terminated or truncated:
            # Decisive outcome from the model's perspective (see ModelVsScriptedEnv):
            # +1 win, -1 loss, 0 undecided (a step-cap truncation).
            info["outcome"] = 1 if reward > 0 else (-1 if reward < 0 else 0)
        info["game_meta"] = self._game_meta
        info["decision_idx"] = self._decision_idx
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._env.action_masks()

    def close(self):
        self._env.close()

    def set_shaping_scale(self, value):
        """Accept shaping-scale broadcasts uniformly (fixed-model applies no shaping)."""
        self.shaping_scale = float(value)

    def _training_has_priority(self, obs: np.ndarray) -> bool:
        a_has_priority = obs[_SELF_IS_A_IDX] > 0.5
        return a_has_priority if self._training_is_a else not a_has_priority

    def _handle_opponent_turns(self, obs, reward, terminated, truncated, info):
        # Same accumulation contract as SelfPlayEnv._handle_opponent_turns: bo3
        # game rewards may land on any step in the loop, so sum them (A's
        # perspective) and OR game_result instead of keeping only the last step's.
        game_result = info.get("game_result", False)
        while not (terminated or truncated) and not self._training_has_priority(obs):
            num_choices = self._env._num_choices
            self._opp_mask[:] = False
            self._opp_mask[:num_choices] = True
            action, _ = self._opponent.predict(obs, action_masks=self._opp_mask, deterministic=False)
            action = int(action)
            obs, step_reward, terminated, truncated, info = self._env.step(action)
            reward += step_reward
            game_result = game_result or info.get("game_result", False)
        info["game_result"] = game_result
        return obs, reward, terminated, truncated, info


# ── Back-compat re-export ───────────────────────────────────────────────────
# The scripted agent now lives in scripted_agent.py.  Importers historically do
# `from env import scripted_action`, so re-export it here.  This import runs at
# the bottom of env.py (after every constant/helper scripted_agent needs is
# defined), so when env is imported first the chain resolves cleanly.  If
# scripted_agent is imported first, scripted_action is not yet bound when this
# line runs, so fall back to a lazy wrapper that resolves it on first call.
try:
    from scripted_agent import scripted_action  # noqa: E402,F401
except ImportError:
    def scripted_action(obs, num_choices):  # noqa: E402
        from scripted_agent import scripted_action as _sa
        return _sa(obs, num_choices)
