"""
Per-entity feature extractor for RoboMage.

Splits the flat observation into sections that are each encoded by a shared-weight
MLP, then aggregated via mean+max pooling.  The policy head receives a fixed-size
representation that is invariant to card ordering and slot position.

State is always from the PRIORITY PLAYER'S perspective ("self").

NOTE: Both exile zones are serialized right after the graveyard blocks and
      consumed through the shared entity encoder like the graveyards.
NOTE: ActionChoice.description is never part of the observation — it is for
      human-readable display only (CLI) and is not passed to the ML model.

Card identity is a single normalized id float per slot (idx/N_CARD_TYPES, or
-1/N_CARD_TYPES for empty/unknown), decoded with round(val*N_CARD_TYPES) and
looked up in a learned nn.Embedding. This decouples the observation size from the
vocab size — growing N_CARD_TYPES costs one embedding row, not 252 one-hot slots.

Index layout must stay in sync with src/machine_io.h (STATE_SIZE = 6329):
  obs[0:36]            global context (player stats, step, flags, stack size)
  obs[36:3684]         96 permanent slots × 38 floats
                         slots 0-47: self; slots 48-95: opponent
                         0-10  status: power, toughness, tapped, attacking, blocking,
                               sickness, damage, controller_is_self, is_creature,
                               is_land, loyalty
                         11    p1p1_net (signed, /10)
                         12    other_counters (/10)
                         13-16 entity refs (normalized (idx+1)/108): attached_to,
                               attached_by, attack_target, blocking_target
                         17    is_blocked
                         18    is_phased_out
                         19-34 keyword multi-hot (N_OBS_KEYWORDS = 16)
                         35    chosen-name id (Pithing Needle / Disruptor Flute named
                               card, Petrified Hamlet named land; sentinel = none)
                         36    returnable-exile id (card this permanent exiled that still
                               has a return path — Static Prison / Phelia; sentinel = none)
                         37    card id (LAST)
  obs[3684:4128]       12 stack slots × 37 floats (controller_is_self + card id +
                         is_spell + x_or_amount/10 + 7 cast qualifiers +
                         chosen-mode multi-hot(6) + 4 announced-target sub-slots ×
                         [present, is_player, controller_is_self, slot_ref, card id])
  obs[4128:4256]      128 graveyard slots × 1 float (card id, recency-ordered)
                         slots 0-63: self; slots 64-127: opponent
  obs[4256:4384]      128 exile slots × 1 float (card id, recency-ordered)
                         slots 0-63: self; slots 64-127: opponent
  obs[4384:4394]       10 hand slots    × 1 float  (card id)
  obs[4394:4906]      128 action history entries × 4 floats (newest first)
                         per entry: category_norm, card_id_norm, is_self, turn/50
  obs[4906:4910]       match context (4 floats: game_number, self_wins, opp_wins, sideboard_phase)
  obs[4910:4913]       library counts & post-board (self_lib/60, opp_lib/60, is_post_board)
  obs[4913]            current game turn / 50
  obs[4914:4919]       5 known top-of-library slots × 1 float (card id, sentinel = unknown)
  obs[4919:5943]       opponent revealed-cards multi-hot (N_CARD_TYPES floats, accumulated across the match)
  obs[5943:5953]       10 known opponent-hand slots × 1 float (card id)
  obs[5953:5955]       pending-decision context (source card id + ctrl_is_self)
  obs[5955:5977]       global extras (self/opp lands played, viewer_has_priority,
                         self/opp monarch, city's blessing, revolt, pending extra
                         turns, is_day, is_night, MandatoryChoice one-hot(6),
                         self_plays_first, sideboard swaps made, sideboard delta)
  obs[5977:6073]       48 self live-library slots × (card id, count)
  obs[6073:6201]       the viewer's own live 75: 48 maindeck + 16 sideboard slots
                         × (card id, count). 16, not 15: mid-swap a cut card is
                         momentarily the sideboard's 16th (DECKLIST_SIDE_SLOTS).
  obs[6201:6329]       the opponent's REGISTERED 75 (frozen at match start):
                         48 maindeck + 16 sideboard slots × (card id, count)
  obs[6329:]           action metadata (cats|ids|ctrl|zone|refs|ords) + cost
                         features (appended by env.py; refs are normalized
                         entity-slot references, (idx+1)/108 with 0.0 = none)
"""

from functools import partial

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

try:
    from card_costs import N_CARD_TYPES
except ImportError:
    from train.card_costs import N_CARD_TYPES

# ACTION_CATEGORY_MAX (mirrors src/classes/action.h via codegen) is needed to
# decode the per-action category-norm floats back to integer category ids for the
# per-action logit head. Same source of truth env.py uses for the action block.
try:
    from _enums import (ACTION_CATEGORY_MAX, N_OBS_KEYWORDS, N_MANDATORY_CHOICES,
                        DECKLIST_MAIN_SLOTS, DECKLIST_SIDE_SLOTS,
                        MAX_BATTLEFIELD_SLOTS, MAX_STACK_DISPLAY, MAX_STACK_MODES,
                        MAX_STACK_TGTS, MAX_GY_SLOTS, MAX_HAND_SLOTS,
                        KNOWN_TOP_LIBRARY_SIZE, ACTION_HISTORY_SIZE,
                        CARD_ID_SLOT_SIZE, STACK_HEAD_FIELDS, STACK_XAMT_FIELDS,
                        STACK_QUAL_FIELDS, STACK_TGT_FIELDS, HIST_ENTRY_SIZE,
                        PENDING_DECISION_SIZE, EXTRAS_SCALARS,
                        EXTRAS_SB_CTX_SIZE, DECKLIST_SLOT_SIZE)
except ImportError:
    from train._enums import (ACTION_CATEGORY_MAX, N_OBS_KEYWORDS, N_MANDATORY_CHOICES,
                             DECKLIST_MAIN_SLOTS, DECKLIST_SIDE_SLOTS,
                             MAX_BATTLEFIELD_SLOTS, MAX_STACK_DISPLAY, MAX_STACK_MODES,
                             MAX_STACK_TGTS, MAX_GY_SLOTS, MAX_HAND_SLOTS,
                             KNOWN_TOP_LIBRARY_SIZE, ACTION_HISTORY_SIZE,
                             CARD_ID_SLOT_SIZE, STACK_HEAD_FIELDS, STACK_XAMT_FIELDS,
                             STACK_QUAL_FIELDS, STACK_TGT_FIELDS, HIST_ENTRY_SIZE,
                             PENDING_DECISION_SIZE, EXTRAS_SCALARS,
                             EXTRAS_SB_CTX_SIZE, DECKLIST_SLOT_SIZE)


def _masked_mean_max(emb: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """Aggregate per-slot embeddings, ignoring empty slots.

    Empty card slots are all-zero one-hots, but the encoder maps them to a
    nonzero bias vector; pooling over them with a plain mean dilutes the real
    signal and a plain max can be pinned by that bias. This pools only the
    occupied slots:
      - masked mean: sum over present slots / count  (count-robust composition)
      - masked max:  per-feature max over present slots (single-card presence)
    Rows with no occupied slots aggregate to zeros for both halves.

    emb     : (B, S, D) per-slot embeddings
    present : (B, S) bool/float, 1.0 where the slot holds a real card
    returns : (B, 2*D) = [masked_mean, masked_max]
    """
    m = present.unsqueeze(-1).to(emb.dtype)          # (B, S, 1)
    counts = m.sum(1).clamp(min=1.0)                 # (B, 1); empty zone -> 1 (mean stays 0)
    masked_mean = (emb * m).sum(1) / counts          # (B, D)

    neg_inf = torch.finfo(emb.dtype).min
    masked_max = emb.masked_fill(m == 0, neg_inf).max(1).values  # (B, D)
    has_any = present.any(1, keepdim=True)           # (B, 1)
    masked_max = torch.where(has_any, masked_max, torch.zeros_like(masked_max))

    return torch.cat([masked_mean, masked_max], dim=-1)

# ── Layout constants (mirror src/machine_io.h) ──────────────────────────────
# Card identity is a single normalized id float per slot; decode via
# round(val * N_CARD_TYPES) and look up in self.card_emb.
# (_GLOBAL_SIZE is derived from env._GLOBAL_SIZE just below, after that import.)

# Every width below comes from _enums (generated from src/machine_io.h +
# src/classes/gamestate.h), never a bare literal — this module keeps its OWN
# offset chain, so a hand-copied width here silently misreads the observation.
# The chain is cross-checked block-by-block against env.py's at the bottom.
_PERM_SLOTS      = 2 * MAX_BATTLEFIELD_SLOTS  # 48 self + 48 opponent (unified: creatures, lands, other)
# 11 status (incl. loyalty) + 2 counters + 4 entity refs + is_blocked +
# is_phased_out + keyword multi-hot + chosen-name id + returnable-exile id +
# card id (LAST) = 38
_PERM_SLOT_SIZE  = 19 + N_OBS_KEYWORDS + 3  # 38
_PERM_STATUS_FLOATS   = _PERM_SLOT_SIZE - 3  # 35 non-id status/ref/keyword floats
_PERM_CHOSEN_NAME_OFF = _PERM_SLOT_SIZE - 3  # 35 (chosen-name id, 3rd-last)
_PERM_RETURNABLE_OFF  = _PERM_SLOT_SIZE - 2  # 36 (returnable-exile id, 2nd-last)
_PERM_CARD_OFF   = _PERM_SLOT_SIZE - 1      # 37 (card id is always LAST)

_STACK_SLOTS      = MAX_STACK_DISPLAY
_STACK_XAMT_OFF   = STACK_HEAD_FIELDS   # x_or_amount / 10 within a stack slot
_STACK_QUALS      = STACK_QUAL_FIELDS   # cast qualifiers (is_copy, kicked, flashback, evoke, ...)
_STACK_MODE_SLOTS = MAX_STACK_MODES     # chosen-mode multi-hot width per stack slot
_STACK_MODE_OFF   = _STACK_XAMT_OFF + STACK_XAMT_FIELDS + _STACK_QUALS  # 11
_STACK_TGT_SLOTS  = MAX_STACK_TGTS      # announced-target sub-slots per stack slot
_STACK_TGT_FIELDS = STACK_TGT_FIELDS    # present + is_player + ctrl_is_self + slot_ref + card id (LAST)
_STACK_TGT_OFF    = _STACK_MODE_OFF + _STACK_MODE_SLOTS     # 17
# ctrl(1) + card id(1) + is_spell(1) + x_or_amount + qualifiers + modes +
# target sub-slots (37 total)
_STACK_SLOT_SIZE  = _STACK_TGT_OFF + _STACK_TGT_SLOTS * _STACK_TGT_FIELDS

_GY_SLOTS        = 2 * MAX_GY_SLOTS  # 64 self + 64 opponent
_GY_SLOT_SIZE    = CARD_ID_SLOT_SIZE # card id only

_EXILE_SLOTS     = 2 * MAX_GY_SLOTS  # 64 self + 64 opponent (same layout as graveyard)
_EXILE_SLOT_SIZE = CARD_ID_SLOT_SIZE # card id only

_HAND_SLOTS      = MAX_HAND_SLOTS
_HAND_SLOT_SIZE  = CARD_ID_SLOT_SIZE # card id only

_HIST_ENTRIES    = ACTION_HISTORY_SIZE  # action history entries (newest first)
_HIST_ENTRY_SIZE = HIST_ENTRY_SIZE      # category_norm, card_id_norm, is_self, turn/50

_KNOWN_TOP_LIB_SLOTS     = KNOWN_TOP_LIBRARY_SIZE  # known top-of-library cards
_KNOWN_TOP_LIB_SLOT_SIZE = CARD_ID_SLOT_SIZE  # card id per slot
_REVEALED_SIZE           = N_CARD_TYPES  # opponent revealed-cards multi-hot (dense, not one-hot-per-slot)
_OPP_KNOWN_HAND_SLOTS    = MAX_HAND_SLOTS  # known opponent-hand card identities
_OPP_KNOWN_HAND_SLOT_SIZE = CARD_ID_SLOT_SIZE  # card id per slot

# Deck-identity tail blocks (mirror machine_io.h [5977-6328] / env.py): five
# (card_id, count) slot blocks — the viewer's live LIBRARY tally, the viewer's own
# LIVE maindeck + sideboard, and the opponent's REGISTERED maindeck + sideboard.
# card id first (norm_card_id, -1 empty sentinel), count second (/4.0). All five
# share one decklist_encoder.
_DECKLIST_MAIN_SLOTS = DECKLIST_MAIN_SLOTS   # 48 (self live lib, self/opp main)
_DECKLIST_SIDE_SLOTS = DECKLIST_SIDE_SLOTS   # 16 (self/opp side)
_DECKLIST_SLOT_SIZE  = DECKLIST_SLOT_SIZE     # card id + count per slot
_DECKLIST_CARD_OFF   = 0                      # card id first within a slot
_DECKLIST_COUNT_OFF  = 1                      # count second

_CARD_EMBED_DIM  = 32   # dimension of the learned card-identity embedding

# ── Per-action logit head (opt-in) ──────────────────────────────────────────
# The action block env.py appends after the state vector is, per action slot:
#   cats[MAX_ACTIONS] (category/ACTION_CATEGORY_MAX) | ids[MAX_ACTIONS] (norm card
#   id of the action's referenced entity — e.g. a target's card) | ctrl[MAX_ACTIONS]
#   (controller_is_self) | zone[MAX_ACTIONS] (ActionRefZone/REF_ZONE_MAX — which
#   zone/side the entity lives in) | refs[MAX_ACTIONS] (normalized entity-slot ref
#   (idx+1)/N_ENTITY_REF_SLOTS: 0-47 self perm, 48-95 opp perm, 96-107 stack,
#   0.0 = none). When per_action_head=True the extractor encodes each slot
#   (category embed + target-card embed + ctrl + zone embed + the GATHERED
#   board/stack embedding of the referenced entity) into a per-action feature so
#   the policy can score "target THIS specific permanent" against that action's
#   OWN features instead of a flat positional Linear. See PerActionMaskablePolicy.
# MAX_ACTIONS and STATE_SIZE come from env.py (single source of truth for the
# action-block layout the engine emits).
try:
    # The module itself too, so the block-by-block chain comparison at the bottom
    # can getattr() env's offsets by name instead of importing 18 more aliases.
    import env as _env_mod
    from env import (MAX_ACTIONS as _MAX_ACTIONS, STATE_SIZE as _ENV_STATE_SIZE,
                     _GLOBAL_SIZE as _ENV_GLOBAL_SIZE, N_ENTITY_REF_SLOTS,
                     OBS_SIZE as _ENV_OBS_SIZE, BUCKET_IDX as _BUCKET_IDX,
                     ARCH_ONEHOT_START as _ARCH_ONEHOT_START,
                     ARCH_ONEHOT_END as _ARCH_ONEHOT_END)
except ImportError:
    import train.env as _env_mod
    from train.env import (MAX_ACTIONS as _MAX_ACTIONS, STATE_SIZE as _ENV_STATE_SIZE,
                           _GLOBAL_SIZE as _ENV_GLOBAL_SIZE, N_ENTITY_REF_SLOTS,
                           OBS_SIZE as _ENV_OBS_SIZE, BUCKET_IDX as _BUCKET_IDX,
                           ARCH_ONEHOT_START as _ARCH_ONEHOT_START,
                           ARCH_ONEHOT_END as _ARCH_ONEHOT_END)
try:
    from archetypes import N_VALUE_BUCKETS
except ImportError:
    from train.archetypes import N_VALUE_BUCKETS
# Width of the matchup tail's trunk-visible part (the two archetype one-hots) and
# of the whole tail. Both derived from env.py's offsets — never re-typed here.
_ARCH_ONEHOT_FEATS  = _ARCH_ONEHOT_END - _ARCH_ONEHOT_START
_MATCHUP_TAIL_FEATS = _ARCH_ONEHOT_END - _BUCKET_IDX
_GLOBAL_SIZE = _ENV_GLOBAL_SIZE   # header width (single source of truth: env.py)
try:
    from _enums import REF_ZONE_MAX, N_REF_ZONES
except ImportError:
    from train._enums import REF_ZONE_MAX, N_REF_ZONES
_ACTION_CAT_EMBED  = 8    # learned embedding dim for the action category
_REF_ZONE_EMBED    = 4    # learned embedding dim for the per-action zone_ref
_PER_ACTION_DIM    = 32   # per-action feature width fed to the action scorer
_HIST_RECENT_K     = 16   # most-recent history entries embedded per-entry

# Offset chain is fully derived from the block-size constants above and pinned by
# the `assert _STATE_END == _ENV_STATE_SIZE` below, so absolute indices are NOT
# annotated here (they went stale when the layout last changed). Cross-reference
# the byte ranges in machine_io.h's layout block, not literals in this file.
_PERM_START  = _GLOBAL_SIZE
_PERM_END    = _PERM_START + _PERM_SLOTS * _PERM_SLOT_SIZE
_STACK_START = _PERM_END
_STACK_END   = _STACK_START + _STACK_SLOTS * _STACK_SLOT_SIZE
_GY_START    = _STACK_END
_GY_END      = _GY_START + _GY_SLOTS * _GY_SLOT_SIZE
_EXILE_START = _GY_END
_EXILE_END   = _EXILE_START + _EXILE_SLOTS * _EXILE_SLOT_SIZE
_HAND_START  = _EXILE_END
_HAND_END    = _HAND_START + _HAND_SLOTS * _HAND_SLOT_SIZE
_HIST_START  = _HAND_END
_HIST_END    = _HIST_START + _HIST_ENTRIES * _HIST_ENTRY_SIZE
# Then: match context (4 floats: game_number, self_wins, opp_wins, sideboard_phase),
# library counts & post-board (self_lib/60, opp_lib/60, is_post_board).
_MATCH_CTX_START      = _HIST_END
_MATCH_CTX_END        = _MATCH_CTX_START + 4                   # library ctx start
_LIBRARY_CTX_END      = _MATCH_CTX_END + 3                     # current turn idx
_CUR_TURN_IDX         = _LIBRARY_CTX_END
_KNOWN_TOP_LIB_START  = _CUR_TURN_IDX + 1
_KNOWN_TOP_LIB_END    = _KNOWN_TOP_LIB_START + _KNOWN_TOP_LIB_SLOTS * _KNOWN_TOP_LIB_SLOT_SIZE
_REVEALED_START       = _KNOWN_TOP_LIB_END
_REVEALED_END         = _REVEALED_START + _REVEALED_SIZE
_OPP_KNOWN_HAND_START = _REVEALED_END
_OPP_KNOWN_HAND_END   = _OPP_KNOWN_HAND_START + _OPP_KNOWN_HAND_SLOTS * _OPP_KNOWN_HAND_SLOT_SIZE
# Pending decision context: card id of the spell/ability currently making a
# mid-resolution choice (sentinel = none) + its controller-is-viewer flag.
_PENDING_START        = _OPP_KNOWN_HAND_END
_PENDING_SIZE         = PENDING_DECISION_SIZE
_PENDING_END          = _PENDING_START + _PENDING_SIZE
# Global extras: self/opp lands played, priority, monarch, city's blessing,
# revolt, pending extra turns, day/night flags, the MandatoryChoice one-hot, then
# self_plays_first and the two sideboard-progress scalars (swaps made, maindeck
# drift). Cheap scalar facts — passed through raw.
_EXTRAS_START         = _PENDING_END
_EXTRAS_SIZE          = EXTRAS_SCALARS + N_MANDATORY_CHOICES + EXTRAS_SB_CTX_SIZE  # 22
_EXTRAS_END           = _EXTRAS_START + _EXTRAS_SIZE
# Deck-identity tail blocks: self live library, the viewer's own live 75, then the
# opponent's registered main + side.
_SELF_LIVE_LIB_START  = _EXTRAS_END
_SELF_LIVE_LIB_END    = _SELF_LIVE_LIB_START + _DECKLIST_MAIN_SLOTS * _DECKLIST_SLOT_SIZE
_SELF_DECK_MAIN_START = _SELF_LIVE_LIB_END
_SELF_DECK_MAIN_END   = _SELF_DECK_MAIN_START + _DECKLIST_MAIN_SLOTS * _DECKLIST_SLOT_SIZE
_SELF_DECK_SIDE_START = _SELF_DECK_MAIN_END
_SELF_DECK_SIDE_END   = _SELF_DECK_SIDE_START + _DECKLIST_SIDE_SLOTS * _DECKLIST_SLOT_SIZE
_OPP_DECK_MAIN_START  = _SELF_DECK_SIDE_END
_OPP_DECK_MAIN_END    = _OPP_DECK_MAIN_START + _DECKLIST_MAIN_SLOTS * _DECKLIST_SLOT_SIZE
_OPP_DECK_SIDE_START  = _OPP_DECK_MAIN_END
_OPP_DECK_SIDE_END    = _OPP_DECK_SIDE_START + _DECKLIST_SIDE_SLOTS * _DECKLIST_SLOT_SIZE
_STATE_END            = _OPP_DECK_SIDE_END
# obs[_STATE_END:] = action metadata + cost features + matchup tail (env.py)
# Guard against the two layout mirrors drifting apart (env.py owns STATE_SIZE and
# the obs tail offsets; these asserts are the mirror check).
assert _STATE_END == _ENV_STATE_SIZE, (_STATE_END, _ENV_STATE_SIZE)
assert _ARCH_ONEHOT_END == _ENV_OBS_SIZE, (_ARCH_ONEHOT_END, _ENV_OBS_SIZE)
assert _BUCKET_IDX > _STATE_END, (_BUCKET_IDX, _STATE_END)

# ...and compare the two chains BLOCK BY BLOCK, not just on their total. A
# total-only assert passes a compensating change (a float moved from one block
# into the next) while every field between the two blocks reads from the wrong
# offset — a silent, training-corrupting failure rather than a loud one. env.py
# is in turn checked against src/machine_io.h's OFFSET_CHAIN by ci_check.py's
# `actorobs` tier, so all three reconstructions are transitively pinned.
_ENV_CHAIN_PAIRS = [
    ("_GLOBAL_SIZE",          _GLOBAL_SIZE),
    ("_STACK_START",          _STACK_START),
    ("_GY_START",             _GY_START),
    ("_EXILE_START",          _EXILE_START),
    ("_HAND_START",           _HAND_START),
    ("_HIST_START",           _HIST_START),
    ("_MATCH_CTX_START",      _MATCH_CTX_START),
    ("_CUR_TURN_IDX",         _CUR_TURN_IDX),
    ("_KNOWN_TOP_LIB_START",  _KNOWN_TOP_LIB_START),
    ("_REVEALED_START",       _REVEALED_START),
    ("_OPP_KNOWN_HAND_START", _OPP_KNOWN_HAND_START),
    ("_EXTRAS_START",         _EXTRAS_START),
    ("_EXTRAS_END",           _EXTRAS_END),
    ("_SELF_LIVE_LIB_START",  _SELF_LIVE_LIB_START),
    ("_SELF_DECK_MAIN_START", _SELF_DECK_MAIN_START),
    ("_SELF_DECK_SIDE_START", _SELF_DECK_SIDE_START),
    ("_OPP_DECK_MAIN_START",  _OPP_DECK_MAIN_START),
    ("_OPP_DECK_SIDE_START",  _OPP_DECK_SIDE_START),
]
for _name, _mine in _ENV_CHAIN_PAIRS:
    _theirs = getattr(_env_mod, _name)
    assert _mine == _theirs, (
        f"extractor.py and env.py disagree on {_name}: {_mine} vs {_theirs} — "
        "the two state-vector offset chains have drifted")
del _name, _mine, _theirs


class CardGameExtractor(BaseFeaturesExtractor):
    """
    Shared-weight per-entity encoder with masked mean+max aggregation.

    Card identity is a single normalized id float per slot; it is decoded to a
    vocab index and looked up in a shared nn.Embedding (padding_idx 0 = empty),
    then concatenated with that slot's status floats before encoding. Decoupling
    card identity from a one-hot makes the observation cost independent of vocab
    size.

    Three independent encoders cover the slot formats:
      perm_encoder   (35 status/counter/ref/keyword floats + chosen-name embed +
                     returnable-exile embed + card_embed → embed_dim): permanents
                     (entity refs enter as raw normalized floats in v1)
      stack_encoder  (32 scalars + card_embed + target-embed mean → embed_dim//2):
                     stack items with x/qualifiers and announced modes/targets
      entity_encoder (card_embed → embed_dim): graveyard, exile, hand, known top-library

    Empty slots (id sentinel) are masked out of the perm / stack / graveyard /
    exile / hand pooling so they neither dilute the mean nor pin the max.

    Output fed into the policy MLP head:
      global(36) + hist(512 raw) + hist_recent(K × (cat_emb+card_emb+2) embedded) +
      meta_ctx(8) + known_top_lib_agg(embed) + revealed_agg(embed) +
      pending_feat(card_emb+1: what's asking for the current choice) +
      extras(19 raw: lands played, priority, monarch, ..., MandatoryChoice one-hot) +
      action_extras(action metadata cats|ids|ctrl|zone|refs + cost feats) +
      arch_onehot(2*N_ARCH raw: one-hot self archetype | one-hot opp archetype;
                  the tail's raw bucket index is stripped and stashed as
                  ``self.last_bucket`` for the multi-head critic) +
      perm_agg(embed*2: masked mean+max) + stack_agg(embed//2 * 2) +
      graveyard_agg(embed*2: masked mean+max) + exile_agg(embed*2: masked mean+max) +
      hand_agg(embed*2: masked mean+max) + opp_known_hand_agg(embed*2: masked mean+max) +
      self_live_lib_agg + opp_deck_main_agg + opp_deck_side_agg
        (each embed*2: masked mean+max over a (card_id, count) deck-identity block)
    """

    def __init__(
        self,
        observation_space: gym.Space,
        embed_dim: int = 64,
        card_embed_dim: int = _CARD_EMBED_DIM,
        per_action_head: bool = False,
    ):
        half = embed_dim // 2
        _hist_size = _HIST_ENTRIES * _HIST_ENTRY_SIZE     # 512
        _meta_ctx_size = _KNOWN_TOP_LIB_START - _HIST_END  # 8 (match+lib+turn)
        # Embedded recent-history block: the K most recent action-history entries,
        # each flattened as [cat_emb | card_emb | is_self | turn]. Positional
        # (recency-ordered) on purpose.
        _hist_recent_size = _HIST_RECENT_K * (_ACTION_CAT_EMBED + card_embed_dim + 2)
        base_features_dim = (
            _GLOBAL_SIZE                                 # 36
            + _hist_size                                 # 512 action history (raw)
            + _hist_recent_size                          # embedded recent-K history
            + _meta_ctx_size                             # 8 match + lib + turn
            + embed_dim                                  # known-top library mean
            + embed_dim                                  # opponent revealed-cards multi-hot
            + card_embed_dim + 1                         # pending-decision source embed + ctrl flag
            + _EXTRAS_SIZE                               # 22 global extras (raw passthrough)
            # action extras (6 blocks incl. refs + ords + costs): everything after
            # the state EXCEPT the matchup tail, which is handled separately.
            + (observation_space.shape[0] - _STATE_END - _MATCHUP_TAIL_FEATS)
            + _ARCH_ONEHOT_FEATS                         # self/opp archetype one-hots
                                                         # (the raw bucket float is
                                                         # stripped, see forward)
            + embed_dim * 2                              # perm masked mean+max (creatures, lands, other)
            + half * 2                                   # stack mean+max
            + embed_dim * 2                              # graveyard masked-mean + max
            + embed_dim * 2                              # exile masked-mean + max
            + embed_dim * 2                              # hand masked-mean + max
            + embed_dim * 2                              # known opponent-hand masked-mean + max
            + embed_dim * 2                              # self live-library masked-mean + max
            + embed_dim * 2                              # self maindeck masked-mean + max
            + embed_dim * 2                              # self sideboard masked-mean + max
            + embed_dim * 2                              # opponent maindeck masked-mean + max
            + embed_dim * 2                              # opponent sideboard masked-mean + max
        )
        # When the per-action logit head is enabled, the encoded per-action tensor
        # (MAX_ACTIONS × _PER_ACTION_DIM) is appended at the END of the returned
        # features so PerActionMaskablePolicy can slice it back out by offset.
        assert observation_space.shape[0] == _ENV_OBS_SIZE, (
            f"CardGameExtractor got an obs of {observation_space.shape[0]} floats but "
            f"this build's OBS_SIZE is {_ENV_OBS_SIZE} — a checkpoint saved against a "
            f"different observation layout cannot be loaded (retrain from scratch)")
        self.per_action_head = per_action_head
        if per_action_head:
            self.per_action_slots = _MAX_ACTIONS
            self.per_action_dim = _PER_ACTION_DIM
            self.per_action_offset = base_features_dim
            features_dim = base_features_dim + self.per_action_slots * self.per_action_dim
        else:
            features_dim = base_features_dim
        super().__init__(observation_space, features_dim=features_dim)

        # Side-channel set by every forward(): the (B,) long tensor of value-bucket
        # indices sliced off the matchup tail. PerActionMaskablePolicy reads it to
        # pick each sample's column out of the multi-head value_net. Not a buffer —
        # it is per-batch scratch, never part of the checkpoint.
        self.last_bucket = None

        # Shared card-identity embedding. Slot id -1 (empty) maps to padding row 0;
        # real ids 0..N_CARD_TYPES-1 map to rows 1..N_CARD_TYPES.
        self.card_emb = nn.Embedding(N_CARD_TYPES + 1, card_embed_dim, padding_idx=0)

        # Encoder for permanent slots (35 non-id floats + chosen-name embedding +
        # card embedding). Status, counters, is_blocked/is_phased_out, and the
        # keyword multi-hot are scalars; the 4 entity refs enter as raw normalized
        # floats (v1). The chosen-name id (Pithing Needle / Disruptor Flute named
        # card) is embedded via the shared card embedding, like the card id.
        self.perm_encoder = nn.Sequential(
            nn.Linear(_PERM_STATUS_FLOATS + 3 * card_embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # Encoder for stack slots: controller_is_self + is_spell, x_or_amount,
        # cast qualifiers, chosen-mode multi-hot, per-target scalar flags
        # (present/is_player/ctrl_is_self/slot_ref × 4 sub-slots), the object's
        # card embedding, and the masked mean of its announced targets' card
        # embeddings.  2 + 1 + 7 + 6 + 4*4 = 32 scalars.
        _stack_scalars = (2 + 1 + _STACK_QUALS + _STACK_MODE_SLOTS
                          + _STACK_TGT_SLOTS * (_STACK_TGT_FIELDS - 1))
        self.stack_encoder = nn.Sequential(
            nn.Linear(_stack_scalars + 2 * card_embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, half),
            nn.ReLU(),
        )

        # Shared encoder for pure card-identity slots (graveyard, hand, top-library)
        self.entity_encoder = nn.Sequential(
            nn.Linear(card_embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # Shared encoder for the three deck-identity blocks (self live library,
        # opponent static maindeck + sideboard). Each slot is a (card_id, count)
        # pair: the card id is embedded via the shared card embedding and the
        # normalized count appended, so one encoder covers all three blocks.
        self.decklist_encoder = nn.Sequential(
            nn.Linear(card_embed_dim + 1, embed_dim),
            nn.ReLU(),
        )

        # Encoder for the dense opponent revealed-cards multi-hot (many bits set,
        # not a single id), so it gets its own weights.
        self.revealed_encoder = nn.Sequential(
            nn.Linear(_REVEALED_SIZE, embed_dim),
            nn.ReLU(),
        )

        # Action-category embedding — shared by the embedded recent-history block
        # (always) and the per-action encoder (when enabled). History and action
        # slots use the same category/ACTION_CATEGORY_MAX normalization.
        self.action_cat_emb = nn.Embedding(ACTION_CATEGORY_MAX + 1, _ACTION_CAT_EMBED)

        # Per-action encoder (opt-in): category embed + referenced-card embed +
        # controller_is_self + zone_ref embed + the gathered board/stack embedding
        # of the action's referenced entity (via the refs block) → a per-action
        # feature. Shares self.card_emb for the target card identity so a target
        # land's id is embedded, not a raw float.
        self._embed_dim = embed_dim
        if per_action_head:
            self.zone_emb = nn.Embedding(N_REF_ZONES, _REF_ZONE_EMBED)
            self.action_encoder = nn.Sequential(
                nn.Linear(_ACTION_CAT_EMBED + card_embed_dim + 1 + _REF_ZONE_EMBED
                          + embed_dim + 1,  # +1 ctrl flag, +1 option_ordinal scalar
                          embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, self.per_action_dim),
                nn.ReLU(),
            )

    def _embed_ids(self, id_floats: torch.Tensor):
        """Map normalized id floats → (card embeddings, present mask).

        id_floats : (..., ) normalized ids (idx/N_CARD_TYPES; -1/N = empty)
        returns   : (emb (..., card_embed_dim), present (...,) bool)
        """
        idx = torch.round(id_floats * N_CARD_TYPES).long()
        present = idx >= 0
        emb = self.card_emb((idx + 1).clamp(0, N_CARD_TYPES))  # -1 → 0 (padding)
        return emb, present

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        global_ctx    = obs[:, :_GLOBAL_SIZE]
        hist_ctx      = obs[:, _HIST_START:_HIST_END]           # action history (128 × 4)
        meta_ctx      = obs[:, _HIST_END:_KNOWN_TOP_LIB_START]  # match ctx + library ctx + current turn (8)
        revealed      = obs[:, _REVEALED_START:_REVEALED_END]   # opponent revealed-cards multi-hot
        pending       = obs[:, _PENDING_START:_PENDING_END]     # pending-decision source id + ctrl flag
        extras        = obs[:, _EXTRAS_START:_EXTRAS_END]       # 19 global extras (raw passthrough)
        action_extras = obs[:, _STATE_END:_BUCKET_IDX]          # action cats|ids|ctrl|zone|refs + cost features
        # Matchup tail: the raw value-bucket index is STRIPPED here (a bucket id is
        # a meaningless magnitude to the trunk) and stashed for the policy's
        # multi-head critic to gather with; the two archetype one-hots DO feed the
        # trunk as explicit matchup conditioning.
        self.last_bucket = torch.round(obs[:, _BUCKET_IDX]).long().clamp_(
            0, N_VALUE_BUCKETS - 1)
        arch_onehot   = obs[:, _ARCH_ONEHOT_START:_ARCH_ONEHOT_END]

        # Embedded recent history: card ids as raw floats are unlearnable (vocab
        # order is meaningless), so the K most recent entries get their card id
        # and category embedded per entry, flattened recency-ordered. The full
        # raw block is still passed through below for the older tail.
        hist_entries = hist_ctx.reshape(-1, _HIST_ENTRIES, _HIST_ENTRY_SIZE)
        recent = hist_entries[:, :_HIST_RECENT_K]               # (B, K, 4) newest first
        rec_cat_idx = torch.round(recent[:, :, 0] * ACTION_CATEGORY_MAX).long() \
            .clamp_(0, ACTION_CATEGORY_MAX)
        rec_cat_e = self.action_cat_emb(rec_cat_idx)            # (B, K, cat_embed)
        rec_card_e, _ = self._embed_ids(recent[:, :, 1])        # (B, K, card_embed)
        hist_recent = torch.cat([rec_cat_e, rec_card_e, recent[:, :, 2:4]], dim=-1) \
            .reshape(recent.shape[0], -1)                       # (B, K*(cat+card+2))

        # Pending-decision context: embed WHAT is asking for the current choice
        # (may not be on the stack yet — targets are announced pre-push).
        pending_emb, _ = self._embed_ids(pending[:, 0])         # (B, card_embed)
        pending_feat = torch.cat([pending_emb, pending[:, 1:2]], dim=-1)

        perms     = obs[:, _PERM_START:_PERM_END].reshape(-1, _PERM_SLOTS, _PERM_SLOT_SIZE)
        stack     = obs[:, _STACK_START:_STACK_END].reshape(-1, _STACK_SLOTS, _STACK_SLOT_SIZE)
        graveyard = obs[:, _GY_START:_GY_END].reshape(-1, _GY_SLOTS, _GY_SLOT_SIZE)
        exile     = obs[:, _EXILE_START:_EXILE_END].reshape(-1, _EXILE_SLOTS, _EXILE_SLOT_SIZE)
        hand      = obs[:, _HAND_START:_HAND_END].reshape(-1, _HAND_SLOTS, _HAND_SLOT_SIZE)
        opp_hand  = obs[:, _OPP_KNOWN_HAND_START:_OPP_KNOWN_HAND_END].reshape(
            -1, _OPP_KNOWN_HAND_SLOTS, _OPP_KNOWN_HAND_SLOT_SIZE)
        top_lib   = obs[:, _KNOWN_TOP_LIB_START:_KNOWN_TOP_LIB_END].reshape(
            -1, _KNOWN_TOP_LIB_SLOTS, _KNOWN_TOP_LIB_SLOT_SIZE)
        self_lib  = obs[:, _SELF_LIVE_LIB_START:_SELF_LIVE_LIB_END].reshape(
            -1, _DECKLIST_MAIN_SLOTS, _DECKLIST_SLOT_SIZE)
        self_main = obs[:, _SELF_DECK_MAIN_START:_SELF_DECK_MAIN_END].reshape(
            -1, _DECKLIST_MAIN_SLOTS, _DECKLIST_SLOT_SIZE)
        self_side = obs[:, _SELF_DECK_SIDE_START:_SELF_DECK_SIDE_END].reshape(
            -1, _DECKLIST_SIDE_SLOTS, _DECKLIST_SLOT_SIZE)
        opp_main  = obs[:, _OPP_DECK_MAIN_START:_OPP_DECK_MAIN_END].reshape(
            -1, _DECKLIST_MAIN_SLOTS, _DECKLIST_SLOT_SIZE)
        opp_side  = obs[:, _OPP_DECK_SIDE_START:_OPP_DECK_SIDE_END].reshape(
            -1, _DECKLIST_SIDE_SLOTS, _DECKLIST_SLOT_SIZE)

        # Embed card identity per slot, then build each slot's encoder input. The
        # chosen-name id and the returnable-exile id are each embedded through the
        # same shared card embedding as the card id and appended after the 35 status
        # floats (chosen-name, then returnable-exile, then card id).
        perm_card_emb, perm_present = self._embed_ids(perms[:, :, _PERM_CARD_OFF])
        perm_chosen_emb, _ = self._embed_ids(perms[:, :, _PERM_CHOSEN_NAME_OFF])
        perm_returnable_emb, _ = self._embed_ids(perms[:, :, _PERM_RETURNABLE_OFF])
        perm_in = torch.cat([perms[:, :, :_PERM_STATUS_FLOATS],
                             perm_chosen_emb, perm_returnable_emb, perm_card_emb], dim=-1)

        stk_card_emb, stk_present = self._embed_ids(stack[:, :, 1])
        # Announced-target sub-slots: (B, 12, 4, 5) of
        # [present, is_player, ctrl, slot_ref, card id].
        stk_tgts = stack[:, :, _STACK_TGT_OFF:].reshape(
            -1, _STACK_SLOTS, _STACK_TGT_SLOTS, _STACK_TGT_FIELDS)
        stk_tgt_emb, _ = self._embed_ids(stk_tgts[:, :, :, _STACK_TGT_FIELDS - 1])  # (B, 12, 4, card_embed)
        stk_tgt_mask = stk_tgts[:, :, :, 0:1]                        # present flag
        stk_tgt_agg = (stk_tgt_emb * stk_tgt_mask).sum(2) / stk_tgt_mask.sum(2).clamp(min=1.0)
        stk_xquals = stack[:, :, _STACK_XAMT_OFF:_STACK_MODE_OFF]    # x_or_amount + 7 qualifiers
        stk_modes = stack[:, :, _STACK_MODE_OFF:_STACK_TGT_OFF]      # chosen-mode multi-hot
        stk_tgt_scalars = stk_tgts[:, :, :, :_STACK_TGT_FIELDS - 1].reshape(
            -1, _STACK_SLOTS, _STACK_TGT_SLOTS * (_STACK_TGT_FIELDS - 1))
        stk_in = torch.cat([stack[:, :, 0:1], stack[:, :, 2:3], stk_xquals, stk_modes,
                            stk_tgt_scalars, stk_card_emb, stk_tgt_agg], dim=-1)

        gy_emb_in, gy_present = self._embed_ids(graveyard[:, :, 0])
        ex_emb_in, ex_present = self._embed_ids(exile[:, :, 0])
        hand_emb_in, hand_present = self._embed_ids(hand[:, :, 0])
        opp_hand_emb_in, opp_hand_present = self._embed_ids(opp_hand[:, :, 0])
        top_lib_emb_in, _ = self._embed_ids(top_lib[:, :, 0])

        # Deck-identity blocks: embed the card id, append the normalized count,
        # encode with the shared decklist_encoder, pool masked mean+max per block.
        self_lib_emb, self_lib_present = self._embed_ids(self_lib[:, :, _DECKLIST_CARD_OFF])
        self_main_emb, self_main_present = self._embed_ids(self_main[:, :, _DECKLIST_CARD_OFF])
        self_side_emb, self_side_present = self._embed_ids(self_side[:, :, _DECKLIST_CARD_OFF])
        opp_main_emb, opp_main_present = self._embed_ids(opp_main[:, :, _DECKLIST_CARD_OFF])
        opp_side_emb, opp_side_present = self._embed_ids(opp_side[:, :, _DECKLIST_CARD_OFF])
        self_lib_in = torch.cat(
            [self_lib_emb, self_lib[:, :, _DECKLIST_COUNT_OFF:_DECKLIST_COUNT_OFF + 1]], dim=-1)
        self_main_in = torch.cat(
            [self_main_emb, self_main[:, :, _DECKLIST_COUNT_OFF:_DECKLIST_COUNT_OFF + 1]], dim=-1)
        self_side_in = torch.cat(
            [self_side_emb, self_side[:, :, _DECKLIST_COUNT_OFF:_DECKLIST_COUNT_OFF + 1]], dim=-1)
        opp_main_in = torch.cat(
            [opp_main_emb, opp_main[:, :, _DECKLIST_COUNT_OFF:_DECKLIST_COUNT_OFF + 1]], dim=-1)
        opp_side_in = torch.cat(
            [opp_side_emb, opp_side[:, :, _DECKLIST_COUNT_OFF:_DECKLIST_COUNT_OFF + 1]], dim=-1)

        # Encode each slot type with its shared-weight encoder
        perm_emb    = self.perm_encoder(perm_in)       # (B, 96, embed)
        stk_emb     = self.stack_encoder(stk_in)       # (B, 12, embed//2)
        gy_emb      = self.entity_encoder(gy_emb_in)   # (B, 128, embed)
        ex_emb      = self.entity_encoder(ex_emb_in)   # (B, 128, embed)  — shared weights
        hand_emb    = self.entity_encoder(hand_emb_in) # (B, 10, embed)  — shared weights
        opp_hand_emb = self.entity_encoder(opp_hand_emb_in)  # (B, 10, embed)  — shared weights
        top_lib_emb = self.entity_encoder(top_lib_emb_in)  # (B, 5, embed)  — shared weights
        self_lib_enc = self.decklist_encoder(self_lib_in)  # (B, 48, embed)  — shared weights
        self_main_enc = self.decklist_encoder(self_main_in)  # (B, 48, embed) — shared weights
        self_side_enc = self.decklist_encoder(self_side_in)  # (B, 15, embed) — shared weights
        opp_main_enc = self.decklist_encoder(opp_main_in)  # (B, 48, embed)  — shared weights
        opp_side_enc = self.decklist_encoder(opp_side_in)  # (B, 15, embed)  — shared weights

        # Aggregate: masked mean+max for perms, stack, graveyard, and hand (skip
        # empty slots — an unmasked mean over the 12 stack slots diluted a lone
        # real spell 1:12 against the empty-slot bias encodings); mean for
        # top-library.
        perm_agg    = _masked_mean_max(perm_emb, perm_present)
        stk_agg     = _masked_mean_max(stk_emb, stk_present)
        gy_agg      = _masked_mean_max(gy_emb,   gy_present)
        ex_agg      = _masked_mean_max(ex_emb,   ex_present)
        hand_agg    = _masked_mean_max(hand_emb, hand_present)
        opp_hand_agg = _masked_mean_max(opp_hand_emb, opp_hand_present)
        top_lib_agg = top_lib_emb.mean(1)
        revealed_agg = self.revealed_encoder(revealed)  # (B, embed) dense multi-hot encoding
        self_lib_agg = _masked_mean_max(self_lib_enc, self_lib_present)
        self_main_agg = _masked_mean_max(self_main_enc, self_main_present)
        self_side_agg = _masked_mean_max(self_side_enc, self_side_present)
        opp_main_agg = _masked_mean_max(opp_main_enc, opp_main_present)
        opp_side_agg = _masked_mean_max(opp_side_enc, opp_side_present)

        base = torch.cat([global_ctx, hist_ctx, hist_recent, meta_ctx, top_lib_agg,
                          revealed_agg, pending_feat, extras, action_extras,
                          arch_onehot,
                          perm_agg, stk_agg, gy_agg, ex_agg, hand_agg, opp_hand_agg,
                          self_lib_agg, self_main_agg, self_side_agg,
                          opp_main_agg, opp_side_agg], dim=-1)
        if not self.per_action_head:
            return base

        # Encode each candidate action from its own (category, referenced-card,
        # controller, zone_ref, referenced-entity embedding, option ordinal) tuple.
        # The action block is the first 6*MAX_ACTIONS floats of action_extras:
        # cats | ids | ctrl | zone | refs | ords. Appended flat; sliced back out by
        # the policy's action scorer. Padded slots (beyond num_choices) are harmless
        # — their logits are masked out by MaskablePPO's action mask downstream.
        a0 = _STATE_END
        cats = obs[:, a0:a0 + _MAX_ACTIONS]
        act_ids = obs[:, a0 + _MAX_ACTIONS:a0 + 2 * _MAX_ACTIONS]
        ctrl = obs[:, a0 + 2 * _MAX_ACTIONS:a0 + 3 * _MAX_ACTIONS]
        zone = obs[:, a0 + 3 * _MAX_ACTIONS:a0 + 4 * _MAX_ACTIONS]
        refs = obs[:, a0 + 4 * _MAX_ACTIONS:a0 + 5 * _MAX_ACTIONS]
        ords = obs[:, a0 + 5 * _MAX_ACTIONS:a0 + 6 * _MAX_ACTIONS]
        cat_idx = torch.round(cats * ACTION_CATEGORY_MAX).long().clamp_(0, ACTION_CATEGORY_MAX)
        cat_e = self.action_cat_emb(cat_idx)                 # (B, A, cat_embed)
        act_id_e, _ = self._embed_ids(act_ids)               # (B, A, card_embed)
        zone_idx = torch.round(zone * REF_ZONE_MAX).long().clamp_(0, REF_ZONE_MAX)
        zone_e = self.zone_emb(zone_idx)                     # (B, A, zone_embed)

        # Gather each action's referenced entity's ENCODED embedding via the
        # unified ref space (0-47 self perm, 48-95 opp perm, 96-107 stack). The
        # stack embeddings are half-width, so zero-pad them up to embed_dim, then
        # append a zeros row as the "no reference" sentinel (ref 0.0 → index 108).
        E = self._embed_dim
        stk_emb_pad = torch.cat(
            [stk_emb, stk_emb.new_zeros(stk_emb.shape[0], _STACK_SLOTS,
                                        E - stk_emb.shape[-1])], dim=-1)
        ent_table = torch.cat([perm_emb, stk_emb_pad], dim=1)          # (B, 108, E)
        ent_table = torch.cat(
            [ent_table, ent_table.new_zeros(ent_table.shape[0], 1, E)], dim=1)  # (B, 109, E)
        ref_idx = torch.round(refs * N_ENTITY_REF_SLOTS).long() - 1    # -1 = none
        ref_idx = torch.where(ref_idx < 0, ref_idx.new_full((), N_ENTITY_REF_SLOTS),
                              ref_idx).clamp_(0, N_ENTITY_REF_SLOTS)   # none → zeros row
        ref_e = torch.gather(ent_table, 1,
                             ref_idx.unsqueeze(-1).expand(-1, -1, E))  # (B, A, E)

        pa_in = torch.cat([cat_e, act_id_e, ctrl.unsqueeze(-1), zone_e, ref_e,
                           ords.unsqueeze(-1)], dim=-1)
        pa = self.action_encoder(pa_in)                      # (B, A, per_action_dim)
        return torch.cat([base, pa.reshape(pa.shape[0], -1)], dim=-1)


class _ActionScorer(nn.Module):
    """Score each candidate action from (per-action feature, policy latent).

    logit[i] = w · tanh(W_l · latent_pi + W_a · action_feat[i]). The policy latent
    supplies the global context ("what effect is resolving, whose turn, board
    state") and the per-action feature supplies "this candidate is a target on my
    own permanent", so the two are combined for THIS action's logit — instead of a
    single Linear(latent → MAX_ACTIONS) learning a fixed positional map.
    """

    def __init__(self, latent_dim: int, per_action_dim: int, hidden: int = 64):
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, hidden)
        self.action_proj = nn.Linear(per_action_dim, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, latent_pi: torch.Tensor, per_action: torch.Tensor) -> torch.Tensor:
        # latent_pi: (B, L)   per_action: (B, A, D)   ->   logits: (B, A)
        h = torch.tanh(self.latent_proj(latent_pi).unsqueeze(1) + self.action_proj(per_action))
        return self.out(h).squeeze(-1)


class PerActionMaskablePolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy that scores logits per candidate action.

    Requires a CardGameExtractor built with ``per_action_head=True`` (which appends
    the encoded per-action tensor to the features). The stock policy's
    ``action_net`` (a Linear over the pooled latent) is left unused; logits come
    from ``_ActionScorer`` so each action's own encoded features (category, target
    card embedding, controller_is_self) drive its logit.

    The critic is MULTI-HEAD: ``value_net`` has one column per
    (self archetype x opp archetype) value bucket, and each sample's column is
    gathered by the bucket index the extractor sliced off the observation's
    matchup tail. Upstream (GAE, value loss) still sees one scalar per sample, but
    a burn race's value statistics no longer fight a control grind's in the last
    layer.

    Prototype: enabling this changes the network shape, so it is NOT
    checkpoint-compatible with the stock MlpPolicy models.
    """

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        assert self.share_features_extractor, \
            "PerActionMaskablePolicy requires share_features_extractor=True"
        fe = self.features_extractor
        assert getattr(fe, "per_action_head", False), \
            "PerActionMaskablePolicy requires CardGameExtractor(per_action_head=True)"
        self._pa_offset = fe.per_action_offset
        self._pa_slots = fe.per_action_slots
        self._pa_dim = fe.per_action_dim
        # Per-archetype-bucket critic: replace the stock single-output value head
        # with one column per bucket (~N_VALUE_BUCKETS * latent params — trivial).
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, N_VALUE_BUCKETS)
        if self.ortho_init:
            self.value_net.apply(partial(self.init_weights, gain=1.0))
        # Per-bucket PopArt statistics (Hessel et al. 2019). ALWAYS present as
        # buffers — they ride in the checkpoint, and at (mu=0, sigma=1) the
        # de-normalization below is the identity, so a run without --popart
        # behaves exactly as if they weren't here. Only PopArtMaskablePPO
        # (train/popart.py) ever updates them.
        self.register_buffer("popart_mu", torch.zeros(N_VALUE_BUCKETS))
        self.register_buffer("popart_sigma", torch.ones(N_VALUE_BUCKETS))
        self.register_buffer("popart_count", torch.zeros(N_VALUE_BUCKETS))
        # When True, _bucket_values returns the head's NORMALIZED output instead of
        # the de-normalized value. The PopArt algorithm flips this for the duration
        # of its value-loss computation (where the targets are normalized too);
        # rollout collection always sees real-scale values so GAE is unaffected.
        self.popart_normalized_out = False
        self.action_scorer = _ActionScorer(self.mlp_extractor.latent_dim_pi, self._pa_dim)
        if self.ortho_init:
            self.action_scorer.out.apply(partial(self.init_weights, gain=0.01))
        # The scorer was created after super()'s optimizer captured the parameter
        # list, so rebuild the optimizer to include it.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _slice_per_action(self, features: torch.Tensor) -> torch.Tensor:
        pa = features[:, self._pa_offset:self._pa_offset + self._pa_slots * self._pa_dim]
        return pa.reshape(pa.shape[0], self._pa_slots, self._pa_dim)

    def _current_buckets(self, n_samples: int) -> torch.Tensor:
        """The (B,) bucket indices the extractor stashed for the current batch."""
        bucket = self.features_extractor.last_bucket
        assert bucket is not None and bucket.shape[0] == n_samples, (
            "multi-head critic: the features extractor did not stash a value-bucket "
            f"index for this batch ({None if bucket is None else tuple(bucket.shape)} "
            f"vs {n_samples} samples) — extract_features() must run first")
        return bucket

    def _bucket_values(self, latent_vf: torch.Tensor) -> torch.Tensor:
        """Value per sample, gathered from its archetype-bucket head column.

        Uses the bucket index the shared features extractor sliced off the obs's
        matchup tail during the extract_features() call that produced
        ``latent_vf`` — so every caller MUST run extract_features first (all three
        entry points below do). Returns (B, 1), exactly like the stock scalar head.

        The head's output is in NORMALIZED value space and is de-normalized here
        with that bucket's PopArt (mu, sigma) — the identity while PopArt is off.
        """
        all_values = self.value_net(latent_vf)                 # (B, N_VALUE_BUCKETS)
        bucket = self._current_buckets(all_values.shape[0])
        v_norm = all_values.gather(1, bucket.view(-1, 1))      # (B, 1)
        if self.popart_normalized_out:
            return v_norm
        return v_norm * self.popart_sigma[bucket].view(-1, 1) \
            + self.popart_mu[bucket].view(-1, 1)

    @torch.no_grad()
    def popart_update(self, buckets, means, second_moments, counts,
                      beta: float = 0.99, eps: float = 1e-4):
        """Fold a batch's per-bucket return statistics into the PopArt stats.

        ``buckets`` are the bucket indices present in the batch; ``means`` /
        ``second_moments`` their return mean and mean-of-squares; ``counts`` the
        per-bucket sample counts (unused for weighting beyond the first update,
        kept for the debias/first-update rule). For every updated bucket the head
        column is rescaled so the network's OUTPUT is preserved across the stats
        change (Hessel et al. 2019, the "art" half of PopArt):

            w <- w * sigma_old / sigma_new
            b <- (b * sigma_old + mu_old - mu_new) / sigma_new

        Returns ``(mu_new, sigma_new)`` (full-length tensors) so the caller can
        normalize its targets with exactly the values the head was rescaled to.
        """
        idx = torch.as_tensor(buckets, dtype=torch.long, device=self.popart_mu.device)
        m = torch.as_tensor(means, dtype=self.popart_mu.dtype, device=idx.device)
        s2 = torch.as_tensor(second_moments, dtype=self.popart_mu.dtype, device=idx.device)
        n = torch.as_tensor(counts, dtype=self.popart_mu.dtype, device=idx.device)
        mu_old = self.popart_mu[idx].clone()
        sigma_old = self.popart_sigma[idx].clone()
        # First update of a bucket adopts the batch statistics outright (no decay
        # bias); later updates fold them in with decay `beta`.
        first = self.popart_count[idx] == 0
        b = torch.where(first, torch.zeros_like(m), torch.full_like(m, float(beta)))
        mu_new = (1.0 - b) * m + b * mu_old
        nu_old = sigma_old ** 2 + mu_old ** 2          # running second moment
        nu_new = (1.0 - b) * s2 + b * nu_old
        sigma_new = torch.sqrt(torch.clamp(nu_new - mu_new ** 2, min=eps ** 2))
        sigma_new = torch.clamp(sigma_new, min=eps, max=1e6)
        # Output-preserving rescale of the affected head columns.
        scale = (sigma_old / sigma_new).view(-1, 1)
        self.value_net.weight[idx] = self.value_net.weight[idx] * scale
        self.value_net.bias[idx] = (self.value_net.bias[idx] * sigma_old
                                    + mu_old - mu_new) / sigma_new
        self.popart_mu[idx] = mu_new
        self.popart_sigma[idx] = sigma_new
        self.popart_count[idx] = self.popart_count[idx] + n
        return self.popart_mu, self.popart_sigma

    def _dist_from(self, features, latent_pi, action_masks):
        logits = self.action_scorer(latent_pi, self._slice_per_action(features))
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def forward(self, obs, deterministic=False, action_masks=None):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self._bucket_values(latent_vf)
        distribution = self._dist_from(features, latent_pi, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions, action_masks=None):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._dist_from(features, latent_pi, action_masks)
        values = self._bucket_values(latent_vf)
        return values, distribution.log_prob(actions), distribution.entropy()

    def predict_values(self, obs):
        """Critic-only pass (SB3 calls this for the GAE bootstrap value).

        Overridden so the value comes from the sample's archetype-bucket column;
        it must run extract_features itself to stash that bucket index.
        """
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self._bucket_values(latent_vf)

    def get_distribution(self, obs, action_masks=None):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._dist_from(features, latent_pi, action_masks)
