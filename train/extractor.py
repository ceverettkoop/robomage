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

Index layout must stay in sync with src/machine_io.h (STATE_SIZE = 5594):
  obs[0:36]            global context (player stats, step, flags, stack size); the
                         self_is_A seat flag [34] is zeroed before the network sees
                         it (network_global_ctx)
  obs[36:4164]         96 permanent slots × 43 floats
                         slots 0-47: self; slots 48-95: opponent (no controller
                         flag: the halves are split by controller)
                         0-9   status: power, toughness, tapped, attacking, blocking,
                               sickness, damage, is_creature, is_land, loyalty
                         10    p1p1_net (signed, /10)
                         11    other_counters (/10)
                         12-15 entity refs (normalized (idx+1)/108): attached_to,
                               attached_by, attack_target, blocking_target
                         16    is_blocked
                         17    is_phased_out
                         18    entered_this_turn
                         19    triggered-ability resolutions this turn (/10)
                         20    once-per-turn-gated activations this turn (/10)
                         21    cant_be_blocked_this_turn
                         22    combat_damage_prevented
                         23    pending_delayed_subject (watched by / a subject of a
                               waiting delayed trigger)
                         24-39 keyword multi-hot (N_OBS_KEYWORDS = 16)
                         40    chosen-name id (Pithing Needle / Disruptor Flute named
                               card, Petrified Hamlet named land; sentinel = none)
                         41    returnable-exile id (card this permanent exiled that still
                               has a return path — Static Prison / Phelia; sentinel = none)
                         42    card id (LAST)
  obs[4164:4608]       12 stack slots × 37 floats (controller_is_self + card id +
                         is_spell + x_or_amount/10 + 7 cast qualifiers +
                         chosen-mode multi-hot(6) + 4 announced-target sub-slots ×
                         [present, is_player, controller_is_self, slot_ref, card id])
  obs[4608:4736]      128 graveyard slots × 1 float (card id, recency-ordered)
                         slots 0-63: self; slots 64-127: opponent
  obs[4736:4864]      128 exile slots × 1 float (card id, recency-ordered)
                         slots 0-63: self; slots 64-127: opponent
  obs[4864:4874]       10 hand slots    × 1 float  (card id)
  obs[4874:4878]       match context (4 floats: game_number, self_wins, opp_wins, sideboard_phase)
  obs[4878:4880]       library counts (self_lib/60, opp_lib/60)
  obs[4880]            current game turn / 50
  obs[4881:4886]       5 known top-of-library slots × 1 float (card id, sentinel = unknown)
  obs[4886:4896]       10 known opponent-hand slots × 1 float (card id)
  obs[4896:4898]       pending-decision context (source card id + ctrl_is_self)
  obs[4898:4925]       global extras (self/opp lands played, self/opp monarch, city's
                         blessing, revolt, pending extra turns, is_day, is_night,
                         self/opp has_passed + is_priority_window, self/opp
                         mulligans taken + self bottom remaining, MandatoryChoice
                         one-hot(6), self_plays_first, sideboard swaps made,
                         sideboard delta)
  obs[4925:5021]       48 self live-library slots × (card id, count)
  obs[5021:5149]       the viewer's own live 75: 48 maindeck + 16 sideboard slots
                         × (card id, count). 16, not 15: mid-swap a cut card is
                         momentarily the sideboard's 16th (DECKLIST_SIDE_SLOTS).
  obs[5149:5341]       the opponent's REGISTERED 75 (frozen at match start):
                         48 maindeck + 16 sideboard slots × (card id, count,
                         revealed — the opponent has shown that card this match)
  obs[5341:5360]       mana development: self (10 floats: potential W,U,B,R,G,C,
                         potential_total, lands_in_play, lands_in_hand,
                         land_drops_remaining) then opponent
                         (9 — no lands_in_hand, which is hidden information)
  obs[5360:5364]       log-scaled vitals: self (log1p(max(life,0))/log1p(20),
                         log1p(library)/log1p(60)) then opponent — the same counts
                         as the linear floats above, re-warped for resolution near
                         zero (see the LOG VITALS block in machine_io.h)
  obs[5364:5386]       per-turn counters: self (11 floats: spells, noncreature and
                         instant/sorcery spells cast, cards drawn (/10), life gained,
                         life lost (/20), spell-color multi-hot W,U,B,R,G) then opponent
  obs[5386:5594]       16 pending delayed-trigger slots × 13 floats (present,
                         controller_is_self, state (0 waiting / 1 on the stack),
                         stack_ref, creator card id, creator_ref, subject_ref,
                         subject card id, fire_on one-hot(4), fires_this_turn),
                         packed in registration order
  obs[5594:]           action metadata (cats|ids|ctrl|zone|refs|ords) + matchup
                         tail (appended by env.py; refs are normalized
                         entity-slot references, (idx+1)/108 with 0.0 = none)
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

try:
    from card_costs import N_CARD_TYPES
    from card_props import _CARD_PROP_MATRIX, N_CARD_PROPS
except ImportError:
    from train.card_costs import N_CARD_TYPES
    from train.card_props import _CARD_PROP_MATRIX, N_CARD_PROPS

# ACTION_CATEGORY_MAX (mirrors src/classes/action.h via codegen) is needed to
# decode the per-action category-norm floats back to integer category ids for the
# per-action logit head. Same source of truth env.py uses for the action block.
try:
    from _enums import (ACTION_CATEGORY_MAX, N_OBS_KEYWORDS, N_MANDATORY_CHOICES,
                        DECKLIST_MAIN_SLOTS, DECKLIST_SIDE_SLOTS,
                        MAX_BATTLEFIELD_SLOTS, MAX_STACK_DISPLAY, MAX_STACK_MODES,
                        MAX_STACK_TGTS, MAX_GY_SLOTS, MAX_HAND_SLOTS,
                        KNOWN_TOP_LIBRARY_SIZE,
                        CARD_ID_SLOT_SIZE, STACK_HEAD_FIELDS, STACK_XAMT_FIELDS,
                        STACK_QUAL_FIELDS, STACK_TGT_FIELDS,
                        MATCH_CTX_SIZE, LIBRARY_CTX_SIZE, CUR_TURN_SIZE,
                        PENDING_DECISION_SIZE, EXTRAS_SCALARS,
                        EXTRAS_PRIORITY_SIZE, EXTRAS_MULLIGAN_SIZE,
                        EXTRAS_SB_CTX_SIZE, DECKLIST_SLOT_SIZE, OPP_DECKLIST_SLOT_SIZE,
                        OPP_DECKLIST_REVEALED_OFF,
                        MANA_DEV_SELF_SIZE, MANA_DEV_OPP_SIZE,
                        LOG_VITALS_PLAYER_SIZE, PER_TURN_PLAYER_SIZE,
                        MAX_DELAYED_TRIGGER_SLOTS, DELAYED_SLOT_SIZE)
except ImportError:
    from train._enums import (ACTION_CATEGORY_MAX, N_OBS_KEYWORDS, N_MANDATORY_CHOICES,
                             DECKLIST_MAIN_SLOTS, DECKLIST_SIDE_SLOTS,
                             MAX_BATTLEFIELD_SLOTS, MAX_STACK_DISPLAY, MAX_STACK_MODES,
                             MAX_STACK_TGTS, MAX_GY_SLOTS, MAX_HAND_SLOTS,
                             KNOWN_TOP_LIBRARY_SIZE,
                             CARD_ID_SLOT_SIZE, STACK_HEAD_FIELDS, STACK_XAMT_FIELDS,
                             STACK_QUAL_FIELDS, STACK_TGT_FIELDS,
                             MATCH_CTX_SIZE, LIBRARY_CTX_SIZE, CUR_TURN_SIZE,
                             PENDING_DECISION_SIZE, EXTRAS_SCALARS,
                             EXTRAS_PRIORITY_SIZE, EXTRAS_MULLIGAN_SIZE,
                             EXTRAS_SB_CTX_SIZE, DECKLIST_SLOT_SIZE, OPP_DECKLIST_SLOT_SIZE,
                             OPP_DECKLIST_REVEALED_OFF,
                             MANA_DEV_SELF_SIZE, MANA_DEV_OPP_SIZE,
                             LOG_VITALS_PLAYER_SIZE, PER_TURN_PLAYER_SIZE,
                             MAX_DELAYED_TRIGGER_SLOTS, DELAYED_SLOT_SIZE)


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
# 10 status (incl. loyalty) + 2 counters + 4 entity refs + is_blocked +
# is_phased_out + 5 per-turn statuses + keyword multi-hot + chosen-name id +
# returnable-exile id + card id (LAST) = 43 (the 5 per-turn statuses are followed
# by pending_delayed_subject)
_PERM_SLOT_SIZE  = 24 + N_OBS_KEYWORDS + 3  # 43
_PERM_STATUS_FLOATS   = _PERM_SLOT_SIZE - 3  # 40 non-id status/ref/keyword floats
_PERM_CHOSEN_NAME_OFF = _PERM_SLOT_SIZE - 3  # 40 (chosen-name id, 3rd-last)
_PERM_RETURNABLE_OFF  = _PERM_SLOT_SIZE - 2  # 41 (returnable-exile id, 2nd-last)
_PERM_CARD_OFF   = _PERM_SLOT_SIZE - 1      # 42 (card id is always LAST)

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

_KNOWN_TOP_LIB_SLOTS     = KNOWN_TOP_LIBRARY_SIZE  # known top-of-library cards
_KNOWN_TOP_LIB_SLOT_SIZE = CARD_ID_SLOT_SIZE  # card id per slot
_OPP_KNOWN_HAND_SLOTS    = MAX_HAND_SLOTS  # known opponent-hand card identities
_OPP_KNOWN_HAND_SLOT_SIZE = CARD_ID_SLOT_SIZE  # card id per slot

# Deck-identity tail blocks (mirror machine_io.h's deck-identity tail / env.py): five
# slot blocks — the viewer's live LIBRARY tally, the viewer's own LIVE maindeck +
# sideboard as (card_id, count), and the opponent's REGISTERED maindeck + sideboard
# as (card_id, count, revealed). card id first (norm_card_id, -1 empty sentinel),
# count second (/4.0), the opp slots' match-scoped revealed bit third. All five
# share one decklist_encoder; the self slots enter it with revealed = 0.
_DECKLIST_MAIN_SLOTS = DECKLIST_MAIN_SLOTS   # 48 (self live lib, self/opp main)
_DECKLIST_SIDE_SLOTS = DECKLIST_SIDE_SLOTS   # 16 (self/opp side)
_DECKLIST_SLOT_SIZE  = DECKLIST_SLOT_SIZE     # card id + count per self slot
_OPP_DECKLIST_SLOT_SIZE = OPP_DECKLIST_SLOT_SIZE  # card id + count + revealed per opp slot
_DECKLIST_CARD_OFF   = 0                      # card id first within a slot
_DECKLIST_COUNT_OFF  = 1                      # count second
_DECKLIST_REVEALED_OFF = OPP_DECKLIST_REVEALED_OFF  # revealed third (opp slots only)

# Pending delayed-trigger slots (mirror machine_io.h's DELAYED TRIGGERS block /
# env.py's _DT_* offsets): present, controller_is_self, state, stack_ref, creator
# card id, creator_ref, subject_ref, subject card id, fire_on one-hot,
# fires_this_turn. The two card ids go through the shared card embedding.
_DELAYED_SLOTS       = MAX_DELAYED_TRIGGER_SLOTS
_DELAYED_SLOT_SIZE   = DELAYED_SLOT_SIZE
_DT_PRESENT_OFF      = 0
_DT_CREATOR_ID_OFF   = 4
_DT_SUBJECT_ID_OFF   = 7
# Scalar columns fed to the encoder: ctrl, state, stack_ref (1..3), creator_ref,
# subject_ref (5..6), fire_on one-hot + fires_this_turn (8..end).
_DT_SCALARS          = 3 + 2 + (DELAYED_SLOT_SIZE - 8)

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
                     _SELF_IS_A_IDX,
                     OBS_SIZE as _ENV_OBS_SIZE, BUCKET_IDX as _BUCKET_IDX,
                     ARCH_ONEHOT_START as _ARCH_ONEHOT_START,
                     ARCH_ONEHOT_END as _ARCH_ONEHOT_END,
                     _OFF_POWER, _OFF_TOUGHNESS, _OFF_IS_TAPPED,
                     _OFF_IS_ATTACKING, _OFF_IS_BLOCKING, _OFF_IS_CREATURE,
                     _OFF_ATTACHED_TO)
except ImportError:
    import train.env as _env_mod
    from train.env import (MAX_ACTIONS as _MAX_ACTIONS, STATE_SIZE as _ENV_STATE_SIZE,
                           _GLOBAL_SIZE as _ENV_GLOBAL_SIZE, N_ENTITY_REF_SLOTS,
                           _SELF_IS_A_IDX,
                           OBS_SIZE as _ENV_OBS_SIZE, BUCKET_IDX as _BUCKET_IDX,
                           ARCH_ONEHOT_START as _ARCH_ONEHOT_START,
                           ARCH_ONEHOT_END as _ARCH_ONEHOT_END,
                           _OFF_POWER, _OFF_TOUGHNESS, _OFF_IS_TAPPED,
                           _OFF_IS_ATTACKING, _OFF_IS_BLOCKING, _OFF_IS_CREATURE,
                           _OFF_ATTACHED_TO)
try:
    from archetypes import N_VALUE_BUCKETS
except ImportError:
    from train.archetypes import N_VALUE_BUCKETS
# Width of the matchup tail's trunk-visible part (the two archetype one-hots) and
# of the whole tail. Both derived from env.py's offsets — never re-typed here.
_ARCH_ONEHOT_FEATS  = _ARCH_ONEHOT_END - _ARCH_ONEHOT_START
_MATCHUP_TAIL_FEATS = _ARCH_ONEHOT_END - _BUCKET_IDX
_GLOBAL_SIZE = _ENV_GLOBAL_SIZE   # header width (single source of truth: env.py)
# The header's "self is Player A" float. The drivers route seats by it, but no
# network may condition on which seat it plays: network_global_ctx zeroes it.
_SEAT_FLAG_IDX = _SELF_IS_A_IDX


def network_global_ctx(obs: torch.Tensor, global_size: int,
                       seat_flag_idx: int) -> torch.Tensor:
    """The header slice ``obs[:, :global_size]`` as the networks see it: the
    seat flag at ``seat_flag_idx`` replaced by 0.0. The ONE definition shared by
    CardGameExtractor and az_net.ScriptTrunk (which passes its baked constants),
    so the two trunks stay bit-identical. Plain-int arguments keep it
    TorchScript-compilable."""
    return torch.cat([obs[:, :seat_flag_idx],
                      torch.zeros_like(obs[:, seat_flag_idx:seat_flag_idx + 1]),
                      obs[:, seat_flag_idx + 1:global_size]], dim=1)

try:
    from _enums import REF_ZONE_MAX, N_REF_ZONES
except ImportError:
    from train._enums import REF_ZONE_MAX, N_REF_ZONES
_ACTION_CAT_EMBED  = 8    # learned embedding dim for the action category
_REF_ZONE_EMBED    = 4    # learned embedding dim for the per-action zone_ref
_PER_ACTION_DIM    = 32   # per-action feature width fed to the action scorer
_ATTN_HEADS        = 4    # heads of the entity-table self-attention layer
# Per-side board counts computed in forward from the permanent block (the pooled
# mean+max is count-invariant, so magnitudes must be handed to the trunk
# explicitly): n_permanents, n_creatures, n_untapped_creatures, n_attacking,
# n_blocking, total power, total toughness — for self, then the opponent.
_BOARD_COUNT_FEATS = 2 * 7
# The 4 per-permanent entity refs resolved by the one-hop gather, contiguous at
# _OFF_ATTACHED_TO: attached_to | attached_by | attack_target | blocking_target.
_PERM_N_REFS       = 4

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
# Then: match context (game_number, self_wins, opp_wins, sideboard_phase),
# library counts (self_lib/60, opp_lib/60), and the current turn.
_MATCH_CTX_START      = _HAND_END
_MATCH_CTX_END        = _MATCH_CTX_START + MATCH_CTX_SIZE      # library ctx start
_LIBRARY_CTX_END      = _MATCH_CTX_END + LIBRARY_CTX_SIZE      # current turn idx
_CUR_TURN_IDX         = _LIBRARY_CTX_END
_KNOWN_TOP_LIB_START  = _CUR_TURN_IDX + CUR_TURN_SIZE
_KNOWN_TOP_LIB_END    = _KNOWN_TOP_LIB_START + _KNOWN_TOP_LIB_SLOTS * _KNOWN_TOP_LIB_SLOT_SIZE
_OPP_KNOWN_HAND_START = _KNOWN_TOP_LIB_END
_OPP_KNOWN_HAND_END   = _OPP_KNOWN_HAND_START + _OPP_KNOWN_HAND_SLOTS * _OPP_KNOWN_HAND_SLOT_SIZE
# Pending decision context: card id of the spell/ability currently making a
# mid-resolution choice (sentinel = none) + its controller-is-viewer flag.
_PENDING_START        = _OPP_KNOWN_HAND_END
_PENDING_SIZE         = PENDING_DECISION_SIZE
_PENDING_END          = _PENDING_START + _PENDING_SIZE
# Global extras: self/opp lands played, monarch, city's blessing, revolt, pending
# extra turns, day/night flags, the priority-window context (pass flags +
# is_priority_window), the mulligan state, the MandatoryChoice one-hot, then
# self_plays_first and the two sideboard-progress scalars (swaps made, maindeck
# drift). Cheap scalar facts — passed through raw.
_EXTRAS_START         = _PENDING_END
_EXTRAS_SIZE          = (EXTRAS_SCALARS + EXTRAS_PRIORITY_SIZE + EXTRAS_MULLIGAN_SIZE
                         + N_MANDATORY_CHOICES + EXTRAS_SB_CTX_SIZE)  # 27
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
_OPP_DECK_MAIN_END    = _OPP_DECK_MAIN_START + _DECKLIST_MAIN_SLOTS * _OPP_DECKLIST_SLOT_SIZE
_OPP_DECK_SIDE_START  = _OPP_DECK_MAIN_END
_OPP_DECK_SIDE_END    = _OPP_DECK_SIDE_START + _DECKLIST_SIDE_SLOTS * _OPP_DECKLIST_SLOT_SIZE
# Mana development: the self half (per-color untapped-source potential, total,
# lands in play, lands in hand, land drops remaining) then the
# opponent's (same minus lands in hand). Cheap normalized scalars with no card
# identity, so — like the global extras — they are passed through RAW into the
# trunk rather than encoded; the point of the block is that the net gets the
# summary directly instead of re-deriving it from 96 pooled permanent slots.
_MANA_DEV_START       = _OPP_DECK_SIDE_END
_MANA_DEV_SIZE        = MANA_DEV_SELF_SIZE + MANA_DEV_OPP_SIZE
_MANA_DEV_END         = _MANA_DEV_START + _MANA_DEV_SIZE
# Log-scaled vitals: (log_life, log_library) for self, then the same for the
# opponent. Like the mana-development block these are plain normalized scalars with
# no card identity, so they pass through RAW into the trunk; the whole point is that
# the net receives the log warping directly instead of having to learn it from the
# linear life/library floats it already gets.
_LOG_VITALS_START     = _MANA_DEV_END
_LOG_VITALS_SIZE      = 2 * LOG_VITALS_PLAYER_SIZE
_LOG_VITALS_END       = _LOG_VITALS_START + _LOG_VITALS_SIZE
# Per-turn counters: spells / noncreature / instant-sorcery spells cast, cards drawn,
# life gained / lost and the spell-color multi-hot, self then opponent. Plain
# normalized scalars with no card identity, passed through RAW like the blocks above.
_PER_TURN_START       = _LOG_VITALS_END
_PER_TURN_SIZE        = 2 * PER_TURN_PLAYER_SIZE
_PER_TURN_END         = _PER_TURN_START + _PER_TURN_SIZE
# Pending delayed triggers: 16 slots encoded by delayed_encoder and pooled.
_DELAYED_START        = _PER_TURN_END
_DELAYED_END          = _DELAYED_START + _DELAYED_SLOTS * _DELAYED_SLOT_SIZE
_STATE_END            = _DELAYED_END
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
    ("_MATCH_CTX_START",      _MATCH_CTX_START),
    ("_CUR_TURN_IDX",         _CUR_TURN_IDX),
    ("_KNOWN_TOP_LIB_START",  _KNOWN_TOP_LIB_START),
    ("_OPP_KNOWN_HAND_START", _OPP_KNOWN_HAND_START),
    ("_EXTRAS_START",         _EXTRAS_START),
    ("_EXTRAS_END",           _EXTRAS_END),
    ("_SELF_LIVE_LIB_START",  _SELF_LIVE_LIB_START),
    ("_SELF_DECK_MAIN_START", _SELF_DECK_MAIN_START),
    ("_SELF_DECK_SIDE_START", _SELF_DECK_SIDE_START),
    ("_OPP_DECK_MAIN_START",  _OPP_DECK_MAIN_START),
    ("_OPP_DECK_SIDE_START",  _OPP_DECK_SIDE_START),
    ("_MANA_DEV_START",       _MANA_DEV_START),
    ("_MANA_DEV_END",         _MANA_DEV_END),
    ("_LOG_VITALS_START",     _LOG_VITALS_START),
    ("_LOG_VITALS_END",       _LOG_VITALS_END),
    ("_PER_TURN_START",       _PER_TURN_START),
    ("_PER_TURN_END",         _PER_TURN_END),
    ("_DELAYED_START",        _DELAYED_START),
    ("_DELAYED_END",          _DELAYED_END),
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

    Encoders over the slot formats:
      perm_encoder   (40 status/counter/ref/keyword floats + chosen-name embed +
                     returnable-exile embed + card_embed → embed_dim): permanents
      stack_encoder  (33 scalars incl. a stack-position float + card_embed +
                     target-embed mean → embed_dim//2): stack items
      delayed_encoder (10 scalars (ctrl, state, stack/creator/subject refs,
                     fire_on one-hot, fires_this_turn) + creator embed + subject
                     embed → embed_dim//2): pending delayed triggers
      entity_encoder (card_embed + draw-distance float → embed_dim): graveyard,
                     exile, known opp hand (distance 0), and the combined
                     hand + known-top-library block (top slot i at (i+1)/5)
      ref_combiner   (embed + 4*embed → embed): each permanent's 4 entity refs
                     (attached_to/by, attack/block target) gathered from the
                     encoded entity table and combined into its embedding
      stk_combiner   (embed//2 + embed → embed//2): same one-hop resolution for a
                     stack item's announced-target slot_refs
      entity_attn    one self-attention layer (MultiheadAttention, _ATTN_HEADS)
                     over the ref-combined 108-row entity table (residual add;
                     absent rows masked from keys and re-zeroed after), so every
                     entity's final embedding can pull from any other (all-pairs
                     combat structure). Pooling and the per-action gather read
                     the ATTENDED rows.

    Empty slots (id sentinel) are masked out of every pooled block so they
    neither dilute the mean nor pin the max.

    Output fed into the policy MLP head:
      global(36, seat flag zeroed — network_global_ctx) +
      meta_ctx(7) + board_counts(14: per-side permanent/creature/untapped/
                attacker/blocker counts + total P/T — magnitudes the
                count-invariant pooling cannot express) +
      pending_feat(card_emb+1: what's asking for the current choice) +
      extras(27 raw: lands played, monarch, ..., pass flags + is_priority_window,
             mulligan state, MandatoryChoice one-hot, sideboard context) +
      mana_dev(19 raw: per-color untapped-source potential, total, lands in play /
                in hand, land drops remaining — self then opponent) +
      log_vitals(4 raw: log1p-scaled life and library size, self then opponent) +
      per_turn(22 raw: spells / noncreature / instant-sorcery cast, cards drawn,
               life gained / lost, spell-color multi-hot — self then opponent) +
      [action_extras raw — ONLY when per_action_head=False (the stock head has no
       other action channel); with the per-action head the metadata feeds the
       per-action encoder instead and never enters base raw] +
      arch_onehot(2*N_ARCH raw: one-hot self archetype | one-hot opp archetype;
                  the tail's raw bucket index is stripped and stashed as
                  ``self.last_bucket`` for the multi-head critic) +
      perm_agg(embed*2: masked mean+max over ATTENDED rows) +
      stack_agg(embed*2: attended rows are full-width) +
      top_stack_feat(embed: the attended top-of-stack row, positional) +
      delayed_agg(embed: masked mean+max of the embed//2 delayed-trigger rows) +
      graveyard_agg(embed*2) + exile_agg(embed*2) +
      hand_lib_agg(embed*2: hand + known top-library combined, draw-distance
                   distinguished) +
      next_draw_feat(card_emb: the top-of-library slot's raw card embed,
                     positional — the sharp "what do I draw next" signal) +
      opp_known_hand_agg(embed*2) +
      self_live_lib_agg + self_deck_main/side_agg + opp_deck_main/side_agg
        (each embed*2: masked mean+max over a deck-identity block; the opp
         blocks' slots also carry the match-scoped revealed bit)

    The raw per-action metadata is deliberately NOT in base (except on the stock
    head, which has no other action channel): index-valued floats (vocab ids,
    category enums, slot refs) carry no learnable structure for a dense layer and
    are the obs's best game-fingerprint memorization channel — everything the
    trunk should know from them arrives embedded (the per-action encoder).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        embed_dim: int = 64,
        card_embed_dim: int = _CARD_EMBED_DIM,
        per_action_head: bool = False,
    ):
        half = embed_dim // 2
        # The full per-card vector every _embed_ids lookup returns: the trainable
        # identity embedding concatenated with the FROZEN printed-property block
        # (card_props). Every consumer below sizes against this, not the identity
        # width alone.
        card_feat = card_embed_dim + N_CARD_PROPS
        _meta_ctx_size = _KNOWN_TOP_LIB_START - _MATCH_CTX_START  # 7 (match+lib+turn)
        base_features_dim = (
            _GLOBAL_SIZE                                 # 36
            + _meta_ctx_size                             # 7 match + lib + turn
            + _BOARD_COUNT_FEATS                         # per-side board counts / P-T sums
            + card_feat + 1                              # pending-decision source embed + ctrl flag
            + _EXTRAS_SIZE                               # 27 global extras (raw passthrough)
            + _MANA_DEV_SIZE                             # 19 mana development (raw passthrough)
            + _LOG_VITALS_SIZE                           # 4 log-scaled vitals (raw passthrough)
            + _PER_TURN_SIZE                             # 22 per-turn counters (raw passthrough)
            # Raw action metadata: stock-head fallback ONLY (that path has no
            # other action channel). With the per-action head the same blocks
            # feed the per-action encoder and never enter base raw.
            + (0 if per_action_head else
               observation_space.shape[0] - _STATE_END - _MATCHUP_TAIL_FEATS)
            + _ARCH_ONEHOT_FEATS                         # self/opp archetype one-hots
                                                         # (the raw bucket float is
                                                         # stripped, see forward)
            + embed_dim * 2                              # perm masked mean+max (attended rows)
            + embed_dim * 2                              # stack mean+max (attended, full width)
            + embed_dim                                  # top-of-stack attended row (positional)
            + embed_dim                                  # delayed triggers mean+max (half width)
            + embed_dim * 2                              # graveyard masked-mean + max
            + embed_dim * 2                              # exile masked-mean + max
            + embed_dim * 2                              # hand + known-top-library masked-mean + max
            + card_feat                                  # next-draw (top-lib slot 0) card embed
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
        # real ids 0..N_CARD_TYPES-1 map to rows 1..N_CARD_TYPES. This is the
        # TRAINABLE half of the card vector — it carries only the card-specific
        # behavioral residual the frozen property block below cannot express.
        self.card_emb = nn.Embedding(N_CARD_TYPES + 1, card_embed_dim, padding_idx=0)

        # FROZEN printed-property block (card_props codegen): a registered buffer,
        # not a parameter — never in any optimizer group, never decayed, rides in
        # the state_dict so checkpoints stay self-contained. Row 0 is the zero
        # padding row, mirroring padding_idx above.
        self.register_buffer(
            "card_props",
            torch.from_numpy(np.concatenate(
                [np.zeros((1, N_CARD_PROPS), dtype=np.float32),
                 _CARD_PROP_MATRIX])))

        # Encoder for permanent slots (35 non-id floats + chosen-name embedding +
        # card embedding). Status, counters, is_blocked/is_phased_out, and the
        # keyword multi-hot are scalars; the 4 entity refs enter as raw normalized
        # floats (v1). The chosen-name id (Pithing Needle / Disruptor Flute named
        # card) is embedded via the shared card embedding, like the card id.
        self.perm_encoder = nn.Sequential(
            nn.Linear(_PERM_STATUS_FLOATS + 3 * card_feat, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # Encoder for stack slots: controller_is_self + is_spell, x_or_amount,
        # cast qualifiers, chosen-mode multi-hot, per-target scalar flags
        # (present/is_player/ctrl_is_self/slot_ref × 4 sub-slots), a normalized
        # stack-position float (mean+max pooling would otherwise erase resolution
        # order), the object's card embedding, and the masked mean of its
        # announced targets' card embeddings.  2 + 1 + 7 + 6 + 4*4 + 1 = 33 scalars.
        _stack_scalars = (2 + 1 + _STACK_QUALS + _STACK_MODE_SLOTS
                          + _STACK_TGT_SLOTS * (_STACK_TGT_FIELDS - 1) + 1)
        self.stack_encoder = nn.Sequential(
            nn.Linear(_stack_scalars + 2 * card_feat, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, half),
            nn.ReLU(),
        )

        # Encoder for the pending delayed-trigger slots: the scalar columns (see
        # _DT_SCALARS; refs enter as raw normalized floats), the creator's card
        # embedding and the subject's card embedding.
        self.delayed_encoder = nn.Sequential(
            nn.Linear(_DT_SCALARS + 2 * card_feat, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, half),
            nn.ReLU(),
        )

        # Shared encoder for pure card-identity slots (graveyard, exile, known
        # opponent hand, and the combined hand + known-top-library block). The
        # +1 input is the draw-distance float: 0.0 for a card in hand (and for
        # every other zone this encoder serves), (i+1)/5 for known top-of-library
        # slot i — distinguishing "in hand now" from "drawn i turns from now"
        # and preserving the top-5 ORDER that pooling would otherwise erase.
        self.entity_encoder = nn.Sequential(
            nn.Linear(card_feat + 1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # One-hop entity-reference resolution + one all-pairs attention layer.
        # ref_combiner folds each permanent's 4 gathered ref embeddings
        # (attached_to/by, attack/block target — norm_ref (idx+1)/108, the same
        # unified slot space the per-action refs use) into its own embedding;
        # stk_combiner does the same for a stack item's announced-target
        # slot_refs (masked mean over its 4 sub-slots). entity_attn then runs
        # one self-attention pass over the combined 108-row table (residual),
        # so combat math (each blocker vs each attacker, board-wide effects)
        # is computable per entity before the count-invariant pooling.
        self.ref_combiner = nn.Sequential(
            nn.Linear(embed_dim + _PERM_N_REFS * embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.stk_combiner = nn.Sequential(
            nn.Linear(half + embed_dim, half),
            nn.ReLU(),
        )
        self.entity_attn = nn.MultiheadAttention(
            embed_dim, _ATTN_HEADS, batch_first=True)

        # Shared encoder for the five deck-identity blocks (self live library,
        # self live maindeck + sideboard, opponent registered maindeck +
        # sideboard). The card id is embedded via the shared card embedding and
        # the normalized count + revealed bit appended (revealed = 0 for the self
        # blocks, which carry no such bit), so one encoder covers all five.
        self.decklist_encoder = nn.Sequential(
            nn.Linear(card_feat + 2, embed_dim),
            nn.ReLU(),
        )

        # Action-category embedding for the per-action encoder (when enabled);
        # action slots carry category/ACTION_CATEGORY_MAX.
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
                nn.Linear(_ACTION_CAT_EMBED + card_feat + 1 + _REF_ZONE_EMBED
                          + embed_dim + 1,  # +1 ctrl flag, +1 option_ordinal scalar
                          embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, self.per_action_dim),
                nn.ReLU(),
            )

    def _embed_ids(self, id_floats: torch.Tensor):
        """Map normalized id floats → (card embeddings, present mask).

        id_floats : (..., ) normalized ids (idx/N_CARD_TYPES; -1/N = empty)
        returns   : (emb (..., card_embed_dim + N_CARD_PROPS), present (...,) bool)

        The returned vector is [trainable identity | frozen printed props] —
        both tables are looked up through the same clamped index so the padding
        row (all-zero in both) and out-of-range behavior stay identical.
        """
        idx = torch.round(id_floats * N_CARD_TYPES).long()
        present = idx >= 0
        safe = (idx + 1).clamp(0, N_CARD_TYPES)  # -1 → 0 (padding)
        emb = torch.cat([self.card_emb(safe), self.card_props[safe]], dim=-1)
        return emb, present

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        global_ctx    = network_global_ctx(obs, _GLOBAL_SIZE, _SEAT_FLAG_IDX)
        meta_ctx      = obs[:, _MATCH_CTX_START:_KNOWN_TOP_LIB_START]  # match ctx + library ctx + current turn (7)
        pending       = obs[:, _PENDING_START:_PENDING_END]     # pending-decision source id + ctrl flag
        extras        = obs[:, _EXTRAS_START:_EXTRAS_END]       # 27 global extras (raw passthrough)
        mana_dev      = obs[:, _MANA_DEV_START:_MANA_DEV_END]   # mana development (raw passthrough)
        log_vitals    = obs[:, _LOG_VITALS_START:_LOG_VITALS_END]  # log-scaled life/library (raw)
        per_turn      = obs[:, _PER_TURN_START:_PER_TURN_END]   # per-turn counters (raw)
        # Matchup tail: the raw value-bucket index is STRIPPED here (a bucket id is
        # a meaningless magnitude to the trunk) and stashed for the policy's
        # multi-head critic to gather with; the two archetype one-hots DO feed the
        # trunk as explicit matchup conditioning.
        self.last_bucket = torch.round(obs[:, _BUCKET_IDX]).long().clamp_(
            0, N_VALUE_BUCKETS - 1)
        arch_onehot   = obs[:, _ARCH_ONEHOT_START:_ARCH_ONEHOT_END]

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
            -1, _DECKLIST_MAIN_SLOTS, _OPP_DECKLIST_SLOT_SIZE)
        opp_side  = obs[:, _OPP_DECK_SIDE_START:_OPP_DECK_SIDE_END].reshape(
            -1, _DECKLIST_SIDE_SLOTS, _OPP_DECKLIST_SLOT_SIZE)

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
        # Normalized stack position (0 = top): pooling is order-invariant, so the
        # slot's resolution order must ride inside its own encoding.
        stk_pos = (torch.arange(_STACK_SLOTS, dtype=stack.dtype, device=stack.device)
                   / _STACK_SLOTS).view(1, _STACK_SLOTS, 1).expand(stack.shape[0], -1, -1)
        stk_in = torch.cat([stack[:, :, 0:1], stack[:, :, 2:3], stk_xquals, stk_modes,
                            stk_tgt_scalars, stk_pos, stk_card_emb, stk_tgt_agg], dim=-1)

        # Pending delayed triggers: scalar columns + creator and subject embeds.
        delayed = obs[:, _DELAYED_START:_DELAYED_END].reshape(
            -1, _DELAYED_SLOTS, _DELAYED_SLOT_SIZE)
        dt_creator_emb, _ = self._embed_ids(delayed[:, :, _DT_CREATOR_ID_OFF])
        dt_subject_emb, _ = self._embed_ids(delayed[:, :, _DT_SUBJECT_ID_OFF])
        dt_present = delayed[:, :, _DT_PRESENT_OFF] > 0.5
        dt_in = torch.cat([delayed[:, :, 1:_DT_CREATOR_ID_OFF],
                           delayed[:, :, _DT_CREATOR_ID_OFF + 1:_DT_SUBJECT_ID_OFF],
                           delayed[:, :, _DT_SUBJECT_ID_OFF + 1:],
                           dt_creator_emb, dt_subject_emb], dim=-1)

        gy_emb_in, gy_present = self._embed_ids(graveyard[:, :, 0])
        ex_emb_in, ex_present = self._embed_ids(exile[:, :, 0])
        opp_hand_emb_in, opp_hand_present = self._embed_ids(opp_hand[:, :, 0])
        # Combined hand + known-top-library block: 10 hand slots (draw distance
        # 0.0) then the 5 known top-lib slots (distance (i+1)/5, preserving the
        # top-5 order). Unknown top slots carry the -1 sentinel and mask out like
        # empty hand slots. The other entity_encoder zones feed a 0.0 distance.
        hl_emb, hl_present = self._embed_ids(
            torch.cat([hand[:, :, 0], top_lib[:, :, 0]], dim=1))  # (B, 15, card_feat)
        hl_dist = hl_emb.new_zeros(hl_emb.shape[0],
                                   _HAND_SLOTS + _KNOWN_TOP_LIB_SLOTS, 1)
        hl_dist[:, _HAND_SLOTS:, 0] = (
            torch.arange(1, _KNOWN_TOP_LIB_SLOTS + 1, dtype=hl_emb.dtype,
                         device=hl_emb.device) / _KNOWN_TOP_LIB_SLOTS)
        hl_in = torch.cat([hl_emb, hl_dist], dim=-1)
        # The next-draw positional feature: the top slot's raw card embedding
        # (sharp "what do I draw next" signal, same pattern as pending_feat).
        next_draw_feat = hl_emb[:, _HAND_SLOTS]
        zero_dist = hl_emb.new_zeros(hl_emb.shape[0], 1, 1)
        gy_in = torch.cat([gy_emb_in, zero_dist.expand(-1, _GY_SLOTS, -1)], dim=-1)
        ex_in = torch.cat([ex_emb_in, zero_dist.expand(-1, _EXILE_SLOTS, -1)], dim=-1)
        opp_hand_in = torch.cat(
            [opp_hand_emb_in, zero_dist.expand(-1, _OPP_KNOWN_HAND_SLOTS, -1)], dim=-1)

        # Deck-identity blocks: embed the card id, append the normalized count and
        # the revealed bit (a zero column for the self blocks), encode with the
        # shared decklist_encoder, pool masked mean+max per block.
        self_lib_emb, self_lib_present = self._embed_ids(self_lib[:, :, _DECKLIST_CARD_OFF])
        self_main_emb, self_main_present = self._embed_ids(self_main[:, :, _DECKLIST_CARD_OFF])
        self_side_emb, self_side_present = self._embed_ids(self_side[:, :, _DECKLIST_CARD_OFF])
        opp_main_emb, opp_main_present = self._embed_ids(opp_main[:, :, _DECKLIST_CARD_OFF])
        opp_side_emb, opp_side_present = self._embed_ids(opp_side[:, :, _DECKLIST_CARD_OFF])
        self_lib_in = torch.cat(
            [self_lib_emb, self_lib[:, :, _DECKLIST_COUNT_OFF:_DECKLIST_COUNT_OFF + 1],
             torch.zeros_like(self_lib[:, :, :1])], dim=-1)
        self_main_in = torch.cat(
            [self_main_emb, self_main[:, :, _DECKLIST_COUNT_OFF:_DECKLIST_COUNT_OFF + 1],
             torch.zeros_like(self_main[:, :, :1])], dim=-1)
        self_side_in = torch.cat(
            [self_side_emb, self_side[:, :, _DECKLIST_COUNT_OFF:_DECKLIST_COUNT_OFF + 1],
             torch.zeros_like(self_side[:, :, :1])], dim=-1)
        opp_main_in = torch.cat(
            [opp_main_emb, opp_main[:, :, _DECKLIST_COUNT_OFF:_DECKLIST_REVEALED_OFF + 1]], dim=-1)
        opp_side_in = torch.cat(
            [opp_side_emb, opp_side[:, :, _DECKLIST_COUNT_OFF:_DECKLIST_REVEALED_OFF + 1]], dim=-1)

        # Encode each slot type with its shared-weight encoder. perm/stack are
        # the FIRST pass — their 4 entity refs / announced-target slot_refs still
        # enter as raw floats here and are resolved by the gathers below.
        perm_emb    = self.perm_encoder(perm_in)       # (B, 96, embed)
        stk_emb     = self.stack_encoder(stk_in)       # (B, 12, embed//2)
        dt_emb      = self.delayed_encoder(dt_in)      # (B, 16, embed//2)
        gy_emb      = self.entity_encoder(gy_in)       # (B, 128, embed)
        ex_emb      = self.entity_encoder(ex_in)       # (B, 128, embed)  — shared weights
        hand_lib_emb = self.entity_encoder(hl_in)      # (B, 15, embed)  — shared weights
        opp_hand_emb = self.entity_encoder(opp_hand_in)  # (B, 10, embed)  — shared weights
        self_lib_enc = self.decklist_encoder(self_lib_in)  # (B, 48, embed)  — shared weights
        self_main_enc = self.decklist_encoder(self_main_in)  # (B, 48, embed) — shared weights
        self_side_enc = self.decklist_encoder(self_side_in)  # (B, 15, embed) — shared weights
        opp_main_enc = self.decklist_encoder(opp_main_in)  # (B, 48, embed)  — shared weights
        opp_side_enc = self.decklist_encoder(opp_side_in)  # (B, 15, embed)  — shared weights

        # Per-side board counts: the pooled mean+max below is count-invariant by
        # design (one 3/3 == four 3/3s), so the magnitudes are computed here and
        # handed to the trunk raw. Counts /10 like the engine's count floats;
        # power/toughness floats are already /10 so their sums are used as-is.
        B = perm_emb.shape[0]
        E = self._embed_dim
        m = perm_present.to(perm_emb.dtype)                            # (B, 96)
        is_cre = perms[:, :, _OFF_IS_CREATURE] * m
        untapped_cre = is_cre * (1.0 - perms[:, :, _OFF_IS_TAPPED])
        attacking = perms[:, :, _OFF_IS_ATTACKING] * m
        blocking = perms[:, :, _OFF_IS_BLOCKING] * m
        power = perms[:, :, _OFF_POWER] * is_cre
        tough = perms[:, :, _OFF_TOUGHNESS] * is_cre
        sc = _PERM_SLOTS // 2                                          # self slots 0..sc-1
        board_counts = torch.stack([
            m[:, :sc].sum(1) / 10.0, is_cre[:, :sc].sum(1) / 10.0,
            untapped_cre[:, :sc].sum(1) / 10.0, attacking[:, :sc].sum(1) / 10.0,
            blocking[:, :sc].sum(1) / 10.0,
            power[:, :sc].sum(1), tough[:, :sc].sum(1),
            m[:, sc:].sum(1) / 10.0, is_cre[:, sc:].sum(1) / 10.0,
            untapped_cre[:, sc:].sum(1) / 10.0, attacking[:, sc:].sum(1) / 10.0,
            blocking[:, sc:].sum(1) / 10.0,
            power[:, sc:].sum(1), tough[:, sc:].sum(1),
        ], dim=-1)                                                     # (B, 14)

        # ── Second pass: one-hop reference resolution ───────────────────────
        # Gather table T1 = [perm v1 | zero-padded stack v1 | zeros sentinel]
        # over the unified ref space (0-47 self perm, 48-95 opp perm, 96-107
        # stack, row 108 = "no reference" — the norm_ref (idx+1)/108 encoding
        # decodes identically for perm refs, stack target refs, and action refs).
        stk_v1_pad = torch.cat(
            [stk_emb, stk_emb.new_zeros(B, _STACK_SLOTS, E - stk_emb.shape[-1])],
            dim=-1)
        t1 = torch.cat([perm_emb, stk_v1_pad], dim=1)                  # (B, 108, E)
        t1 = torch.cat([t1, t1.new_zeros(B, 1, E)], dim=1)             # (B, 109, E)

        # Each permanent's 4 refs (attached_to/by, attack/block target) resolved
        # to the referenced entity's ENCODED embedding and combined — so a
        # blocker's embedding can reflect WHAT it blocks, an aura its host.
        perm_refs = perms[:, :, _OFF_ATTACHED_TO:_OFF_ATTACHED_TO + _PERM_N_REFS]
        pref_idx = torch.round(perm_refs * N_ENTITY_REF_SLOTS).long() - 1
        pref_idx = torch.where(pref_idx < 0,
                               pref_idx.new_full((), N_ENTITY_REF_SLOTS),
                               pref_idx).clamp_(0, N_ENTITY_REF_SLOTS)
        pref_e = torch.gather(
            t1, 1, pref_idx.reshape(B, -1).unsqueeze(-1).expand(-1, -1, E)
        ).reshape(B, _PERM_SLOTS, _PERM_N_REFS * E)
        perm_emb2 = self.ref_combiner(torch.cat([perm_emb, pref_e], dim=-1))

        # Same one-hop for each stack item's announced-target slot_refs (masked
        # mean over its 4 sub-slots; player targets carry ref 0.0 → zeros row).
        sref = stk_tgts[:, :, :, _STACK_TGT_FIELDS - 2]
        sref_idx = torch.round(sref * N_ENTITY_REF_SLOTS).long() - 1
        sref_idx = torch.where(sref_idx < 0,
                               sref_idx.new_full((), N_ENTITY_REF_SLOTS),
                               sref_idx).clamp_(0, N_ENTITY_REF_SLOTS)
        sref_e = torch.gather(
            t1, 1, sref_idx.reshape(B, -1).unsqueeze(-1).expand(-1, -1, E)
        ).reshape(B, _STACK_SLOTS, _STACK_TGT_SLOTS, E)
        sref_agg = (sref_e * stk_tgt_mask).sum(2) / stk_tgt_mask.sum(2).clamp(min=1.0)
        stk_emb2 = self.stk_combiner(torch.cat([stk_emb, sref_agg], dim=-1))

        # ── All-pairs pass: one self-attention layer over the entity table ──
        # Residual add; absent rows are masked out of the KEYS and re-zeroed
        # after (a row whose every key is masked — e.g. a fully empty board at a
        # mulligan decision — yields NaN from the masked softmax, and absent
        # rows are exactly the rows discarded here). Pooling and the per-action
        # gather both read the ATTENDED rows.
        stk_v2_pad = torch.cat(
            [stk_emb2, stk_emb2.new_zeros(B, _STACK_SLOTS, E - stk_emb2.shape[-1])],
            dim=-1)
        ent0 = torch.cat([perm_emb2, stk_v2_pad], dim=1)               # (B, 108, E)
        present108 = torch.cat([perm_present, stk_present], dim=1)     # (B, 108)
        attn_out, _ = self.entity_attn(ent0, ent0, ent0,
                                       key_padding_mask=~present108,
                                       need_weights=False)
        ent = torch.where(present108.unsqueeze(-1), ent0 + attn_out,
                          torch.zeros_like(ent0))
        perm_att = ent[:, :_PERM_SLOTS]                                # (B, 96, E)
        stk_att = ent[:, _PERM_SLOTS:]                                 # (B, 12, E)

        # Aggregate: masked mean+max everywhere (skip empty slots — an unmasked
        # mean over the 12 stack slots diluted a lone real spell 1:12 against
        # the empty-slot bias encodings).
        perm_agg    = _masked_mean_max(perm_att, perm_present)
        stk_agg     = _masked_mean_max(stk_att, stk_present)
        top_stack_feat = stk_att[:, 0]        # positional: what resolves NEXT
        delayed_agg = _masked_mean_max(dt_emb, dt_present)
        gy_agg      = _masked_mean_max(gy_emb,   gy_present)
        ex_agg      = _masked_mean_max(ex_emb,   ex_present)
        hand_lib_agg = _masked_mean_max(hand_lib_emb, hl_present)
        opp_hand_agg = _masked_mean_max(opp_hand_emb, opp_hand_present)
        self_lib_agg = _masked_mean_max(self_lib_enc, self_lib_present)
        self_main_agg = _masked_mean_max(self_main_enc, self_main_present)
        self_side_agg = _masked_mean_max(self_side_enc, self_side_present)
        opp_main_agg = _masked_mean_max(opp_main_enc, opp_main_present)
        opp_side_agg = _masked_mean_max(opp_side_enc, opp_side_present)

        parts = [global_ctx, meta_ctx, board_counts,
                 pending_feat, extras, mana_dev, log_vitals, per_turn]
        if not self.per_action_head:
            # Stock-head fallback only: raw action metadata (this path has no
            # per-action encoder, so it is its only action channel).
            parts.append(obs[:, _STATE_END:_BUCKET_IDX])
        parts += [arch_onehot,
                  perm_agg, stk_agg, top_stack_feat, delayed_agg, gy_agg, ex_agg,
                  hand_lib_agg, next_draw_feat, opp_hand_agg,
                  self_lib_agg, self_main_agg, self_side_agg,
                  opp_main_agg, opp_side_agg]
        base = torch.cat(parts, dim=-1)
        if not self.per_action_head:
            return base

        # Encode each candidate action from its own (category, referenced-card,
        # controller, zone_ref, referenced-entity embedding, option ordinal) tuple.
        # The action block is the 6*MAX_ACTIONS floats after the state:
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

        # Gather each action's referenced entity's ATTENDED embedding (the same
        # unified ref space as t1 above; row 108 = zeros "no reference" row).
        ent_table = torch.cat([ent, ent.new_zeros(B, 1, E)], dim=1)    # (B, 109, E)
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
