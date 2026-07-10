"""
Per-entity feature extractor for RoboMage.

Splits the flat observation into sections that are each encoded by a shared-weight
MLP, then aggregated via mean+max pooling.  The policy head receives a fixed-size
representation that is invariant to card ordering and slot position.

State is always from the PRIORITY PLAYER'S perspective ("self").

NOTE: Exile zones are tracked in GameState but not serialized to the observation.
NOTE: ActionChoice.description is never part of the observation — it is for
      human-readable display only (CLI) and is not passed to the ML model.

Card identity is a single normalized id float per slot (idx/N_CARD_TYPES, or
-1/N_CARD_TYPES for empty/unknown), decoded with round(val*N_CARD_TYPES) and
looked up in a learned nn.Embedding. This decouples the observation size from the
vocab size — growing N_CARD_TYPES costs one embedding row, not 252 one-hot slots.

Index layout must stay in sync with src/machine_io.h:
  obs[0:36]            global context (player stats, step, flags, stack size)
  obs[36:3588]         96 permanent slots × 37 floats
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
                         36    card id (LAST)
  obs[3588:4032]       12 stack slots × 37 floats (controller_is_self + card id +
                         is_spell + x_or_amount/10 + 7 cast qualifiers +
                         chosen-mode multi-hot(6) + 4 announced-target sub-slots ×
                         [present, is_player, controller_is_self, slot_ref, card id])
  obs[4032:4160]      128 graveyard slots × 1 float (card id, recency-ordered)
                         slots 0-63: self; slots 64-127: opponent
  obs[4160:4170]       10 hand slots    × 1 float  (card id)
  obs[4170:4682]      128 action history entries × 4 floats (newest first)
                         per entry: category_norm, card_id_norm, is_self, turn/50
  obs[4682:4686]       match context (4 floats: game_number, self_wins, opp_wins, sideboard_phase)
  obs[4686:4689]       library counts & post-board (self_lib/60, opp_lib/60, is_post_board)
  obs[4689]            current game turn / 50
  obs[4690:4695]       5 known top-of-library slots × 1 float (card id, sentinel = unknown)
  obs[4695:5719]       opponent revealed-cards multi-hot (N_CARD_TYPES floats, accumulated across the match)
  obs[5719:5729]       10 known opponent-hand slots × 1 float (card id)
  obs[5729:5731]       pending-decision context (source card id + ctrl_is_self)
  obs[5731:5750]       global extras (self/opp lands played, viewer_has_priority,
                         self/opp monarch, city's blessing, revolt, pending extra
                         turns, is_day, is_night, MandatoryChoice one-hot(6))
  obs[5750:]           action metadata (cats|ids|ctrl|zone|refs) + cost features
                         (appended by env.py; refs are normalized entity-slot
                         references, (idx+1)/108 with 0.0 = none)
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
    from _enums import ACTION_CATEGORY_MAX, N_OBS_KEYWORDS, N_MANDATORY_CHOICES
except ImportError:
    from train._enums import ACTION_CATEGORY_MAX, N_OBS_KEYWORDS, N_MANDATORY_CHOICES


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

_PERM_SLOTS      = 96   # 48 self + 48 opponent (unified: creatures, lands, other)
# 11 status (incl. loyalty) + 2 counters + 4 entity refs + is_blocked +
# is_phased_out + keyword multi-hot + chosen-name id + card id (LAST) = 37
_PERM_SLOT_SIZE  = 19 + N_OBS_KEYWORDS + 2  # 37
_PERM_STATUS_FLOATS   = _PERM_SLOT_SIZE - 2  # 35 non-id status/ref/keyword floats
_PERM_CHOSEN_NAME_OFF = _PERM_SLOT_SIZE - 2  # 35 (chosen-name id, 2nd-last)
_PERM_CARD_OFF   = _PERM_SLOT_SIZE - 1      # 36 (card id is always LAST)

_STACK_SLOTS      = 12
_STACK_XAMT_OFF   = 3   # x_or_amount / 10 within a stack slot
_STACK_QUALS      = 7   # cast qualifiers (is_copy, kicked, flashback, evoke, ...)
_STACK_MODE_SLOTS = 6   # chosen-mode multi-hot width per stack slot
_STACK_MODE_OFF   = _STACK_XAMT_OFF + 1 + _STACK_QUALS      # 11
_STACK_TGT_SLOTS  = 4   # announced-target sub-slots per stack slot
_STACK_TGT_FIELDS = 5   # present + is_player + ctrl_is_self + slot_ref + card id (LAST)
_STACK_TGT_OFF    = _STACK_MODE_OFF + _STACK_MODE_SLOTS     # 17
# ctrl(1) + card id(1) + is_spell(1) + x_or_amount + qualifiers + modes +
# target sub-slots (37 total)
_STACK_SLOT_SIZE  = _STACK_TGT_OFF + _STACK_TGT_SLOTS * _STACK_TGT_FIELDS

_GY_SLOTS        = 128  # 64 self + 64 opponent
_GY_SLOT_SIZE    = 1    # card id only

_HAND_SLOTS      = 10
_HAND_SLOT_SIZE  = 1    # card id only

_HIST_ENTRIES    = 128  # action history entries (newest first)
_HIST_ENTRY_SIZE = 4    # category_norm, card_id_norm, is_self, turn/50

_KNOWN_TOP_LIB_SLOTS     = 5   # known top-of-library cards
_KNOWN_TOP_LIB_SLOT_SIZE = 1   # card id per slot
_REVEALED_SIZE           = N_CARD_TYPES  # opponent revealed-cards multi-hot (dense, not one-hot-per-slot)
_OPP_KNOWN_HAND_SLOTS    = 10  # known opponent-hand card identities
_OPP_KNOWN_HAND_SLOT_SIZE = 1  # card id per slot

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
    from env import (MAX_ACTIONS as _MAX_ACTIONS, STATE_SIZE as _ENV_STATE_SIZE,
                     _GLOBAL_SIZE as _ENV_GLOBAL_SIZE, N_ENTITY_REF_SLOTS)
except ImportError:
    from train.env import (MAX_ACTIONS as _MAX_ACTIONS, STATE_SIZE as _ENV_STATE_SIZE,
                           _GLOBAL_SIZE as _ENV_GLOBAL_SIZE, N_ENTITY_REF_SLOTS)
_GLOBAL_SIZE = _ENV_GLOBAL_SIZE   # header width (single source of truth: env.py)
try:
    from _enums import REF_ZONE_MAX, N_REF_ZONES
except ImportError:
    from train._enums import REF_ZONE_MAX, N_REF_ZONES
_ACTION_CAT_EMBED  = 8    # learned embedding dim for the action category
_REF_ZONE_EMBED    = 4    # learned embedding dim for the per-action zone_ref
_PER_ACTION_DIM    = 32   # per-action feature width fed to the action scorer
_HIST_RECENT_K     = 16   # most-recent history entries embedded per-entry

_PERM_START  = _GLOBAL_SIZE                                    # 36
_PERM_END    = _PERM_START + _PERM_SLOTS * _PERM_SLOT_SIZE     # 3588
_STACK_START = _PERM_END                                       # 3588
_STACK_END   = _STACK_START + _STACK_SLOTS * _STACK_SLOT_SIZE  # 4032
_GY_START    = _STACK_END                                      # 4032
_GY_END      = _GY_START + _GY_SLOTS * _GY_SLOT_SIZE           # 4160
_HAND_START  = _GY_END                                         # 4160
_HAND_END    = _HAND_START + _HAND_SLOTS * _HAND_SLOT_SIZE     # 4170
_HIST_START  = _HAND_END                                       # 4170
_HIST_END    = _HIST_START + _HIST_ENTRIES * _HIST_ENTRY_SIZE  # 4682
# obs[4682:4686] = match context (4 floats: game_number, self_wins, opp_wins, sideboard_phase)
# obs[4686:4689] = library counts & post-board (self_lib/60, opp_lib/60, is_post_board)
_MATCH_CTX_START      = _HIST_END                              # 4682
_MATCH_CTX_END        = _MATCH_CTX_START + 4                   # 4686 (library ctx start)
_LIBRARY_CTX_END      = _MATCH_CTX_END + 3                     # 4689 (current turn idx)
_CUR_TURN_IDX         = _LIBRARY_CTX_END                       # 4689
_KNOWN_TOP_LIB_START  = _CUR_TURN_IDX + 1                      # 4690
_KNOWN_TOP_LIB_END    = _KNOWN_TOP_LIB_START + _KNOWN_TOP_LIB_SLOTS * _KNOWN_TOP_LIB_SLOT_SIZE  # 4695
_REVEALED_START       = _KNOWN_TOP_LIB_END                    # 4695
_REVEALED_END         = _REVEALED_START + _REVEALED_SIZE      # 5719
_OPP_KNOWN_HAND_START = _REVEALED_END                         # 5719
_OPP_KNOWN_HAND_END   = _OPP_KNOWN_HAND_START + _OPP_KNOWN_HAND_SLOTS * _OPP_KNOWN_HAND_SLOT_SIZE  # 5729
# Pending decision context: card id of the spell/ability currently making a
# mid-resolution choice (sentinel = none) + its controller-is-viewer flag.
_PENDING_START        = _OPP_KNOWN_HAND_END                   # 5729
_PENDING_SIZE         = 2
_PENDING_END          = _PENDING_START + _PENDING_SIZE        # 5731
# Global extras (machine_io.h [5731:5750]): self/opp lands played, priority,
# monarch, city's blessing, revolt, pending extra turns, day/night flags, plus
# the MandatoryChoice one-hot. Cheap scalar facts — passed through raw.
_EXTRAS_START         = _PENDING_END                          # 5731
_EXTRAS_SIZE          = 13 + N_MANDATORY_CHOICES              # 19
_EXTRAS_END           = _EXTRAS_START + _EXTRAS_SIZE          # 5750
_STATE_END            = _EXTRAS_END
# obs[_STATE_END:] = action metadata + cost features appended by env.py
# Guard against the two layout mirrors drifting apart (env.py owns STATE_SIZE).
assert _STATE_END == _ENV_STATE_SIZE, (_STATE_END, _ENV_STATE_SIZE)


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
                     card_embed → embed_dim): permanents (entity refs enter as raw
                     normalized floats in v1)
      stack_encoder  (32 scalars + card_embed + target-embed mean → embed_dim//2):
                     stack items with x/qualifiers and announced modes/targets
      entity_encoder (card_embed → embed_dim): graveyard, hand, known top-library

    Empty slots (id sentinel) are masked out of the perm / stack / graveyard /
    hand pooling so they neither dilute the mean nor pin the max.

    Output fed into the policy MLP head:
      global(36) + hist(512 raw) + hist_recent(K × (cat_emb+card_emb+2) embedded) +
      meta_ctx(8) + known_top_lib_agg(embed) + revealed_agg(embed) +
      pending_feat(card_emb+1: what's asking for the current choice) +
      extras(19 raw: lands played, priority, monarch, ..., MandatoryChoice one-hot) +
      action_extras(action metadata cats|ids|ctrl|zone|refs + cost feats) +
      perm_agg(embed*2: masked mean+max) + stack_agg(embed//2 * 2) +
      graveyard_agg(embed*2: masked mean+max) + hand_agg(embed*2: masked mean+max) +
      opp_known_hand_agg(embed*2: masked mean+max)
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
            + _EXTRAS_SIZE                               # 19 global extras (raw passthrough)
            + (observation_space.shape[0] - _STATE_END)  # action extras (5 blocks incl. refs + costs)
            + embed_dim * 2                              # perm masked mean+max (creatures, lands, other)
            + half * 2                                   # stack mean+max
            + embed_dim * 2                              # graveyard masked-mean + max
            + embed_dim * 2                              # hand masked-mean + max
            + embed_dim * 2                              # known opponent-hand masked-mean + max
        )
        # When the per-action logit head is enabled, the encoded per-action tensor
        # (MAX_ACTIONS × _PER_ACTION_DIM) is appended at the END of the returned
        # features so PerActionMaskablePolicy can slice it back out by offset.
        self.per_action_head = per_action_head
        if per_action_head:
            self.per_action_slots = _MAX_ACTIONS
            self.per_action_dim = _PER_ACTION_DIM
            self.per_action_offset = base_features_dim
            features_dim = base_features_dim + self.per_action_slots * self.per_action_dim
        else:
            features_dim = base_features_dim
        super().__init__(observation_space, features_dim=features_dim)

        # Shared card-identity embedding. Slot id -1 (empty) maps to padding row 0;
        # real ids 0..N_CARD_TYPES-1 map to rows 1..N_CARD_TYPES.
        self.card_emb = nn.Embedding(N_CARD_TYPES + 1, card_embed_dim, padding_idx=0)

        # Encoder for permanent slots (35 non-id floats + chosen-name embedding +
        # card embedding). Status, counters, is_blocked/is_phased_out, and the
        # keyword multi-hot are scalars; the 4 entity refs enter as raw normalized
        # floats (v1). The chosen-name id (Pithing Needle / Disruptor Flute named
        # card) is embedded via the shared card embedding, like the card id.
        self.perm_encoder = nn.Sequential(
            nn.Linear(_PERM_STATUS_FLOATS + 2 * card_embed_dim, embed_dim),
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
                          + embed_dim,
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
        action_extras = obs[:, _STATE_END:]                     # action cats|ids|ctrl|zone|refs + cost features

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
        hand      = obs[:, _HAND_START:_HAND_END].reshape(-1, _HAND_SLOTS, _HAND_SLOT_SIZE)
        opp_hand  = obs[:, _OPP_KNOWN_HAND_START:_OPP_KNOWN_HAND_END].reshape(
            -1, _OPP_KNOWN_HAND_SLOTS, _OPP_KNOWN_HAND_SLOT_SIZE)
        top_lib   = obs[:, _KNOWN_TOP_LIB_START:_KNOWN_TOP_LIB_END].reshape(
            -1, _KNOWN_TOP_LIB_SLOTS, _KNOWN_TOP_LIB_SLOT_SIZE)

        # Embed card identity per slot, then build each slot's encoder input. The
        # chosen-name id is embedded through the same shared card embedding as the
        # card id and appended after the 35 status floats.
        perm_card_emb, perm_present = self._embed_ids(perms[:, :, _PERM_CARD_OFF])
        perm_chosen_emb, _ = self._embed_ids(perms[:, :, _PERM_CHOSEN_NAME_OFF])
        perm_in = torch.cat([perms[:, :, :_PERM_STATUS_FLOATS],
                             perm_chosen_emb, perm_card_emb], dim=-1)

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
        hand_emb_in, hand_present = self._embed_ids(hand[:, :, 0])
        opp_hand_emb_in, opp_hand_present = self._embed_ids(opp_hand[:, :, 0])
        top_lib_emb_in, _ = self._embed_ids(top_lib[:, :, 0])

        # Encode each slot type with its shared-weight encoder
        perm_emb    = self.perm_encoder(perm_in)       # (B, 96, embed)
        stk_emb     = self.stack_encoder(stk_in)       # (B, 12, embed//2)
        gy_emb      = self.entity_encoder(gy_emb_in)   # (B, 128, embed)
        hand_emb    = self.entity_encoder(hand_emb_in) # (B, 10, embed)  — shared weights
        opp_hand_emb = self.entity_encoder(opp_hand_emb_in)  # (B, 10, embed)  — shared weights
        top_lib_emb = self.entity_encoder(top_lib_emb_in)  # (B, 5, embed)  — shared weights

        # Aggregate: masked mean+max for perms, stack, graveyard, and hand (skip
        # empty slots — an unmasked mean over the 12 stack slots diluted a lone
        # real spell 1:12 against the empty-slot bias encodings); mean for
        # top-library.
        perm_agg    = _masked_mean_max(perm_emb, perm_present)
        stk_agg     = _masked_mean_max(stk_emb, stk_present)
        gy_agg      = _masked_mean_max(gy_emb,   gy_present)
        hand_agg    = _masked_mean_max(hand_emb, hand_present)
        opp_hand_agg = _masked_mean_max(opp_hand_emb, opp_hand_present)
        top_lib_agg = top_lib_emb.mean(1)
        revealed_agg = self.revealed_encoder(revealed)  # (B, embed) dense multi-hot encoding

        base = torch.cat([global_ctx, hist_ctx, hist_recent, meta_ctx, top_lib_agg,
                          revealed_agg, pending_feat, extras, action_extras,
                          perm_agg, stk_agg, gy_agg, hand_agg, opp_hand_agg], dim=-1)
        if not self.per_action_head:
            return base

        # Encode each candidate action from its own (category, referenced-card,
        # controller, zone_ref, referenced-entity embedding) tuple. The action
        # block is the first 5*MAX_ACTIONS floats of action_extras:
        # cats | ids | ctrl | zone | refs. Appended flat; sliced back out by the
        # policy's action scorer. Padded slots (beyond num_choices) are harmless —
        # their logits are masked out by MaskablePPO's action mask downstream.
        a0 = _STATE_END
        cats = obs[:, a0:a0 + _MAX_ACTIONS]
        act_ids = obs[:, a0 + _MAX_ACTIONS:a0 + 2 * _MAX_ACTIONS]
        ctrl = obs[:, a0 + 2 * _MAX_ACTIONS:a0 + 3 * _MAX_ACTIONS]
        zone = obs[:, a0 + 3 * _MAX_ACTIONS:a0 + 4 * _MAX_ACTIONS]
        refs = obs[:, a0 + 4 * _MAX_ACTIONS:a0 + 5 * _MAX_ACTIONS]
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

        pa_in = torch.cat([cat_e, act_id_e, ctrl.unsqueeze(-1), zone_e, ref_e], dim=-1)
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

    def _dist_from(self, features, latent_pi, action_masks):
        logits = self.action_scorer(latent_pi, self._slice_per_action(features))
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def forward(self, obs, deterministic=False, action_masks=None):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._dist_from(features, latent_pi, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions, action_masks=None):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._dist_from(features, latent_pi, action_masks)
        values = self.value_net(latent_vf)
        return values, distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs, action_masks=None):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._dist_from(features, latent_pi, action_masks)
