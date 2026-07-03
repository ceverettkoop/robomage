"""
Per-entity feature extractor for RoboMage.

Splits the flat observation into sections that are each encoded by a shared-weight
MLP, then aggregated via mean+max pooling.  The policy head receives a fixed-size
representation that is invariant to card ordering and slot position.

State is always from the PRIORITY PLAYER'S perspective ("self").

NOTE: Exile zones are tracked in GameState but not serialized to the observation.
NOTE: ActionChoice.description is never part of the observation — it is for
      human-readable display only (GUI/CLI) and is not passed to the ML model.

Card identity is a single normalized id float per slot (idx/N_CARD_TYPES, or
-1/N_CARD_TYPES for empty/unknown), decoded with round(val*N_CARD_TYPES) and
looked up in a learned nn.Embedding. This decouples the observation size from the
vocab size — growing N_CARD_TYPES costs one embedding row, not 252 one-hot slots.

Index layout must stay in sync with src/machine_io.h:
  obs[0:34]            global context (player stats, step, flags, stack size)
  obs[34:1186]         96 permanent slots × 12 floats  (11 status + 1 card id)
                         slots 0-47: self; slots 48-95: opponent
                         status: power, toughness, tapped, attacking, blocking,
                                 sickness, damage, controller_is_self, is_creature, is_land, loyalty
  obs[1186:1486]       12 stack slots   × 25 floats (controller_is_self + card id + is_spell +
                         chosen-mode multi-hot(6) + 4 announced-target sub-slots ×
                         [present, is_player, controller_is_self, card id])
  obs[1486:1614]      128 graveyard slots × 1 float (card id)
                         slots 0-63: self; slots 64-127: opponent
  obs[1614:1624]       10 hand slots    × 1 float  (card id)
  obs[1624:2136]      128 action history entries × 4 floats (newest first)
                         per entry: category_norm, card_id_norm, is_self, turn/50
  obs[2136:2140]       match context (4 floats: game_number, self_wins, opp_wins, sideboard_phase)
  obs[2140:2143]       library counts & post-board (self_lib/60, opp_lib/60, is_post_board)
  obs[2143]            current game turn / 50
  obs[2144:2149]       5 known top-of-library slots × 1 float (card id, sentinel = unknown)
  obs[2149:3173]       opponent revealed-cards multi-hot (N_CARD_TYPES floats, accumulated across the match)
  obs[3173:3183]       10 known opponent-hand slots × 1 float (card id)
  obs[3183:]           action metadata + cost features (appended by env.py)
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from card_costs import N_CARD_TYPES
except ImportError:
    from train.card_costs import N_CARD_TYPES


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
_GLOBAL_SIZE     = 34

_PERM_SLOTS      = 96   # 48 self + 48 opponent (unified: creatures, lands, other)
_PERM_SLOT_SIZE  = 12   # 11 status floats (incl. loyalty) + 1 card id
_PERM_CARD_OFF   = 11   # card id follows the 11 status floats

_STACK_SLOTS      = 12
_STACK_MODE_SLOTS = 6   # chosen-mode multi-hot width per stack slot
_STACK_TGT_SLOTS  = 4   # announced-target sub-slots per stack slot
_STACK_TGT_FIELDS = 4   # present + is_player + ctrl_is_self + card id
_STACK_TGT_OFF    = 3 + _STACK_MODE_SLOTS  # target sub-slots follow ctrl/id/is_spell + modes
# controller_is_self(1) + card id(1) + is_spell(1) + modes + target sub-slots (25)
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

_PERM_START  = _GLOBAL_SIZE                                    # 34
_PERM_END    = _PERM_START + _PERM_SLOTS * _PERM_SLOT_SIZE     # 1186
_STACK_START = _PERM_END                                       # 1186
_STACK_END   = _STACK_START + _STACK_SLOTS * _STACK_SLOT_SIZE  # 1486
_GY_START    = _STACK_END                                      # 1486
_GY_END      = _GY_START + _GY_SLOTS * _GY_SLOT_SIZE           # 1614
_HAND_START  = _GY_END                                         # 1614
_HAND_END    = _HAND_START + _HAND_SLOTS * _HAND_SLOT_SIZE     # 1624
_HIST_START  = _HAND_END                                       # 1624
_HIST_END    = _HIST_START + _HIST_ENTRIES * _HIST_ENTRY_SIZE  # 2136
# obs[2136:2140] = match context (4 floats: game_number, self_wins, opp_wins, sideboard_phase)
# obs[2140:2143] = library counts & post-board (self_lib/60, opp_lib/60, is_post_board)
_MATCH_CTX_START      = _HIST_END                              # 2136
_MATCH_CTX_END        = _MATCH_CTX_START + 4                   # 2140 (library ctx start)
_LIBRARY_CTX_END      = _MATCH_CTX_END + 3                     # 2143 (current turn idx)
_CUR_TURN_IDX         = _LIBRARY_CTX_END                       # 2143
_KNOWN_TOP_LIB_START  = _CUR_TURN_IDX + 1                      # 2144
_KNOWN_TOP_LIB_END    = _KNOWN_TOP_LIB_START + _KNOWN_TOP_LIB_SLOTS * _KNOWN_TOP_LIB_SLOT_SIZE  # 2149
_REVEALED_START       = _KNOWN_TOP_LIB_END                    # 2149
_REVEALED_END         = _REVEALED_START + _REVEALED_SIZE      # 3173
_OPP_KNOWN_HAND_START = _REVEALED_END
_OPP_KNOWN_HAND_END   = _OPP_KNOWN_HAND_START + _OPP_KNOWN_HAND_SLOTS * _OPP_KNOWN_HAND_SLOT_SIZE
_STATE_END            = _OPP_KNOWN_HAND_END
# obs[_STATE_END:] = action metadata + cost features appended by env.py


class CardGameExtractor(BaseFeaturesExtractor):
    """
    Shared-weight per-entity encoder with masked mean+max aggregation.

    Card identity is a single normalized id float per slot; it is decoded to a
    vocab index and looked up in a shared nn.Embedding (padding_idx 0 = empty),
    then concatenated with that slot's status floats before encoding. Decoupling
    card identity from a one-hot makes the observation cost independent of vocab
    size.

    Three independent encoders cover the slot formats:
      perm_encoder   (11 status + card_embed → embed_dim): permanents
      stack_encoder  (20 scalars + card_embed + target-embed mean → embed_dim//2):
                     stack items with their announced modes/targets
      entity_encoder (card_embed → embed_dim): graveyard, hand, known top-library

    Empty slots (id sentinel) are masked out of the perm / graveyard / hand
    pooling so they neither dilute the mean nor pin the max.

    Output fed into the policy MLP head:
      global(34) + hist(512) + meta_ctx(8) + known_top_lib_agg(embed) +
      revealed_agg(embed) +
      action_extras(match+lib+turn skipped; action metadata + cost feats) +
      perm_agg(embed*2: masked mean+max) + stack_agg(embed//2 * 2) +
      graveyard_agg(embed*2: masked mean+max) + hand_agg(embed*2: masked mean+max) +
      opp_known_hand_agg(embed*2: masked mean+max)
    """

    def __init__(
        self,
        observation_space: gym.Space,
        embed_dim: int = 64,
        card_embed_dim: int = _CARD_EMBED_DIM,
    ):
        half = embed_dim // 2
        _hist_size = _HIST_ENTRIES * _HIST_ENTRY_SIZE     # 512
        _meta_ctx_size = _KNOWN_TOP_LIB_START - _HIST_END  # 8 (match+lib+turn)
        features_dim = (
            _GLOBAL_SIZE                                 # 34
            + _hist_size                                 # 512 action history
            + _meta_ctx_size                             # 8 match + lib + turn
            + embed_dim                                  # known-top library mean
            + embed_dim                                  # opponent revealed-cards multi-hot
            + (observation_space.shape[0] - _STATE_END)  # action extras
            + embed_dim * 2                              # perm masked mean+max (creatures, lands, other)
            + half * 2                                   # stack mean+max
            + embed_dim * 2                              # graveyard masked-mean + max
            + embed_dim * 2                              # hand masked-mean + max
            + embed_dim * 2                              # known opponent-hand masked-mean + max
        )
        super().__init__(observation_space, features_dim=features_dim)

        # Shared card-identity embedding. Slot id -1 (empty) maps to padding row 0;
        # real ids 0..N_CARD_TYPES-1 map to rows 1..N_CARD_TYPES.
        self.card_emb = nn.Embedding(N_CARD_TYPES + 1, card_embed_dim, padding_idx=0)

        # Encoder for permanent slots (11 status floats + card embedding)
        self.perm_encoder = nn.Sequential(
            nn.Linear(_PERM_CARD_OFF + card_embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # Encoder for stack slots: controller_is_self + is_spell, chosen-mode multi-hot,
        # per-target scalar flags (present/is_player/ctrl_is_self × 4 sub-slots), the
        # object's card embedding, and the masked mean of its announced targets' card
        # embeddings.
        _stack_scalars = 2 + _STACK_MODE_SLOTS + _STACK_TGT_SLOTS * (_STACK_TGT_FIELDS - 1)
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
        action_extras = obs[:, _STATE_END:]                     # action cats + card IDs + cost features

        perms     = obs[:, _PERM_START:_PERM_END].reshape(-1, _PERM_SLOTS, _PERM_SLOT_SIZE)
        stack     = obs[:, _STACK_START:_STACK_END].reshape(-1, _STACK_SLOTS, _STACK_SLOT_SIZE)
        graveyard = obs[:, _GY_START:_GY_END].reshape(-1, _GY_SLOTS, _GY_SLOT_SIZE)
        hand      = obs[:, _HAND_START:_HAND_END].reshape(-1, _HAND_SLOTS, _HAND_SLOT_SIZE)
        opp_hand  = obs[:, _OPP_KNOWN_HAND_START:_OPP_KNOWN_HAND_END].reshape(
            -1, _OPP_KNOWN_HAND_SLOTS, _OPP_KNOWN_HAND_SLOT_SIZE)
        top_lib   = obs[:, _KNOWN_TOP_LIB_START:_KNOWN_TOP_LIB_END].reshape(
            -1, _KNOWN_TOP_LIB_SLOTS, _KNOWN_TOP_LIB_SLOT_SIZE)

        # Embed card identity per slot, then build each slot's encoder input.
        perm_card_emb, perm_present = self._embed_ids(perms[:, :, _PERM_CARD_OFF])
        perm_in = torch.cat([perms[:, :, :_PERM_CARD_OFF], perm_card_emb], dim=-1)

        stk_card_emb, _ = self._embed_ids(stack[:, :, 1])
        # Announced-target sub-slots: (B, 12, 4, 4) of [present, is_player, ctrl, card id].
        stk_tgts = stack[:, :, _STACK_TGT_OFF:].reshape(
            -1, _STACK_SLOTS, _STACK_TGT_SLOTS, _STACK_TGT_FIELDS)
        stk_tgt_emb, _ = self._embed_ids(stk_tgts[:, :, :, 3])       # (B, 12, 4, card_embed)
        stk_tgt_mask = stk_tgts[:, :, :, 0:1]                        # present flag
        stk_tgt_agg = (stk_tgt_emb * stk_tgt_mask).sum(2) / stk_tgt_mask.sum(2).clamp(min=1.0)
        stk_modes = stack[:, :, 3:_STACK_TGT_OFF]                    # chosen-mode multi-hot
        stk_tgt_scalars = stk_tgts[:, :, :, :3].reshape(-1, _STACK_SLOTS, _STACK_TGT_SLOTS * 3)
        stk_in = torch.cat([stack[:, :, 0:1], stack[:, :, 2:3], stk_modes, stk_tgt_scalars,
                            stk_card_emb, stk_tgt_agg], dim=-1)

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

        # Aggregate: masked mean+max for perms, graveyard, and hand (skip empty
        # slots); mean+max for stack; mean for top-library.
        perm_agg    = _masked_mean_max(perm_emb, perm_present)
        stk_agg     = torch.cat([stk_emb.mean(1),  stk_emb.max(1).values],  dim=-1)
        gy_agg      = _masked_mean_max(gy_emb,   gy_present)
        hand_agg    = _masked_mean_max(hand_emb, hand_present)
        opp_hand_agg = _masked_mean_max(opp_hand_emb, opp_hand_present)
        top_lib_agg = top_lib_emb.mean(1)
        revealed_agg = self.revealed_encoder(revealed)  # (B, embed) dense multi-hot encoding

        return torch.cat([global_ctx, hist_ctx, meta_ctx, top_lib_agg, revealed_agg, action_extras,
                          perm_agg, stk_agg, gy_agg, hand_agg, opp_hand_agg], dim=-1)
