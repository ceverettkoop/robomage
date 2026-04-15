"""
Per-entity feature extractor for RoboMage.

Splits the flat observation into sections that are each encoded by a shared-weight
MLP, then aggregated via mean+max pooling.  The policy head receives a fixed-size
representation that is invariant to card ordering and slot position.

State is always from the PRIORITY PLAYER'S perspective ("self").

NOTE: Exile zones are tracked in GameState but not serialized to the observation.
NOTE: ActionChoice.description is never part of the observation — it is for
      human-readable display only (GUI/CLI) and is not passed to the ML model.

Index layout must stay in sync with src/machine_io.h:
  obs[0:34]            global context (player stats, step, flags, stack size)
  obs[34:13282]        96 permanent slots × 138 floats  (10 status + 128 card one-hot)
                         slots 0-47: self; slots 48-95: opponent
                         status: power, toughness, tapped, attacking, blocking,
                                 sickness, damage, controller_is_self, is_creature, is_land
  obs[13282:14842]     12 stack slots   × 130 floats  (controller_is_self + 128 card one-hot + is_spell)
  obs[14842:31226]    128 graveyard slots × 128 floats (128 card one-hot)
                         slots 0-63: self; slots 64-127: opponent
  obs[31226:32506]     10 hand slots    × 128 floats  (128 card one-hot)
  obs[32506:33018]    128 action history entries × 4 floats (newest first)
                         per entry: category_norm, card_id_norm, is_self, turn/50
  obs[33018:33022]     match context (4 floats: game_number, self_wins, opp_wins, sideboard_phase)
  obs[33022:33025]     library counts & post-board (self_lib/60, opp_lib/60, is_post_board)
  obs[33025]           current game turn / 50
  obs[33026:33666]     5 known top-of-library slots × 128 floats (card one-hot, all zeros = unknown)
  obs[33666:]          action metadata + cost features (appended by env.py)
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ── Layout constants (mirror src/machine_io.h) ──────────────────────────────
_GLOBAL_SIZE     = 34

_PERM_SLOTS      = 96   # 48 self + 48 opponent (unified: creatures, lands, other)
_PERM_SLOT_SIZE  = 138  # 10 status floats + 128 card one-hot

_STACK_SLOTS     = 12
_STACK_SLOT_SIZE = 130  # controller_is_self(1) + card one-hot(128) + is_spell(1)

_GY_SLOTS        = 128  # 64 self + 64 opponent
_GY_SLOT_SIZE    = 128  # card one-hot only

_HAND_SLOTS      = 10
_HAND_SLOT_SIZE  = 128  # card one-hot only

_HIST_ENTRIES    = 128  # action history entries (newest first)
_HIST_ENTRY_SIZE = 4    # category_norm, card_id_norm, is_self, turn/50

_KNOWN_TOP_LIB_SLOTS     = 5   # known top-of-library cards
_KNOWN_TOP_LIB_SLOT_SIZE = 128 # card one-hot per slot

_PERM_START  = _GLOBAL_SIZE                                    # 34
_PERM_END    = _PERM_START + _PERM_SLOTS * _PERM_SLOT_SIZE     # 13282
_STACK_START = _PERM_END                                       # 13282
_STACK_END   = _STACK_START + _STACK_SLOTS * _STACK_SLOT_SIZE  # 14842
_GY_START    = _STACK_END                                      # 14842
_GY_END      = _GY_START + _GY_SLOTS * _GY_SLOT_SIZE           # 31226
_HAND_START  = _GY_END                                         # 31226
_HAND_END    = _HAND_START + _HAND_SLOTS * _HAND_SLOT_SIZE     # 32506
_HIST_START  = _HAND_END                                       # 32506
_HIST_END    = _HIST_START + _HIST_ENTRIES * _HIST_ENTRY_SIZE  # 33018
# obs[33018:33022] = match context (4 floats: game_number, self_wins, opp_wins, sideboard_phase)
# obs[33022:33025] = library counts & post-board (self_lib/60, opp_lib/60, is_post_board)
_MATCH_CTX_START      = _HIST_END                              # 33018
_MATCH_CTX_END        = _MATCH_CTX_START + 4                   # 33022 (library ctx start)
_LIBRARY_CTX_END      = _MATCH_CTX_END + 3                     # 33025 (current turn idx)
_CUR_TURN_IDX         = _LIBRARY_CTX_END                       # 33025
_KNOWN_TOP_LIB_START  = _CUR_TURN_IDX + 1                      # 33026
_KNOWN_TOP_LIB_END    = _KNOWN_TOP_LIB_START + _KNOWN_TOP_LIB_SLOTS * _KNOWN_TOP_LIB_SLOT_SIZE  # 33666
_STATE_END            = _KNOWN_TOP_LIB_END                     # 33666
# obs[33666:] = action metadata + cost features appended by env.py


class CardGameExtractor(BaseFeaturesExtractor):
    """
    Shared-weight per-entity encoder with mean+max aggregation.

    Four independent encoders cover the four slot formats:
      perm_encoder   (138 → embed_dim): permanents (10 status + 128 card one-hot)
      stack_encoder  (130 → embed_dim//2): stack items
      entity_encoder (128 → embed_dim): graveyard, hand, and known top-library

    Output fed into the policy MLP head:
      global(34) + hist(512) + meta_ctx(8) + known_top_lib_agg(embed) +
      action_extras(match+lib+turn skipped; action metadata + cost feats) +
      perm_agg(embed*2) + stack_agg(embed//2 * 2) +
      graveyard_agg(embed) + hand_agg(embed)
    """

    def __init__(
        self,
        observation_space: gym.Space,
        embed_dim: int = 64,
    ):
        half = embed_dim // 2
        _hist_size = _HIST_ENTRIES * _HIST_ENTRY_SIZE     # 512
        _meta_ctx_size = _KNOWN_TOP_LIB_START - _HIST_END  # 8 (match+lib+turn)
        features_dim = (
            _GLOBAL_SIZE                                 # 34
            + _hist_size                                 # 512 action history
            + _meta_ctx_size                             # 8 match + lib + turn
            + embed_dim                                  # known-top library mean
            + (observation_space.shape[0] - _STATE_END)  # action extras
            + embed_dim * 2                              # perm mean+max (creatures, lands, other)
            + half * 2                                   # stack mean+max
            + embed_dim                                  # graveyard mean
            + embed_dim                                  # hand mean
        )
        super().__init__(observation_space, features_dim=features_dim)

        # Shared encoder for 42-float unified permanent slots
        self.perm_encoder = nn.Sequential(
            nn.Linear(_PERM_SLOT_SIZE, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # Encoder for 33-float stack slots
        self.stack_encoder = nn.Sequential(
            nn.Linear(_STACK_SLOT_SIZE, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, half),
            nn.ReLU(),
        )

        # Shared encoder for 32-float card-identity slots (graveyard AND hand)
        self.entity_encoder = nn.Sequential(
            nn.Linear(_HAND_SLOT_SIZE, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        global_ctx    = obs[:, :_GLOBAL_SIZE]
        hist_ctx      = obs[:, _HIST_START:_HIST_END]           # action history (128 × 4)
        meta_ctx      = obs[:, _HIST_END:_KNOWN_TOP_LIB_START]  # match ctx + library ctx + current turn (8)
        action_extras = obs[:, _STATE_END:]                     # action cats + card IDs + cost features

        perms     = obs[:, _PERM_START:_PERM_END].reshape(-1, _PERM_SLOTS, _PERM_SLOT_SIZE)
        stack     = obs[:, _STACK_START:_STACK_END].reshape(-1, _STACK_SLOTS, _STACK_SLOT_SIZE)
        graveyard = obs[:, _GY_START:_GY_END].reshape(-1, _GY_SLOTS, _GY_SLOT_SIZE)
        hand      = obs[:, _HAND_START:_HAND_END].reshape(-1, _HAND_SLOTS, _HAND_SLOT_SIZE)
        top_lib   = obs[:, _KNOWN_TOP_LIB_START:_KNOWN_TOP_LIB_END].reshape(
            -1, _KNOWN_TOP_LIB_SLOTS, _KNOWN_TOP_LIB_SLOT_SIZE)

        # Encode each slot type with its shared-weight encoder
        perm_emb    = self.perm_encoder(perms)        # (B, 96, embed)
        stk_emb     = self.stack_encoder(stack)       # (B, 12, embed//2)
        gy_emb      = self.entity_encoder(graveyard)  # (B, 128, embed)
        hand_emb    = self.entity_encoder(hand)       # (B, 10, embed)  — shared weights
        top_lib_emb = self.entity_encoder(top_lib)    # (B, 5, embed)  — shared weights

        # Aggregate: mean+max for perms and stack; mean for graveyard, hand, top-library
        perm_agg    = torch.cat([perm_emb.mean(1), perm_emb.max(1).values], dim=-1)
        stk_agg     = torch.cat([stk_emb.mean(1),  stk_emb.max(1).values],  dim=-1)
        gy_agg      = gy_emb.mean(1)
        hand_agg    = hand_emb.mean(1)
        top_lib_agg = top_lib_emb.mean(1)

        return torch.cat([global_ctx, hist_ctx, meta_ctx, top_lib_agg, action_extras,
                          perm_agg, stk_agg, gy_agg, hand_agg], dim=-1)
