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
  obs[0:33]          global context (player stats, step, stack)
  obs[33:4065]       96 permanent slots × 42 floats  (10 status + 32 card one-hot)
                       slots 0-47: self; slots 48-95: opponent
                       status: power, toughness, tapped, attacking, blocking,
                               sickness, damage, controller_is_self, is_creature, is_land
  obs[4065:4473]     12 stack slots   × 34 floats  (controller_is_self + 32 card one-hot + is_spell)
  obs[4473:8569]    128 graveyard slots × 32 floats (32 card one-hot)
                       slots 0-63: self; slots 64-127: opponent
  obs[8569:8889]     10 hand slots    × 32 floats  (32 card one-hot)
  obs[8889:8921]     32 action-category features   (appended by env.py)
  obs[8921:8953]     32 action card-ID features    (appended by env.py)
  obs[8953:8985]     32 action controller_is_self  (appended by env.py)
  obs[8985:9055]     70 hand cast-cost features    (10 slots × 7 cost feats)
  obs[9055:9391]    336 BF ability-cost features   (48 slots × 7 cost feats)
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ── Layout constants (mirror src/machine_io.h) ──────────────────────────────
_GLOBAL_SIZE     = 33

_PERM_SLOTS      = 96   # 48 self + 48 opponent (unified: creatures, lands, other)
_PERM_SLOT_SIZE  = 42   # 10 status floats + 32 card one-hot

_STACK_SLOTS     = 12
_STACK_SLOT_SIZE = 34   # controller_is_self(1) + card one-hot(32) + is_spell(1)

_GY_SLOTS        = 128  # 64 self + 64 opponent
_GY_SLOT_SIZE    = 32   # card one-hot only

_HAND_SLOTS      = 10
_HAND_SLOT_SIZE  = 32   # card one-hot only

_PERM_START  = _GLOBAL_SIZE                                    # 33
_PERM_END    = _PERM_START + _PERM_SLOTS * _PERM_SLOT_SIZE     # 4065
_STACK_START = _PERM_END                                       # 4065
_STACK_END   = _STACK_START + _STACK_SLOTS * _STACK_SLOT_SIZE  # 4473
_GY_START    = _STACK_END                                      # 4473
_GY_END      = _GY_START + _GY_SLOTS * _GY_SLOT_SIZE           # 8569
_HAND_START  = _GY_END                                         # 8569
_HAND_END    = _HAND_START + _HAND_SLOTS * _HAND_SLOT_SIZE     # 8889
# obs[8889:] = action metadata + cost features appended by env.py


class CardGameExtractor(BaseFeaturesExtractor):
    """
    Shared-weight per-entity encoder with mean+max aggregation.

    Four independent encoders cover the four slot formats:
      perm_encoder   (40 → embed_dim): creatures and lands (same 40-float format)
      stack_encoder  (33 → embed_dim//2): stack items
      entity_encoder (32 → embed_dim): graveyard and hand (card one-hot only)

    Output fed into the policy MLP head:
      global(33) + action_extras(274) +
      creature_agg(embed*2) + land_agg(embed*2) +
      stack_agg(embed//2 * 2) + graveyard_agg(embed) + hand_agg(embed)

    With default embed_dim=64:
      33 + 274 + 128 + 128 + 64 + 64 + 64 = 755 floats.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        embed_dim: int = 64,
    ):
        half = embed_dim // 2
        features_dim = (
            _GLOBAL_SIZE                                 # 33
            + (observation_space.shape[0] - _HAND_END)  # action extras
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
        action_extras = obs[:, _HAND_END:]   # action cats + card IDs + cost features

        perms     = obs[:, _PERM_START:_PERM_END].reshape(-1, _PERM_SLOTS, _PERM_SLOT_SIZE)
        stack     = obs[:, _STACK_START:_STACK_END].reshape(-1, _STACK_SLOTS, _STACK_SLOT_SIZE)
        graveyard = obs[:, _GY_START:_GY_END].reshape(-1, _GY_SLOTS, _GY_SLOT_SIZE)
        hand      = obs[:, _HAND_START:_HAND_END].reshape(-1, _HAND_SLOTS, _HAND_SLOT_SIZE)

        # Encode each slot type with its shared-weight encoder
        perm_emb = self.perm_encoder(perms)        # (B, 96, embed)
        stk_emb  = self.stack_encoder(stack)       # (B, 12, embed//2)
        gy_emb   = self.entity_encoder(graveyard)  # (B, 128, embed)
        hand_emb = self.entity_encoder(hand)       # (B, 10, embed)  — shared weights

        # Aggregate: mean+max for perms and stack; mean for graveyard and hand
        perm_agg = torch.cat([perm_emb.mean(1), perm_emb.max(1).values], dim=-1)
        stk_agg  = torch.cat([stk_emb.mean(1),  stk_emb.max(1).values],  dim=-1)
        gy_agg   = gy_emb.mean(1)
        hand_agg = hand_emb.mean(1)

        return torch.cat([global_ctx, action_extras, perm_agg, stk_agg, gy_agg, hand_agg], dim=-1)
