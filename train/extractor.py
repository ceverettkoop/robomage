"""
Per-entity feature extractor for RoboMage.

Splits the flat observation into sections that are each encoded by a shared-weight
MLP, then aggregated via mean+max pooling.  The policy head receives a fixed-size
representation that is invariant to card ordering and slot position.

Index layout must stay in sync with src/machine_io.h:
  obs[0:33]         global context (player stats, step, stack)
  obs[33:833]       20 creature slots × 40 floats  (8 status + 32 card one-hot)
  obs[833:1633]     20 land slots    × 40 floats  (same format; status fields 0 for lands)
  obs[1633:1798]     5 stack slots   × 33 floats  (controller_is_A + 32 card one-hot)
  obs[1798:2438]    20 graveyard slots × 32 floats (32 card one-hot)
  obs[2438:2758]    10 hand slots    × 32 floats  (32 card one-hot)
  obs[2758:2790]    32 action-category features   (appended by env.py)
  obs[2790:2822]    32 action card-ID features    (appended by env.py)
  obs[2822:2892]    70 hand cast-cost features    (10 slots × 7 cost feats)
  obs[2892:3032]   140 BF ability-cost features   (20 slots × 7 cost feats)
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ── Layout constants (mirror src/machine_io.h) ──────────────────────────────
_GLOBAL_SIZE     = 33

_CREATURE_SLOTS  = 20
_PERM_SLOT_SIZE  = 40   # 8 status floats + 32 card one-hot

_LAND_SLOTS      = 20
# land slots use the same 40-float format as creature slots

_STACK_SLOTS     = 5
_STACK_SLOT_SIZE = 33   # controller_is_A(1) + card one-hot(32)

_GY_SLOTS        = 20   # 10 per player
_GY_SLOT_SIZE    = 32   # card one-hot only

_HAND_SLOTS      = 10
_HAND_SLOT_SIZE  = 32   # card one-hot only

_CREATURE_START = _GLOBAL_SIZE                                       # 33
_CREATURE_END   = _CREATURE_START + _CREATURE_SLOTS * _PERM_SLOT_SIZE  # 833
_LAND_START     = _CREATURE_END                                      # 833
_LAND_END       = _LAND_START + _LAND_SLOTS * _PERM_SLOT_SIZE          # 1633
_STACK_START    = _LAND_END                                          # 1633
_STACK_END      = _STACK_START + _STACK_SLOTS * _STACK_SLOT_SIZE       # 1798
_GY_START       = _STACK_END                                         # 1798
_GY_END         = _GY_START + _GY_SLOTS * _GY_SLOT_SIZE                # 2438
_HAND_START     = _GY_END                                            # 2438
_HAND_END       = _HAND_START + _HAND_SLOTS * _HAND_SLOT_SIZE          # 2758
# obs[2758:] = action metadata + cost features appended by env.py


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
            _GLOBAL_SIZE                   # 33
            + (observation_space.shape[0] - _HAND_END)  # action extras (274)
            + embed_dim * 2                # creature mean+max
            + embed_dim * 2                # land mean+max
            + half * 2                     # stack mean+max
            + embed_dim                    # graveyard mean
            + embed_dim                    # hand mean
        )
        super().__init__(observation_space, features_dim=features_dim)

        # Shared encoder for 40-float permanent slots (creatures AND lands)
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
        global_ctx  = obs[:, :_GLOBAL_SIZE]
        action_extras = obs[:, _HAND_END:]   # action cats + card IDs + cost features

        creatures  = obs[:, _CREATURE_START:_CREATURE_END].reshape(-1, _CREATURE_SLOTS, _PERM_SLOT_SIZE)
        lands      = obs[:, _LAND_START:_LAND_END].reshape(-1, _LAND_SLOTS, _PERM_SLOT_SIZE)
        stack      = obs[:, _STACK_START:_STACK_END].reshape(-1, _STACK_SLOTS, _STACK_SLOT_SIZE)
        graveyard  = obs[:, _GY_START:_GY_END].reshape(-1, _GY_SLOTS, _GY_SLOT_SIZE)
        hand       = obs[:, _HAND_START:_HAND_END].reshape(-1, _HAND_SLOTS, _HAND_SLOT_SIZE)

        # Encode each slot type with its shared-weight encoder
        cr_emb  = self.perm_encoder(creatures)    # (B, 20, embed)
        land_emb = self.perm_encoder(lands)        # (B, 20, embed)  — shared weights
        stk_emb  = self.stack_encoder(stack)       # (B,  5, embed//2)
        gy_emb   = self.entity_encoder(graveyard)  # (B, 20, embed)
        hand_emb = self.entity_encoder(hand)       # (B, 10, embed)  — shared weights

        # Aggregate: mean+max for creatures, lands, stack; mean for graveyard and hand
        cr_agg   = torch.cat([cr_emb.mean(1),   cr_emb.max(1).values],   dim=-1)
        land_agg = torch.cat([land_emb.mean(1), land_emb.max(1).values], dim=-1)
        stk_agg  = torch.cat([stk_emb.mean(1),  stk_emb.max(1).values],  dim=-1)
        gy_agg   = gy_emb.mean(1)
        hand_agg = hand_emb.mean(1)

        return torch.cat([global_ctx, action_extras, cr_agg, land_agg, stk_agg, gy_agg, hand_agg], dim=-1)
