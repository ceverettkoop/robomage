"""
Per-entity feature extractor for RoboMage.

Splits the flat observation into three parts:
  - Global context (game state scalars)
  - Battlefield entity slots (processed with a shared-weight MLP)
  - Hand entity slots (processed with a separate shared-weight MLP)

Battlefield and hand slots are each encoded independently then aggregated via
mean+max pooling, giving the policy head a fixed-size representation that is
invariant to card ordering and slot position.

Index layout must stay in sync with src/machine_io.h:
  obs[0:33]       global context (player stats, step, stack)
  obs[33:833]     20 battlefield slots × 40 floats each  (8 status + 32 card one-hot)
  obs[833:1153]   10 hand slots × 32 floats each         (32 card one-hot)
  obs[1153:1185]  32 action-category features (appended by env.py)
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ── layout constants (mirror src/machine_io.h) ─────────────────────────────
_GLOBAL_SIZE    = 33
_BF_SLOTS       = 20
_BF_SLOT_SIZE   = 40   # 8 status floats + 32 card one-hot (N_CARD_TYPES)
_HAND_SLOTS     = 10
_HAND_SLOT_SIZE = 32   # N_CARD_TYPES card one-hot
_ACTION_CATS    = 32   # MAX_ACTIONS, appended by env.py

_BF_START   = _GLOBAL_SIZE                             # 33
_BF_END     = _BF_START + _BF_SLOTS * _BF_SLOT_SIZE   # 833
_HAND_START = _BF_END                                  # 833
_HAND_END   = _HAND_START + _HAND_SLOTS * _HAND_SLOT_SIZE  # 1153
# obs[1153:1185] = action categories


class CardGameExtractor(BaseFeaturesExtractor):
    """
    Shared-weight per-entity encoder with mean+max aggregation.

    Output fed into the policy MLP head:
      global(33) + action_cats(32) + bf_agg(embed*2) + hand_agg(embed) floats
    With default embed_dim=64: 33 + 32 + 128 + 64 = 257 floats.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        bf_embed_dim: int = 64,
        hand_embed_dim: int = 64,
    ):
        features_dim = _GLOBAL_SIZE + _ACTION_CATS + bf_embed_dim * 2 + hand_embed_dim
        super().__init__(observation_space, features_dim=features_dim)

        # Shared encoder for each battlefield slot (40 → bf_embed_dim)
        self.bf_encoder = nn.Sequential(
            nn.Linear(_BF_SLOT_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, bf_embed_dim),
            nn.ReLU(),
        )

        # Shared encoder for each hand slot (32 → hand_embed_dim)
        self.hand_encoder = nn.Sequential(
            nn.Linear(_HAND_SLOT_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, hand_embed_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        global_ctx  = obs[:, :_GLOBAL_SIZE]
        battlefield = obs[:, _BF_START:_BF_END].reshape(-1, _BF_SLOTS, _BF_SLOT_SIZE)
        hand        = obs[:, _HAND_START:_HAND_END].reshape(-1, _HAND_SLOTS, _HAND_SLOT_SIZE)
        action_cats = obs[:, _HAND_END:]

        # Encode every slot with the shared weights
        bf_emb   = self.bf_encoder(battlefield)   # (B, 20, bf_embed_dim)
        hand_emb = self.hand_encoder(hand)         # (B, 10, hand_embed_dim)

        # Mean + max aggregation: captures both average board state and salient entities
        bf_agg = torch.cat(
            [bf_emb.mean(dim=1), bf_emb.max(dim=1).values], dim=-1
        )  # (B, bf_embed_dim * 2)
        hand_agg = hand_emb.mean(dim=1)  # (B, hand_embed_dim)

        return torch.cat([global_ctx, action_cats, bf_agg, hand_agg], dim=-1)
