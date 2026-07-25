"""AlphaZero network (AZNet) for RoboMage — Phase C.

AZNet reuses the Phase B/PPO feature stack so a trained PPO checkpoint can warm-
start it:

  trunk  = a CardGameExtractor(obs_space, embed_dim, per_action_head=True) whose
           encoder submodules are adopted into a TorchScript-scriptable twin
           (:class:`ScriptTrunk`) — see the DEVIATION note below. It returns
           [base features | per-action tail], identical to the extractor.
  policy = an MLP body (net_arch [256,256], Tanh) over the FULL trunk features,
           feeding extractor._ActionScorer to score each candidate action from
           its OWN encoded per-action features (mirrors PerActionMaskablePolicy)
  value  = an MLP body (net_arch [256,256], Tanh) over the FULL trunk features,
           -> Linear(latent, N_VALUE_BUCKETS): a MULTI-HEAD critic with one column
           per (self archetype x opp archetype) value bucket. The column named by
           the obs matchup tail's bucket index is gathered, then tanh -> [-1, 1].

The policy/value bodies + scorer + value Linear are laid out to mirror a
PerActionMaskablePolicy checkpoint 1:1 (mlp_extractor.policy_net /
mlp_extractor.value_net / action_scorer / value_net), so ``from_ppo`` can copy
those weights when warm-starting from that flavor. A stock MlpPolicy checkpoint
shares only the CardGameExtractor trunk; its policy/value heads have a different
shape (no per-action tail, a flat action_net), so they start fresh.

``forward(obs, mask) -> (logits, value)`` masks in-graph and is the future
TorchScript export surface (Phase D). ``--export-check`` scripts the net and
compares scripted vs eager outputs to prove that path now.

DEVIATION (reported to the caller): the plan said ``trunk = CardGameExtractor``
directly, but ``torch.jit.script`` in torch 2.13 rejects that module — its
``forward`` slices with module-level int globals ("python value of type 'int'
cannot be used as a value. Perhaps it is a closed over global variable?"), and
jit eagerly compiles every submodule's forward, so wrapping the extractor makes
AZNet unscriptable. Since scriptability + a passing ``--export-check`` are hard
requirements (and the plan asks for a plain, SB3-class-free scriptable module),
AZNet builds a CardGameExtractor, ADOPTS its encoder submodules (identical names
+ shapes, so warm-start and state_dict stay 1:1 compatible), and drives them
through :class:`ScriptTrunk`'s scriptable forward with the layout constants
baked in as ``__constants__``. ``ScriptTrunk`` is guarded against drift from the
real extractor by :func:`verify_trunk_matches_extractor` (run in --export-check).
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces

import extractor as _ex
from extractor import CardGameExtractor, _ActionScorer
try:
    from env import OBS_SIZE, MAX_ACTIONS, make_observation_space
    from card_costs import N_CARD_TYPES
    from cli_spec import EMBED_DIM, NET_ARCH
    from _enums import REF_ZONE_MAX, N_REF_ZONES, ACTION_CATEGORY_MAX
    from archetypes import N_VALUE_BUCKETS
except ImportError:  # pragma: no cover
    from train.env import OBS_SIZE, MAX_ACTIONS, make_observation_space
    from train.card_costs import N_CARD_TYPES
    from train.cli_spec import EMBED_DIM, NET_ARCH
    from train._enums import REF_ZONE_MAX, N_REF_ZONES, ACTION_CATEGORY_MAX
    from train.archetypes import N_VALUE_BUCKETS

_AZ_CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints", "az")


def obs_space_from_const() -> gym.Space:
    """The env's observation Box, built from OBS_SIZE without spawning an engine.

    Delegates to env.make_observation_space so the per-dimension bounds (notably
    the raw value-bucket slot) are identical to a live env's space."""
    return make_observation_space()


class ScriptTrunk(nn.Module):
    """TorchScript-scriptable mirror of ``CardGameExtractor(per_action_head=True)``.

    It ADOPTS the encoder submodules of a real CardGameExtractor (so parameters,
    names and shapes are identical — warm-start and state_dict are 1:1) and
    reproduces its ``forward`` with every layout offset baked in as a
    ``__constants__`` int (module-global ints are not scriptable in torch 2.13).
    Kept faithful by :func:`verify_trunk_matches_extractor`.
    """

    __constants__ = [
        "N_CARD_TYPES", "ACTION_CATEGORY_MAX", "REF_ZONE_MAX",
        "GLOBAL_SIZE", "HIST_START", "HIST_END", "HIST_ENTRIES",
        "HIST_ENTRY_SIZE", "HIST_RECENT_K", "KNOWN_TOP_LIB_START",
        "KNOWN_TOP_LIB_END", "KNOWN_TOP_LIB_SLOTS", "REVEALED_START",
        "REVEALED_END", "PENDING_START", "PENDING_END", "EXTRAS_START",
        "EXTRAS_END", "STATE_END", "PERM_START", "PERM_END", "PERM_SLOTS",
        "PERM_SLOT_SIZE", "PERM_STATUS_FLOATS", "PERM_CHOSEN_NAME_OFF",
        "PERM_RETURNABLE_OFF", "PERM_CARD_OFF", "STACK_START", "STACK_END",
        "STACK_SLOTS", "STACK_SLOT_SIZE", "STACK_XAMT_OFF", "STACK_MODE_OFF",
        "STACK_TGT_OFF", "STACK_TGT_SLOTS", "STACK_TGT_FIELDS", "GY_START",
        "GY_END", "GY_SLOTS", "EXILE_START", "EXILE_END", "EXILE_SLOTS",
        "HAND_START", "HAND_END", "HAND_SLOTS", "OPP_KNOWN_HAND_START",
        "OPP_KNOWN_HAND_END", "OPP_KNOWN_HAND_SLOTS", "SELF_LIVE_LIB_START",
        "SELF_LIVE_LIB_END", "OPP_DECK_MAIN_START", "OPP_DECK_MAIN_END",
        "OPP_DECK_SIDE_START", "OPP_DECK_SIDE_END", "DECKLIST_MAIN_SLOTS",
        "DECKLIST_SIDE_SLOTS", "DECKLIST_SLOT_SIZE", "DECKLIST_CARD_OFF",
        "DECKLIST_COUNT_OFF", "MAX_ACTIONS", "BUCKET_IDX",
        "ARCH_ONEHOT_START", "ARCH_ONEHOT_END",
        "N_ENTITY_REF_SLOTS", "EMBED_DIM", "PER_ACTION_DIM",
        "features_dim", "per_action_offset", "per_action_slots", "per_action_dim",
    ]

    def __init__(self, fe: CardGameExtractor):
        super().__init__()
        assert getattr(fe, "per_action_head", False), \
            "ScriptTrunk requires a per_action_head=True CardGameExtractor"
        # Adopt the extractor's encoder submodules (same names -> state_dict 1:1).
        self.card_emb = fe.card_emb
        self.perm_encoder = fe.perm_encoder
        self.stack_encoder = fe.stack_encoder
        self.entity_encoder = fe.entity_encoder
        self.decklist_encoder = fe.decklist_encoder
        self.revealed_encoder = fe.revealed_encoder
        self.action_cat_emb = fe.action_cat_emb
        self.zone_emb = fe.zone_emb
        self.action_encoder = fe.action_encoder

        # Layout constants (single-sourced from the extractor module, so their
        # VALUES never drift; only the forward STRUCTURE below is a mirror).
        self.N_CARD_TYPES = int(N_CARD_TYPES)
        self.ACTION_CATEGORY_MAX = int(ACTION_CATEGORY_MAX)
        self.REF_ZONE_MAX = int(REF_ZONE_MAX)
        self.GLOBAL_SIZE = int(_ex._GLOBAL_SIZE)
        self.HIST_START = int(_ex._HIST_START)
        self.HIST_END = int(_ex._HIST_END)
        self.HIST_ENTRIES = int(_ex._HIST_ENTRIES)
        self.HIST_ENTRY_SIZE = int(_ex._HIST_ENTRY_SIZE)
        self.HIST_RECENT_K = int(_ex._HIST_RECENT_K)
        self.KNOWN_TOP_LIB_START = int(_ex._KNOWN_TOP_LIB_START)
        self.KNOWN_TOP_LIB_END = int(_ex._KNOWN_TOP_LIB_END)
        self.KNOWN_TOP_LIB_SLOTS = int(_ex._KNOWN_TOP_LIB_SLOTS)
        self.REVEALED_START = int(_ex._REVEALED_START)
        self.REVEALED_END = int(_ex._REVEALED_END)
        self.PENDING_START = int(_ex._PENDING_START)
        self.PENDING_END = int(_ex._PENDING_END)
        self.EXTRAS_START = int(_ex._EXTRAS_START)
        self.EXTRAS_END = int(_ex._EXTRAS_END)
        self.STATE_END = int(_ex._STATE_END)
        self.PERM_START = int(_ex._PERM_START)
        self.PERM_END = int(_ex._PERM_END)
        self.PERM_SLOTS = int(_ex._PERM_SLOTS)
        self.PERM_SLOT_SIZE = int(_ex._PERM_SLOT_SIZE)
        self.PERM_STATUS_FLOATS = int(_ex._PERM_STATUS_FLOATS)
        self.PERM_CHOSEN_NAME_OFF = int(_ex._PERM_CHOSEN_NAME_OFF)
        self.PERM_RETURNABLE_OFF = int(_ex._PERM_RETURNABLE_OFF)
        self.PERM_CARD_OFF = int(_ex._PERM_CARD_OFF)
        self.STACK_START = int(_ex._STACK_START)
        self.STACK_END = int(_ex._STACK_END)
        self.STACK_SLOTS = int(_ex._STACK_SLOTS)
        self.STACK_SLOT_SIZE = int(_ex._STACK_SLOT_SIZE)
        self.STACK_XAMT_OFF = int(_ex._STACK_XAMT_OFF)
        self.STACK_MODE_OFF = int(_ex._STACK_MODE_OFF)
        self.STACK_TGT_OFF = int(_ex._STACK_TGT_OFF)
        self.STACK_TGT_SLOTS = int(_ex._STACK_TGT_SLOTS)
        self.STACK_TGT_FIELDS = int(_ex._STACK_TGT_FIELDS)
        self.GY_START = int(_ex._GY_START)
        self.GY_END = int(_ex._GY_END)
        self.GY_SLOTS = int(_ex._GY_SLOTS)
        self.EXILE_START = int(_ex._EXILE_START)
        self.EXILE_END = int(_ex._EXILE_END)
        self.EXILE_SLOTS = int(_ex._EXILE_SLOTS)
        self.HAND_START = int(_ex._HAND_START)
        self.HAND_END = int(_ex._HAND_END)
        self.HAND_SLOTS = int(_ex._HAND_SLOTS)
        self.OPP_KNOWN_HAND_START = int(_ex._OPP_KNOWN_HAND_START)
        self.OPP_KNOWN_HAND_END = int(_ex._OPP_KNOWN_HAND_END)
        self.OPP_KNOWN_HAND_SLOTS = int(_ex._OPP_KNOWN_HAND_SLOTS)
        self.SELF_LIVE_LIB_START = int(_ex._SELF_LIVE_LIB_START)
        self.SELF_LIVE_LIB_END = int(_ex._SELF_LIVE_LIB_END)
        self.OPP_DECK_MAIN_START = int(_ex._OPP_DECK_MAIN_START)
        self.OPP_DECK_MAIN_END = int(_ex._OPP_DECK_MAIN_END)
        self.OPP_DECK_SIDE_START = int(_ex._OPP_DECK_SIDE_START)
        self.OPP_DECK_SIDE_END = int(_ex._OPP_DECK_SIDE_END)
        self.DECKLIST_MAIN_SLOTS = int(_ex._DECKLIST_MAIN_SLOTS)
        self.DECKLIST_SIDE_SLOTS = int(_ex._DECKLIST_SIDE_SLOTS)
        self.DECKLIST_SLOT_SIZE = int(_ex._DECKLIST_SLOT_SIZE)
        self.DECKLIST_CARD_OFF = int(_ex._DECKLIST_CARD_OFF)
        self.DECKLIST_COUNT_OFF = int(_ex._DECKLIST_COUNT_OFF)
        self.MAX_ACTIONS = int(_ex._MAX_ACTIONS)
        # Matchup-tail offsets (env.py owns them; the extractor re-exports them).
        self.BUCKET_IDX = int(_ex._BUCKET_IDX)
        self.ARCH_ONEHOT_START = int(_ex._ARCH_ONEHOT_START)
        self.ARCH_ONEHOT_END = int(_ex._ARCH_ONEHOT_END)
        self.N_ENTITY_REF_SLOTS = int(_ex.N_ENTITY_REF_SLOTS)
        self.EMBED_DIM = int(fe._embed_dim)
        self.PER_ACTION_DIM = int(fe.per_action_dim)

        self.features_dim = int(fe.features_dim)
        self.per_action_offset = int(fe.per_action_offset)
        self.per_action_slots = int(fe.per_action_slots)
        self.per_action_dim = int(fe.per_action_dim)

    def _embed_ids(self, id_floats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.round(id_floats * self.N_CARD_TYPES).long()
        present = idx >= 0
        emb = self.card_emb((idx + 1).clamp(0, self.N_CARD_TYPES))
        return emb, present

    def _mean_max(self, emb: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        m = present.unsqueeze(-1).to(emb.dtype)
        counts = m.sum(1).clamp(min=1.0)
        masked_mean = (emb * m).sum(1) / counts
        # float32 finfo().min (torch.finfo is not TorchScript-scriptable); present
        # rows are always selected below, so the exact fill value is immaterial.
        neg_inf = -3.4028234663852886e+38
        masked_max = emb.masked_fill(m == 0, neg_inf).max(1)[0]
        has_any = present.any(1, keepdim=True)
        masked_max = torch.where(has_any, masked_max, torch.zeros_like(masked_max))
        return torch.cat([masked_mean, masked_max], dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        global_ctx = obs[:, :self.GLOBAL_SIZE]
        hist_ctx = obs[:, self.HIST_START:self.HIST_END]
        meta_ctx = obs[:, self.HIST_END:self.KNOWN_TOP_LIB_START]
        revealed = obs[:, self.REVEALED_START:self.REVEALED_END]
        pending = obs[:, self.PENDING_START:self.PENDING_END]
        extras = obs[:, self.EXTRAS_START:self.EXTRAS_END]
        # Mirror the extractor: the action/cost block stops BEFORE the matchup
        # tail's raw bucket float (stripped from the trunk input), and the two
        # archetype one-hots enter the base cat right after it.
        action_extras = obs[:, self.STATE_END:self.BUCKET_IDX]
        arch_onehot = obs[:, self.ARCH_ONEHOT_START:self.ARCH_ONEHOT_END]

        hist_entries = hist_ctx.reshape(-1, self.HIST_ENTRIES, self.HIST_ENTRY_SIZE)
        recent = hist_entries[:, :self.HIST_RECENT_K]
        rec_cat_idx = torch.round(recent[:, :, 0] * self.ACTION_CATEGORY_MAX).long(
        ).clamp(0, self.ACTION_CATEGORY_MAX)
        rec_cat_e = self.action_cat_emb(rec_cat_idx)
        rec_card_e, _ = self._embed_ids(recent[:, :, 1])
        hist_recent = torch.cat([rec_cat_e, rec_card_e, recent[:, :, 2:4]], dim=-1
                                ).reshape(recent.shape[0], -1)

        pending_emb, _ = self._embed_ids(pending[:, 0])
        pending_feat = torch.cat([pending_emb, pending[:, 1:2]], dim=-1)

        perms = obs[:, self.PERM_START:self.PERM_END].reshape(
            -1, self.PERM_SLOTS, self.PERM_SLOT_SIZE)
        stack = obs[:, self.STACK_START:self.STACK_END].reshape(
            -1, self.STACK_SLOTS, self.STACK_SLOT_SIZE)
        graveyard = obs[:, self.GY_START:self.GY_END].reshape(-1, self.GY_SLOTS, 1)
        exile = obs[:, self.EXILE_START:self.EXILE_END].reshape(-1, self.EXILE_SLOTS, 1)
        hand = obs[:, self.HAND_START:self.HAND_END].reshape(-1, self.HAND_SLOTS, 1)
        opp_hand = obs[:, self.OPP_KNOWN_HAND_START:self.OPP_KNOWN_HAND_END].reshape(
            -1, self.OPP_KNOWN_HAND_SLOTS, 1)
        top_lib = obs[:, self.KNOWN_TOP_LIB_START:self.KNOWN_TOP_LIB_END].reshape(
            -1, self.KNOWN_TOP_LIB_SLOTS, 1)
        self_lib = obs[:, self.SELF_LIVE_LIB_START:self.SELF_LIVE_LIB_END].reshape(
            -1, self.DECKLIST_MAIN_SLOTS, self.DECKLIST_SLOT_SIZE)
        opp_main = obs[:, self.OPP_DECK_MAIN_START:self.OPP_DECK_MAIN_END].reshape(
            -1, self.DECKLIST_MAIN_SLOTS, self.DECKLIST_SLOT_SIZE)
        opp_side = obs[:, self.OPP_DECK_SIDE_START:self.OPP_DECK_SIDE_END].reshape(
            -1, self.DECKLIST_SIDE_SLOTS, self.DECKLIST_SLOT_SIZE)

        perm_card_emb, perm_present = self._embed_ids(perms[:, :, self.PERM_CARD_OFF])
        perm_chosen_emb, _ = self._embed_ids(perms[:, :, self.PERM_CHOSEN_NAME_OFF])
        perm_returnable_emb, _ = self._embed_ids(perms[:, :, self.PERM_RETURNABLE_OFF])
        perm_in = torch.cat([perms[:, :, :self.PERM_STATUS_FLOATS],
                             perm_chosen_emb, perm_returnable_emb, perm_card_emb], dim=-1)

        stk_card_emb, stk_present = self._embed_ids(stack[:, :, 1])
        stk_tgts = stack[:, :, self.STACK_TGT_OFF:].reshape(
            -1, self.STACK_SLOTS, self.STACK_TGT_SLOTS, self.STACK_TGT_FIELDS)
        stk_tgt_emb, _ = self._embed_ids(stk_tgts[:, :, :, self.STACK_TGT_FIELDS - 1])
        stk_tgt_mask = stk_tgts[:, :, :, 0:1]
        stk_tgt_agg = (stk_tgt_emb * stk_tgt_mask).sum(2) / stk_tgt_mask.sum(2).clamp(min=1.0)
        stk_xquals = stack[:, :, self.STACK_XAMT_OFF:self.STACK_MODE_OFF]
        stk_modes = stack[:, :, self.STACK_MODE_OFF:self.STACK_TGT_OFF]
        stk_tgt_scalars = stk_tgts[:, :, :, :self.STACK_TGT_FIELDS - 1].reshape(
            -1, self.STACK_SLOTS, self.STACK_TGT_SLOTS * (self.STACK_TGT_FIELDS - 1))
        stk_in = torch.cat([stack[:, :, 0:1], stack[:, :, 2:3], stk_xquals, stk_modes,
                            stk_tgt_scalars, stk_card_emb, stk_tgt_agg], dim=-1)

        gy_emb_in, gy_present = self._embed_ids(graveyard[:, :, 0])
        ex_emb_in, ex_present = self._embed_ids(exile[:, :, 0])
        hand_emb_in, hand_present = self._embed_ids(hand[:, :, 0])
        opp_hand_emb_in, opp_hand_present = self._embed_ids(opp_hand[:, :, 0])
        top_lib_emb_in, _ = self._embed_ids(top_lib[:, :, 0])

        self_lib_emb, self_lib_present = self._embed_ids(self_lib[:, :, self.DECKLIST_CARD_OFF])
        opp_main_emb, opp_main_present = self._embed_ids(opp_main[:, :, self.DECKLIST_CARD_OFF])
        opp_side_emb, opp_side_present = self._embed_ids(opp_side[:, :, self.DECKLIST_CARD_OFF])
        self_lib_in = torch.cat(
            [self_lib_emb, self_lib[:, :, self.DECKLIST_COUNT_OFF:self.DECKLIST_COUNT_OFF + 1]], dim=-1)
        opp_main_in = torch.cat(
            [opp_main_emb, opp_main[:, :, self.DECKLIST_COUNT_OFF:self.DECKLIST_COUNT_OFF + 1]], dim=-1)
        opp_side_in = torch.cat(
            [opp_side_emb, opp_side[:, :, self.DECKLIST_COUNT_OFF:self.DECKLIST_COUNT_OFF + 1]], dim=-1)

        perm_emb = self.perm_encoder(perm_in)
        stk_emb = self.stack_encoder(stk_in)
        gy_emb = self.entity_encoder(gy_emb_in)
        ex_emb = self.entity_encoder(ex_emb_in)
        hand_emb = self.entity_encoder(hand_emb_in)
        opp_hand_emb = self.entity_encoder(opp_hand_emb_in)
        top_lib_emb = self.entity_encoder(top_lib_emb_in)
        self_lib_enc = self.decklist_encoder(self_lib_in)
        opp_main_enc = self.decklist_encoder(opp_main_in)
        opp_side_enc = self.decklist_encoder(opp_side_in)

        perm_agg = self._mean_max(perm_emb, perm_present)
        stk_agg = self._mean_max(stk_emb, stk_present)
        gy_agg = self._mean_max(gy_emb, gy_present)
        ex_agg = self._mean_max(ex_emb, ex_present)
        hand_agg = self._mean_max(hand_emb, hand_present)
        opp_hand_agg = self._mean_max(opp_hand_emb, opp_hand_present)
        top_lib_agg = top_lib_emb.mean(1)
        revealed_agg = self.revealed_encoder(revealed)
        self_lib_agg = self._mean_max(self_lib_enc, self_lib_present)
        opp_main_agg = self._mean_max(opp_main_enc, opp_main_present)
        opp_side_agg = self._mean_max(opp_side_enc, opp_side_present)

        base = torch.cat([global_ctx, hist_ctx, hist_recent, meta_ctx, top_lib_agg,
                          revealed_agg, pending_feat, extras, action_extras,
                          arch_onehot,
                          perm_agg, stk_agg, gy_agg, ex_agg, hand_agg, opp_hand_agg,
                          self_lib_agg, opp_main_agg, opp_side_agg], dim=-1)

        a0 = self.STATE_END
        cats = obs[:, a0:a0 + self.MAX_ACTIONS]
        act_ids = obs[:, a0 + self.MAX_ACTIONS:a0 + 2 * self.MAX_ACTIONS]
        ctrl = obs[:, a0 + 2 * self.MAX_ACTIONS:a0 + 3 * self.MAX_ACTIONS]
        zone = obs[:, a0 + 3 * self.MAX_ACTIONS:a0 + 4 * self.MAX_ACTIONS]
        refs = obs[:, a0 + 4 * self.MAX_ACTIONS:a0 + 5 * self.MAX_ACTIONS]
        ords = obs[:, a0 + 5 * self.MAX_ACTIONS:a0 + 6 * self.MAX_ACTIONS]
        cat_idx = torch.round(cats * self.ACTION_CATEGORY_MAX).long().clamp(
            0, self.ACTION_CATEGORY_MAX)
        cat_e = self.action_cat_emb(cat_idx)
        act_id_e, _ = self._embed_ids(act_ids)
        zone_idx = torch.round(zone * self.REF_ZONE_MAX).long().clamp(0, self.REF_ZONE_MAX)
        zone_e = self.zone_emb(zone_idx)

        E = self.EMBED_DIM
        stk_emb_pad = torch.cat(
            [stk_emb, stk_emb.new_zeros(stk_emb.shape[0], self.STACK_SLOTS,
                                        E - stk_emb.shape[-1])], dim=-1)
        ent_table = torch.cat([perm_emb, stk_emb_pad], dim=1)
        ent_table = torch.cat(
            [ent_table, ent_table.new_zeros(ent_table.shape[0], 1, E)], dim=1)
        ref_idx = torch.round(refs * self.N_ENTITY_REF_SLOTS).long() - 1
        ref_idx = torch.where(ref_idx < 0,
                              torch.full_like(ref_idx, self.N_ENTITY_REF_SLOTS),
                              ref_idx).clamp(0, self.N_ENTITY_REF_SLOTS)
        ref_e = torch.gather(ent_table, 1, ref_idx.unsqueeze(-1).expand(-1, -1, E))

        pa_in = torch.cat([cat_e, act_id_e, ctrl.unsqueeze(-1), zone_e, ref_e,
                           ords.unsqueeze(-1)], dim=-1)
        pa = self.action_encoder(pa_in)
        return torch.cat([base, pa.reshape(pa.shape[0], -1)], dim=-1)


def _mlp_body(in_dim: int, arch) -> nn.Sequential:
    """SB3-MlpExtractor-shaped body: Linear/Tanh pairs over ``arch`` (indices
    0,2,... are the Linears, matching mlp_extractor.policy_net.{0,2} keys)."""
    layers: list = []
    last = in_dim
    for h in arch:
        layers.append(nn.Linear(last, h))
        layers.append(nn.Tanh())
        last = h
    return nn.Sequential(*layers)


class AZNet(nn.Module):
    """Policy+value net over the shared CardGameExtractor trunk (scriptable twin)."""

    def __init__(self, observation_space: Optional[gym.Space] = None,
                 embed_dim: int = EMBED_DIM, net_arch=None):
        super().__init__()
        if observation_space is None:
            observation_space = obs_space_from_const()
        arch = list(net_arch) if net_arch is not None else list(NET_ARCH)
        self.embed_dim = int(embed_dim)
        self.obs_size = int(observation_space.shape[0])

        # Build a real CardGameExtractor (per the plan), then drive its adopted
        # submodules through a scriptable forward. See the DEVIATION note above.
        fe = CardGameExtractor(observation_space, embed_dim=embed_dim,
                               per_action_head=True)
        self.trunk = ScriptTrunk(fe)
        feat_dim = self.trunk.features_dim
        self._pa_offset = int(self.trunk.per_action_offset)
        self._pa_slots = int(self.trunk.per_action_slots)
        self._pa_dim = int(self.trunk.per_action_dim)
        latent_dim = arch[-1]

        self.policy_body = _mlp_body(feat_dim, arch)
        self.value_body = _mlp_body(feat_dim, arch)
        self.action_scorer = _ActionScorer(latent_dim, self._pa_dim)
        # Multi-head critic, mirroring PerActionMaskablePolicy: one column per
        # (self archetype x opp archetype) value bucket, gathered by the bucket
        # index in the obs's matchup tail. AZ targets are already bounded [-1,1],
        # so this is purely about separating matchup statistics in the last layer.
        self.value_head = nn.Linear(latent_dim, N_VALUE_BUCKETS)
        self._bucket_idx = int(self.trunk.BUCKET_IDX)
        self._n_buckets = int(N_VALUE_BUCKETS)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        """(obs[B,OBS], mask[B,MAX_ACTIONS] bool) -> (masked logits, value[B]).

        Masking is in-graph (illegal actions -> -inf). TorchScript export surface."""
        features = self.trunk(obs)
        pa = features[:, self._pa_offset:].reshape(
            features.shape[0], self._pa_slots, self._pa_dim)
        latent_pi = self.policy_body(features)
        latent_vf = self.value_body(features)
        logits = self.action_scorer(latent_pi, pa)
        neg = -3.4028234663852886e+38   # float32 finfo().min (jit-safe literal)
        logits = torch.where(mask, logits, torch.full_like(logits, neg))
        # Gather this sample's archetype-bucket column, then squash.
        bucket = torch.round(obs[:, self._bucket_idx]).long().clamp(
            0, self._n_buckets - 1)
        v_all = self.value_head(latent_vf)
        value = torch.tanh(v_all.gather(1, bucket.unsqueeze(1))).squeeze(-1)
        return logits, value

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def meta(self, steps: int = 0) -> dict:
        return {
            "embed_dim": self.embed_dim,
            "obs_size": self.obs_size,
            "OBS_SIZE": OBS_SIZE,
            "MAX_ACTIONS": MAX_ACTIONS,
            "N_CARD_TYPES": N_CARD_TYPES,
            "N_VALUE_BUCKETS": N_VALUE_BUCKETS,
            "steps": int(steps),
        }

    def save(self, path: str, steps: int = 0) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)
        with open(_meta_path(path), "w") as f:
            json.dump(self.meta(steps), f, indent=2)
        return path


def _meta_path(path: str) -> str:
    base, _ = os.path.splitext(path)
    return base + ".meta.json"


def torchscript_export_path(ckpt_path: str) -> str:
    """Map a state_dict checkpoint path to its TorchScript sibling.

    ``foo__azfinal.pt`` -> ``foo__azfinal.ts.pt`` (the distinct ``.ts.pt`` suffix
    keeps the serialized module from colliding with the state_dict ``.pt`` that
    :func:`resolve_az_checkpoint` loads). Used by the C++ actor pipeline."""
    base, _ = os.path.splitext(ckpt_path)
    return base + ".ts.pt"


def save_torchscript(net: "AZNet", path: str, steps: Optional[int] = None) -> str:
    """Script ``net`` and persist the serialized TorchScript module to ``path``.

    ``path`` must end with ``.ts.pt`` (the distinct suffix that keeps the artifact
    from colliding with state_dict ``.pt`` files). Also writes ``<path>.meta.json``
    with the same :meth:`AZNet.meta` handshake dict so a C++ loader can verify
    OBS_SIZE / MAX_ACTIONS / N_CARD_TYPES before running the module."""
    assert path.endswith(".ts.pt"), \
        f"TorchScript export path must end with .ts.pt, got {path!r}"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    net.eval()
    scripted = torch.jit.script(net)
    scripted.save(path)
    with open(_meta_path(path), "w") as f:
        json.dump(net.meta(steps or 0), f, indent=2)
    return path


# The single generalist AZ checkpoint stem — mirrors opponents.GEN_STEM for PPO.
# There is now ONE AZ net that pilots ANY deck; the deck it plays travels as a
# separate explicit parameter, never inferred from the filename. Its checkpoints
# are 'gen__azfinal.pt' (the gate-promoted incumbent) plus periodic
# 'gen__azv{steps}.pt' candidate snapshots (+ '.ts.pt' / '.meta.json' siblings).
AZ_GEN_STEM = "gen"


def az_checkpoint_path(steps: Optional[int] = None,
                       checkpoint_dir: str = _AZ_CKPT_DIR) -> str:
    """``{dir}/gen__azfinal.pt`` (steps None) or ``{dir}/gen__azv{steps}.pt``.

    Keyed to the single generalist stem — there is no per-deck AZ net anymore."""
    name = (f"{AZ_GEN_STEM}__azfinal.pt" if steps is None
            else f"{AZ_GEN_STEM}__azv{int(steps)}.pt")
    return os.path.join(checkpoint_dir, name)


def resolve_az_checkpoint(spec: str,
                          checkpoint_dir: str = _AZ_CKPT_DIR,
                          prefer: str = "final") -> Optional[str]:
    """Resolve an AZ checkpoint spec to a path, generalist contract. Accepts:

      - an explicit existing path (``.pt``) — returned as-is;
      - the reserved stem ``'gen'`` → ``gen__azfinal.pt``, else the newest
        ``gen__azv*.pt`` (respecting ``prefer``).

    Returns ``None`` for anything else — a nonexistent path, or a bare (non-``gen``)
    deck shorthand — so boolean probes stay well-behaved and callers can fall back
    to a PPO warm-start. The user-facing "deck shorthands are gone" error is raised
    at the controller layer (see opponents._load_az_evaluator), which pairs the
    generalist net with an explicit deck.

    ``prefer="snapshot"`` flips the final/snapshot order — the trainer uses it to
    continue the CANDIDATE line (newest snapshot) while ``__azfinal`` stays the
    gate-promoted incumbent that self-play and opponent specs default to."""
    if spec and spec.endswith(".pt") and os.path.exists(spec):
        return spec
    if not spec or spec.strip().lower() != AZ_GEN_STEM:
        return None
    final = az_checkpoint_path(None, checkpoint_dir)
    if not os.path.exists(final):
        final = None
    import glob as _glob
    snaps = _glob.glob(os.path.join(checkpoint_dir, f"{AZ_GEN_STEM}__azv*.pt"))

    def _steps(p):
        try:
            return int(os.path.basename(p).split("__azv")[1].split(".pt")[0])
        except (IndexError, ValueError):
            return -1
    snaps = [p for p in snaps if _steps(p) >= 0]
    snap = max(snaps, key=_steps) if snaps else None
    return (snap or final) if prefer == "snapshot" else (final or snap)


def load_az(path: str, map_location="cpu") -> "AZNet":
    """Load an AZNet from ``path`` (+ its .meta.json), asserting the layout
    handshake against the current OBS_SIZE / MAX_ACTIONS / N_CARD_TYPES."""
    meta = {}
    mp = _meta_path(path)
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
        for key, cur in (("OBS_SIZE", OBS_SIZE), ("MAX_ACTIONS", MAX_ACTIONS),
                         ("N_CARD_TYPES", N_CARD_TYPES),
                         ("N_VALUE_BUCKETS", N_VALUE_BUCKETS)):
            if key in meta and int(meta[key]) != int(cur):
                raise RuntimeError(
                    f"AZ checkpoint {path} layout mismatch: {key}={meta[key]} "
                    f"but current build has {cur} — regenerate / retrain")
    embed_dim = int(meta.get("embed_dim", EMBED_DIM))
    net = AZNet(obs_space_from_const(), embed_dim=embed_dim)
    sd = torch.load(path, map_location=map_location)
    net.load_state_dict(sd)
    net.eval()
    return net


# ----------------------------------------------------------------------
# Warm-start from a PPO checkpoint
# ----------------------------------------------------------------------

def _detect_ppo_flavor(sd: dict) -> str:
    """'per_action' if the checkpoint has the per-action scorer/encoder,
    else 'stock'."""
    has_scorer = any(k.startswith("action_scorer.") for k in sd)
    has_action_encoder = any(k.startswith("features_extractor.action_encoder.")
                             for k in sd)
    return "per_action" if (has_scorer or has_action_encoder) else "stock"


def from_ppo(ckpt_path: str, map_location="cpu") -> "AZNet":
    """Build an AZNet warm-started from a PPO ``.zip`` checkpoint.

    Two flavors are handled by inspecting the state-dict keys:
      * PerActionMaskablePolicy — trunk + action scorer + policy/value bodies +
        value head map 1:1 (the value head loses its tanh in AZNet — noted).
      * stock MlpPolicy — only the shared CardGameExtractor trunk transfers; the
        policy/value heads have a different shape and start fresh.
    Weights that don't transfer are reported."""
    try:
        from sb3_contrib import MaskablePPO as _PPO
    except ImportError:  # pragma: no cover
        from stable_baselines3 import PPO as _PPO
    model = _PPO.load(ckpt_path, device="cpu")
    pk = getattr(model, "policy_kwargs", {}) or {}
    embed_dim = int(pk.get("features_extractor_kwargs", {}).get("embed_dim",
                                                                EMBED_DIM))
    sd = model.policy.state_dict()
    flavor = _detect_ppo_flavor(sd)

    net = AZNet(obs_space_from_const(), embed_dim=embed_dim)
    net_sd = net.state_dict()

    transferred: list = []
    # 1) Trunk (features_extractor.* -> trunk.*). Shared by both flavors.
    for k, v in sd.items():
        if not k.startswith("features_extractor."):
            continue
        tk = "trunk." + k[len("features_extractor."):]
        if tk in net_sd and net_sd[tk].shape == v.shape:
            net_sd[tk] = v.clone()
            transferred.append(tk)

    notes: list = []
    if flavor == "per_action":
        pa_map = {
            "mlp_extractor.policy_net.": "policy_body.",
            "mlp_extractor.value_net.": "value_body.",
            "action_scorer.": "action_scorer.",
        }
        for k, v in sd.items():
            for src, dst in pa_map.items():
                if k.startswith(src):
                    tk = dst + k[len(src):]
                    if tk in net_sd and net_sd[tk].shape == v.shape:
                        net_sd[tk] = v.clone()
                        transferred.append(tk)
        for k, v in sd.items():
            if k.startswith("value_net."):
                tk = "value_head." + k[len("value_net."):]
                if tk in net_sd and net_sd[tk].shape == v.shape:
                    net_sd[tk] = v.clone()
                    transferred.append(tk)
        notes.append("multi-head value head copied 1:1 from PPO value_net "
                     f"({N_VALUE_BUCKETS} archetype-bucket columns, now behind tanh)")
    else:
        notes.append("stock MlpPolicy: policy/value heads start fresh "
                     "(shape-incompatible with the per-action AZ head)")

    net.load_state_dict(net_sd)

    tset = set(transferred)
    fresh = [k for k in net_sd if k not in tset]
    print(f"[from_ppo] flavor={flavor} embed_dim={embed_dim}: "
          f"transferred {len(transferred)} tensors, {len(fresh)} fresh.")
    for n in notes:
        print(f"[from_ppo]   note: {n}")
    fresh_top = sorted({k.split(".")[0] for k in fresh})
    if fresh:
        print(f"[from_ppo]   fresh modules: {', '.join(fresh_top)}")
    net.eval()
    return net


# ----------------------------------------------------------------------
# Evaluator for MCTS
# ----------------------------------------------------------------------

class AZEvaluator:
    """mcts.Evaluator over an AZNet: softmax(masked logits[:num_choices]) priors
    and the tanh value passthrough (already in [-1,1], current-mover view)."""

    def __init__(self, net: "AZNet", device: str = "cpu"):
        self._net = net.to(device).eval()
        self._device = device
        self._mask = np.zeros(MAX_ACTIONS, dtype=bool)

    def evaluate(self, obs: np.ndarray, num_choices: int):
        self._mask[:] = False
        self._mask[:num_choices] = True
        with torch.no_grad():
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                    device=self._device).unsqueeze(0)
            mask_t = torch.as_tensor(self._mask, device=self._device).unsqueeze(0)
            logits, value = self._net(obs_t, mask_t)
            probs = torch.softmax(logits[0, :num_choices], dim=-1)
            priors = probs.detach().cpu().numpy().astype(np.float64)
            v = float(value.item())
        total = priors.sum()
        if not np.isfinite(total) or total <= 0.0:
            priors = np.full(num_choices, 1.0 / num_choices)
            total = 1.0
        return priors / total, v


# ----------------------------------------------------------------------
# Consistency guard + --export-check CLI
# ----------------------------------------------------------------------

def verify_trunk_matches_extractor(embed_dim: int = 32, seed: int = 0,
                                   tol: float = 1e-5) -> float:
    """Assert ScriptTrunk reproduces CardGameExtractor(per_action_head=True) bit-
    for-bit on random input (guards the mirrored forward against drift). Returns
    the max abs diff."""
    torch.manual_seed(seed)
    space = obs_space_from_const()
    fe = CardGameExtractor(space, embed_dim=embed_dim, per_action_head=True).eval()
    trunk = ScriptTrunk(fe).eval()          # adopts fe's submodules (same weights)
    obs = torch.randn(4, OBS_SIZE)
    with torch.no_grad():
        ref = fe(obs)
        got = trunk(obs)
    d = float((ref - got).abs().max())
    if d > tol:
        raise RuntimeError(f"ScriptTrunk drifted from CardGameExtractor: "
                           f"max|Δ|={d:.2e} > {tol:.0e}")
    return d


def _export_check(embed_dim: int = EMBED_DIM, seed: int = 0,
                  tol: float = 1e-4) -> int:
    # 1) mirror faithfulness
    d = verify_trunk_matches_extractor(embed_dim=min(embed_dim, 32), seed=seed)
    print(f"trunk-vs-extractor: max|Δ|={d:.2e} -> PASS")

    # 2) scripted vs eager AZNet
    torch.manual_seed(seed)
    net = AZNet(obs_space_from_const(), embed_dim=embed_dim).eval()
    obs = torch.randn(3, OBS_SIZE)
    mask = torch.zeros(3, MAX_ACTIONS, dtype=torch.bool)
    for i in range(3):
        mask[i, : (i + 2)] = True
    with torch.no_grad():
        eager_logits, eager_value = net(obs, mask)
    try:
        scripted = torch.jit.script(net)
    except Exception as e:  # pragma: no cover
        print(f"FAIL: torch.jit.script(AZNet) raised: {e}")
        return 1
    with torch.no_grad():
        s_logits, s_value = scripted(obs, mask)
    dl = 0.0
    for i in range(3):
        n = int(mask[i].sum())
        dl = max(dl, float((eager_logits[i, :n] - s_logits[i, :n]).abs().max()))
    dv = float((eager_value - s_value).abs().max())
    ok = dl <= tol and dv <= tol
    print(f"export-check: max|Δlogits|={dl:.2e} max|Δvalue|={dv:.2e} "
          f"tol={tol:.0e} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        return 1

    # 3) persisted TorchScript round-trips: save -> torch.jit.load -> re-run.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ts_path = os.path.join(td, "aznet_export_check.ts.pt")
        save_torchscript(net, ts_path)
        reloaded = torch.jit.load(ts_path)
        reloaded.eval()
        with torch.no_grad():
            r_logits, r_value = reloaded(obs, mask)
        rdl = 0.0
        for i in range(3):
            n = int(mask[i].sum())
            rdl = max(rdl, float((eager_logits[i, :n] - r_logits[i, :n]).abs().max()))
        rdv = float((eager_value - r_value).abs().max())
        with open(_meta_path(ts_path)) as f:
            rt_meta = json.load(f)
        meta_ok = all(int(rt_meta[k]) == int(v) for k, v in
                      (("OBS_SIZE", OBS_SIZE), ("MAX_ACTIONS", MAX_ACTIONS),
                       ("N_CARD_TYPES", N_CARD_TYPES)))
    reload_ok = rdl == 0.0 and rdv == 0.0 and meta_ok
    print(f"export-reload: max|Δlogits|={rdl:.2e} max|Δvalue|={rdv:.2e} "
          f"meta_ok={meta_ok} -> {'PASS' if reload_ok else 'FAIL'}")
    return 0 if reload_ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AZNet utilities")
    ap.add_argument("--export-check", action="store_true",
                    help="Script the net and compare scripted vs eager outputs")
    ap.add_argument("--from-ppo", default=None,
                    help="Warm-start from a PPO checkpoint and report transfer")
    ap.add_argument("--export", default=None,
                    help="Export an AZ checkpoint (path or deck shorthand) to its "
                         "sibling .ts.pt TorchScript module + meta")
    ap.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    args = ap.parse_args()
    rc = 0
    if args.from_ppo:
        from_ppo(args.from_ppo)
    if args.export:
        ckpt = resolve_az_checkpoint(args.export)
        if ckpt is None:
            raise SystemExit(f"no AZ checkpoint found for {args.export!r}")
        net = load_az(ckpt)
        out = save_torchscript(net, torchscript_export_path(ckpt))
        print(out)
    if args.export_check or (not args.from_ppo and not args.export):
        rc = _export_check(embed_dim=args.embed_dim)
    raise SystemExit(rc)
