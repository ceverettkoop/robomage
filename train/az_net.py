"""AlphaZero network (AZNet) for RoboMage — Phase C.

AZNet reuses the Phase B/PPO feature stack so a trained PPO checkpoint can warm-
start it:

  trunk  = extractor.CardGameExtractor(obs_space, embed_dim, per_action_head=True)
           -> the per-entity encoder, returning [base features | per-action tail]
  policy = an MLP body (net_arch [256,256], Tanh) over the FULL trunk features,
           feeding extractor._ActionScorer to score each candidate action from
           its OWN encoded per-action features (mirrors PerActionMaskablePolicy)
  value  = an MLP body (net_arch [256,256], Tanh) over the FULL trunk features,
           -> Linear -> tanh scalar in [-1, 1]

The policy/value bodies + scorer + value Linear are laid out to mirror a
PerActionMaskablePolicy checkpoint 1:1 (mlp_extractor.policy_net /
mlp_extractor.value_net / action_scorer / value_net), so ``from_ppo`` can copy
those weights when warm-starting from that flavor. A stock MlpPolicy checkpoint
shares only the CardGameExtractor trunk; its policy/value heads have a different
shape (no per-action tail, a flat action_net), so they start fresh.

``forward(obs, mask) -> (logits, value)`` masks in-graph and is the future
TorchScript export surface (Phase D). ``--export-check`` scripts the net and
compares scripted vs eager outputs to prove that path now.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces

from extractor import CardGameExtractor, _ActionScorer, _PER_ACTION_DIM
try:
    from env import OBS_SIZE, MAX_ACTIONS
    from card_costs import N_CARD_TYPES
    from cli_spec import EMBED_DIM, NET_ARCH
except ImportError:  # pragma: no cover
    from train.env import OBS_SIZE, MAX_ACTIONS
    from train.card_costs import N_CARD_TYPES
    from train.cli_spec import EMBED_DIM, NET_ARCH

_AZ_CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints", "az")


def obs_space_from_const() -> gym.Space:
    """The env's observation Box, built from OBS_SIZE without spawning an engine."""
    return spaces.Box(low=-10.0, high=10.0, shape=(OBS_SIZE,), dtype=np.float32)


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
    """Policy+value net over the shared CardGameExtractor trunk."""

    def __init__(self, observation_space: Optional[gym.Space] = None,
                 embed_dim: int = EMBED_DIM, net_arch=None):
        super().__init__()
        if observation_space is None:
            observation_space = obs_space_from_const()
        arch = list(net_arch) if net_arch is not None else list(NET_ARCH)
        self.embed_dim = int(embed_dim)
        self.obs_size = int(observation_space.shape[0])

        self.trunk = CardGameExtractor(observation_space, embed_dim=embed_dim,
                                       per_action_head=True)
        feat_dim = self.trunk.features_dim               # base + per-action tail
        self._pa_offset = int(self.trunk.per_action_offset)
        self._pa_slots = int(self.trunk.per_action_slots)   # == MAX_ACTIONS
        self._pa_dim = int(self.trunk.per_action_dim)       # == _PER_ACTION_DIM
        latent_dim = arch[-1]

        # Mirror PerActionMaskablePolicy: policy/value bodies consume the FULL
        # trunk features (including the per-action tail, as SB3's mlp_extractor
        # does), then the scorer combines the policy latent with each action's
        # per-action feature.
        self.policy_body = _mlp_body(feat_dim, arch)
        self.value_body = _mlp_body(feat_dim, arch)
        self.action_scorer = _ActionScorer(latent_dim, self._pa_dim)
        self.value_head = nn.Linear(latent_dim, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        """(obs[B,OBS], mask[B,MAX_ACTIONS] bool) -> (masked logits, value[B]).

        Masking is in-graph (illegal actions -> -inf) so the exported module is
        self-contained. TorchScript export surface — keep it script-friendly."""
        features = self.trunk(obs)
        base_end = self._pa_offset
        pa = features[:, base_end:].reshape(features.shape[0], self._pa_slots,
                                            self._pa_dim)
        latent_pi = self.policy_body(features)
        latent_vf = self.value_body(features)
        logits = self.action_scorer(latent_pi, pa)          # (B, MAX_ACTIONS)
        neg = torch.finfo(logits.dtype).min
        logits = torch.where(mask, logits, torch.full_like(logits, neg))
        value = torch.tanh(self.value_head(latent_vf)).squeeze(-1)   # (B,)
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


def az_checkpoint_path(deck: str, steps: Optional[int] = None,
                       checkpoint_dir: str = _AZ_CKPT_DIR) -> str:
    """``{dir}/{deck}__azfinal.pt`` (steps None) or ``{deck}__azv{steps}.pt``.

    A subfolder deck ('league/ur_delver') keeps its subfolder under the az dir."""
    sub = os.path.join(checkpoint_dir, os.path.dirname(deck))
    stem = os.path.basename(deck)
    name = f"{stem}__azfinal.pt" if steps is None else f"{stem}__azv{int(steps)}.pt"
    return os.path.join(sub, name)


def resolve_az_checkpoint(spec: str,
                          checkpoint_dir: str = _AZ_CKPT_DIR) -> Optional[str]:
    """Resolve an AZ checkpoint shorthand/path. Priority: exact path; then
    ``{deck}__azfinal.pt``; then the newest ``{deck}__azv*.pt``. Returns None if
    nothing matches (so callers can fall back to a PPO warm-start)."""
    if spec and os.path.exists(spec):
        return spec
    final = az_checkpoint_path(spec, None, checkpoint_dir)
    if os.path.exists(final):
        return final
    import glob as _glob
    sub = os.path.join(checkpoint_dir, os.path.dirname(spec))
    stem = os.path.basename(spec)
    snaps = _glob.glob(os.path.join(sub, f"{stem}__azv*.pt"))

    def _steps(p):
        try:
            return int(os.path.basename(p).split("__azv")[1].split(".pt")[0])
        except (IndexError, ValueError):
            return -1
    snaps = [p for p in snaps if _steps(p) >= 0]
    if snaps:
        return max(snaps, key=_steps)
    return None


def load_az(path: str, map_location="cpu") -> "AZNet":
    """Load an AZNet from ``path`` (+ its .meta.json), asserting the layout
    handshake against the current OBS_SIZE / MAX_ACTIONS / N_CARD_TYPES."""
    meta = {}
    mp = _meta_path(path)
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
        for key, cur in (("OBS_SIZE", OBS_SIZE), ("MAX_ACTIONS", MAX_ACTIONS),
                         ("N_CARD_TYPES", N_CARD_TYPES)):
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

    transferred: list[str] = []
    # 1) Trunk (features_extractor.* -> trunk.*). Shared by both flavors.
    for k, v in sd.items():
        if not k.startswith("features_extractor."):
            continue
        tk = "trunk." + k[len("features_extractor."):]
        if tk in net_sd and net_sd[tk].shape == v.shape:
            net_sd[tk] = v.clone()
            transferred.append(tk)

    notes: list[str] = []
    if flavor == "per_action":
        # 2) policy/value bodies + scorer + value head map 1:1.
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
        # PPO value_net is Linear(latent,1) with NO tanh; AZNet wraps it in
        # tanh, so the copied weights now feed a tanh — a reasonable warm start.
        for k, v in sd.items():
            if k.startswith("value_net."):
                tk = "value_head." + k[len("value_net."):]
                if tk in net_sd and net_sd[tk].shape == v.shape:
                    net_sd[tk] = v.clone()
                    transferred.append(tk)
        notes.append("value head copied from PPO value_net (now behind tanh)")
    else:
        notes.append("stock MlpPolicy: policy/value heads start fresh "
                     "(shape-incompatible with the per-action AZ head)")

    net.load_state_dict(net_sd)

    fresh = [k for k in net_sd if k not in set(transferred)]
    print(f"[from_ppo] flavor={flavor} embed_dim={embed_dim}: "
          f"transferred {len(transferred)} tensors, {len(fresh)} fresh.")
    for n in notes:
        print(f"[from_ppo]   note: {n}")
    # Summarize the fresh top-level module groups so the caller sees what wasn't
    # warm-started (per-action encoder on stock, both heads on stock, etc.).
    fresh_groups = sorted({k.split(".")[0] + ("." + k.split(".")[1]
                          if k.startswith("trunk.") else "") for k in fresh})
    if fresh:
        print(f"[from_ppo]   fresh groups: {', '.join(fresh_groups)}")
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
# --export-check CLI (proves the TorchScript export path)
# ----------------------------------------------------------------------

def _export_check(embed_dim: int = EMBED_DIM, seed: int = 0,
                  tol: float = 1e-4) -> int:
    torch.manual_seed(seed)
    net = AZNet(obs_space_from_const(), embed_dim=embed_dim).eval()
    obs = torch.randn(3, OBS_SIZE)
    mask = torch.zeros(3, MAX_ACTIONS, dtype=torch.bool)
    for i in range(3):
        mask[i, : (i + 2)] = True   # varying legal counts, all rows non-empty
    with torch.no_grad():
        eager_logits, eager_value = net(obs, mask)
    try:
        scripted = torch.jit.script(net)
    except Exception as e:  # pragma: no cover
        print(f"FAIL: torch.jit.script(AZNet) raised: {e}")
        return 1
    with torch.no_grad():
        s_logits, s_value = scripted(obs, mask)
    # Compare only the LEGAL logits (masked slots are -inf in both; inf-inf=nan).
    dl = 0.0
    for i in range(3):
        n = int(mask[i].sum())
        dl = max(dl, float((eager_logits[i, :n] - s_logits[i, :n]).abs().max()))
    dv = float((eager_value - s_value).abs().max())
    ok = dl <= tol and dv <= tol
    print(f"export-check: max|Δlogits|={dl:.2e} max|Δvalue|={dv:.2e} "
          f"tol={tol:.0e} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AZNet utilities")
    ap.add_argument("--export-check", action="store_true",
                    help="Script the net and compare scripted vs eager outputs")
    ap.add_argument("--from-ppo", default=None,
                    help="Warm-start from a PPO checkpoint and report transfer")
    ap.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    args = ap.parse_args()
    rc = 0
    if args.from_ppo:
        from_ppo(args.from_ppo)
    if args.export_check or not args.from_ppo:
        rc = _export_check(embed_dim=args.embed_dim)
    raise SystemExit(rc)
