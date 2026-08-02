"""PopArt value normalization for the per-archetype-bucket critic (opt-in).

The multi-head critic (``extractor.PerActionMaskablePolicy``) gives every
(self archetype x opponent archetype) value bucket its own final head column,
which removes *last-layer* interference between matchup classes. PopArt (Hessel
et al. 2019, "Multi-task Deep Reinforcement Learning with PopArt") additionally
removes *gradient-scale* interference in the shared torso: each bucket keeps a
running (mu, sigma) of its returns, the head predicts NORMALIZED values, and on
every stats update that bucket's head column is rescaled so the network's output
is preserved ("preserving outputs, precisely adapting").

Wiring — deliberately thin, because the invasive part is where the value loss is
computed:

* the (mu, sigma) statistics and the output-preserving rescale live on the policy
  (``popart_mu`` / ``popart_sigma`` / ``popart_count`` buffers +
  ``popart_update``), so they travel inside the checkpoint;
* rollout collection is untouched: ``policy.forward`` / ``predict_values`` return
  DE-normalized (real-scale) values, so GAE, advantages and the stored
  ``values``/``returns`` are in return space exactly as without PopArt;
* this subclass, once per rollout (the collected batch), updates the stats from
  that rollout's per-bucket returns, then normalizes the buffer's ``returns`` and
  ``values`` in place and flips the policy into normalized-output mode for the
  duration of the stock ``train()`` — so SB3's unmodified value loss compares
  normalized predictions against normalized targets, with no copy of its epoch
  loop here.

Default OFF (``train.py --popart`` enables it). With PopArt off the stats stay at
(0, 1) and every formula above is the identity.
"""

import numpy as np

try:
    from sb3_contrib import MaskablePPO
except ImportError:  # pragma: no cover — mirrors train.py's fallback
    from stable_baselines3 import PPO as MaskablePPO

try:
    from env import BUCKET_IDX
    from archetypes import N_VALUE_BUCKETS, bucket_name
except ImportError:  # pragma: no cover
    from train.env import BUCKET_IDX
    from train.archetypes import N_VALUE_BUCKETS, bucket_name

# Decay of the running per-bucket statistics, applied once per rollout.
POPART_BETA = 0.99
# Floor on a bucket's sigma. Beyond div-by-zero safety, this caps target
# amplification for near-deterministic matchups (the doomsday row sits at
# sigma 0.09-0.22): a bucket pinned at the floor keeps full mean-centering but
# only partial variance equalization, so its upset noise no longer draws an
# equal share of value-loss gradient on top of PFSP already oversampling it.
POPART_EPS = 0.25


class PopArtMaskablePPO(MaskablePPO):
    """MaskablePPO whose value targets are per-bucket PopArt-normalized.

    Requires a policy with the multi-head critic API (``popart_update`` +
    ``popart_normalized_out``), i.e. ``extractor.PerActionMaskablePolicy``.
    ``clip_range_vf`` is rejected: SB3 would clip normalized predictions against
    the stored real-scale values, which is not the same objective.
    """

    def __init__(self, *args, popart_beta: float = POPART_BETA, **kwargs):
        super().__init__(*args, **kwargs)
        self.popart_beta = float(popart_beta)
        # SB3's load() constructs with _init_setup_model=False (no policy yet) and
        # builds it afterwards, so only check when there is something to check;
        # load() re-checks once the policy exists.
        if getattr(self, "policy", None) is not None:
            self._assert_popart_capable()

    def _assert_popart_capable(self) -> None:
        if not hasattr(self.policy, "popart_update"):
            raise RuntimeError(
                "--popart needs the multi-head critic policy "
                "(extractor.PerActionMaskablePolicy); this model's policy is "
                f"{type(self.policy).__name__} — drop --popart or drop --stock-head")
        if self.clip_range_vf is not None:
            raise RuntimeError(
                "--popart is incompatible with clip_range_vf: the value clip would "
                "compare normalized predictions against real-scale stored values")

    # ------------------------------------------------------------------
    def _rollout_buckets(self) -> np.ndarray:
        """Per-sample value-bucket index for the rollout buffer's observations."""
        obs = np.asarray(self.rollout_buffer.observations)
        return np.rint(obs[..., BUCKET_IDX]).astype(np.int64).ravel()

    def _update_popart_stats(self) -> None:
        """Fold this rollout's per-bucket return statistics into the policy's
        PopArt stats (rescaling the affected head columns), then normalize the
        buffer's returns AND values in place with the updated stats.

        Normalizing the stored values too keeps everything SB3 derives from the
        buffer during ``train()`` (its explained-variance log) in one consistent
        space; the advantages were already computed from real-scale values before
        this runs, so they are untouched.
        """
        buf = self.rollout_buffer
        buckets = self._rollout_buckets()
        returns = buf.returns.reshape(-1)
        values = buf.values.reshape(-1)
        if buckets.size != returns.size:
            raise RuntimeError(
                f"popart: rollout buffer has {returns.size} returns but "
                f"{buckets.size} bucket ids — obs layout mismatch")
        present, idx = [], []
        means, second_moments, counts = [], [], []
        for b in np.unique(buckets):
            sel = buckets == b
            r = returns[sel].astype(np.float64)
            present.append(int(b))
            idx.append(sel)
            means.append(float(r.mean()))
            second_moments.append(float((r ** 2).mean()))
            counts.append(float(r.size))
        if not present:
            return
        mu, sigma = self.policy.popart_update(
            present, means, second_moments, counts,
            beta=self.popart_beta, eps=POPART_EPS)
        mu_np = mu.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        for b, sel in zip(present, idx):
            returns[sel] = (returns[sel] - mu_np[b]) / sigma_np[b]
            values[sel] = (values[sel] - mu_np[b]) / sigma_np[b]
            tag = bucket_name(b)
            self.logger.record(f"popart/mu_{tag}", float(mu_np[b]))
            self.logger.record(f"popart/sigma_{tag}", float(sigma_np[b]))
        buf.returns[:] = returns.reshape(buf.returns.shape)
        buf.values[:] = values.reshape(buf.values.shape)
        self.logger.record("popart/n_buckets", len(present))

    def train(self) -> None:
        """Normalize this rollout's targets, then run the stock PPO update."""
        self._update_popart_stats()
        self.policy.popart_normalized_out = True
        try:
            super().train()
        finally:
            self.policy.popart_normalized_out = False

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, *args, **kwargs):
        """Load a checkpoint (PPO or PopArt — the stats live in the policy's
        buffers, so both flavors load) and re-check PopArt compatibility."""
        model = super().load(*args, **kwargs)
        if not hasattr(model, "popart_beta"):
            model.popart_beta = POPART_BETA
        model._assert_popart_capable()
        return model


def popart_stats(policy) -> dict:
    """{bucket_name: (mu, sigma, count)} for every bucket with recorded stats.

    Small introspection helper for reports/debugging (the buffers are otherwise
    only meaningful inside the value loss)."""
    out = {}
    if not hasattr(policy, "popart_mu"):
        return out
    mu = policy.popart_mu.detach().cpu().numpy()
    sigma = policy.popart_sigma.detach().cpu().numpy()
    cnt = policy.popart_count.detach().cpu().numpy()
    for b in range(N_VALUE_BUCKETS):
        if cnt[b] > 0:
            out[bucket_name(b)] = (float(mu[b]), float(sigma[b]), float(cnt[b]))
    return out
