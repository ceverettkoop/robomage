"""The R-NaD learner (docs/rnad_training_plan.md §4 L4, B3–B5): the update of OpenSpiel's
``RNaDSolver`` (``rnad.py`` at 8156646cdef0c753162fc6b68a11d57cce642f95, ``loss`` and
``update_parameters``) over the step-5 network, its state, the iteration boundary and the pushes to
the inference server.

    cfg = RNaDConfig(entropy_schedule_sizes=(10_000,), chunk_len=32)
    learner = RNaDLearner.from_bc_checkpoint("bc/final.pt", tables, cfg, device="cuda")
    learner.add_hook(ServerLink(server, push_every=50))      # pushes now, every 50 steps, at boundaries
    metrics = learner.step(batch)                             # rnad_data.RNaDBatch (or the same names)
    torch.save(learner.state_dict(), "ckpt/learner.pt")       # everything an exact resume needs
    learner = RNaDLearner.from_state_dict(torch.load(...), tables, device="cuda")

**State.** The current network ``model`` (an ``RMNet``, trained), the target network ``target``
(``params_target``: an average of ``model`` at rate ``target_network_avg``), the two
regularisation policies ``regs[0]`` = π_m,reg (the reference's ``params_prev``) and ``regs[1]`` =
π_m−1,reg (``params_prev_``), each a frozen network with a version and a smoothing ε; the Adam
state; ``learner_steps``, ``reg_updates`` (m), the served policy version, the fp16 loss scale and
the torch generators' states.

**One step** (``step``), the reference's ``loss`` + ``update_parameters``:

1. α and the boundary flag from the Δm schedule (``EntropySchedule`` over ``learner_steps``).
2. Per regularisation policy, each step's log π: the actor recorded it (``batch.evals[p]`` for p in
   ``reg_policy_ids``, matched by *version*, so a step recorded under π_m,reg at id 1 is still used
   after a boundary moves that policy to π_m−1,reg); else recomputed here from the frozen network
   (``stale_reg="recompute"``) or the whole game dropped (``"drop"``); both counted.
3. The chunks, last to first: the current network's forward with gradient on the chunk's valid
   steps (logits, value), the target network's value (no gradient); the reward transform
   ``log π − (α·log π_m,reg + (1 − α)·log π_m−1,reg)``; per player v-trace carried across chunks;
   the critic loss and NeuRD normalised over the whole batch's steps; backward.
4. Adam (b1 0, b2 0.999, ε 1e-8, eps_root 0), scaled by −lr, each update clipped to
   ±``clip_gradient`` (the reference's ``optax.chain(scale_by_adam, scale(-lr), clip)``); the target
   network moved toward the new parameters; at a boundary π_m−1,reg ← π_m,reg and π_m,reg ← the
   target network (the reference's ``lax.cond(update_target_net, ...)``); ``hooks``.

**Smoothed regularisation (B3).** π_0,reg is the behaviour-cloned network mixed with
uniform-over-legal: ``(1 − ε)·π + ε/n`` with ``reg_eps`` (``reg_eps_sideboarding`` at sideboarding
steps). The smoothing is **learner-side**: the server serves every regularisation policy's raw
network, the actors record its raw log π, and the learner applies the policy's (ε, ε_sb) to the
recorded or recomputed log π with the step's ``sideboarding`` flag (``model.smoothed_log_policy``).
Nothing on the server or in the tensors has to know which decisions are sideboarding. Later
regularisation policies (target networks) have ε = 0.

**The server** (``ServerLink``): id ``current_id`` (0) serves ``model`` at version
``learner_steps`` (pushed at attach and every ``push_every`` steps), ids ``reg_policy_ids`` (1, 2)
serve π_m,reg and π_m−1,reg's raw networks at their versions (at attach and at each boundary). A
regularisation policy's version is the learner step at which it was frozen (π_0,reg:
``initial_reg_version``, 0), so versions never repeat.

**Omniscient critic** (``ModelConfig.omniscient_critic``, an A/B flag): the batch's ``omni``
inputs feed the critic, whose value (the target network's, for v-trace; the current network's, for
the critic loss) replaces the seat value head's; the seat value head is still regressed to the same
v-trace targets (``critic_trains_value_head``) so the exported policy's value stays meaningful.

**Steps without μ** cannot be trained on (v-trace's importance ratios divide by μ(a)):
``rnad_data.filter_episodes`` raises or drops them, and ``step`` raises if one reaches it.

**Differences from the reference** (and rnad.py's list, which this module inherits):

1. *Regularisation log π from the trajectory.* The reference recomputes ``log_pi_prev`` and
   ``log_pi_prev_`` with the frozen networks every step; here the inference server's evaluations,
   recorded by the actor, are used when their versions match (step 2), and only the rest are
   recomputed. Equal up to float (and the server's precision).
2. *Irregular seats and variable games.* Batches are full games padded to the longest (T rounded
   up to ``chunk_len``) and scanned chunk by chunk (rnad.py difference 2); the seat sequence is
   whatever the engine asked, which the reference's per-step ``player_id`` already allows.
3. *The value head is ``tanh``* (the model's, shared with behaviour cloning and the exports), where
   the reference's is linear; the v-trace targets of the transformed game can leave [−1, 1] by the
   accumulated η·KL terms. ``value_activation="linear"`` reads the head before its tanh.
4. *Adam ε* is 1e-8 (DeepNash's supplement, the plan); the reference's ``AdamConfig`` writes
   ``10e-8`` (= 1e-7).
5. *Fine-tuning discretisation* defaults to 16 steps (the plan and the nn: agent's ``disc=16``),
   the reference's to 32; the rounding is the engine's (rnad.py difference 3).
6. *Schedule changes.* The schedule is the reference's ``EntropySchedule``, a function of
   ``learner_steps`` (so a changed ``entropy_schedule_sizes`` reinterprets the run's position).
"""

import copy
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as Fn

from . import rnad as R
from .batch import INPUTS
from .model import ModelConfig, RMNet, load_checkpoint, smoothed_log_policy


@dataclass
class RNaDConfig:
    # The reward transform's weight η.
    eta: float = 0.2
    # Adam, as the reference's optax chain (scale_by_adam, scale(-lr), clip).
    learning_rate: float = 5e-5
    adam_b1: float = 0.0
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    clip_gradient: float = 10_000.0
    # The rate at which the target network follows the current one.
    target_network_avg: float = 0.001
    # Δm: iteration sizes and their repeats (the last size repeats forever).
    entropy_schedule_sizes: tuple = (10_000,)
    entropy_schedule_repeats: tuple = (1,)
    # NeuRD's logit threshold β and advantage clip; v-trace's c̄ and ρ̄ (the reference passes ρ̄ = ∞).
    nerd_beta: float = 2.0
    nerd_clip: float = 10_000.0
    c_vtrace: float = 1.0
    rho_vtrace: float = math.inf
    # Fine-tuning (B5): from this learner step on (negative: never) the merged policy is projected.
    finetune_from_learner_steps: int = -1
    finetune_threshold: float = 0.03
    finetune_discretization: int = 16
    # Time steps per chunk (forward + backward unit; memory ~ chunk_len × games).
    chunk_len: int = 32
    # π_0,reg's smoothing (B3) and the served ids of the regularisation policies.
    reg_eps: float = 0.1
    reg_eps_sideboarding: float = 0.25
    reg_policy_ids: tuple = (1, 2)
    initial_reg_version: int = 0
    # Steps whose recorded regularisation versions are not the learner's: "recompute" or "drop".
    stale_reg: str = "recompute"
    value_activation: str = "tanh"
    critic_trains_value_head: bool = True
    # Forward precision ("fp32", "fp16" with dynamic loss scaling, "bf16"); the loss math is fp32.
    precision: str = "fp32"
    loss_scale_init: float = 2.0 ** 15
    loss_scale_period: int = 1000

    def to_dict(self):
        d = asdict(self)
        d["rho_vtrace"] = None if math.isinf(self.rho_vtrace) else self.rho_vtrace
        return d

    @classmethod
    def from_dict(cls, d):
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if d.get("rho_vtrace", 0) is None:
            d["rho_vtrace"] = math.inf
        for k in ("entropy_schedule_sizes", "entropy_schedule_repeats", "reg_policy_ids"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


@dataclass
class RegPolicy:
    """A frozen regularisation policy: its network, version and smoothing (ε, ε at sideboarding)."""
    model: RMNet
    version: int
    eps: float = 0.0
    eps_sideboarding: float = 0.0

    def smoothed(self):
        return self.eps > 0 or self.eps_sideboarding > 0


def _frozen(model):
    m = copy.deepcopy(model)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class _Adam:
    """optax.chain(scale_by_adam(b1, b2, eps, eps_root=0), scale(-lr), clip(clip)) over a parameter
    list (torch._foreach ops)."""

    def __init__(self, params, lr, b1, b2, eps, clip):
        self.params = params
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, b1, b2, eps, clip
        self.t = 0
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]

    @torch.no_grad()
    def step(self, grads):
        self.t += 1
        torch._foreach_mul_(self.m, self.b1)
        torch._foreach_add_(self.m, grads, alpha=1.0 - self.b1)
        torch._foreach_mul_(self.v, self.b2)
        torch._foreach_addcmul_(self.v, grads, grads, value=1.0 - self.b2)
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        den = torch._foreach_div(self.v, bc2)
        torch._foreach_sqrt_(den)
        torch._foreach_add_(den, self.eps)
        upd = torch._foreach_div(self.m, den)
        torch._foreach_mul_(upd, -self.lr / bc1)
        torch._foreach_clamp_min_(upd, -self.clip)
        torch._foreach_clamp_max_(upd, self.clip)
        torch._foreach_add_(self.params, upd)
        return torch.stack([u.abs().max() for u in upd]).max() if upd else torch.zeros(())

    def state_dict(self):
        return {"t": self.t, "m": [x.detach().cpu().clone() for x in self.m],
                "v": [x.detach().cpu().clone() for x in self.v]}

    def load_state_dict(self, sd):
        self.t = int(sd["t"])
        for dst, src in zip(self.m, sd["m"]):
            dst.copy_(src)
        for dst, src in zip(self.v, sd["v"]):
            dst.copy_(src)


class ServerLink:
    """Pushes the learner's policies to an ``InferenceServer`` (server.py): the current network at
    id ``current_id`` on attach and every ``push_every`` learner steps, the regularisation policies'
    raw networks at ``reg_policy_ids`` on attach and at each iteration boundary (with the current
    network). ``server`` needs only ``set_policy(pid, policy, version)``."""

    def __init__(self, server, current_id=0, push_every=100):
        self.server = server
        self.current_id = current_id
        self.push_every = push_every
        self.pushes = 0
        self.current_at = None

    def push_current(self, learner):
        if self.current_at == learner.learner_steps:
            return                                  # already served at this step
        self.current_at = learner.learner_steps
        self.server.set_policy(self.current_id, learner.model, learner.learner_steps)
        learner.policy_version = learner.learner_steps
        self.pushes += 1

    def push_regs(self, learner):
        for pid, reg in zip(learner.cfg.reg_policy_ids, learner.regs):
            self.server.set_policy(pid, reg.model, reg.version)
            self.pushes += 1

    def __call__(self, learner, event):
        if event in ("attach", "iteration"):
            self.push_regs(learner)
            self.push_current(learner)
        elif event == "step" and self.push_every > 0 and learner.learner_steps % self.push_every == 0:
            self.push_current(learner)


class RNaDLearner:
    """See the module docstring."""

    def __init__(self, model, cfg=None, device="cpu", regs=None):
        self.cfg = cfg = cfg or RNaDConfig()
        if cfg.stale_reg not in ("recompute", "drop"):
            raise ValueError("stale_reg is 'recompute' or 'drop'")
        if cfg.value_activation not in ("tanh", "linear"):
            raise ValueError("value_activation is 'tanh' or 'linear'")
        if cfg.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision is fp32, fp16 or bf16")
        if len(cfg.reg_policy_ids) != 2:
            raise ValueError("reg_policy_ids names the two regularisation policies' server ids")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.train()
        self.target = _frozen(self.model)
        if regs is None:
            r0 = RegPolicy(_frozen(self.model), cfg.initial_reg_version, cfg.reg_eps, cfg.reg_eps_sideboarding)
            regs = [r0, r0]
        self.regs = regs
        for r in self.regs:
            r.model.to(self.device)
        self.schedule = R.EntropySchedule(sizes=cfg.entropy_schedule_sizes, repeats=cfg.entropy_schedule_repeats)
        self.finetune = R.FineTuning(cfg.finetune_from_learner_steps, cfg.finetune_threshold,
                                     cfg.finetune_discretization)
        self.params = [p for _, p in self.model.named_parameters()]
        self.target_params = [p for _, p in self.target.named_parameters()]
        self.opt = _Adam(self.params, cfg.learning_rate, cfg.adam_b1, cfg.adam_b2, cfg.adam_eps, cfg.clip_gradient)
        self.learner_steps = 0
        self.reg_updates = 0
        self.policy_version = 0
        self.loss_scale = cfg.loss_scale_init if cfg.precision == "fp16" else 1.0
        self.good_steps = 0
        self.totals = {"steps_skipped": 0, "reg_recorded": 0, "reg_recomputed": 0, "games_dropped_stale": 0,
                       "decisions": 0, "games": 0}
        self.hooks = []

    # ── construction ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_bc_checkpoint(cls, path, tables, cfg=None, device="cpu", model_cfg=None):
        """B3: the network = the behaviour-cloned policy (and its value head); π_0,reg = π_−1,reg =
        that policy smoothed (``reg_eps``, ``reg_eps_sideboarding``). `model_cfg` overrides the
        checkpoint's config fields that do not change the parameters' shapes (e.g. id_dropout); a
        critic absent from the checkpoint starts from its initialisation."""
        model = load_checkpoint(path, tables) if model_cfg is None else _load_with_config(path, tables, model_cfg)
        return cls(model, cfg, device)

    @classmethod
    def from_state_dict(cls, sd, tables, device="cpu"):
        """A learner rebuilt from ``state_dict()`` (over `tables`, whose hashes must be the model's)."""
        for k in ("spec_hash", "vocab_hash", "text_model"):
            if sd["hashes"].get(k) != tables.hashes[k]:
                raise ValueError("learner state: %s %s is not the tables' %s" % (k, sd["hashes"].get(k),
                                                                                tables.hashes[k]))
        model = RMNet(ModelConfig.from_dict(sd["model_config"]), tables)
        learner = cls(model, RNaDConfig.from_dict(sd["config"]), device)
        learner.load_state_dict(sd)
        return learner

    def configure(self, **changes):
        """Changes config fields mid-run (``rmtrain set``: learning_rate, eta, the schedule, the
        fine-tuning, the regularisation smoothing of later seeds, ...), updating what depends on them.
        A changed schedule is read at the current learner step (the reference's EntropySchedule is a
        function of the step), so it can move α and the next boundary."""
        unknown = set(changes) - set(RNaDConfig.__dataclass_fields__)
        if unknown:
            raise ValueError("configure: no config field %s" % ", ".join(sorted(unknown)))
        d = self.cfg.to_dict()
        d.update(changes)
        cfg = RNaDConfig.from_dict(d)
        self.cfg = cfg
        self.opt.lr, self.opt.b1, self.opt.b2, self.opt.eps, self.opt.clip = (
            cfg.learning_rate, cfg.adam_b1, cfg.adam_b2, cfg.adam_eps, cfg.clip_gradient)
        self.schedule = R.EntropySchedule(sizes=cfg.entropy_schedule_sizes, repeats=cfg.entropy_schedule_repeats)
        self.finetune = R.FineTuning(cfg.finetune_from_learner_steps, cfg.finetune_threshold,
                                     cfg.finetune_discretization)
        return cfg

    def add_hook(self, hook):
        """`hook(learner, event)` is called with "attach" now, "iteration" after each boundary (the
        regularisation policies moved) and "step" after every step (after "iteration")."""
        self.hooks.append(hook)
        hook(self, "attach")

    def _fire(self, event):
        for h in self.hooks:
            h(self, event)

    # ── state ─────────────────────────────────────────────────────────────────────────────

    def state_dict(self):
        """Everything an exact resume needs (tensors on the CPU; torch.save-able)."""
        def sd(m):
            return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        regs = []
        for i, r in enumerate(self.regs):
            shared = next((j for j in range(i) if self.regs[j] is r), None)
            regs.append({"same_as": shared} if shared is not None else
                        {"state": sd(r.model), "version": r.version, "eps": r.eps,
                         "eps_sideboarding": r.eps_sideboarding})
        rng = {"torch": torch.get_rng_state()}
        if self.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(self.device)
        return {"format": 1, "reference": R.REFERENCE, "config": self.cfg.to_dict(),
                "model_config": self.model.cfg.to_dict(), "hashes": dict(self.model.hashes),
                "model": sd(self.model), "target": sd(self.target), "regs": regs, "adam": self.opt.state_dict(),
                "learner_steps": self.learner_steps, "reg_updates": self.reg_updates,
                "policy_version": self.policy_version, "loss_scale": self.loss_scale,
                "good_steps": self.good_steps, "totals": dict(self.totals), "rng": rng}

    def load_state_dict(self, sd):
        """Restores ``state_dict()`` into this learner (same architecture), the global torch
        generators included (the id-dropout draws come from them)."""
        self.model.load_state_dict(sd["model"])
        self.target.load_state_dict(sd["target"])
        regs = []
        for r in sd["regs"]:
            if r.get("same_as") is not None:
                regs.append(regs[r["same_as"]])
                continue
            m = _frozen(self.model)
            m.load_state_dict(r["state"])
            regs.append(RegPolicy(m.to(self.device), int(r["version"]), float(r["eps"]),
                                  float(r["eps_sideboarding"])))
        self.regs = regs
        self.opt.load_state_dict(sd["adam"])
        self.learner_steps = int(sd["learner_steps"])
        self.reg_updates = int(sd["reg_updates"])
        self.policy_version = int(sd["policy_version"])
        self.loss_scale = float(sd["loss_scale"])
        self.good_steps = int(sd["good_steps"])
        self.totals = dict(sd["totals"])
        torch.set_rng_state(sd["rng"]["torch"])
        if "cuda" in sd["rng"] and self.device.type == "cuda":
            torch.cuda.set_rng_state(sd["rng"]["cuda"], self.device)

    # ── the step ──────────────────────────────────────────────────────────────────────────

    def _autocast(self):
        if self.cfg.precision == "fp32":
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type,
                              dtype=torch.float16 if self.cfg.precision == "fp16" else torch.bfloat16)

    def _inputs(self, x):
        return [torch.from_numpy(x[k]).to(self.device, non_blocking=True) for k in INPUTS]

    def _value(self, model, out, critic):
        """The value the learner trains / bootstraps from: the critic's when `critic`, else the seat
        head's (before its tanh under value_activation="linear")."""
        if critic:
            return out["critic_value_raw"] if self.cfg.value_activation == "linear" else out["critic_value"]
        if self.cfg.value_activation == "linear":
            return model.value.fc2(Fn.gelu(model.value.fc1(out["summary"]))).squeeze(-1)
        return out["value"]

    def _forward(self, model, inputs, omni, seat_rows):
        """The model's outputs (+ ``critic_value_raw`` under value_activation="linear")."""
        out = model(*inputs, omni=omni, seat=seat_rows)
        if omni is not None and self.cfg.value_activation == "linear":
            c = model.critic
            o = omni
            extra = model.cards.gather(o[3], o[4], model.cfg.id_dropout) + c.seat_emb(seat_rows).unsqueeze(1)
            xo = c.encoder(o[0], o[1], o[2], o[5], o[6], o[7], extra)
            g = c.summary(xo, o[5])
            out["critic_value_raw"] = c.value.fc2(Fn.gelu(c.value.fc1(g))).squeeze(-1)
        return out

    def _reg_log_pi(self, batch, k, legal, valid):
        """Regularisation policy k's recorded log π [T, G, A] and where it is known [T, G]."""
        reg = self.regs[k]
        T, G, A = legal.shape
        lp = torch.zeros(T, G, A, device=self.device)
        known = torch.zeros(T, G, dtype=torch.bool, device=self.device)
        for pid in self.cfg.reg_policy_ids:
            ev = batch.evals.get(pid) if hasattr(batch, "evals") else None
            if ev is None:
                continue
            hit = torch.from_numpy(ev["present"] & (ev["version"] == reg.version)).to(self.device) & ~known
            if hit.any():
                lp = torch.where(hit.unsqueeze(-1), torch.from_numpy(ev["log_pi"]).to(self.device), lp)
                known |= hit
        return lp, known & valid

    def _smooth(self, reg, lp, legal, sideboarding):
        """The reference convention (0 at illegal rows) of reg's smoothed log π, from its raw log π."""
        if not reg.smoothed():
            return torch.where(legal, lp, torch.zeros_like(lp))
        shape = lp.shape
        a = shape[-1]
        x = torch.where(legal, lp, torch.full_like(lp, -float("inf"))).reshape(-1, a)
        s = smoothed_log_policy(x, legal.reshape(-1, a), reg.eps, sideboarding.reshape(-1), reg.eps_sideboarding)
        return torch.where(legal, s.reshape(shape), torch.zeros_like(lp))

    def step(self, batch):
        """One learner step on `batch` (an ``rnad_data.RNaDBatch`` or an object with its names);
        returns a metrics dict."""
        cfg = self.cfg
        t_start = time.time()
        alpha, update_target_net = self.schedule(self.learner_steps)
        finetune_on = self.finetune.enabled(self.learner_steps)
        info = self.accumulate_gradients(batch, alpha, finetune_on)
        G, T = batch.G, batch.T
        sums, n_valid, dropped_games = info["sums"], info["decisions"], info["games_dropped_stale"]
        recorded, recomputed = info["reg_recorded"], info["reg_recomputed"]
        scale = self.loss_scale
        # The update.
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        skipped = False
        if scale != 1.0:
            torch._foreach_mul_(grads, 1.0 / scale)
        norms = torch.stack(torch._foreach_norm(grads))
        grad_norm = float(torch.linalg.vector_norm(norms))
        finite = math.isfinite(grad_norm)
        if not finite:
            if cfg.precision != "fp16":
                raise FloatingPointError("R-NaD step %d: non-finite gradients" % self.learner_steps)
            skipped = True
            self.loss_scale = max(self.loss_scale / 2.0, 1.0)
            self.good_steps = 0
            self.totals["steps_skipped"] += 1
        update_max = 0.0
        boundary = False
        if not skipped:
            update_max = float(self.opt.step(grads))
            with torch.no_grad():
                torch._foreach_lerp_(self.target_params, self.params, cfg.target_network_avg)
            if cfg.precision == "fp16":
                self.good_steps += 1
                if self.good_steps % cfg.loss_scale_period == 0:
                    self.loss_scale = min(self.loss_scale * 2.0, 2.0 ** 24)
            self.learner_steps += 1
            if update_target_net:
                self._iteration_boundary()
                boundary = True
        for p in self.params:
            p.grad = None
        self.totals["reg_recorded"] += recorded
        self.totals["reg_recomputed"] += recomputed
        self.totals["games_dropped_stale"] += dropped_games
        self.totals["decisions"] += n_valid
        self.totals["games"] += G - dropped_games
        sums = {k: float(v) for k, v in sums.items()}
        nv = max(n_valid, 1)
        pv = batch.policy_version[batch.valid]
        metrics = {"learner_steps": self.learner_steps, "reg_updates": self.reg_updates, "alpha": alpha,
                   "boundary": boundary, "finetune": finetune_on, "skipped": skipped,
                   "loss": sums["loss_v"] + sums["loss_nerd"] + sums["loss_value_head"],
                   "loss_v": sums["loss_v"], "loss_nerd": sums["loss_nerd"],
                   "loss_value_head": sums["loss_value_head"], "kl_reg": sums["kl_reg"] / nv,
                   "entropy": sums["entropy"] / nv, "adv_abs": sums["adv_abs"] / nv,
                   "v_target": sums["v_target"] / nv, "ratio_pi_mu": sums["ratio"] / nv,
                   "grad_norm": grad_norm, "update_max": update_max, "loss_scale": self.loss_scale,
                   "decisions": n_valid, "games": G - dropped_games, "T": T,
                   "reg_recorded": recorded, "reg_recomputed": recomputed, "games_dropped_stale": dropped_games,
                   "policy_lag": float(self.learner_steps - pv.mean()) if len(pv) and (pv >= 0).all() else -1.0,
                   "seconds": time.time() - t_start}
        if not skipped:
            if boundary:
                self._fire("iteration")
            self._fire("step")
        return metrics

    def accumulate_gradients(self, batch, alpha, finetune_on=False):
        """The reference's loss on `batch` with mixing weight `alpha`, back-propagated chunk by chunk
        into the parameters' ``.grad`` (scaled by the loss scale; cleared first). Returns ``sums``
        (per-batch sums of the losses and diagnostics, tensors), ``decisions``, ``reg_recorded``,
        ``reg_recomputed``, ``games_dropped_stale``."""
        cfg = self.cfg
        dev = self.device
        T, G, A, L = batch.T, batch.G, batch.A, batch.chunk_len
        nt = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
        valid = nt(batch.valid)
        if bool((valid & ~nt(batch.has_mu)).any()):
            raise ValueError("R-NaD step: a valid step has no behaviour distribution (filter_episodes drops "
                             "or refuses such episodes)")
        seat = nt(batch.seat)
        legal = nt(batch.legal)
        mu = nt(batch.mu)
        chosen = nt(batch.chosen)
        reward = nt(batch.reward)
        sideboarding = nt(batch.sideboarding)
        if bool(((chosen < 0) & valid).any()):
            raise ValueError("R-NaD step: a valid step without a chosen menu row")
        # Invalid steps get a dummy one-row menu (the reference's dummy state), so no NaN arises.
        inv = ~valid
        row0 = torch.zeros(A, dtype=torch.bool, device=dev)
        row0[0] = True
        legal = torch.where(inv.unsqueeze(-1), row0, legal)
        mu = torch.where(inv.unsqueeze(-1), row0.float(), mu)
        chosen = torch.where(inv, torch.zeros_like(chosen), chosen)
        actions_oh = Fn.one_hot(chosen, A).float()
        if bool((((actions_oh * mu).sum(-1) <= 0) & valid).any()):
            raise ValueError("R-NaD step: a chosen row with μ = 0")
        # Regularisation policies: recorded where the version matches, else recompute or drop.
        reg_lp, reg_known = [], []
        for k in range(2):
            lp, known = self._reg_log_pi(batch, k, legal, valid)
            reg_lp.append(lp)
            reg_known.append(known)
        missing = valid & ~(reg_known[0] & reg_known[1])
        dropped_games = 0
        if cfg.stale_reg == "drop" and bool(missing.any()):
            keep = ~missing.any(0)
            dropped_games = int((~keep).sum())
            valid = valid & keep.unsqueeze(0)
            missing = missing & keep.unsqueeze(0)
        recomputed = int(sum(int((valid & ~reg_known[k]).sum()) for k in range(2))) \
            if cfg.stale_reg == "recompute" else 0
        recorded = int(sum(int((valid & reg_known[k]).sum()) for k in range(2)))
        n_player = [(valid & (seat == p)).sum().float() for p in range(2)]
        n_valid = int(valid.sum())
        critic = self.model.critic is not None
        if critic and not getattr(batch, "omniscient", False):
            raise ValueError("R-NaD step: the omniscient critic needs a batch with the omniscient inputs")
        carry = [None, None]
        sums = {k: torch.zeros((), device=dev) for k in ("loss_v", "loss_nerd", "loss_value_head", "kl_reg",
                                                         "entropy", "adv_abs", "v_target", "ratio")}
        for p in self.params:
            p.grad = None
        scale = self.loss_scale
        for c in range(batch.chunks - 1, -1, -1):
            t0, t1 = c * L, (c + 1) * L
            sl = slice(t0, t1)
            x, flat = batch.chunk_inputs(c)
            idx = torch.from_numpy(np.asarray(flat, np.int64) - t0 * G).to(dev)
            logits = torch.zeros(L * G, A, device=dev)
            v = torch.zeros(L * G, device=dev)
            vt = torch.zeros(L * G, device=dev)
            v_head = None
            lps = [reg_lp[k][sl].clone() for k in range(2)]
            if x is not None:
                inputs = self._inputs(x)
                seat_rows = seat[sl].reshape(-1)[idx]
                omni = None
                if critic:
                    xo, _ = batch.chunk_inputs(c, viewer="omni")
                    omni = self._inputs(xo)[:8]
                with self._autocast():
                    out = self._forward(self.model, inputs, omni, seat_rows)
                    with torch.no_grad():
                        tout = self._forward(self.target, inputs, omni, seat_rows)
                logits = logits.index_copy(0, idx, out["logits"].float())
                v = v.index_copy(0, idx, self._value(self.model, out, critic).float())
                vt = vt.index_copy(0, idx, self._value(self.target, tout, critic).float().detach())
                if critic and cfg.critic_trains_value_head:
                    v_head = torch.zeros(L * G, device=dev).index_copy(
                        0, idx, self._value(self.model, out, False).float())
                if cfg.stale_reg == "recompute":
                    for k in range(2):
                        need = (valid[sl] & ~reg_known[k][sl]).reshape(-1)[idx]
                        if k == 1 and self.regs[1] is self.regs[0]:
                            # π_0,reg in both slots: the rows slot 0 recomputed are slot 1's too.
                            lps[1] = torch.where((valid[sl] & ~reg_known[1][sl]).unsqueeze(-1), lps[0], lps[1])
                            break
                        if bool(need.any()):
                            rows = torch.nonzero(need).squeeze(-1)
                            sub = [t.index_select(0, rows) for t in inputs]
                            with torch.no_grad(), self._autocast():
                                rl = self.regs[k].model(*sub)["logits"].float()
                            rl = torch.log_softmax(rl.masked_fill(~sub[9], -float("inf")), -1)
                            flat_lp = lps[k].reshape(L * G, A)
                            flat_lp[idx[rows]] = torch.where(sub[9], rl, torch.zeros_like(rl))
            logits = logits.view(L, G, A)
            v = v.view(L, G, 1)
            vt = vt.view(L, G, 1)
            lg, val_c, seat_c = legal[sl], valid[sl], seat[sl]
            logits = torch.where(lg, logits, torch.zeros_like(logits))
            lps = [self._smooth(self.regs[k], lps[k], lg, sideboarding[sl]) for k in range(2)]
            pi = R.legal_policy(logits, lg)
            log_pi = R.legal_log_policy(logits, lg)
            pi_d = pi.detach()
            policy_pprocessed = self.finetune.post_process_policy(pi_d, lg) if finetune_on else pi_d
            log_policy_reg = (log_pi - (alpha * lps[0] + (1 - alpha) * lps[1])).detach()
            v_targets, has_played, q = [], [], []
            for p in range(2):
                vt_p, hp_p, q_p, carry[p] = R.v_trace(
                    vt, val_c, seat_c, mu[sl], policy_pprocessed, log_policy_reg,
                    R._player_others(seat_c, val_c, p), actions_oh[sl], reward[sl, :, p], p,
                    eta=cfg.eta, lambda_=1.0, c=cfg.c_vtrace, rho=cfg.rho_vtrace, carry=carry[p])
                v_targets.append(vt_p)
                has_played.append(hp_p)
                q.append(q_p)
            loss_v = R.get_loss_v([v] * 2, v_targets, has_played, normalization=n_player)
            is_vector = torch.ones_like(val_c, dtype=torch.float32).unsqueeze(-1)
            loss_nerd = R.get_loss_nerd([logits] * 2, [pi_d] * 2, q, val_c, seat_c, lg, [is_vector] * 2,
                                        clip=cfg.nerd_clip, threshold=cfg.nerd_beta, normalization=n_player)
            loss = loss_v + loss_nerd
            if v_head is not None:
                lvh = R.get_loss_v([v_head.view(L, G, 1)] * 2, v_targets, has_played, normalization=n_player)
                loss = loss + lvh
                sums["loss_value_head"] += lvh.detach()
            if x is not None and loss.requires_grad:
                (loss * scale).backward()
            with torch.no_grad():
                vf = val_c.float()
                sums["loss_v"] += loss_v.detach()
                sums["loss_nerd"] += loss_nerd.detach()
                kl = (pi_d * (log_pi.detach() - (alpha * lps[0] + (1 - alpha) * lps[1]))).sum(-1)
                sums["kl_reg"] += (kl * vf).sum()
                sums["entropy"] += (-(pi_d * log_pi.detach()).sum(-1) * vf).sum()
                own = [torch.where(seat_c == p, 1.0, 0.0) * vf for p in range(2)]
                adv = sum((q[p] - (pi_d * q[p]).sum(-1, keepdim=True)).abs().sum(-1) * own[p] for p in range(2))
                sums["adv_abs"] += adv.sum()
                sums["v_target"] += sum((v_targets[p].squeeze(-1) * own[p]).sum() for p in range(2))
                sums["ratio"] += (((actions_oh[sl] * pi_d).sum(-1) / (actions_oh[sl] * mu[sl]).sum(-1)) * vf).sum()
        return {"sums": sums, "decisions": n_valid, "reg_recorded": recorded, "reg_recomputed": recomputed,
                "games_dropped_stale": dropped_games}

    def _iteration_boundary(self):
        """π_m−1,reg ← π_m,reg; π_m,reg ← the target network (unsmoothed), version = the learner step."""
        new = RegPolicy(_frozen(self.target), self.learner_steps, 0.0, 0.0)
        self.regs = [new, self.regs[0]]
        self.reg_updates += 1


def _load_with_config(path, tables, model_cfg):
    data = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("spec_hash", "vocab_hash", "text_model"):
        if data["hashes"].get(k) != tables.hashes[k]:
            raise ValueError("%s: the checkpoint's %s is not the tables'" % (path, k))
    cfg = ModelConfig.from_dict({**data["config"], **(model_cfg if isinstance(model_cfg, dict) else
                                                      model_cfg.to_dict())})
    model = RMNet(cfg, tables)
    missing, unexpected = model.load_state_dict(data["state_dict"], strict=False)
    bad = [k for k in missing if not k.startswith("critic.")] + list(unexpected)
    if bad:
        raise ValueError("%s: the checkpoint does not fit the model config: %s" % (path, bad[:5]))
    return model
