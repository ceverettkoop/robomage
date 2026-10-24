"""Behaviour cloning (docs/rnad_training_plan.md §4 L3, §6 B1): the network imitates the label of
each step of a replay window (the teacher's, or the reply of a replayed scripted seat).

    trainer = BCTrainer(model, window, BCConfig(batch_size=128), device="cuda", categories=cats)
    m = trainer.train(2000)            # dict: losses, agreement, lr, decisions/s, loader wait, step
    e = trainer.evaluate()             # dict: held-out agreement (overall, per group, per category), value, aux
    trainer.save("run/ckpt/bc_2000.pt", version=1)   # model checkpoint + optimizer state beside it
    trainer.export("run/ckpt/p1", version=1)         # p1.ts.pt, p1.pt2 for nn:<model>
    trainer.close()

    python -m rmlearn.bc --tables DIR --episodes DIR [DIR ...] --out RUN [--minutes 25] [--device cuda]
        [--heldout 0.05] [--heldout-by game|match] [--split-seed N] [--eval-train F]

The command line evaluates the held-out games every ``--eval-every`` steps and, with
``--eval-train F``, a sample of the training games too (the fraction F of them, by the split's hash;
their keys ``evaltrain_*``: the train / held-out gap).

**Losses.**
- Policy: cross-entropy of the masked log-softmax over the menu rows to the label, with label
  smoothing ``label_smoothing`` (the smoothed target puts ε uniformly over the legal rows). The
  batch's mean is weighted: a sideboarding decision weighs ``sideboard_weight``, any other 1
  (``sum(w·loss) / sum(w)``; at 1 the plain mean). The training metrics carry the sideboarding
  decisions' own loss and agreement (``train_policy_sideboarding``, ``train_agree_sideboarding``,
  ``train_frac_sideboarding``; left out of a call that drew none).
- Value: mean squared error of the value head (tanh, the contract's one value) to the deciding
  seat's game result (+1 / -1 / 0), weighted ``value_weight``. A scalar regression rather than a
  win/draw/loss head, so the exported value is what the actor and R-NaD read without a change.
- Auxiliary: ``model.aux_losses`` (the opponent's hand, both libraries' tops)
  on steps of omniscient episodes, weighted ``aux_weight`` each (``aux_head_weights`` overrides
  per head). Id-dropout is the model's own (``ModelConfig.id_dropout``).
- Event dropout (``event_dropout`` p, a spec with recent-event history only, docs/featurizer.md
  §3.7): in training each history event token is masked out with probability p, its card tokens
  with it (``event_dropout``), so the policy cannot lean on one remembered event.

Adam (or AdamW with ``weight_decay``), linear warm-up then an optional cosine decay to ``lr_min``,
gradient clipping by global norm, batches drawn ``sort_pool`` at a time and cut by token count
(uniform over the window's decisions, less padding per batch), mixed precision by autocast (``amp`` "fp16" with a gradient
scaler, "bf16", or "off"). Batches come from a ``replay.Prefetcher`` (background threads expanding
and copying the next batches while the GPU computes).

**Evaluation** walks the window's held-out games (split by game, ``WindowConfig.heldout_fraction``):
top-1 agreement with the label overall, per decision group (``mulligan``: a menu with a mulligan or
keep row; ``sideboarding``; ``in_game``: the rest; with each group's unweighted policy loss) and per
action category of the label's menu row (the menu token's ``category`` slot, found in the spec by name;
the names the spec's ``enums``), the value's MSE and sign agreement on decisive games, and the auxiliary heads'
top-1 accuracies (the hand's: the top card is in the hand).
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from . import export as EX
from . import model as M
from . import replay as RP

MULLIGAN_CATEGORIES = ("MULLIGAN", "KEEP_HAND")
AUX_HEADS = ("hand", "own_library", "opp_library")


@dataclass
class BCConfig:
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    warmup: int = 300
    decay_steps: int = 0          # cosine decay to lr_min over this many steps after warm-up (0: constant)
    lr_min: float = 3e-5
    label_smoothing: float = 0.05
    event_dropout: float = 0.0    # history event tokens masked out in training (with their cards)
    value_weight: float = 0.5
    aux_weight: float = 0.05
    aux_head_weights: dict = field(default_factory=dict)
    grad_clip: float = 1.0
    sideboard_weight: float = 1.0  # the policy loss's weight of a sideboarding decision (others 1)
    amp: str = "fp16"             # "fp16" | "bf16" | "off" (only on a GPU)
    loader_workers: int = 2
    sort_pool: int = 4            # draw this many batches at once, sorted by token count (less padding)
    prefetch: int = 8
    seed: int = 0
    eval_limit: int = 30000       # held-out decisions per evaluation
    eval_batch: int = 256

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Categories:
    """The menu token's action-category slot, from the spec: ``slot`` (its column in ``tok_cat``),
    ``offset`` (its first id, the none) and ``names`` (the enum's names in registry order, from the
    spec unless given)."""

    def __init__(self, spec, names=None):
        self.slot = self.offset = None
        self.names = names if names is not None else spec.enum_values("action_category")
        t = spec.type_by_id[spec.menu_type]
        for c in t.get("categorical", []):
            if c.get("column") == "category":
                self.slot, self.offset, self.size = c["col"], c["offset"], c["size"]
        self.mulligan_ids = set()
        if names:
            self.mulligan_ids = {names.index(n) for n in MULLIGAN_CATEGORIES if n in names}

    def of_rows(self, tok_cat, act_tok):
        """[B, A] the category value of each menu row (-1 none / unknown)."""
        if self.slot is None:
            return torch.full(act_tok.shape, -1, dtype=torch.long, device=act_tok.device)
        ids = tok_cat[:, :, self.slot].gather(1, act_tok)
        v = ids - self.offset - 1
        return torch.where((v >= 0) & (v < self.size - 1), v, torch.full_like(v, -1))

    def name(self, v):
        if v < 0:
            return "none"
        return self.names[v] if self.names and v < len(self.names) else "cat%d" % v


def event_dropout(inputs, p, spec):
    """The ten inputs with ``tok_mask`` cleared, independently with probability `p`, at each history
    event token and at the card tokens related to it (``hist_card.event``); `inputs` unchanged
    without history tokens or at p 0."""
    if p <= 0 or spec.hist_event_type is None:
        return inputs
    tok_type, tok_mask, rel, rel_mask = inputs[0], inputs[5], inputs[6], inputs[7]
    drop = (tok_type == spec.hist_event_type) & tok_mask & (torch.rand(tok_type.shape, device=tok_type.device) < p)
    if spec.hist_card_rel is not None:
        is_card = (rel[:, :, 2] == spec.hist_card_rel) & rel_mask
        ev_dropped = drop.gather(1, rel[:, :, 1]) & is_card
        hits = torch.zeros(tok_type.shape, dtype=torch.int32, device=tok_type.device)
        hits.scatter_add_(1, rel[:, :, 0], ev_dropped.to(torch.int32))
        drop = drop | (hits > 0)
    out = list(inputs)
    out[5] = tok_mask & ~drop
    return out


def policy_loss(logits, act_mask, label, smoothing):
    """Per decision: the label-smoothed cross-entropy over the real menu rows; and log π."""
    lp = M.masked_log_softmax(logits.float(), act_mask)
    lpz = lp.masked_fill(~act_mask, 0.0)
    nll = -lpz.gather(1, label.unsqueeze(1)).squeeze(1)
    uniform = -lpz.sum(-1) / act_mask.sum(-1).clamp(min=1).to(lpz.dtype)
    return (1.0 - smoothing) * nll + smoothing * uniform, lp


def weighted_policy_loss(pl, sideboarding, sideboard_weight):
    """The batch's policy loss from the per-decision `pl` [B]: its mean, or at a `sideboard_weight`
    other than 1 the mean weighted by it at the sideboarding decisions (`sideboarding` [B] bool)."""
    if sideboard_weight == 1.0:
        return pl.mean()
    w = torch.where(sideboarding, torch.as_tensor(float(sideboard_weight), dtype=pl.dtype, device=pl.device),
                    torch.ones_like(pl))
    return (w * pl).sum() / w.sum().clamp(min=1e-12)


class BCTrainer:
    """Behaviour cloning of `model` (an RMNet) on `window` (a replay.ReplayWindow)."""

    def __init__(self, model, window, cfg=None, device="cpu", categories=None, log=None):
        self.cfg = cfg or BCConfig()
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.window = window
        self.categories = categories or Categories(model.spec)
        self.log = log or (lambda m: None)
        c = self.cfg
        params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=c.lr, weight_decay=c.weight_decay) if c.weight_decay \
            else torch.optim.Adam(params, lr=c.lr)
        on_gpu = self.device.type == "cuda"
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(c.amp) if on_gpu else None
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)
        self.step = 0
        self.decisions = 0
        self.n_cards = model.cards.n_cards
        self.depth = model.spec.library_depth
        self._pf = None
        self._pending = []

    # ── batches ──

    def _make_batch(self, rng):
        """`sort_pool` batches: that many batches' decisions drawn uniformly, sorted by token count
        and cut into batches (in a random order), so each batch pads to similar sizes."""
        b, k = self.cfg.batch_size, max(1, self.cfg.sort_pool)
        items = self.window.sample_decisions(b * k, rng, "train")
        items.sort(key=lambda it: it[0].seat_feat.tokens(it[1]))
        out = []
        for j in rng.permutation(k):
            db = RP.decision_batch(items[j * b:(j + 1) * b], self.n_cards, self.depth)
            out.append(db.to_torch(self.device, pin=self.device.type == "cuda"))
        return out

    def _next_batch(self, pf):
        if not self._pending:
            self._pending = list(pf.get())
        return self._pending.pop()

    def _prefetcher(self):
        if self._pf is None:
            depth = max(1, self.cfg.prefetch // max(1, self.cfg.sort_pool))
            self._pf = RP.Prefetcher(self._make_batch, depth=depth, workers=self.cfg.loader_workers,
                                     seed=self.cfg.seed * 7919 + self.step)
        return self._pf

    def close(self):
        if self._pf is not None:
            self._pf.stop()
            self._pf = None
        self._pending = []

    def lr_at(self, step):
        c = self.cfg
        if step < c.warmup:
            return c.lr * (step + 1) / c.warmup
        if c.decay_steps <= 0:
            return c.lr
        f = min(1.0, (step - c.warmup) / c.decay_steps)
        return c.lr_min + 0.5 * (c.lr - c.lr_min) * (1 + math.cos(math.pi * f))

    def _autocast(self):
        if self.amp_dtype is None:
            return torch.autocast(self.device.type, enabled=False)
        return torch.autocast(self.device.type, dtype=self.amp_dtype)

    def losses(self, b, out):
        """{name: scalar tensor}: total, policy, value and each aux head's; plus the batch's agreement
        and the sideboarding decisions' sums (``sum_policy_sideboarding``, ``sum_agree_sideboarding``,
        ``n_sideboarding``, ``n_decisions``)."""
        c = self.cfg
        mask = b.inputs[9]
        pl, lp = policy_loss(out["logits"], mask, b.label, c.label_smoothing)
        sb = b.sideboarding
        loss = {"policy": weighted_policy_loss(pl, sb, c.sideboard_weight),
                "value": ((out["value"].float() - b.outcome) ** 2).mean()}
        total = loss["policy"] + c.value_weight * loss["value"]
        if b.targets is not None and "aux_hand" in out:
            al = M.aux_losses(out, b.targets)
            for k, v in al.items():
                loss["aux_" + k] = v
                total = total + c.aux_head_weights.get(k, c.aux_weight) * v
        loss["total"] = total
        hit = (lp.argmax(-1) == b.label).float()
        loss["agree"] = hit.mean()
        loss["sum_policy_sideboarding"] = (pl.detach() * sb).sum()
        loss["sum_agree_sideboarding"] = (hit * sb).sum()
        loss["n_sideboarding"] = sb.sum().float()
        loss["n_decisions"] = torch.tensor(float(b.n), device=pl.device)
        return loss

    # ── training ──

    def train(self, steps):
        """`steps` optimizer steps; the means of the losses and rates over them (a dict)."""
        self.model.train()
        pf = self._prefetcher()
        sums = {}
        gn_sum = 0.0
        gn_n = 0.0
        t0 = time.time()
        w0 = pf.wait_s
        n0 = self.decisions
        for _ in range(steps):
            b = self._next_batch(pf)
            lr = self.lr_at(self.step)
            for g in self.opt.param_groups:
                g["lr"] = lr
            inputs = event_dropout(b.inputs, self.cfg.event_dropout, self.model.spec)
            self.model.cards.mark_seen(inputs[3])
            with self._autocast():
                out = self.model(*inputs)
            loss = self.losses(b, out)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss["total"]).backward()
            self.scaler.unscale_(self.opt)
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            for k, v in loss.items():
                sums[k] = sums.get(k, 0.0) + v.detach()
            fin = torch.isfinite(gn).float()      # a step the scaler skips has an infinite norm
            gn_sum += torch.nan_to_num(gn.detach().float() * fin, nan=0.0, posinf=0.0)
            gn_n += fin
            self.step += 1
            self.decisions += b.n
        dt = time.time() - t0
        sb = {k: float(sums.pop(k)) for k in ("sum_policy_sideboarding", "sum_agree_sideboarding", "n_sideboarding",
                                              "n_decisions")}
        m = {k: float(v) / steps for k, v in sums.items()}
        m = {("train_" + k): v for k, v in m.items()}
        if sb["n_sideboarding"] > 0:
            m["train_policy_sideboarding"] = sb["sum_policy_sideboarding"] / sb["n_sideboarding"]
            m["train_agree_sideboarding"] = sb["sum_agree_sideboarding"] / sb["n_sideboarding"]
            m["train_frac_sideboarding"] = sb["n_sideboarding"] / max(sb["n_decisions"], 1.0)
        m.update({"step": self.step, "lr": self.lr_at(self.step - 1), "grad_norm": float(gn_sum) / max(float(gn_n), 1.0),
                  "skipped_steps": steps - int(float(gn_n)),
                  "decisions": self.decisions, "decisions_per_s": (self.decisions - n0) / dt,
                  "loader_wait_frac": (pf.wait_s - w0) / dt, "seconds": dt, "amp_scale": float(self.scaler.get_scale())
                  if self.scaler.is_enabled() else 1.0})
        return m

    # ── evaluation ──

    @torch.no_grad()
    def evaluate(self, split="heldout", limit=None):
        """Agreement, value and aux metrics over the split's labelled decisions (a dict)."""
        c = self.cfg
        limit = c.eval_limit if limit is None else limit
        self.model.eval()
        cat = self.categories
        agree_n = {}
        tot_n = {}
        v_se = 0.0
        v_n = 0
        v_sign = v_dec = 0
        aux_hit = {k: 0.0 for k in AUX_HEADS}
        aux_n = {k: 0.0 for k in AUX_HEADS}
        loss_sum = 0.0
        group_loss = {}
        n = 0

        def add(key, hits, count):
            agree_n[key] = agree_n.get(key, 0) + hits
            tot_n[key] = tot_n.get(key, 0) + count

        for items in self.window.iter_decisions(split, True, c.eval_batch, limit):
            b = RP.decision_batch(items, self.n_cards, self.depth).to_torch(self.device)
            with self._autocast():
                out = self.model(*b.inputs)
            mask = b.inputs[9]
            pl, lp = policy_loss(out["logits"], mask, b.label, c.label_smoothing)
            loss_sum += float(pl.sum())
            hit = (lp.argmax(-1) == b.label)
            rows = cat.of_rows(b.inputs[2], b.inputs[8])
            lab_cat = rows.gather(1, b.label.unsqueeze(1)).squeeze(1)
            mull = torch.zeros_like(hit)
            for mid in cat.mulligan_ids:
                mull |= ((rows == mid) & mask).any(-1)
            group = torch.where(b.sideboarding, 2, torch.where(mull, 1, 0))
            add("all", int(hit.sum()), b.n)
            for gi, gname in enumerate(("in_game", "mulligan", "sideboarding")):
                sel = group == gi
                add("group:" + gname, int((hit & sel).sum()), int(sel.sum()))
                group_loss[gname] = group_loss.get(gname, 0.0) + float((pl * sel).sum())
            for v in torch.unique(lab_cat).tolist():
                sel = lab_cat == v
                add("cat:" + cat.name(int(v)), int((hit & sel).sum()), int(sel.sum()))
            val = out["value"].float()
            v_se += float(((val - b.outcome) ** 2).sum())
            v_n += b.n
            dec = b.outcome != 0
            v_sign += int(((val > 0) == (b.outcome > 0))[dec].sum())
            v_dec += int(dec.sum())
            if b.targets is not None and "aux_hand" in out:
                t = b.targets
                hand = t["opp_hand"]
                ok = hand.sum(-1) > 0
                top = out["aux_hand"].argmax(-1)
                aux_hit["hand"] += float((hand.gather(1, top.unsqueeze(1)).squeeze(1) > 0)[ok].sum())
                aux_n["hand"] += float(ok.sum())
                for k in ("own_library", "opp_library"):
                    tg = t[k]
                    ok = tg >= 0
                    aux_hit[k] += float((out["aux_" + k].argmax(-1) == tg)[ok].sum())
                    aux_n[k] += float(ok.sum())
            n += b.n
        self.model.train()
        if not n:
            return {"eval_decisions": 0}
        m = {"eval_decisions": n, "eval_policy_loss": loss_sum / n, "eval_value_mse": v_se / max(v_n, 1),
             "eval_value_sign_acc": v_sign / max(v_dec, 1), "eval_step": self.step}
        for k in sorted(tot_n):
            if tot_n[k]:
                m["eval_agree_" + k] = agree_n[k] / tot_n[k]
                m["eval_count_" + k] = tot_n[k]
                if k.startswith("group:"):
                    m["eval_policy_loss_" + k] = group_loss[k[len("group:"):]] / tot_n[k]
        for k in AUX_HEADS:
            if aux_n[k]:
                m["eval_aux_acc_" + k] = aux_hit[k] / aux_n[k]
        return m

    # ── checkpoints and exports ──

    def save(self, path, version=0, extra=None):
        """The model checkpoint at `path` (model.save_checkpoint) and the optimizer state at `path`.opt."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        info = {"step": self.step, "decisions": self.decisions, "version": int(version), "bc_config": self.cfg.to_dict()}
        info.update(extra or {})
        M.save_checkpoint(self.model, path, info)
        tmp = path + ".opt.tmp"
        torch.save({"opt": self.opt.state_dict(), "scaler": self.scaler.state_dict(), "step": self.step,
                    "decisions": self.decisions}, tmp)
        os.replace(tmp, path + ".opt")
        return path

    def restore(self, path):
        """Loads the optimizer state saved beside checkpoint `path` (the model is loaded separately)."""
        st = torch.load(path + ".opt", map_location=self.device, weights_only=False)
        self.opt.load_state_dict(st["opt"])
        self.scaler.load_state_dict(st["scaler"])
        self.step, self.decisions = st["step"], st["decisions"]

    def export(self, prefix, version, formats=("ts", "pt2")):
        was = self.model.training
        paths = EX.export_policy(self.model, prefix, version, formats)
        self.model.train(was)
        return paths


# ── the command line ─────────────────────────────────────────────────────────────────────────

def main():
    from . import spec as S
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", required=True, help="rmfeat --export-tables DIR")
    ap.add_argument("--episodes", nargs="+", required=True, help="trajectory directories")
    ap.add_argument("--out", required=True, help="run directory: metrics.jsonl, ckpt/")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--minutes", type=float, default=25.0, help="stop after this long")
    ap.add_argument("--steps", type=int, default=0, help="stop after this many steps (0: by time)")
    ap.add_argument("--chunk", type=int, default=250, help="steps between metrics lines")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--heldout", type=float, default=0.05)
    ap.add_argument("--heldout-by", choices=("game", "match"), default="game",
                    help="hold out games, or whole matches")
    ap.add_argument("--split-seed", type=int, default=None, help="the held-out split's seed (default: the BC seed)")
    ap.add_argument("--eval-train", type=float, default=0.0,
                    help="also evaluate this fraction of the training games (0: none)")
    ap.add_argument("--max-decisions", type=int, default=4_000_000)
    ap.add_argument("--version", type=int, default=1, help="the exported policy's version")
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--config", help="a JSON object of BCConfig fields")
    a = ap.parse_args()
    cfg = BCConfig.from_dict(json.loads(a.config)) if a.config else BCConfig()
    torch.manual_seed(cfg.seed)
    os.makedirs(os.path.join(a.out, "ckpt"), exist_ok=True)
    metrics = open(os.path.join(a.out, "metrics.jsonl"), "a")

    def emit(m):
        m = dict(m, time=time.time())
        metrics.write(json.dumps(m) + "\n")
        metrics.flush()
        print(json.dumps(m), flush=True)

    tables = S.load_tables(a.tables)
    win = RP.ReplayWindow(RP.WindowConfig(dirs=a.episodes, max_decisions=a.max_decisions,
                                          spec_hash=tables.accepted_spec_hashes(),
                                          heldout_fraction=a.heldout, heldout_by=a.heldout_by,
                                          probe_fraction=a.eval_train,
                                          split_seed=cfg.seed if a.split_seed is None else a.split_seed))
    t = time.time()
    win.poll()
    emit(dict(win.stats(), event="window", seconds=time.time() - t))
    model = M.RMNet(M.ModelConfig(d=a.d, layers=a.layers, heads=a.heads), tables)
    cats = Categories(tables.spec)
    trainer = BCTrainer(model, win, cfg, a.device, cats)
    emit({"event": "start", "params": model.param_count(), "bc_config": cfg.to_dict(),
          "model_config": model.cfg.to_dict(), "device": a.device})
    end = time.time() + 60 * a.minutes
    evaluated = -1

    def evaluate():
        emit(trainer.evaluate())
        if a.eval_train:
            m = trainer.evaluate("probe")
            emit({"evaltrain_" + k[len("eval_"):]: v for k, v in m.items()})

    try:
        while (not a.steps or trainer.step < a.steps) and time.time() < end:
            n = a.chunk if not a.steps else min(a.chunk, a.steps - trainer.step)
            emit(trainer.train(n))
            if trainer.step % a.eval_every < a.chunk:
                evaluate()
                trainer.save(os.path.join(a.out, "ckpt", "bc.pt"), a.version)
                evaluated = trainer.step
        if evaluated != trainer.step:
            evaluate()
            trainer.save(os.path.join(a.out, "ckpt", "bc.pt"), a.version)
        emit({"event": "export", "paths": trainer.export(os.path.join(a.out, "ckpt", "p%d" % a.version), a.version)})
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
