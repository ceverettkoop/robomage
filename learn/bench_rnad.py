#!/usr/bin/env python3
"""The R-NaD learner step's throughput (docs/rnad_training_plan.md §4 L4, §5): decisions per second
and peak memory of ``RNaDLearner.step`` for the given model, batch shapes and precisions.

    HSA_OVERRIDE_GFX_VERSION=10.3.0 flock /tmp/rm-gpu.lock learn/.venv/bin/python learn/bench_rnad.py \\
        --tables DIR --episodes DIR --device cuda --games 32 --chunk 8,16 --precision fp32,fp16

G games per batch are drawn round-robin from the episodes in ``--episodes`` (any *.rmtraj with μ; the
random agent's will do). Each configuration runs ``--warmup`` steps, then ``--steps`` timed ones on
the same batch (``replay.TrajectoryBatch``; ``prep`` is building it, the chunks' inputs are
expanded inside the step). ``--recorded-regs`` marks every step's regularisation log π as recorded at the
learner's versions (the server-supplied path: no recompute forwards); without it the learner
recomputes them (one extra forward while π_0,reg sits in both slots). On the GPU this must be the
only process on it (``flock /tmp/rm-gpu.lock``).
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rmlearn import model as M  # noqa: E402
from rmlearn import episodes as EP  # noqa: E402
from rmlearn import replay as RP  # noqa: E402
from rmlearn import rnad_learner as RL  # noqa: E402
from rmlearn import spec as S  # noqa: E402


def make_batch(eps, games, chunk, recorded, omniscient, learner):
    pick = [eps[i % len(eps)] for i in range(games)]
    b = RP.TrajectoryBatch(pick, chunk_len=chunk, policies=(1, 2), omniscient=omniscient)
    if recorded:
        for k, pid in enumerate(learner.cfg.reg_policy_ids):
            ev = b.evals[pid]
            ev["present"][:] = b.valid
            ev["version"][:] = learner.regs[k].version
            ev["log_pi"][:] = np.where(b.legal, -np.log(np.maximum(b.legal.sum(-1, keepdims=True), 1)), 0.0)
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--games", default="32", help="games per batch (comma list)")
    ap.add_argument("--chunk", default="16", help="chunk lengths (comma list)")
    ap.add_argument("--precision", default="fp32,fp16")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--recorded-regs", action="store_true")
    ap.add_argument("--omniscient", action="store_true", help="the omniscient critic (needs omniscient episodes)")
    ap.add_argument("--forward-tokens", type=int, default=65536)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()
    dev = torch.device(args.device)
    tables = S.load_tables(args.tables)
    eps = [EP.read_episode(f) for f in sorted(glob.glob(os.path.join(args.episodes, "*.rmtraj")))]
    eps, dropped = RL.trainable_episodes(eps, spec_hash=tables.hashes["spec_hash"])
    lens = np.array([e.steps for e in eps])
    toks = np.concatenate([np.diff(e.seat_feat.tok_off) for e in eps])
    print("episodes: %d (%s dropped), decisions per game mean %.0f max %d, tokens per decision mean %.0f max %d"
          % (len(eps), dropped or "none", lens.mean(), lens.max(), toks.mean(), toks.max()), flush=True)
    for prec in args.precision.split(","):
        for games in (int(x) for x in args.games.split(",")):
            for chunk in (int(x) for x in args.chunk.split(",")):
                torch.manual_seed(0)
                net = M.RMNet(M.ModelConfig(d=args.d, layers=args.layers, omniscient_critic=args.omniscient), tables)
                cfg = RL.RNaDConfig(chunk_len=chunk, precision=prec, forward_tokens=args.forward_tokens)
                learner = RL.RNaDLearner(net, cfg, device=dev)
                t = time.time()
                batch = make_batch(eps, games, chunk, args.recorded_regs, args.omniscient, learner)
                prep = time.time() - t
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                    torch.cuda.reset_peak_memory_stats(dev)
                try:
                    for _ in range(args.warmup):
                        learner.step(batch)
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    t = time.time()
                    for _ in range(args.steps):
                        m = learner.step(batch)
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    dt = (time.time() - t) / args.steps
                except torch.OutOfMemoryError:
                    print("%s G %d chunk %d: out of memory" % (prec, games, chunk), flush=True)
                    del learner, net
                    torch.cuda.empty_cache()
                    continue
                mem = ""
                if dev.type == "cuda":
                    mem = ", peak allocated %.2f GB, reserved %.2f GB" % (
                        torch.cuda.max_memory_allocated(dev) / 2 ** 30, torch.cuda.max_memory_reserved(dev) / 2 ** 30)
                print("%s G %d chunk %d (T %d, %d decisions, ≤ %d per chunk): %.3f s/step, %.0f decisions/s, prep "
                      "%.2f s, loss %.4f, %d log π_reg recomputed%s"
                      % (prec, games, chunk, batch.T, m["decisions"], int(batch.valid.reshape(-1, chunk, games)
                                                                          .sum((1, 2)).max()),
                         dt, m["decisions"] / dt, prep, m["loss"], m["reg_recomputed"], mem), flush=True)
                del learner, net, batch
                if dev.type == "cuda":
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
