#!/usr/bin/env python3
"""Phase B gate evaluation: MCTS-guided policy vs the SAME checkpoint raw.

Runs the A/B strength gate from the AlphaZero plan (docs/alphazero_status.md):
``mcts:<ckpt>?sims=&worlds=`` vs the raw checkpoint, mirror decks, seats
alternating (half the games each way, split into per-worker batches with
disjoint seed ranges), plus optional scripted:hard reference batches. Batches
run across parallel worker processes (each pinned to 1 torch thread); results
print per batch with a running total, then a final summary. Success bar per
the plan: >= 55% for the search side over ~200 games.

Example (the real-hardware gate):
    train/.venv/bin/python train/eval_search_gate.py \
        --checkpoint league/ur_delver --deck league/ur_delver \
        --games 192 --sims 128 --worlds 4 --workers 4

The checkpoint spec is anything ``opponents.resolve_checkpoint`` accepts (deck
shorthand or path). ``--ref-games N`` adds N games of mcts-vs-scripted:hard and
2N of raw-vs-scripted:hard as an absolute reference (0 disables).
"""

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TRAIN_DIR)


def _run_batch(args):
    tag, a_spec, b_spec, deck, games, seed = args
    sys.path.insert(0, _TRAIN_DIR)
    os.chdir(_REPO_ROOT)
    import torch
    torch.set_num_threads(1)
    from runner import run_match
    from opponents import make_controller
    ca = make_controller(a_spec)
    cb = make_controller(b_spec)
    t0 = time.time()
    r = run_match(ca, cb, deck_a=deck, deck_b=deck,
                  games=games, bo3=True, seed=seed, transcript="quiet")
    stats = getattr(ca, "stats", None) or getattr(cb, "stats", None) or {}
    return tag, r.wins, r.losses, r.draws, dict(stats), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True,
                    help="Checkpoint spec (deck shorthand or path) piloting BOTH sides")
    ap.add_argument("--deck", required=True, help="Deck for the mirror match")
    ap.add_argument("--games", type=int, default=192,
                    help="Total gate games, split evenly across seats (default 192)")
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--sb-sims", type=int, default=None,
                    help="bo3 sideboard-root sims (default: the SearchController "
                         "sideboard budget, DEFAULT_SB_SIMS)")
    ap.add_argument("--sb-worlds", type=int, default=None,
                    help="bo3 sideboard-root determinized worlds (default budget)")
    ap.add_argument("--sb-max-depth", type=int, default=None,
                    help="bo3 sideboard-root descent depth cap (default budget)")
    ap.add_argument("--sb-rollout-turns", type=int, default=None,
                    help="bo3 sideboard-root leaf-rollout horizon in player "
                         "turns, 0 = off (default budget)")
    ap.add_argument("--c", type=float, default=1.5, dest="c_puct")
    ap.add_argument("--vscale", type=float, default=1.0,
                    help="Value squash scale for the PPO value head")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--batch-size", type=int, default=24,
                    help="Games per batch (a batch is one worker unit of work)")
    ap.add_argument("--ref-games", type=int, default=16,
                    help="MCTS-vs-scripted:hard reference games (0 disables; raw "
                         "reference runs 2x this)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    spec = (f"mcts:{args.checkpoint}?sims={args.sims}&worlds={args.worlds}"
            f"&c={args.c_puct}&vscale={args.vscale}")
    for knob, val in (("sb_sims", args.sb_sims), ("sb_worlds", args.sb_worlds),
                      ("sb_max_depth", args.sb_max_depth),
                      ("sb_rollout_turns", args.sb_rollout_turns)):
        if val is not None:
            spec += f"&{knob}={val}"
    raw = args.checkpoint

    batches = []
    n_batches = max(2, (args.games + args.batch_size - 1) // args.batch_size)
    n_batches += n_batches % 2  # even count so seats split exactly evenly
    per = args.games // n_batches
    for i in range(n_batches):
        seed = args.seed + 1000 * i
        if i % 2 == 0:
            batches.append((f"gate_mctsA_{i}", spec, raw, args.deck, per, seed))
        else:
            batches.append((f"gate_mctsB_{i}", raw, spec, args.deck, per, seed))
    if args.ref_games:
        batches.append(("ref_mcts_vs_hard", spec, "scripted", args.deck,
                        args.ref_games, args.seed + 900001))
        batches.append(("ref_raw_vs_hard", raw, "scripted", args.deck,
                        2 * args.ref_games, args.seed + 950001))

    print(f"[gate] {sum(b[4] for b in batches if b[0].startswith('gate_'))} gate games "
          f"({n_batches} batches), spec={spec}, workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    mcts_w = mcts_l = mcts_d = 0
    agg = {"searched": 0, "fallback": 0, "sims": 0, "sim_steps": 0}
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_batch, b): b[0] for b in batches}
        for fut in as_completed(futs):
            tag, w, l, d, stats, dt = fut.result()
            results[tag] = (w, l, d)
            print(f"[{tag}] {w}W {l}L {d}D  wall={dt / 60:.1f}min  stats={stats}",
                  flush=True)
            if tag.startswith("gate_"):
                if "mctsA" in tag:
                    mcts_w += w; mcts_l += l
                else:
                    mcts_w += l; mcts_l += w
                mcts_d += d
                for k in agg:
                    agg[k] += stats.get(k, 0)
                print(f"  == gate running total: MCTS {mcts_w}W {mcts_l}L {mcts_d}D"
                      f"  winrate={mcts_w / max(1, mcts_w + mcts_l):.1%}", flush=True)

    print("\n===== FINAL =====", flush=True)
    tot = mcts_w + mcts_l + mcts_d
    wr = mcts_w / max(1, mcts_w + mcts_l)
    print(f"GATE (mcts vs raw, same ckpt): {mcts_w}W {mcts_l}L {mcts_d}D of {tot}"
          f"  winrate={wr:.1%}  -> {'PASS' if wr >= 0.55 else 'FAIL'} (bar 55%)",
          flush=True)
    print(f"search stats: {agg}  safe-fraction="
          f"{agg['searched'] / max(1, agg['searched'] + agg['fallback']):.1%}",
          flush=True)
    for tag in ("ref_mcts_vs_hard", "ref_raw_vs_hard"):
        if tag in results:
            w, l, d = results[tag]
            print(f"{tag}: {w}W {l}L {d}D  winrate={w / max(1, w + l):.1%}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
