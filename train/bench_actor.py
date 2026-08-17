#!/usr/bin/env python3
"""AlphaZero self-play throughput benchmark: C++ actor vs Python reference.

Times the two self-play producers on the SAME work (same exported net, same deck,
same sims/worlds, both single-process single-thread) and reports games/hour and
per-decision cost so the C++ actor's speedup is a headline number.

  Leg A — bin/az_actor --selfplay: run N games as a subprocess, wall-timed;
          "decisions" = the searched-sample count from its
          `SELFPLAY: total_samples=` line.
  Leg B — az_selfplay in-process, single worker: the same N games driven through
          az_selfplay's own _play_match loop (no multiprocessing), wall-timed;
          "decisions" = the searched-sample count.

Both legs load the identical deterministic AZNet (torch.manual_seed(0), exported
to .ts.pt for C++ and loaded via load_az for Python). Single-thread on both sides
(torch.set_num_threads(1)) so the comparison measures the engine/search path, not
BLAS parallelism.

Run:
  train/.venv/bin/python train/bench_actor.py --games 2 --sims 32 --worlds 2
  train/.venv/bin/python train/bench_actor.py --games 2 --sims 128 --worlds 4
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from az_net import (AZNet, obs_space_from_const, load_az, AZEvaluator,
                    save_torchscript, torchscript_export_path)
from cli_spec import BIN_DIR, BUILD_DIR
import az_selfplay

# BUILD_DIR (bin/<config>/) is where the actor binary lives; BIN_DIR (bin/) stays
# the launch cwd used below for resource lookup.
ACTOR_BIN = os.path.join(BUILD_DIR, "az_actor")
_TOTAL = re.compile(r"^SELFPLAY: total_samples=(\d+) shards=(\d+)$")


def _cpp_leg(ts_path, out_dir, args):
    # Shared argv builder (az_selfplay.actor_selfplay_cmd) pins the same
    # noise/temperature knobs _python_leg passes to _play_match, so the two legs
    # measure the identical workload by construction.
    cmd = az_selfplay.actor_selfplay_cmd(
        ACTOR_BIN, deck=args.deck, deck_b=getattr(args, "deck_b", None),
        seed=args.seed, games=args.games,
        sims=args.sims, worlds=args.worlds, model=ts_path, out_dir=out_dir)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=BIN_DIR, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=env)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        print("FAIL: az_actor exited nonzero:\n"
              + proc.stderr.decode("utf-8", "replace"), file=sys.stderr)
        return None
    decisions = 0
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        m = _TOTAL.match(line.strip())
        if m:
            decisions = int(m.group(1))
    return dt, decisions


def _python_leg(ckpt, out_dir, args):
    torch.set_num_threads(1)
    net = load_az(ckpt)
    evaluator = AZEvaluator(net)
    rng = np.random.default_rng(args.seed + 100003)
    from search_env import SearchRoboMageEnv
    deck_b = getattr(args, "deck_b", None) or args.deck
    env = SearchRoboMageEnv(deck_a=args.deck, deck_b=deck_b)
    decisions = 0
    t0 = time.perf_counter()
    try:
        for g in range(args.games):
            seed = args.seed + g
            (samples, _game_winners, _searched, _fallback,
             _dropped, _sb_stats) = az_selfplay._play_match(
                env, evaluator, rng, sims=args.sims, worlds=args.worlds,
                temp_moves=az_selfplay.DEFAULT_TEMP_MOVES,
                root_noise_eps=az_selfplay.DEFAULT_ROOT_NOISE_EPS,
                root_noise_alpha=az_selfplay.DEFAULT_ROOT_NOISE_ALPHA, seed=seed)
            decisions += len(samples)
    finally:
        env.close()
    dt = time.perf_counter() - t0
    return dt, decisions


def _row(name, games, dt, decisions):
    spg = dt / max(1, games)
    gph = 3600.0 * games / dt if dt > 0 else float("nan")
    mspd = 1000.0 * dt / max(1, decisions)
    return {"name": name, "s_game": spg, "games_hr": gph,
            "decisions": decisions, "ms_dec": mspd}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--deck", default="league/ur_delver")
    ap.add_argument("--deck-b", default=None,
                    help="Player B deck (default: mirror = --deck)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if not os.path.exists(ACTOR_BIN):
        print(f"FAIL: {ACTOR_BIN} not found — build it with `make actor`",
              file=sys.stderr)
        return 1

    print(f"[bench] deck={args.deck} games={args.games} sims={args.sims} "
          f"worlds={args.worlds} seed={args.seed} (single-thread both legs)")

    with tempfile.TemporaryDirectory() as td:
        # One deterministic net, shared by both legs.
        torch.manual_seed(0)
        net = AZNet(obs_space_from_const()).eval()
        ckpt = os.path.join(td, "bench__azfinal.pt")
        net.save(ckpt)
        ts_path = torchscript_export_path(ckpt)
        save_torchscript(net, ts_path)

        print("[bench] leg A: C++ bin/az_actor --selfplay ...", flush=True)
        a = _cpp_leg(ts_path, os.path.join(td, "cpp"), args)
        if a is None:
            return 1
        print("[bench] leg B: Python az_selfplay (in-process, 1 worker) ...",
              flush=True)
        b = _python_leg(ckpt, os.path.join(td, "py"), args)

    ra = _row("C++  (az_actor)", args.games, a[0], a[1])
    rb = _row("Python (az_selfplay)", args.games, b[0], b[1])

    print("\n" + "=" * 66)
    print(f"{'leg':<22}{'s/game':>10}{'games/hr':>12}{'decisions':>11}{'ms/dec':>10}")
    print("-" * 66)
    for r in (ra, rb):
        print(f"{r['name']:<22}{r['s_game']:>10.3f}{r['games_hr']:>12.1f}"
              f"{r['decisions']:>11d}{r['ms_dec']:>10.2f}")
    print("=" * 66)
    speedup = rb["s_game"] / ra["s_game"] if ra["s_game"] > 0 else float("nan")
    dec_speedup = rb["ms_dec"] / ra["ms_dec"] if ra["ms_dec"] > 0 else float("nan")
    print(f"speedup (C++ vs Python): {speedup:.2f}x per game, "
          f"{dec_speedup:.2f}x per decision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
