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

  Leg C — (the eval server, AUTO like the az-* commands) the C++ actor fleet
          evaluating over one central az_eval_server's socket.

Run (``train.py bench-actor`` is the entry point; flags in cli_spec):
  train/.venv/bin/python train/train.py bench-actor --games 2 --sims 32 --worlds 2
  train/.venv/bin/python train/train.py bench-actor --games 2 --batch 1,4 --no-python
"""

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
from cli_spec import (BENCH_PLAYER_SELF, BIN_DIR, INTERACTIVE_BINARY,
                      INTERACTIVE_BUILD_DIR, parse_int_list)
import az_selfplay

# Both legs run the release-by-default tier the self-play drivers use
# (ROBOMAGE_BUILD overrides); BIN_DIR (bin/) stays the launch cwd used below
# for resource lookup.
ACTOR_BIN = os.path.join(INTERACTIVE_BUILD_DIR, "az_actor")
_TOTAL = re.compile(r"^SELFPLAY: total_samples=(\d+) shards=(\d+)$")


def _cpp_leg(ts_path, out_dir, args, batch=1, cross_world=False,
             device="cpu", eval_server=None, fleet=1, scripted=False):
    # Shared argv builder (az_selfplay.actor_selfplay_cmd) pins the same
    # noise/temperature knobs _python_leg passes to _play_match, so the two legs
    # measure the identical workload by construction. ``fleet`` > 1 launches
    # that many CONCURRENT actor processes, each playing args.games games on a
    # disjoint seed range — the Stage C shape (decisions/dt then measures
    # fleet-wide throughput, not single-process latency). ``scripted`` puts
    # scripted:hard on seat B via the oracle (net seat A, one shared oracle
    # process), matching _python_leg's agent/net_is_a=True mode.
    oracle = None
    if scripted:
        oracle = az_selfplay._spawn_oracle()  # (proc, sock, tmpdir)

    def _cmd(i):
        return az_selfplay.actor_selfplay_cmd(
            ACTOR_BIN, deck=args.deck_a, deck_b=getattr(args, "deck_b", None),
            seed=args.seed + i * 100000, games=args.games,
            sims=args.sims, worlds=args.worlds, model=ts_path, out_dir=out_dir,
            batch=batch, cross_world=cross_world,
            device=device, eval_server=eval_server,
            scripted_seat=("B" if scripted else None),
            scripted_oracle=(oracle[1] if scripted else None))
    # Single-thread BLAS; a local-GPU leg also needs the ROCm overrides.
    base = (az_selfplay.actor_gpu_env() if (device != "cpu" and not eval_server)
            else os.environ)
    env = dict(base, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.perf_counter()
    try:
        procs = [subprocess.Popen(_cmd(i), cwd=BIN_DIR, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, env=env)
                 for i in range(fleet)]
        outs = [p.communicate() for p in procs]
    finally:
        if oracle is not None:
            import shutil
            oracle[0].terminate()
            oracle[0].wait()
            shutil.rmtree(oracle[2], ignore_errors=True)
    dt = time.perf_counter() - t0
    decisions = 0
    for p, (out, err) in zip(procs, outs):
        if p.returncode != 0:
            print("FAIL: az_actor exited nonzero:\n"
                  + err.decode("utf-8", "replace"), file=sys.stderr)
            return None
        for line in out.decode("utf-8", "replace").splitlines():
            m = _TOTAL.match(line.strip())
            if m:
                decisions += int(m.group(1))
    return dt, decisions


def _python_leg(ckpt, out_dir, args, scripted=False):
    torch.set_num_threads(1)
    net = load_az(ckpt)
    evaluator = AZEvaluator(net)
    rng = np.random.default_rng(args.seed + 100003)
    from search_env import SearchRoboMageEnv
    deck_b = getattr(args, "deck_b", None) or args.deck_a
    env = SearchRoboMageEnv(deck_a=args.deck_a, deck_b=deck_b,
                            binary_path=INTERACTIVE_BINARY)
    agent = None
    if scripted:
        from scripted_agent import make_agent
        agent = make_agent("scripted:hard")
        agent.set_deck_names(args.deck_a, deck_b)
    decisions = 0
    t0 = time.perf_counter()
    try:
        for g in range(args.games):
            seed = args.seed + g
            (samples, _game_winners, _searched, _fallback,
             _dropped, _sb_stats) = az_selfplay._play_match(
                env, evaluator, rng, sims=args.sims, worlds=args.worlds,
                root_noise_eps=az_selfplay.DEFAULT_ROOT_NOISE_EPS,
                root_noise_alpha=az_selfplay.DEFAULT_ROOT_NOISE_ALPHA, seed=seed,
                agent=agent, net_is_a=(True if scripted else None))
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


def run(args):
    """``train.py bench-actor`` entry; returns the process exit code."""
    batches = parse_int_list(args.batch, "--batch")
    scripted = args.player_b != BENCH_PLAYER_SELF
    if not os.path.exists(ACTOR_BIN):
        print(f"FAIL: {ACTOR_BIN} not found — build it with `make actor "
              "BUILD=RELEASE` (or set ROBOMAGE_BUILD=debug)",
              file=sys.stderr)
        return 1

    print(f"[bench] deck={args.deck_a} games={args.games} sims={args.sims} "
          f"worlds={args.worlds} seed={args.seed} (single-thread both legs)")

    with tempfile.TemporaryDirectory() as td:
        # One deterministic net, shared by both legs.
        torch.manual_seed(0)
        net = AZNet(obs_space_from_const()).eval()
        ckpt = os.path.join(td, "bench__azfinal.pt")
        net.save(ckpt)
        ts_path = torchscript_export_path(ckpt)
        save_torchscript(net, ts_path)

        # Fleet legs play fleet * games games total; _row needs the real count.
        leg_games = args.games * args.fleet
        dev_lbl = "" if args.actor_device == "cpu" else f" {args.actor_device}"
        fleet_lbl = "" if args.fleet == 1 else f" n={args.fleet}"
        scr_lbl = ", scripted B" if scripted else ""
        rows = []
        for k in batches:
            print(f"[bench] leg A: C++ bin/az_actor --selfplay (batch={k}"
                  f"{dev_lbl}{fleet_lbl}{scr_lbl}) ...", flush=True)
            a = _cpp_leg(ts_path, os.path.join(td, f"cpp_b{k}"), args, batch=k,
                         device=args.actor_device, fleet=args.fleet,
                         scripted=scripted)
            if a is None:
                return 1
            rows.append(_row(f"C++ b={k}{dev_lbl}{fleet_lbl}", leg_games,
                             a[0], a[1]))
        if not args.no_cross_world:
            print(f"[bench] leg A: C++ bin/az_actor --selfplay (cross-world"
                  f"{dev_lbl}{fleet_lbl}{scr_lbl}) ...", flush=True)
            a = _cpp_leg(ts_path, os.path.join(td, "cpp_xw"), args,
                         cross_world=True, device=args.actor_device,
                         fleet=args.fleet, scripted=scripted)
            if a is None:
                return 1
            rows.append(_row(f"C++ b=xw{dev_lbl}{fleet_lbl}", leg_games,
                             a[0], a[1]))
        eval_server = az_selfplay.resolve_eval_server(args)
        if eval_server is not False:
            # Stage C: one server owns the device; the whole fleet shares it.
            # Cross-world keeps each actor's request K = worlds with no quality
            # cost, so it is the natural pairing.
            srv_dev = _server_device(args)
            print(f"[bench] leg C: az_eval_server({srv_dev}) + "
                  f"{args.fleet} actor(s), cross-world ...", flush=True)
            try:
                server, sock, server_dir = az_selfplay.start_eval_server(
                    ts_path, device=srv_dev, forced=bool(eval_server),
                    tag="bench")
            except RuntimeError as exc:
                print(f"FAIL: {exc} (no usable GPU?)", file=sys.stderr)
                return 1
            if server is None:
                print("[bench] eval-server AUTO: the server failed to start "
                      "(no usable GPU?) — skipping leg C", flush=True)
            else:
                try:
                    a = _cpp_leg(ts_path, os.path.join(td, "cpp_srv"), args,
                                 cross_world=True, eval_server=sock,
                                 fleet=args.fleet, scripted=scripted)
                finally:
                    az_selfplay.stop_eval_server(server, server_dir)
                if a is None:
                    return 1
                rows.append(_row(f"C++ xw srv-{srv_dev}{fleet_lbl}",
                                 leg_games, a[0], a[1]))
        rb = None
        if not args.no_python:
            print(f"[bench] leg B: Python az_selfplay (in-process, 1 worker"
                  f"{scr_lbl}) ...", flush=True)
            b = _python_leg(ckpt, os.path.join(td, "py"), args,
                            scripted=scripted)
            rb = _row("Python (az_selfplay)", args.games, b[0], b[1])

    ra = rows[0]
    print("\n" + "=" * 78)
    print(f"{'leg':<22}{'s/game':>10}{'games/hr':>12}{'decisions':>11}{'ms/dec':>10}"
          f"{'evals/s':>12}")
    print("-" * 78)
    for r in rows + ([rb] if rb is not None else []):
        # ~sims leaf evals per searched root: the fleet-wide net throughput,
        # directly comparable to the sanity sweep's rows/s numbers.
        evals_s = args.sims * 1000.0 / r["ms_dec"] if r["ms_dec"] > 0 else 0.0
        print(f"{r['name']:<22}{r['s_game']:>10.3f}{r['games_hr']:>12.1f}"
              f"{r['decisions']:>11d}{r['ms_dec']:>10.2f}{evals_s:>12.0f}")
    print("=" * 78)
    # Batch sweep: per-decision speedup of each batch>1 leg over the first
    # (baseline) batch value. Decision counts differ across batch values (the
    # trees diverge), so ms/dec — same sims per searched root — is the metric.
    for r in rows[1:]:
        s = ra["ms_dec"] / r["ms_dec"] if r["ms_dec"] > 0 else float("nan")
        print(f"batch speedup ({r['name'].split('b=')[-1]:>2} vs "
              f"{ra['name'].split('b=')[-1]}): {s:.2f}x per decision")
    if rb is not None:
        speedup = rb["s_game"] / ra["s_game"] if ra["s_game"] > 0 else float("nan")
        dec_speedup = rb["ms_dec"] / ra["ms_dec"] if ra["ms_dec"] > 0 else float("nan")
        print(f"speedup (C++ vs Python): {speedup:.2f}x per game, "
              f"{dec_speedup:.2f}x per decision")
    return 0


def _server_device(args):
    """The eval-server leg's device: --eval-server-device, else the az-*
    commands' rule (--actor-device, with cpu meaning a cuda server)."""
    if args.eval_server_device:
        return args.eval_server_device
    return args.actor_device if args.actor_device != "cpu" else "cuda"


if __name__ == "__main__":
    sys.exit("bench_actor.py was removed; use `train.py bench-actor` "
             "(e.g. --games 2 --sims 32 --worlds 2)")
