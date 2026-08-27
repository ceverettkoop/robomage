#!/usr/bin/env python
"""Benchmark AlphaZero self-play throughput across worker counts.

Runs one bo3 self-play generation leg per worker count (default 32,48,64,74)
with everything else held constant, times each leg, and prints a throughput
comparison table (matches/hr, samples/sec, speedup, per-worker efficiency).

Defaults REPLICATE the reset_league_8_24 curriculum's az-league slots (the
2026-08 in-flight run) so the measured throughput is the production workload:
the EXHAUSTIVE-SELFPLAY matrix (every unordered roster pair x2 = 110 matches +
20 rotating vs-scripted cells = 130 matches/leg on the 10-deck roster; the
slot index advances per leg exactly like consecutive league slots) at the
shared cli_spec training budget — sims=1028 worlds=8 c_puct=2.5 td_n=10
temp_moves=20 and the sb plan-search knobs. Pass --random-draw for the old
mirror_frac-draw schedule instead (then --games/--mirror-frac apply).

Legs call ``az_selfplay.generate(..., bo3=True)`` IN-PROCESS — the same code
path the ``az`` / ``az-league`` cycles use — NOT the standalone ``train.py
az-selfplay`` subcommand, which is bo1: the pooled ``az_data/gen`` window is
bo3-only and bo1 shards would mix silently. Shards therefore land directly in
the normal training pool (``train/az_data/gen`` unless ``--out``) in the pooled
bo3 schema, so the next az-train window picks them up like any other self-play
pass. Each leg uses a DISTINCT seed: every leg contributes fresh games (no
near-duplicate samples in the pool); the matchup schedules are statistically
equivalent across legs, not identical.

Actor-backend caveat the table accounts for: the C++ actor runs ONE process per
distinct (deck_a, deck_b, scripted-seat) matchup group, and ``workers`` only
caps how many groups run at once — so the EFFECTIVE concurrency of a leg is
``min(workers, n_groups)``. Single-focus schedules top out around the roster
size, which is why ``--decks`` defaults to ``league`` (multi-focus over the
whole roster, up to ~100 distinct groups on the 10-deck roster). Each leg's
group count and effective cap are computed from the leg's exact schedule (same
builder, same seed) and printed/recorded; a leg whose cap < workers is flagged,
since its measurement says nothing new beyond the cap. The pure-Python backend
(``--no-actor``) splits matches evenly over exactly ``workers`` processes, so
there the counts always bind.

Optionally finishes with an az-train pass over the freshly written shards and
an az-eval gate (``--train``; both run as ``train.py`` subprocesses, like a
curriculum phase). These too mirror the curriculum's training slot: az-train
runs AUTO half-epoch batches (``--batches 0`` + the default epoch_frac 0.5,
q_mix 0.75) over a window defaulting to the bench legs' shards PLUS the most
recent az-league run's pool shards (``--pool-extra auto`` — completed slots
from the league progress sidecar, plus pooled gate shards and any interrupted
slot's partial shards), so an interrupted league run's data is trained
alongside the fresh bench data; the stock az-train window of 50 would silently
drop most of it — and the gate runs at the training budget (az-eval's default
sims/worlds ARE the training budget) with ``--promote`` and gate-shard
recording+pooling on (``az_data/gate/gate_<ts>``), like the league's gate.
``--no-promote`` / ``--no-gate-shards`` opt out.

Example (backgrounded so the log can be tailed):

    train/.venv/bin/python train/bench_az_workers.py \
        --games 148 --train --promote > /tmp/bench_az_workers.log 2>&1 &

Per-leg results are appended to ``train/az_data/bench_workers_<ts>/results.json``
after every leg (crash-safe: an interrupted run keeps its completed legs), and
the final table is also written to ``summary.txt`` next to it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TRAIN_DIR)
TRAIN_PY = os.path.join(TRAIN_DIR, "train.py")
POOL_DIR = os.path.join(TRAIN_DIR, "az_data", "gen")
LEAGUE_SIDECAR = os.path.join(TRAIN_DIR, "checkpoints", "az",
                              "_az_league_progress.json")


def parse_counts(text: str) -> list:
    counts = [int(t) for t in text.split(",") if t.strip()]
    if not counts:
        raise argparse.ArgumentTypeError("empty --counts list")
    if any(c < 1 for c in counts):
        raise argparse.ArgumentTypeError("worker counts must be >= 1")
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Benchmark AZ self-play throughput across worker counts "
                    "(bo3, shards pooled for the next az-train)")
    ap.add_argument("--counts", type=parse_counts, default=[32, 48, 64, 74],
                    help="Comma-separated worker counts, one leg each "
                         "(default 32,48,64,74)")
    ap.add_argument("--random-draw", action="store_true",
                    help="Use the random mirror_frac-draw schedule instead of "
                         "the curriculum-matching exhaustive-selfplay matrix "
                         "(--games/--mirror-frac apply only in this mode)")
    ap.add_argument("--repeats", type=int, default=None,
                    help="Exhaustive mode: play every self-play cell N times "
                         "per leg (default: the shared cli_spec default, 2 — "
                         "what the 8_24 curriculum runs)")
    ap.add_argument("--scripted-cells", type=int, default=None,
                    help="Exhaustive mode: rotating vs-scripted:hard cells per "
                         "leg (default: cli_spec default, 20 — the curriculum "
                         "value); the slot index advances per leg so legs tile "
                         "different cells")
    ap.add_argument("--slot-base", type=int, default=0,
                    help="Exhaustive mode: slot index of the FIRST leg for the "
                         "rotating scripted-cell slice (leg i uses slot-base+i)")
    ap.add_argument("--games", type=int, default=148,
                    help="MATCHES per leg in --random-draw mode (default 148; "
                         "must be >= the largest worker count or generate() "
                         "clamps workers down to it). IGNORED by the default "
                         "exhaustive matrix, which fixes the count itself")
    ap.add_argument("--decks", default="league",
                    help="Focus-deck pool: 'league' = every decks/league/*.dk "
                         "(default; maximizes distinct actor matchup groups so "
                         "high worker counts can actually bind), or a "
                         "comma-separated list of .dk stems")
    ap.add_argument("--sims", type=int, default=None,
                    help="PUCT sims per decision, TOTAL across --worlds "
                         "(default: the az-cycle default)")
    ap.add_argument("--worlds", type=int, default=None,
                    help="Determinized worlds per search (default: az-cycle default)")
    ap.add_argument("--c-puct", type=float, default=None)
    ap.add_argument("--mirror-frac", type=float, default=None,
                    help="P(opponent == focus deck) per match (default: az-cycle default)")
    ap.add_argument("--td-n", type=int, default=None,
                    help="n-step TD horizon baked into the shards (default: az-cycle default)")
    ap.add_argument("--checkpoint", default=None,
                    help="AZ (.pt) / PPO (.zip) checkpoint or 'gen' "
                         "(default: generalist AZ ckpt, else gen PPO warm-start)")
    ap.add_argument("--seed", type=int, default=1,
                    help="Base seed; leg i uses seed + i*1000003 so every leg "
                         "plays fresh games (default 1)")
    ap.add_argument("--out", default=None,
                    help="Shard output dir (default: the az_data/gen training "
                         "pool, so the next az-train incorporates the shards; "
                         "point elsewhere to keep the bench data OUT of the pool)")
    # Backend knobs (mirror the az-selfplay/az flags)
    ap.add_argument("--actor", action="store_true",
                    help="Force the C++ actor backend (default AUTO: actor iff built)")
    ap.add_argument("--no-actor", action="store_true",
                    help="Force the pure-Python backend (workers bind exactly)")
    ap.add_argument("--actor-device", default="cpu")
    ap.add_argument("--eval-server", action="store_true",
                    help="Force the central GPU eval server (default AUTO)")
    ap.add_argument("--no-eval-server", action="store_true")
    ap.add_argument("--no-cross-world", action="store_true")
    # Final legs
    ap.add_argument("--train", action="store_true",
                    help="After all legs, run `train.py az-train` over the fresh "
                         "shards, then `train.py az-eval` gating the candidate")
    ap.add_argument("--train-deck", default="delver",
                    help="--deck label passed to az-train/az-eval (default delver; "
                         "the shard pool itself is deck-agnostic)")
    ap.add_argument("--train-window", type=int, default=0,
                    help="az-train --window (0 = AUTO: the shards this "
                         "benchmark wrote PLUS the most recent az-league run's "
                         "pool shards per --pool-extra, falling back to 2x the "
                         "bench shards when the league count is unavailable; "
                         "default 0)")
    ap.add_argument("--pool-extra", default="auto",
                    help="Pre-existing pool shards to ALSO cover in the auto "
                         "az-train window, on top of the bench legs' own. "
                         "'auto' (default) counts the most recent az-league "
                         "run's shards — its completed slots' self-play shards "
                         "(from the league progress sidecar) plus everything "
                         "newer in the pool (pooled gate shards + the partial "
                         "shards of an interrupted slot). An integer overrides "
                         "the count exactly; 0 disables (auto window back to "
                         "2x bench shards)")
    ap.add_argument("--train-batches", type=int, default=0,
                    help="az-train --batches (default 0 = AUTO half-epoch via "
                         "the shared epoch_frac default, matching the az cycle "
                         "and the 8_24 curriculum)")
    ap.add_argument("--eval-games", type=int, default=None,
                    help="az-eval --games (default: az-eval's own default)")
    ap.add_argument("--no-eval", action="store_true",
                    help="With --train: skip the az-eval gate")
    ap.add_argument("--no-promote", action="store_true",
                    help="Do NOT pass --promote to az-eval. Default is to pass "
                         "it, like the az cycle's gate (promotion is still "
                         "decided by the sequential test + per-deck floor)")
    ap.add_argument("--no-gate-shards", action="store_true",
                    help="Do NOT record/pool the gate's candidate-vs-incumbent "
                         "matches as training shards. Default on, like the "
                         "curriculum (az-eval --record-dir az_data/gate/...)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print each leg's plan (schedule size, distinct matchup "
                         "groups, effective concurrency cap) and the train/eval "
                         "subprocess argv, then exit without playing anything")
    return ap


def tri_state(force_on: bool, force_off: bool, name: str):
    if force_on and force_off:
        sys.exit(f"error: --{name} and --no-{name} are mutually exclusive")
    return True if force_on else (False if force_off else None)


def leg_seed(base: int, i: int) -> int:
    return base + i * 1000003


def schedule_preview(az_selfplay, args, focus, roster, seed, slot):
    """(n_matches, n_groups) for the EXACT schedule a leg will play.

    Same builders + seed as generate(), so both counts are exact, not
    estimates. Actor groups are keyed (deck_a, deck_b, scripted-seat) — the
    exhaustive mode's vs-scripted cells key separately from same-pair
    self-play cells, exactly as _generate_actor groups them."""
    if args.random_draw:
        sched = az_selfplay.build_matchup_schedule_ex(
            focus, roster, args.games, args.mirror_frac, seed)
        seats = [None] * len(sched)
    else:
        sched, seats = az_selfplay.build_exhaustive_schedule_ex(
            focus, roster, seed, include_scripted=False,
            repeats=args.repeats, scripted_cells=args.scripted_cells,
            slot=slot)
    keys = {(a, b, None if s is None else ("B" if s else "A"))
            for (a, b, _f), s in zip(sched, seats)}
    return len(sched), len(keys)


def fmt_row(cols, widths):
    return "  ".join(str(c).rjust(w) for c, w in zip(cols, widths))


def render_table(legs) -> list:
    """Final comparison table as printable lines."""
    header = ("workers", "eff.cap", "matches", "games", "samples", "shards",
              "elapsed", "match/hr", "samp/s", "samp/s/wkr", "speedup")
    widths = [max(len(h), 9) for h in header]
    lines = [fmt_row(header, widths), fmt_row(["-" * w for w in widths], widths)]
    base_rate = legs[0]["samples_per_sec"] if legs else 0.0
    for leg in legs:
        rate = leg["samples_per_sec"]
        lines.append(fmt_row((
            leg["workers"], leg["effective_cap"], leg["matches"],
            leg["games"] if leg["games"] is not None else "-",
            leg["samples"], leg["shards"],
            f"{leg['elapsed_secs']:.0f}s",
            f"{leg['matches_per_hr']:.1f}",
            f"{rate:.2f}",
            f"{rate / leg['workers']:.3f}",
            f"{rate / base_rate:.2f}x" if base_rate else "-",
        ), widths))
    return lines


def dump_results(bench_dir, payload):
    path = os.path.join(bench_dir, "results.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)
    return path


def run_leg(az_selfplay, args, focus, roster, workers, seed, slot, n_matches,
            use_actor, eval_server):
    t0 = time.monotonic()
    res = az_selfplay.generate(
        focus[0], games=args.games,
        sims=args.sims, worlds=args.worlds, c_puct=args.c_puct,
        workers=workers, checkpoint=args.checkpoint,
        out_dir=args.out, seed=seed, use_actor=use_actor,
        focus_decks=focus, roster=roster, mirror_frac=args.mirror_frac,
        bo3=True, td_n=args.td_n,
        exhaustive_selfplay=not args.random_draw,
        exhaustive_repeats=args.repeats,
        scripted_cells=args.scripted_cells, slot=slot,
        actor_device=args.actor_device, eval_server=eval_server,
        cross_world=not args.no_cross_world)
    elapsed = time.monotonic() - t0
    stats = res.get("stats", {})
    return {
        "workers": workers,
        "seed": seed,
        "slot": slot,
        "matches": n_matches,
        # The actor backend doesn't tally per-game counts (bo3 matches span
        # 1-3 games); None renders as "-" in the table.
        "games": stats.get("games") or None,
        "samples": res["samples"],
        "shards": len(res["shards"]),
        "shard_paths": list(res["shards"]),
        "out_dir": res["out_dir"],
        "elapsed_secs": elapsed,
        "matches_per_hr": n_matches * 3600.0 / elapsed if elapsed else 0.0,
        "samples_per_sec": res["samples"] / elapsed if elapsed else 0.0,
    }


def league_run_pool_shards() -> int:
    """Shard count of the most recent az-league run still sitting in the pool.

    Anchored on the league progress sidecar's per-slot ``shards`` counts (the
    completed slots' self-play shards). Those S files are the newest PLAIN
    (non-gate) shards in the pool apart from anything written after the run
    (nothing, when the run was interrupted), so the S-th-newest plain shard's
    mtime marks the run's start; every pool file at or after that mtime — the
    completed slots, the pooled ``shard_gate-*`` recordings of the run's
    gates, and the partial shards of an interrupted slot — belongs to the run.
    Off by at most the interrupted slot's partial count at the OLD end (a few
    of slot 1's shards fall outside the anchor), which only shrinks the
    window's tail into data the run already trained on. Returns 0 (with a
    printed warning) when the sidecar or pool is unreadable."""
    try:
        with open(LEAGUE_SIDECAR) as fh:
            results = json.load(fh).get("results", [])
        s = sum(int(r.get("shards", 0)) for r in results)
    except (OSError, ValueError) as exc:
        print(f"[bench-az] WARNING: cannot read league sidecar "
              f"{LEAGUE_SIDECAR} ({exc}) — pool-extra auto resolves to 0",
              flush=True)
        return 0
    every = sorted(glob.glob(os.path.join(POOL_DIR, "shard_*.npz")),
                   key=os.path.getmtime)
    plain = [p for p in every
             if not os.path.basename(p).startswith("shard_gate-")]
    if s <= 0 or not plain:
        print("[bench-az] WARNING: league sidecar reports no completed-slot "
              "shards — pool-extra auto resolves to 0", flush=True)
        return 0
    cutoff = os.path.getmtime(plain[max(0, len(plain) - s)])
    return sum(1 for p in every if os.path.getmtime(p) >= cutoff)


def final_train_eval(args, total_new_shards: int, pool_extra: int,
                     dry_run: bool):
    """Compose (and run) the az-train and az-eval subprocess legs."""
    window = args.train_window
    if window == 0:
        if pool_extra > 0:
            window = max(1, total_new_shards + pool_extra)
            print(f"[bench-az] az-train window auto: {total_new_shards} bench "
                  f"+ {pool_extra} league-run shard(s) = {window}", flush=True)
        else:
            window = max(1, 2 * total_new_shards)
    train_cmd = [sys.executable, TRAIN_PY, "az-train",
                 "--deck", args.train_deck, "--window", str(window),
                 "--batches", str(args.train_batches)]
    cmds = [train_cmd]
    if not args.no_eval:
        # Mirror the az cycle's gate: training-budget sims/worlds (az-eval's
        # defaults), --promote (the sequential test + floor still decide), and
        # gate-shard recording pooled into az_data/gen afterward.
        # --gate-max-rounds 1 pins the gate to ONE panel: the real gate buys
        # rounds until its SPRT decides, which is the right behaviour for a
        # promotion verdict and the wrong one for a benchmark, whose leg must
        # cost the same every run to be comparable.
        eval_cmd = [sys.executable, TRAIN_PY, "az-eval",
                    "--deck", args.train_deck, "--candidate", "gen",
                    "--gate-max-rounds", "1"]
        if args.eval_games is not None:
            eval_cmd += ["--games", str(args.eval_games)]
        if not args.no_promote:
            eval_cmd += ["--promote"]
        if args.no_gate_shards:
            eval_cmd += ["--no-pool-shards"]   # else az-eval records by default
        else:
            eval_cmd += ["--record-dir",
                         os.path.join(TRAIN_DIR, "az_data", "gate",
                                      time.strftime("gate_%Y%m%d_%H%M%S"))]
        cmds.append(eval_cmd)
    for cmd in cmds:
        print(f"[bench-az] {'would run' if dry_run else 'running'}: "
              + " ".join(cmd), flush=True)
        if not dry_run:
            subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def main() -> None:
    args = build_arg_parser().parse_args()
    use_actor = tri_state(args.actor, args.no_actor, "actor")
    eval_server = tri_state(args.eval_server, args.no_eval_server, "eval-server")

    # Heavy import kept out of module load so --help stays instant.
    import az_selfplay
    if args.sims is None:
        args.sims = az_selfplay.DEFAULT_AZ_SIMS
    if args.worlds is None:
        args.worlds = az_selfplay.DEFAULT_AZ_WORLDS
    if args.c_puct is None:
        args.c_puct = az_selfplay.DEFAULT_AZ_C_PUCT
    if args.mirror_frac is None:
        args.mirror_frac = az_selfplay.DEFAULT_MIRROR_FRAC
    if args.td_n is None:
        args.td_n = az_selfplay.DEFAULT_TD_N
    if args.repeats is None:
        args.repeats = az_selfplay.DEFAULT_AZ_EXHAUSTIVE_REPEATS
    if args.scripted_cells is None:
        args.scripted_cells = az_selfplay.DEFAULT_AZ_SCRIPTED_CELLS

    # Resolve the opponent roster to an explicit list and hand the SAME list to
    # both the schedule preview and generate() — generate(roster=None) means
    # "the league roster", but build_matchup_schedule_ex(pool=None) means "no
    # pool" (every match a mirror), so the preview must never see None.
    roster = az_selfplay.league_roster()
    if args.decks.strip() == "league":
        focus = list(roster)
        if not focus:
            sys.exit("error: no decks found in decks/league/")
    else:
        focus = [d.strip() for d in args.decks.split(",") if d.strip()]

    if args.train and args.out is not None:
        print("[bench-az] WARNING: --train reads the az_data/gen pool, but "
              "--out routes the bench shards elsewhere — the az-train leg "
              "will NOT see this benchmark's data", flush=True)

    # Resolve --pool-extra BEFORE any leg writes to the pool: 'auto' counts the
    # most recent az-league run's shards by anchoring on the league sidecar's
    # completed-slot totals, and the count must not include this benchmark's
    # own output.
    if args.train:
        if str(args.pool_extra).strip().lower() == "auto":
            pool_extra = league_run_pool_shards()
            print(f"[bench-az] pool-extra auto: {pool_extra} shard(s) from the "
                  f"most recent az-league run will be covered by the az-train "
                  f"window alongside the bench shards", flush=True)
        else:
            pool_extra = max(0, int(args.pool_extra))
    else:
        pool_extra = 0

    max_count = max(args.counts)
    if args.random_draw and args.games < max_count:
        print(f"[bench-az] WARNING: --games {args.games} < largest worker count "
              f"{max_count}; generate() clamps workers to the match count, so "
              f"the biggest legs will all measure the same thing", flush=True)

    bench_dir = os.path.join(
        TRAIN_DIR, "az_data", "bench_workers_" + time.strftime("%Y%m%d_%H%M%S"))
    payload = {
        "config": {k: v for k, v in vars(args).items()},
        "focus_decks": focus,
        "pool_extra_resolved": pool_extra,
        "legs": [],
    }

    mode = ("random-draw" if args.random_draw else
            f"exhaustive-selfplay x{args.repeats} + "
            f"{args.scripted_cells} scripted cells (curriculum-matching)")
    print(f"[bench-az] counts={args.counts} schedule={mode} "
          f"sims={args.sims} worlds={args.worlds} c_puct={args.c_puct} "
          f"td_n={args.td_n} focus={len(focus)} deck(s) "
          f"out={args.out or 'az_data/gen (pool)'}", flush=True)

    # Per-leg plan: exact schedule size + distinct actor matchup groups.
    plans = []
    for i, count in enumerate(args.counts):
        seed = leg_seed(args.seed, i)
        slot = args.slot_base + i
        n_matches, n_groups = schedule_preview(
            az_selfplay, args, focus, roster, seed, slot)
        cap = min(count, n_groups) if use_actor is not False else count
        note = ""
        if use_actor is not False and n_groups < count:
            note = (f"  <-- actor concurrency saturates at {n_groups} groups; "
                    f"this count cannot bind on the actor backend")
        print(f"[bench-az] leg {i + 1}: workers={count} seed={seed} "
              f"slot={slot} matches={n_matches} matchup-groups={n_groups} "
              f"effective-cap={cap}{note}", flush=True)
        plans.append((count, seed, slot, n_matches, n_groups, cap))

    if args.dry_run:
        if args.train:
            # Estimate bench output as one ~leg-per-worker-flush shard batch
            # per leg is unknowable up front; show the window math with 0 bench
            # shards (the real run substitutes the true count).
            final_train_eval(args, total_new_shards=0, pool_extra=pool_extra,
                             dry_run=True)
        print("[bench-az] dry run — nothing played")
        return

    os.makedirs(bench_dir, exist_ok=True)
    print(f"[bench-az] results dir: {bench_dir}", flush=True)

    try:
        for i, (count, seed, slot, n_matches, n_groups, cap) in enumerate(plans):
            print(f"\n[bench-az] ===== leg {i + 1}/{len(plans)}: "
                  f"workers={count} =====", flush=True)
            leg = run_leg(az_selfplay, args, focus, roster, count, seed,
                          slot, n_matches, use_actor, eval_server)
            leg["matchup_groups"] = n_groups
            leg["effective_cap"] = cap
            payload["legs"].append(leg)
            dump_results(bench_dir, payload)
            print(f"[bench-az] leg {i + 1} done: {leg['samples']} samples, "
                  f"{leg['shards']} shards in {leg['elapsed_secs']:.0f}s "
                  f"({leg['samples_per_sec']:.2f} samp/s, "
                  f"{leg['matches_per_hr']:.1f} matches/hr)", flush=True)
    finally:
        if payload["legs"]:
            lines = render_table(payload["legs"])
            print("\n[bench-az] throughput comparison (speedup vs the "
                  f"{payload['legs'][0]['workers']}-worker leg):")
            for ln in lines:
                print("  " + ln)
            with open(os.path.join(bench_dir, "summary.txt"), "w") as fh:
                fh.write("\n".join(lines) + "\n")
            dump_results(bench_dir, payload)
            print(f"[bench-az] results: {os.path.join(bench_dir, 'results.json')}",
                  flush=True)

    total_new = sum(leg["shards"] for leg in payload["legs"])
    if args.out is None:
        print(f"[bench-az] {total_new} new shard(s) are in the az_data/gen "
              f"training pool — the next az-train window picks them up "
              f"(use --window >= {total_new} to cover all of them)", flush=True)
    if args.train:
        final_train_eval(args, total_new_shards=total_new,
                         pool_extra=pool_extra, dry_run=False)


if __name__ == "__main__":
    main()
