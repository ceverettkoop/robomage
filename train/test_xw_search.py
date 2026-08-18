#!/usr/bin/env python3
"""Cross-world batched leaf evaluation gates for the PYTHON search stack
(train/mcts.py cross_world — the GUI play / analysis modes' scheduler; the C++
actor's twin has its own gates in test_mcts_parity.py).

Torch-free EXACT legs (run in the default `make check` xwsearch tier):
an evaluator without ``evaluate_batch`` is called row-by-row, so cross-world
scheduling must reproduce the sequential search BIT-FOR-BIT —

  * run_search(cross_world=True) == run_search() under UniformEvaluator, at
    several searched roots of a real deterministic game (pinned world seeds):
    visits, q, w_sum, root_value, world_values, sims_run all equal;
  * IncrementalSearch(cross_world=True) in ragged chunks == the same totals;
  * a rollout budget (sb-style) leaves cross_world INERT: the flag on must
    equal the flag off exactly (the sequential fallback path).

Torch legs (auto-skip when torch is absent — CI containers stay green):

  * AZEvaluator.evaluate_batch row-consistency: batched rows match the
    single-row evaluate() to float tolerance (batched-GEMM ulps only), on a
    fresh deterministic net;
  * a real-net cross-vs-sequential run_search: equal total visits and an
    argmax-agreement REPORT (not asserted at 1.0 — ulp flips on near-ties are
    legitimate — but a floor guards gross breakage).

Run: train/.venv/bin/python train/test_xw_search.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli_spec import BIN_DIR  # noqa: E402
from mcts import IncrementalSearch, UniformEvaluator, run_search  # noqa: E402
from search_env import SearchRoboMageEnv  # noqa: E402

SEED = 7
WORLD_SEEDS = [101, 202, 303, 404]
WORLDS = 4
SIMS = 64
N_ROOTS = 4          # searched roots compared per game


class XwTestError(AssertionError):
    pass


def _write_decks():
    """The fast lopsided matchup test_analysis_session uses (bears vs swamps),
    so a game reaches searchable roots quickly and deterministically."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    a = os.path.join(d, "xw_test_a.dk")
    b = os.path.join(d, "xw_test_b.dk")
    with open(a, "w") as f:
        f.write("36 Grizzly Bears\n24 Forest\nSIDEBOARD:\n15 Mountain\n")
    with open(b, "w") as f:
        f.write("60 Swamp\nSIDEBOARD:\n15 Mountain\n")
    return "temp/xw_test_a", "temp/xw_test_b", [a, b]


def _rm(paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _drive_to_safe(env, min_nc=2, cap=400):
    for _ in range(cap):
        if env.last_search_safe and env._num_choices >= min_nc:
            return
        env.step(0)
    raise XwTestError(f"no safe decision with >= {min_nc} choices in {cap} steps")


def _cmp_results(tag, a, b):
    """Exact equality of two SearchResults (a = cross, b = sequential)."""
    if a.sims_run != b.sims_run:
        raise XwTestError(f"[{tag}] sims_run {a.sims_run} != {b.sims_run}")
    if not np.array_equal(a.visits, b.visits):
        raise XwTestError(f"[{tag}] visits differ:\n  cross {a.visits}\n"
                          f"  seq   {b.visits}")
    if not np.array_equal(a.w_sum, b.w_sum):
        raise XwTestError(f"[{tag}] w_sum differ")
    if not np.array_equal(a.q, b.q):
        raise XwTestError(f"[{tag}] q differ")
    if a.root_value != b.root_value:
        raise XwTestError(f"[{tag}] root_value {a.root_value} != {b.root_value}")
    if not np.array_equal(a.world_values, b.world_values):
        raise XwTestError(f"[{tag}] world_values differ")


def test_uniform_exact():
    """Uniform evaluator: cross-world == sequential bit-exact, run_search AND
    ragged-chunk IncrementalSearch, over several searched roots of one game."""
    deck_a, deck_b, paths = _write_decks()
    ev = UniformEvaluator()
    total = 0
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b)
        try:
            env.reset(options={"engine_seed": SEED})
            for r in range(N_ROOTS):
                _drive_to_safe(env)
                seq = run_search(env, ev, sims=SIMS, worlds=WORLDS,
                                 world_seeds=WORLD_SEEDS)
                xw = run_search(env, ev, sims=SIMS, worlds=WORLDS,
                                world_seeds=WORLD_SEEDS, cross_world=True)
                _cmp_results(f"root {r} run_search", xw, seq)

                s = IncrementalSearch(env, ev, worlds=WORLDS,
                                      world_seeds=WORLD_SEEDS,
                                      cross_world=True)
                try:
                    for n in (5, 16, 43):          # ragged, totals 64
                        stats = s.run_chunk(n)
                    if not np.array_equal(stats.visits, seq.visits):
                        raise XwTestError(
                            f"root {r}: chunked cross visits != sequential "
                            f"run_search visits")
                    if not np.array_equal(stats.q, seq.q):
                        raise XwTestError(f"root {r}: chunked cross q differ")
                finally:
                    s.close()
                total += int(seq.visits.sum())
                # Step along the searched line so each root is a new position.
                env.step(seq.best_action())
        finally:
            env.close()
    finally:
        _rm(paths)
    print(f"PASS [xw-uniform-exact]: cross-world == sequential bit-exact over "
          f"{N_ROOTS} searched roots ({total} total root visits; run_search + "
          f"ragged-chunk IncrementalSearch)")
    return 0


def test_rollout_inert():
    """A budget with rollouts on must leave cross_world INERT (the sequential
    fallback): flag on == flag off, bit-exact."""
    deck_a, deck_b, paths = _write_decks()
    ev = UniformEvaluator()
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b)
        try:
            env.reset(options={"engine_seed": SEED})
            _drive_to_safe(env)
            base = run_search(env, ev, sims=16, worlds=2,
                              world_seeds=WORLD_SEEDS, rollout_turns=2)
            xw = run_search(env, ev, sims=16, worlds=2,
                            world_seeds=WORLD_SEEDS, rollout_turns=2,
                            cross_world=True)
            _cmp_results("rollout-inert", xw, base)
        finally:
            env.close()
    finally:
        _rm(paths)
    print("PASS [xw-rollout-inert]: cross_world is a no-op for a rollout "
          "budget (sequential fallback, bit-exact)")
    return 0


def test_torch_legs():
    """AZEvaluator.evaluate_batch row-consistency + real-net cross-vs-seq
    agreement. Self-skips without torch."""
    try:
        import torch
    except ImportError:
        print("SKIP [xw-torch]: torch not installed — evaluate_batch and "
              "real-net legs skipped (uniform EXACT legs above still gate "
              "the scheduler)")
        return 0
    from az_net import AZNet, AZEvaluator, obs_space_from_const

    torch.manual_seed(0)
    torch.set_num_threads(1)
    net = AZNet(obs_space_from_const()).eval()
    ev = AZEvaluator(net, warn=False)

    deck_a, deck_b, paths = _write_decks()
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b)
        try:
            env.reset(options={"engine_seed": SEED})
            _drive_to_safe(env)

            # (a) evaluate_batch rows match evaluate() (batched-GEMM ulps).
            obs = env._obs.copy()
            nc = env._num_choices
            rows = [(obs, nc), (obs, max(2, nc - 1)), (obs, nc)]
            batched = ev.evaluate_batch([o for o, _ in rows],
                                        [n for _, n in rows])
            for i, ((o, n), (bp, bv)) in enumerate(zip(rows, batched)):
                sp, sv = ev.evaluate(o, n)
                if bp.shape != sp.shape:
                    raise XwTestError(f"evaluate_batch row {i} shape mismatch")
                if not np.allclose(bp, sp, atol=1e-5):
                    raise XwTestError(
                        f"evaluate_batch row {i} priors drift > 1e-5:\n"
                        f"  batch {bp}\n  single {sp}")
                if abs(bv - sv) > 1e-5:
                    raise XwTestError(
                        f"evaluate_batch row {i} value drift: {bv} vs {sv}")
            print("PASS [xw-batch-rows]: evaluate_batch rows match evaluate() "
                  "to 1e-5")

            # (b) real net: equal budgets, agreement report with a gross floor.
            agree = 0
            n_roots = 0
            for r in range(N_ROOTS):
                _drive_to_safe(env)
                seq = run_search(env, ev, sims=SIMS, worlds=WORLDS,
                                 world_seeds=WORLD_SEEDS)
                xw = run_search(env, ev, sims=SIMS, worlds=WORLDS,
                                world_seeds=WORLD_SEEDS, cross_world=True)
                if int(xw.visits.sum()) != int(seq.visits.sum()):
                    raise XwTestError(
                        f"root {r}: cross total visits "
                        f"{int(xw.visits.sum())} != {int(seq.visits.sum())}")
                agree += int(np.argmax(xw.visits) == np.argmax(seq.visits))
                n_roots += 1
                env.step(seq.best_action())
            frac = agree / n_roots
            print(f"REPORT [xw-net]: cross vs sequential argmax agreement "
                  f"{agree}/{n_roots} = {frac:.3f} (ulp-level drift only)")
            if frac < 0.5:
                raise XwTestError(
                    f"cross-world argmax agreement {frac:.3f} < 0.5 — far "
                    f"beyond batched-GEMM ulp drift; scheduler bug likely")
        finally:
            env.close()
    finally:
        _rm(paths)
    return 0


def main() -> int:
    test_uniform_exact()
    test_rollout_inert()
    test_torch_legs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
