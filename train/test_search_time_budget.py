"""Standalone check for the wall-clock per-decision search budget.

Verifies mcts.run_search(..., time_budget_s=T): on a real search env parked at a
loop-safe root, a timed search terminates in roughly the budget (loose,
non-flaky bounds) and returns a valid visit distribution with at least the
one-sim-per-world floor of visits. Also covers the min-deadline stability
early-stop (time_budget_min_s): an obvious root (peaked priors) stops well
before the max deadline with ``stopped_early`` set, and a contested root
(uniform priors, neutral values) runs to the max deadline without stopping.

The time_budget_s=None (fixed-budget) path is NOT re-checked here — it is covered
bit-exactly by test_mcts_parity.py (the actor visit-parity corpus), which would
catch any reordering of the sequential loop this change refactored.

Run standalone (not wired into a ci_check tier — wall-clock assertions can be
flaky on a loaded CI machine; the bounds here are deliberately generous):
    train/.venv/bin/python train/test_search_time_budget.py
"""

import sys
import time

import numpy as np

from mcts import UniformEvaluator, run_search
from search_env import SearchRoboMageEnv

DECK = "league/ur_delver"
SEED = 1
WORLDS = 4
BUDGET_S = 0.3


class PeakedEvaluator:
    """Near-certain priors on action 0, neutral value: an 'obvious' root that
    the stability early-stop should terminate long before the max deadline."""

    def evaluate(self, obs: np.ndarray, num_choices: int) -> tuple[np.ndarray, float]:
        p = np.full(num_choices, 0.03 / max(1, num_choices - 1), dtype=np.float64)
        p[0] = 0.97
        return p / p.sum(), 0.0


def _first_search_root(env):
    """Step the env (passing / first-choice) until it is parked at a loop-safe,
    branching root a search controller would actually search. Returns True there,
    False if the game ended first."""
    for _ in range(500):
        if getattr(env, "last_search_safe", False) and env._num_choices > 1:
            return True
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            return False
    return False


def _check_timed_no_min(env, n) -> bool:
    """Original timed-budget case: clock is the sole terminator, no early stop."""
    ok = True
    t0 = time.monotonic()
    result = run_search(env, UniformEvaluator(), sims=0, worlds=WORLDS,
                        time_budget_s=BUDGET_S)
    dt = time.monotonic() - t0

    # Loose wall-clock bounds: at least most of the budget, and not wildly over
    # (one in-flight sim + Python overhead). Generous ceiling so a loaded machine
    # does not make this flaky.
    if not (0.2 <= dt <= BUDGET_S + 3.0):
        print(f"FAIL: timed search took {dt:.3f}s, expected ~{BUDGET_S}s "
              f"(bounds 0.2..{BUDGET_S + 3.0})", file=sys.stderr)
        ok = False

    # Visit floor: at least one sim per world (the guaranteed minimum), and
    # every visit landed in the root menu.
    total = int(result.visits.sum())
    if total < WORLDS:
        print(f"FAIL: visits sum {total} < worlds {WORLDS} (floor violated)",
              file=sys.stderr)
        ok = False
    if result.sims_run < WORLDS:
        print(f"FAIL: sims_run {result.sims_run} < worlds {WORLDS}", file=sys.stderr)
        ok = False

    # Valid argmax over the real menu.
    best = result.best_action()
    if not (0 <= best < n) or result.num_choices != n:
        print(f"FAIL: best_action {best} out of range for {n} choices "
              f"(num_choices={result.num_choices})", file=sys.stderr)
        ok = False

    # No min deadline was given, so the stability stop must never have fired.
    if result.stopped_early:
        print("FAIL: stopped_early set without time_budget_min_s", file=sys.stderr)
        ok = False

    # The env must be back at the true root, ready for the real step.
    if env._num_choices != n:
        print(f"FAIL: env not restored to root ({env._num_choices} vs {n})",
              file=sys.stderr)
        ok = False

    if ok:
        print(f"PASS: timed search ran {result.sims_run} sims across {WORLDS} "
              f"worlds in {dt:.3f}s (budget {BUDGET_S}s), {total} root visits, "
              f"argmax={best}/{n}")
    return ok


def _check_early_stop_obvious(env, n) -> bool:
    """Peaked priors concentrate visits on one action; the stability check must
    end the search well before the (deliberately huge) max deadline."""
    ok = True
    max_s = 6.0
    t0 = time.monotonic()
    result = run_search(env, PeakedEvaluator(), sims=0, worlds=WORLDS,
                        time_budget_s=max_s, time_budget_min_s=0.25)
    dt = time.monotonic() - t0

    if not result.stopped_early:
        print(f"FAIL: obvious root did not early-stop (ran {dt:.3f}s, "
              f"{result.sims_run} sims)", file=sys.stderr)
        ok = False
    # Generous ceiling: min deadline (0.25s) + a couple of 32-sim check windows.
    if not (0.2 <= dt <= 4.0):
        print(f"FAIL: obvious root took {dt:.3f}s, expected well under the "
              f"{max_s}s max (bounds 0.2..4.0)", file=sys.stderr)
        ok = False
    if env._num_choices != n:
        print(f"FAIL: env not restored to root ({env._num_choices} vs {n})",
              file=sys.stderr)
        ok = False

    if ok:
        print(f"PASS: obvious root early-stopped in {dt:.3f}s "
              f"({result.sims_run} sims, max budget {max_s}s)")
    return ok


def _check_no_stop_contested(env, n) -> bool:
    """Uniform priors + neutral values keep root visits near-even, so the
    stability check must NOT fire and the search runs to the max deadline."""
    ok = True
    max_s = 1.0
    t0 = time.monotonic()
    result = run_search(env, UniformEvaluator(), sims=0, worlds=WORLDS,
                        time_budget_s=max_s, time_budget_min_s=0.25)
    dt = time.monotonic() - t0

    if result.stopped_early:
        print(f"FAIL: contested (uniform) root early-stopped after {dt:.3f}s",
              file=sys.stderr)
        ok = False
    if dt < 0.9 * max_s:
        print(f"FAIL: contested root returned in {dt:.3f}s, expected to run to "
              f"the {max_s}s deadline", file=sys.stderr)
        ok = False
    if env._num_choices != n:
        print(f"FAIL: env not restored to root ({env._num_choices} vs {n})",
              file=sys.stderr)
        ok = False

    if ok:
        print(f"PASS: contested root ran to the deadline ({dt:.3f}s, "
              f"{result.sims_run} sims, no early stop)")
    return ok


def main() -> int:
    env = SearchRoboMageEnv(deck_a=DECK, deck_b=DECK, bo3=False)
    env.MAX_STEPS = env.MAX_STEPS_BO3 = 1 << 30  # no training-only truncation
    try:
        env.reset(options={"engine_seed": SEED})
        if not _first_search_root(env):
            print("FAIL: never reached a loop-safe branching root", file=sys.stderr)
            return 1
        n = env._num_choices
        # run_search restores the env to this same root after every call, so
        # the three cases search the identical decision back-to-back.
        ok = _check_timed_no_min(env, n)
        ok = _check_early_stop_obvious(env, n) and ok
        ok = _check_no_stop_contested(env, n) and ok
    finally:
        env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
