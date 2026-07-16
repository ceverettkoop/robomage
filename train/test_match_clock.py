"""Standalone checks for the match-level chess clock and paced responses.

Covers MatchClock's allocation math (horizon estimate, difficulty multiplier,
clamps, Fischer floor, debit/reset) with pure Python — no engine binary — and
the SearchController paced-response floor on the fallback path (no bound env)
with a stub evaluator.

Run standalone (not wired into a ci_check tier — the pacing case asserts real
wall-clock sleeps, which can be flaky on a loaded CI machine):
    train/.venv/bin/python train/test_match_clock.py
"""

import sys
import time

import numpy as np

from opponents import MatchClock, SearchController

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        return
    FAILURES.append(msg)
    print(f"FAIL: {msg}", file=sys.stderr)


class StubEvaluator:
    """Uniform priors, neutral value — torch-free SearchController evaluator."""

    def evaluate(self, obs, num_choices):
        return np.full(num_choices, 1.0 / num_choices, dtype=np.float64), 0.0


def test_allocation_math() -> None:
    clock = MatchClock(1500.0)

    # Fresh 25-min bank, game 1, turn 0: horizon clamps to 120 -> base 12.5s;
    # no priors -> multiplier 1.0.
    lo, hi = clock.allocate(turn=0, game_number=1, self_wins=0, opp_wins=0,
                            is_sideboard=False, priors=None, num_choices=5)
    check(abs(hi - 12.5) < 1e-9, f"fresh-bank base: expected 12.5, got {hi}")
    check(abs(lo - 0.25 * hi) < 1e-9, f"t_lo should be hi/4, got {lo} vs hi {hi}")

    # Peaked priors clamp the multiplier low; uniform wide menus clamp it high.
    peaked = np.array([0.97, 0.01, 0.01, 0.01])
    _, hi_peak = clock.allocate(turn=0, game_number=1, self_wins=0, opp_wins=0,
                                is_sideboard=False, priors=peaked, num_choices=4)
    check(abs(hi_peak - 12.5 * clock._M_LO) < 1e-9,
          f"peaked multiplier should clamp to {clock._M_LO}, got hi {hi_peak}")
    uniform8 = np.full(8, 1.0 / 8.0)
    _, hi_uni = clock.allocate(turn=0, game_number=1, self_wins=0, opp_wins=0,
                               is_sideboard=False, priors=uniform8, num_choices=8)
    check(abs(hi_uni - 12.5 * clock._M_HI) < 1e-9,
          f"uniform-8 multiplier should clamp to {clock._M_HI}, got hi {hi_uni}")

    # Deep in a decided-length game the horizon floors at _HORIZON_LO.
    lo, hi = clock.allocate(turn=30, game_number=3, self_wins=1, opp_wins=1,
                            is_sideboard=False, priors=None, num_choices=5)
    check(abs(hi - min(1500.0 / clock._HORIZON_LO, clock.t_max,
                       1500.0 / 4.0)) < 1e-9,
          f"late-game allocation wrong: {hi}")

    # bo1 (game_number 0) has no extra-game term: same horizon as a 1-1 bo3.
    _, hi_bo1 = clock.allocate(turn=10, game_number=0, self_wins=0, opp_wins=0,
                               is_sideboard=False, priors=None, num_choices=5)
    _, hi_g3 = clock.allocate(turn=10, game_number=3, self_wins=1, opp_wins=1,
                              is_sideboard=False, priors=None, num_choices=5)
    check(abs(hi_bo1 - hi_g3) < 1e-9,
          f"bo1 vs 1-1 horizons should match: {hi_bo1} vs {hi_g3}")

    # A 0-0 score budgets for more future games than a 1-0 lead.
    _, hi_00 = clock.allocate(turn=5, game_number=1, self_wins=0, opp_wins=0,
                              is_sideboard=False, priors=None, num_choices=5)
    _, hi_10 = clock.allocate(turn=5, game_number=2, self_wins=1, opp_wins=0,
                              is_sideboard=False, priors=None, num_choices=5)
    check(hi_00 < hi_10, f"0-0 should allocate less per decision than 1-0 "
                         f"(longer horizon): {hi_00} vs {hi_10}")

    # Sideboard roots cap at sb_t_max.
    _, hi_sb = clock.allocate(turn=0, game_number=2, self_wins=1, opp_wins=0,
                              is_sideboard=True, priors=None, num_choices=20)
    check(hi_sb <= clock.sb_t_max + 1e-9,
          f"sideboard allocation {hi_sb} exceeds sb_t_max {clock.sb_t_max}")

    # remaining/4 cap: a nearly-empty bank can't be blown on one decision.
    small = MatchClock(2.0)
    lo, hi = small.allocate(turn=0, game_number=1, self_wins=0, opp_wins=0,
                            is_sideboard=False, priors=None, num_choices=5)
    check(abs(hi - small.t_min) < 1e-9,
          f"2s bank should floor at t_min ({small.t_min}), got {hi}")

    # Fischer floor on an empty bank; debit clamps at zero; reset restores.
    small.debit(10.0)
    check(small.remaining == 0.0, f"debit should clamp at 0, got {small.remaining}")
    lo, hi = small.allocate(turn=0, game_number=1, self_wins=0, opp_wins=0,
                            is_sideboard=False, priors=None, num_choices=5)
    check(lo == small.t_min and hi == small.t_min,
          f"empty bank should allocate (t_min, t_min), got ({lo}, {hi})")
    small.reset()
    check(small.remaining == 2.0, f"reset should restore the bank, got {small.remaining}")

    if not FAILURES:
        print("PASS: MatchClock allocation math")


def test_paced_floor() -> None:
    # No bound env -> every choose() takes the fallback path (evaluator argmax).
    paced = SearchController(StubEvaluator(), paced=True, rng_seed=1,
                             clock=1500.0)
    obs = np.zeros(8192, dtype=np.float32)  # fallback path never indexes deep
    t0 = time.monotonic()
    a = paced.choose(obs, 3)
    dt = time.monotonic() - t0
    check(0 <= a < 3, f"paced choose returned invalid action {a}")
    check(0.5 <= dt <= 1.5, f"paced fallback took {dt:.3f}s, expected the "
                            f"~0.5-0.9s jittered floor")
    # The pad must not be debited from the bank (fallback thinks in ~ms).
    spent = paced.stats["clock_bank"] - paced.stats["clock_remaining"]
    check(spent < 0.1, f"paced pad leaked into the clock bank: {spent:.3f}s")

    unpaced = SearchController(StubEvaluator(), rng_seed=1)
    t0 = time.monotonic()
    unpaced.choose(obs, 3)
    dt = time.monotonic() - t0
    check(dt < 0.1, f"unpaced fallback took {dt:.3f}s, expected instant")

    # bind_env resets the bank for the next match.
    paced._clock.debit(100.0)
    paced.bind_env(None)
    check(paced.stats["clock_remaining"] == paced.stats["clock_bank"],
          "bind_env should reset the clock bank")

    if not FAILURES:
        print("PASS: paced-response floor and clock debit")


def main() -> int:
    test_allocation_math()
    test_paced_floor()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s)", file=sys.stderr)
        return 1
    print("match clock: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
