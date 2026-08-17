"""Standalone checks for the match-level chess clock and paced responses.

Covers MatchClock's allocation math (horizon estimate, difficulty multiplier,
clamps, Fischer floor, debit/reset), its non-linear front-loaded spend curve
with the final-five-minutes reserve, the match-point value closeout, and the
symmetric giveup deadline cap (a decisively-lost current game) — all
pure Python, no engine binary — plus the SearchController paced-response floor
on the fallback path (no bound env) with a stub evaluator.

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

    # Fresh 25-min bank, game 1, turn 0, nothing observed: the horizon is the
    # match prior — a full game (120) plus 1.5 expected future games — ≈309,
    # so ~300 decisions share the bank; no priors -> multiplier 1.0. A FULL
    # bank sits at the top of the pacing curve, so the proportional share is
    # front-loaded by exactly 2x (frac == 1 -> 1 + sqrt(1)).
    h0 = clock._GAME_DEC_EST + 1.5 * (clock._GAME_DEC_EST + clock._SB_DEC_EST)
    full = 2.0 * 1500.0 / h0
    lo, hi = clock.allocate(turn=0, game_number=1, self_wins=0, opp_wins=0,
                            is_sideboard=False, priors=None, num_choices=5)
    check(abs(hi - full) < 1e-9,
          f"fresh-bank base: expected {full:.3f}, got {hi}")
    check(abs(lo - 0.25 * hi) < 1e-9, f"t_lo should be hi/4, got {lo} vs hi {hi}")

    # Peaked priors clamp the multiplier low; uniform wide menus clamp it high.
    base = full
    peaked = np.array([0.97, 0.01, 0.01, 0.01])
    _, hi_peak = clock.allocate(turn=0, game_number=1, self_wins=0, opp_wins=0,
                                is_sideboard=False, priors=peaked, num_choices=4)
    check(abs(hi_peak - base * clock._M_LO) < 1e-9,
          f"peaked multiplier should clamp to {clock._M_LO}, got hi {hi_peak}")
    uniform8 = np.full(8, 1.0 / 8.0)
    _, hi_uni = clock.allocate(turn=0, game_number=1, self_wins=0, opp_wins=0,
                               is_sideboard=False, priors=uniform8, num_choices=8)
    check(abs(hi_uni - base * clock._M_HI) < 1e-9,
          f"uniform-8 multiplier should clamp to {clock._M_HI}, got hi {hi_uni}")

    # SHORTER-than-prior revision: the match reached game 3 after only 90
    # decisions (two ~45-decision games), so the per-game estimate shrinks
    # and each remaining decision gets MORE time than the unrevised prior.
    fast = MatchClock(1500.0)
    for g, n in ((1, 45), (2, 45)):
        for _ in range(n):
            fast.note_decision(g)
    fast.note_decision(3)
    _, hi_fast = fast.allocate(turn=0, game_number=3, self_wins=1, opp_wins=1,
                               is_sideboard=False, priors=None, num_choices=5)
    check(hi_fast > 1500.0 / clock._GAME_DEC_EST,
          f"short-game history should raise per-decision spend: {hi_fast}")
    check(fast._horizon(0, 3, 1, 1) < clock._GAME_DEC_EST,
          "game 3 after 90 decisions should expect a shorter-than-prior tail")

    # LONGER-than-prior revision: 250 decisions and still in game 1 — the
    # observed per-turn rate projects a longer match than the 300 prior, so
    # per-decision spend shrinks vs an on-pace game at the same turn.
    slow = MatchClock(1500.0)
    for _ in range(250):
        slow.note_decision(1)
    _, hi_slow = slow.allocate(turn=20, game_number=1, self_wins=0, opp_wins=0,
                               is_sideboard=False, priors=None, num_choices=5)
    pace = MatchClock(1500.0)
    for _ in range(100):
        pace.note_decision(1)
    _, hi_pace = pace.allocate(turn=20, game_number=1, self_wins=0, opp_wins=0,
                               is_sideboard=False, priors=None, num_choices=5)
    check(hi_slow < hi_pace,
          f"a hot game should shrink per-decision spend: {hi_slow} vs {hi_pace}")
    check(slow.decisions + slow._horizon(20, 1, 0, 0) > slow._MATCH_DEC_EST,
          "250 decisions into game 1 should project past the 300 prior")

    # Deep in the last game the horizon floors at _HORIZON_LO (t_max caps).
    lo, hi = clock.allocate(turn=30, game_number=3, self_wins=1, opp_wins=1,
                            is_sideboard=False, priors=None, num_choices=5)
    check(abs(hi - min(2.0 * 1500.0 / clock._HORIZON_LO, clock.t_max,
                       1500.0 / 4.0)) < 1e-9,
          f"late-game allocation wrong: {hi}")

    # bo1 (game_number 0) budgets a single game: horizon = the per-game prior.
    _, hi_bo1 = clock.allocate(turn=0, game_number=0, self_wins=0, opp_wins=0,
                               is_sideboard=False, priors=None, num_choices=5)
    check(abs(hi_bo1 - 2.0 * 1500.0 / clock._GAME_DEC_EST) < 1e-9,
          f"fresh bo1 should divide the bank over one game: {hi_bo1}")

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

    # Fischer floor on an empty bank; debit clamps at zero; reset restores
    # the bank AND the pace counters.
    small.debit(10.0)
    check(small.remaining == 0.0, f"debit should clamp at 0, got {small.remaining}")
    lo, hi = small.allocate(turn=0, game_number=1, self_wins=0, opp_wins=0,
                            is_sideboard=False, priors=None, num_choices=5)
    check(lo == small.t_min and hi == small.t_min,
          f"empty bank should allocate (t_min, t_min), got ({lo}, {hi})")
    small.note_decision(1)
    small.reset()
    check(small.remaining == 2.0, f"reset should restore the bank, got {small.remaining}")
    check(small.decisions == 0 and not small._game_start_dec,
          "reset should clear the pace counters")

    if not FAILURES:
        print("PASS: MatchClock allocation math")


def legacy_allocate(clock, *, turn, game_number, self_wins, opp_wins,
                    is_sideboard, priors, num_choices):
    """The PRE-CHANGE ``_horizon``/``allocate``, spelled out here as the
    regression oracle: uniform proportional spend, no value input."""
    cur_start = clock._game_start_dec[-1] if clock._game_start_dec else 0
    in_cur = max(0, clock.decisions - cur_start)
    completed = max(0, game_number - 1)
    per_game = clock._GAME_DEC_EST
    if completed > 0:
        observed = cur_start / completed
        per_game = ((clock._PRIOR_GAME_W * clock._GAME_DEC_EST
                     + completed * observed)
                    / (clock._PRIOR_GAME_W + completed))
    turn = max(0, turn)
    turns_left = max(clock._TURN_END_EST - turn, clock._MIN_TURNS_LEFT)
    prior_rate = per_game / clock._TURN_END_EST
    rate = ((clock._RATE_PRIOR_W * prior_rate + in_cur)
            / (clock._RATE_PRIOR_W + turn))
    in_game = rate * turns_left
    if game_number <= 0 or (self_wins >= 1 and opp_wins >= 1):
        extra_games = 0.0
    elif self_wins == 0 and opp_wins == 0:
        extra_games = 1.5
    else:
        extra_games = 0.5
    horizon = in_game + extra_games * (per_game + clock._SB_DEC_EST)
    c = min(max(horizon, clock._HORIZON_LO), clock._HORIZON_HI)
    base = clock.remaining / c
    m = clock._difficulty(priors, num_choices)
    t_cap = clock.sb_t_max if is_sideboard else clock.t_max
    t_hi = min(base * m, t_cap, clock.remaining / 4.0)
    t_hi = max(t_hi, clock.t_min)
    t_lo = min(max(0.25 * t_hi, clock.t_min), t_hi)
    return t_lo, t_hi


def _hi(clock, **kwargs) -> float:
    """The max deadline for a decision, with unopinionated defaults."""
    call = dict(turn=0, game_number=1, self_wins=0, opp_wins=0,
                is_sideboard=False, priors=None, num_choices=5)
    call.update(kwargs)
    return clock.allocate(**call)[1]


def test_pacing_curve() -> None:
    # 30-min bank, fresh: horizon is the match prior (309 decisions).
    h0 = (MatchClock._GAME_DEC_EST
          + 1.5 * (MatchClock._GAME_DEC_EST + MatchClock._SB_DEC_EST))
    reserve = MatchClock._RESERVE_S

    # Full bank -> frac 1.0 -> 2x the proportional share (hand-computed).
    clock = MatchClock(1800.0)
    check(abs(_hi(clock) - 2.0 * (1800.0 / h0)) < 1e-9,
          f"full bank should spend 2x proportional: {_hi(clock)} vs "
          f"{2.0 * (1800.0 / h0)}")

    # frac 0.25 (remaining 675 of an above-reserve bank of 1500) -> 1.5x.
    clock.debit(1800.0 - 675.0)
    check(clock.remaining == 675.0, f"debit bookkeeping: {clock.remaining}")
    frac = (675.0 - reserve) / (1800.0 - reserve)
    check(abs(frac - 0.25) < 1e-12, f"test setup: frac {frac}")
    check(abs(_hi(clock) - 1.5 * (675.0 / h0)) < 1e-9,
          f"quarter bank should spend 1.5x proportional: {_hi(clock)}")

    # Exactly at the reserve: the boost is gone, spend is plain proportional.
    clock.debit(675.0 - reserve)
    check(abs(_hi(clock) - reserve / h0) < 1e-9,
          f"at the reserve the spend should be 1x: {_hi(clock)}")

    # Inside the reserve: bit-identical to the legacy formula.
    clock.debit(100.0)
    check(clock.remaining == 200.0, f"debit bookkeeping: {clock.remaining}")
    got = clock.allocate(turn=0, game_number=1, self_wins=0, opp_wins=0,
                         is_sideboard=False, priors=None, num_choices=5)
    check(got == legacy_allocate(clock, turn=0, game_number=1, self_wins=0,
                                 opp_wins=0, is_sideboard=False, priors=None,
                                 num_choices=5),
          f"inside the reserve should be legacy-identical: {got}")

    # A bank that never exceeds the reserve never boosts, at any fill level.
    for bank_s in (300.0, 250.0, 60.0):
        small = MatchClock(bank_s)
        for spent in (0.0, bank_s / 3.0, bank_s / 2.0):
            small.debit(spent)
            got = small.allocate(turn=3, game_number=1, self_wins=0, opp_wins=0,
                                 is_sideboard=False, priors=None, num_choices=5)
            check(got == legacy_allocate(small, turn=3, game_number=1,
                                         self_wins=0, opp_wins=0,
                                         is_sideboard=False, priors=None,
                                         num_choices=5),
                  f"bank {bank_s} at {small.remaining}s should not boost: {got}")

    # Monotone: the boost factor decays with the bank, never below 1x nor
    # above 2x, and the sqrt makes it concave (still >1.5x at half a bank).
    ref = MatchClock(1800.0)
    factors = []
    for rem in (1800.0, 1500.0, 1200.0, 900.0, 600.0, 400.0, 350.0, 300.0):
        ref.remaining = rem
        factors.append(_hi(ref) / (rem / h0))
    check(all(1.0 - 1e-12 <= f <= 2.0 + 1e-12 for f in factors),
          f"boost factor out of [1, 2]: {factors}")
    check(all(a >= b - 1e-12 for a, b in zip(factors, factors[1:])),
          f"boost factor should decay monotonically: {factors}")
    ref.remaining = 1050.0  # half of the above-reserve bank
    check(_hi(ref) / (1050.0 / h0) > 1.5,
          "sqrt curve should still be >1.5x at half the above-reserve bank")

    # The rails still clamp AFTER the boost: t_cap, then remaining/4, then the
    # t_min Fischer floor.
    capped = MatchClock(1800.0)
    check(abs(_hi(capped, turn=30, game_number=3, self_wins=1, opp_wins=1)
              - capped.t_max) < 1e-9,
          "a boosted late-game allocation should clamp to t_max")
    check(abs(_hi(capped, turn=0, game_number=2, self_wins=1, opp_wins=0,
                 is_sideboard=True, num_choices=20) - capped.sb_t_max) < 1e-9,
          "a boosted sideboard allocation should clamp to sb_t_max")
    wide = MatchClock(1800.0, t_max=1.0e6, sb_t_max=1.0e6)
    uniform8 = np.full(8, 1.0 / 8.0)
    check(abs(_hi(wide, turn=30, game_number=3, self_wins=1, opp_wins=1,
                  priors=uniform8, num_choices=8) - 1800.0 / 4.0) < 1e-9,
          "the remaining/4 rail should clamp a boosted, boosted-difficulty root")
    drained = MatchClock(1800.0)
    drained.debit(1800.0)
    check(drained.allocate(turn=5, game_number=1, self_wins=0, opp_wins=0,
                           is_sideboard=False, priors=None, num_choices=5)
          == (drained.t_min, drained.t_min),
          "an empty bank should still grant the Fischer floor")

    if not FAILURES:
        print("PASS: front-loaded pacing curve and reserve")


def test_closeout() -> None:
    # Wide caps so the horizon effect is visible instead of being clamped.
    clock = MatchClock(1800.0, t_max=1.0e6, sb_t_max=1.0e6)
    base_state = dict(turn=5, game_number=2, self_wins=1, opp_wins=0)
    none_hi = _hi(clock, **base_state)

    # Fires ONLY at self_wins == 1 with value >= _CLOSEOUT_V. (Negatives stay
    # above _GIVEUP_V so this isolates the closeout from the giveup cap, which
    # test_giveup covers separately.)
    for v in (-0.7, 0.0, 0.5, 0.79, 0.8, 0.9, 1.0):
        check(_hi(clock, turn=5, game_number=1, self_wins=0, opp_wins=0,
                  value=v)
              == _hi(clock, turn=5, game_number=1, self_wins=0, opp_wins=0),
              f"0 wins with value {v} must be value-free")
        check(_hi(clock, turn=5, game_number=3, self_wins=0, opp_wins=1,
                  value=v)
              == _hi(clock, turn=5, game_number=3, self_wins=0, opp_wins=1),
              f"opponent at match point with value {v} must be value-free")
        check(_hi(clock, turn=5, game_number=3, self_wins=2, opp_wins=0,
                  value=v)
              == _hi(clock, turn=5, game_number=3, self_wins=2, opp_wins=0),
              f"2 wins with value {v} must be value-free (guard)")
    for v in (-0.7, 0.0, 0.5, 0.7999):
        check(_hi(clock, value=v, **base_state) == none_hi,
              f"match point with a sub-threshold value {v} must be value-free")

    # At match point with a decisive value the horizon shrinks -> bigger slice,
    # monotonically in the value.
    hi_80 = _hi(clock, value=0.8, **base_state)
    hi_90 = _hi(clock, value=0.9, **base_state)
    hi_100 = _hi(clock, value=1.0, **base_state)
    check(hi_90 > none_hi,
          f"closeout should allocate MORE per decision: {hi_90} vs {none_hi}")
    check(hi_100 >= hi_90 >= hi_80 > none_hi,
          f"closeout should be monotone in value: {hi_80}, {hi_90}, {hi_100}")

    # The 1-0 future-game budget is dropped entirely and the in-game share is
    # scaled — hand-check the horizon against the documented ramp.
    scale = 1.0 - (1.0 - clock._CLOSEOUT_MIN_SCALE) * (0.9 - 0.8) / 0.2
    check(abs(scale - 0.675) < 1e-12, f"test setup: scale {scale}")
    h_none = clock._horizon(5, 2, 1, 0)
    h_80 = clock._horizon(5, 2, 1, 0, 0.8)
    h_90 = clock._horizon(5, 2, 1, 0, 0.9)
    check(abs(h_90 - h_80 * scale) < 1e-9,
          f"closeout scale wrong: {h_90} vs {h_80 * scale}")
    check(h_80 < h_none,
          f"closeout should drop the future-game budget: {h_80} vs {h_none}")
    check(abs(clock._horizon(5, 2, 1, 0, 1.0)
              - h_80 * clock._CLOSEOUT_MIN_SCALE) < 1e-9,
          "value 1.0 should scale by _CLOSEOUT_MIN_SCALE")

    # A 1-1 game 3 has no future-game budget to drop, so only the scale moves.
    h3_none = clock._horizon(5, 3, 1, 1)
    check(abs(clock._horizon(5, 3, 1, 1, 0.9) - h3_none * scale) < 1e-9,
          "game 3 closeout should be the in-game scale alone")

    # _HORIZON_LO still clamps last: deep in a closed-out game the scaled
    # horizon collapses below 8 and is floored there.
    check(clock._horizon(30, 3, 1, 1, 1.0) == clock._HORIZON_LO,
          f"_HORIZON_LO clamp broken: {clock._horizon(30, 3, 1, 1, 1.0)}")

    # Value is inert on the pacing side: an inside-reserve closeout still just
    # spends its (shorter-horizon) proportional share, no boost.
    inside = MatchClock(1800.0, t_max=1.0e6)
    inside.debit(1800.0 - 200.0)
    hi_in = _hi(inside, value=0.9, **base_state)
    check(abs(hi_in - 200.0 / inside._horizon(5, 2, 1, 0, 0.9)) < 1e-9,
          f"closeout inside the reserve should not boost: {hi_in}")

    if not FAILURES:
        print("PASS: match-point value closeout")


def test_giveup() -> None:
    # Wide caps so the giveup cap is the binding constraint, not t_max.
    clock = MatchClock(1800.0, t_max=1.0e6, sb_t_max=1.0e6)
    base = dict(turn=5, game_number=2, self_wins=0, opp_wins=1)
    none_hi = _hi(clock, **base)
    check(none_hi > clock._GIVEUP_MAX_S,
          f"test setup: uncapped deadline {none_hi} must exceed the cap")

    # Fires at/below _GIVEUP_V regardless of match score, in-game only.
    for v in (-1.0, -0.9, -0.75):
        for state in (dict(turn=5, game_number=1, self_wins=0, opp_wins=0),
                      dict(turn=5, game_number=2, self_wins=1, opp_wins=0),
                      dict(turn=5, game_number=3, self_wins=1, opp_wins=1)):
            check(_hi(clock, value=v, **state) == clock._GIVEUP_MAX_S,
                  f"giveup should cap at {clock._GIVEUP_MAX_S} for value {v}")

    # Above the threshold and with no value, unchanged.
    for v in (-0.7499, -0.5, 0.0, 0.5, None):
        check(_hi(clock, value=v, **base) == none_hi,
              f"value {v} above the giveup threshold must not cap")

    # Mutually exclusive with the match-point closeout: a decisively-WON game
    # on match point takes the closeout path (bigger slice), never the giveup
    # cap. A decisively-LOST game is never a closeout, so it always caps.
    won_mp = dict(turn=5, game_number=2, self_wins=1, opp_wins=0)
    check(_hi(clock, value=0.9, **won_mp) > clock._GIVEUP_MAX_S,
          "a won match-point game must not be giveup-capped")

    # Never at a sideboard root — that budget is for the NEXT game, not a
    # doomed current one.
    sb = dict(turn=0, game_number=2, self_wins=0, opp_wins=1, is_sideboard=True)
    check(_hi(clock, value=-1.0, **sb) == _hi(clock, **sb),
          "giveup must not cap a sideboard root")

    # The Fischer floor still wins on an exhausted bank (cap is a ceiling only).
    empty = MatchClock(1800.0, t_min=5.0, t_max=1.0e6)
    empty.debit(1800.0)
    check(_hi(empty, value=-1.0, **base) == empty.t_min,
          "giveup cap must not drop below the t_min Fischer floor")

    if not FAILURES:
        print("PASS: giveup deadline cap")


def test_legacy_regression_grid() -> None:
    """Without a value and inside the reserve, allocation must be
    bit-identical to the pre-change formula across a grid of match states."""
    peaked = np.array([0.9, 0.05, 0.03, 0.02])
    uniform4 = np.full(4, 0.25)
    menus = ((None, 5), (peaked, 4), (uniform4, 4))
    states = (
        (0, 0, 0, 0, False),    # bo1, turn 0
        (3, 1, 0, 0, False),    # game 1, 0-0
        (12, 1, 0, 0, False),
        (30, 1, 0, 0, False),   # past _TURN_END_EST
        (0, 2, 1, 0, True),     # sideboard before game 2, on match point
        (0, 2, 0, 1, True),     # sideboard, down a game
        (7, 2, 1, 0, False),
        (7, 2, 0, 1, False),
        (18, 3, 1, 1, False),   # decider
        (30, 3, 1, 1, False),
    )
    checked = 0
    for bank_s, spent, paces in ((300.0, 0.0, ()),
                                 (300.0, 120.0, ((1, 40),)),
                                 (240.0, 90.0, ((1, 60), (2, 55))),
                                 (1800.0, 1600.0, ((1, 30), (2, 30))),
                                 (1800.0, 1800.0, ((1, 90),))):
        clock = MatchClock(bank_s)
        for game, n in paces:
            for _ in range(n):
                clock.note_decision(game)
        clock.debit(spent)
        check(clock.remaining <= clock._RESERVE_S,
              f"grid setup: remaining {clock.remaining} must be in the reserve")
        for turn, game_number, self_wins, opp_wins, is_sb in states:
            for priors, num_choices in menus:
                call = dict(turn=turn, game_number=game_number,
                            self_wins=self_wins, opp_wins=opp_wins,
                            is_sideboard=is_sb, priors=priors,
                            num_choices=num_choices)
                got = clock.allocate(**call)
                want = legacy_allocate(clock, **call)
                check(got == want,
                      f"regression: bank {bank_s} rem {clock.remaining} "
                      f"state {call} -> {got}, legacy {want}")
                checked += 1
    check(checked == 150, f"grid should cover 150 states, covered {checked}")

    if not FAILURES:
        print(f"PASS: legacy regression grid ({checked} states)")


def test_paced_floor() -> None:
    # No bound env -> every choose() takes the fallback path (evaluator argmax).
    paced = SearchController(StubEvaluator(), paced=True, rng_seed=1,
                             clock=1500.0)
    obs = np.zeros(8192, dtype=np.float32)  # fallback path never indexes deep
    t0 = time.monotonic()
    a = paced.choose(obs, 3)
    dt = time.monotonic() - t0
    check(0 <= a < 3, f"paced choose returned invalid action {a}")
    check(paced._PACE_FLOOR_S <= dt <= 0.5,
          f"paced fallback took {dt:.3f}s, expected the "
          f"~{paced._PACE_FLOOR_S}-{paced._PACE_FLOOR_S + paced._PACE_JITTER_S}s "
          f"jittered floor")
    # The pad must not be debited from the bank (fallback thinks in ~ms).
    spent = paced.stats["clock_bank"] - paced.stats["clock_remaining"]
    check(spent < 0.1, f"paced pad leaked into the clock bank: {spent:.3f}s")

    unpaced = SearchController(StubEvaluator(), rng_seed=1)
    t0 = time.monotonic()
    unpaced.choose(obs, 3)
    dt = time.monotonic() - t0
    check(dt < 0.01, f"unpaced fallback took {dt:.3f}s, expected instant")

    # bind_env resets the bank for the next match.
    paced._clock.debit(100.0)
    paced.bind_env(None)
    check(paced.stats["clock_remaining"] == paced.stats["clock_bank"],
          "bind_env should reset the clock bank")

    if not FAILURES:
        print("PASS: paced-response floor and clock debit")


def test_pace_idle() -> None:
    # pace_idle returns a duration for the DRIVER to hold (it never sleeps
    # itself): usually 0, occasionally a fake-think beat in the idle range.
    paced = SearchController(StubEvaluator(), paced=True, rng_seed=1)
    beats = [paced.pace_idle() for _ in range(400)]
    holds = [b for b in beats if b > 0.0]
    frac = len(holds) / len(beats)
    check(0.1 <= frac <= 0.45,
          f"pace_idle held {frac:.2f} of calls, expected ~{paced._PACE_IDLE_P}")
    check(all(paced._PACE_IDLE_MIN_S <= b <= paced._PACE_IDLE_MAX_S
              for b in holds),
          f"pace_idle beat outside [{paced._PACE_IDLE_MIN_S}, "
          f"{paced._PACE_IDLE_MAX_S}]: {holds[:5]}")

    # Unpaced controllers never ask for a hold.
    unpaced = SearchController(StubEvaluator(), rng_seed=1)
    check(all(unpaced.pace_idle() == 0.0 for _ in range(100)),
          "unpaced pace_idle should always return 0.0")

    if not FAILURES:
        print("PASS: pace_idle fake-think beats")


def main() -> int:
    # --math-only: just the deterministic pure-arithmetic legs (what the
    # ci_check `matchclock` tier gates on). The two timing legs below assert
    # real wall-clock sleeps and a sampled probability, which is why the full
    # run stays out of default CI (see the module docstring's flakiness note).
    math_only = "--math-only" in sys.argv[1:]
    test_allocation_math()
    test_pacing_curve()
    test_closeout()
    test_giveup()
    test_legacy_regression_grid()
    if not math_only:
        test_paced_floor()
        test_pace_idle()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s)", file=sys.stderr)
        return 1
    print("match clock: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
