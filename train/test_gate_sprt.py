#!/usr/bin/env python3
"""Sequential promotion-gate test regression (train/gate_sprt.py).

Asserts the properties the AZ gate's verdict actually depends on:

  * the hypotheses are SYMMETRIC about 0.5, so the test carries no inherent
    incumbent bias;
  * a draw is worth exactly half a win and half a loss, and under symmetric
    hypotheses contributes nothing to the evidence;
  * the verdict is monotone in the score, mirror-symmetric under swapping the
    candidate's wins and losses, and never decides on an empty tally;
  * the bounds match Wald's formulas for the configured error rates, and a
    tighter alpha demands strictly more evidence;
  * at the round cap the incumbent keeps the seat unless the score reached the
    promote bar, and a score inside the indifference region is flagged
    UNDECIDED (kept, but not a failed gate);
  * the per-deck floor lock fires only when no remaining match could lift the
    veto, and never on a rescuable deficit;
  * simulated gates behave: with the verdict re-asked after every match, a
    clearly stronger candidate is accepted inside the first round or two, a
    clearly weaker one is rejected, an EQUAL candidate is close to a coin flip
    in either direction, and per-match checking costs fewer matches than
    per-round checking at the same error rates.

Stdlib-only (no torch, no engine, no actor binary), instant. Rides ci_check's
`gatesprt` tier. Run: train/.venv/bin/python train/test_gate_sprt.py
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gate_sprt import (DEFAULT_GATE_ALPHA, VERDICT_ACCEPT, VERDICT_CONTINUE,
                       VERDICT_REJECT, floor_locked, sprt_bounds,
                       sprt_cap_verdict, sprt_expected_matches,
                       sprt_hypotheses, sprt_llr, sprt_plan_line,
                       sprt_verdict)

THRESHOLD = 0.55
ROUND = 28          # DEFAULT_AZ_EVAL_GAMES: matches per panel round
MAX_ROUNDS = 8      # DEFAULT_AZ_GATE_MAX_ROUNDS


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _v(w, l, d=0, threshold=THRESHOLD, alpha=DEFAULT_GATE_ALPHA):
    return sprt_verdict(w, l, d, threshold=threshold, alpha=alpha)


def test_symmetry() -> int:
    """Hypotheses straddle 0.5 evenly and the verdict is mirror-symmetric."""
    p0, p1 = sprt_hypotheses(THRESHOLD)
    if abs((p0 + p1) - 1.0) > 1e-12:
        return _fail(f"hypotheses not symmetric about 0.5: {p0} / {p1}")
    if abs(p1 - THRESHOLD) > 1e-12:
        return _fail(f"H1 {p1} is not the promote threshold {THRESHOLD}")
    # Swapping wins and losses must flip the verdict and negate the evidence.
    for w, l in ((30, 10), (28, 26), (56, 40), (14, 14)):
        a, b = _v(w, l), _v(l, w)
        if abs(a["llr"] + b["llr"]) > 1e-9:
            return _fail(f"llr not antisymmetric at {w}-{l}: "
                         f"{a['llr']} vs {b['llr']}")
        flip = {VERDICT_ACCEPT: VERDICT_REJECT, VERDICT_REJECT: VERDICT_ACCEPT,
                VERDICT_CONTINUE: VERDICT_CONTINUE}
        if flip[a["verdict"]] != b["verdict"]:
            return _fail(f"verdict not mirror-symmetric at {w}-{l}: "
                         f"{a['verdict']} vs {b['verdict']}")
    # An even record is dead neutral — evidence for neither side, and in
    # particular never a rejection.
    even = _v(112, 112)
    if even["verdict"] != VERDICT_CONTINUE or abs(even["llr"]) > 1e-9:
        return _fail(f"an even 112-112 record is not neutral: {even}")
    print("PASS [symmetry]: hypotheses straddle 0.5, verdict mirror-symmetric, "
          "an even record is neutral")
    return 0


def test_draws() -> int:
    """A draw is half a win and half a loss, worth zero evidence here."""
    if abs(sprt_llr(10, 6, 4, *sprt_hypotheses(THRESHOLD))
           - sprt_llr(12, 8, 0, *sprt_hypotheses(THRESHOLD))) > 1e-12:
        return _fail("a draw is not scored as half a win + half a loss")
    if abs(_v(0, 0, 40)["llr"]) > 1e-12:
        return _fail("all-draws produced non-zero evidence")
    if abs(_v(0, 0, 40)["score"] - 0.5) > 1e-12:
        return _fail("all-draws score is not 0.5")
    print("PASS [draws]: draws score half/half and carry no evidence")
    return 0


def test_bounds() -> int:
    """Wald's bounds, and a tighter alpha demanding strictly more evidence."""
    lo, hi = sprt_bounds(0.1)
    if abs(hi - math.log(0.9 / 0.1)) > 1e-12 or abs(lo + hi) > 1e-12:
        return _fail(f"bounds are not Wald's symmetric pair: {lo}, {hi}")
    _, hi_tight = sprt_bounds(0.01)
    if not hi_tight > hi:
        return _fail("alpha=0.01 did not demand more evidence than alpha=0.1")
    # The same 20-8 record decides at alpha=0.1 but not at alpha=0.01.
    if _v(20, 8, alpha=0.1)["verdict"] != VERDICT_ACCEPT:
        return _fail("20-8 did not accept at alpha=0.1")
    if _v(20, 8, alpha=0.01)["verdict"] != VERDICT_CONTINUE:
        return _fail("20-8 accepted at alpha=0.01 (bounds ignored?)")
    if _v(0, 0)["verdict"] != VERDICT_CONTINUE:
        return _fail("an empty tally produced a verdict")
    print("PASS [bounds]: Wald bounds, tighter alpha needs more evidence, "
          "empty tally never decides")
    return 0


def test_monotone() -> int:
    """More wins never weakens the case; the verdict walks reject->accept."""
    prev = -1e9
    seen = []
    for w in range(0, 57):
        r = _v(w, 56 - w)
        if r["llr"] < prev - 1e-12:
            return _fail(f"llr decreased when a loss became a win at w={w}")
        prev = r["llr"]
        if not seen or seen[-1] != r["verdict"]:
            seen.append(r["verdict"])
    if seen != [VERDICT_REJECT, VERDICT_CONTINUE, VERDICT_ACCEPT]:
        return _fail(f"verdict did not walk reject->continue->accept: {seen}")
    print("PASS [monotone]: llr monotone in wins, verdict walks "
          "reject -> continue -> accept")
    return 0


def test_cap_tiebreak() -> int:
    """At the cap: promote iff the score reached the bar; an in-between score
    is a REJECT flagged undecided; at or below H0 a plain rejection."""
    # 51.8% — above even but under the 0.55 bar — keeps the incumbent, and is
    # flagged undecided rather than counted as a failure.
    r = sprt_cap_verdict(58, 54, 0, threshold=THRESHOLD)
    if r["verdict"] != VERDICT_REJECT or not r["capped"] or not r["undecided"]:
        return _fail(f"a 58-54 cap record was not kept as undecided: {r}")
    # 57.1% — at the bar — is promoted.
    r = sprt_cap_verdict(64, 48, 0, threshold=THRESHOLD)
    if r["verdict"] != VERDICT_ACCEPT or r["undecided"]:
        return _fail(f"a 64-48 cap record was not accepted: {r}")
    # Exactly the bar promotes; a hair under does not.
    if sprt_cap_verdict(55, 45, 0, threshold=THRESHOLD)["verdict"] \
            != VERDICT_ACCEPT:
        return _fail("a cap record exactly at the bar was not accepted")
    if sprt_cap_verdict(54, 46, 0, threshold=THRESHOLD)["verdict"] \
            != VERDICT_REJECT:
        return _fail("a cap record just under the bar was accepted")
    # 42.9% — at or below H0 — is a plain rejection, not undecided.
    r = sprt_cap_verdict(48, 64, 0, threshold=THRESHOLD)
    if r["verdict"] != VERDICT_REJECT or r["undecided"]:
        return _fail(f"a losing cap record was not plainly rejected: {r}")
    # Dead even: kept, undecided.
    r = sprt_cap_verdict(56, 56, 0, threshold=THRESHOLD)
    if r["verdict"] != VERDICT_REJECT or not r["undecided"]:
        return _fail(f"a dead-even cap record was not kept as undecided: {r}")
    r = sprt_cap_verdict(50, 50, 12, threshold=THRESHOLD)
    if r["verdict"] != VERDICT_REJECT or not r["undecided"]:
        return _fail("a dead-even cap record with draws was not undecided")
    print("PASS [cap]: cap keeps the incumbent under the bar, promotes at it, "
          "flags the indifference region undecided")
    return 0


def test_floor_lock() -> int:
    """The floor lock fires only when no remaining match could lift the veto."""
    floor = 0.2   # bar: cand - inc < -0.6
    # 0-for-8 vs 8-for-8 with 4 matches left each: best case 4/12 - 8/12 =
    # -0.33, above the bar -> NOT locked (a rescue is still possible).
    if floor_locked(0, 8, 8, 8, 4, 4, floor):
        return _fail("a rescuable 0-8 deficit was reported locked")
    # 0-for-14 vs 14-for-14 with 2 left each: best case 2/16 - 14/16 = -0.75
    # -> locked.
    if not floor_locked(0, 14, 14, 14, 2, 2, floor):
        return _fail("an unrescuable 0-14 deficit was not reported locked")
    # Nothing left to play: locked iff it fires now.
    if not floor_locked(1, 8, 7, 8, 0, 0, floor):
        return _fail("a firing veto with no matches left was not locked")
    if floor_locked(2, 8, 6, 8, 0, 0, floor):
        return _fail("a deficit exactly at the bar (-0.5 >= -0.6) locked")
    # A veto that does not fire now never locks; a disabled floor never locks.
    if floor_locked(4, 8, 4, 8, 0, 0, floor) or floor_locked(0, 8, 8, 8, 0, 0, 0.0):
        return _fail("a non-firing or disabled veto reported locked")
    if floor_locked(0, 0, 0, 0, 10, 10, floor):
        return _fail("an empty tally reported locked")
    print("PASS [floor-lock]: locks only beyond rescue, never on a live deficit")
    return 0


def _simulate(p: float, trials: int, seed: int, check_every: int = 1) -> dict:
    """Run `trials` whole sequential gates against a candidate whose true match
    win-rate is `p`, as az_eval drives it: up to MAX_ROUNDS rounds of ROUND
    matches, the verdict re-asked after every `check_every` matches (1 = the
    live gate's per-match check), the cap rule after the last round."""
    rng = random.Random(seed)
    promoted = 0
    matches = 0
    for _ in range(trials):
        w = l = 0
        r = None
        for i in range(1, MAX_ROUNDS * ROUND + 1):
            if rng.random() < p:
                w += 1
            else:
                l += 1
            if i % check_every:
                continue
            r = sprt_verdict(w, l, 0, threshold=THRESHOLD)
            if r["verdict"] != VERDICT_CONTINUE:
                break
        else:
            r = sprt_cap_verdict(w, l, 0, threshold=THRESHOLD)
        promoted += r["verdict"] == VERDICT_ACCEPT
        matches += w + l
    return {"promote_rate": promoted / trials,
            "mean_matches": matches / trials}


def test_simulated_gates() -> int:
    """End-to-end behavior of the sequential loop against known strengths."""
    strong = _simulate(0.65, 400, seed=11)
    weak = _simulate(0.35, 400, seed=12)
    equal = _simulate(0.50, 600, seed=13)
    # Per-match checking is the point: at the same error rates it must not
    # cost more matches than checking only at round boundaries.
    equal_rounds = _simulate(0.50, 600, seed=13, check_every=ROUND)
    if equal["mean_matches"] > equal_rounds["mean_matches"]:
        return _fail(f"per-match checking cost more matches "
                     f"({equal['mean_matches']:.0f}) than per-round "
                     f"({equal_rounds['mean_matches']:.0f})")
    print(f"       p=0.65 -> promote {strong['promote_rate']:.2f}, "
          f"mean {strong['mean_matches']:.0f} matches")
    print(f"       p=0.35 -> promote {weak['promote_rate']:.2f}, "
          f"mean {weak['mean_matches']:.0f} matches")
    print(f"       p=0.50 -> promote {equal['promote_rate']:.2f}, "
          f"mean {equal['mean_matches']:.0f} matches")
    if strong["promote_rate"] < 0.9:
        return _fail(f"a clearly stronger candidate (p=0.65) promoted only "
                     f"{strong['promote_rate']:.2f} of the time")
    if weak["promote_rate"] > 0.1:
        return _fail(f"a clearly weaker candidate (p=0.35) promoted "
                     f"{weak['promote_rate']:.2f} of the time")
    # A decisive candidate must settle inside the first couple of rounds — the
    # sequential test is meant to be cheap when the answer is obvious.
    if strong["mean_matches"] > 2 * ROUND:
        return _fail(f"p=0.65 cost {strong['mean_matches']:.0f} matches — more "
                     f"than the {2 * ROUND} of a two-round verdict")
    # The headline property: an EQUAL candidate is a coin flip, not a rejection.
    # A gate that leans either way here is systematically biased, and at ~50%
    # candidate strength that bias decides most gates.
    if not 0.35 <= equal["promote_rate"] <= 0.65:
        return _fail(f"an EQUAL candidate (p=0.50) promoted "
                     f"{equal['promote_rate']:.2f} of the time — the test is "
                     f"biased toward one side")
    if equal["mean_matches"] > MAX_ROUNDS * ROUND:
        return _fail("the round cap did not bound the gate's cost")
    print("PASS [simulated]: strong accepted inside two rounds, weak rejected, "
          "EQUAL is a coin flip")
    return 0


def test_expected_matches() -> int:
    """The plan line's cost estimates are finite and ordered the right way."""
    edge = sprt_expected_matches(THRESHOLD, threshold=THRESHOLD)
    strong = sprt_expected_matches(THRESHOLD + 0.1, threshold=THRESHOLD)
    if not (0 < strong < edge < float("inf")):
        return _fail(f"expected-match estimates out of order: edge={edge}, "
                     f"strong={strong}")
    if sprt_expected_matches(0.5, threshold=THRESHOLD) != float("inf"):
        return _fail("a driftless candidate did not report an unbounded "
                     "expected sample size (that is what the cap is for)")
    if "SPRT" not in sprt_plan_line(THRESHOLD):
        return _fail("plan line does not describe the test")
    print(f"PASS [estimates]: {sprt_plan_line(THRESHOLD)}")
    return 0


def main() -> int:
    for t in (test_symmetry, test_draws, test_bounds, test_monotone,
              test_cap_tiebreak, test_floor_lock, test_simulated_gates,
              test_expected_matches):
        rc = t()
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
