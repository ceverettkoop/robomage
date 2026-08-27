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
  * the round cap's tie-break is the UNBIASED one (promote iff score > 0.5),
    with the promote bar playing no part there;
  * simulated gates behave: a clearly stronger candidate is accepted inside the
    first round or two, a clearly weaker one is rejected, and an EQUAL candidate
    is close to a coin flip in either direction.

Stdlib-only (no torch, no engine, no actor binary), instant. Rides ci_check's
`gatesprt` tier. Run: train/.venv/bin/python train/test_gate_sprt.py
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gate_sprt import (DEFAULT_GATE_ALPHA, VERDICT_ACCEPT, VERDICT_CONTINUE,
                       VERDICT_REJECT, sprt_bounds, sprt_cap_verdict,
                       sprt_expected_matches, sprt_hypotheses, sprt_llr,
                       sprt_plan_line, sprt_verdict)

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
    """At the cap the tie-break is score > 0.5; the promote bar plays no part."""
    # 51.8% — above even but well under the 0.55 bar — is still promoted.
    r = sprt_cap_verdict(58, 54, 0, threshold=THRESHOLD)
    if r["verdict"] != VERDICT_ACCEPT or not r["capped"]:
        return _fail(f"a 58-54 cap record was not accepted: {r}")
    if sprt_cap_verdict(54, 58, 0, threshold=THRESHOLD)["verdict"] \
            != VERDICT_REJECT:
        return _fail("a losing cap record was not rejected")
    # Dead even: no evidence either way, so the incumbent keeps the seat.
    if sprt_cap_verdict(56, 56, 0, threshold=THRESHOLD)["verdict"] \
            != VERDICT_REJECT:
        return _fail("a dead-even cap record was accepted (llr > 0?)")
    if sprt_cap_verdict(50, 50, 12, threshold=THRESHOLD)["verdict"] \
            != VERDICT_REJECT:
        return _fail("a dead-even cap record with draws was accepted")
    print("PASS [cap]: cap tie-break is the unbiased score > 0.5, bar-free")
    return 0


def _simulate(p: float, trials: int, seed: int) -> dict:
    """Run `trials` whole sequential gates against a candidate whose true match
    win-rate is `p`, exactly as az_eval drives it: rounds of ROUND matches, a
    verdict after each, the cap tie-break at MAX_ROUNDS."""
    rng = random.Random(seed)
    promoted = 0
    matches = 0
    for _ in range(trials):
        w = l = 0
        for rnd in range(1, MAX_ROUNDS + 1):
            for _ in range(ROUND):
                if rng.random() < p:
                    w += 1
                else:
                    l += 1
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
    """End-to-end behavior of the round loop against known candidate strengths."""
    strong = _simulate(0.65, 400, seed=11)
    weak = _simulate(0.35, 400, seed=12)
    equal = _simulate(0.50, 600, seed=13)
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
              test_cap_tiebreak, test_simulated_gates, test_expected_matches):
        rc = t()
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
