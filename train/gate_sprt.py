#!/usr/bin/env python3
"""Sequential probability ratio test (SPRT) for the AZ promotion gate.

The gate asks one question — is the CANDIDATE net better than the incumbent? —
and answers it from as much evidence as that question needs and no more. The
gate plays balanced ROUNDS of its matchup panel and asks for a verdict after
each; the test stops as soon as the evidence is decisive in EITHER direction and
keeps buying rounds while it is not, up to a hard cap.

Hypotheses are SYMMETRIC around 0.5, derived from the operating bar `threshold`
(`--promote-threshold`):

    H1 (promote): p = threshold          e.g. 0.55
    H0 (keep):    p = 1 - threshold      e.g. 0.45

so neither hypothesis is privileged and a candidate of genuinely equal strength
is a coin flip rather than a near-certain rejection. The log-likelihood ratio
for a Bernoulli score, with draws counted as half a win and half a loss, is

    llr = (w + d/2) * ln(p1/p0) + (l + d/2) * ln((1-p1)/(1-p0))

which under symmetric hypotheses makes a draw contribute exactly 0. Wald's
bounds for error rates (alpha, beta) are

    upper = ln((1 - beta) / alpha)       llr >= upper  -> ACCEPT (promote)
    lower = ln(beta / (1 - alpha))       llr <= lower  -> REJECT (keep)

At the cap the test is undecided by construction. The tie-break keeps the
incumbent unless the candidate's score has reached the promote bar: a score
inside the indifference region (between H0 and H1) is reported as UNDECIDED —
kept, but not counted as a failed gate by the league — while a score at or
below H0 is a plain rejection.

The verdict is checked after every completed match, not only at round
boundaries, so a decisive candidate stops as soon as the evidence is in.

Stdlib-only (no numpy/torch/engine) so it can be imported and unit-tested
anywhere. Regression: train/test_gate_sprt.py (ci_check tier `gatesprt`).
"""

import math

# Verdicts returned by :func:`sprt_verdict`.
VERDICT_ACCEPT = "accept"       # promote the candidate
VERDICT_REJECT = "reject"       # keep the incumbent
VERDICT_CONTINUE = "continue"   # undecided — play another round

# Symmetric error rates. 0.15 is the operating point for a gate whose cap is a
# few hundred matches: against 0.1 it cuts the expected sample by about a
# fifth at every candidate strength (simulated: an equal candidate ~150 -> ~100
# matches with per-match checking), promoting a true-0.45 candidate 8% instead
# of 7% of the time and detecting a true-0.55 one 82% instead of 90%.
# Tightening to 0.05 roughly doubles the expected sample size for the same
# indifference region, which the gate cannot afford at the training-budget
# sim count (see cli_spec's DEFAULT_AZ_EVAL_SIMS note — the fix for gate cost
# is more matches per verdict, never fewer sims per match).
DEFAULT_GATE_ALPHA = 0.15


def sprt_hypotheses(threshold: float) -> tuple:
    """(p0, p1) for an operating bar ``threshold``, symmetric about 0.5.

    ``threshold`` is clamped into (0.5, 1) — a bar at or below 0.5 has no
    indifference region and would make the test degenerate."""
    p1 = min(0.999, max(0.5 + 1e-6, float(threshold)))
    return 1.0 - p1, p1


def sprt_bounds(alpha: float = DEFAULT_GATE_ALPHA,
                beta: float = None) -> tuple:
    """Wald's (lower, upper) log-likelihood-ratio bounds for error rates
    ``alpha`` (accepting a bad candidate) and ``beta`` (rejecting a good one;
    defaults to ``alpha``, i.e. a symmetric test)."""
    a = min(0.49, max(1e-4, float(alpha)))
    b = a if beta is None else min(0.49, max(1e-4, float(beta)))
    return math.log(b / (1.0 - a)), math.log((1.0 - b) / a)


def sprt_llr(wins: int, losses: int, draws: int, p0: float, p1: float) -> float:
    """Log-likelihood ratio of H1(p=``p1``) over H0(p=``p0``) for a match score
    of ``wins``/``losses``/``draws``, draws split half a win and half a loss."""
    w = wins + 0.5 * draws
    l = losses + 0.5 * draws
    return w * math.log(p1 / p0) + l * math.log((1.0 - p1) / (1.0 - p0))


def sprt_verdict(wins: int, losses: int, draws: int, *,
                 threshold: float, alpha: float = DEFAULT_GATE_ALPHA,
                 beta: float = None) -> dict:
    """Verdict on the accumulated aggregate score.

    Returns ``{"verdict", "llr", "lower", "upper", "p0", "p1", "n", "score"}``
    where ``score`` is the draw-adjusted win-rate and ``verdict`` is one of
    :data:`VERDICT_ACCEPT` / :data:`VERDICT_REJECT` / :data:`VERDICT_CONTINUE`.
    A zero-match tally is always ``continue``."""
    p0, p1 = sprt_hypotheses(threshold)
    lower, upper = sprt_bounds(alpha, beta)
    n = wins + losses + draws
    llr = sprt_llr(wins, losses, draws, p0, p1) if n else 0.0
    if n == 0:
        verdict = VERDICT_CONTINUE
    elif llr >= upper:
        verdict = VERDICT_ACCEPT
    elif llr <= lower:
        verdict = VERDICT_REJECT
    else:
        verdict = VERDICT_CONTINUE
    return {"verdict": verdict, "llr": llr, "lower": lower, "upper": upper,
            "p0": p0, "p1": p1, "n": n,
            "score": ((wins + 0.5 * draws) / n) if n else 0.0}


def sprt_cap_verdict(wins: int, losses: int, draws: int, *,
                     threshold: float, alpha: float = DEFAULT_GATE_ALPHA,
                     beta: float = None) -> dict:
    """The verdict to use once the round cap is reached: the test is undecided,
    so the incumbent keeps the seat unless the candidate's draw-adjusted score
    has reached the promote bar (H1). A score strictly inside the indifference
    region — above H0, below H1 — is a REJECT flagged ``undecided``: the
    candidate is not demonstrably worse, only not demonstrably better, and the
    league does not count it as a failed gate. A score at or below H0 is a
    plain rejection. The returned dict is :func:`sprt_verdict`'s with
    ``verdict`` overwritten and ``capped`` / ``undecided`` set."""
    r = sprt_verdict(wins, losses, draws, threshold=threshold, alpha=alpha,
                     beta=beta)
    p0, p1 = sprt_hypotheses(threshold)
    score = r["score"]
    if score >= p1 - 1e-12:
        r["verdict"] = VERDICT_ACCEPT
        r["undecided"] = False
    else:
        r["verdict"] = VERDICT_REJECT
        r["undecided"] = score > p0 + 1e-12
    r["capped"] = True
    return r


def floor_locked(cand_wins: int, cand_n: int, inc_wins: int, inc_n: int,
                 cand_left: int, inc_left: int, gate_floor: float) -> bool:
    """Is a per-deck floor veto already beyond rescue? The veto compares the
    candidate's win-rate piloting a deck against the incumbent's on like
    pairings and fires on a deficit below ``2*gate_floor - 1``. It is LOCKED
    when it fires now and would still fire in the candidate's best case — the
    candidate winning every one of its ``cand_left`` remaining matches on the
    deck and the incumbent losing every one of its ``inc_left`` — so playing
    the remaining rounds cannot change the outcome. The gate stops there
    instead of buying rounds a veto will overturn. ``cand_n``/``inc_n`` of
    zero, or ``gate_floor`` of zero, never lock."""
    if gate_floor <= 0 or cand_n <= 0 or inc_n <= 0:
        return False
    bar = 2.0 * gate_floor - 1.0
    now = cand_wins / cand_n - inc_wins / inc_n
    if not now < bar:
        return False
    best_cand = (cand_wins + max(0, cand_left)) / (cand_n + max(0, cand_left))
    worst_inc = inc_wins / (inc_n + max(0, inc_left))
    return best_cand - worst_inc < bar


def sprt_cap_line(threshold: float) -> str:
    """One line describing the cap tie-break for the gate's header."""
    p0, p1 = sprt_hypotheses(threshold)
    return (f"at the round cap: promote iff score >= {p1:.3f}, undecided "
            f"(kept, not a failed gate) if score in ({p0:.3f}, {p1:.3f})")


def sprt_expected_matches(p: float, *, threshold: float,
                          alpha: float = DEFAULT_GATE_ALPHA,
                          beta: float = None) -> float:
    """Wald's approximate expected number of matches before the test stops when
    the candidate's true win-rate is ``p``. Used only for the gate's printed
    plan line, so callers can see what a round budget buys.

    Uses the textbook E[N] = E[llr at stop] / drift with the stopping side
    priced by the error rates: a candidate at or above H1 stops at the upper
    bound with probability 1-beta, one at or below H0 stops there with
    probability alpha, and one in between is interpolated. Returns ``inf`` at
    the driftless point where the walk has no expected direction — the case the
    round cap exists for."""
    p0, p1 = sprt_hypotheses(threshold)
    lower, upper = sprt_bounds(alpha, beta)
    a = min(0.49, max(1e-4, float(alpha)))
    b = a if beta is None else min(0.49, max(1e-4, float(beta)))
    step_w, step_l = math.log(p1 / p0), math.log((1.0 - p1) / (1.0 - p0))
    drift = p * step_w + (1.0 - p) * step_l
    if abs(drift) < 1e-12:
        return float("inf")
    if p1 - p0 <= 0:
        return float("inf")
    t = min(1.0, max(0.0, (p - p0) / (p1 - p0)))   # 0 at H0, 1 at H1
    pa = a + t * ((1.0 - b) - a)                   # P(stop at the ACCEPT bound)
    return (pa * upper + (1.0 - pa) * lower) / drift


def sprt_plan_line(threshold: float, alpha: float = DEFAULT_GATE_ALPHA,
                   beta: float = None) -> str:
    """One human-readable line describing the configured test, for the gate's
    header: the hypotheses, the bounds, and the expected match counts at a
    decisive and a marginal candidate strength."""
    p0, p1 = sprt_hypotheses(threshold)
    lower, upper = sprt_bounds(alpha, beta)
    strong = sprt_expected_matches(p1 + 0.1, threshold=threshold, alpha=alpha,
                                   beta=beta)
    edge = sprt_expected_matches(p1, threshold=threshold, alpha=alpha,
                                 beta=beta)
    return (f"SPRT H0 p={p0:.3f} vs H1 p={p1:.3f}, bounds [{lower:+.2f}, "
            f"{upper:+.2f}] (alpha=beta={alpha:.2f}); expected ~{edge:.0f} "
            f"matches at p={p1:.2f}, ~{strong:.0f} at p={p1 + 0.1:.2f}")
