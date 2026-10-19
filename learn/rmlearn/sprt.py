"""The SPRT match runner (docs/rnad_training_plan.md T8, B1/B4's "an SPRT against scripted:hard"):
a policy against an opponent over the league matchups, bo3 with ``swap: sides``, through N engine
actors, with a sequential probability ratio test on the **per-game** win rate that stops the actors
as soon as it decides.

    r = run_sprt("runs/r1/ckpt/p12.pt2", "scripted", SPRTConfig(actors=8), "runs/r1/eval/sprt-p12")
    r["verdict"]                    # "H1" (p >= p1), "H0" (p <= p0) or "undecided" (the game cap)
    r["report"]["game1"], r["report"]["games23"], r["report"]["by_matchup"]   # matches.report()

    python -m rmlearn.sprt --policy runs/r1/ckpt/p12.pt2 --opponent scripted --out runs/r1/eval/sprt-p12
    python -m rmlearn.sprt --policy nn:remote:/tmp/rm/s.sock:argmax --opponent explore --out ...

**Policy and opponent** are names as ``matches.agent_side`` reads them: an export path (played as
``nn:<path>:<options>`` in process, on the CPU: ``policy_options`` / ``opponent_options``, e.g.
``argmax`` or ``temp=1:thresh=0.03:disc=16``), an ``nn:`` / ``nn:remote:<socket>[:policies=…]``
spec (a server the caller runs; the orchestrator's), ``scripted`` / ``scripted:<tier>`` (side
``first`` with the scripted teacher at mix 1), ``explore``, ``random``.

**The test** (Wald; the archive's ``gate_sprt.py``, with the plan's per-game scoring): H1 p = p1
(0.55) against H0 p = p0 (0.45) for the probability that the policy wins a game; each decisive game
adds ln(p1/p0) (a win) or ln((1−p1)/(1−p0)) (a loss) to the log-likelihood ratio; H1 is accepted at
llr ≥ ln((1−β)/α), H0 at llr ≤ ln(β/(1−α)) (α = β = 0.05: ±2.94). Draws and capped games are not
evidence for either hypothesis: they are counted, excluded from the test and from the win rate, and
listed as findings (under self-play they are bugs to inspect). Games are consumed in the order the
actors finish them; the test is checked after every game. At the game cap (``max_games``) the
verdict is ``undecided`` (with the llr and the win rate so far); ``sequential=False`` plays the cap
whatever the llr (the ladder's fixed budget) and reports the test at the end.

**Independence.** The SPRT treats games as independent Bernoulli trials, as decided (the plan scores
games, not matches), but games of one match are correlated (same decks, sideboarding against what
was seen, game 3 only at 1–1), so the test's error rates are somewhat optimistic; the report gives
the match-clustered standard error and the design effect beside the binomial one (``matches.tally``).
Two smaller biases: games that finish early (short games) are counted first, and stopping the
actors writes off their unfinished matches (a bo3 cut after game 1 or 2).

**Stopping.** When the test decides (or the cap or ``timeout_s`` is reached, or ``cancel`` is set)
every actor gets SIGTERM, stops before its next decision and exits 0; the games it finished are in
its results file and are reported (a few may arrive after the verdict: the report counts every
finished game, the test's figures are those at its decision). An actor that fails is reported with
its log's tail; the run ends when no actor is left.

Output: ``out_dir/sprt.json`` (the result dict) and ``out_dir/summary.txt``; the actors' files
(``a<k>.results.jsonl``, ``a<k>.status.json``, ``a<k>.log``, ``rmlog/``) beside them.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

from . import matches as MT
from .corpus import BIN_DIR


@dataclass
class SPRTConfig:
    p0: float = 0.45
    p1: float = 0.55
    alpha: float = 0.05
    beta: float = 0.05
    max_games: int = 600            # the game cap (verdict "undecided" there)
    min_games: int = 0              # no verdict before this many decisive games
    sequential: bool = True         # False: play max_games whatever the test says
    actors: int = 8
    pairs: list = None              # deck pairs [(a, b)] (None: every league pair)
    weights: dict = field(default_factory=dict)   # {"a-b": weight}
    format: str = "bo3"
    policy_options: str = "argmax"
    opponent_options: str = "argmax"
    seed: int = 11
    knowledge: str = "expert"
    max_game_decisions: int = 5000
    timeout_s: float = 0.0          # wall-clock cap (0: none)
    poll_s: float = 0.5
    engine: str = None              # robomage_nn (default bin/release/robomage_nn)
    bin_dir: str = BIN_DIR

    def to_dict(self):
        d = asdict(self)
        d["pairs"] = [list(p) for p in self.pairs] if self.pairs else None
        return d

    @classmethod
    def from_dict(cls, d):
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if d.get("pairs"):
            d["pairs"] = [tuple(p) for p in d["pairs"]]
        return cls(**d)


class SPRT:
    """Wald's sequential test of H1 p = p1 against H0 p = p0 on Bernoulli games."""

    def __init__(self, p0=0.45, p1=0.55, alpha=0.05, beta=0.05, min_games=0):
        if not 0 < p0 < p1 < 1:
            raise ValueError("SPRT needs 0 < p0 < p1 < 1")
        self.p0, self.p1, self.alpha, self.beta, self.min_games = p0, p1, alpha, beta, min_games
        self.win_step = math.log(p1 / p0)
        self.loss_step = math.log((1 - p1) / (1 - p0))
        self.lower = math.log(beta / (1 - alpha))
        self.upper = math.log((1 - beta) / alpha)
        self.llr = 0.0
        self.wins = self.losses = 0
        self.verdict = None             # "H1" / "H0" once decided
        self.decided_at = None          # decisive games at the decision

    @property
    def n(self):
        return self.wins + self.losses

    def add(self, result):
        """Adds one game (+1 win, -1 loss; 0 is no evidence); the verdict (None while undecided).
        Games after the decision still count in wins / losses / llr."""
        if result > 0:
            self.wins += 1
            self.llr += self.win_step
        elif result < 0:
            self.losses += 1
            self.llr += self.loss_step
        if self.verdict is None and self.n >= self.min_games and result != 0:
            if self.llr >= self.upper:
                self.verdict, self.decided_at = "H1", self.n
            elif self.llr <= self.lower:
                self.verdict, self.decided_at = "H0", self.n
        return self.verdict

    def state(self):
        return {"p0": self.p0, "p1": self.p1, "alpha": self.alpha, "beta": self.beta, "lower": self.lower,
                "upper": self.upper, "llr": self.llr, "wins": self.wins, "losses": self.losses, "n": self.n,
                "verdict": self.verdict, "decided_at": self.decided_at}


def run_sprt(policy, opponent, cfg=None, out_dir="sprt", log=print, on_game=None, cancel=None):
    """Plays `policy` (side a) against `opponent` (side b) until the SPRT decides, the game cap, the
    timeout or `cancel` (a threading.Event) is set; returns the result dict (also written to
    `out_dir`/sprt.json, with `out_dir`/summary.txt). `on_game(record, sprt)` is called per game."""
    cfg = cfg or SPRTConfig()
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    sched = MT.eval_schedule(policy, opponent, out_dir, cfg.pairs, cfg.weights, cfg.format, cfg.knowledge,
                             cfg.max_game_decisions, "sprt", cfg.policy_options, cfg.opponent_options,
                             teacher_seed=cfg.seed)
    policy_spec = sched["entries"][0]["agent_a"]
    opponent_spec = sched["entries"][0]["agent_b"]
    pool = MT.ActorPool(sched, out_dir, cfg.actors, cfg.engine, cfg.bin_dir, cfg.seed)
    test = SPRT(cfg.p0, cfg.p1, cfg.alpha, cfg.beta, cfg.min_games)
    records = []
    reason = None
    t0 = time.time()
    log("sprt: %s vs %s, %d actors, %d pairs, cap %d games -> %s"
        % (policy_spec, opponent_spec, cfg.actors, len(sched["entries"]), cfg.max_games, out_dir))
    pool.start()
    try:
        while reason is None:
            time.sleep(cfg.poll_s)
            for line in pool.poll():
                rec = MT.game_record(line)
                records.append(rec)
                decided = test.verdict
                test.add(0 if rec["end"] == "cap" else rec["result"])
                if on_game:
                    on_game(rec, test)
                if decided is None and test.verdict and cfg.sequential and reason is None:
                    reason = "llr %.3f %s %.3f after %d decisive games" % (
                        test.llr, ">=" if test.verdict == "H1" else "<=",
                        test.upper if test.verdict == "H1" else test.lower, test.n)
            if reason is None and len(records) >= cfg.max_games:
                reason = "game cap %d" % cfg.max_games
            if reason is None and cfg.timeout_s and time.time() - t0 > cfg.timeout_s:
                reason = "timeout %.0f s" % cfg.timeout_s
            if reason is None and cancel is not None and cancel.is_set():
                reason = "cancelled"
            if reason is None and pool.alive() == 0:
                reason = "no actor left"
    finally:
        decisions = pool.decisions()
        pool.stop()
    for line in pool.poll():           # games that finished while the actors were stopping
        records.append(MT.game_record(line))
    secs = time.time() - t0
    verdict = test.verdict or "undecided"
    final = SPRT(cfg.p0, cfg.p1, cfg.alpha, cfg.beta)
    for r in records:
        final.add(0 if r["end"] == "cap" else r["result"])
    result = {"kind": "sprt", "policy": policy, "opponent": opponent, "policy_spec": policy_spec,
              "opponent_spec": opponent_spec, "verdict": verdict, "reason": reason, "sprt": test.state(),
              "sprt_all_games": final.state(), "report": MT.report(records), "games": len(records),
              "decisions": decisions, "seconds": round(secs, 1), "games_per_s": round(len(records) / max(secs, 1e-9), 3),
              "decisions_per_s": round(decisions / max(secs, 1e-9), 1), "failed_actors": pool.failures(),
              "config": cfg.to_dict(), "out_dir": out_dir, "finished_at": time.time()}
    MT.write_json(os.path.join(out_dir, "sprt.json"), result)
    text = format_result(result)
    with open(os.path.join(out_dir, "summary.txt"), "w") as fp:
        fp.write(text + "\n")
    log(text)
    return result


def format_result(r):
    s = r["sprt"]
    head = ["SPRT %s vs %s: %s (%s)" % (r["policy_spec"], r["opponent_spec"], r["verdict"], r["reason"]),
            "  H1 p=%.2f vs H0 p=%.2f, alpha %.2f, beta %.2f: llr %.3f in [%.3f, %.3f] over %d decisive games "
            "(%d-%d)%s" % (s["p1"], s["p0"], s["alpha"], s["beta"], s["llr"], s["lower"], s["upper"], s["n"],
                           s["wins"], s["losses"],
                           ", decided at %d" % s["decided_at"] if s["decided_at"] is not None else ""),
            "  %d games, %d decisions in %.0f s (%.2f games/s, %.0f decisions/s)%s"
            % (r["games"], r["decisions"], r["seconds"], r["games_per_s"], r["decisions_per_s"],
               ", %d actors FAILED" % len(r["failed_actors"]) if r["failed_actors"] else "")]
    for f in r["failed_actors"]:
        head.append("  actor %d exit %s: %s" % (f["actor"], f["exit"], " | ".join(f["log_tail"][-2:])))
    return "\n".join(head) + "\n" + MT.format_report(r["report"])


def add_common_args(ap):
    """The match-runner flags shared by sprt / ladder (SPRTConfig fields)."""
    ap.add_argument("--actors", type=int, default=8)
    ap.add_argument("--pairs", default="all", help="deck pairs a-b,c-d (league deck names) or all")
    ap.add_argument("--weights", default="", help="per-pair weights a-b=2,c-d=0.5")
    ap.add_argument("--format", default="bo3", choices=("bo3", "bo1"))
    ap.add_argument("--policy-options", default="argmax", help="nn: options of an export policy")
    ap.add_argument("--opponent-options", default="argmax", help="nn: options of an export opponent")
    ap.add_argument("--seed", type=int, default=11, help="the actors' run seed")
    ap.add_argument("--knowledge", default="expert")
    ap.add_argument("--max-game-decisions", type=int, default=5000)
    ap.add_argument("--timeout", type=float, default=0.0, help="wall-clock cap, seconds")
    ap.add_argument("--engine", help="robomage_nn (default bin/release/robomage_nn)")


def config_from_args(a, **extra):
    weights = {}
    for item in filter(None, a.weights.split(",")):
        k, v = item.rsplit("=", 1)
        weights[k] = float(v)
    return SPRTConfig(actors=a.actors, pairs=MT.parse_pairs(a.pairs), weights=weights, format=a.format,
                      policy_options=a.policy_options, opponent_options=a.opponent_options, seed=a.seed,
                      knowledge=a.knowledge, max_game_decisions=a.max_game_decisions, timeout_s=a.timeout,
                      engine=a.engine, **extra)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", required=True, help="an export (.pt2 / .ts.pt) or an nn: / nn:remote: spec")
    ap.add_argument("--opponent", default="scripted", help="scripted[:tier], explore, random, an export or a spec")
    ap.add_argument("--out", required=True)
    ap.add_argument("--p0", type=float, default=0.45)
    ap.add_argument("--p1", type=float, default=0.55)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--max-games", type=int, default=600)
    ap.add_argument("--min-games", type=int, default=0)
    ap.add_argument("--fixed", action="store_true", help="play --max-games whatever the test says")
    add_common_args(ap)
    a = ap.parse_args(argv)
    cfg = config_from_args(a, p0=a.p0, p1=a.p1, alpha=a.alpha, beta=a.beta, max_games=a.max_games,
                           min_games=a.min_games, sequential=not a.fixed)
    r = run_sprt(a.policy, a.opponent, cfg, a.out)
    return 0 if not r["failed_actors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
