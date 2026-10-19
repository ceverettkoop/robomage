"""Evaluation (docs/rmtrain.md §7; docs/rnad_training_plan.md T8): an evaluator registry and the
evaluation job that plays an evaluator's matches with evaluation actors.

    @evaluator("mine")
    def mine(ctx):                    # -> [Group]: named sets of schedule entries, matches, policies to load
        return [Group("vs-x", entries, matches=24)]

    job = EvalJob(run, cfg, seq, "it3", groups, meta, engine, bin_dir, log); job.start()
    while not job.poll(): ...         # True once every evaluation actor has finished
    job.result                        # {"groups": {name: per-game record}, ...}

The evaluated policy is the inference server's policy 3 (the orchestrator's ``snapshot`` of the
current network at the evaluation's start), played as ``nn:remote:<socket>:policies=3`` with the
phase's options (``argmax`` in B1–B2, sampling in B4, the B5 projection); opponents are agents, the
scripted teacher (``scripted`` = side ``first`` + ``teacher.py`` at mix 1) or other server policies
the group loads (ids 10 and up: the ladder of past R-NaD fixed points and the BC policy). Every game
is scored on its own (game 1 apart from games 2–3; ``evaluate.summarize``). The registry is the
extension point the T8 SPRT runner and exploiter will plug into.
"""

import os
import time

from .. import evaluate as EV
from .actors import ActorPool
from .rundir import write_json
from .schedules import deck, nn_remote

SUBJECT_PID = 3
LADDER_PID0 = 10
EVALUATORS = {}


def evaluator(name):
    def reg(fn):
        EVALUATORS[name] = fn
        return fn
    return reg


class Group:
    """One opponent's matches: `entries` (schedule entries, ids prefixed ``<name>|``), `matches` in
    all, `loads` [{pid, path, version}] (server policies to load first)."""

    def __init__(self, name, entries, matches, loads=None):
        self.name, self.entries, self.matches, self.loads = name, entries, int(matches), list(loads or [])


class Context:
    def __init__(self, run, cfg, subject_options, ladder):
        self.run = run
        self.cfg = cfg
        self.subject = nn_remote(run.rmip_sock, str(SUBJECT_PID), subject_options)
        self.ladder = ladder        # [{"label", "path", "version"}], oldest first


def _pairs_entries(name, subject, agent, teacher, pairs):
    out = []
    for a, b in pairs:
        e = {"id": "%s|%s-%s" % (name, a, b), "purpose": "eval", "deck_a": deck(a), "deck_b": deck(b),
             "swap": "sides", "agent_a": subject, "agent_b": agent}
        if teacher:
            e.update({"teacher_b": teacher, "teacher_mix": 1})
        out.append(e)
    return out


@evaluator("fixed")
def fixed_opponents(ctx):
    """The policy against each of eval.opponents (scripted = scripted:hard, explore, random, any agent)."""
    e = ctx.cfg["eval"]
    return [Group(opp, _pairs_entries(opp, ctx.subject, *EV.opponent_side(opp), e["pairs"]), e["matches"])
            for opp in e["opponents"] if e["matches"] > 0]


@evaluator("ladder")
def ladder(ctx):
    """The policy against the most recent eval.ladder R-NaD fixed points and the BC policy."""
    e = ctx.cfg["eval"]
    if e["ladder_matches"] <= 0 or not ctx.ladder:
        return []
    rungs = [r for r in ctx.ladder if r["label"] == "bc"][:1]
    fixed = [r for r in ctx.ladder if r["label"] != "bc"]
    rungs += fixed[-e["ladder"]:] if e["ladder"] > 0 else []
    groups = []
    for j, r in enumerate(rungs):
        pid = LADDER_PID0 + j
        agent = nn_remote(ctx.run.rmip_sock, str(pid), e["opponent_nn_options"])
        name = "ladder-" + r["label"]
        groups.append(Group(name, _pairs_entries(name, ctx.subject, agent, None, e["pairs"]), e["ladder_matches"],
                            [{"pid": pid, "path": r["path"], "version": r.get("version", 0)}]))
    return groups


def build(ctx, names):
    groups = []
    for n in names:
        fn = EVALUATORS.get(n)
        if fn is None:
            raise ValueError("no evaluator %r (registered: %s)" % (n, ", ".join(sorted(EVALUATORS))))
        groups += fn(ctx)
    return groups


class EvalJob:
    def __init__(self, run, cfg, seq, label, groups, meta, engine, bin_dir, log):
        self.run, self.cfg, self.seq, self.label = run, cfg, seq, label
        self.groups = groups
        self.meta = dict(meta)
        self.engine, self.bin_dir, self.log = engine, bin_dir, log
        self.dir = os.path.join(run.eval, "%04d-%s" % (seq, label))
        self.pool = None
        self.result = None
        self.started = None
        self.crashes = []

    def loads(self):
        return [ld for g in self.groups for ld in g.loads]

    def start(self):
        os.makedirs(self.dir, exist_ok=True)
        entries = []
        for g in self.groups:
            for e in g.entries:
                entries.append(dict(e, weight=g.matches / len(g.entries)))
        total = sum(g.matches for g in self.groups)
        sched = os.path.join(self.dir, "schedule.json")
        write_json(sched, {"version": "eval-%d" % self.seq,
                           "outputs": {"eval": {"rmlog_dir": os.path.join(self.dir, "rmlog")}},
                           "defaults": {"format": self.cfg["league"]["format"],
                                        "knowledge_a": self.cfg["league"]["knowledge"],
                                        "knowledge_b": self.cfg["league"]["knowledge"],
                                        "max_game_decisions": self.cfg["league"]["max_game_decisions"]},
                           "entries": entries})
        n = max(1, min(int(self.cfg["eval"]["actors"]), total))
        per = {k: total // n + (1 if k < total % n else 0) for k in range(n)}
        self.pool = ActorPool(self.engine, self.bin_dir, sched, self.dir, os.path.join(self.dir, "logs"),
                              self.cfg["run"]["seed"] * 1000 + self.seq, self.cfg["actors"], self.log, matches=per)
        self.pool.set_count(n)
        self.started = time.time()
        self.log("eval %s: %d matches (%s) on %d actors in %s" % (self.label, total,
                                                                   ", ".join(g.name for g in self.groups), n,
                                                                   self.dir))

    def poll(self):
        if self.result is not None:
            return True
        self.crashes += self.pool.tick()
        if not self.pool.finished():
            return False
        self.pool.stop_all(timeout=5)
        groups = EV.summarize(self.dir, subject=lambda agent: ":policies=%d" % SUBJECT_PID in str(agent))
        self.result = dict(self.meta, seq=self.seq, label=self.label, dir=self.dir, groups=groups,
                           seconds=round(time.time() - self.started, 1), failed_actors=len(self.crashes))
        return True

    def cancel(self):
        if self.pool is not None:
            self.pool.stop_all(timeout=5)


def brief(result):
    """{group: "wr 0.43 (n games)"} of an evaluation result."""
    out = {}
    for name, g in (result or {}).get("groups", {}).items():
        wr = g.get("win_rate")
        out[name] = "%s over %d games (g1 %s, g2-3 %s)" % (
            "%.3f" % wr if wr is not None else "-", g.get("games", 0), _fmt(g["game1"].get("win_rate")),
            _fmt(g["games23"].get("win_rate")))
    return out


def _fmt(x):
    return "%.3f" % x if x is not None else "-"
