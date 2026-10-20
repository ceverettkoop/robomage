"""Evaluation (docs/rmtrain.md §7; docs/rnad_training_plan.md T8): an evaluator registry and the
evaluation job that plays an evaluator's matches with evaluation actors.

    @evaluator("mine")
    def mine(ctx):                    # -> [Group]: named sets of schedule entries, matches, policies to load
        return [Group("vs-x", entries, matches=24)]

    job = EvalJob(run, cfg, seq, "it3", groups, meta, engine, bin_dir, log); job.start()
    while not job.poll(): ...         # True once every evaluation actor has finished
    job.result                        # {"groups": {name: per-game record}, ...}

**Kinds.** An evaluator returns ``Group``s, played by one ``EvalJob`` (a fixed number of matches,
or, for a group with ``sprt``, until its sequential test decides: ``sprt``), or one ``ExploitSpec``,
run by an ``ExploitJob`` (``exploiter``: a PPO best response trained inside the GPU process against
the current policy, ``rmlearn/exploiter.py``). ``SEPARATE`` evaluators run as their own evaluation
(the orchestrator splits a request naming several), since they stop on their own terms. After
every evaluation the orchestrator records its per-game results in the run's ladder
(``record_ladder``: ``rmlearn/ladder.py``, ``<run>/ladder/ladder.json``) and refreshes the
Bradley–Terry ratings (``status``, ``evals.jsonl``).

The evaluated policy is the inference server's policy 3: an R-NaD iteration's evaluation loads
that iteration's fixed point (``ckpt/fixed/m<k>.pt``), any other evaluation the orchestrator's
``snapshot`` of the current network when it starts. It plays as ``nn:remote:<socket>:policies=3``
with the phase's options — ``argmax`` by default in every phase but B5 (the fine-tuning
projection, ``thresh=…:disc=…``) — and, with ``eval.also_sampled``, the ``fixed`` evaluator also
plays it with the actors' sampling options (groups ``<opponent>@sampled``); every group carries its
``options``. Opponents are agents, the scripted teacher (``scripted`` = side ``first`` +
``teacher.py`` at mix 1) or other server policies the group loads (ids 10 and up: the ladder of past
R-NaD fixed points, the evaluated one excluded, and the BC policy). With ``eval.policy_eval`` local the
evaluated policy and the ladder rungs play in process instead (``nn:<export>``): each is exported
first by the live exporter (``LocalExports``; the evaluation waits for them as an ``ExportWait``),
and no server policy is loaded. Every game is scored on its own
(game 1 apart from games 2–3; ``evaluate.summarize``). An evaluation whose groups would collide (a
repeated group or entry id) is refused; one that plays no game is an error (the orchestrator logs
it), never an empty result.
"""

import glob
import json
import os
import time

from .. import evaluate as EV
from .. import matches as MT
from ..ladder import Ladder
from ..sprt import SPRT
from .actors import ActorPool
from .rundir import write_json
from .live import export_path
from .schedules import deck, league_pairs, nn_local, nn_remote

SUBJECT_PID = 3
LADDER_PID0 = 10
EXPLOITER_PIDS = (20, 21)       # the exploiter in training, its victim (the current policy, frozen)
EVALUATORS = {}
SEPARATE = {"sprt", "exploiter"}
SAMPLED = "@sampled"            # the suffix of the sampled (secondary) groups


def options_label(options):
    """How an evaluated policy's nn: options read in the status: ``argmax``, ``sampled`` (the empty
    options: temperature 1), or the options themselves."""
    if options is None:
        return "-"
    return options if options else "sampled"


def evaluator(name):
    def reg(fn):
        EVALUATORS[name] = fn
        return fn
    return reg


class Group:
    """One opponent's matches: `entries` (schedule entries, ids prefixed ``<name>|``), `matches` in
    all, `loads` [{pid, path, version}] (server policies to load first), `sprt` (a sequential test:
    {p0, p1, alpha, beta, max_games, min_games}; the group stops when it decides)."""

    def __init__(self, name, entries, matches, loads=None, sprt=None, options=None):
        self.name, self.entries, self.matches, self.loads = name, entries, int(matches), list(loads or [])
        self.sprt = sprt
        self.options = options      # the evaluated policy's nn: options in this group


class ExploitSpec:
    """An exploiter run (``ExploitJob``): the GPU process's ``exploit_start`` arguments."""

    def __init__(self, config, actors, pause_learner):
        self.config, self.actors, self.pause_learner = config, int(actors), bool(pause_learner)
        self.loads = []


class LocalExports:
    """The exports a local evaluation plays (eval.policy_eval local): ``export(name, checkpoint,
    version)`` names ``exports/eval/<name>-v<version>.<ext>`` and records it in ``needs`` (checkpoint,
    out, version) for the live exporter; `subject` is the evaluated policy's (checkpoint, version, name)."""

    def __init__(self, run, fmt, subject):
        self.dir = os.path.join(run.exports, "eval")
        self.fmt = fmt
        self.needs = []
        self.subject = self.export(subject[2], subject[0], subject[1])

    def export(self, name, checkpoint, version):
        out = export_path(self.dir, "%s-v%d" % (name, int(version)), self.fmt)
        if all(n[1] != out for n in self.needs):
            self.needs.append((checkpoint, out, int(version)))
        return out


class Context:
    """`subject_label`: the evaluated fixed point's ladder label (its rung is not an opponent);
    `sampled_options`: the options of the secondary sampled groups (None: none); `local`: a
    ``LocalExports`` (the policies play in process) or None (server policies)."""

    def __init__(self, run, cfg, subject_options, ladder, subject_label=None, sampled_options=None, local=None):
        self.run = run
        self.cfg = cfg
        self.local = local
        self.options = subject_options
        self.subject = self.subject_with(subject_options)
        self.ladder = ladder        # [{"label", "path", "version"}], oldest first
        self.subject_label = subject_label
        self.sampled_options = sampled_options

    def subject_with(self, options):
        if self.local is not None:
            return nn_local(self.local.subject, options)
        return nn_remote(self.run.rmip_sock, str(SUBJECT_PID), options)


def _pairs_entries(name, subject, agent, teacher, pairs):
    out = []
    for a, b in [p for pair in pairs for p in (tuple(pair), tuple(pair)[::-1])]:
        # Both orders: ``swap: sides`` moves a side's deck with its agent, so side a keeps deck a.
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
    variants = [("", ctx.options)] + ([(SAMPLED, ctx.sampled_options)] if ctx.sampled_options is not None else [])
    return [Group(opp + sfx, _pairs_entries(opp + sfx, ctx.subject_with(opts), *EV.opponent_side(opp), e["pairs"]),
                  e["matches"], options=opts)
            for sfx, opts in variants for opp in e["opponents"] if e["matches"] > 0]


@evaluator("ladder")
def ladder(ctx):
    """The policy against the most recent eval.ladder R-NaD fixed points and the BC policy."""
    e = ctx.cfg["eval"]
    if e["ladder_matches"] <= 0 or not ctx.ladder:
        return []
    by_label = {}
    for r in ctx.ladder:            # one rung per label (the latest), the evaluated fixed point left out
        if r["label"] != ctx.subject_label:
            by_label.pop(r["label"], None)
            by_label[r["label"]] = r
    rungs = [r for r in by_label.values() if r["label"] == "bc"][:1]
    fixed = [r for r in by_label.values() if r["label"] != "bc"]
    rungs += fixed[-e["ladder"]:] if e["ladder"] > 0 else []
    groups = []
    for j, r in enumerate(rungs):
        pid = LADDER_PID0 + j
        name = "ladder-" + r["label"]
        if ctx.local is not None:
            agent = nn_local(ctx.local.export(r["label"], r["path"], r.get("version", 0)), e["opponent_nn_options"])
            loads = []
        else:
            agent = nn_remote(ctx.run.rmip_sock, str(pid), e["opponent_nn_options"])
            loads = [{"pid": pid, "path": r["path"], "version": r.get("version", 0)}]
        groups.append(Group(name, _pairs_entries(name, ctx.subject, agent, None, e["pairs"]), e["ladder_matches"],
                            loads, options=ctx.options))
    return groups


def _league_or(cfg, pairs):
    return [tuple(p[:2]) for p in pairs] if pairs else [(a, b) for a, b, _ in league_pairs(cfg)]


@evaluator("sprt")
def sprt(ctx):
    """The policy against each of eval.sprt_opponents (default scripted:hard) over eval.sprt_pairs
    (default the league), a sequential test on the per-game win rate (H1 eval.sprt_p1 against H0
    eval.sprt_p0) that stops the group when it decides, or at eval.sprt_max_games games."""
    e = ctx.cfg["eval"]
    test = {"p0": e["sprt_p0"], "p1": e["sprt_p1"], "alpha": e["sprt_alpha"], "beta": e["sprt_beta"],
            "max_games": e["sprt_max_games"], "min_games": min(30, e["sprt_max_games"])}
    pairs = _league_or(ctx.cfg, e["sprt_pairs"])
    return [Group("sprt-" + opp, _pairs_entries("sprt-" + opp, ctx.subject, *EV.opponent_side(opp), pairs),
                  e["sprt_max_games"], sprt=test, options=ctx.options)
            for opp in e["sprt_opponents"] if e["sprt_max_games"] > 0]


@evaluator("exploiter")
def exploiter(ctx):
    """A best response trained against the current policy (rmlearn/exploiter.py) inside the GPU
    process, eval.exploiter_minutes of training, over eval.exploiter_pairs (default eval.pairs)."""
    e = ctx.cfg["eval"]
    if e["exploiter_minutes"] <= 0:
        return []
    lg = ctx.cfg["league"]
    conf = {"minutes": e["exploiter_minutes"], "baseline_games": e["exploiter_baseline_games"],
            "final_games": e["exploiter_final_games"], "pairs": [list(p) for p in
                                                                  _league_or(ctx.cfg, e["exploiter_pairs"] or e["pairs"])],
            "victim_options": e["exploiter_victim_options"], "lr": e["exploiter_lr"],
            "rollout_decisions": e["exploiter_rollout_decisions"], "format": lg["format"],
            "knowledge": lg["knowledge"], "max_game_decisions": lg["max_game_decisions"],
            "seed": ctx.cfg["run"]["seed"], "actors": e["exploiter_actors"]}
    return [ExploitSpec(conf, e["exploiter_actors"], e["exploiter_pause_learner"])]


def split_request(names):
    """Evaluator lists of the evaluations a request makes: the SEPARATE evaluators one each, the
    others together (in order of first appearance)."""
    rest = [n for n in names if n not in SEPARATE]
    return ([rest] if rest else []) + [[n] for n in names if n in SEPARATE]


def check_groups(groups):
    """ValueError when an evaluation's groups would collide: a repeated group name, entry id or
    server policy id (the engine refuses a schedule with a repeated id; one group's results would
    be scored as another's)."""
    names, ids, pids = set(), set(), set()
    for g in groups:
        if isinstance(g, ExploitSpec):
            continue
        if g.name in names:
            raise ValueError("group %s appears twice" % g.name)
        names.add(g.name)
        for e in g.entries:
            if e["id"] in ids:
                raise ValueError("schedule entry id %s appears twice" % e["id"])
            ids.add(e["id"])
        for ld in g.loads:
            if ld["pid"] in pids or ld["pid"] == SUBJECT_PID:
                raise ValueError("server policy %d is loaded twice" % ld["pid"])
            pids.add(ld["pid"])


def build(ctx, names):
    groups = []
    for n in names:
        fn = EVALUATORS.get(n)
        if fn is None:
            raise ValueError("no evaluator %r (registered: %s)" % (n, ", ".join(sorted(EVALUATORS))))
        groups += fn(ctx)
    return groups


class ExportWait:
    """A local evaluation waiting for its exports (``LocalExports.needs``, queued to the live
    exporter): ``ready()`` once every one exists, ``error`` when one failed. The orchestrator then
    starts its ``EvalJob`` (``groups``, ``meta``)."""

    def __init__(self, seq, label, groups, meta, needs):
        self.seq, self.label, self.groups, self.meta = seq, label, groups, meta
        self.outs = [out for _, out, _ in needs]
        self.error = None
        self.request = None

    def ready(self):
        return all(os.path.exists(o) for o in self.outs)

    def loads(self):
        return []

    def cancel(self):
        pass


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
        self.tests = {g.name: SPRT(g.sprt["p0"], g.sprt["p1"], g.sprt["alpha"], g.sprt["beta"],
                                   g.sprt.get("min_games", 0)) for g in groups if g.sprt}
        self.records = {name: [] for name in self.tests}
        self.reasons = {}
        self.reader = None

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
        self.reader = MT.ResultsReader(self.dir)
        self.started = time.time()
        self.log("eval %s: %d matches (%s) on %d actors in %s" % (self.label, total,
                                                                   ", ".join(g.name for g in self.groups), n,
                                                                   self.dir))

    def _sprt_tick(self):
        """Feeds the sequential tests; True when every group of the job is a test that is done."""
        caps = {g.name: g.sprt["max_games"] for g in self.groups if g.sprt}
        for line in self.reader.poll():
            name = line["entry"].split("|", 1)[0]
            if name not in self.tests:
                continue
            rec = MT.game_record(line)
            self.records[name].append(rec)
            t = self.tests[name]
            before = t.verdict
            t.add(0 if rec["end"] == "cap" else rec["result"])
            if before is None and t.verdict and name not in self.reasons:
                self.reasons[name] = "%s after %d decisive games (llr %.3f)" % (t.verdict, t.n, t.llr)
        for name, recs in self.records.items():
            if name not in self.reasons and len(recs) >= caps[name]:
                self.reasons[name] = "game cap %d" % caps[name]
        return len(self.tests) == len(self.groups) and len(self.reasons) == len(self.tests)

    def poll(self):
        if self.result is not None:
            return True
        self.crashes += self.pool.tick()
        stop = bool(self.tests) and self._sprt_tick()
        if not stop and not self.pool.finished():
            return False
        errors = sorted({(h.get("last_exit") or {}).get("reason") or "exit %s" % (h.get("last_exit") or {}).get("code")
                         for h in self.pool.health() if (h.get("last_exit") or {}).get("code") not in (0, None)})
        self.pool.stop_all(timeout=5)
        groups = EV.summarize(self.dir, subject=lambda agent: ":policies=%d" % SUBJECT_PID in str(agent))
        for g in self.groups:
            if g.name in groups:
                groups[g.name]["options"] = g.options
        if self.tests:
            self._sprt_tick()
        for name, t in self.tests.items():
            g = groups.setdefault(name, {"all": {}, "game1": {}, "games23": {}, "pairs": {}, "games": 0,
                                         "win_rate": None})
            g["sprt"] = dict(t.state(), verdict=t.verdict or "undecided",
                             reason=self.reasons.get(name, "the actors finished"))
            g["report"] = MT.report(self.records[name])
        self.result = dict(self.meta, seq=self.seq, label=self.label, dir=self.dir, groups=groups,
                           seconds=round(time.time() - self.started, 1), failed_actors=len(self.crashes),
                           actor_errors=errors[:4], draw_lines=draw_lines(self.dir))
        return True

    def cancel(self):
        if self.pool is not None:
            self.pool.stop_all(timeout=5)


def draw_lines(out_dir):
    """The results lines of an actors' directory whose game was drawn."""
    out = []
    for path in sorted(glob.glob(os.path.join(out_dir, "a*.results.jsonl"))):
        with open(path) as fp:
            for ln in fp:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if r.get("end_reason") == "draw":
                    out.append(dict(r, results_file=path))
    return out


class ExploitJob:
    """An exploiter run (``ExploitSpec``): the GPU process trains it (``exploit_start``: the current
    policy frozen as the victim at ``EXPLOITER_PIDS[1]``, the exploiter from a copy of it at
    ``EXPLOITER_PIDS[0]``; with ``pause_learner`` the learner idles meanwhile), these actors play
    exploiter against victim (until stopped), and the job ends when the GPU process reports the final
    games in (``exploit_status``) or at the budget plus a margin; ``exploit_finish`` returns the
    result (``rmlearn/exploiter.py``)."""

    def __init__(self, run, cfg, seq, label, spec, gpu, meta, engine, bin_dir, log):
        self.run, self.cfg, self.seq, self.label, self.spec, self.gpu = run, cfg, seq, label, spec, gpu
        self.meta = dict(meta)
        self.engine, self.bin_dir, self.log = engine, bin_dir, log
        self.dir = os.path.join(run.eval, "%04d-%s" % (seq, label))
        self.pool = None
        self.result = None
        self.started = None
        self.crashes = []
        self.last_status = 0.0
        self.status = {}
        self.groups = []

    def loads(self):
        return []

    def start(self):
        os.makedirs(self.dir, exist_ok=True)
        r = self.gpu.call("exploit_start", dir=self.dir, config=self.spec.config, pids=list(EXPLOITER_PIDS),
                          pause_learner=self.spec.pause_learner)
        self.meta["version"] = r.get("victim_version", self.meta.get("version"))
        sched = os.path.join(self.dir, "schedule.json")
        write_json(sched, r["schedule"])
        self.pool = ActorPool(self.engine, self.bin_dir, sched, self.dir, os.path.join(self.dir, "logs"),
                              self.cfg["run"]["seed"] * 1000 + self.seq, self.cfg["actors"], self.log)
        self.pool.set_count(self.spec.actors)
        self.started = time.time()
        self.log("exploiter %s: %.1f min against the policy at version %s on %d actors in %s"
                 % (self.label, self.spec.config["minutes"], self.meta["version"], self.spec.actors, self.dir))

    def poll(self):
        if self.result is not None:
            return True
        self.crashes += self.pool.tick()
        now = time.time()
        if now - self.last_status < 5.0:
            return False
        self.last_status = now
        self.status = self.gpu.call("exploit_status")
        c = self.spec.config
        overdue = now - self.started > 60 * c["minutes"] + 1800
        if not self.status.get("done") and not overdue:
            return False
        decisions = sum(int(a.get("decisions") or 0) for a in self.pool.health())
        self.pool.stop_all(timeout=5)
        res = self.gpu.call("exploit_finish", decisions=decisions, failed_actors=len(self.crashes))["result"]
        self.result = dict(self.meta, seq=self.seq, label=self.label, dir=self.dir, exploiter=res,
                           groups=exploit_groups(res), seconds=round(time.time() - self.started, 1),
                           failed_actors=len(self.crashes))
        return True

    def cancel(self):
        if self.pool is not None:
            self.pool.stop_all(timeout=5)
        if self.result is None and self.started is not None:
            try:
                self.gpu.call("exploit_cancel")
            except Exception as err:  # the GPU process may be gone; the run handles that
                self.log("exploiter %s: cancel: %s" % (self.label, err))


def compat_group(report):
    """A matches.report() as an evaluate.summarize() group (win_rate, games, game1, games23) with the
    full report under ``report``."""
    a = report["all"]
    return {"win_rate": a["win_rate"], "games": a["games"], "game1": {"win_rate": report["game1"]["win_rate"]},
            "games23": {"win_rate": report["games23"]["win_rate"]}, "report": report}


def exploit_groups(res):
    out = {"exploiter-baseline": compat_group(res["baseline"])}
    if res.get("final"):
        out["exploiter-final"] = compat_group(res["final"])
    return out


def _counts(g):
    """(wins, losses, draws, caps) of an evaluation group (summarize's or a report's)."""
    if "report" in g and g["report"]:
        a = g["report"]["all"]
        return a["wins"], a["losses"], a["draws"], a["caps"]
    a = g.get("all") or {}
    return a.get("win", 0), a.get("loss", 0), a.get("draw", 0), a.get("caps", 0)


def record_ladder(run, result, rungs, anchor="scripted"):
    """Records an evaluation's games in the run's ladder (``<run>/ladder``): the subject is the
    evaluated policy at its version (``v<version>``), each group's opponent a player (a ladder rung
    by its label, a fixed opponent by name; an exploiter's groups are not pairings); returns the
    ratings (anchored at `anchor` when it has played, else at the first player)."""
    lad = Ladder(os.path.join(run.path, "ladder"))
    subject = result.get("subject") or "v%s" % result.get("version")
    by_label = {r["label"]: r for r in rungs}
    if lad.get(subject) is None:
        r = by_label.get(subject)
        lad.add_fixed_point(subject, r["path"] if r else "nn:remote:policy-%d" % SUBJECT_PID, result.get("version"),
                            result.get("options") or "", {"label": result.get("label"), "phase": result.get("phase")})
    for name, g in result.get("groups", {}).items():
        if name.startswith("exploiter") or name.endswith(SAMPLED):
            continue
        opp = name[len("ladder-"):] if name.startswith("ladder-") else name[len("sprt-"):] if name.startswith(
            "sprt-") else name
        if lad.get(opp) is None:
            r = by_label.get(opp)
            lad.add_fixed_point(opp, r["path"] if r else opp, r.get("version") if r else None, "", {})
        w, lo, d, caps = _counts(g)
        lad.record_counts(subject, opp, w, lo, d, caps, {"label": result.get("label"), "group": name,
                                                          "dir": result.get("dir")})
    return lad.ratings(anchor=anchor)


def roll_back_ladder(run, iterations, steps):
    """Removes from the run's ladder record the players of learner steps a restore lost: fixed points
    ``m<k>`` past `iterations` and R-NaD snapshots ``v<version>`` (evaluated in B4–B5) past `steps`
    (their names are issued again for other weights), with their pairings. (removed ids, the
    ratings or None without a record)."""
    path = os.path.join(run.path, "ladder", "ladder.json")
    if not os.path.exists(path):
        return [], None
    lad = Ladder(os.path.dirname(path))
    phase = {f["id"]: (f.get("meta") or {}).get("phase") for f in lad.fixed_points()}
    gone = []
    for fid in set(phase) | {x for p in lad.pairings() for x in (p["a"], p["b"])}:
        if (fid[:1] == "m" and fid[1:].isdigit() and int(fid[1:]) > iterations) or \
                (fid[:1] == "v" and fid[1:].isdigit() and int(fid[1:]) > steps and phase.get(fid) in ("B4", "B5")):
            gone.append(fid)
    if gone:
        lad.remove(gone)
    return sorted(gone), lad.ratings(anchor="scripted")


def brief(result):
    """{group: "wr 0.43 over n games (…) [options]"} of an evaluation result."""
    out = {}
    if (result or {}).get("error"):
        out["error"] = result["error"]
    for name, g in (result or {}).get("groups", {}).items():
        wr = g.get("win_rate")
        out[name] = "%s over %d games (g1 %s, g2-3 %s)" % (
            "%.3f" % wr if wr is not None else "-", g.get("games", 0), _fmt(g["game1"].get("win_rate")),
            _fmt(g["games23"].get("win_rate")))
        if g.get("sprt"):
            out[name] += ", SPRT %s" % g["sprt"]["verdict"]
        if "options" in g:
            out[name] += " [%s]" % options_label(g["options"])
    return out


def _fmt(x):
    return "%.3f" % x if x is not None else "-"
