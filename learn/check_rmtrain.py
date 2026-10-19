#!/usr/bin/env python3
"""rmtrain's checks (learn/rmtrain, learn/rmlearn/orchestrator/; docs/rmtrain.md §9).

    learn/.venv/bin/python learn/check_rmtrain.py                 # CPU only, no engine: RMTRAIN-CHECK: ok
    learn/.venv/bin/python learn/check_rmtrain.py --smoke [--run DIR] [--keep]   # the end-to-end tiny run (GPU)

**CPU checks** (the GPU hidden; no engine): the config (defaults valid, the defaults' TOML parsing back
to the defaults, files and ``--set`` overrides, refusals naming the key, live parameters); the
control socket (round trips, errors, a dead socket); an orchestrator over a fake run directory
driven through its control socket (status, pause / resume, set with persistence, phase
hold / release / next, eval refused without a policy, stop) and its status from the files once it
is gone; the resume bookkeeping: an actor's first match after a crash (the match in progress written
off, never below its results), the results tail (offsets, a torn line, counts per schedule version,
the state round trip), the actor pool against a fake engine (a crash restarted at the next match,
crash-loop back-off, SIGTERM on stop), the replay window's restore (recent files by budget, a
checkpoint's window with its fresh episodes, older files left out), checkpoint pruning, the
phase schedules, and the T8 evaluators (sprt groups and an SPRT evaluation stopping its actors when
the test decides, the exploiter's spec, the split of a request into separate evaluations, the
ladder record and ratings).

**The smoke** (``--smoke``; needs ``bin/release/robomage_nn`` and ``rmfeat``, the learner's venv and
the GPU): a tiny run (3 actors, a d 64 / 2-layer model, B0 a few thousand decisions, B1 a few
hundred BC steps ended by ``phase next``, one short DAgger round, R-NaD with Δm 30) started detached
(``systemd-run --user`` when available), every CLI command exercised against it, everything
``kill -9``'ed in B4 and the run resumed from its checkpoint (by systemd's restart, else by
``start``; same phase, learner step not behind the checkpoint), then ``stop`` and ``start`` again,
then run to its end (B5, done); per-iteration SPRT evaluations and one exploiter (``rmtrain eval
--evaluators exploiter``) run on the way, and the ladder's ratings land in ``state.json``. Prints
per-phase timings and ``RMTRAIN-SMOKE: ok``.
"""

import os
import sys

if "--smoke" not in sys.argv:          # the CPU checks hide the GPU; the smoke's run needs it
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import tomllib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from rmlearn.orchestrator import actors as A  # noqa: E402
from rmlearn.orchestrator import config as C  # noqa: E402
from rmlearn.orchestrator import control as CT  # noqa: E402
from rmlearn.orchestrator import evaluation as EVAL  # noqa: E402
from rmlearn.orchestrator import gpu as G  # noqa: E402
from rmlearn.orchestrator import rundir as RD  # noqa: E402
from rmlearn.orchestrator import schedules as SCH  # noqa: E402

FAILURES = []


def check(cond, what):
    if not cond:
        FAILURES.append(what)
        print("FAIL: " + what, flush=True)
    return cond


def raises(fn, *needles):
    try:
        fn()
    except ValueError as err:
        return all(n in str(err) for n in needles)
    return False


# ── config ───────────────────────────────────────────────────────────────────────────────────

def check_config(tmp):
    d = C.defaults()
    check(C.validate(d) is d, "the defaults validate")
    back = tomllib.loads(C.defaults_toml())
    check(back == d, "the defaults' TOML parses back to the defaults")
    p = os.path.join(tmp, "c.toml")
    with open(p, "w") as fp:
        fp.write('[model]\nd = 64\nheads = 2\n[rnad]\nentropy_schedule_sizes = [100, 200]\n'
                 'entropy_schedule_repeats = [2, 1]\nlearning_rate = 1\n')
    cfg = C.resolve(C.load_file(p), ["actors.count=4", "eval.opponents=[\"random\"]", "b4.nn_options=temp=2",
                                     "run.gpu_lock_wait=true"])
    check(cfg["model"]["d"] == 64 and cfg["actors"]["count"] == 4, "a file and --set over the defaults")
    check(cfg["rnad"]["learning_rate"] == 1.0 and isinstance(cfg["rnad"]["learning_rate"], float),
          "an int where a float is expected becomes a float")
    check(cfg["eval"]["opponents"] == ["random"] and cfg["b4"]["nn_options"] == "temp=2", "--set lists and strings")
    check(cfg["run"]["gpu_lock_wait"] is True, "--set booleans")
    jp = os.path.join(tmp, "c.json")
    with open(jp, "w") as fp:
        json.dump({"actors": {"count": 2}}, fp)
    check(C.resolve(C.load_file(jp))["actors"]["count"] == 2, "a JSON config file")
    check(raises(lambda: C.resolve({"modle": {"d": 1}}), "[modle]"), "an unknown section is refused")
    check(raises(lambda: C.resolve(None, ["model.dd=3"]), "model.dd"), "an unknown key is refused")
    check(raises(lambda: C.resolve(None, ["model.d=abc"]), "model.d"), "a bad type is refused")
    check(raises(lambda: C.resolve(None, ["model.d=1.5"]), "model.d"), "a fractional int is refused")
    check(raises(lambda: C.resolve(None, ["league.format=bo5"]), "league.format"), "a bad choice is refused")
    check(raises(lambda: C.resolve(None, ["model.d=65"]), "model.d"), "d not a multiple of heads is refused")
    check(raises(lambda: C.resolve(None, ["rnad.entropy_schedule_repeats=[1,2]"]), "repeats"),
          "schedule repeats of the wrong length are refused")
    check(raises(lambda: C.resolve(None, ["b1.min_steps=10", "b1.max_steps=5"]), "b1"), "min > max refused")
    check(raises(lambda: C.resolve(None, ["model.d"]), "section.key=value"), "--set without = is refused")
    check(C.parse_live("rnad.eta", "0.1") == ("rnad.eta", 0.1), "parse_live of a live float")
    check(C.parse_live("rnad.entropy_schedule_sizes", "500") == ("rnad.entropy_schedule_sizes", [500]),
          "an int Δm becomes a one-size schedule")
    check(C.parse_live("actors.count", "8") == ("actors.count", 8), "parse_live of actors.count")
    check(raises(lambda: C.parse_live("model.d", "64"), "not settable"), "a non-live key is refused by set")
    check(raises(lambda: C.parse_live("rnad.eta", "abc"), "rnad.eta"), "a bad live value is refused")
    check(raises(lambda: C.parse_live("eta", "1"), "section.key"), "a key without a section is refused")
    check(C.GPU_LIVE <= C.LIVE, "every GPU live key is live")
    check(raises(lambda: C.set_key(d, "actors.count", -1), "actors.count"), "set_key validates")


# ── control socket ───────────────────────────────────────────────────────────────────────────

def check_control(tmp):
    path = os.path.join(tempfile.mkdtemp(prefix="rmt", dir="/tmp"), "c.sock")

    def handler(req):
        if req.get("cmd") == "boom":
            raise ValueError("bad")
        return {"ok": True, "echo": req}

    srv = CT.ControlServer(path, handler)
    srv.start()
    try:
        r = CT.request(path, {"cmd": "x", "n": [1, 2]})
        check(r.get("ok") and r["echo"]["n"] == [1, 2], "control: a round trip")
        r = CT.request(path, {"cmd": "boom"})
        check(not r.get("ok") and "bad" in r.get("error", ""), "control: a handler error is the reply")
        check(CT.alive(path), "control: alive")
        rs = []
        ts = [threading.Thread(target=lambda: rs.append(CT.request(path, {"cmd": "y"}))) for _ in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        check(len(rs) == 8 and all(r["ok"] for r in rs), "control: concurrent clients")
    finally:
        srv.stop()
    check(not os.path.exists(path), "control: the socket is removed on stop")
    check(not CT.alive(path), "control: a dead socket is not alive")
    try:
        CT.request(path, {"cmd": "x"}, timeout=1)
        check(False, "control: a request to a dead socket raises")
    except ConnectionError:
        pass


# ── a fake run driven through the orchestrator's control socket ────────────────────────────

def fake_run(tmp, name="fake"):
    run = RD.RunDir(os.path.join(tmp, name))
    run.make()
    cfg = C.resolve(None, ["actors.count=0", "b0.decisions=1000000", "run.bin_dir=" + tmp])
    RD.write_json(run.config, cfg)
    return run, cfg


def check_orchestrator(tmp):
    from rmlearn.orchestrator.orchestrator import Orchestrator, status_from_files
    from rmlearn.rmtrain import format_status
    run, cfg = fake_run(tmp)
    o = Orchestrator(run)
    # main() without the engine: the pieces it sets up, then a loop thread serving requests.
    o.state = o.new_state()
    o.state["stage"] = "collect"
    o.metrics = RD.JsonLines(run.metrics)
    o.evals_file = RD.JsonLines(run.evals)
    o.tail = A.ResultsTail(run.actors)
    o.counters = A.Counters()
    o.pool = A.ActorPool("/bin/false", tmp, run.schedule, run.actors, run.actor_logs, 1, cfg["actors"], o.log)
    with open(os.path.join(run.actors, "a0.results.jsonl"), "w") as fp:
        fp.write(json.dumps({"match": 0, "game": 1, "schedule": "b0", "decisions": 120, "end_reason": "win"}) + "\n")
    srv = CT.ControlServer(run.control_sock, o.handle_request)
    srv.start()
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            for line in o.tail.poll():
                o.counters.add(line)
            o.serve_requests()
            o.write_status()
            time.sleep(0.05)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    try:
        time.sleep(0.3)
        req = lambda **kw: CT.request(run.control_sock, kw)  # noqa: E731
        s = req(cmd="status")
        check(s.get("ok") and s["phase"] == "B0" and s["progress"]["decisions"] == 120,
              "fake run: status over the socket (B0 progress from the results)")
        check(req(cmd="pause")["paused"] and o.state["paused"], "fake run: pause")
        check(RD.read_json(run.state)["paused"], "fake run: pause persisted in state.json")
        check(not req(cmd="resume")["paused"] and not o.state["paused"], "fake run: resume")
        r = req(cmd="set", key="rnad.eta", value="0.15")
        check(r.get("ok") and r["value"] == 0.15 and r["old"] == 0.2, "fake run: set a live key")
        check(RD.read_json(run.config)["rnad"]["eta"] == 0.15, "fake run: set persisted in config.json")
        r = req(cmd="set", key="model.d", value="64")
        check(not r.get("ok") and "not settable" in r["error"], "fake run: set refuses a non-live key")
        r = req(cmd="set", key="actors.count", value="-3")
        check(not r.get("ok") and "actors.count" in r["error"], "fake run: set refuses an invalid value")
        r = req(cmd="phase", what="hold")
        check(r.get("ok") and r["hold"], "fake run: phase hold")
        r = req(cmd="phase", what="status")
        check(r["hold"] and r["phase"] == "B0" and r["progress"]["target"] == 1000000, "fake run: phase status")
        r = req(cmd="phase", what="release")
        check(not r["hold"], "fake run: phase release")
        r = req(cmd="phase", what="next")
        check(r.get("ok") and o.force_next, "fake run: phase next is recorded for the loop")
        o.force_next = False
        r = req(cmd="eval")
        check(not r.get("ok") and "no policy" in r["error"], "fake run: eval refused before a policy exists")
        r = req(cmd="nope")
        check(not r.get("ok") and "unknown command" in r["error"], "fake run: an unknown command")
        r = req(cmd="stop")
        check(r.get("ok") and o.stopping, "fake run: stop")
        mets = RD.read_jsonl(run.metrics)
        check([m["event"] for m in mets] == ["pause", "resume", "set", "hold", "release"],
              "fake run: the steering events in metrics.jsonl (%s)" % [m["event"] for m in mets])
    finally:
        stop.set()
        t.join()
        srv.stop()
        o.save_state()
        o.write_status(running=False)
    doc = status_from_files(run)
    check(doc["running"] is False and doc["phase"] == "B0" and doc["progress"]["decisions"] == 120,
          "status from the files of a run that is not running")
    text = format_status(doc)
    check("NOT RUNNING" in text and "phase B0" in text, "the human status of a stopped run")
    st = RD.read_json(run.state)
    check(st["results"]["offsets"].get("a0.results.jsonl", 0) > 0 and
          st["results"]["counters"]["by_version"]["b0"]["decisions"] == 120,
          "the results offsets and counts are in state.json")
    o.metrics.close()
    o.evals_file.close()


# ── resume bookkeeping ──────────────────────────────────────────────────────────────────────

FAKE_ENGINE = r'''#!/usr/bin/env python3
import json, os, signal, sys, time
a = sys.argv
def opt(k, d=None):
    return a[a.index(k) + 1] if k in a else d
out, k, start = opt("--actor-out"), int(opt("--actor-id")), int(opt("--actor-start"))
mode = open(os.path.join(out, "mode")).read().strip()
def status(state, current):
    json.dump({"state": state, "next_match": start, "current": current, "exit_reason": "fake " + state},
              open(os.path.join(out, "a%d.status.json" % k), "w"))
with open(os.path.join(out, "starts.txt"), "a") as fp:
    fp.write("%d %d\n" % (k, start))
if mode == "crash":
    status("failed", {"match": start})
    sys.exit(1)
stopped = []
signal.signal(signal.SIGTERM, lambda s, f: stopped.append(1))
status("running", {"match": start})
while not stopped:
    time.sleep(0.05)
status("stopped", {"match": start})
'''


def check_resume(tmp):
    out = os.path.join(tmp, "act")
    os.makedirs(out)
    check(A.first_match(out, 0) == 0, "first match of a new actor is 0")
    RD.write_json(os.path.join(out, "a0.status.json"), {"next_match": 7, "current": {"match": 7}})
    check(A.first_match(out, 0) == 8, "a match in progress is written off")
    RD.write_json(os.path.join(out, "a0.status.json"), {"next_match": 7, "current": None})
    check(A.first_match(out, 0) == 7, "between matches the next one")
    with open(os.path.join(out, "a0.results.jsonl"), "w") as fp:
        for m in (3, 9, 8):
            fp.write(json.dumps({"match": m}) + "\n")
    check(A.first_match(out, 0) == 10, "never below the results' last match + 1 (a lagging status file)")
    # results tail
    tdir = os.path.join(tmp, "tail")
    os.makedirs(tdir)
    p = os.path.join(tdir, "a1.results.jsonl")
    with open(p, "w") as fp:
        fp.write(json.dumps({"schedule": "b0", "decisions": 10}) + "\n" + '{"schedule": "b0", "deci')
    tail = A.ResultsTail(tdir)
    c = A.Counters()
    for ln in tail.poll():
        c.add(ln)
    check(c.of("b0") == {"games": 1, "decisions": 10}, "a torn last line is left for later")
    with open(p, "a") as fp:
        fp.write('sions": 5, "end_reason": "cap"}\n' + json.dumps({"schedule": "b2-r1", "decisions": 7}) + "\n")
    for ln in tail.poll():
        c.add(ln)
    check(c.of("b0")["decisions"] == 15 and c.of("b2-r1")["games"] == 1 and c.total["caps"] == 1,
          "the completed line and counts per schedule version")
    c2 = A.Counters(json.loads(json.dumps(c.state())))
    tail2 = A.ResultsTail(tdir, json.loads(json.dumps(tail.offsets)))
    check(not tail2.poll() and c2.of("b0") == c.of("b0"), "the tail and counters resume from the state")
    # the actor pool against a fake engine
    eng = os.path.join(tmp, "fake_engine.py")
    with open(eng, "w") as fp:
        fp.write(FAKE_ENGINE)
    os.chmod(eng, 0o755)
    pdir = os.path.join(tmp, "pool")
    os.makedirs(pdir)
    with open(os.path.join(pdir, "mode"), "w") as fp:
        fp.write("crash")
    settings = dict(C.defaults()["actors"], restart_delay_s=0.05, crash_loop_restarts=2, crash_loop_window_s=60,
                    max_backoff_s=0.4, stop_timeout_s=5)
    logs = []
    pool = A.ActorPool(eng, tmp, os.path.join(tmp, "sched.json"), pdir, os.path.join(pdir, "logs"), 1, settings,
                       logs.append)
    pool.set_count(1)
    end = time.time() + 15
    while time.time() < end and pool.actors[0].restarts < 4:
        pool.tick()
        time.sleep(0.02)
    starts = [int(ln.split()[1]) for ln in open(os.path.join(pdir, "starts.txt"))]
    check(starts[:5] == [0, 1, 2, 3, 4], "a crashed actor restarts at the next match (%s)" % starts[:5])
    delays = [float(m.split("restart in ")[1].split(" s")[0]) for m in logs if "restart in" in m]
    check(len(delays) >= 4 and delays[0] < 1 and delays[3] >= delays[1] and delays[3] > 0.05,
          "a crash loop backs off (%s)" % delays)
    with open(os.path.join(pdir, "mode"), "w") as fp:
        fp.write("run")
    end = time.time() + 15
    while time.time() < end and not (pool.actors[0].proc and pool.actors[0].proc.poll() is None):
        pool.tick()
        time.sleep(0.05)
    time.sleep(0.3)
    check(pool.running() == 1, "the restarted actor runs")
    h = pool.health()
    check(h and h[0]["restarts"] >= 4 and h[0]["alive"], "health reports restarts")
    pool.stop_all(timeout=5)
    st = RD.read_json(os.path.join(pdir, "a0.status.json"))
    check(st["state"] == "stopped" and pool.running() == 0, "stop_all sends SIGTERM (the actor stopped cleanly)")
    # checkpoint pruning
    cdir = os.path.join(tmp, "ck")
    os.makedirs(cdir)
    for i, n in enumerate(["bc-p10.pt", "bc-p20.pt", "bc-p30.pt", "bc-b1.pt", "bc-p40.pt"]):
        with open(os.path.join(cdir, n), "w") as fp:
            fp.write("x")
        open(os.path.join(cdir, n + ".opt"), "w").close()
        os.utime(os.path.join(cdir, n), (1000 + i, 1000 + i))
    G.prune(cdir, {"prefix": "bc-p", "count": 2})
    left = sorted(f for f in os.listdir(cdir) if f.endswith(".pt"))
    check(left == ["bc-b1.pt", "bc-p30.pt", "bc-p40.pt"] and not os.path.exists(os.path.join(cdir, "bc-p10.pt.opt")),
          "pruning keeps the newest periodic checkpoints and the phase checkpoints (%s)" % left)
    check_window_restore(tmp)


class _FakeWindow:
    def __init__(self):
        self.skipped = set()
        self.added = []

    def skip(self, paths):
        self.skipped.update(paths)

    def add(self, ep, fresh=True):
        self.added.append((ep.path, fresh))
        return True


class _Ep:
    def __init__(self, path):
        self.path = path


def check_window_restore(tmp):
    from rmlearn import episodes as EP
    d = os.path.join(tmp, "eps")
    os.makedirs(d)
    paths = []
    for i in range(6):
        p = os.path.join(d, "e%d.rmtraj" % i)
        with open(p, "wb") as fp:
            fp.write(b"x" * 100)
        os.utime(p, (2000 + i, 2000 + i))
        paths.append(p)
    w = _FakeWindow()
    n = G.restore_recent(w, [d], 350)
    check(n == 3 and w.skipped == set(paths[:3]), "restore_recent leaves out the oldest files past the budget")
    real = EP.read_episode
    EP.read_episode = lambda p: _Ep(p)
    try:
        w = _FakeWindow()
        state = {"files": paths[2:5], "fresh": [paths[4]], "min_mtime": 2002}
        n = G.restore_window(w, state, [d])
        check(n == 3 and w.added == [(paths[2], False), (paths[3], False), (paths[4], True)],
              "restore_window: the checkpoint's episodes, only its fresh ones fresh (%s)" % w.added)
        check(paths[0] in w.skipped and paths[1] in w.skipped and paths[5] not in w.skipped,
              "restore_window: older files left out, newer ones polled as fresh")
    finally:
        EP.read_episode = real


def check_schedules(tmp):
    run = RD.RunDir(os.path.join(tmp, "sch"))
    run.make()
    cfg = C.resolve(None, ["league.decks=[\"a\", \"b\", \"c\"]"])
    allowed = {"id", "weight", "purpose", "deck_a", "deck_b", "format", "swap", "agent_a", "agent_b", "teacher_a",
               "teacher_b", "teacher_mix", "knowledge_a", "knowledge_b", "max_game_decisions", "trajectory_seats",
               "trajectory_omniscient", "rmlog_dir", "trajectory_dir"}
    for sched in (SCH.b0(run, cfg), SCH.dagger(run, cfg, 2, 0.25), SCH.selfplay(run, cfg, "b4")):
        ok = all(set(e) <= allowed and e["weight"] >= 0 for e in sched["entries"])
        ids = [e["id"] for e in sched["entries"]]
        check(ok and len(ids) == len(set(ids)) and set(sched["defaults"]) <= allowed,
              "schedule %s: entries well formed" % sched["version"])
    b0 = SCH.b0(run, cfg)
    check(len(b0["entries"]) == 3 + 6 and abs(sum(e["weight"] for e in b0["entries"]) - 1) < 1e-9,
          "b0: every pair scripted vs scripted and every ordered pair vs explore, weights summing to 1")
    dg = SCH.dagger(run, cfg, 2, 0.25)
    check(dg["version"] == "b2-r2" and all(e["teacher_mix"] == 0.25 and "policies=0" in e["agent_a"]
                                           for e in dg["entries"]), "dagger: round version, mix, policy 0")
    sp = SCH.selfplay(run, C.set_key(cfg, "b4.nn_options", "temp=2"), "b5")
    check(all(e["agent_a"].endswith(":policies=0,1,2:temp=2") for e in sp["entries"]) and sp["version"] == "b5",
          "selfplay: policies 0,1,2 and the options")
    check(sp["defaults"]["trajectory_omniscient"] is False, "selfplay: no omniscient part without the critic")
    from rmlearn.orchestrator.orchestrator import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    o.cfg = C.resolve(None, ["b2.rounds_max=5", "b2.teacher_mix_start=0.5", "b2.teacher_mix_end=0"])
    check([o.dagger_mix(r) for r in range(1, 6)] == [0.5, 0.375, 0.25, 0.125, 0.0], "the DAgger mix schedule")
    ctx = EVAL.Context(run, cfg, "argmax", [{"label": "bc", "path": "b.pt", "version": 1}] +
                       [{"label": "m%d" % i, "path": "m%d.pt" % i, "version": 10 * i} for i in range(1, 6)])
    groups = EVAL.build(ctx, ["fixed", "ladder"])
    names = [g.name for g in groups]
    check(names == ["scripted", "explore", "random", "ladder-bc", "ladder-m3", "ladder-m4", "ladder-m5"],
          "the evaluators' groups (%s)" % names)
    lad = [g for g in groups if g.name == "ladder-m5"][0]
    check(lad.loads[0]["pid"] == 13 and ":policies=13" in lad.entries[0]["agent_b"] and
          ":policies=3:argmax" in lad.entries[0]["agent_a"], "a ladder rung's server policy")
    check(raises(lambda: EVAL.build(ctx, ["nope"]), "no evaluator"), "an unknown evaluator is refused")


def check_t8(tmp):
    """The T8 evaluators: sprt (groups, the test stopping an evaluation), exploiter (its spec and the
    GPU process's config), the request split, the ladder record and ratings."""
    from rmlearn import exploiter as X
    from rmlearn.orchestrator.orchestrator import Orchestrator
    run = RD.RunDir(os.path.join(tmp, "t8"))
    run.make()
    cfg = C.resolve(None, ["league.decks=[\"a\", \"b\", \"c\"]", "eval.sprt_max_games=40"])
    check(raises(lambda: C.validate(C.set_key(C.resolve(None, []), "eval.sprt_p0", 0.6)), "eval.sprt_p0"),
          "config: sprt_p0 >= sprt_p1 refused")
    check(EVAL.split_request(["sprt", "fixed", "exploiter", "ladder"]) == [["fixed", "ladder"], ["sprt"], ["exploiter"]],
          "the request split: sprt and exploiter on their own")
    o = Orchestrator.__new__(Orchestrator)
    o.state = {"eval": {"queue": []}, "phase": "B4"}
    o.eval_job = None
    o.log = lambda m: None
    o.queue_eval("it3", "", ["sprt", "fixed", "ladder"])
    o.queue_eval("manual9", "", ["exploiter"])
    check([(x["label"], x["evaluators"]) for x in o.state["eval"]["queue"]] ==
          [("it3", ["fixed", "ladder"]), ("it3-sprt", ["sprt"]), ("manual9", ["exploiter"])],
          "queue_eval splits the separate evaluators (%s)" % [x["label"] for x in o.state["eval"]["queue"]])
    check(o.eval_queued_kind("exploiter") and not o.eval_queued_kind("nope"), "an exploiter is queued")
    ctx = EVAL.Context(run, cfg, "", [])
    groups = EVAL.build(ctx, ["sprt"])
    g = groups[0]
    decks = sorted({(e["deck_a"], e["deck_b"]) for e in g.entries})
    check(g.name == "sprt-scripted" and g.sprt["max_games"] == 40 and len(g.entries) == 6 and
          ("league/a", "league/b") in decks and ("league/b", "league/a") in decks and
          all(e["swap"] == "sides" and e["teacher_b"] for e in g.entries),
          "sprt evaluator: every league pair in both orders vs scripted:hard")
    spec = EVAL.build(EVAL.Context(run, C.set_key(cfg, "eval.exploiter_minutes", 0.5), "", []), ["exploiter"])[0]
    xc = X.ExploiterConfig(**dict(spec.config, pairs=[tuple(p) for p in spec.config["pairs"]], device="cpu"))
    check(isinstance(spec, EVAL.ExploitSpec) and xc.minutes == 0.5 and spec.pause_learner,
          "exploiter evaluator: its config is an ExploiterConfig")
    # An SPRT evaluation stops when its test decides: 30 games, the subject losing 25.
    job = EVAL.EvalJob(run, cfg, 1, "it3-sprt", groups, {"version": 7, "label": "it3-sprt"}, "/bin/false", tmp,
                       lambda m: None)
    os.makedirs(job.dir, exist_ok=True)

    class _Pool:
        stopped = False

        def tick(self):
            return []

        def finished(self):
            return False

        def stop_all(self, timeout=None):
            _Pool.stopped = True

    job.pool = _Pool()
    job.reader = EVAL.MT.ResultsReader(job.dir)
    job.started = time.time()
    check(not job.poll(), "sprt job: undecided without games")
    with open(os.path.join(job.dir, "a0.results.jsonl"), "w") as fp:
        for i in range(30):
            side_a = "AB"[i % 2]
            res = 1 if i % 6 == 0 else -1
            fp.write(json.dumps({"actor": 0, "match": i // 2, "game": 1 + i % 2, "entry": "sprt-scripted|a-b",
                                 "side_a_seat": side_a, "first": "A", "deck_a": "league/a", "deck_b": "league/b",
                                 "agent_a": "nn:remote:s:policies=3" if side_a == "A" else "first",
                                 "agent_b": "first" if side_a == "A" else "nn:remote:s:policies=3",
                                 "result_a": res if side_a == "A" else -res, "result_b": -res if side_a == "A" else res,
                                 "end_reason": "win", "decisions": 50}) + "\n")
    check(job.poll() and _Pool.stopped, "sprt job: stops its actors once the test decides")
    sg = job.result["groups"]["sprt-scripted"]
    check(sg["sprt"]["verdict"] == "H0" and sg["report"]["all"]["wins"] + sg["report"]["all"]["losses"] == 30 and
          abs(sg["win_rate"] - 5 / 30) < 1e-9, "sprt job: verdict H0 and the per-game record (%s)" % sg["sprt"])
    ratings = EVAL.record_ladder(run, dict(job.result, version=7, options=""), [])
    check(ratings["scripted"]["elo"] == 0 and ratings["v7"]["elo"] < -100 and ratings["v7"]["games"] == 30,
          "ladder: the evaluation recorded, rated against scripted (%s)" % ratings)
    check(os.path.exists(os.path.join(run.path, "ladder", "ladder.json")), "ladder: ladder.json in the run")
    check("SPRT H0" in EVAL.brief(job.result)["sprt-scripted"], "the brief names the verdict")


# ── the smoke ───────────────────────────────────────────────────────────────────────────────

SMOKE_CONFIG = """
[run]
seed = 5
memory_max = "24G"
gpu_lock_wait = true
restart_delay_s = 5
[league]
decks = ["ur_delver", "burn", "bw_dnt"]
max_game_decisions = 3000
[actors]
count = 3
[model]
d = 64
heads = 2
layers = 2
[replay]
heldout_fraction = 0.2
[b0]
decisions = 5000
[bc]
batch_size = 32
warmup = 20
unit_steps = 10
report_every = 50
eval_every = 100
eval_limit = 3000
[b1]
min_steps = 100000
max_steps = 100000
patience = 2
[b2]
rounds_min = 1
rounds_max = 1
decisions_per_round = 3000
min_steps_per_round = 50
[rnad]
games_per_step = 2
chunk_len = 16
entropy_schedule_sizes = [30]
push_every = 10
report_every = 10
[b4]
max_learner_steps = 250
[b5]
max_learner_steps = 300
[eval]
pairs = [["ur_delver", "burn"]]
matches = 2
ladder = 2
ladder_matches = 2
actors = 2
opponents = ["scripted", "random"]
every_minutes = 0
evaluators = ["sprt", "fixed", "ladder"]
sprt_max_games = 4
exploiter_minutes = 0.3
exploiter_actors = 2
exploiter_baseline_games = 2
exploiter_final_games = 2
exploiter_rollout_decisions = 256
[checkpoint]
every_minutes = 0.5
"""


def sh(argv, timeout=600, check_ok=True):
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if check_ok and r.returncode != 0:
        print(r.stdout + r.stderr)
    return r


def smoke(args):
    rmtrain = os.path.join(HERE, "rmtrain")
    run = RD.RunDir(args.run or os.path.join(REPO_ROOT, "runs", "smoke-rmtrain"))
    if os.path.exists(run.path):
        shutil.rmtree(run.path)
    cfgp = os.path.join(tempfile.mkdtemp(), "smoke.toml")
    with open(cfgp, "w") as fp:
        fp.write(SMOKE_CONFIG)
    env_note = "systemd-run" if shutil.which("systemd-run") else "daemon"
    t0 = time.time()
    r = sh([rmtrain, "start", run.path, "--config", cfgp])
    check(r.returncode == 0 and "started as" in r.stdout, "start detaches (%s)" % env_note)
    print(r.stdout.strip(), flush=True)
    daemon = RD.read_json(os.path.join(run.path, "daemon.json")) or {}

    def status():
        try:
            return CT.request(run.control_sock, {"cmd": "status"}, 15)
        except (ConnectionError, ValueError):
            return RD.read_json(run.status) or {}

    phase_at = {}

    def wait(cond, what, timeout):
        end = time.time() + timeout
        while time.time() < end:
            s = status()
            p = s.get("phase")
            if p and p not in phase_at:
                phase_at[p] = time.time() - t0
                print("smoke: %6.1f s  phase %s" % (phase_at[p], p), flush=True)
            if cond(s):
                return s
            if not CT.alive(run.control_sock) and not cond(s):
                if what != "the run to end":
                    check(False, "the run died while waiting for %s (see %s)" % (what, run.logs))
                    return s
            time.sleep(2)
        check(False, "timed out waiting for %s" % what)
        return status()

    # B0 → B1: exercise the CLI while BC trains.
    wait(lambda s: s.get("phase") == "B1" and s.get("stage") == "train" and (s.get("progress") or {}).get("bc_step"),
         "B1", 900)
    out = sh([rmtrain, "status", run.path]).stdout
    check("phase B1" in out and "gpu:" in out, "status (human) while running")
    print(out, flush=True)
    js = json.loads(sh([rmtrain, "status", run.path, "--json"]).stdout)
    check(js.get("running") and js["gpu"].get("server") is not None, "status --json has the server stats")
    check(sh([rmtrain, "tail", run.path, "gpu", "-n", "5", "--no-follow"]).returncode == 0, "tail gpu")
    check(sh([rmtrain, "tail", run.path, "metrics", "-n", "3", "--no-follow"]).returncode == 0, "tail metrics")
    check("rnad.eta = 0.18" in sh([rmtrain, "set", run.path, "rnad.eta", "0.18"]).stdout, "set rnad.eta")
    check(sh([rmtrain, "set", run.path, "model.d", "32"], check_ok=False).returncode != 0, "set refuses model.d")
    check(sh([rmtrain, "phase", run.path, "hold"]).returncode == 0, "phase hold")
    check("[hold" in sh([rmtrain, "phase", run.path, "status"]).stdout, "phase status shows the hold")
    check(sh([rmtrain, "pause", run.path]).returncode == 0, "pause")
    time.sleep(4)
    s1 = status()
    time.sleep(4)
    s2 = status()
    check(s2.get("paused") and s1.get("progress", {}).get("bc_step") == s2.get("progress", {}).get("bc_step"),
          "pause stops the learner")
    check(sh([rmtrain, "resume", run.path]).returncode == 0, "resume")
    time.sleep(3)
    check(sh([rmtrain, "phase", run.path, "release"]).returncode == 0, "phase release")
    s = status()
    check(s.get("phase") == "B1" and s.get("stage") == "train", "B1 trains on (its step cap is out of reach)")
    check(sh([rmtrain, "phase", run.path, "next"]).returncode == 0, "phase next (ends B1)")
    # B1's end, its evaluation, DAgger, R-NaD.
    wait(lambda s: s.get("phase") == "B4" and (s.get("progress") or {}).get("learner_steps", 0) >= 20, "B4", 1800)
    check(sh([rmtrain, "eval", run.path]).returncode == 0, "eval")
    check(sh([rmtrain, "eval", run.path, "--evaluators", "exploiter"]).returncode == 0, "eval --evaluators exploiter")
    s = wait(lambda s: (s.get("progress") or {}).get("iterations", 0) >= 2 and (s.get("checkpoint") or {}).get(
        "kind") == "rnad" and (s["checkpoint"].get("learner_steps") or 0) >= 60, "two R-NaD iterations", 1200)
    ck_steps = s["checkpoint"]["learner_steps"]
    # kill -9 everything, then resume.
    t_kill = time.time()
    killed = kill9(run, daemon)
    check(killed > 0, "kill -9 of the run's processes (%d)" % killed)
    time.sleep(2)
    check(not CT.alive(run.control_sock), "the run is gone after kill -9")
    out = sh([rmtrain, "status", run.path]).stdout
    check("NOT RUNNING" in out and "phase B4" in out, "status from the files after kill -9")
    if daemon.get("kind") == "systemd":
        print("smoke: waiting for systemd to restart the run (Restart=on-failure)", flush=True)
        end = time.time() + 120
        while time.time() < end and not CT.alive(run.control_sock):
            time.sleep(1)
        check(CT.alive(run.control_sock), "systemd restarted the killed run")
    else:
        r = sh([rmtrain, "start", run.path])
        check(r.returncode == 0 and "resuming" in r.stdout, "start resumes the run")
    s = wait(lambda s: s.get("running") and (s.get("gpu") or {}).get("learner_steps") is not None, "the resume", 300)
    check(s.get("phase") == "B4" and s["gpu"]["learner_steps"] >= ck_steps,
          "resumed in B4 at the checkpoint's learner step (%s >= %s)" % (s["gpu"].get("learner_steps"), ck_steps))
    resumed_in = time.time() - t_kill
    print("smoke: resumed %.1f s after the kill at learner step %s" % (resumed_in, s["gpu"]["learner_steps"]),
          flush=True)
    check(sh([rmtrain, "set", run.path, "actors.count", "2"]).returncode == 0, "set actors.count")
    time.sleep(6)
    check((status().get("actors") or {}).get("wanted") == 2, "actors.count applied")
    # stop, then start: the CLI's resume.
    steps = (status().get("gpu") or {}).get("learner_steps", 0)
    r = sh([rmtrain, "stop", run.path], timeout=900)
    check(r.returncode == 0 and "stopped" in r.stdout and not CT.alive(run.control_sock), "stop")
    ck = (RD.read_json(run.state) or {}).get("checkpoint") or {}
    check(ck.get("kind") == "rnad" and ck.get("learner_steps", 0) >= steps, "stop checkpoints the learner")
    r = sh([rmtrain, "start", run.path])
    check(r.returncode == 0 and "resuming" in r.stdout, "start resumes the stopped run")
    s = wait(lambda s: s.get("running") and (s.get("gpu") or {}).get("learner_steps") is not None, "the resume", 300)
    check(s["gpu"]["learner_steps"] == ck.get("learner_steps") or s["gpu"]["learner_steps"] >= ck.get(
        "learner_steps", 0), "resumed at the stop's checkpoint")
    check((status().get("actors") or {}).get("wanted") == 2, "a set value survives the restart")
    wait(lambda s: s.get("phase") == "B5", "B5", 1200)
    check(sh([rmtrain, "phase", run.path, "next"]).returncode == 0, "phase next (B5 -> done)")
    wait(lambda s: not CT.alive(run.control_sock), "the run to end", 900)
    st = RD.read_json(run.state)
    check(st and st["phase"] == "done", "the run ended in phase done")
    total = time.time() - t0
    mets = RD.read_jsonl(run.metrics)
    evs = [m for m in mets if m["event"] == "eval"]
    tr = [(m["from"], m["to"], m["reason"]) for m in mets if m["event"] == "transition"]
    print("smoke: transitions %s" % tr)
    print("smoke: evaluations %s" % [(e["label"], {k: v["win_rate"] for k, v in e["results"].items()}) for e in evs])
    check(any(e["label"] == "b1" for e in evs) and any(e["label"].startswith("b2-r") for e in evs) and
          any(e["label"].startswith("it") for e in evs), "B1, DAgger and per-iteration evaluations ran")
    check(any(e["label"].endswith("-sprt") and "sprt-scripted" in e["results"] for e in evs),
          "per-iteration SPRT evaluations ran")
    check(any("exploiter-baseline" in e["results"] for e in evs), "an exploiter ran (rmtrain eval --evaluators)")
    check(bool((RD.read_json(run.state) or {}).get("eval", {}).get("ratings")), "ladder ratings in state.json")
    check(any(m["event"] == "fatal" for m in mets) is False, "no fatal event")
    exports = os.listdir(run.exports)
    check(any(f.endswith(".pt2") for f in exports), "exports written (%s)" % sorted(exports)[:6])
    print("smoke: phase start times (s): %s; total %.1f s" % (json.dumps({k: round(v, 1) for k, v in
                                                                        phase_at.items()}), total))
    if not args.keep and not FAILURES:
        shutil.rmtree(run.path)


def kill9(run, daemon):
    """SIGKILL to the orchestrator, the GPU process and every actor of the run (as a crash or a reboot)."""
    pids = set()
    st = RD.read_json(run.status) or {}
    for k in ("pid",):
        if st.get(k):
            pids.add(int(st[k]))
    if (st.get("gpu") or {}).get("pid"):
        pids.add(int(st["gpu"]["pid"]))
    for h in (st.get("actors") or {}).get("health", []):
        if h.get("pid"):
            pids.add(int(h["pid"]))
    if daemon.get("kind") == "systemd":
        subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", daemon["unit"] + ".service"])
        return len(pids) or 1
    n = 0
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGKILL)
            n += 1
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
                n += 1
            except ProcessLookupError:
                pass
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true", help="the end-to-end tiny run (GPU, engine)")
    ap.add_argument("--run", help="--smoke: the run directory (default runs/smoke-rmtrain)")
    ap.add_argument("--keep", action="store_true", help="--smoke: keep the run directory")
    args = ap.parse_args()
    t0 = time.time()
    if args.smoke:
        smoke(args)
        tag = "RMTRAIN-SMOKE"
    else:
        tmp = tempfile.mkdtemp(prefix="rmtrain-check-")
        try:
            check_config(tmp)
            check_control(tmp)
            check_orchestrator(tmp)
            check_resume(tmp)
            check_schedules(tmp)
            check_t8(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        tag = "RMTRAIN-CHECK"
    if FAILURES:
        print("%s: %d failures (%.1f s)" % (tag, len(FAILURES), time.time() - t0))
        return 1
    print("%s: ok (%.1f s)" % (tag, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
