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
phase schedules, the sideboarding warm start (``actors.teacher_tier`` the teacher of B0 and DAgger,
evaluations against scripted:hard, ``bc.sideboard_weight``, ``rnad.reg_eps_sideboarding`` near 0),
structured sideboard exploration (``p_explore``'s schedule, the coverage counts from results lines, the
weights, the self-play schedule's ``sb_explore`` and no other schedule's, the orchestrator's snapshot
cadence, metrics, status and state; the engine's own schedule check when a built engine knows the key),
and the T8 evaluators (sprt groups and an SPRT evaluation stopping its actors when
the test decides, the exploiter's spec, the split of a request into separate evaluations, the
ladder record and ratings), and the end-of-run benchmark (its policy set, summary, sideboarding report
and ``--opponent``).

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
import glob  # noqa: E402
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
from rmlearn import rmtrain as RT  # noqa: E402

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
    agg = type("W", (), {"agg": None})()
    for i in range(4):
        m = {"learner_steps": i + 1, "loss": 1.0 + i, "value_mse_first_sight": 0.5}
        if i % 2:
            m.update(value_mse_after=0.25 * i, value_mse_gap=0.5 - 0.25 * i)
        G.Worker._aggregate(agg, m, ("learner_steps",), ())
    out = G.Worker._flush(agg, 4)
    check(out["loss"] == 2.5 and out["value_mse_first_sight"] == 0.5 and out["value_mse_after"] == 0.5 and
          out["value_mse_gap"] == 0.0, "metrics aggregation: a mean over the steps that reported it (%s)" % out)
    doc["metrics"] = {"rnad_step": dict(out, time=time.time())}
    check("value MSE: first sight 0.5, after the update 0.5, gap 0" in format_status(doc),
          "the R-NaD value diagnostic in the human status")
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
    def started():
        try:
            return [int(ln.split()[1]) for ln in open(os.path.join(pdir, "starts.txt"))]
        except FileNotFoundError:
            return []

    # The fifth start is written by the fourth restart's process, after the pool counted it.
    end = time.time() + 15
    while time.time() < end and (pool.actors[0].restarts < 4 or len(started()) < 5):
        pool.tick()
        time.sleep(0.02)
    starts = started()
    check(starts[:5] == [0, 1, 2, 3, 4], "a crashed actor restarts at the next match (%s)" % starts[:5])
    delays = [float(m.split("restart in ")[1].split(" s")[0]) for m in logs if "restart in" in m]
    check(len(delays) >= 4 and delays[0] < 1 and delays[3] >= delays[1] and delays[3] > 0.05,
          "a crash loop backs off (%s)" % delays)
    with open(os.path.join(pdir, "mode"), "w") as fp:
        fp.write("run")
    def fake_running():
        st = RD.read_json(os.path.join(pdir, "a0.status.json")) or {}
        return st.get("state") == "running" and pool.actors[0].proc and pool.actors[0].proc.poll() is None

    # A start still in flight from the crash loop may read the old mode: wait for one that runs.
    end = time.time() + 15
    while time.time() < end and not fake_running():
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
        # The checkpoint: e2, e3 taken by the learner, e4 not yet; e0, e1 evicted earlier (still on
        # disk); e5 written after the checkpoint, before the crash (taken by the lost steps).
        w = _FakeWindow()
        state = {"files": paths[2:5], "fresh": [paths[4]], "complete": True}
        n, left = G.restore_window(w, state, [d])
        check(n == 3 and w.added == [(paths[2], False), (paths[3], False), (paths[4], True)],
              "restore_window: the checkpoint's episodes, only its fresh ones fresh (%s)" % w.added)
        check(paths[0] in w.skipped and paths[1] in w.skipped and left == 3,
              "restore_window: episodes the learner took before the checkpoint are not fresh again")
        check(paths[5] in w.skipped,
              "restore_window: episodes written after the checkpoint (the lost steps' games) are not fresh")
        later = os.path.join(d, "e9.rmtraj")
        with open(later, "wb") as fp:
            fp.write(b"x")
        os.utime(later, (time.time() + 60, time.time() + 60))
        check(later not in w.skipped, "restore_window: episodes written after the restore are polled as fresh")
    finally:
        EP.read_episode = real

    class _PollWindow:
        def __init__(self):
            self.listed = ["a"]

        def poll(self):
            self.listed.append("b")         # an episode on disk the polling thread had not seen yet

        def files(self):
            return list(self.listed), ["b"]

    st = G.window_state(_PollWindow(), [_Ep("c")])
    check(st["files"] == ["a", "b"] and st["fresh"] == ["b", "c"] and st["complete"],
          "window_state polls first: every episode on disk is in the checkpoint (%s)" % st)


def check_schedules(tmp):
    run = RD.RunDir(os.path.join(tmp, "sch"))
    run.make()
    cfg = C.resolve(None, ["league.decks=[\"a\", \"b\", \"c\"]", "actors.policy_eval=remote",
                           "eval.policy_eval=remote"])
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


def check_sideboarding_warm_start(tmp):
    """The sideboarding warm start (docs/rnad_training_plan.md §6.2): actors.teacher_tier is the
    teacher of B0's scripted seats and DAgger's labels while every evaluation opponent stays
    scripted:hard; bc.sideboard_weight weights the policy loss (1: the plain mean); rnad.reg_eps_sideboarding
    at 0 or 0.01 validates and smooths finitely."""
    import types
    import torch
    from rmlearn import bc as BC
    from rmlearn import benchmark as BM
    from rmlearn import evaluate as EV
    from rmlearn import matches as MT
    from rmlearn import model as M
    from rmlearn.rnad_learner import RegPolicy
    run = RD.RunDir(os.path.join(tmp, "sbw"))
    run.make()
    cfg = C.resolve(None, ["league.decks=[\"a\", \"b\", \"c\"]", "actors.teacher_tier=hard+sb",
                           "actors.policy_eval=remote", "eval.policy_eval=remote"])
    t = SCH.teacher(cfg)
    check(" --tier hard+sb " in t + " ", "the teacher command carries --tier hard+sb (%s)" % t)
    check(raises(lambda: C.resolve(None, ["actors.teacher_tier=hard; rm -rf x"]), "actors.teacher_tier"),
          "a teacher tier that is no tier name is refused")
    b0 = SCH.b0(run, cfg)
    ss = [e for e in b0["entries"] if e["id"].startswith("ss-")]
    se = [e for e in b0["entries"] if e["id"].startswith("se-")]
    check(ss and all(e["agent_a"] == e["agent_b"] == "first" and e["teacher_a"] == e["teacher_b"] == t and
                     e["teacher_mix"] == 1 for e in ss), "b0: both scripted seats are the hard+sb teacher")
    check(se and all(e["agent_a"] == "first" and e["teacher_a"] == t and e["agent_b"] == "explore" and
                     "teacher_b" not in e for e in se), "b0: the scripted seat vs explore is the hard+sb teacher")
    dg = SCH.dagger(run, cfg, 1, 0.5)
    check(all(e["teacher_a"] == e["teacher_b"] == t for e in dg["entries"]), "dagger: hard+sb labels both seats")
    sp = SCH.selfplay(run, cfg, "b4")
    check(not any(k.startswith("teacher") for e in sp["entries"] for k in e), "selfplay: no teacher")

    def hard(cmd):
        return bool(cmd) and " --tier hard " in cmd + " " and "hard+sb" not in cmd

    ctx = EVAL.Context(run, cfg, "argmax", [])
    groups = {g.name: g for g in EVAL.build(ctx, ["fixed", "sprt"])}
    for name in ("scripted", "sprt-scripted"):
        g = groups.get(name)
        check(g is not None and all(hard(e.get("teacher_b")) and "teacher_a" not in e for e in g.entries),
              "evaluation %s: the opponent stays scripted:hard under a hard+sb teacher" % name)
    check(hard(EV.opponent_side("scripted")[1]) and hard(MT.agent_side("scripted")[1]) and
          hard(MT.agent_side(BM.OPPONENT)[1]), "rmtrain eval / sprt / bench: scripted is scripted:hard")
    check(MT.agent_side("scripted:hard+sb")[1] and "--tier hard+sb" in MT.agent_side("scripted:hard+sb")[1],
          "scripted:hard+sb names the sideboarding tier as an opponent")

    # bc.sideboard_weight on a synthetic batch: 6 decisions, 2 of them sideboarding.
    g = torch.Generator().manual_seed(5)
    n, a = 6, 5
    mask = torch.ones(n, a, dtype=torch.bool)
    mask[0, 3:] = False
    mask[4, 2:] = False
    label = torch.tensor([1, 0, 4, 2, 1, 3])
    sbf = torch.tensor([False, True, False, False, True, False])
    out = {"logits": torch.randn(n, a, generator=g), "value": torch.tanh(torch.randn(n, generator=g))}
    b = types.SimpleNamespace(inputs=[None] * 9 + [mask], label=label, outcome=torch.tensor([1., -1, 1, 0, -1, 1]),
                              targets=None, sideboarding=sbf, n=n)

    def losses(w):
        bc = BC.BCConfig(sideboard_weight=w)
        return BC.BCTrainer.losses(types.SimpleNamespace(cfg=bc), b, out), bc

    pl, _ = BC.policy_loss(out["logits"], mask, label, BC.BCConfig().label_smoothing)
    l1, c1 = losses(1.0)
    check(torch.equal(l1["policy"], pl.mean()), "sideboard_weight 1: the plain mean (the old loss) exactly")
    check(torch.equal(l1["total"], pl.mean() + c1.value_weight * l1["value"]), "sideboard_weight 1: the old total")
    l3, _ = losses(3.0)
    want = (pl[~sbf].sum() + 3 * pl[sbf].sum()) / (4 + 3 * 2)
    check(torch.allclose(l3["policy"], want) and torch.equal(l3["value"], l1["value"]),
          "sideboard_weight 3: sum(w·loss) / sum(w), the value loss unweighted")
    l0, _ = losses(0.0)
    check(torch.allclose(l0["policy"], pl[~sbf].mean()), "sideboard_weight 0: the other decisions' mean")
    check(float(l1["n_sideboarding"]) == 2 and torch.allclose(l1["sum_policy_sideboarding"], pl[sbf].sum()),
          "the sideboarding decisions' own loss sums")
    allsb = types.SimpleNamespace(**dict(vars(b), sideboarding=torch.ones(n, dtype=torch.bool)))
    z = BC.BCTrainer.losses(types.SimpleNamespace(cfg=BC.BCConfig(sideboard_weight=0.0)), allsb, out)
    check(bool(torch.isfinite(z["policy"])) and float(z["policy"]) == 0.0, "sideboard_weight 0 on an all-sideboarding batch: 0, finite")
    check(raises(lambda: C.resolve(None, ["bc.sideboard_weight=-1"]), "bc.sideboard_weight"), "a negative weight refused")
    check(BC.BCConfig.from_dict(C.resolve(None, ["bc.sideboard_weight=4"])["bc"]).sideboard_weight == 4.0,
          "bc.sideboard_weight reaches BCConfig")

    # rnad.reg_eps_sideboarding about 0.
    for eps in (0.0, 0.01):
        C.resolve(None, ["rnad.reg_eps_sideboarding=%s" % eps])
        lg = out["logits"].clone().requires_grad_(True)
        sm = M.smoothed_log_policy(lg, mask, 0.1, sbf, eps)
        sm[mask].sum().backward()
        ok = bool(torch.isfinite(sm[mask]).all()) and bool(torch.isfinite(lg.grad).all())
        if eps == 0.0:
            ok = ok and torch.allclose(sm[sbf], M.masked_log_softmax(out["logits"], mask)[sbf])
        check(ok and RegPolicy(None, 0, 0.1, eps).smoothed(),
              "reg_eps_sideboarding %s: validates, finite log π and gradients%s" %
              (eps, ", the raw π at sideboarding steps" if eps == 0.0 else ""))


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
    o.cfg = C.resolve(None, [])
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

        def health(self):
            return []

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


# ── a resume in B4 and its evaluations (the first validation run's findings) ─────────────────

class _FakeGpu:
    """The GPU process's link: canned replies per command (a dict or a function of the arguments),
    every call recorded."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []
        self.proc = None
        self.fatal = ""

    def call(self, cmd, timeout=None, **kw):
        self.calls.append((cmd, kw))
        r = self.replies.get(cmd, {})
        return dict(r(kw) if callable(r) else r)

    def alive(self):
        return True

    def events(self):
        return []


def _b4_orchestrator(tmp, name, sets=()):
    """An orchestrator in B4 over a fake run (no engine: evaluation actors are /bin/false), its GPU
    process a _FakeGpu; returns (orchestrator, run, logs)."""
    from rmlearn.orchestrator.orchestrator import Orchestrator
    run = RD.RunDir(os.path.join(tmp, name))
    run.make()
    cfg = C.resolve(None, ["actors.count=0", "run.bin_dir=" + tmp, "eval.actors=2", "eval.matches=2",
                           "actors.policy_eval=remote", "eval.policy_eval=remote",
                           "eval.ladder_matches=2", "eval.sprt_max_games=2", "league.decks=[\"a\", \"b\"]",
                           "eval.pairs=[[\"a\", \"b\"]]"] + list(sets))
    RD.write_json(run.config, cfg)
    o = Orchestrator(run)
    logs = []
    o.log = lambda m: logs.append(m)
    o.state = o.new_state()
    o.state.update(phase="B4", stage="selfplay", selfplay_version="b4")
    o.metrics = RD.JsonLines(run.metrics)
    o.evals_file = RD.JsonLines(run.evals)
    o.tail = A.ResultsTail(run.actors)
    o.counters = A.Counters()
    o.pool = A.ActorPool("/bin/false", tmp, run.schedule, run.actors, run.actor_logs, 1, cfg["actors"], o.log)
    o.engine = "/bin/false"
    o.bin_dir = tmp
    return o, run, logs


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        fp.write("x")
    return path


def _run_eval(o, timeout=20):
    """eval_tick until the started evaluation is finished (or cancelled)."""
    o.eval_tick()
    end = time.time() + timeout
    while o.eval_job is not None and time.time() < end:
        o.eval_tick()
        time.sleep(0.05)


def check_resume_and_evaluations(tmp):
    """The val1 run: a GPU-process kill in B4 mid-evaluation (it1), the restore at an earlier step,
    the iteration crossed again, a pause around the re-queued evaluation."""
    from rmlearn.rmtrain import format_status
    o, run, logs = _b4_orchestrator(tmp, "b4resume")
    st = o.state
    m1 = _touch(os.path.join(run.fixed, "m1.pt"))
    ck = _touch(os.path.join(run.ckpt, "rnad-p79.pt"))
    bc = _touch(os.path.join(run.ckpt, "bc-final.pt"))
    # Before the crash: step 110, iteration 1, its fixed point a rung, its evaluation running.
    st["rnad"].update(learner_steps=110, iterations=1)
    st["checkpoint"] = {"kind": "rnad", "path": ck, "learner_steps": 79, "version": 79}
    st["ladder"] = [{"label": "bc", "path": bc, "version": 12000}, {"label": "m1", "path": m1, "version": 100}]
    st["eval"]["queue"] = [{"label": "it1", "options": "", "evaluators": ["fixed", "ladder"], "phase": "B4",
                            "policy": {"label": "m1", "path": m1, "version": 100, "iteration": 1}},
                           {"label": "t95", "options": "", "evaluators": ["fixed"], "phase": "B4"}]
    o.last_metrics["rnad_step"] = {"event": "rnad_step", "step": 110, "time": time.time()}
    lad = EVAL.Ladder(os.path.join(run.path, "ladder"))
    for fid, phase in (("v105", "B4"), ("v60", "B4"), ("v11050", "B2")):     # v11050: a BC step
        lad.add_fixed_point(fid, "nn:remote:policy-3", int(fid[1:]), "argmax", {"phase": phase})
        lad.record_counts(fid, "scripted", 3, 1)
    lad.record_counts("m1", "scripted", 2, 2)
    stats = {"learner_steps": 79}
    o.gpu = _FakeGpu({"rnad_init": {"learner_steps": 79, "reg_updates": 0},
                      "stats": lambda kw: {"learner_steps": stats["learner_steps"], "mode": "rnad"},
                      "snapshot": lambda kw: {"pid": kw["pid"], "version": stats["learner_steps"]},
                      "checkpoint": lambda kw: {"kind": "rnad", "learner_steps": 81, "reg_updates": 0, "version": 81,
                                                "path": kw["path"].replace("{step}", "81"), "bytes": 1}})
    o.ensure_rnad_model()
    # 3: progress and metrics follow the restored learner.
    o.write_status()
    p = o.status_cache["progress"]
    check(p["learner_steps"] == 79 and p["iterations"] == 0,
          "resume: the progress is the restored learner's (%s), not the run's last (110, 1)" % p)
    check("rnad_step" not in o.status_cache["metrics"] and o.status_cache["metrics"].get("restore", {}).get(
        "learner_steps") == 79, "resume: the pre-crash rnad_step line is gone; a restore line names step 79")
    stats["learner_steps"] = 95
    o.gpu_stats_at = 0
    o.write_status()
    check(o.status_cache["progress"]["learner_steps"] == 95, "the progress follows the GPU process's learner step")
    r = o.checkpoint()
    check(o.gpu.calls[-1][1]["path"].endswith("rnad-p{step}.pt") and r["path"].endswith("rnad-p81.pt") and
          st["rnad"]["learner_steps"] >= 81, "a periodic checkpoint is named by the GPU process's step")
    # 1: the lost iteration's rung, evaluation and ladder players are dropped; crossing it again
    # adds the rung once, however often the event comes.
    check([r["label"] for r in st["ladder"]] == ["bc"] and [x["label"] for x in st["eval"]["queue"]] == ["t95"],
          "resume: the lost iteration's rung and evaluation dropped (%s, %s)" % (
              st["ladder"], [x["label"] for x in st["eval"]["queue"]]))
    players = {f["id"] for f in EVAL.Ladder(os.path.join(run.path, "ladder")).fixed_points()} | {
        x for pr in EVAL.Ladder(os.path.join(run.path, "ladder")).pairings() for x in (pr["a"], pr["b"])}
    check("m1" not in players and "v105" not in players and "v60" in players and "v11050" in players,
          "resume: the ladder record's players of lost steps removed (%s)" % sorted(players))
    st["eval"]["queue"] = []
    ev = {"reg_updates": 1, "learner_steps": 100, "version": 100, "fixed_point": m1}
    o.on_iteration(dict(ev))
    o.on_iteration(dict(ev))
    check([r["label"] for r in st["ladder"]] == ["bc", "m1"], "an iteration crossed again: one rung m1 (%s)" %
          [r["label"] for r in st["ladder"]])
    stale = [{"label": "bc", "path": bc, "version": 1}, {"label": "m1", "path": m1, "version": 100},
             {"label": "m1", "path": m1, "version": 100}]
    names = [g.name for g in EVAL.build(EVAL.Context(run, o.cfg, "argmax", stale), ["ladder"])]
    check(names == ["ladder-bc", "ladder-m1"], "the ladder evaluator: one group per rung (%s)" % names)
    dup = [EVAL.Group("x", [{"id": "x|a-b"}], 1), EVAL.Group("x", [{"id": "x|a-b"}], 1)]
    check(raises(lambda: EVAL.check_groups(dup), "twice"), "colliding groups are refused")
    q = st["eval"]["queue"]
    check([x["label"] for x in q] == ["it1", "it1-sprt"] and all(x["policy"]["path"] == m1 for x in q),
          "the iteration's evaluations name its fixed point (%s)" % [(x["label"], x.get("policy")) for x in q])
    # 4: the iteration's evaluation loads m1.pt as the subject (not the current network); the
    # fixed point is not its own ladder opponent. The evaluation actors fail (/bin/false): 1's
    # silent {} is now an error.
    st["eval"]["queue"] = q[:1]
    o.gpu.calls = []
    o.gpu_model = "rnad"
    o.eval_tick()
    job = o.eval_job
    calls = [(c, kw.get("pid"), kw.get("path")) for c, kw in o.gpu.calls]
    check(("load_policy", EVAL.SUBJECT_PID, m1) in calls and not any(c == "snapshot" for c, _, _ in calls),
          "an iteration's evaluation evaluates its fixed point (%s)" % calls)
    check(job is not None and [g.name for g in job.groups] == ["scripted", "explore", "random", "ladder-bc"] and
          job.meta["subject"] == "m1" and job.meta["version"] == 100,
          "its groups: the fixed opponents and the other rungs (%s)" % ([g.name for g in job.groups] if job else None))
    # A pause stops it and queues it again, still for m1.
    st["paused"] = True
    o.eval_tick()
    check(o.eval_job is None and st["eval"]["queue"][0]["policy"]["path"] == m1,
          "a paused evaluation is queued again for the same fixed point")
    st["paused"] = False
    _run_eval(o)
    last = st["eval"]["last"]
    check(last.get("error", "").startswith("no games played") and "exit 1" in last["error"],
          "an evaluation that played no game is an error (%s)" % last)
    check(any(m.startswith("ERROR: evaluation it1") for m in logs), "the error is in the log")
    mets = RD.read_jsonl(run.metrics)
    check(any(m["event"] == "eval_failed" and m["label"] == "it1" for m in mets) and
          not any(m["event"] == "eval" for m in mets), "metrics.jsonl: eval_failed, no eval line")
    check("ERROR no games played" in format_status(dict(o.status_cache, eval={"last": last})),
          "the status shows the failed evaluation")
    # A cadence evaluation evaluates the current network when it runs.
    o.gpu.calls = []
    stats["learner_steps"] = 120
    o.queue_eval("t120", o.eval_options(), ["fixed"])
    o.eval_tick()
    check(o.eval_job is not None and o.eval_job.meta["version"] == 120 and o.eval_job.meta["subject"] == "v120" and
          any(c == "snapshot" for c, _ in o.gpu.calls), "a cadence evaluation snapshots the current network")
    o.cancel_eval()
    o.metrics.close()
    o.evals_file.close()


def check_eval_options_and_findings(tmp):
    """5: evaluations at argmax by default (B5: the projection), the sampled policy as an optional
    secondary group, every result labelled; 6: draws as findings in the status."""
    from rmlearn.rmtrain import format_status
    d = C.defaults()["eval"]
    check(d["nn_options_rnad"] == "argmax" and d["nn_options_bc"] == "argmax" and d["also_sampled"] is False,
          "evaluations play argmax by default; no sampled entry by default")
    o, run, logs = _b4_orchestrator(tmp, "evopts", ["eval.also_sampled=true", "b4.nn_options=temp=1.5"])
    check(o.eval_options() == "argmax", "B4: the evaluation options are argmax")
    o.state["phase"] = "B5"
    check(o.eval_options() == C.defaults()["eval"]["nn_options_finetune"], "B5: the fine-tuning projection")
    o.state["phase"] = "B4"
    old_cfg, _ = C.fill_defaults({k: {kk: vv for kk, vv in v.items() if kk != "also_sampled"}
                                  for k, v in C.defaults().items()})
    check(old_cfg == C.defaults(), "a run created before a key existed resumes with its default")
    o.queue_eval("t5", o.eval_options(), ["fixed"])
    req = o.state["eval"]["queue"][0]
    check(req["options"] == "argmax" and req["sampled_options"] == "temp=1.5", "the request carries both options")
    ctx = EVAL.Context(run, o.cfg, req["options"], [], sampled_options=req["sampled_options"])
    groups = EVAL.build(ctx, ["fixed"])
    by = {g.name: g for g in groups}
    check(sorted(by) == ["explore", "explore@sampled", "random", "random@sampled", "scripted", "scripted@sampled"] and
          by["scripted"].entries[0]["agent_a"].endswith(":policies=3:argmax") and
          by["scripted@sampled"].entries[0]["agent_a"].endswith(":policies=3:temp=1.5") and
          by["scripted@sampled"].options == "temp=1.5", "also_sampled: a sampled group per fixed opponent")
    # A finished evaluation: options per group, the brief labelled, sampled groups out of the ratings,
    # its draws noted as findings.
    job = EVAL.EvalJob(run, o.cfg, 9, "t5", [by["scripted"], by["scripted@sampled"]],
                       {"version": 7, "subject": "v7", "options": "argmax", "phase": "B4"}, "/bin/false", tmp, o.log)
    os.makedirs(job.dir, exist_ok=True)

    class _Done:
        def tick(self):
            return []

        def finished(self):
            return True

        def health(self):
            return []

        def stop_all(self, timeout=None):
            pass

    job.pool = _Done()
    job.started = time.time()
    rmlog = os.path.join(job.dir, "rmlog", "a0-m1.rmlog")
    with open(os.path.join(job.dir, "a0.results.jsonl"), "w") as fp:
        for i, (entry, res, end) in enumerate([("scripted|a-b", 1, "win"), ("scripted|a-b", 0, "draw"),
                                               ("scripted@sampled|a-b", -1, "win")]):
            fp.write(json.dumps({"actor": 0, "match": 1, "game": i + 1, "entry": entry, "side_a_seat": "A",
                                 "deck_a": "league/a", "deck_b": "league/b", "agent_a": "nn:remote:s:policies=3",
                                 "agent_b": "first", "result_a": res, "result_b": -res, "end_reason": end,
                                 "decisions": 50, "rmlog": rmlog}) + "\n")
    job.request = {"label": "t5", "evaluators": ["fixed"]}
    o.eval_job = None
    check(job.poll(), "the finished evaluation")
    o.gpu = _FakeGpu({})
    o.finish_eval(job)
    res = RD.read_jsonl(run.evals)[-1]
    check(res["groups"]["scripted"]["options"] == "argmax" and res["groups"]["scripted@sampled"]["options"] ==
          "temp=1.5" and res["options"] == "argmax", "evals.jsonl: the options of the result and of each group")
    b = o.state["eval"]["last"]["brief"]
    check(b["scripted"].endswith("[argmax]") and b["scripted@sampled"].endswith("[temp=1.5]"),
          "the brief labels each group's options (%s)" % b)
    check(o.state["eval"]["last"]["options"] == "argmax" and o.state["eval"]["last"]["subject"] == "v7",
          "the last evaluation names its subject and options")
    ratings = o.state["eval"]["ratings"]
    check(ratings["v7"]["games"] == 2, "the ratings count the primary games only (%s)" % ratings)
    # Training actors' draws.
    with open(os.path.join(run.actors, "a3.results.jsonl"), "w") as fp:
        fp.write(json.dumps({"actor": 3, "match": 126, "game": 2, "entry": "sp-a-b", "schedule": "b4",
                             "deck_a": "league/ur_delver", "deck_b": "league/burn", "result_a": 0, "result_b": 0,
                             "end_reason": "draw", "decisions": 64,
                             "rmlog": os.path.join(run.path, "rmlog", "selfplay", "a3-m126.rmlog")}) + "\n")
        fp.write(json.dumps({"actor": 3, "match": 127, "game": 1, "schedule": "b4", "result_a": 1, "result_b": -1,
                             "end_reason": "win", "decisions": 80}) + "\n")
    o.ingest_results()
    fd = o.state["findings"]
    check(fd["draws"] == 2 and fd["recent"][-1]["rmlog"].endswith("a3-m126.rmlog") and
          fd["recent"][0]["source"] == "eval t5", "draws as findings: the evaluation's and the actors' (%s)" % fd)
    o.write_status()
    text = format_status(o.status_cache)
    check("findings: 2 draws" in text and "rmlog/selfplay/a3-m126.rmlog" in text and
          "actors/a3.results.jsonl" in text and "ur_delver vs burn" in text,
          "the status's findings line names each draw's results file and decision log")
    check("FAIL" not in text and not any("ERROR" in m for m in logs), "a draw is not an error")
    o.metrics.close()
    o.evals_file.close()


# ── local actors (actors.policy_eval / eval.policy_eval local) ────────────────────────────────

FAKE_EXPORTER = r"""
import json, os, sys, time
for line in sys.stdin:
    r = json.loads(line)
    time.sleep(0.05)
    if "FAIL" in r["checkpoint"]:
        print(json.dumps({"re": r["id"], "ok": False, "error": "boom", "seconds": 0.05}), flush=True)
        continue
    os.makedirs(os.path.dirname(r["out"]), exist_ok=True)
    with open(r["out"], "w") as fp:
        fp.write("v%d" % r["version"])
    print(json.dumps({"re": r["id"], "ok": True, "out": r["out"], "version": r["version"], "seconds": 0.05}),
          flush=True)
"""


def _fake_live(run, cfg, log):
    """A LiveExporter whose process is FAKE_EXPORTER (writes "v<version>" to the out file)."""
    import types
    from rmlearn.orchestrator.live import LiveExporter
    live = LiveExporter(run, cfg, log)

    def start(self):
        import queue as Q
        self.proc = subprocess.Popen([sys.executable, "-c", FAKE_EXPORTER], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        q, proc = self.replies, self.proc

        def read():
            for line in proc.stdout:
                q.put(json.loads(line))
            q.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        del Q
    live._start = types.MethodType(start, live)
    return live


def _drain(o, cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        o.live_tick()
        o.eval_tick()
        if cond():
            return True
        time.sleep(0.02)
    return False


def check_local(tmp):
    """actors.policy_eval / eval.policy_eval local: the config, the schedules' in-process specs, the
    GPU process's save cadence, the live exporter's coalescing, the actors waiting for the first
    export, local evaluations exporting their policies first, the policy-lag cap and percentiles."""
    import types
    import numpy as np
    from rmlearn.orchestrator.live import export_path
    # Config.
    check(raises(lambda: C.resolve(None, ["actors.policy_eval=gpu"]), "actors.policy_eval"),
          "local: actors.policy_eval is remote or local")
    check(C.parse_live("actors.export_every_s", "10") == ("actors.export_every_s", 10.0) and
          C.parse_live("rnad.max_policy_lag", "8") == ("rnad.max_policy_lag", 8) and
          raises(lambda: C.parse_live("actors.policy_eval", "local"), "not settable"),
          "local: the export cadence and the lag cap are live, policy_eval is not")
    check(raises(lambda: C.resolve(None, ["actors.export_every_s=0"]), "export_every_s"), "local: a cadence > 0")
    # Schedules.
    run = RD.RunDir(os.path.join(tmp, "loc"))
    run.make()
    cfg = C.resolve(None, ["league.decks=[\"a\", \"b\"]", "actors.policy_eval=local", "b4.nn_options=temp=2"])
    cur = os.path.join(run.exports, "live", "current.ts.pt")
    sp, dg = SCH.selfplay(run, cfg, "b4"), SCH.dagger(run, cfg, 1, 0.5)
    check(all(e["agent_a"] == e["agent_b"] == "nn:%s:temp=2" % cur for e in sp["entries"]) and
          all(e["agent_a"] == "nn:%s" % cur and e["teacher_mix"] == 0.5 for e in dg["entries"]),
          "local: self-play and DAgger play the live export in process (%s)" % sp["entries"][0]["agent_a"])
    pt2 = SCH.selfplay(run, C.set_key(cfg, "actors.export_format", "pt2"), "b4")
    check(pt2["entries"][0]["agent_a"].startswith("nn:" + cur[:-len(".ts.pt")] + ".pt2"), "local: a .pt2 export")
    check(raises(lambda: SCH.nn_local("/a:b/c.ts.pt"), "cannot contain"), "local: a model path with ':' is refused")
    # The GPU process's save cadence.
    RD.write_json(run.config, cfg)
    w = G.Worker(run.path, None)
    w.cfg = C.resolve(None, ["actors.policy_eval=local", "actors.export_every_s=30"])
    w.learner = types.SimpleNamespace(learner_steps=10)
    saved, events = [], []
    w.save_model = lambda path: saved.append(path) or {"path": path, "version": w.learner.learner_steps,
                                                       "seconds": 0.0}
    w.event = lambda name, **kw: events.append((name, kw["version"]))
    w.maybe_publish()
    w.maybe_publish()
    t_first = w.published[1]
    w.learner.learner_steps = 12
    w.maybe_publish()                                   # stepped, not due
    w.published = (12 - 2, t_first - 31)
    w.maybe_publish()                                   # due, stepped: saved
    w.published = (12, time.time() - 31)
    w.maybe_publish()                                   # due, but the learner has not stepped
    w.cfg["actors"]["export_every_steps"] = 3
    w.learner.learner_steps = 15
    w.maybe_publish()                                   # 3 steps: saved before the time is due
    check(events == [("policy_saved", 10), ("policy_saved", 12), ("policy_saved", 15)] and
          saved[0].endswith("exports/live/src-v10.pt"),
          "local: the GPU process saves at once, then every export_every_s / export_every_steps when stepped (%s)"
          % events)
    w.cfg["actors"]["policy_eval"] = "remote"
    w.learner.learner_steps = 30
    w.published = (15, 0)
    w.maybe_publish()
    check(len(events) == 3, "remote: nothing saved for the actors")
    # The policy-lag cap and percentiles.
    eps = [types.SimpleNamespace(policy_version=np.array(v)) for v in ([98, 99], [90, 99], [-1, 100], [95])]
    kept, n = G.drop_stale(eps, 100, 5)
    check(n == 1 and kept == [eps[0], eps[2], eps[3]] and G.drop_stale(eps, 100, 0) == (eps, 0),
          "the policy-lag cap drops games older than rnad.max_policy_lag (0: off)")
    import collections
    from rmlearn import replay as RP
    win = RP.ReplayWindow(RP.WindowConfig(dirs=[]))
    fr = [types.SimpleNamespace(split="train", i=i) for i in range(5)]
    win._fresh = collections.deque(fr)
    got = [e.i for e in win.take_fresh(2, timeout=0, newest=True)] + [e.i for e in win.take_fresh(2, timeout=0)]
    check(got == [4, 3, 0, 1], "rnad.fresh_order newest: a step takes the newest fresh games (%s)" % got)
    check(C.parse_live("rnad.fresh_order", "newest") == ("rnad.fresh_order", "newest") and
          raises(lambda: C.resolve(None, ["rnad.fresh_order=random"]), "fresh_order"), "rnad.fresh_order is live")
    pc = G.lag_percentiles([np.arange(0, 50), np.arange(50, 101)])
    check(pc["policy_lag_p50"] == 50 and pc["policy_lag_max"] == 100 and abs(pc["policy_lag_p95"] - 95) < 1e-9,
          "the policy lag's percentiles (%s)" % pc)
    # The orchestrator: the actors wait for the first export; exports coalesce; evaluations in process.
    o, run, logs = _b4_orchestrator(tmp, "b4local", ["actors.policy_eval=local", "eval.policy_eval=local",
                                                     "actors.count=2", "actors.local_count=3",
                                                     "eval.evaluators=[\"fixed\", \"ladder\"]",
                                                     "eval.opponents=[\"random\"]"])
    o.live = _fake_live(run, o.cfg, o.log)
    o.gpu = _FakeGpu({"save_model": lambda kw: {"path": kw["path"], "version": 42}})
    o.gpu_model = "rnad"
    o.gpu_mode = o.gpu_paused = "x"
    o.live_reset()
    o.sync()
    check(o.pool.count == 0, "local: no actor starts before the policy's first export")
    srcs = [_touch(os.path.join(run.exports, "live", "src-v%d.pt" % v)) for v in (5, 6, 7)]
    for v, src in zip((5, 6, 7), srcs):
        o.on_gpu_event({"event": "policy_saved", "path": src, "version": v, "seconds": 0.01})
    check(not os.path.exists(srcs[1]), "local: a queued live export is replaced by a newer one (its source deleted)")
    check(_drain(o, lambda: o.live_ok and not o.live.pending()), "local: the live exports finish")
    with open(SCH.live_export(run, o.cfg)) as fp:
        content = fp.read()
    lines = [m for m in RD.read_jsonl(run.metrics) if m["event"] == "live_export"]
    check(content == "v7" and [m["version"] for m in lines] == [5, 7] and not any(os.path.exists(p) for p in srcs),
          "local: exported 5 then 7 over the actors' file, the sources deleted (%s, %s)" % (content, lines))
    o.sync()
    check(o.pool.count == 3 and o.live_status()["ok"],
          "local: actors.local_count actors start once the export is in place (%d)" % o.pool.count)
    o.state["phase"] = "B0"
    check(o.actor_count() == 2, "local: actors.count in the phases without the network (B0)")
    o.state["phase"] = "B4"
    o.pool.stop_all(timeout=1)
    o.live_reset()
    o.pool.count = 0
    o.sync()
    check(o.pool.count == 0, "local: a new model (a resume, a DAgger round) makes the actors wait again")
    # A local evaluation: the fixed point and the ladder rungs exported first, played in process.
    m1 = _touch(os.path.join(run.fixed, "m1.pt"))
    o.state["ladder"] = [{"label": "bc", "path": _touch(os.path.join(run.ckpt, "bc-final.pt")), "version": 3},
                         {"label": "m1", "path": m1, "version": 100}]
    o.queue_eval("it1", "argmax", ["fixed", "ladder"], policy={"label": "m1", "path": m1, "version": 100,
                                                               "iteration": 1})
    o.eval_tick()
    wait = o.eval_job
    want = [export_path(os.path.join(run.exports, "eval"), s, "ts") for s in ("m1-v100", "bc-v3")]
    check(isinstance(wait, EVAL.ExportWait) and sorted(wait.outs) == sorted(want),
          "local evaluation: waits for the subject's and the rungs' exports (%s)" % getattr(wait, "outs", None))
    check(_drain(o, lambda: isinstance(o.eval_job, EVAL.EvalJob)), "local evaluation: started once exported")
    job = o.eval_job
    ents = [e for g in job.groups for e in g.entries]
    check(all(e["agent_a"].startswith("nn:" + want[0]) for e in ents) and
          any(e["agent_b"] == "nn:%s:argmax" % want[1] for e in ents) and not job.loads() and
          not any(c[0] in ("load_policy", "snapshot") for c in o.gpu.calls),
          "local evaluation: the subject and the rungs in process, no server policy loaded")
    o.cancel_eval()
    done = o.live.done
    o.queue_eval("it1b", "argmax", ["fixed"], policy={"label": "m1", "path": m1, "version": 100, "iteration": 1})
    check(_drain(o, lambda: isinstance(o.eval_job, EVAL.EvalJob)) and o.live.done == done,
          "local evaluation: an export newer than its checkpoint is reused")
    o.cancel_eval()
    time.sleep(0.02)
    _touch(m1)                  # the fixed point written again (a resume crossed the boundary again)
    o.queue_eval("it1c", "argmax", ["fixed"], policy={"label": "m1", "path": m1, "version": 100, "iteration": 1})
    check(_drain(o, lambda: isinstance(o.eval_job, EVAL.EvalJob)) and o.live.done == done + 1,
          "local evaluation: an export older than its checkpoint is exported again")
    o.cancel_eval()
    o.queue_eval("t9", "argmax", ["fixed"])
    o.eval_tick()
    snap = [c for c in o.gpu.calls if c[0] == "save_model"]
    check(snap and isinstance(o.eval_job, EVAL.ExportWait) and o.eval_job.meta["subject"] == "v42",
          "local evaluation of the current network: saved (save_model), exported, named v<version>")
    check(_drain(o, lambda: isinstance(o.eval_job, EVAL.EvalJob)), "local evaluation: the snapshot exported")
    o.cancel_eval()
    o.queue_eval("it2", "argmax", ["fixed"], policy={"label": "m2", "path": _touch(os.path.join(run.fixed, "FAIL.pt")),
                                                     "version": 200, "iteration": 2})
    check(_drain(o, lambda: o.eval_job is None), "local evaluation: a failed export ends it")
    check(any(m["event"] == "eval_failed" and m["label"] == "it2" and "export" in m["error"]
              for m in RD.read_jsonl(run.metrics)), "local evaluation: a failed export is an eval_failed error")
    o.live.stop()


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
local_count = 0
teacher_tier = "hard+sb"
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
sideboard_weight = 4.0
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
reg_eps_sideboarding = 0.01
push_every = 10
report_every = 10
[b4]
max_learner_steps = 250
[b5]
max_learner_steps = 300
[sb_explore]
p_start = 0.5
p_end = 0.1
hold_frac = 0.5
rewrite_every_s = 5
[eval]
pairs = [["ur_delver", "burn"]]
matches = 2
ladder = 2
ladder_matches = 2
actors = 2
opponents = ["scripted", "random"]
every_minutes = 0
evaluators = ["sprt", "fixed", "ladder"]
also_sampled = true
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
        stop_unit(run)
        shutil.rmtree(run.path)
    cfgp = os.path.join(tempfile.mkdtemp(), "smoke.toml")
    with open(cfgp, "w") as fp:
        fp.write(SMOKE_CONFIG)
    env_note = "systemd-run" if shutil.which("systemd-run") else "daemon"
    t0 = time.time()
    mode = args.policy_eval or C.defaults()["actors"]["policy_eval"]
    local = mode == "local"
    sets = ["--set", "actors.policy_eval=" + mode, "--set", "eval.policy_eval=" + mode]
    r = sh([rmtrain, "start", run.path, "--config", cfgp] + sets)
    check(r.returncode == 0 and "started as" in r.stdout, "start detaches (%s%s)" % (
        env_note, ", local actors and evaluations" if local else ""))
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
    fails = [(m["label"], m.get("error")) for m in mets if m["event"] == "eval_failed"]
    check(not fails, "no evaluation failed (%s)" % fails)
    its = [e for e in evs if e["label"].startswith("it")]
    check(its and all(e.get("subject") == "m" + e["label"][2:].split("-")[0] for e in its),
          "each iteration's evaluation evaluated its fixed point (%s)" % [(e["label"], e.get("subject")) for e in its])
    check(all(e.get("options") is not None for e in evs if "exploiter-baseline" not in e["results"]),
          "every evaluation names its options")
    opts = [(k, g.get("options")) for e in its for k, g in e["results"].items() if not k.startswith("sprt-")]
    check(any(k.endswith("@sampled") for k, _ in opts) and
          all(o == ("" if k.endswith("@sampled") else "argmax") for k, o in opts),
          "eval.also_sampled: sampled groups beside the argmax ones, each labelled (%s)" % opts)
    labels = [r["label"] for r in st["ladder"]]
    check(len(labels) == len(set(labels)), "the ladder's rungs are unique after the kill (%s)" % labels)
    rest = [m for m in mets if m["event"] == "restore" and m.get("model") == "rnad"]
    check(rest and all(m["learner_steps"] <= m["previous_learner_steps"] for m in rest),
          "the resumes' restore lines (%s)" % [(m["learner_steps"], m["previous_learner_steps"]) for m in rest])
    check(any(m["event"] == "fatal" for m in mets) is False, "no fatal event")
    smoke_sideboarding_checks(run, mets)
    exports = os.listdir(run.exports)
    check(any(f.endswith(".pt2") for f in exports), "exports written (%s)" % sorted(exports)[:6])
    if local:
        smoke_local_checks(run, mets, evs)
    print("smoke: phase start times (s): %s; total %.1f s" % (json.dumps({k: round(v, 1) for k, v in
                                                                        phase_at.items()}), total))
    if not args.keep and not FAILURES:
        stop_unit(run)
        shutil.rmtree(run.path)


def stop_unit(run):
    """Stops the run's systemd unit, also one waiting to restart, before its directory is deleted."""
    subprocess.run(["systemctl", "--user", "stop", RT.unit_name(run) + ".service"], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def smoke_sideboarding_checks(run, mets):
    """The sideboarding teacher's phases in B0–B2 and the actors' exploring phases in B4, from the
    results lines and the orchestrator's sb_explore metrics."""
    lines = [json.loads(ln) for f in glob.glob(os.path.join(run.actors, "*.results.jsonl")) for ln in open(f)]
    phases = [(ln.get("schedule"), seat) for ln in lines for seat in (ln.get("sb") or {}).values()]
    taught = [s for v, s in phases if v not in ("b4", "b5")]
    check(any(s["swaps"] > 0 for s in taught), "the hard+sb teacher sideboarded in B0–B2 (%d phases, %d with swaps)"
          % (len(taught), sum(1 for s in taught if s["swaps"] > 0)))
    b4 = [s for v, s in phases if v == "b4"]
    check(any(s["explored"] for s in b4), "the actors explored sideboarding in B4 (%d of %d phases)"
          % (sum(1 for s in b4 if s["explored"]), len(b4)))
    check(not any(s["explored"] for v, s in phases if v != "b4"), "no exploring phase outside B4")
    sx = [m for m in mets if m["event"] == "sb_explore"]
    check(sx and sx[-1].get("explored_phases", 0) > 0, "sb_explore metrics lines with explored phases (%s)"
          % (sx[-1] if sx else None))


def smoke_local_checks(run, mets, evs):
    """--policy-eval local: the live exports followed the learner, the actors played them (each
    episode's policy versions are exported versions), the learner recomputed the regularisation log π,
    and the evaluations played their policies in process."""
    live = [m for m in mets if m["event"] == "live_export"]
    check(len(live) >= 3, "live exports of the current policy (%d)" % len(live))
    vers = [m["version"] for m in live if m["phase"] in ("B4", "B5")]
    check(vers and vers == sorted(vers) or any(m["event"] == "restore" for m in mets),
          "live export versions follow the learner (%s)" % vers[:12])
    check(any(m["phase"] == "B2" for m in live), "DAgger played a live export")
    steps = [m for m in mets if m["event"] == "rnad_step"]
    check(steps and all(m.get("reg_recorded", 0) == 0 for m in steps) and sum(m.get("reg_recomputed", 0)
                                                                                for m in steps) > 0,
          "the learner recomputed every regularisation log π (none recorded)")
    check(any("policy_lag_p95" in m for m in steps), "the steps report the policy lag's percentiles")
    results = []
    for path in glob.glob(os.path.join(run.actors, "a*.results.jsonl")):
        results += RD.read_jsonl(path)
    sp = [r for r in results if str(r.get("schedule", "")).startswith("b") and r["schedule"] in ("b4", "b5")]
    check(sp and all(str(r.get("agent_a", "")).startswith("nn:") and "remote" not in r["agent_a"] for r in sp),
          "self-play played in process (%s)" % (sp[0].get("agent_a") if sp else None))
    exported = set(vers)
    played = {v for r in sp for v in (r.get("policy_versions_a") or []) + (r.get("policy_versions_b") or [])}
    check(played and played <= exported | {0}, "the actors' policy versions are exported versions (%s not in %s)"
          % (sorted(played - exported)[:5], sorted(exported)[:8]))
    local_evs = [e for e in RD.read_jsonl(run.evals) if e.get("policy_eval") == "local"]
    check(local_evs and not any(e.get("error") for e in local_evs), "local evaluations ran (%d)" % len(local_evs))


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


def check_eval_scoring(tmp):
    """An evaluation's groups score the schedule's side a (the evaluated policy) from its
    ``side_a_seat``, whatever the agent specs: local exports, server policies, a ladder's two networks."""
    from rmlearn import evaluate as EV
    out = os.path.join(tmp, "scoring")
    os.makedirs(out, exist_ok=True)
    pol = "nn:/r/exports/eval/v7-v7.ts.pt:argmax"
    rung = "nn:/r/exports/eval/m1-v5.ts.pt:argmax"
    # (entry, side a's seat, side a's result): side a wins 3 of 4 against each opponent.
    games = [(opp, seat, res) for opp in ("random", "ladder-m1", "remote") for seat, res in
             (("A", 1), ("B", 1), ("A", -1), ("B", 1))]
    with open(os.path.join(out, "a0.results.jsonl"), "w") as fp:
        for i, (opp, seat, res) in enumerate(games):
            side_a = {"random": pol, "ladder-m1": pol, "remote": "nn:remote:/r/rmip.sock:argmax:policies=3"}[opp]
            side_b = {"random": "random", "ladder-m1": rung, "remote": "first"}[opp]
            agents = (side_a, side_b) if seat == "A" else (side_b, side_a)
            results = (res, -res) if seat == "A" else (-res, res)
            fp.write(json.dumps({"actor": 0, "match": i, "game": 1, "entry": opp + "|a-b", "side_a_seat": seat,
                                 "swapped": seat == "B", "agent_a": agents[0], "agent_b": agents[1],
                                 "result_a": results[0], "result_b": results[1], "end_reason": "win"}) + "\n")
    s = EV.summarize(out)
    for opp in ("random", "ladder-m1", "remote"):
        check(s[opp]["all"]["win"] == 3 and s[opp]["all"]["loss"] == 1,
              "summarize scores side a from side_a_seat (%s: %s)" % (opp, s[opp]["all"]))


def check_benchmark(tmp):
    """The end-of-run benchmark's policy set and summary: the default policies from a run's exports,
    per-deck win rates playing and facing a deck, the difference from the baseline, the tables."""
    from rmlearn import benchmark as BM
    from rmlearn import matches as MT
    from rmlearn.orchestrator.rundir import RunDir
    run = RunDir(os.path.join(tmp, "bench-run"))
    os.makedirs(run.exports)
    for n in ("bc-final", "m2", "m10", "m1", "final", "bc-r1"):
        open(os.path.join(run.exports, n + ".ts.pt"), "w").close()
    with open(run.config, "w") as fp:
        json.dump({"eval": {"nn_options_finetune": "thresh=0.03:disc=16"}}, fp)
    pol = BM.default_policies(run)
    check(pol == ["bc-final", "m1", "m2", "m10", "final", "final@thresh=0.03:disc=16"], "default policies: %s" % pol)
    for item, want in (("ref-m3=/r/exports/m3.ts.pt", ("ref-m3", "/r/exports/m3.ts.pt", "argmax")),
                       ("m2@thresh=0.03:disc=16", ("m2@thresh=0.03:disc=16", os.path.join(run.exports, "m2.ts.pt"),
                                                   "thresh=0.03:disc=16")),
                       ("p1=final@temp=1", ("p1", os.path.join(run.exports, "final.ts.pt"), "temp=1"))):
        got = BM.resolve_policy(run, item)
        check(got == want, "resolve_policy(%s): %s" % (item, got))
    os.makedirs(run.fixed)
    open(os.path.join(run.fixed, "m4.pt"), "w").close()
    open(os.path.join(run.ckpt, "rnad-final.pt"), "w").close()
    check(BM.checkpoint_of(run, "m4") == os.path.join(run.fixed, "m4.pt")
          and BM.checkpoint_of(run, "final") == os.path.join(run.ckpt, "rnad-final.pt")
          and BM.checkpoint_of(run, "m9") is None, "checkpoint_of")
    lab, path, opts = BM.resolve_policy(run, pol[-1])
    check(lab == "final@thresh=0.03:disc=16" and path == os.path.join(run.exports, "final.ts.pt")
          and opts == "thresh=0.03:disc=16", "resolve_policy: %s" % [lab, path, opts])

    def line(k, deck, opp, res):
        return {"actor": 0, "match": k, "game": 1, "entry": "%s-%s" % (deck, opp), "side_a_seat": "A",
                "deck_a": "league/" + deck, "deck_b": "league/" + opp, "result_a": res, "result_b": -res,
                "end_reason": "win" if res else "draw", "first": "A"}
    # The baseline wins 1 of 4 with burn against uw_control; the other policy 3 of 4 (and draws one).
    base = [line(k, "burn", "uw_control", 1 if k == 0 else -1) for k in range(4)]
    other = [line(k, "burn", "uw_control", -1 if k == 0 else 1) for k in range(4)] + [line(4, "burn", "uw_control", 0)]
    reps = {"bc-final": MT.report([MT.game_record(x) for x in base]),
            "final": MT.report([MT.game_record(x) for x in other])}
    s = BM.summarize(reps)
    f = s["policies"]["final"]
    check(s["baseline"] == "bc-final" and f["plays"]["burn"]["win_rate"] == 0.75
          and f["faces"]["uw_control"]["win_rate"] == 0.75 and abs(f["delta_plays"]["burn"]["delta"] - 0.5) < 1e-9
          and f["all"]["draws"] == 1 and len(f["findings"]) == 1, "benchmark summary: %s" % json.dumps(f)[:600])
    # Paired by match: the same matches, the other policy flips match 0 (a loss) and wins match 4's
    # game where the baseline drew nothing: identical elsewhere.
    mb = {("0", k): {"wins": int(r["result_a"] > 0), "decisive": 1, "deck": "burn", "decisions": [1]}
          for k, r in enumerate(base)}
    mo = {("0", k): {"wins": int(k != 0) if k else 0, "decisive": 1, "deck": "burn", "decisions": [1 if k > 1 else 2]}
          for k in range(4)}
    pd = BM.paired_delta(mb, mo)
    check(pd and abs(pd["delta"] - 0.5) < 1e-9 and pd["identical"] == 0.5 and pd["matches"] == 4
          and pd["two_se"] > 0, "paired_delta: %s" % pd)
    sp = BM.summarize(reps, matches={"bc-final": mb, "final": mo})
    check(sp["policies"]["final"].get("paired") and sp["policies"]["final"]["delta_all"]["identical"] == 0.5,
          "summarize paired: %s" % json.dumps(sp["policies"]["final"].get("delta_all")))
    md = BM.format_summary(s, "t")
    check("| final | 4 | 0.750 |" in md and "+0.500 ±" in md and "## Matchups: final" in md
          and "| burn | - | - | - | - | - | - | - | 0.75 | - | - |" in md, "benchmark tables:\n" + md)
    # run_benchmark: each policy played with the match count asked for, the summary paired by match.
    calls = []

    def fake_play(lab, path, opts, out_dir, matches, *a, **k):
        calls.append((lab, matches))
        return reps["bc-final" if lab == "bc-final" else "final"]
    play, recs = BM.play_policy, BM.match_records
    BM.play_policy = fake_play
    BM.match_records = lambda d: mb if d.endswith("bc-final") else mo
    try:
        sr = BM.run_benchmark(run, ["bc-final", "final"], label="t", matches=7, actors=2, log=lambda m: None)
    finally:
        BM.play_policy, BM.match_records = play, recs
    check(calls == [("bc-final", 7), ("final", 7)] and sr["matches"] == 7 and sr["policies"]["final"].get("paired")
          and os.path.exists(os.path.join(run.path, "bench", "t", "summary.md")), "run_benchmark: %s %s" % (calls, sr.get("matches")))


def _bench_line(match, game, seat_a, deck_a, deck_b, result_a, sb=None, first=None, end="win"):
    line = {"actor": 0, "match": match, "game": game, "entry": "x", "side_a_seat": seat_a,
            "deck_a": "league/" + deck_a, "deck_b": "league/" + deck_b, "result_a": result_a, "result_b": -result_a,
            "end_reason": end, "decisions": 10}
    if first:
        line["first"] = first
    if sb is not None:
        line["sb"] = sb
    return line


def check_benchmark_sideboarding(tmp):
    """The benchmark's sideboarding report (docs/rmtrain.md §7.1): the policy's own seat only, the
    configuration tracked across the match (game 3 from game 2's), play / draw from ``first`` or
    the previous game's loser, guide agreement on the play and on the draw (a game-3 flip, the draw
    land), held-in vs held-out, frame names (front faces, accents) against the guides', the swap
    rate and histogram, the top cards, the next game's win rate after swaps vs Done, explored and
    missing phases left out, the opponent's side apart; ``--opponent``'s directory and the reuse of
    ``report.json`` against the same opponent only."""
    from rmlearn import benchmark as BM
    from rmlearn import matches as MT
    from rmlearn import sb_report as SB
    from rmlearn.orchestrator.rundir import RunDir
    seat = _sb_seat
    burn_bug_in = ["Roiling Vortex", "Searing Blood", "Roiling Vortex", "Searing Blood"]
    burn_bug_out = ["Light Up the Stage", "Skewer the Critics", "Skewer the Critics", "Skewer the Critics"]
    lines = [
        # 1: the policy (burn, seat A) vs bug. Game 2 on the play (A lost game 1): the guide's board;
        # game 3 on the draw (B lost game 2): from game 2's maindeck the draw swap (Mountain out,
        # Dead/Gone's front face in).
        _bench_line(1, 1, "A", "burn", "bug", -1, first="A"),
        _bench_line(1, 2, "A", "burn", "bug", 1, first="A",
                    sb={"A": seat(False, 4, burn_bug_in, burn_bug_out), "B": seat(False, 0)}),
        _bench_line(1, 3, "A", "burn", "bug", -1, first="B",
                    sb={"A": seat(False, 1, ["Dead"], ["Mountain"]), "B": seat(False, 0)}),
        # 2: the policy (burn, seat B) vs uw_control, held out. Game 2 on the draw: Done (not the
        # guide's); game 3 without "first" (B lost game 2: on the play) boarded as the guide on the play.
        _bench_line(2, 1, "B", "uw_control", "burn", -1, first="A"),
        _bench_line(2, 2, "B", "uw_control", "burn", 1, first="A",
                    sb={"A": seat(False, 1, ["Containment Priest"], ["Island"]), "B": seat(False, 0)}),
        _bench_line(2, 3, "B", "uw_control", "burn", -1,
                    sb={"A": seat(False, 0), "B": seat(False, 4, ["Roiling Vortex", "Roiling Vortex",
                                                                  "Smash to Smithereens", "Smash to Smithereens"],
                                                         burn_bug_out)}),
        # 3: burn vs bug, game 2 on the play, a board that is not the guide's; the game capped.
        _bench_line(3, 1, "A", "burn", "bug", -1, first="A"),
        _bench_line(3, 2, "A", "burn", "bug", 0, first="A", end="cap",
                    sb={"A": seat(False, 1, ["Searing Blood"], ["Mountain"]), "B": seat(False, 0)}),
        # 4: an engine from before the sb key: no phase, counted as missing.
        _bench_line(4, 1, "A", "burn", "bug", -1, first="A"),
        _bench_line(4, 2, "A", "burn", "bug", 1, first="A"),
        # 5: an explored phase: left out, counted.
        _bench_line(5, 1, "A", "burn", "bug", -1, first="A"),
        _bench_line(5, 2, "A", "burn", "bug", 1, first="A",
                    sb={"A": seat(True, 2, ["Dead", "Dead"], ["Mountain", "Mountain"]), "B": seat(False, 0)}),
        # 6: bug vs bw_dnt on the play, the guide's board, cutting the frames' "Lórien Revealed".
        _bench_line(6, 1, "A", "bug", "bw_dnt", -1, first="A"),
        _bench_line(6, 2, "A", "bug", "bw_dnt", 1, first="A",
                    sb={"A": seat(False, 3, ["Pernicious Deed", "Pernicious Deed", "Toxic Deluge"],
                                  ["Thoughtseize", "Thoughtseize", "Lórien Revealed"]), "B": seat(False, 0)}),
    ]
    recs = SB.phases(lines, "a")
    got = [(r["deck"], r["game"], r["on_play"], r["swaps"], r["held_out"], r["agree"]) for r in recs]
    want = [("burn", 2, True, 4, False, True), ("burn", 3, False, 1, False, True),
            ("burn", 2, False, 0, True, False), ("burn", 3, True, 4, True, True),
            ("burn", 2, True, 1, False, False), ("burn", 2, True, 2, False, False),
            ("bug", 2, True, 3, False, True)]
    check(got == want, "sb_report phases: %s" % got)
    # A wrong game-2 board leaves game 3's draw swap off the guide too (the configuration carries over).
    alt = [dict(x) for x in lines[:3]]
    alt[1]["sb"] = {"A": seat(False, 3, burn_bug_in[:3], burn_bug_out[:3]), "B": seat(False, 0)}
    check([r["agree"] for r in SB.phases(alt, "a")] == [False, False], "sb_report: game 3 from game 2's maindeck")
    rep = SB.report(lines, "a")
    a = rep["all"]
    ag = a["agreement"]
    check(rep["explored"] == 1 and rep["missing"] == 1 and a["phases"] == 6
          and a["hist"] == [1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0] and abs(a["swap_rate"] - 5 / 6) < 1e-9,
          "sb_report counts: %s" % json.dumps(rep)[:800])
    check(a["after_swap"] == {"wins": 3, "decisive": 4, "win_rate": 0.75}
          and a["after_done"] == {"wins": 0, "decisive": 1, "win_rate": 0.0}, "sb_report next game: %s / %s"
          % (a["after_swap"], a["after_done"]))
    check((ag["held_in"]["phases"], ag["held_in"]["agree"]) == (4, 3)
          and (ag["held_in"]["on_play"]["phases"], ag["held_in"]["on_play"]["agree"]) == (3, 2)
          and (ag["held_in"]["on_draw"]["phases"], ag["held_in"]["on_draw"]["agree"]) == (1, 1)
          and (ag["held_out"]["phases"], ag["held_out"]["agree"], ag["held_out"]["rate"]) == (2, 1, 0.5),
          "sb_report agreement: %s" % ag)
    burn = rep["by_deck"]["burn"]
    check(set(rep["by_deck"]) == {"burn", "bug"} and burn["phases"] == 5
          and burn["in"] == [["Roiling Vortex", 4], ["Searing Blood", 3], ["Smash to Smithereens", 2], ["Dead", 1]]
          and burn["out"][:3] == [["Skewer the Critics", 6], ["Light Up the Stage", 2], ["Mountain", 2]]
          and burn["agreement"]["held_out"]["rate"] == 0.5 and rep["by_deck"]["bug"]["agreement"]["held_in"]["rate"] == 1.0,
          "sb_report per deck: %s" % json.dumps(rep["by_deck"])[:800])
    opp = SB.report(lines, "b")
    check(opp["all"]["phases"] == 7 and opp["all"]["hist"][:2] == [6, 1] and opp["by_deck"]["uw_control"]["in"]
          == [["Containment Priest", 1]] and "bw_dnt" in opp["by_deck"], "sb_report opponent: %s" % json.dumps(opp)[:600])
    # The summary: the section with the policy's row and, against hard+sb, the opponent's.
    reps = {"p": MT.report([MT.game_record(x) for x in lines])}
    s = BM.summarize(reps, sideboards={"p": {"policy": rep, "opponent": opp}})
    s["opponent"] = "scripted:hard+sb"
    md = BM.format_summary(s, "t")
    check("## Sideboarding" in md and "| p | 6 | 0.833 | 1/2/0/1/2/0/0/0/0/0/0 | 0.750 (4) | 0.000 (1) | 0.75 (3/4) | "
          "0.50 (1/2) | 0.67 (2/3) / 1.00 (1/1) |" in md and "| scripted:hard+sb (vs p) | 7 |" in md
          and "### Sideboarding per deck: scripted:hard+sb vs p" in md and "Roiling Vortex ×4" in md
          and "Explored phases left out" in md and "without sideboarding data" in md
          and "Every policy against scripted:hard+sb" in md, "summary sideboarding:\n" + md)
    # --opponent: the directory, the schedule's opponent, report.json reused only against its opponent.
    check(BM.bench_dir("end") == "end" and BM.bench_dir("end", "scripted") == "end"
          and BM.bench_dir("end", "scripted:hard+sb") == "end-vs-hard_sb", "bench_dir")
    ap = argparse.ArgumentParser()
    BM.add_args(ap)
    check(ap.parse_args([]).opponent == "scripted:hard"
          and ap.parse_args(["--opponent", "scripted:hard+sb"]).opponent == "scripted:hard+sb", "bench --opponent")
    sched = MT.eval_schedule("/x/p.ts.pt", "scripted:hard+sb", tmp)
    check(all("--tier hard+sb" in e["teacher_b"] and "teacher_a" not in e for e in sched["entries"]),
          "eval_schedule against hard+sb")
    pd = os.path.join(tmp, "bench-reuse")
    os.makedirs(pd)
    MT.write_json(os.path.join(pd, "report.json"), {"policy": {"label": "p"}, "all": {}})
    check(BM.play_policy("p", "/x", "argmax", pd, 1, 1, 11, log=lambda m: None) is not None, "reuse: hard (no key)")
    refused = False
    try:
        BM.play_policy("p", "/x", "argmax", pd, 1, 1, 11, log=lambda m: None, opponent="scripted:hard+sb")
    except SystemExit:
        refused = True
    check(refused, "reuse: a report against scripted:hard refused for hard+sb")
    run = RunDir(os.path.join(tmp, "bench-run-opp"))
    os.makedirs(run.exports)
    calls = []

    def fake_play(lab, path, opts, out_dir, matches, *a, **k):
        calls.append((lab, k.get("opponent"), out_dir))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "a0.results.jsonl"), "w") as fp:
            fp.write("".join(json.dumps(x) + "\n" for x in lines))
        return reps["p"]
    export = os.path.join(run.exports, "p.ts.pt")
    open(export, "w").close()
    play = BM.play_policy
    BM.play_policy = fake_play
    try:
        sr = BM.run_benchmark(run, [export], label="t", matches=2, actors=1, log=lambda m: None,
                              opponent="scripted:hard+sb")
    finally:
        BM.play_policy = play
    root = os.path.join(run.path, "bench", "t-vs-hard_sb")
    check(calls and calls[0][1] == "scripted:hard+sb" and calls[0][2].startswith(root)
          and sr["opponent"] == "scripted:hard+sb" and sr["dir"] == "t-vs-hard_sb"
          and sr["policies"]["p"]["sideboarding"]["policy"]["all"]["phases"] == 6
          and sr["policies"]["p"]["sideboarding"]["opponent"]["all"]["phases"] == 7
          and os.path.exists(os.path.join(root, "summary.md")), "run_benchmark --opponent: %s" % calls)


# ── structured sideboard exploration: p_explore, coverage counts, the schedule's weights ────────

def _sb_line(schedule, deck_a, deck_b, sb, purpose="train", game=2):
    line = {"actor": 0, "match": 0, "game": game, "schedule": schedule, "purpose": purpose, "deck_a": deck_a,
            "deck_b": deck_b, "decisions": 10, "end_reason": "win"}
    if sb is not None:
        line["sb"] = sb
    return line


def _sb_seat(explored, swaps, cards_in=(), cards_out=()):
    return {"explored": explored, "swaps": swaps, "in": list(cards_in), "out": list(cards_out)}


def _engine_with_sb_explore(tmp):
    """An engine binary (bin/debug, else bin/release) whose --actor schedule reader knows sb_explore,
    or None (a probe: p 1.5 must be refused as sb_explore's p)."""
    bad = os.path.join(tmp, "probe.json")
    RD.write_json(bad, {"entries": [{"id": "x", "deck_a": "league/bug", "deck_b": "league/burn", "agent_a": "first",
                                     "agent_b": "first", "sb_explore": {"p": 1.5}}]})
    for build in ("debug", "release"):
        eng = os.path.join(REPO_ROOT, "bin", build, "robomage")
        if not os.path.exists(eng):
            continue
        r = subprocess.run([eng, "--actor", "--actor-schedule", bad, "--actor-out", os.path.join(tmp, "probe"),
                            "--actor-matches", "0"], cwd=os.path.join(REPO_ROOT, "bin"), capture_output=True,
                           text=True, timeout=60)
        if r.returncode != 0 and "sb_explore.p" in r.stdout + r.stderr:
            return eng
    return None


def check_sb_explore(tmp):
    """Structured sideboard exploration's orchestrator side (docs/rmtrain.md §6.1): p_explore's
    schedule (hold, ramp, end; 0 outside B4; live), the coverage counts from results lines (the seat's
    deck after a swap, explored vs policy phases, distinct cards per phase), the weights
    (1 + n)^-power, the self-play schedule's sb_explore (both decks' weights; none at p 0, in B5, in B0,
    DAgger and evaluation schedules), the orchestrator's snapshot cadence, metrics, status and state
    round trip, the league deck files' card keys, and the engine's own schedule check when built."""
    from rmlearn.orchestrator import sb_explore as SBX
    from rmlearn.rmtrain import format_status
    d = C.resolve(None, [])
    check(d["sb_explore"]["p_start"] == 0.0 and d["sb_explore"]["count_weights"] == [1.0] * 11 and
          d["sb_explore"]["coverage"] is True and SBX.p_explore(d, "B4", 0) == 0.0,
          "sb_explore: off by default (p_start 0)")
    cfg = C.resolve(None, ["sb_explore.p_start=0.4", "sb_explore.p_end=0.1", "sb_explore.hold_frac=0.5",
                           "b4.max_learner_steps=1000"])
    ps = [SBX.p_explore(cfg, "B4", s) for s in (0, 500, 750, 1000, 5000)]
    check(all(abs(a - b) < 1e-12 for a, b in zip(ps, [0.4, 0.4, 0.25, 0.1, 0.1])),
          "sb_explore: p held, then linear, then the end (%s)" % ps)
    check(all(SBX.p_explore(cfg, ph, 750) == 0.0 for ph in ("B0", "B1", "B2", "B3", "B5", "done")),
          "sb_explore: p 0 outside B4")
    check(SBX.p_explore(C.set_key(cfg, "b4.max_learner_steps", 0), "B4", 10 ** 6) == 0.4,
          "sb_explore: p_start throughout B4 without b4.max_learner_steps")
    check(SBX.p_explore(C.set_key(cfg, "sb_explore.hold_frac", 1.0), "B4", 999) == 0.4 and
          SBX.p_explore(C.set_key(cfg, "sb_explore.hold_frac", 1.0), "B4", 1000) == 0.1,
          "sb_explore: hold_frac 1 holds to the end")
    for key in ("p_start", "p_end", "hold_frac", "count_weights", "coverage", "coverage_power", "rewrite_every_s"):
        check("sb_explore." + key in C.LIVE and "sb_explore." + key not in C.GPU_LIVE,
              "sb_explore.%s is live (the orchestrator's)" % key)
    key, v = C.parse_live("sb_explore.p_start", "0.2")
    check(abs(SBX.p_explore(C.set_key(cfg, key, v), "B4", 0) - 0.2) < 1e-12, "sb_explore: a live p_start")
    check(C.parse_live("sb_explore.count_weights", "[0,1,1,1,1,1,1,1,1,1,2]")[1][10] == 2, "live count_weights")
    for bad, needle in (("sb_explore.p_start=1.5", "p_start"), ("sb_explore.count_weights=[1,1]", "count_weights"),
                        ("sb_explore.count_weights=[0,0,0,0,0,0,0,0,0,0,0]", "count_weights"),
                        ("sb_explore.coverage_power=-1", "coverage_power"),
                        ("sb_explore.rewrite_every_s=0", "rewrite_every_s"), ("sb_explore.hold_frac=2", "hold_frac")):
        check(raises(lambda: C.resolve(None, [bad]), needle), "sb_explore: %s refused" % bad)
    check(SBX.p_moved(0.3, 0.29) and not SBX.p_moved(0.3, 0.295) and SBX.p_moved(0.001, 0.0) and
          not SBX.p_moved(0.0, 0.0), "sb_explore: when p moved enough to rewrite")
    check(SBX.card_key("Dead/Gone") == SBX.card_key("Dead") == SBX.card_key("Dead // Gone") == "dead" and
          SBX.card_key("Lórien Revealed") == SBX.card_key("Lorien Revealed") == "lorienrevealed",
          "sb_explore: card keys (front face, accents folded)")

    # Counting from results lines: seat A plays deck_a, seat B deck_b (per seat after the swap).
    X, Y = "league/x", "league/y"
    cov = SBX.Coverage()
    n = cov.add(_sb_line("b4", X, Y, {"A": _sb_seat(True, 2, ["Dead", "Pyroblast"], ["Bolt", "Bolt"]),
                                      "B": _sb_seat(False, 0)}))
    n += cov.add(_sb_line("b4", Y, X, {"A": _sb_seat(True, 1, ["Hydroblast"], ["Brainstorm"]),       # swapped
                                       "B": _sb_seat(True, 12, ["Dead", "Dead", "Smash"], ["Bolt", "Rift Bolt"])}))
    n += cov.add(_sb_line("b5", X, Y, {"A": _sb_seat(False, 3, ["Smash"], ["Bolt"]), "B": _sb_seat(False, 1)}))
    ignored = [cov.add(_sb_line("b4", X, Y, None, game=1)),
               cov.add(_sb_line("b4", X, Y, {"A": _sb_seat(True, 1, ["Dead"], ["Bolt"])}, purpose="eval")),
               cov.add(_sb_line("b0", X, Y, {"A": _sb_seat(True, 1, ["Dead"], ["Bolt"])})),
               cov.add(_sb_line("b2-r1", X, Y, {"A": _sb_seat(False, 1, ["Dead"], ["Bolt"])}))]
    xy, yx = cov.explored.get("league/x|league/y", {}), cov.explored.get("league/y|league/x", {})
    check(n == 6 and ignored == [0, 0, 0, 0], "sb_explore counts: 6 phases counted, game 1 / eval / B0 / DAgger "
          "lines ignored (%d, %s)" % (n, ignored))
    check(xy.get("phases") == 2 and xy["in"] == {"Dead": 2, "Pyroblast": 1, "Smash": 1} and
          xy["out"] == {"Bolt": 2, "Rift Bolt": 1} and xy["hist"][2] == 1 and xy["hist"][10] == 1,
          "sb_explore counts: x vs y from seat A unswapped and seat B swapped, a card once per phase, 10+ (%s)" % xy)
    check(yx.get("phases") == 1 and yx["in"] == {"Hydroblast": 1} and yx["out"] == {"Brainstorm": 1},
          "sb_explore counts: y vs x from the swapped line's seat A (%s)" % yx)
    pol = cov.policy
    check(pol.get("league/y|league/x", {}).get("hist", [0])[0] == 1 and
          pol.get("league/x|league/y", {}).get("phases") == 1 and pol["league/x|league/y"]["hist"][3] == 1 and
          pol["league/y|league/x"]["phases"] == 2 and pol["league/y|league/x"]["hist"][1] == 1,
          "sb_explore counts: the policy's phases and swap counts (%s)" % pol)
    check(SBX.Coverage(json.loads(json.dumps(cov.state()))).state() == cov.state(),
          "sb_explore counts: the state round trip")
    check(cov.take_recent() == [1, 1, 0, 1] + [0] * 7 and cov.take_recent() == [0] * 11,
          "sb_explore counts: the policy's recent histogram, taken and reset")

    # Weights and the self-play schedule.
    run = RD.RunDir(os.path.join(tmp, "sbx"))
    run.make()
    cfg = C.resolve(None, ["league.decks=[\"x\", \"y\", \"z\"]", "actors.policy_eval=remote",
                           "eval.policy_eval=remote", "sb_explore.p_start=0.3",
                           "sb_explore.count_weights=[1,2,1,1,1,1,1,1,1,1,0]"])
    snap = SBX.snapshot(cfg, cov, 0.3)
    w = snap["weights"]
    def close(got, want):
        return set(got) == set(want) and all(abs(got[k] - want[k]) < 1e-6 for k in want)

    check(close(w["league/x|league/y"]["in"], {"Dead": 1 / 3, "Pyroblast": 0.5, "Smash": 0.5}) and
          close(w["league/x|league/y"]["out"], {"Bolt": 1 / 3, "Rift Bolt": 0.5}),
          "sb_explore weights: (1 + n)^-1 (%s)" % w.get("league/x|league/y"))
    w2 = SBX.snapshot(C.set_key(cfg, "sb_explore.coverage_power", 2.0), cov, 0.3)["weights"]
    check(abs(w2["league/x|league/y"]["in"]["Dead"] - 1 / 9) < 1e-6, "sb_explore weights: power 2")
    check(SBX.snapshot(C.set_key(cfg, "sb_explore.coverage", False), cov, 0.3)["weights"] == {} and
          SBX.snapshot(C.set_key(cfg, "sb_explore.coverage_power", 0.0), cov, 0.3)["weights"] == {},
          "sb_explore weights: none with coverage off or power 0 (uniform)")
    sp = SCH.selfplay(run, cfg, "b4", snap)
    by = {e["id"]: e for e in sp["entries"]}
    sx = by["sp-x-y"].get("sb_explore") or {}
    check(sx.get("p") == 0.3 and sx.get("count_weights") == [1.0, 2.0] + [1.0] * 8 + [0.0] and
          sx.get("default_weight") == 1.0 and sx.get("weights") == {X: w["league/x|league/y"], Y: w["league/y|league/x"]},
          "selfplay: sb_explore with p, the count weights and both decks' weights (%s)" % sx)
    check(by["sp-x-z"]["sb_explore"]["weights"] == {} and by["sp-y-z"]["sb_explore"]["p"] == 0.3,
          "selfplay: an unexplored matchup explores with uniform weights")
    check(not any("sb_explore" in e for e in SCH.selfplay(run, cfg, "b4", SBX.snapshot(cfg, cov, 0.0))["entries"]),
          "selfplay: no sb_explore at p 0")
    check(not any("sb_explore" in e for e in SCH.selfplay(run, cfg, "b4")["entries"]), "selfplay: none without a snapshot")
    others = [SCH.b0(run, cfg), SCH.dagger(run, cfg, 1, 0.5)]
    ctx = EVAL.Context(run, cfg, "argmax", [{"label": "bc", "path": "b.pt", "version": 1}])
    ents = [e for s_ in others for e in s_["entries"]] + [e for g in EVAL.build(ctx, ["fixed", "ladder", "sprt"])
                                                          for e in g.entries]
    check(ents and not any("sb_explore" in e for e in ents) and
          not any("sb_explore" in s_["defaults"] for s_ in others), "B0, DAgger and evaluation schedules never explore")

    # The orchestrator: counts from the results files, the snapshot's cadence, metrics, status, state.
    o, orun, logs = _b4_orchestrator(tmp, "sbxo", ["sb_explore.p_start=0.5", "sb_explore.p_end=0.1",
                                                   "b4.max_learner_steps=1000", "sb_explore.rewrite_every_s=3600"])
    o.bin_dir = os.path.join(tmp, "sbx-bin")
    ddir = os.path.join(o.bin_dir, "resources", "decks", "league")
    os.makedirs(ddir)
    with open(os.path.join(ddir, "a.dk"), "w") as fp:
        fp.write("4 Lightning Bolt\n4 Lórien Revealed\n\nSIDEBOARD:\n2 Dead/Gone\n2 Pyroblast\n")
    with open(os.path.join(ddir, "b.dk"), "w") as fp:
        fp.write("4 Brainstorm\n\nSIDEBOARD:\n1 Hydroblast\n")
    lines = [_sb_line("b4", "league/a", "league/b", {"A": _sb_seat(True, 1, ["Dead"], ["Lorien Revealed"]),
                                                     "B": _sb_seat(False, 0)})] * 10
    lines.append(_sb_line("b4", "league/b", "league/a", {"A": _sb_seat(False, 1, ["Hydroblast"], ["Brainstorm"]),
                                                         "B": _sb_seat(True, 2, ["Pyroblast", "Dead"],
                                                                       ["Lightning Bolt", "Lightning Bolt"])}))
    with open(os.path.join(orun.actors, "a0.results.jsonl"), "w") as fp:
        fp.write("".join(json.dumps(x) + "\n" for x in lines))
    o.ingest_results()
    o.sb_tick()
    snap = o.sb_snap
    entry = o.wanted()[0]["entries"][0]
    ab = (entry.get("sb_explore") or {}).get("weights", {}).get("league/a", {})
    check(snap and snap["p"] == 0.5 and abs(ab.get("in", {}).get("Dead", 0) - 1 / 12) < 1e-6 and
          abs(ab["in"]["Pyroblast"] - 0.5) < 1e-6 and abs(ab["out"]["Lorien Revealed"] - 1 / 11) < 1e-6,
          "orchestrator: B4's schedule carries the counted weights (%s)" % ab)
    mets = [m for m in RD.read_jsonl(orun.metrics) if m["event"] == "sb_explore"]
    m = mets[-1] if mets else {}
    check(len(mets) == 1 and m.get("p") == 0.5 and m.get("explored_phases") == 11 and m.get("policy_phases") == 11
          and m.get("explored_hist", [])[1:3] == [10, 1] and m.get("policy_hist", [])[:2] == [10, 1] and
          abs(m.get("policy_swap_rate", 0) - 1 / 11) < 1e-9 and m.get("policy_hist_recent", [])[:2] == [10, 1],
          "orchestrator: an sb_explore metrics line with both histograms and the swap rate (%s)" % m)
    check(m.get("in_cells") == 3 and abs(m.get("in_tried_1", 0) - 2 / 3) < 1e-9 and
          abs(m.get("in_tried_10", 0) - 1 / 3) < 1e-9 and m.get("out_cells") == 3 and
          abs(m.get("out_tried_1", 0) - 2 / 3) < 1e-9 and abs(m.get("out_tried_10", 0) - 1 / 3) < 1e-9 and
          m.get("phases_min") == 0 and m.get("phases_median") == 5.5 and m.get("matchups") == 2,
          "orchestrator: the coverage cells from the deck files (Dead/Gone is Dead, Lórien folded) (%s)" % m)
    o.sb_tick()
    check(len([m for m in RD.read_jsonl(orun.metrics) if m["event"] == "sb_explore"]) == 1 and o.sb_snap is snap,
          "orchestrator: the snapshot is kept until it is due")
    o.state["rnad"]["learner_steps"] = 760
    o.sb_tick()
    check(o.sb_snap is not snap and abs(o.sb_snap["p"] - 0.292) < 1e-9, "orchestrator: p moved, a new snapshot")
    o.cfg = C.set_key(o.cfg, "sb_explore.coverage", False)
    o.sb_tick()
    check(not o.wanted()[0]["entries"][0]["sb_explore"]["weights"], "orchestrator: a set key, a new snapshot (uniform)")
    o.write_status()
    doc = RD.read_json(orun.status)
    text = format_status(doc)
    check(abs(doc["sb_explore"]["p"] - 0.292) < 1e-9 and "sb explore: p 0.292" in text and "11 explored phases" in text
          and "in-cells 3 tried >=1 67%" in text and "policy swap rate 0.091 (11 phases)" in text,
          "status: p_explore and the coverage line (%s)" % [x for x in text.splitlines() if "sb explore" in x])
    o.save_state()
    o2 = type(o).__new__(type(o))
    o2.sb_cov = SBX.Coverage(RD.read_json(orun.state).get("sb_explore"))
    check(o2.sb_cov.state() == o.sb_cov.state() and o2.sb_cov.explored, "state.json keeps the counts")
    o.state["phase"] = "B5"
    o.sb_tick()
    check(o.sb_snap["p"] == 0.0 and not any("sb_explore" in e for e in o.wanted()[0]["entries"]),
          "orchestrator: B5's schedule does not explore")
    o.state["phase"] = "B2"
    o.sb_tick()
    check(o.sb_snap is None, "orchestrator: no snapshot outside B4 and B5")
    o.metrics.close()
    o.evals_file.close()

    # The league deck files: every card reaches a key; Dead/Gone is the frames' Dead.
    decks = SBX.read_decks(os.path.join(REPO_ROOT, "bin"), ["league/" + x for x in C.LEAGUE])
    if decks:
        check(len(decks) == len(C.LEAGUE) and SBX.card_key("Dead") in decks["league/burn"][1] and
              all(all(k for k in m | sd) for m, sd in decks.values()), "the league deck files' card keys")

    # The engine's own schedule check, when an engine with sb_explore is built.
    eng = _engine_with_sb_explore(tmp)
    if eng is None:
        print("check_sb_explore: no engine with sb_explore built; the engine's schedule check skipped", flush=True)
        return
    rcfg = C.resolve(None, ["league.decks=[\"bug\", \"burn\", \"uw_control\"]", "actors.policy_eval=remote",
                            "sb_explore.p_start=0.25"])
    rcov = SBX.Coverage()
    rcov.add(_sb_line("b4", "league/burn", "league/bug", {"A": _sb_seat(True, 1, ["Dead"], ["Lightning Bolt"]),
                                                          "B": _sb_seat(True, 1, ["Pernicious Deed"], ["Ponder"])}))
    sched = SCH.selfplay(run, rcfg, "b4", SBX.snapshot(rcfg, rcov, SBX.p_explore(rcfg, "B4", 0)))
    path = os.path.join(tmp, "sbx-schedule.json")
    RD.write_json(path, sched)
    r = subprocess.run([eng, "--actor", "--actor-schedule", path, "--actor-out", os.path.join(tmp, "sbx-actor"),
                        "--actor-matches", "0"], cwd=os.path.join(REPO_ROOT, "bin"), capture_output=True, text=True,
                       timeout=60)
    check(r.returncode == 0 and all("sb_explore" in e for e in sched["entries"]),
          "the engine accepts the written self-play schedule (%s: %s)" % (eng, (r.stdout + r.stderr)[-300:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true", help="the end-to-end tiny run (GPU, engine)")
    ap.add_argument("--run", help="--smoke: the run directory (default runs/smoke-rmtrain)")
    ap.add_argument("--keep", action="store_true", help="--smoke: keep the run directory")
    ap.add_argument("--policy-eval", choices=("remote", "local"),
                    help="--smoke: the actors' and evaluations' policy_eval (actors.policy_eval, eval.policy_eval; "
                    "default: the config's default)")
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
            check_sideboarding_warm_start(tmp)
            check_t8(tmp)
            check_resume_and_evaluations(tmp)
            check_eval_options_and_findings(tmp)
            check_eval_scoring(tmp)
            check_benchmark(tmp)
            check_benchmark_sideboarding(tmp)
            check_local(tmp)
            check_sb_explore(tmp)
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
