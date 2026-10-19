"""The actor pool and the actors' results (docs/rmtrain.md §5; docs/actor.md).

    pool = ActorPool(engine, bin_dir, schedule, out_dir, log_dir, seed, settings, log)
    pool.set_count(26)        # starts / stops actors to reach the count
    pool.tick()               # reaps exits, restarts crashed actors (with a crash-loop back-off)
    pool.stop_all()           # SIGTERM, then SIGKILL after the timeout
    tail = ResultsTail(out_dir); for line in tail.poll(): ...

An actor is ``robomage_nn --actor`` (the GPU hidden from it), in its own process group (its
teachers with it), its stdout and stderr appended to ``<log_dir>/a<k>.log``. Its first match index
comes from its own files: the status file's ``next_match`` (+1 when a match was in progress, which
is thereby written off) and never below the results file's last match + 1, so a restarted actor
(after a crash, a ``kill -9`` of the run or a stop) never replays a match index it used.
"""

import collections
import glob
import json
import os
import signal
import subprocess
import time

GPU_HIDDEN = {"HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""}


def cpu_env():
    env = dict(os.environ)
    env.update(GPU_HIDDEN)
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    return env


def _read_json(path):
    try:
        with open(path) as fp:
            return json.load(fp)
    except (FileNotFoundError, ValueError):
        return None


def last_match_in_results(path):
    """The largest match index in an actor's results file (-1 when none)."""
    last = -1
    try:
        with open(path, "rb") as fp:
            fp.seek(0, 2)
            size = fp.tell()
            fp.seek(max(0, size - 65536))
            tail = fp.read().decode(errors="replace").splitlines()
    except FileNotFoundError:
        return last
    for ln in tail:
        try:
            last = max(last, int(json.loads(ln)["match"]))
        except (ValueError, KeyError, TypeError):
            continue
    return last


def first_match(out_dir, k):
    """Actor k's next match index from its status and results files (0 for a new actor)."""
    st = _read_json(os.path.join(out_dir, "a%d.status.json" % k))
    start = 0
    if st:
        start = int(st.get("next_match", 0))
        if st.get("current"):
            start += 1          # the match in progress is written off
    return max(start, last_match_in_results(os.path.join(out_dir, "a%d.results.jsonl" % k)) + 1)


class _Actor:
    def __init__(self, k):
        self.k = k
        self.proc = None
        self.log = None
        self.started_at = None
        self.start_match = None
        self.restart_at = None      # a pending restart's time
        self.crashes = collections.deque()
        self.restarts = 0
        self.last_exit = None
        self.stopping = False


class ActorPool:
    """Training actors (or, with `matches`, a fixed batch of evaluation actors that are not restarted)."""

    def __init__(self, engine, bin_dir, schedule, out_dir, log_dir, seed, settings, log, matches=None, prefix="a"):
        self.engine = engine
        self.bin_dir = bin_dir
        self.schedule = schedule
        self.out_dir = out_dir
        self.log_dir = log_dir
        self.seed = int(seed)
        self.settings = settings        # the [actors] config section (read live)
        self.log = log
        self.matches = matches          # {k: matches} for an evaluation batch
        self.prefix = prefix
        self.actors = {}
        self.count = 0
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    # ── processes ──

    def _start(self, a):
        start = first_match(self.out_dir, a.k)
        argv = [self.engine, "--actor", "--actor-schedule", self.schedule, "--actor-out", self.out_dir,
                "--actor-id", str(a.k), "--actor-seed", str(self.seed), "--actor-start", str(start)]
        if self.matches is not None:
            argv += ["--actor-matches", str(self.matches[a.k])]
        if a.log is None:
            a.log = open(os.path.join(self.log_dir, "%s%d.log" % (self.prefix, a.k)), "a", buffering=1)
        a.log.write("=== %s start match %d: %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), start, " ".join(argv)))
        a.proc = subprocess.Popen(argv, cwd=self.bin_dir, stdin=subprocess.DEVNULL, stdout=a.log,
                                  stderr=subprocess.STDOUT, env=cpu_env(), start_new_session=True)
        a.started_at = time.time()
        a.start_match = start
        a.restart_at = None
        a.stopping = False

    def set_count(self, n):
        """Runs actors 0..n-1: starts the missing ones, stops the others."""
        self.count = n
        for k in range(n):
            a = self.actors.get(k)
            if a is None:
                a = self.actors[k] = _Actor(k)
            if a.proc is None and a.restart_at is None:
                self._start(a)
        for k, a in list(self.actors.items()):
            if k >= n:
                if a.proc is not None and a.proc.poll() is None:
                    a.stopping = True
                    a.proc.send_signal(signal.SIGTERM)
                a.restart_at = None

    def tick(self):
        """Reaps exits; schedules and performs restarts. Returns [(k, returncode)] of crashes seen."""
        now = time.time()
        crashes = []
        s = self.settings
        for k, a in list(self.actors.items()):
            if a.proc is not None and a.proc.poll() is not None:
                rc = a.proc.returncode
                a.last_exit = {"code": rc, "at": now, "reason": self._exit_reason(k)}
                a.proc = None
                a.log.write("=== %s exit %s%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), rc,
                                                    " (%s)" % a.last_exit["reason"] if a.last_exit["reason"] else ""))
                if a.stopping or k >= self.count:
                    a.stopping = False
                    continue
                if self.matches is not None and rc == 0:
                    continue        # an evaluation actor that finished
                crashes.append((k, rc))
                if self.matches is not None:
                    continue        # evaluation actors are not restarted
                a.crashes.append(now)
                while a.crashes and now - a.crashes[0] > s["crash_loop_window_s"]:
                    a.crashes.popleft()
                delay = s["restart_delay_s"]
                over = len(a.crashes) - s["crash_loop_restarts"]
                if over >= 0:
                    delay = min(s["max_backoff_s"], max(delay, 1.0) * 2 ** (over + 1))
                a.restart_at = now + delay
                self.log("actor %s%d exited %s (%s); restart in %.2f s (%d crashes in the window)"
                         % (self.prefix, k, rc, a.last_exit["reason"] or "no status", delay, len(a.crashes)))
            if a.proc is None and a.restart_at is not None and now >= a.restart_at and k < self.count:
                a.restarts += 1
                self._start(a)
        return crashes

    def _exit_reason(self, k):
        st = _read_json(os.path.join(self.out_dir, "a%d.status.json" % k)) or {}
        return st.get("exit_reason") or st.get("last_error") or ""

    def running(self):
        return sum(1 for a in self.actors.values() if a.proc is not None and a.proc.poll() is None)

    def finished(self):
        """An evaluation batch: every actor has exited."""
        return all(a.proc is None or a.proc.poll() is not None for a in self.actors.values())

    def stop_all(self, timeout=None):
        """SIGTERM to every actor, SIGKILL to its process group after `timeout` seconds."""
        timeout = self.settings["stop_timeout_s"] if timeout is None else timeout
        self.count = 0
        live = []
        for a in self.actors.values():
            a.restart_at = None
            if a.proc is not None and a.proc.poll() is None:
                a.stopping = True
                try:
                    a.proc.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
                live.append(a)
        end = time.time() + timeout
        for a in live:
            try:
                a.proc.wait(timeout=max(0.1, end - time.time()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(a.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                a.proc.wait()
        self.tick()
        for a in self.actors.values():
            if a.log is not None:
                a.log.close()
                a.log = None
        self.actors = {}

    def kill_all(self):
        for a in self.actors.values():
            if a.proc is not None and a.proc.poll() is None:
                try:
                    os.killpg(a.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def health(self):
        """Per actor: id, pid, alive, restarts, the status file's state and rates."""
        out = []
        for k in sorted(self.actors):
            a = self.actors[k]
            st = _read_json(os.path.join(self.out_dir, "a%d.status.json" % k)) or {}
            out.append({"id": k, "pid": a.proc.pid if a.proc is not None else None,
                        "alive": a.proc is not None and a.proc.poll() is None, "restarts": a.restarts,
                        "recent_crashes": len(a.crashes), "restart_at": a.restart_at, "state": st.get("state"),
                        "matches": st.get("matches"), "games": st.get("games"), "decisions": st.get("decisions"),
                        "decisions_per_s": st.get("recent_decisions_per_s"), "caps": st.get("caps"),
                        "draws": st.get("draws"), "next_match": st.get("next_match"),
                        "last_error": st.get("last_error") or "", "last_exit": a.last_exit})
        return out


class ResultsTail:
    """Reads the new lines of every ``a*.results.jsonl`` of a directory (offsets kept; a torn last
    line is left for the next poll)."""

    def __init__(self, out_dir, offsets=None):
        self.out_dir = out_dir
        self.offsets = dict(offsets or {})

    def poll(self):
        out = []
        for path in sorted(glob.glob(os.path.join(self.out_dir, "a*.results.jsonl"))):
            name = os.path.basename(path)
            off = self.offsets.get(name, 0)
            try:
                size = os.path.getsize(path)
            except FileNotFoundError:
                continue
            if size < off:
                off = 0
            if size == off:
                continue
            with open(path, "rb") as fp:
                fp.seek(off)
                data = fp.read()
            end = data.rfind(b"\n") + 1
            for ln in data[:end].splitlines():
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    continue
            self.offsets[name] = off + end
        return out


def draw_finding(line, source, results_file=None, base=None):
    """A drawn game's finding (None for any other game): what played it (`source`: ``train`` or the
    evaluation), the decks and game, the results file and the match's decision log (paths shown
    relative to `base`). A draw is a finding to review, not an error (CR 104.4a: both players lost at
    once is a legitimate end)."""
    if line.get("end_reason") != "draw":
        return None
    results_file = results_file or line.get("results_file")

    def rel(p):
        return os.path.relpath(p, base) if p and base and os.path.isabs(p) and p.startswith(base) else p

    def short(d):
        return str(d or "?").rsplit("/", 1)[-1]

    f = {"at": time.time(), "source": source, "schedule": line.get("schedule"), "entry": line.get("entry"),
         "actor": line.get("actor"), "match": line.get("match"), "game": line.get("game"),
         "decks": "%s vs %s" % (short(line.get("deck_a")), short(line.get("deck_b"))),
         "end_reason": "draw", "results": results_file, "rmlog": line.get("rmlog")}
    f["text"] = "%s a%s m%s g%s %s: %s (results %s, log %s)" % (
        source, f["actor"], f["match"], f["game"], f["decks"], f["end_reason"], rel(results_file) or "?",
        rel(f["rmlog"]) or "?")
    return f


class Counters:
    """Games and decisions per schedule version, plus rates over a sliding window."""

    def __init__(self, state=None):
        state = state or {}
        self.by_version = state.get("by_version", {})
        self.total = state.get("total", {"games": 0, "decisions": 0, "caps": 0, "draws": 0})
        self._hist = collections.deque()

    def add(self, line):
        v = str(line.get("schedule", ""))
        c = self.by_version.setdefault(v, {"games": 0, "decisions": 0})
        d = int(line.get("decisions", 0))
        c["games"] += 1
        c["decisions"] += d
        self.total["games"] += 1
        self.total["decisions"] += d
        self.total["caps"] += line.get("end_reason") == "cap"
        self.total["draws"] += line.get("end_reason") == "draw"

    def sample(self):
        now = time.time()
        self._hist.append((now, self.total["games"], self.total["decisions"]))
        while len(self._hist) > 2 and now - self._hist[0][0] > 60:
            self._hist.popleft()

    def rates(self):
        if len(self._hist) < 2:
            return {"games_per_s": 0.0, "decisions_per_s": 0.0}
        (t0, g0, d0), (t1, g1, d1) = self._hist[0], self._hist[-1]
        dt = max(t1 - t0, 1e-6)
        return {"games_per_s": (g1 - g0) / dt, "decisions_per_s": (d1 - d0) / dt}

    def of(self, version):
        return self.by_version.get(str(version), {"games": 0, "decisions": 0})

    def state(self):
        return {"by_version": self.by_version, "total": self.total}
