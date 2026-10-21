"""The run directory (docs/rmtrain.md §4): its paths, atomic JSON files, the append-only JSON-lines
files and the short socket directory.

Everything durable lives under the run directory; the Unix sockets live under
``/tmp/rmtrain-<hash>/`` (a socket path must stay under 108 bytes), recorded in ``sockets.json``.
"""

import hashlib
import json
import os
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
# The run daemon's exit status when its run directory has no config.json (deleted): systemd does not
# restart it.
NO_RUN_EXIT = 3


def resolve_run(name):
    """The run directory of `name`: a path when it contains a slash (or is '.'), else runs/<name>."""
    if "/" in name or name in (".", ".."):
        return os.path.abspath(name)
    return os.path.join(RUNS_DIR, name)


def write_json(path, doc):
    """`doc` written whole: a temporary file renamed over `path` (fsynced)."""
    tmp = "%s.tmp%d.%d" % (path, os.getpid(), threading.get_ident() % 100000)
    with open(tmp, "w") as fp:
        json.dump(doc, fp, indent=1)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as fp:
            return json.load(fp)
    except (FileNotFoundError, ValueError):
        return default


class JsonLines:
    """An append-only JSON-lines file (one object per line, flushed per line; thread-safe)."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._fp = open(path, "a", buffering=1)

    def write(self, obj):
        line = json.dumps(obj, default=_jsonable)
        with self._lock:
            self._fp.write(line + "\n")
            self._fp.flush()

    def close(self):
        with self._lock:
            self._fp.close()


def _jsonable(x):
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
    except ImportError:
        pass
    return str(x)


def read_jsonl(path, last=None):
    """The objects of a JSON-lines file (the `last` ones only when given); a torn last line is skipped."""
    out = []
    try:
        with open(path) as fp:
            lines = fp.readlines()
    except FileNotFoundError:
        return out
    if last is not None:
        lines = lines[-last:]
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


class RunDir:
    """The paths of one run."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(self.path.rstrip("/"))
        p = lambda *a: os.path.join(self.path, *a)  # noqa: E731
        self.config = p("config.json")
        self.state = p("state.json")
        self.status = p("status.json")
        self.sockets = p("sockets.json")
        self.metrics = p("metrics.jsonl")
        self.evals = p("evals.jsonl")
        self.schedule = p("schedule.json")
        self.lock = p("orchestrator.lock")
        self.logs = p("logs")
        self.actor_logs = p("logs", "actors")
        self.actors = p("actors")
        self.episodes = p("episodes")
        self.rmlog = p("rmlog")
        self.ckpt = p("ckpt")
        self.fixed = p("ckpt", "fixed")
        self.exports = p("exports")
        self.eval = p("eval")
        self.tables = p("tables")
        self.cards = p("cards.rmstate")
        h = hashlib.blake2b(self.path.encode(), digest_size=5).hexdigest()
        self.sock_dir = "/tmp/rmtrain-%s" % h
        self.control_sock = os.path.join(self.sock_dir, "control.sock")
        self.rmip_sock = os.path.join(self.sock_dir, "rmip.sock")

    def log(self, component):
        if component.startswith("a") and component[1:].isdigit():
            return os.path.join(self.actor_logs, component + ".log")
        return os.path.join(self.logs, component + ".log")

    def episode_dir(self, phase):
        return os.path.join(self.episodes, phase)

    def rmlog_dir(self, phase):
        return os.path.join(self.rmlog, phase)

    def exists(self):
        return os.path.exists(self.config)

    def make(self):
        for d in (self.path, self.logs, self.actor_logs, self.actors, self.episodes, self.rmlog, self.ckpt,
                  self.fixed, self.exports, self.eval):
            os.makedirs(d, exist_ok=True)
        os.makedirs(self.sock_dir, mode=0o700, exist_ok=True)
        write_json(self.sockets, {"control": self.control_sock, "rmip": self.rmip_sock})


class Log:
    """A plain-text, line-buffered log file with timestamps (and an optional echo)."""

    def __init__(self, path, echo=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fp = open(path, "a", buffering=1)
        self._lock = threading.Lock()
        self.echo = echo

    def __call__(self, msg):
        line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        with self._lock:
            self._fp.write(line + "\n")
        if self.echo:
            self.echo(line)

    def close(self):
        self._fp.close()
