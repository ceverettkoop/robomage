"""The live exporter (docs/rmtrain.md §5, "Local evaluation"): one long-lived CPU process
(``exporter --serve``) that turns model checkpoints into the actors' exports, one at a time, so the
policy the local actors play (``actors.policy_eval = "local"``) follows the learner.

    live = LiveExporter(run, cfg, log)
    live.request("live", "exports/live/src-v120.pt", "exports/live/current.ts.pt", "ts", 120)  # coalesces
    live.request("eval", ck, out, "ts", 7)       # every request of another kind is kept
    for job in live.poll(): ...                  # finished jobs: {"kind", "checkpoint", "out", "version", "ok", ...}
    live.stop()

Requests of kind ``live`` coalesce: a new one replaces a queued (not yet started) one, whose source
checkpoint is deleted, so the exports follow the newest policy and never queue up. A ``live``
job's source checkpoint is deleted once exported (any job's with ``delete_source``). The process is (re)started on demand; a job it
dies in fails.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

from .actors import cpu_env
from .rundir import REPO_ROOT

LEARN_DIR = os.path.join(REPO_ROOT, "learn")


def export_path(directory, stem, fmt):
    return os.path.join(directory, stem + (".ts.pt" if fmt == "ts" else ".pt2"))


class LiveExporter:
    def __init__(self, run, cfg, log):
        self.run = run
        self.cfg = cfg
        self.log = log
        self.proc = None
        self.reader = None
        self.replies = queue.Queue()
        self.queue = []          # jobs not yet sent
        self.current = None      # the job in the process
        self.next_id = 0
        self.done = 0
        self.seconds = 0.0
        self.log_fp = None

    def tables(self):
        return self.cfg["run"]["tables"] or self.run.tables

    def _start(self):
        argv = [sys.executable, "-u", "-m", "rmlearn.orchestrator.exporter", "--tables", self.tables(), "--serve"]
        if self.log_fp is None:
            self.log_fp = open(self.run.log("export"), "a", buffering=1)
        self.log_fp.write("=== %s live exporter: %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(argv)))
        env = cpu_env()
        env["PYTHONPATH"] = LEARN_DIR
        self.proc = subprocess.Popen(argv, cwd=LEARN_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log_fp, env=env, start_new_session=True, text=True, bufsize=1)
        q = self.replies
        proc = self.proc

        def read():
            for line in proc.stdout:
                try:
                    q.put(json.loads(line))
                except ValueError:
                    continue
            q.put(None)         # the process is gone
        self.reader = threading.Thread(target=read, name="live-exporter", daemon=True)
        self.reader.start()

    def request(self, kind, checkpoint, out, fmt, version, delete_source=None, **extra):
        """Queues an export of `checkpoint` to `out`; a ``live`` request replaces a queued one. The
        source is deleted once exported with `delete_source` (default: for ``live`` requests)."""
        job = dict(extra, kind=kind, checkpoint=checkpoint, out=out, format=fmt, version=int(version),
                   queued_at=time.time(), delete_source=kind == "live" if delete_source is None else delete_source)
        if kind == "live":
            for old in [j for j in self.queue if j["kind"] == "live"]:
                self.queue.remove(old)
                _unlink(old["checkpoint"])
        self.queue.append(job)
        self._send()

    def pending(self, kind=None):
        jobs = self.queue + ([self.current] if self.current else [])
        return [j for j in jobs if kind is None or j["kind"] == kind]

    def _send(self):
        if self.current is not None or not self.queue:
            return
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        job = self.queue.pop(0)
        self.next_id += 1
        job["id"] = self.next_id
        job["started_at"] = time.time()
        try:
            self.proc.stdin.write(json.dumps({"id": job["id"], "checkpoint": job["checkpoint"], "out": job["out"],
                                              "format": job["format"], "version": job["version"]}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass                # the reader reports the process gone; the job fails then
        self.current = job

    def poll(self):
        """The jobs finished since the last poll (``ok``, ``seconds``, ``error``)."""
        out = []
        while True:
            try:
                msg = self.replies.get_nowait()
            except queue.Empty:
                break
            job, self.current = self.current, None
            if job is None:
                continue
            if msg is None:
                job.update(ok=False, error="the live exporter exited %s (see logs/export.log)" % self.proc.poll())
                self.proc = None
            else:
                job.update(ok=bool(msg.get("ok")), error=msg.get("error"), seconds=msg.get("seconds", 0.0))
            job["finished_at"] = time.time()
            if job["delete_source"]:
                _unlink(job["checkpoint"])
            if job["ok"]:
                self.done += 1
                self.seconds += float(job.get("seconds") or 0.0)
            out.append(job)
            self._send()
        if self.current is None:
            self._send()
        return out

    def stop(self):
        for j in self.queue:
            if j["delete_source"]:
                _unlink(j["checkpoint"])
        self.queue = []
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None
        if self.log_fp is not None:
            self.log_fp.close()
            self.log_fp = None


def _unlink(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
