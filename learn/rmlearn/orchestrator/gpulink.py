"""The orchestrator's side of the GPU process (gpu.py): start it, call its commands, read its events.

    gpu = GpuLink(run, cfg, log); gpu.start()           # waits for its "ready" event
    reply = gpu.call("bc_init", dirs=[...])              # GpuError on a failed command or a dead process
    for ev in gpu.events(): ...                          # bc_train, bc_eval, rnad_step, rnad_iteration, fatal
    gpu.stop()
"""

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time

from .rundir import REPO_ROOT


class GpuError(RuntimeError):
    pass


class GpuLink:
    def __init__(self, run, cfg, log):
        self.run = run
        self.cfg = cfg
        self.log = log
        self.proc = None
        self._id = 0
        self._lock = threading.Lock()
        self._waiting = {}
        self._events = queue.Queue()
        self._reader = None
        self.fatal = None
        self.started_at = None

    def env(self):
        env = dict(os.environ)
        hsa = self.cfg["run"]["hsa_override_gfx_version"]
        if hsa:
            env["HSA_OVERRIDE_GFX_VERSION"] = hsa
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.path.join(REPO_ROOT, "learn") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if self.cfg["run"]["device"] == "cpu":
            env.update({"HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": ""})
        else:
            for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
                if env.get(k) == "":            # hidden by the caller's environment: the GPU process needs it
                    del env[k]
        return env

    def start(self, timeout=None, on_wait=None):
        err = open(self.run.log("gpu"), "a", buffering=1)
        err.write("=== %s start\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
        self.proc = subprocess.Popen([sys.executable, "-u", "-m", "rmlearn.orchestrator.gpu", self.run.path],
                                     cwd=os.path.join(REPO_ROOT, "learn"), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=err, env=self.env(), start_new_session=True,
                                     text=True, bufsize=1)
        err.close()
        self.started_at = time.time()
        self._reader = threading.Thread(target=self._read, name="gpu-reader", daemon=True)
        self._reader.start()
        end = None if timeout is None else time.time() + timeout
        while True:
            try:
                ev = self._events.get(timeout=0.5)
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise GpuError("the GPU process exited %s before it was ready (see logs/gpu.log)"
                                   % self.proc.returncode)
                if end is not None and time.time() > end:
                    raise GpuError("the GPU process was not ready within %.0f s" % timeout)
                continue
            if ev.get("event") == "ready":
                return ev
            if ev.get("event") == "fatal":
                raise GpuError(ev.get("message", "fatal"))
            if ev.get("event") == "lock_wait":
                (on_wait or (lambda p: self.log("the GPU process waits for the GPU lock %s" % p)))(ev.get("path"))

    def _read(self):
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "re" in msg:
                with self._lock:
                    slot = self._waiting.pop(msg["re"], None)
                if slot is not None:
                    slot[1].append(msg)
                    slot[0].set()
                continue
            if msg.get("event") == "fatal":
                self.fatal = msg.get("message")
            self._events.put(msg)
        with self._lock:
            waiting, self._waiting = self._waiting, {}
        for ev, box in waiting.values():
            ev.set()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def call(self, cmd, timeout=None, **kw):
        """Runs a command in the GPU process; its reply (GpuError when it fails or the process is gone)."""
        if not self.alive():
            raise GpuError("the GPU process is not running")
        with self._lock:
            self._id += 1
            i = self._id
            ev, box = threading.Event(), []
            self._waiting[i] = (ev, box)
        try:
            self.proc.stdin.write(json.dumps(dict(kw, id=i, cmd=cmd)) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as err:
            raise GpuError("the GPU process is gone (%s)" % err)
        end = None if timeout is None else time.time() + timeout
        while not ev.wait(0.5):
            if not self.alive():
                raise GpuError("the GPU process exited %s during %s" % (self._code(), cmd))
            if end is not None and time.time() > end:
                with self._lock:
                    self._waiting.pop(i, None)
                raise GpuError("%s: no reply within %.0f s" % (cmd, timeout))
        if not box:
            raise GpuError("the GPU process exited %s during %s" % (self._code(), cmd))
        r = box[0]
        if not r.get("ok"):
            raise GpuError("%s: %s" % (cmd, r.get("error")))
        return r

    def _code(self):
        try:
            return self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return None

    def events(self):
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def stop(self, timeout=120):
        if self.proc is None:
            return
        if self.alive():
            try:
                self.call("stop", timeout=timeout)
            except GpuError as err:
                self.log("gpu stop: %s" % err)
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.log("the GPU process did not exit; killing it")
            self.kill()
        if self._reader is not None:
            self._reader.join(timeout=5)

    def kill(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()
