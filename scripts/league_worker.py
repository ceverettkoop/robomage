#!/usr/bin/env python3
"""League worker supervisor: crash-restart loop + dashboard control agent.

Wraps one machine's `train.py league` shard for distributed training
(docs/distributed_league_training.md). Stdlib-only. Responsibilities:

  * Launch the league driver (appending `--shard i/n`, and `--resume` whenever
    that shard's sidecar already exists) and relaunch it with `--resume` after
    a crash, with a short backoff.
  * Heartbeat a small `_worker_status{tag}.json` into the (Syncthing-synced)
    checkpoint dir so the dashboard on the other machine can see liveness and
    the last log line without any network RPC.
  * Poll `_control{tag}.json` (written by scripts/league_dashboard.py and
    replicated by Syncthing) for stop / resume / restart commands, deduped by
    their `issued_at` stamp, which is echoed back as `last_command_ack`.

Usage (everything after `--` is the base league command, WITHOUT --shard/--resume):

  scripts/league_worker.py --shard 0/2 -- \\
      train/.venv/bin/python train/train.py league \\
      --binary bin/robomage --decks league/a,league/b,... --bo3 ...

Stop semantics: SIGINT to the trainer's process group (the driver plus its
SubprocVecEnv/engine children). The league sidecar is persisted at every
snapshot, so at most one snapshot interval of work is lost; `--resume` recovers
the rest.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

# Shard-tag / spec helpers live in the dependency-free train/cli_spec.py.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "train"))
from cli_spec import parse_shard, shard_tag  # noqa: E402

DEFAULT_CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "train", "checkpoints")
HEARTBEAT_EVERY = 30.0   # seconds between status-file writes
CONTROL_POLL = 3.0       # seconds between control-file reads
RESTART_BACKOFF = 10.0   # seconds before relaunching a crashed trainer
STOP_GRACE = 60.0        # seconds to wait after SIGINT before SIGKILL


def _atomic_write_json(path: str, obj: dict):
    """tmp + os.replace, same crash-safe pattern as train.py's sidecar writes."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"[worker] WARNING: could not write {path}: {exc}", file=sys.stderr)


def _read_json(path: str):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _install_stignore(checkpoint_dir: str):
    """Copy the Syncthing ignore template into the checkpoint dir if absent
    (the dir is gitignored, so the template can't live there in the repo)."""
    dst = os.path.join(checkpoint_dir, ".stignore")
    src = os.path.join(_REPO_ROOT, "scripts", "checkpoints.stignore")
    if not os.path.exists(dst) and os.path.exists(src):
        try:
            with open(src) as fh:
                content = fh.read()
            with open(dst, "w") as fh:
                fh.write(content)
            print(f"[worker] installed {dst}")
        except OSError as exc:
            print(f"[worker] WARNING: could not install .stignore: {exc}",
                  file=sys.stderr)


class Worker:
    def __init__(self, base_cmd: list[str], shard: str | None,
                 checkpoint_dir: str):
        parse_shard(shard)  # validate up front
        self.base_cmd = base_cmd
        self.shard = shard
        self.dir = checkpoint_dir
        tag = shard_tag(shard)
        self.status_path = os.path.join(checkpoint_dir, f"_worker_status{tag}.json")
        self.control_path = os.path.join(checkpoint_dir, f"_control{tag}.json")
        self.sidecar_path = os.path.join(checkpoint_dir, f"_league_progress{tag}.json")
        self.proc: subprocess.Popen | None = None
        self.state = "starting"          # running | stopped | crashed | done
        self.restarts = 0
        self.last_log_line = ""
        self.last_command_ack = None     # issued_at of the last handled command
        self._reader: threading.Thread | None = None

    # ── trainer process management ────────────────────────────────────────────
    def _cmd(self) -> list[str]:
        cmd = list(self.base_cmd)
        if self.shard:
            cmd += ["--shard", self.shard]
        if os.path.exists(self.sidecar_path):
            cmd += ["--resume"]
        return cmd

    def launch(self):
        cmd = self._cmd()
        print(f"[worker] launching: {' '.join(cmd)}")
        # New session => the driver and all its SubprocVecEnv/engine children
        # form one process group we can signal together.
        self.proc = subprocess.Popen(
            cmd, cwd=_REPO_ROOT, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.state = "running"
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self):
        """Echo the trainer's output and remember the last non-empty line."""
        proc = self.proc
        for line in proc.stdout:
            sys.stdout.write(line)
            stripped = line.strip()
            if stripped:
                self.last_log_line = stripped[:300]

    def terminate(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        pgid = os.getpgid(self.proc.pid)
        print("[worker] sending SIGINT to trainer process group")
        os.killpg(pgid, signal.SIGINT)
        deadline = time.monotonic() + STOP_GRACE
        while self.proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.5)
        if self.proc.poll() is None:
            print("[worker] trainer ignored SIGINT; SIGKILLing group")
            os.killpg(pgid, signal.SIGKILL)
            self.proc.wait()

    # ── coordination files ────────────────────────────────────────────────────
    def write_status(self):
        _atomic_write_json(self.status_path, {
            "hostname": socket.gethostname(),
            "pid": self.proc.pid if self.proc and self.proc.poll() is None else None,
            "shard": self.shard,
            "state": self.state,
            "written_at": time.time(),
            "last_log_line": self.last_log_line,
            "restarts": self.restarts,
            "last_command_ack": self.last_command_ack,
        })

    def poll_control(self) -> str | None:
        """Return a not-yet-handled command from the control file, if any."""
        ctl = _read_json(self.control_path)
        if not ctl:
            return None
        issued = ctl.get("issued_at")
        cmd = ctl.get("command")
        if issued is None or cmd not in ("stop", "resume", "restart"):
            return None
        if issued == self.last_command_ack:
            return None  # already handled
        self.last_command_ack = issued
        return cmd

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self) -> int:
        _install_stignore(self.dir)
        self.launch()
        next_heartbeat = 0.0
        crashed_at = None
        while True:
            now = time.monotonic()
            if now >= next_heartbeat:
                self.write_status()
                next_heartbeat = now + HEARTBEAT_EVERY

            cmd = self.poll_control()
            if cmd == "stop" and self.state in ("running", "crashed"):
                print("[worker] control: stop")
                self.terminate()
                self.state = "stopped"
                self.write_status()
            elif cmd == "resume" and self.state in ("stopped", "crashed"):
                print("[worker] control: resume")
                self.launch()
                self.write_status()
            elif cmd == "restart":
                print("[worker] control: restart")
                self.terminate()
                self.launch()
                self.write_status()

            if self.state == "running" and self.proc.poll() is not None:
                rc = self.proc.returncode
                if rc == 0:
                    # League budget complete — nothing left to supervise.
                    print("[worker] trainer finished cleanly; exiting")
                    self.state = "done"
                    self.write_status()
                    return 0
                print(f"[worker] trainer exited rc={rc}; "
                      f"restarting with --resume in {RESTART_BACKOFF:.0f}s")
                self.state = "crashed"
                self.write_status()
                crashed_at = now

            if self.state == "crashed" and crashed_at is not None \
                    and now - crashed_at >= RESTART_BACKOFF:
                self.restarts += 1
                crashed_at = None
                self.launch()
                self.write_status()

            time.sleep(CONTROL_POLL)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        split = argv.index("--")
        own, base_cmd = argv[:split], argv[split + 1:]
    else:
        own, base_cmd = argv, []
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Everything after '--' is the base league command "
               "(without --shard/--resume; the worker appends those).")
    ap.add_argument("--shard", default=None, metavar="i/n",
                    help="Roster shard this machine trains (matches train.py "
                         "league --shard). Omit for an unsharded run.")
    ap.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR,
                    help=f"Synced checkpoint dir (default {DEFAULT_CHECKPOINT_DIR})")
    args = ap.parse_args(own)
    if not base_cmd:
        ap.error("missing base league command after '--' "
                 "(e.g. -- train/.venv/bin/python train/train.py league ...)")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    worker = Worker(base_cmd, args.shard, args.checkpoint_dir)
    try:
        return worker.run()
    except KeyboardInterrupt:
        print("\n[worker] interrupted; stopping trainer")
        worker.terminate()
        worker.state = "stopped"
        worker.write_status()
        return 130


if __name__ == "__main__":
    sys.exit(main())
