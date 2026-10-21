"""rmtrain: the training program's command line (docs/rmtrain.md; docs/rnad_training_plan.md §4 L6).

    learn/rmtrain start <run> [--config FILE] [--set section.key=value ...] [--foreground]
    learn/rmtrain status <run> [--json]
    learn/rmtrain tail <run> [orchestrator|gpu|export|metrics|evals|a<k>|actors|all] [-n N] [--no-follow]
    learn/rmtrain pause|resume|stop <run>
    learn/rmtrain set <run> <section.key> <value>
    learn/rmtrain eval <run>
    learn/rmtrain bench <run> [--wait] [--policies ...]   # the end-of-run benchmark (rmlearn/benchmark.py)
    learn/rmtrain phase <run> hold|release|next|status
    learn/rmtrain config <run>          # the resolved config (TOML)
    learn/rmtrain defaults              # every key with its default and documentation (TOML)

<run> is a name (``runs/<name>``) or a path. ``start`` on an existing run resumes it.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from .orchestrator import config as C
from .orchestrator.control import alive, request
from .orchestrator.rundir import NO_RUN_EXIT, REPO_ROOT, RunDir, read_json, resolve_run, write_json

LEARN_DIR = os.path.join(REPO_ROOT, "learn")
PASS_ENV = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH", "XDG_RUNTIME_DIR",
            "DBUS_SESSION_BUS_ADDRESS")
PASS_PREFIX = ("HSA_", "HIP_", "ROC", "PYTORCH_", "TORCH", "AMD_", "MIOPEN_")


def die(msg, code=1):
    print("rmtrain: " + msg, file=sys.stderr)
    sys.exit(code)


def unit_name(run):
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run.name)
    return "rmtrain-%s-%s" % (safe, os.path.basename(run.sock_dir).split("-")[-1])


# ── start ────────────────────────────────────────────────────────────────────────────────────

def prepare(run, args):
    """Writes (a new run) or updates (an existing one) the resolved config; returns it."""
    if run.exists():
        if args.config:
            die("run %s exists: its config is %s (change keys with --set, or `set` while it runs)"
                % (run.path, run.config))
        cfg, added = C.fill_defaults(read_json(run.config))
        try:
            cfg = C.absolute_paths(C.apply_sets(cfg, args.set))
            C.validate(cfg)
        except ValueError as err:
            die(str(err))
        if added:
            print("rmtrain: keys new since the run was created, at their defaults: %s" % ", ".join(added))
        if args.set or added:
            write_json(run.config, cfg)
        return cfg, False
    try:
        cfg = C.absolute_paths(C.resolve(C.load_file(args.config) if args.config else None, args.set))
    except (ValueError, OSError) as err:
        die(str(err))
    run.make()
    write_json(run.config, cfg)
    with open(os.path.join(run.path, "config.toml"), "w") as fp:
        fp.write("# The run's config as resolved at its creation (config.json is the live one).\n")
        fp.write(C.to_toml(cfg))
    return cfg, True


def cmd_start(args):
    run = RunDir(resolve_run(args.run))
    if run.exists() and alive(run.control_sock):
        die("run %s is already running" % run.name)
    if run.exists() and not args.foreground and _unit_active(unit_name(run)):
        die("run %s's unit %s is active (systemd is restarting it?); see rmtrain status" % (run.name, unit_name(run)))
    cfg, new = prepare(run, args)
    state = read_json(run.state) or {}
    if state.get("phase") == "done":
        die("run %s is finished (phase done)" % run.name)
    print("rmtrain: %s run %s (%s)%s" % ("new" if new else "resuming", run.name, run.path,
                                         ", phase %s" % state["phase"] if state.get("phase") else ""))
    if args.foreground:
        from .orchestrator.orchestrator import Orchestrator
        return Orchestrator(run, echo=True).main()
    return detach(run, cfg)


def detach(run, cfg):
    py = sys.executable
    env = {k: v for k, v in os.environ.items() if k in PASS_ENV or k.startswith(PASS_PREFIX)}
    env["PYTHONPATH"] = LEARN_DIR
    env["PYTHONUNBUFFERED"] = "1"
    argv = [py, "-u", "-m", "rmlearn.rmtrain", "_run", run.path]
    unit = unit_name(run)
    how = None
    if shutil.which("systemd-run") and _systemd_user_ok():
        subprocess.run(["systemctl", "--user", "reset-failed", unit + ".service"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        cmd = ["systemd-run", "--user", "--unit=" + unit, "--collect", "--quiet",
               "-p", "MemoryMax=" + cfg["run"]["memory_max"], "-p", "KillMode=mixed", "-p", "TimeoutStopSec=600",
               "-p", "Restart=" + ("on-failure" if cfg["run"]["restart_on_failure"] else "no"),
               "-p", "RestartSec=%ds" % cfg["run"]["restart_delay_s"], "-p", "StartLimitIntervalSec=0",
               "-p", "RestartPreventExitStatus=%d" % NO_RUN_EXIT,
               "--working-directory=" + LEARN_DIR, "--description=rmtrain run " + run.path]
        cmd += ["--setenv=%s=%s" % (k, v) for k, v in sorted(env.items())]
        r = subprocess.run(cmd + ["--"] + argv, capture_output=True, text=True)
        if r.returncode == 0:
            how = "systemd unit %s (MemoryMax=%s)" % (unit, cfg["run"]["memory_max"])
            write_json(os.path.join(run.path, "daemon.json"), {"kind": "systemd", "unit": unit, "at": time.time()})
        else:
            print("rmtrain: systemd-run failed (%s); falling back to a plain daemon" % r.stderr.strip(),
                  file=sys.stderr)
    if how is None:
        print("rmtrain: WARNING: systemd-run --user is unavailable: the run is a plain daemon with no MemoryMax "
              "and no supervision", file=sys.stderr)
        pid = double_fork(argv, env, os.path.join(run.logs, "daemon.out"))
        how = "daemon pid %d" % pid
        write_json(os.path.join(run.path, "daemon.json"), {"kind": "daemon", "pid": pid, "at": time.time()})
    end = time.time() + 120
    while time.time() < end:
        if alive(run.control_sock):
            print("rmtrain: run %s started as %s" % (run.name, how))
            print("rmtrain: logs: %s (rmtrain tail %s)" % (run.logs, run.name))
            try:
                print(format_status(request(run.control_sock, {"cmd": "status"}, 10)))
            except (ConnectionError, ValueError):
                pass
            return 0
        if how.startswith("systemd") and not _unit_active(unit):
            break
        time.sleep(0.5)
    print("rmtrain: the run did not come up; the end of its log:", file=sys.stderr)
    _print_tail(run.log("orchestrator"), 20)
    if how.startswith("systemd"):
        subprocess.run(["journalctl", "--user", "-u", unit, "-n", "20", "--no-pager"])
    return 1


def _systemd_user_ok():
    r = subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True, text=True)
    return r.stdout.strip() in ("running", "degraded", "starting")


def _unit_active(unit):
    r = subprocess.run(["systemctl", "--user", "is-active", unit + ".service"], capture_output=True, text=True)
    return r.stdout.strip() in ("active", "activating", "deactivating")


def double_fork(argv, env, out_path):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        os.setsid()
        pid2 = os.fork()
        if pid2:
            os.write(w, str(pid2).encode())
            os._exit(0)
        os.close(w)
        os.chdir(LEARN_DIR)
        fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        nul = os.open(os.devnull, os.O_RDONLY)
        os.dup2(nul, 0)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.execve(argv[0], argv, env)
    os.close(w)
    data = os.read(r, 64)
    os.close(r)
    os.waitpid(pid, 0)
    return int(data or 0)


# ── the other commands ───────────────────────────────────────────────────────────────────────

def existing(name):
    run = RunDir(resolve_run(name))
    if not run.exists():
        die("no run %s (%s)" % (name, run.path))
    return run


def call(run, req, timeout=180):
    if not alive(run.control_sock):
        die("run %s is not running (rmtrain start %s resumes it)" % (run.name, run.name))
    r = request(run.control_sock, req, timeout)
    if not r.get("ok"):
        die(r.get("error", "failed"))
    return r


def cmd_status(args):
    run = existing(args.run)
    if alive(run.control_sock):
        doc = request(run.control_sock, {"cmd": "status"}, 15)
    else:
        from .orchestrator.orchestrator import status_from_files
        doc = status_from_files(run)
    if args.json:
        print(json.dumps(doc, indent=1, default=str))
    else:
        print(format_status(doc))
    return 0


def cmd_simple(args):
    run = existing(args.run)
    r = call(run, {"cmd": args.cmd})
    if args.cmd == "stop":
        print("rmtrain: stopping %s (checkpoint, actors, GPU process)..." % run.name)
        end = time.time() + 900
        while time.time() < end and alive(run.control_sock, timeout=2):
            time.sleep(1)
        print("rmtrain: %s" % ("stopped" if not alive(run.control_sock) else "still stopping; see rmtrain tail"))
    else:
        print("rmtrain: %s: %s" % (args.cmd, json.dumps({k: v for k, v in r.items() if k != "ok"})))
    return 0


def cmd_set(args):
    run = existing(args.run)
    try:
        C.parse_live(args.key, args.value)
    except ValueError as err:
        die(str(err))
    r = call(run, {"cmd": "set", "key": args.key, "value": args.value})
    print("rmtrain: %s = %s (was %s)" % (r["key"], json.dumps(r["value"]), json.dumps(r["old"])))
    return 0


def cmd_eval(args):
    run = existing(args.run)
    req = {"cmd": "eval"}
    if args.evaluators:
        req["evaluators"] = args.evaluators.split(",")
    r = call(run, req)
    print("rmtrain: evaluation %s queued%s" % (r["queued"], " (running: %s)" % r["running"] if r.get("running")
                                                else ""))
    return 0


def cmd_bench(args):
    from . import benchmark as BM
    return BM.run_from_args(existing(args.run), args)


def cmd_phase(args):
    run = existing(args.run)
    if args.what == "status" and not alive(run.control_sock):
        st = read_json(run.state) or {}
        print("phase %s/%s, hold %s (not running)" % (st.get("phase"), st.get("stage"), st.get("hold")))
        return 0
    r = call(run, {"cmd": "phase", "what": args.what})
    print("phase %s%s%s; progress %s" % (r["phase"], "/" + r["stage"] if r["stage"] else "",
                                         " [hold%s]" % (": " + r["held_reason"] if r.get("held_reason") else "")
                                         if r["hold"] else "", json.dumps(r["progress"])))
    return 0


def cmd_tail(args):
    run = existing(args.run)
    comp = args.component or "orchestrator"
    if comp == "metrics":
        files = [run.metrics]
    elif comp == "evals":
        files = [run.evals]
    elif comp == "actors":
        files = sorted(os.path.join(run.actor_logs, f) for f in os.listdir(run.actor_logs))
    elif comp == "all":
        files = [run.log("orchestrator"), run.log("gpu"), run.metrics]
    else:
        files = [run.log(comp)]
    files = [f for f in files if os.path.exists(f)] or files
    if args.no_follow:
        for f in files:
            if len(files) > 1:
                print("==> %s <==" % f)
            _print_tail(f, args.n)
        return 0
    os.execvp("tail", ["tail", "-n", str(args.n), "-F"] + files)


def cmd_config(args):
    run = existing(args.run)
    print(C.to_toml(read_json(run.config)))
    return 0


def _print_tail(path, n):
    try:
        with open(path, errors="replace") as fp:
            lines = fp.readlines()[-n:]
    except FileNotFoundError:
        print("(no %s)" % path)
        return
    sys.stdout.write("".join(lines))


# ── the human-readable status ────────────────────────────────────────────────────────────────

def _dur(s):
    s = int(max(0, s or 0))
    if s >= 86400:
        return "%dd%02dh" % (s // 86400, s % 86400 // 3600)
    if s >= 3600:
        return "%dh%02dm" % (s // 3600, s % 3600 // 60)
    if s >= 60:
        return "%dm%02ds" % (s // 60, s % 60)
    return "%ds" % s


def _n(x, fmt="%.4g"):
    if x is None:
        return "-"
    if isinstance(x, float):
        return fmt % x
    return str(x)


def format_status(d):
    now = time.time()
    L = []
    if d.get("running"):
        head = "running, pid %s, up %s" % (d.get("pid"), _dur(d.get("uptime_s")))
    else:
        head = "NOT RUNNING (last status %s ago)" % _dur(now - d["updated_at"]) if d.get("updated_at") else \
            "NOT RUNNING"
        if d.get("finished"):
            head = "FINISHED, " + head
    L.append("run %s (%s): %s" % (d.get("run"), d.get("path"), head))
    flags = "".join([" [HOLD%s]" % (": " + d["held_reason"] if d.get("held_reason") else "") if d.get("hold") else "",
                     " [PAUSED]" if d.get("paused") else "", " [stopping: %s]" % d["stopping"]
                     if d.get("stopping") else ""])
    L.append("phase %s%s, for %s%s" % (d.get("phase"), "/" + d["stage"] if d.get("stage") else "",
                                       _dur(now - d.get("phase_started", now)), flags))
    if d.get("progress"):
        L.append("  progress: " + ", ".join("%s %s" % (k, _n(v)) for k, v in d["progress"].items()
                                             if not isinstance(v, (list, dict))))
    t = d.get("totals")
    if t:
        L.append("games %d (%.2f/s), decisions %d (%.0f/s), caps %d, draws %d" % (
            t.get("games", 0), t.get("games_per_s", 0), t.get("decisions", 0), t.get("decisions_per_s", 0),
            t.get("caps", 0), t.get("draws", 0)))
    a = d.get("actors")
    if a:
        bad = [h for h in a.get("health", []) if not h.get("alive")]
        L.append("actors %d/%d running%s, %d restarts%s" % (a.get("running", 0), a.get("wanted", 0),
                                                             " (waiting for %s)" % a["waiting_for"]
                                                             if a.get("waiting_for") else "", a.get("restarts", 0),
                                                           "; down: " + ", ".join("a%d (%s)" % (h["id"], (h.get(
                                                               "last_exit") or {}).get("reason") or "?")
                                                               for h in bad[:4]) if bad else ""))
    g = d.get("gpu") or {}
    if g and "error" not in g:
        sv = g.get("server", {})
        mem = g.get("gpu_memory", {})
        L.append("gpu: pid %s, mode %s%s; server %.0f rows/s, batch %s, queue p99 %s ms, %d clients, policies %s"
                 % (g.get("pid"), g.get("mode"), " (paused)" if g.get("paused") else "", sv.get("rows_per_s", 0),
                    _n(sv.get("batch_mean"), "%.1f"), _n(sv.get("queue_ms_p99"), "%.1f"), sv.get("clients", 0),
                    sv.get("policies")))
        if mem:
            L.append("     memory %.0f / %.0f MB used (torch %.0f MB, peak %.0f MB)" % (
                mem.get("used_mb", 0), mem.get("total_mb", 0), mem.get("allocated_mb", 0), mem.get("peak_mb", 0)))
        w = g.get("window")
        if w:
            L.append("     window: %d episodes, %d decisions (%d labelled), %.2f GB, %d fresh, %d evicted"
                     % (w.get("episodes", 0), w.get("decisions", 0), w.get("labelled", 0), w.get("bytes", 0) / 2 ** 30,
                        w.get("fresh", 0), w.get("evicted", 0)))
    elif g.get("error"):
        L.append("gpu: %s" % g["error"])
    lv = d.get("live")
    if lv:
        last = lv.get("last") or {}
        L.append("live export (local actors): %s; version %s, %s ago, %s s to export, %s exports in 10 min%s" % (
            "in place" if lv.get("ok") else "WAITING (the actors start once it is in place)", last.get("version", "-"),
            _dur(now - last["at"]) if last.get("at") else "-", _n(last.get("seconds"), "%.2f"),
            lv.get("exports_10min", 0), "; queued %s" % lv["queued"] if lv.get("queued") else ""))
    m = d.get("metrics") or {}
    for key, fields in (("bc_train", ("train_policy", "train_agree", "train_value", "lr", "decisions_per_s")),
                        ("bc_eval", ("eval_agree_all", "eval_agree_group:in_game", "eval_value_mse")),
                        ("rnad_step", ("loss", "loss_v", "loss_nerd", "kl_reg", "entropy", "alpha", "policy_lag",
                                       "policy_lag_p95", "decisions_per_s", "wait_fraction", "reg_recomputed"))):
        if key in m:
            e = m[key]
            L.append("%s @ step %s, %s ago: %s" % (key, e.get("step"), _dur(now - e.get("time", now)),
                                                   ", ".join("%s %s" % (f, _n(e.get(f))) for f in fields if f in e)))
            if key == "rnad_step" and "value_mse_first_sight" in e:
                L.append("  value MSE: first sight %s, after the update %s, gap %s" % (
                    _n(e.get("value_mse_first_sight")), _n(e.get("value_mse_after")), _n(e.get("value_mse_gap"))))
    ev = d.get("eval") or {}
    if ev.get("running") or ev.get("queue"):
        L.append("eval: running %s, queued %s" % (ev.get("running"), ev.get("queue")))
    if ev.get("last"):
        last = ev["last"]
        from .orchestrator.evaluation import options_label
        L.append("last eval %s (%s, %s, version %s, options %s)%s" % (
            last.get("label"), last.get("phase"), last.get("subject") or "-", last.get("version"),
            options_label(last.get("options")), ": ERROR %s" % last["error"] if last.get("error") else ":"))
        for k, v in (last.get("brief") or {}).items():
            if k != "error":
                L.append("  %-16s %s" % (k, v))
    if ev.get("exploiter"):
        x = ev["exploiter"]
        L.append("exploiter: %s, version %s, %s games, rolling win rate %s" % (
            x.get("phase"), x.get("version"), x.get("games"), _n(x.get("recent_win_rate"))))
    if ev.get("ratings"):
        top = sorted(ev["ratings"].items(), key=lambda kv: -kv[1]["elo"])[:8]
        L.append("ratings (Bradley-Terry, Elo scale): " + ", ".join("%s %+.0f (%d g)" % (k, r["elo"], r["games"])
                                                                   for k, r in top))
    fd = d.get("findings") or {}
    if fd.get("draws"):
        L.append("findings: %d draw%s (CR 104.4a, to review, not errors); latest: %s" % (
            fd["draws"], "s" if fd["draws"] != 1 else "", "; ".join(f["text"] for f in fd.get("recent", [])[-3:])))
    ck = d.get("checkpoint")
    if ck:
        L.append("checkpoint %s, %s ago" % (ck.get("path"), _dur(now - ck.get("at", now))))
    for tr in (d.get("transitions") or [])[-3:]:
        L.append("  %s -> %s (%s), %s ago" % (tr["from"], tr["to"], tr["reason"], _dur(now - tr["at"])))
    for e in d.get("recent_metrics") or []:
        L.append("  metrics: %s %s @ step %s" % (e.get("event"), e.get("phase"), e.get("step")))
    return "\n".join(L)


# ── main ─────────────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_run"]:
        from .orchestrator.orchestrator import Orchestrator
        return Orchestrator(RunDir(argv[1])).main()
    ap = argparse.ArgumentParser(prog="rmtrain", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start", help="start (or resume) a run")
    p.add_argument("run")
    p.add_argument("--config", help="a TOML or JSON config (a new run only)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--foreground", action="store_true", help="run in this terminal (tests)")
    p = sub.add_parser("status", help="the run's status (from its files when it is not running)")
    p.add_argument("run")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("tail", help="follow a log")
    p.add_argument("run")
    p.add_argument("component", nargs="?", help="orchestrator (default), gpu, export, metrics, evals, a<k>, actors, all")
    p.add_argument("-n", type=int, default=40)
    p.add_argument("--no-follow", action="store_true")
    for name, hlp in (("pause", "actors stop, the learner stops stepping"), ("resume", "undo pause"),
                      ("stop", "checkpoint and stop")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("run")
    p = sub.add_parser("set", help="change a live parameter")
    p.add_argument("run")
    p.add_argument("key")
    p.add_argument("value")
    p = sub.add_parser("eval", help="queue an evaluation now")
    p.add_argument("run")
    p.add_argument("--evaluators", help="comma-separated (default eval.evaluators)")
    p = sub.add_parser("bench", help="the end-of-run benchmark: the run's policies vs scripted:hard, per deck")
    p.add_argument("run")
    from .benchmark import add_args as bench_args
    bench_args(p)
    p = sub.add_parser("phase", help="hold, release or force the phase transition")
    p.add_argument("run")
    p.add_argument("what", choices=("hold", "release", "next", "status"))
    p = sub.add_parser("config", help="print the run's resolved config")
    p.add_argument("run")
    sub.add_parser("defaults", help="print the default config")
    args = ap.parse_args(argv)
    if args.cmd == "start":
        return cmd_start(args)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd in ("pause", "resume", "stop"):
        return cmd_simple(args)
    if args.cmd == "set":
        return cmd_set(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "bench":
        return cmd_bench(args)
    if args.cmd == "phase":
        return cmd_phase(args)
    if args.cmd == "tail":
        return cmd_tail(args)
    if args.cmd == "config":
        return cmd_config(args)
    if args.cmd == "defaults":
        print(C.defaults_toml())
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
