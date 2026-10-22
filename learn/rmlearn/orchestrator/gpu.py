"""The run's GPU process (docs/rmtrain.md §5): the learner and the inference server in one process,
the only one of the run that touches the GPU (docs/rnad_training_plan.md §4 L5, §7 decision 3).

    python -m rmlearn.orchestrator.gpu RUN_DIR          # started by the orchestrator, never by hand

It takes the GPU lock (``run.gpu_lock``, an flock: fails at once, or waits with
``run.gpu_lock_wait``), loads the run's tables, starts the RMIP server on the run's socket and then
obeys the orchestrator: JSON-line commands on stdin (``{"id": n, "cmd": NAME, ...}``), replies
(``{"re": n, "ok": ...}``) and events (``{"event": NAME, ...}``) on its protocol output (the
original stdout; fd 1 is redirected to stderr, the run's ``logs/gpu.log``, so nothing a library prints
can corrupt the protocol). End of input (the orchestrator gone) ends the process; SIGTERM and SIGINT
are ignored (the orchestrator stops it in order, after a checkpoint).

**Modes.** ``idle``; ``bc``: behaviour cloning (``bc.BCTrainer``) on a replay window over the given
episode directories, in units of ``bc.unit_steps`` steps, a ``bc_train`` event every
``bc.report_every`` steps and a ``bc_eval`` event (held-out agreement) every ``bc.eval_every``;
``rnad``: R-NaD (``rnad_learner.RNaDLearner``) on fresh self-play games (``take_fresh``,
``trainable_episodes``), ``rnad.games_per_step`` games a step, an ``rnad_step`` event every
``rnad.report_every`` steps, an ``rnad_iteration`` event at each iteration boundary (the new
π_m,reg saved as a fixed point under ``ckpt/fixed/``). ``pause`` stops stepping, ``resume`` goes on.

**Local actors** (``actors.policy_eval = "local"``): in ``rnad`` mode the current network is saved
(``model.save_checkpoint``, ``exports/live/src-v<step>.pt``) at once and then every
``actors.export_every_s`` seconds or ``actors.export_every_steps`` learner steps when the learner
stepped, each announced by a ``policy_saved`` event; the orchestrator's live exporter turns it into
the actors' export. ``save_model`` saves the current network on request (DAgger's round start, an
evaluation's snapshot).

**Served policies** (``InferenceServer``): 0 the current policy (BC: pushed by ``serve``; R-NaD:
``ServerLink``, every ``rnad.push_every`` steps), 1 and 2 the regularisation policies (R-NaD), 3 the
evaluation snapshot (``snapshot``), 10+ the evaluation ladder (``load_policy``), 20 and 21 an
exploiter and its victim.

**Exploiter** (``exploit_start`` / ``exploit_status`` / ``exploit_finish`` / ``exploit_cancel``;
``rmlearn/exploiter.py``): an ``ExploiterSession`` against the current policy, ticked once per loop
iteration between the learner's units (its PPO updates on this GPU); with ``pause_learner`` the
learner idles until it finishes, so the two never compete for the GPU.
"""

import fcntl
import json
import os
import queue
import signal
import sys
import threading
import time
import traceback

PROTO_FD = None


def _protocol_out():
    """The protocol stream: the original stdout, with fd 1 sent to stderr from here on."""
    global PROTO_FD
    PROTO_FD = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(PROTO_FD, "w", buffering=1)


def log(msg):
    print("%s gpu: %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg), file=sys.stderr, flush=True)


class Worker:
    def __init__(self, run_path, out):
        from .config import fill_defaults
        from .rundir import RunDir, read_json
        self.rd = RunDir(run_path)
        self.cfg = read_json(self.rd.config)
        if self.cfg is None:
            raise SystemExit("gpu: %s has no config.json" % self.rd.path)
        self.cfg = fill_defaults(self.cfg)[0]
        self.out = out
        self.out_lock = threading.Lock()
        self.cmds = queue.Queue()
        self.mode = "idle"
        self.paused = False
        self.started = time.time()
        self.lock_fp = None
        self.server = None
        self.tables = None
        self.trainer = None
        self.bc_window = None
        self.learner = None
        self.link = None
        self.window = None
        self.pending = []
        self.dropped = {}
        self.last_error = ""
        self.agg = None
        self.last_bc_eval = 0
        self.eof = False
        self.exploit = None             # an exploiter.ExploiterSession (exploit_start … exploit_finish)
        self.exploit_pause = False      # the learner idles while it trains
        self.published = None           # (learner step, time) of the last policy saved for the live export
        self.lags = []                  # the report window's per-decision policy lags
        self.flushed_at = None          # the end of the last metrics window

    # ── protocol ──

    def send(self, obj):
        line = json.dumps(obj, default=_jsonable)
        with self.out_lock:
            try:
                self.out.write(line + "\n")
                self.out.flush()
            except (BrokenPipeError, ValueError):
                self.eof = True

    def event(self, name, **kw):
        kw["event"] = name
        kw["time"] = time.time()
        self.send(kw)

    def reader(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                log("bad command line: %r" % line[:200])
                continue
            if msg.get("cmd") == "stats":
                try:
                    self.send({"re": msg.get("id"), "ok": True, **self.stats()})
                except Exception as err:  # a reply, not a crash
                    self.send({"re": msg.get("id"), "ok": False, "error": repr(err)})
                continue
            self.cmds.put(msg)
        self.eof = True
        self.cmds.put({"cmd": "_eof"})

    # ── setup ──

    def take_lock(self):
        path = self.cfg["run"]["gpu_lock"]
        if not path:
            return
        self.lock_fp = open(path, "a+")
        try:
            fcntl.flock(self.lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not self.cfg["run"]["gpu_lock_wait"]:
                raise SystemExit("gpu: the GPU lock %s is held by another process (another run or GPU job); stop it, "
                                 "or set run.gpu_lock_wait = true to wait for it" % path)
            log("waiting for the GPU lock %s (held by another process)" % path)
            self.event("lock_wait", path=path)
            fcntl.flock(self.lock_fp, fcntl.LOCK_EX)
        log("holding the GPU lock %s" % path)

    def setup(self):
        import torch
        from .. import spec as S
        from ..server import InferenceServer
        c = self.cfg
        torch.set_num_threads(max(1, int(c["run"]["torch_threads"])))
        dev = c["run"]["device"]
        if dev.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("gpu: run.device is %s but torch sees no GPU" % dev)
        self.device = dev
        tdir = c["run"]["tables"] or self.rd.tables
        self.tables = S.load_tables(tdir)
        sv = c["server"]
        self.server = InferenceServer(self.rd.rmip_sock, device=dev, max_batch=sv["max_batch"],
                                      deadline_us=sv["deadline_us"], precision=sv["precision"], name="rmtrain",
                                      log=lambda m: log("server: " + m))
        self.server.start()
        log("ready: device %s, tables %s, server %s" % (dev, tdir, self.rd.rmip_sock))

    # ── BC ──

    def _window_cfg(self, dirs, selfplay):
        from .. import replay as RP
        r = self.cfg["replay"]
        return RP.WindowConfig(dirs=list(dirs), max_decisions=r["max_decisions"], max_bytes=int(r["max_gb"] * (1 << 30)),
                               delete_evicted=bool(selfplay and r["delete_evicted_selfplay"]),
                               poll_interval=r["poll_interval_s"], spec_hash=self.tables.spec.hash,
                               heldout_fraction=0.0 if selfplay else r["heldout_fraction"],
                               split_seed=self.cfg["run"]["seed"], max_decision_tokens=r["max_decision_tokens"])

    def bc_init(self, dirs, resume=None):
        import torch
        from .. import bc as BC
        from .. import model as M
        from .. import replay as RP
        c = self.cfg
        torch.manual_seed(c["run"]["seed"])
        if resume:
            model = M.load_checkpoint(resume, self.tables)
        else:
            model = M.RMNet(M.ModelConfig.from_dict(c["model"]), self.tables)
        win = RP.ReplayWindow(self._window_cfg(dirs, False), log=lambda m: log("bc window: " + m))
        skipped = restore_recent(win, dirs, win.cfg.max_bytes)
        win.start()
        names = BC.category_names(self.rd.cards) if os.path.exists(self.rd.cards) else None
        cats = BC.Categories(self.tables.spec, names)
        trainer = BC.BCTrainer(model, win, BC.BCConfig.from_dict(c["bc"]), device=self.device, categories=cats)
        if resume:
            trainer.restore(resume)
        self.bc_window, self.trainer = win, trainer
        self.last_bc_eval = trainer.step
        log("bc: %s model (%d parameters), window over %s (%d older files left out)"
            % ("resumed" if resume else "new", model.param_count(), ", ".join(dirs), skipped))
        return {"step": trainer.step, "params": model.param_count()}

    def bc_unit(self):
        st = self.bc_window.stats()
        if not st["labelled"] or not st["train_episodes"]:
            time.sleep(0.5)
            return
        c = self.cfg["bc"]
        m = self.trainer.train(int(c["unit_steps"]))
        self._aggregate(m, ("step", "lr", "decisions", "amp_scale"), ("seconds", "skipped_steps"))
        if self.agg["n"] >= c["report_every"]:
            self.event("bc_train", **self._flush(m["step"]), window_skipped_oversize=st.get("skipped_oversize", 0))
        if self.trainer.step - self.last_bc_eval >= c["eval_every"]:
            self.bc_evaluate(event=True)

    def bc_evaluate(self, event=False):
        e = self.trainer.evaluate()
        e["window"] = self.bc_window.stats()
        self.last_bc_eval = self.trainer.step
        if event:
            self.event("bc_eval", **e)
        return e

    def bc_close(self):
        import torch
        if self.mode == "bc":
            self.mode = "idle"
        if self.trainer is not None:
            self.trainer.close()
            self.trainer = None
        if self.bc_window is not None:
            self.bc_window.stop()
            self.bc_window = None
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ── R-NaD ──

    def rnad_config(self):
        from ..rnad_learner import RNaDConfig
        return RNaDConfig.from_dict(dict(self.cfg["rnad"]))

    def rnad_init(self, bc_checkpoint=None, resume=None):
        import torch
        from .. import replay as RP
        from ..rnad_learner import RNaDLearner, ServerLink
        self.bc_close()
        c = self.cfg
        saved_window = None
        if resume:
            data = torch.load(resume, map_location="cpu", weights_only=False)
            learner = RNaDLearner.from_state_dict(data["learner"], self.tables, device=self.device)
            learner.configure(**{k: v for k, v in self.rnad_config().to_dict().items()
                                 if k in ("eta", "learning_rate", "entropy_schedule_sizes", "entropy_schedule_repeats",
                                          "reg_eps", "reg_eps_sideboarding", "finetune_threshold",
                                          "finetune_discretization", "stale_reg", "forward_tokens",
                                          "value_after_every")})
            saved_window = data.get("window")
        else:
            learner = RNaDLearner.from_bc_checkpoint(bc_checkpoint, self.tables, self.rnad_config(), device=self.device)
        self.link = ServerLink(self.server, push_every=int(c["rnad"]["push_every"]))
        learner.add_hook(self.link)
        learner.add_hook(self._on_learner_event)
        self.learner = learner
        sp = self.rd.episode_dir("selfplay")
        os.makedirs(sp, exist_ok=True)
        self.window = RP.ReplayWindow(self._window_cfg([sp], True), log=lambda m: log("rnad window: " + m))
        restored, left_out = restore_window(self.window, saved_window, [sp]) if saved_window else (0, 0)
        self.window.start()
        log("rnad: %s learner at step %d (iteration %d), window restored with %d episodes, %d episode files "
            "not in the checkpoint left out" % ("resumed" if resume else "seeded", learner.learner_steps,
                                                learner.reg_updates, restored, left_out))
        return {"learner_steps": learner.learner_steps, "reg_updates": learner.reg_updates}

    def _on_learner_event(self, learner, event):
        if event != "iteration":
            return
        from .. import model as M
        path = os.path.join(self.rd.fixed, "m%d.pt" % learner.reg_updates)
        reg = learner.regs[0]
        M.save_checkpoint(reg.model, path, {"version": reg.version, "learner_steps": learner.learner_steps,
                                            "reg_updates": learner.reg_updates})
        self.event("rnad_iteration", reg_updates=learner.reg_updates, learner_steps=learner.learner_steps,
                   version=reg.version, fixed_point=path)

    def save_model(self, path):
        """The current network saved to `path` (model.save_checkpoint, atomically); its version."""
        from .. import model as M
        model, version = self.current_model()
        t = time.time()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        M.save_checkpoint(model, path, {"version": version})
        return {"path": path, "version": version, "seconds": round(time.time() - t, 3)}

    def maybe_publish(self):
        """Local actors: the current network saved for the live export when due (see the module docstring)."""
        a = self.cfg["actors"]
        if a.get("policy_eval") != "local" or self.learner is None:
            return
        step = self.learner.learner_steps
        if self.published is not None:
            last_step, last_at = self.published
            due = time.time() - last_at >= float(a["export_every_s"]) or \
                (int(a["export_every_steps"]) > 0 and step - last_step >= int(a["export_every_steps"]))
            if step == last_step or not due:
                return
        r = self.save_model(os.path.join(self.rd.exports, "live", "src-v%d.pt" % step))
        self.published = (step, time.time())
        self.event("policy_saved", **r)

    def rnad_unit(self):
        from .. import replay as RP
        from ..rnad_learner import trainable_episodes
        r = self.cfg["rnad"]
        G = int(r["games_per_step"])
        self.maybe_publish()
        if len(self.pending) < G:
            self.pending += self.window.take_fresh(G - len(self.pending), timeout=0.5,
                                                   newest=r.get("fresh_order") == "newest")
            if len(self.pending) < G:
                return
        eps, self.pending = self.pending, []
        kept, dropped = trainable_episodes(eps, missing_mu="drop", spec_hash=self.tables.spec.hash)
        kept, stale = drop_stale(kept, self.learner.learner_steps, int(r.get("max_policy_lag", 0)))
        if stale:
            dropped["policy_lag"] = stale
        for k, v in dropped.items():
            self.dropped[k] = self.dropped.get(k, 0) + v
        if not kept:
            return
        batch = RP.TrajectoryBatch(kept, chunk_len=int(r["chunk_len"]), policies=self.learner.cfg.reg_policy_ids,
                                   omniscient=self.learner.model.critic is not None)
        pv = batch.policy_version[batch.valid]
        if len(pv) and (pv >= 0).all():
            self.lags.append(self.learner.learner_steps - pv)
        t0 = time.time()
        m = self.learner.step(batch)
        m["step_seconds"] = time.time() - t0
        m["boundary"] = float(m["boundary"])
        self._aggregate(m, ("learner_steps", "reg_updates", "alpha", "loss_scale", "finetune"),
                        ("decisions", "games", "seconds", "step_seconds", "skipped", "boundary", "reg_recorded",
                         "reg_recomputed", "games_dropped_stale"))
        if self.agg["n"] >= r["report_every"] or m["boundary"]:
            out = self._flush(m["learner_steps"])
            out["decisions_per_s"] = out["decisions"] / max(out["step_seconds"], 1e-9)
            out["wait_fraction"] = max(0.0, 1.0 - out["step_seconds"] / max(out["wall_seconds"], 1e-9))
            out.update(lag_percentiles(self.lags))
            self.lags = []
            out["games_dropped"] = dict(self.dropped)
            out["window"] = self.window.stats()
            self.event("rnad_step", **out)

    # ── metrics aggregation ──

    def _aggregate(self, m, last_keys, sum_keys):
        """Adds one unit's metrics: `last_keys` keep the last value, `sum_keys` sum, every other
        number is averaged over the units that reported it (value_mse_after: the measured steps)."""
        a = self.agg
        if a is None:
            # The window starts where the last one ended (its first unit's time included).
            a = self.agg = {"n": 0, "sums": {}, "counts": {}, "last": {}, "t0": getattr(self, "flushed_at", None) or time.time()}
        a["n"] += 1
        for k, v in m.items():
            if isinstance(v, bool):
                v = float(v)
            if not isinstance(v, (int, float)):
                continue
            if k in last_keys:
                a["last"][k] = v
            elif k in sum_keys:
                a["sums"][k] = a["sums"].get(k, 0.0) + v
            else:
                a["sums"]["mean:" + k] = a["sums"].get("mean:" + k, 0.0) + v
                a["counts"][k] = a["counts"].get(k, 0) + 1

    def _flush(self, step):
        a, self.agg = self.agg, None
        self.flushed_at = time.time()
        out = dict(a["last"])
        for k, v in a["sums"].items():
            out[k[5:] if k.startswith("mean:") else k] = v / a["counts"][k[5:]] if k.startswith("mean:") else v
        out["units"] = a["n"]
        out["wall_seconds"] = time.time() - a["t0"]
        out["step"] = step
        return out

    # ── commands ──

    def current_model(self):
        if self.learner is not None:
            return self.learner.model, self.learner.learner_steps
        if self.trainer is not None:
            return self.trainer.model, self.trainer.step
        raise ValueError("no model yet")

    def handle(self, msg):
        from .. import model as M
        cmd = msg.get("cmd")
        if cmd == "hello":
            return {"pid": os.getpid(), "device": self.device, "rmip": self.rd.rmip_sock}
        if cmd == "bc_init":
            return self.bc_init(msg["dirs"], msg.get("resume"))
        if cmd == "rnad_init":
            return self.rnad_init(msg.get("bc_checkpoint"), msg.get("resume"))
        if cmd == "mode":
            if msg["mode"] == "bc" and self.trainer is None or msg["mode"] == "rnad" and self.learner is None:
                raise ValueError("mode %s before its init" % msg["mode"])
            self.mode = msg["mode"]
            self.agg = None
            self.flushed_at = None
            return {"mode": self.mode}
        if cmd == "pause":
            self.paused = True
            return {}
        if cmd == "resume":
            self.paused = False
            self.agg = None
            self.flushed_at = None
            return {}
        if cmd == "save_model":
            return self.save_model(msg["path"])
        if cmd in ("serve", "snapshot"):
            model, version = self.current_model()
            pid = int(msg.get("pid", 0 if cmd == "serve" else 3))
            self.server.set_policy(pid, model, version)
            return {"pid": pid, "version": version}
        if cmd == "load_policy":
            model = M.load_checkpoint(msg["path"], self.tables)
            self.server.set_policy(int(msg["pid"]), model, int(msg.get("version", 0)))
            return {"pid": int(msg["pid"])}
        if cmd == "unload":
            for pid in msg["pids"]:
                self.server.remove_policy(int(pid))
            return {}
        if cmd == "bc_eval":
            return self.bc_evaluate()
        if cmd == "checkpoint":
            return self.checkpoint(msg["path"], msg.get("keep", []))
        if cmd == "set":
            return self.set(msg["key"], msg["value"])
        if cmd == "config":
            self.cfg = msg["config"]
            return {}
        if cmd == "finetune":
            if self.learner is None:
                raise ValueError("finetune before rnad_init")
            if self.learner.cfg.finetune_from_learner_steps < 0:
                self.learner.configure(finetune_from_learner_steps=self.learner.learner_steps)
            return {"from": self.learner.cfg.finetune_from_learner_steps}
        if cmd == "exploit_start":
            return self.exploit_start(msg)
        if cmd == "exploit_status":
            if self.exploit is None:
                raise ValueError("no exploiter running")
            return self.exploit.status()
        if cmd == "exploit_finish":
            if self.exploit is None:
                raise ValueError("no exploiter running")
            x, self.exploit = self.exploit, None
            self.exploit_pause = False
            return {"result": x.result(int(msg.get("decisions", 0)))}
        if cmd == "exploit_cancel":
            if self.exploit is not None:
                x, self.exploit = self.exploit, None
                self.exploit_pause = False
                x.close()
            return {}
        if cmd == "stop":
            raise _Stop()
        if cmd == "_eof":
            raise _Stop()
        if cmd == "crash":         # tests: the GPU process dies
            os._exit(int(msg.get("code", 9)))
        raise ValueError("unknown command %r" % cmd)

    def exploit_start(self, msg):
        """An exploiter (rmlearn/exploiter.py) against the current policy, trained in this process
        between the learner's units: the victim a frozen copy of the current network (served at
        pids[1]), the exploiter a trainable copy (pids[0]); the actors' schedule in the reply."""
        import copy
        from .. import exploiter as X
        if self.exploit is not None:
            raise ValueError("an exploiter is already running")
        model, version = self.current_model()
        victim = copy.deepcopy(model).eval()
        init = copy.deepcopy(model)
        conf = dict(msg.get("config") or {})
        if conf.get("pairs"):
            conf["pairs"] = [tuple(p) for p in conf["pairs"]]
        cfg = X.ExploiterConfig(**dict(conf, device=self.device))
        self.exploit = X.ExploiterSession(self.server, init, victim, cfg, msg["dir"], pids=tuple(msg["pids"]),
                                          victim_version=version, labels={"victim": "v%d" % version,
                                                                          "init": "v%d" % version}, log=log)
        self.exploit_pause = bool(msg.get("pause_learner", True))
        return {"schedule": self.exploit.schedule(), "victim_version": version}

    def checkpoint(self, path, keep):
        """Saves the model (``{step}`` in `path`: the learner's step now); the reply names the file."""
        import torch
        _, step = self.current_model()
        path = path.replace("{step}", str(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.learner is not None:
            data = {"learner": self.learner.state_dict(), "window": window_state(self.window, self.pending)}
            tmp = path + ".tmp"
            torch.save(data, tmp)
            os.replace(tmp, path)
            out = {"kind": "rnad", "learner_steps": self.learner.learner_steps,
                   "reg_updates": self.learner.reg_updates, "version": self.learner.learner_steps}
        elif self.trainer is not None:
            self.trainer.save(path, version=self.trainer.step)
            out = {"kind": "bc", "step": self.trainer.step, "version": self.trainer.step}
        else:
            raise ValueError("nothing to checkpoint")
        out["path"] = path
        out["bytes"] = os.path.getsize(path)
        prune(os.path.dirname(path), keep)
        return out

    def set(self, key, value):
        sec, _, k = key.partition(".")
        self.cfg[sec][k] = value
        if sec == "bc" and self.trainer is not None and k in self.trainer.cfg.__dataclass_fields__:
            setattr(self.trainer.cfg, k, value)
        if sec == "rnad" and self.learner is not None:
            if k in self.learner.cfg.__dataclass_fields__:
                self.learner.configure(**{k: value})
                if k in ("reg_eps", "reg_eps_sideboarding"):
                    for reg in self.learner.regs:
                        if reg.smoothed():
                            setattr(reg, "eps" if k == "reg_eps" else "eps_sideboarding", float(value))
            if k == "push_every" and self.link is not None:
                self.link.push_every = int(value)
        if key == "replay.max_decision_tokens":
            for w in (self.bc_window, self.window):
                if w is not None:
                    w.set_max_decision_tokens(value)
        return {"key": key, "value": value}

    def stats(self):
        import torch
        s = {"pid": os.getpid(), "mode": self.mode, "paused": self.paused, "uptime_s": time.time() - self.started,
             "last_error": self.last_error}
        if self.server is not None:
            s["server"] = self.server.stats()
        if self.trainer is not None:
            s["bc_step"] = self.trainer.step
            s["bc_decisions"] = self.trainer.decisions
        if self.bc_window is not None:
            s["window"] = self.bc_window.stats()
        if self.learner is not None:
            s["learner_steps"] = self.learner.learner_steps
            s["reg_updates"] = self.learner.reg_updates
            s["learner_totals"] = dict(self.learner.totals)
            s["pending_games"] = len(self.pending)
            s["games_dropped"] = dict(self.dropped)
        if self.window is not None:
            s["window"] = self.window.stats()
        if str(self.device).startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            s["gpu_memory"] = {"allocated_mb": torch.cuda.memory_allocated() / 2 ** 20,
                               "peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                               "used_mb": (total - free) / 2 ** 20, "total_mb": total / 2 ** 20}
        return s

    # ── the loop ──

    def run(self):
        threading.Thread(target=self.reader, name="commands", daemon=True).start()
        while True:
            try:
                timeout = 0 if (self.mode != "idle" and not self.paused) else 0.2
                msg = self.cmds.get(timeout=timeout) if timeout else self.cmds.get_nowait()
            except queue.Empty:
                msg = None
            if msg is not None:
                try:
                    reply = self.handle(msg)
                    self.send({"re": msg.get("id"), "ok": True, **(reply or {})})
                except _Stop:
                    self.send({"re": msg.get("id"), "ok": True})
                    return 0
                except Exception as err:  # the orchestrator decides what a failed command means
                    log("command %s failed: %s\n%s" % (msg.get("cmd"), err, traceback.format_exc()))
                    self.send({"re": msg.get("id"), "ok": False, "error": "%s: %s" % (type(err).__name__, err)})
                continue
            if self.exploit is not None:
                self.exploit.tick()
                if self.exploit_pause:
                    time.sleep(0.05)
                    continue
            if self.paused:
                continue
            if self.mode == "bc":
                self.bc_unit()
            elif self.mode == "rnad":
                self.rnad_unit()

    def close(self):
        if self.server is not None:
            self.server.stop()
        for w in (self.window, self.bc_window):
            if w is not None:
                w.stop()
        if self.trainer is not None:
            self.trainer.close()


class _Stop(Exception):
    pass


def _jsonable(x):
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
    except ImportError:
        pass
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.item() if x.numel() == 1 else x.tolist()
    except ImportError:
        pass
    return str(x)


def drop_stale(eps, learner_steps, max_lag):
    """(the episodes whose every step was played by a policy at most `max_lag` learner steps old, the
    number dropped); `max_lag` 0 keeps every one. A step's policy version is the learner step it was
    exported or served at (-1: none, e.g. a teacher's step, never stale)."""
    import numpy as np
    if max_lag <= 0:
        return eps, 0
    kept = []
    for ep in eps:
        pv = np.asarray(ep.policy_version)
        pv = pv[pv >= 0]
        if len(pv) and learner_steps - int(pv.min()) > max_lag:
            continue
        kept.append(ep)
    return kept, len(eps) - len(kept)


def lag_percentiles(lags):
    """The policy lag (learner step − the step's policy version, per trained decision) of a report
    window: its median, 95th percentile, max and mean."""
    import numpy as np
    if not lags:
        return {}
    a = np.concatenate(lags)
    if not len(a):
        return {}
    return {"policy_lag_p50": float(np.percentile(a, 50)), "policy_lag_p95": float(np.percentile(a, 95)),
            "policy_lag_max": float(a.max()), "policy_lag_mean": float(a.mean())}


def prune(directory, keep):
    """Keeps the newest periodic checkpoints named in `keep` ({"prefix": p, "count": n}) of `directory`."""
    if not keep:
        return
    prefix, count = keep["prefix"], int(keep["count"])
    files = sorted((f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".pt")),
                   key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    for f in files[:-count] if count > 0 else []:
        for ext in ("", ".opt"):
            try:
                os.unlink(os.path.join(directory, f + ext))
            except FileNotFoundError:
                pass


def episode_files(dirs):
    """[(mtime, size, path)] of every .rmtraj of `dirs`."""
    out = []
    for d in dirs:
        try:
            it = os.scandir(d)
        except FileNotFoundError:
            continue
        with it:
            for e in it:
                if e.name.endswith(".rmtraj") and e.is_file():
                    try:
                        st = e.stat()
                    except FileNotFoundError:
                        continue
                    out.append((st.st_mtime, st.st_size, e.path))
    return out


def restore_recent(win, dirs, budget):
    """Leaves out (``win.skip``) the oldest episode files beyond `budget` bytes, newest first, so a
    restarted window reads only what it would keep; the number left out."""
    files = sorted(episode_files(dirs), reverse=True)
    total = 0
    skip = []
    for mtime, size, path in files:
        total += size
        if total > budget:
            skip.append(path)
    win.skip(skip)
    return len(skip)


def window_state(win, pending=()):
    """The R-NaD window's part of a checkpoint: the episodes in the window (``files``, ingestion
    order) and those the learner has not taken yet (``fresh``: the window's untaken ones and
    `pending`, taken but not yet trained on). The window is polled first, so every episode file on
    disk at this moment is in ``files`` or ``fresh``: whatever a resume finds on disk beside them
    was written after the checkpoint."""
    win.poll()
    files, fresh = win.files()
    return {"files": files, "fresh": fresh + [e.path for e in pending], "complete": True}


def restore_window(win, state, dirs):
    """A checkpoint's window back (``window_state``): its episodes still on disk re-read, only its
    fresh ones (not yet taken by the learner at the checkpoint) offered as fresh; every other
    episode file already on disk is left out (``win.skip``): taken before the checkpoint, or written
    after it by policies of the lost steps — whose versions the restored learner issues again for
    other weights, so their recorded evaluations would be mistaken for its own. Episodes written
    from now on are polled as fresh. (n restored, n left out)."""
    from .. import episodes as EP
    now = time.time()
    fresh = set(state.get("fresh", []))
    listed = list(state.get("files", []))
    for p in fresh:
        if p not in listed:
            listed.append(p)
    win.skip(listed)
    n = 0
    for p in listed:
        try:
            ep = EP.read_episode(p)
        except (OSError, ValueError):
            continue
        if win.add(ep, fresh=p in fresh):
            n += 1
    known = set(listed)
    left = [p for mtime, _, p in episode_files(dirs) if p not in known and mtime <= now]
    win.skip(left)
    return n, len(left)


def main():
    out = _protocol_out()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal.SIG_IGN)
    if len(sys.argv) != 2:
        print("usage: python -m rmlearn.orchestrator.gpu RUN_DIR", file=sys.stderr)
        return 2
    w = Worker(sys.argv[1], out)
    code = 0
    try:
        w.take_lock()
        w.setup()
        w.event("ready", pid=os.getpid(), device=w.device)
        code = w.run()
    except SystemExit as err:
        msg = str(err)
        log(msg)
        w.event("fatal", message=msg)
        code = 3
    except BaseException as err:  # the orchestrator stops the run on this
        msg = "%s: %s" % (type(err).__name__, err)
        log("FATAL: %s\n%s" % (msg, traceback.format_exc()))
        w.event("fatal", message=msg)
        code = 1
    finally:
        try:
            w.close()
        except Exception as err:  # closing after a failure
            log("close: %r" % err)
    log("exit %d" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
