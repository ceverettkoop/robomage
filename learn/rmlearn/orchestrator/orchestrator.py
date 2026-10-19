"""The orchestrator (docs/rmtrain.md; docs/rnad_training_plan.md §4 L6, §6): one process, no GPU, that
walks the phases B0–B5 of one run, owns the actors, the GPU process, the evaluations and exports,
and answers the control socket.

    Orchestrator(RunDir(path)).main()     # until stopped, finished or failed; the exit status

**The loop** (every 0.5 s): the actors' new results lines (games and decisions per schedule
version), the GPU process's events (metrics, held-out evaluations, R-NaD iteration boundaries), the
phase's exit test, the processes brought to what the phase and stage want (``sync``: the schedule
written, the actor count, the GPU process's mode), the evaluation queue, the export queue, the
checkpoint cadence, ``status.json``. Control requests that change anything are queued to the loop and
answered from it.

**Durability.** ``state.json`` (written whole, often) holds the phase and its stage, the counters
and the results files' offsets, the last checkpoint (written by the GPU process, then recorded
here), the ladder, the evaluation history and the transitions. A restarted orchestrator resumes the
phase from it: the GPU process reloads the recorded checkpoint, the actors start after their last
match (the one in progress written off), counters continue from the recorded offsets.

**Failures.** A crashed actor is restarted (actors.py); the GPU process dying (or failing a command)
stops the run: actors stopped, state saved, exit status 2. ``start`` resumes it from its checkpoint.
"""

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback

from . import config as C
from . import evaluation as EVAL
from . import schedules as SCH
from .actors import ActorPool, Counters, ResultsTail, cpu_env, draw_finding
from .control import ControlServer
from .gpulink import GpuError, GpuLink
from .rundir import REPO_ROOT, JsonLines, Log, read_json, read_jsonl, write_json

PHASES = ["B0", "B1", "B2", "B3", "B4", "B5", "done"]
FINDINGS_KEPT = 10
LEARN_DIR = os.path.join(REPO_ROOT, "learn")


class Fatal(Exception):
    pass


class Orchestrator:
    def __init__(self, run, echo=False):
        self.run = run
        self.cfg = read_json(run.config)
        if self.cfg is None:
            raise SystemExit("rmtrain: %s has no config.json" % run.path)
        self.cfg, added = C.fill_defaults(self.cfg)
        C.validate(self.cfg)
        if added:
            write_json(run.config, self.cfg)
        self.log = Log(run.log("orchestrator"), echo=(lambda line: print(line, flush=True)) if echo else None)
        self.started = time.time()
        self.state = None
        self.metrics = None
        self.evals_file = None
        self.gpu = None
        self.pool = None
        self.tail = None
        self.counters = None
        self.requests = queue.Queue()
        self.stopping = None             # the reason, once a stop began
        self.exit_code = 0
        self.force_next = False
        self.eval_job = None
        self.export_proc = None
        self.export_job = None
        self.export_log = None
        self.export_queue = []
        self.gpu_stats = {}
        self.gpu_stats_at = 0.0
        self.last_metrics = {}
        self.status_cache = {}
        self.schedule_text = None
        self.gpu_mode = None
        self.gpu_paused = None
        self.gpu_model = None           # "bc" / "rnad" once loaded in the GPU process
        self.last_state_save = 0.0
        self.last_checkpoint = time.time()
        self.lock_fp = None
        bin_dir = self.cfg["run"]["bin_dir"]
        self.bin_dir = bin_dir if os.path.isabs(bin_dir) else os.path.join(REPO_ROOT, bin_dir)
        self.engine = os.path.join(self.bin_dir, self.cfg["run"]["build"], "robomage_nn")
        self.status_lock = threading.Lock()

    # ── state ────────────────────────────────────────────────────────────────────────────────

    def new_state(self):
        return {"created": time.time(), "phase": "B0", "stage": "", "phase_started": time.time(), "hold": False,
                "paused": False, "transitions": [], "results": {"offsets": {}, "counters": {}}, "checkpoint": None,
                "bc": {"best_agree": None, "evals_since_best": 0, "last_eval": None, "final": None, "b1_final": None},
                "b2": {"round": 0, "round_start_step": 0, "best": None, "history": []},
                "rnad": {"iterations": 0, "learner_steps": 0}, "ladder": [], "selfplay_version": "b4",
                "eval": {"seq": 0, "last": None, "last_at": None, "queue": [], "history": []}, "exports": [],
                "starts": 0, "held_reason": "", "findings": {"draws": 0, "recent": []}}

    def save_state(self):
        self.state["results"] = {"offsets": self.tail.offsets, "counters": self.counters.state()}
        self.state["saved_at"] = time.time()
        write_json(self.run.state, self.state)
        self.last_state_save = time.time()

    def emit(self, event, **kw):
        """A metrics.jsonl line: time, phase, stage, step and the event's fields."""
        line = {"time": time.time(), "phase": self.state["phase"], "stage": self.state["stage"],
                "step": self.current_step(), "event": event}
        line.update(kw)
        self.metrics.write(line)
        self.last_metrics[event] = line
        return line

    def current_step(self):
        p = self.state["phase"]
        if p in ("B4", "B5", "done"):
            return self.state["rnad"]["learner_steps"]
        return self.state.get("bc_step", 0)

    # ── the main loop ────────────────────────────────────────────────────────────────────────

    def main(self):
        import fcntl
        self.lock_fp = open(self.run.lock, "a+")
        try:
            fcntl.flock(self.lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("rmtrain: run %s already has an orchestrator" % self.run.path, file=sys.stderr)
            return 1
        self.run.make()
        self.state = read_json(self.run.state) or self.new_state()
        self.state["starts"] = self.state.get("starts", 0) + 1
        self.state.setdefault("findings", {"draws": 0, "recent": []})
        lost = self.state.get("eval", {}).pop("running", None)
        if lost:            # an evaluation the last orchestrator died in: played again from its start
            self.state["eval"]["queue"].insert(0, lost)
        self.metrics = JsonLines(self.run.metrics)
        self.evals_file = JsonLines(self.run.evals)
        res = self.state.get("results", {})
        self.tail = ResultsTail(self.run.actors, res.get("offsets"))
        self.counters = Counters(res.get("counters"))
        self.pool = ActorPool(self.engine, self.bin_dir, self.run.schedule, self.run.actors, self.run.actor_logs,
                              self.cfg["run"]["seed"], self.cfg["actors"], self.log)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda s, f: self.request_stop("signal %d" % s))
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        self.log("start #%d: run %s, phase %s%s, pid %d" % (self.state["starts"], self.run.path, self.state["phase"],
                                                           "/" + self.state["stage"] if self.state["stage"] else "",
                                                           os.getpid()))
        self.emit("start", pid=os.getpid(), starts=self.state["starts"])
        control = ControlServer(self.run.control_sock, self.handle_request, self.log)
        try:
            if not os.path.exists(self.engine):
                raise Fatal("no engine %s (make BUILD=%s actor)" % (self.engine, self.cfg["run"]["build"].upper()))
            self.ensure_tables()
            self.save_state()
            self.write_status()
            control.start()
            self.enter_phase(resume=True)
            while not self.stopping:
                self.tick()
                time.sleep(0.5)
            self.shutdown(self.stopping)
        except (Fatal, GpuError) as err:
            self.log("FATAL: %s" % err)
            self.emit("fatal", message=str(err))
            self.exit_code = 2
            self.shutdown("fatal: %s" % err, checkpoint=False)
        except Exception as err:  # the run stops; start resumes it
            self.log("FATAL: %r\n%s" % (err, traceback.format_exc()))
            self.emit("fatal", message=repr(err))
            self.exit_code = 2
            self.shutdown("fatal: %r" % err, checkpoint=False)
        finally:
            control.stop()
            self.write_status(running=False)
            self.metrics.close()
            self.evals_file.close()
        self.log("exit %d" % self.exit_code)
        return self.exit_code

    def ingest_results(self):
        """The training actors' new results lines: the counters, and the draws as findings."""
        for line in self.tail.poll():
            self.counters.add(line)
            self.note_finding(line, "train", os.path.join(self.run.actors, "a%s.results.jsonl" % line.get("actor")))

    def note_finding(self, line, source, results_file=None):
        """A drawn game (CR 104.4a: both players lost at once) noted for review, not as an error:
        state ``findings`` (the count and the latest few, each with its results file and decision log)."""
        f = draw_finding(line, source, results_file, self.run.path)
        if f is None:
            return
        fd = self.state.setdefault("findings", {"draws": 0, "recent": []})
        fd["draws"] += 1
        fd["recent"] = (fd["recent"] + [f])[-FINDINGS_KEPT:]
        self.log("finding: a draw, %s" % f["text"])

    def request_stop(self, reason):
        if not self.stopping:
            self.stopping = reason

    def tick(self):
        self.ingest_results()
        self.counters.sample()
        if self.gpu is not None:
            if not self.gpu.alive():
                raise Fatal("the GPU process died (exit %s; see logs/gpu.log)%s"
                            % (self.gpu.proc.returncode, ": " + self.gpu.fatal if self.gpu.fatal else ""))
            for ev in self.gpu.events():
                self.on_gpu_event(ev)
        self.pool.tick()
        self.serve_requests()
        if not self.state["paused"] and not self.stopping:
            self.phase_tick()
        self.sync()
        self.eval_tick()
        self.export_tick()
        if (self.gpu_model and not self.state["paused"]
                and time.time() - self.last_checkpoint > 60 * self.cfg["checkpoint"]["every_minutes"]):
            self.checkpoint()
        if time.time() - self.last_state_save > 10:
            self.save_state()
        if time.time() - self.status_cache.get("updated_at", 0) >= self.cfg["run"]["status_every_s"]:
            self.write_status()

    # ── processes wanted by the phase ───────────────────────────────────────────────────────

    def wanted(self):
        """(schedule dict or None, GPU mode or None) of the phase and stage."""
        p, s = self.state["phase"], self.state["stage"]
        if p == "B0" and s == "collect":
            return SCH.b0(self.run, self.cfg), None
        if p == "B1":
            return None, "bc" if s == "train" else "idle"
        if p == "B2":
            if s == "collect":
                r = self.state["b2"]["round"]
                return SCH.dagger(self.run, self.cfg, r, self.dagger_mix(r)), "bc"
            return None, "bc"
        if p in ("B4", "B5"):
            return SCH.selfplay(self.run, self.cfg, self.state["selfplay_version"]), "rnad"
        return None, None

    def sync(self):
        if self.stopping:
            return
        sched, mode = self.wanted()
        paused = self.state["paused"]
        if sched is None or paused:
            if self.pool.count or self.pool.running():
                self.pool.stop_all()
        else:
            text = json.dumps(sched, indent=1)
            if text != self.schedule_text:
                write_json(self.run.schedule, sched)
                self.schedule_text = text
                self.log("schedule %s written (%d entries)" % (sched["version"], len(sched["entries"])))
            if self.pool.count != self.cfg["actors"]["count"]:
                self.log("actors: %d -> %d" % (self.pool.count, self.cfg["actors"]["count"]))
                self.pool.set_count(self.cfg["actors"]["count"])
        if self.gpu is not None and mode is not None:
            if mode != self.gpu_mode:
                self.gpu.call("mode", mode=mode)
                self.gpu_mode = mode
            if paused != self.gpu_paused:
                self.gpu.call("pause" if paused else "resume")
                self.gpu_paused = paused

    # ── the GPU process ─────────────────────────────────────────────────────────────────────

    def ensure_gpu(self):
        if self.gpu is not None:
            return
        self.gpu = GpuLink(self.run, self.cfg, self.log)
        self.log("starting the GPU process (device %s, lock %s)" % (self.cfg["run"]["device"],
                                                                   self.cfg["run"]["gpu_lock"] or "none"))
        ev = self.gpu.start(on_wait=self.on_lock_wait)
        self.gpu_mode = None
        self.gpu_paused = None
        self.log("GPU process ready: pid %s" % ev.get("pid"))

    def on_lock_wait(self, path):
        self.log("the GPU process waits for the GPU lock %s (another run or GPU job holds it)" % path)
        with self.status_lock:
            self.status_cache["gpu"] = {"error": "waiting for the GPU lock %s" % path}
        write_json(self.run.status, self.status_cache)

    def bc_dirs(self):
        b0 = self.cfg["b0"]["corpus_dir"]
        b0 = (b0 if os.path.isabs(b0) else os.path.join(REPO_ROOT, b0)) if b0 else self.run.episode_dir("b0")
        dg = self.run.episode_dir("dagger")
        os.makedirs(dg, exist_ok=True)
        return [b0, dg]

    def ensure_bc_model(self):
        self.ensure_gpu()
        if self.gpu_model == "bc":
            return
        ck = self.state["checkpoint"]
        resume = ck["path"] if ck and ck.get("kind") == "bc" and os.path.exists(ck["path"]) else None
        r = self.gpu.call("bc_init", dirs=self.bc_dirs(), resume=resume)
        self.gpu_model = "bc"
        before = self.state.get("bc_step", 0)
        self.state["bc_step"] = r["step"]
        self.log("BC model %s at step %d (%d parameters)" % ("restored" if resume else "initialised", r["step"],
                                                           r["params"]))
        if resume:
            for k in ("bc_train", "bc_eval"):
                self.last_metrics.pop(k, None)
            self.emit("restore", model="bc", checkpoint=resume, bc_step=r["step"], previous_bc_step=before)

    def ensure_rnad_model(self):
        self.ensure_gpu()
        if self.gpu_model == "rnad":
            return
        ck = self.state["checkpoint"]
        if not (ck and ck.get("kind") == "rnad" and os.path.exists(ck["path"])):
            raise Fatal("phase %s without an R-NaD checkpoint" % self.state["phase"])
        r = self.gpu.call("rnad_init", resume=ck["path"])
        self.gpu_model = "rnad"
        rn = self.state["rnad"]
        before = (rn["learner_steps"], rn["iterations"])
        rn["learner_steps"] = r["learner_steps"]
        rn["iterations"] = r["reg_updates"]
        self.log("R-NaD learner restored at step %d, iteration %d (the run had reached step %d, iteration %d)"
                 % (r["learner_steps"], r["reg_updates"], before[0], before[1]))
        self.last_metrics.pop("rnad_step", None)
        self.emit("restore", model="rnad", checkpoint=ck["path"], learner_steps=r["learner_steps"],
                  iterations=r["reg_updates"], previous_learner_steps=before[0], previous_iterations=before[1])
        self.roll_back(r["learner_steps"], r["reg_updates"])

    def roll_back(self, steps, iterations):
        """After a restore: what the lost learner steps made is dropped — the fixed points of later
        iterations from the ladder, their queued evaluations (each comes again when the restored
        learner crosses its boundary, for the fixed point it then writes), and the ladder record's
        players of the lost steps (``m<k>`` past `iterations`, ``v<version>`` past `steps`: those
        names are issued again for other weights)."""
        st = self.state
        lost = [r["label"] for r in st["ladder"] if _iteration_of(r["label"]) > iterations]
        if lost:
            st["ladder"] = [r for r in st["ladder"] if r["label"] not in lost]
        q = st["eval"]["queue"]
        dropped = [x["label"] for x in q if (x.get("policy") or {}).get("iteration", 0) > iterations]
        st["eval"]["queue"] = [x for x in q if x["label"] not in dropped]
        try:
            removed, ratings = EVAL.roll_back_ladder(self.run, iterations, steps)
        except (OSError, ValueError, KeyError) as err:      # the ratings are a report, never fatal
            removed, ratings = [], None
            self.log("roll back: ladder: %s" % err)
        if ratings is not None:
            st["eval"]["ratings"] = ratings
        if lost or dropped or removed:
            self.log("rolled back to step %d, iteration %d: ladder rungs %s, queued evaluations %s, ladder players %s "
                     "dropped" % (steps, iterations, lost or "-", dropped or "-", removed or "-"))

    def on_gpu_event(self, ev):
        kind = ev.pop("event", "")
        ev.pop("time", None)
        if kind == "bc_train":
            self.state["bc_step"] = int(ev.get("step", 0))
            self.emit("bc_train", **ev)
        elif kind == "bc_eval":
            self.state["bc_step"] = int(ev.get("eval_step", self.state.get("bc_step", 0)))
            self.on_bc_eval(ev)
            self.emit("bc_eval", **ev)
        elif kind == "rnad_step":
            self.state["rnad"]["learner_steps"] = int(ev.get("learner_steps", 0))
            self.emit("rnad_step", **ev)
        elif kind == "rnad_iteration":
            self.on_iteration(ev)
        elif kind == "fatal":
            raise Fatal("GPU process: %s" % ev.get("message"))
        else:
            self.emit("gpu_" + kind, **ev)

    def checkpoint(self, name=None, keep_periodic=True):
        """The GPU process's model and optimiser (and, under R-NaD, the window's files) saved; the
        state records it. `name`: a kept checkpoint's file name (else a periodic one, pruned)."""
        if not self.gpu_model:
            return None
        kind = self.gpu_model
        if name is None:
            path = os.path.join(self.run.ckpt, "%s-p{step}.pt" % kind)     # the GPU process's step
            keep = {"prefix": "%s-p" % kind, "count": self.cfg["checkpoint"]["keep"]} if keep_periodic else {}
        else:
            path = os.path.join(self.run.ckpt, name)
            keep = {}
        r = self.gpu.call("checkpoint", path=path, keep=keep)
        path = r.get("path", path)
        r.pop("re", None)
        r.pop("ok", None)
        r["at"] = time.time()
        r["phase"] = self.state["phase"]
        if kind == "rnad":
            self.state["rnad"]["learner_steps"] = int(r["learner_steps"])
        self.state["checkpoint"] = r
        self.last_checkpoint = time.time()
        self.save_state()
        self.emit("checkpoint", **r)
        self.log("checkpoint %s (%.1f MB)" % (path, r.get("bytes", 0) / 2 ** 20))
        return r

    # ── phases ──────────────────────────────────────────────────────────────────────────────

    def enter_phase(self, resume=False):
        """Sets the phase up (also after a restart: the stage recorded in the state)."""
        p = self.state["phase"]
        st = self.state
        if p == "B0":
            if self.cfg["b0"]["corpus_dir"] or self.cfg["b0"]["decisions"] <= 0:
                return self.transition("B1", "corpus reused (b0.corpus_dir)" if self.cfg["b0"]["corpus_dir"]
                                       else "b0.decisions is 0", force=True)
            st["stage"] = "collect"
        elif p == "B1":
            self.ensure_bc_model()
            if st["stage"] not in ("train", "eval"):
                st["stage"] = "train"
            if st["stage"] == "eval" and not self.eval_queued("b1") and not st.get("pending"):
                self.queue_eval("b1", self.cfg["eval"]["nn_options_bc"], ["fixed"])
        elif p == "B2":
            if self.cfg["b2"]["rounds_max"] <= 0:
                return self.transition("B3", "b2.rounds_max is 0", force=True)
            self.ensure_bc_model()
            if st["b2"]["round"] < 1:
                st["b2"]["round"] = 1
                st["stage"] = "collect"
            if st["stage"] not in ("collect", "eval"):
                st["stage"] = "collect"
            if st["stage"] == "collect":
                self.start_round(resume)
            elif not self.eval_queued("b2-r%d" % st["b2"]["round"]) and not st.get("pending"):
                self.queue_eval("b2-r%d" % st["b2"]["round"], self.cfg["eval"]["nn_options_bc"], ["fixed"])
        elif p == "B3":
            self.seed_rnad()
        elif p in ("B4", "B5"):
            self.ensure_rnad_model()
            st["stage"] = "selfplay"
            st["selfplay_version"] = p.lower()
            if p == "B5":
                r = self.gpu.call("finetune")
                self.log("fine-tuning projection on from learner step %d" % r["from"])
        elif p == "done":
            if self.gpu is not None and self.gpu_model == "rnad" and not resume:
                ck = self.checkpoint(name="rnad-final.pt")
                self.queue_export(ck["path"], "final", ck["version"])
            self.request_stop("the run is finished")
        self.save_state()

    def note_held(self, to, reason):
        st = self.state
        if st.get("held_reason") != reason:
            st["held_reason"] = reason
            self.log("transition %s -> %s held (%s)" % (st["phase"], to, reason))
            self.emit("transition_held", to=to, reason=reason)

    def transition(self, to, reason, force=False, pending=False):
        """Moves to phase `to` unless held (`force`: also when held). True when it moved. With
        `pending`, a held transition is recorded and made once released (or forced): for exits decided
        by an event (an evaluation's result) rather than by a test the loop repeats."""
        st = self.state
        if st["hold"] and not force:
            self.note_held(to, reason)
            if pending:
                st["pending"] = {"to": to, "reason": reason}
            return False
        st["pending"] = None
        frm = st["phase"]
        self.log("phase %s -> %s: %s" % (frm, to, reason))
        st["transitions"].append({"from": frm, "to": to, "at": time.time(), "reason": reason})
        self.emit("transition", to=to, reason=reason, **{"from": frm})
        if to in ("B1", "B3", "done"):
            self.pool.stop_all()            # the new phase plays no training matches
        st["phase"] = to
        st["stage"] = ""
        st["held_reason"] = ""
        st["phase_started"] = time.time()
        self.force_next = False
        self.enter_phase()
        return True

    def phase_tick(self):
        st = self.state
        p, s = st["phase"], st["stage"]
        force = self.force_next
        if st.get("pending") and (force or not st["hold"]):
            pend = st["pending"]
            self.transition(pend["to"], pend["reason"] + (" (phase next)" if force else " (released)"), force=True)
            self.force_next = False
            return
        if p == "B0":
            done = self.counters.of("b0")["decisions"]
            if force or done >= self.cfg["b0"]["decisions"]:
                self.transition("B1", "phase next" if force else "%d decisions played" % done, force=force)
        elif p == "B1" and s == "train":
            b, step = st["bc"], st.get("bc_step", 0)
            c = self.cfg["b1"]
            reason = None
            if step >= c["max_steps"]:
                reason = "step cap %d" % c["max_steps"]
            elif step >= c["min_steps"] and b["evals_since_best"] >= c["patience"]:
                reason = "held-out agreement plateaued at %.4f" % (b["best_agree"] or 0)
            if force:
                reason = "phase next"
            if reason and (not st["hold"] or force):
                self.end_b1(reason)
            elif reason:
                self.note_held("B2", reason)
        elif p == "B1" and s == "eval":
            if force:
                self.cancel_eval("b1")
                self.transition("B2", "phase next (B1 evaluation skipped)", force=True)
        elif p == "B2" and s == "collect":
            r = st["b2"]["round"]
            done = self.counters.of("b2-r%d" % r)["decisions"]
            steps = st.get("bc_step", 0) - st["b2"]["round_start_step"]
            c = self.cfg["b2"]
            if force or (done >= c["decisions_per_round"] and steps >= c["min_steps_per_round"]):
                self.end_round(r, "phase next" if force else "%d decisions, %d BC steps" % (done, steps))
        elif p == "B2" and s == "eval":
            if force:
                self.cancel_eval("b2-r%d" % st["b2"]["round"])
                self.transition("B3", "phase next (DAgger ended)", force=True)
        elif p == "B4":
            c = self.cfg["b4"]
            it, ls = st["rnad"]["iterations"], st["rnad"]["learner_steps"]
            reason = "phase next" if force else None
            if not reason and c["iterations"] > 0 and it >= c["iterations"]:
                reason = "%d R-NaD iterations" % it
            if not reason and c["max_learner_steps"] > 0 and ls >= c["max_learner_steps"]:
                reason = "%d learner steps" % ls
            if reason:
                self.transition("B5", reason, force=force)
        elif p == "B5":
            c = self.cfg["b5"]
            ls = st["rnad"]["learner_steps"]
            if force or (c["max_learner_steps"] > 0 and ls >= c["max_learner_steps"]):
                if self.transition("done", "phase next" if force else "%d learner steps" % ls, force=force):
                    return
        if p in ("B4", "B5") and not force:
            e = self.cfg["eval"]
            last = st["eval"]["last_at"] or st["phase_started"]
            if e["every_minutes"] > 0 and time.time() - last > 60 * e["every_minutes"] and not self.eval_job \
                    and not st["eval"]["queue"]:
                self.queue_eval("t%d" % st["rnad"]["learner_steps"], self.eval_options(), self.cfg["eval"]["evaluators"])
            last_x = st["eval"].get("exploiter_last_at") or st["phase_started"]
            if e["exploiter_every_minutes"] > 0 and time.time() - last_x > 60 * e["exploiter_every_minutes"] and \
                    not self.eval_queued_kind("exploiter"):
                self.queue_eval("x%d" % st["rnad"]["learner_steps"], self.eval_options(), ["exploiter"])
        self.force_next = False

    # B1

    def on_bc_eval(self, ev):
        b = self.state["bc"]
        agree = ev.get("eval_agree_all")
        b["last_eval"] = {k: v for k, v in ev.items() if not isinstance(v, dict)}
        if agree is None:
            return
        if b["best_agree"] is None or agree > b["best_agree"] + self.cfg["b1"]["min_delta"]:
            b["best_agree"] = agree
            b["evals_since_best"] = 0
        else:
            b["evals_since_best"] += 1

    def end_b1(self, reason):
        st = self.state
        self.log("B1 ends: %s" % reason)
        ck = self.checkpoint(name="bc-b1.pt")
        st["bc"]["b1_final"] = ck["path"]
        self.set_ladder_bc(ck)
        self.queue_export(ck["path"], "bc-b1", ck["version"])
        if self.cfg["b1"]["play_eval"]:
            st["stage"] = "eval"
            self.queue_eval("b1", self.cfg["eval"]["nn_options_bc"], ["fixed"])
            self.save_state()
        else:
            self.transition("B2", reason, force=True)

    def set_ladder_bc(self, ck):
        lad = [r for r in self.state["ladder"] if r["label"] != "bc"]
        self.state["ladder"] = [{"label": "bc", "path": ck["path"], "version": ck["version"]}] + lad

    # B2

    def dagger_mix(self, r):
        c = self.cfg["b2"]
        n = max(c["rounds_max"], 1)
        f = 0.0 if n == 1 else (r - 1) / (n - 1)
        return round(c["teacher_mix_start"] + (c["teacher_mix_end"] - c["teacher_mix_start"]) * min(f, 1.0), 6)

    def start_round(self, resume=False):
        b2 = self.state["b2"]
        r = b2["round"]
        served = self.gpu.call("serve", pid=0)
        if not resume or not b2.get("round_start_step"):
            b2["round_start_step"] = served["version"]
        self.log("DAgger round %d: teacher_mix %.3f, policy version %d" % (r, self.dagger_mix(r), served["version"]))
        self.emit("dagger_round", round=r, teacher_mix=self.dagger_mix(r), version=served["version"])

    def end_round(self, r, reason):
        st = self.state
        self.log("DAgger round %d ends: %s" % (r, reason))
        self.pool.stop_all()
        ck = self.checkpoint(name="bc-r%d.pt" % r)
        self.queue_export(ck["path"], "bc-r%d" % r, ck["version"])
        st["stage"] = "eval"
        self.queue_eval("b2-r%d" % r, self.cfg["eval"]["nn_options_bc"], ["fixed"])
        self.save_state()

    def after_round_eval(self, result):
        st = self.state
        b2 = st["b2"]
        r = b2["round"]
        wr = (result["groups"].get("scripted") or {}).get("win_rate")
        b2["history"].append({"round": r, "win_rate_scripted": wr, "mix": self.dagger_mix(r)})
        c = self.cfg["b2"]
        best = b2["best"]
        improved = wr is not None and (best is None or wr >= best + c["min_improvement"])
        if wr is not None and (best is None or wr > best):
            b2["best"] = wr
        reason = None
        if r >= c["rounds_max"]:
            reason = "round cap %d" % c["rounds_max"]
        elif r >= c["rounds_min"] and not improved:
            reason = "win rate vs scripted:hard stopped improving (%s after %s)" % (_f(wr), _f(best))
        if reason:
            if not self.transition("B3", reason, pending=True):
                return          # held: stays in B2's evaluation stage until released or forced
        else:
            b2["round"] = r + 1
            st["stage"] = "collect"
            self.start_round()
        self.save_state()

    # B3

    def seed_rnad(self):
        st = self.state
        self.ensure_gpu()
        if self.gpu_model == "bc":
            ck = self.checkpoint(name="bc-final.pt")
            path = ck["path"]
            self.set_ladder_bc(ck)
            self.queue_export(path, "bc-final", ck["version"])
        else:
            ck = st["checkpoint"]
            if ck and ck.get("kind") == "rnad":     # seeded before a restart
                return self.transition("B4", "R-NaD seeded (before the restart)", force=True)
            if not ck or ck.get("kind") != "bc":
                raise Fatal("B3 needs a BC checkpoint")
            path = ck["path"]
        st["bc"]["final"] = path
        r = self.gpu.call("rnad_init", bc_checkpoint=path)
        self.gpu_model = "rnad"
        self.gpu_mode = None
        st["rnad"]["learner_steps"] = r["learner_steps"]
        st["rnad"]["iterations"] = r["reg_updates"]
        self.log("R-NaD seeded from %s (reg ε %.3f, sideboarding ε %.3f)" % (path, self.cfg["rnad"]["reg_eps"],
                                                                           self.cfg["rnad"]["reg_eps_sideboarding"]))
        self.checkpoint(name="rnad-seed.pt")
        self.transition("B4", "R-NaD seeded from the BC policy", force=True)

    # B4 / B5

    def on_iteration(self, ev):
        st = self.state
        m = int(ev["reg_updates"])
        st["rnad"]["iterations"] = m
        st["rnad"]["learner_steps"] = int(ev["learner_steps"])
        rung = {"label": "m%d" % m, "path": ev["fixed_point"], "version": int(ev["version"])}
        st["ladder"] = [r for r in st["ladder"] if r["label"] != rung["label"]] + [rung]
        self.emit("rnad_iteration", **ev)
        self.log("R-NaD iteration %d at learner step %d (fixed point %s)" % (m, ev["learner_steps"],
                                                                            ev["fixed_point"]))
        self.queue_export(ev["fixed_point"], "m%d" % m, int(ev["version"]))
        if self.cfg["eval"]["per_iteration"]:
            self.queue_eval("it%d" % m, self.eval_options(), self.cfg["eval"]["evaluators"],
                            policy=dict(rung, iteration=m))
        self.save_state()

    def eval_options(self):
        """The evaluated policy's options in the current phase (the primary result)."""
        p = self.state["phase"]
        e = self.cfg["eval"]
        if p in ("B1", "B2"):
            return e["nn_options_bc"]
        if p == "B5":
            return e["nn_options_finetune"]
        return e["nn_options_rnad"]

    def sampled_options(self):
        """The options the phase's actors play with (the secondary, sampled result: eval.also_sampled)."""
        return self.cfg["b2" if self.state["phase"] in ("B1", "B2") else "b4"]["nn_options"]

    # ── evaluation ──────────────────────────────────────────────────────────────────────────

    def queue_eval(self, label, options, evaluators, policy=None):
        """Queues an evaluation; the SEPARATE evaluators (sprt, exploiter) of `evaluators` become their
        own evaluations, labelled ``<label>-<name>`` (``<label>`` alone when only one is named).
        `policy` ({label, path, version, iteration}): the evaluated policy is that saved network (an
        iteration's fixed point), loaded when the evaluation starts; without it, the current network
        when the evaluation starts. `options` are the primary result's; with eval.also_sampled the
        ``fixed`` evaluator also plays the policy with the actors' sampling options."""
        q = self.state["eval"]["queue"]
        parts = EVAL.split_request(list(evaluators))
        sampled = self.sampled_options() if self.cfg["eval"]["also_sampled"] else None
        for names in parts:
            lab = label if len(parts) == 1 or names[0] not in EVAL.SEPARATE else "%s-%s" % (label, names[0])
            if any(x["label"] == lab for x in q) or (self.eval_job and self.eval_job.label == lab):
                continue
            q.append({"label": lab, "options": options, "sampled_options": sampled, "evaluators": names,
                      "phase": self.state["phase"], "policy": policy})
            self.log("evaluation %s queued (%s; %s%s)" % (lab, ", ".join(names), policy["label"] if policy else
                                                          "the current policy", ", options " +
                                                          EVAL.options_label(options)))

    def eval_queued_kind(self, name):
        """An evaluation naming evaluator `name` is running or queued."""
        running = self.eval_job is not None and name in (getattr(self.eval_job, "request", None) or {}).get(
            "evaluators", [])
        return running or any(name in x["evaluators"] for x in self.state["eval"]["queue"])

    def eval_queued(self, label):
        return (self.eval_job is not None and self.eval_job.label == label) or \
            any(x["label"] == label for x in self.state["eval"]["queue"])

    def cancel_eval(self, label=None):
        if self.eval_job is not None and (label is None or self.eval_job.label == label):
            self.eval_job.cancel()
            self.log("evaluation %s cancelled" % self.eval_job.label)
            self.eval_job = None
            self.state["eval"].pop("running", None)
        self.state["eval"]["queue"] = [x for x in self.state["eval"]["queue"] if label is not None and
                                       x["label"] != label]

    def eval_tick(self):
        st = self.state
        if self.eval_job is not None:
            if st["paused"]:
                self.eval_job.cancel()
                st["eval"]["queue"].insert(0, self.eval_job.request)
                st["eval"].pop("running", None)
                self.log("evaluation %s stopped by the pause (queued again)" % self.eval_job.label)
                self.eval_job = None
                return
            if not self.eval_job.poll():
                return
            job, self.eval_job = self.eval_job, None
            self.finish_eval(job)
            return
        if st["paused"] or self.stopping or not st["eval"]["queue"] or self.gpu is None or not self.gpu_model:
            return
        req = st["eval"]["queue"].pop(0)
        st["eval"]["seq"] += 1
        seq = st["eval"]["seq"]
        pol = req.get("policy")
        ctx = EVAL.Context(self.run, self.cfg, req["options"], st["ladder"],
                           subject_label=pol["label"] if pol else None, sampled_options=req.get("sampled_options"))
        try:
            groups = EVAL.build(ctx, req["evaluators"])
            EVAL.check_groups(groups)
        except ValueError as err:
            return self.fail_eval(req, seq, "not played: %s" % err)
        if pol and not os.path.exists(pol["path"]):
            return self.fail_eval(req, seq, "not played: the fixed point %s is gone" % pol["path"])
        if not groups:
            self.log("evaluation %s: nothing to play" % req["label"])
            return
        spec = next((g for g in groups if isinstance(g, EVAL.ExploitSpec)), None)
        if spec is not None:
            meta = {"phase": st["phase"], "version": None, "options": None, "evaluators": req["evaluators"],
                    "started_at": time.time()}
            job = EVAL.ExploitJob(self.run, self.cfg, seq, req["label"], spec, self.gpu, meta, self.engine,
                                  self.bin_dir, self.log)
            job.request = req
            st["eval"]["running"] = req
            job.start()
            self.eval_job = job
            return
        if pol:
            self.gpu.call("load_policy", pid=EVAL.SUBJECT_PID, path=pol["path"], version=pol["version"])
            version, subject = pol["version"], pol["label"]
        else:
            version = self.gpu.call("snapshot", pid=EVAL.SUBJECT_PID)["version"]
            subject = "v%d" % version
        for ld in [x for g in groups for x in g.loads]:
            self.gpu.call("load_policy", pid=ld["pid"], path=ld["path"], version=ld["version"])
        meta = {"phase": st["phase"], "version": version, "subject": subject, "policy": pol["path"] if pol else None,
                "options": req["options"], "sampled_options": req.get("sampled_options"),
                "evaluators": req["evaluators"], "started_at": time.time()}
        job = EVAL.EvalJob(self.run, self.cfg, seq, req["label"], groups, meta, self.engine, self.bin_dir, self.log)
        job.request = req
        st["eval"]["running"] = req
        job.start()
        self.eval_job = job

    def fail_eval(self, req, seq, reason):
        """An evaluation that could not be played: an error in the log, metrics.jsonl
        (``eval_failed``), evals.jsonl and the status; the run goes on."""
        st = self.state
        st["eval"].pop("running", None)
        self.log("ERROR: evaluation %s %s" % (req["label"], reason))
        self.emit("eval_failed", label=req["label"], error=reason)
        last = {"label": req["label"], "seq": seq, "phase": st["phase"], "version": None, "error": reason,
                "options": req.get("options"), "brief": {}, "seconds": 0}
        st["eval"]["last"] = last
        st["eval"]["last_at"] = time.time()
        st["eval"]["history"] = (st["eval"]["history"] + [last])[-50:]
        self.evals_file.write(dict(last, evaluators=req["evaluators"], time=time.time()))
        self.save_state()

    def finish_eval(self, job):
        st = self.state
        st["eval"].pop("running", None)
        res = job.result
        games = sum(int(g.get("games") or 0) for g in res.get("groups", {}).values())
        failed = res.get("failed_actors", 0)
        if not games:
            res["error"] = "no games played (%d evaluation actors failed%s)" % (
                failed, ": " + "; ".join(res.get("actor_errors") or []) if res.get("actor_errors") else "")
            self.log("ERROR: evaluation %s: %s; see %s" % (res["label"], res["error"], res["dir"]))
        elif failed:
            self.log("WARNING: evaluation %s: %d evaluation actors failed (%s); its results are partial" % (
                res["label"], failed, "; ".join(res.get("actor_errors") or []) or "?"))
        for line in res.pop("draw_lines", []):
            self.note_finding(line, "eval " + res["label"])
        pids = [ld["pid"] for ld in job.loads()]
        if pids and self.gpu is not None:
            self.gpu.call("unload", pids=pids)
        if isinstance(job, EVAL.ExploitJob):
            st["eval"]["exploiter_last_at"] = time.time()
            st["eval"]["exploiter_last"] = EVAL.brief(res)
        elif not res.get("error"):
            try:
                res["ratings"] = st["eval"]["ratings"] = EVAL.record_ladder(self.run, res, st["ladder"])
            except (OSError, ValueError, KeyError) as err:      # the ratings are a report, never fatal
                self.log("evaluation %s: ladder: %s" % (res["label"], err))
        st["eval"]["last"] = {"label": res["label"], "seq": res["seq"], "phase": res["phase"],
                              "version": res["version"], "subject": res.get("subject"),
                              "options": res.get("options"), "brief": EVAL.brief(res), "seconds": res["seconds"]}
        if res.get("error"):
            st["eval"]["last"]["error"] = res["error"]
        st["eval"]["last_at"] = time.time()
        st["eval"]["history"] = (st["eval"]["history"] + [st["eval"]["last"]])[-50:]
        self.evals_file.write(dict(res, time=time.time()))
        self.emit("eval_failed" if res.get("error") else "eval", label=res["label"], version=res["version"],
                  subject=res.get("subject"), options=res["options"],
                  results={k: {"win_rate": g.get("win_rate"), "games": g.get("games"), "options": g.get("options"),
                               "game1": g["game1"].get("win_rate"), "games23": g["games23"].get("win_rate")}
                           for k, g in res["groups"].items()}, seconds=res["seconds"],
                  failed_actors=res["failed_actors"], **({"error": res["error"]} if res.get("error") else {}))
        if not res.get("error"):
            self.log("evaluation %s (%s, %s) done in %.0f s: %s" % (
                res["label"], res.get("subject"), EVAL.options_label(res["options"]), res["seconds"],
                json.dumps(EVAL.brief(res))))
        if st["phase"] == "B1" and st["stage"] == "eval" and res["label"] == "b1":
            wr = (res["groups"].get("scripted") or {}).get("win_rate")
            st["b2"]["best"] = wr
            st["b2"]["history"].append({"round": 0, "win_rate_scripted": wr})
            self.transition("B2", "BC evaluated (vs scripted:hard %s)" % _f(wr), pending=True)
        elif st["phase"] == "B2" and st["stage"] == "eval" and res["label"] == "b2-r%d" % st["b2"]["round"]:
            self.after_round_eval(res)
        self.save_state()

    # ── exports ─────────────────────────────────────────────────────────────────────────────

    def queue_export(self, checkpoint, name, version):
        if not self.cfg["checkpoint"]["export"]:
            return
        self.export_queue.append((checkpoint, name, version))

    def export_tick(self):
        if self.export_proc is not None:
            if self.export_proc.poll() is None:
                return
            ck, name, version, t0 = self.export_job
            rc = self.export_proc.returncode
            self.export_log.close()
            self.export_proc = None
            if rc == 0:
                self.state["exports"].append({"name": name, "checkpoint": ck, "version": version,
                                              "prefix": os.path.join(self.run.exports, name), "at": time.time()})
                self.emit("export", name=name, version=version, seconds=round(time.time() - t0, 1))
                self.log("exported %s (%.0f s)" % (name, time.time() - t0))
            else:
                self.log("export %s failed (exit %s; see logs/export.log)" % (name, rc))
                self.emit("export_failed", name=name, code=rc)
        if not self.export_queue:
            return
        ck, name, version = self.export_queue.pop(0)
        tables = self.cfg["run"]["tables"] or self.run.tables
        argv = [sys.executable, "-m", "rmlearn.orchestrator.exporter", "--tables", tables, "--checkpoint", ck,
                "--prefix", os.path.join(self.run.exports, name), "--version", str(version), "--formats",
                ",".join(self.cfg["checkpoint"]["export_formats"])]
        self.export_log = open(self.run.log("export"), "a", buffering=1)
        self.export_log.write("=== %s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(argv)))
        env = cpu_env()
        env["PYTHONPATH"] = LEARN_DIR
        self.export_proc = subprocess.Popen(argv, cwd=LEARN_DIR, stdout=self.export_log, stderr=subprocess.STDOUT,
                                            env=env, start_new_session=True)
        self.export_job = (ck, name, version, time.time())

    # ── tables ──────────────────────────────────────────────────────────────────────────────

    def ensure_tables(self):
        env = cpu_env()
        if not os.path.exists(self.run.cards):
            self.log("writing the cards frame %s" % self.run.cards)
            subprocess.run([self.engine, "--cards-frame", self.run.cards], cwd=self.bin_dir, env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if self.cfg["run"]["tables"]:
            return
        if not os.path.exists(os.path.join(self.run.tables, "spec.json")):
            rmfeat = os.path.join(self.bin_dir, self.cfg["run"]["build"], "rmfeat")
            self.log("exporting the card tables into %s" % self.run.tables)
            os.makedirs(self.run.tables, exist_ok=True)
            subprocess.run([rmfeat, "--export-tables", self.run.cards, self.run.tables], cwd=self.bin_dir, env=env,
                           check=True, stdout=subprocess.DEVNULL)

    # ── control ─────────────────────────────────────────────────────────────────────────────

    def handle_request(self, req):
        """The control socket's handler (its own thread): status from the cache, the rest through the loop."""
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True, "pid": os.getpid()}
        if cmd == "status":
            with self.status_lock:
                return dict(self.status_cache, ok=True)
        box = {"done": threading.Event(), "reply": None}
        self.requests.put((req, box))
        if not box["done"].wait(timeout=float(req.get("timeout", 120))):
            return {"ok": False, "error": "the orchestrator did not answer in time"}
        return box["reply"]

    def serve_requests(self):
        while True:
            try:
                req, box = self.requests.get_nowait()
            except queue.Empty:
                return
            try:
                reply = self.command(req)
                reply = dict(reply or {}, ok=True)
            except (ValueError, KeyError) as err:
                reply = {"ok": False, "error": str(err)}
            except GpuError as err:
                reply = {"ok": False, "error": "GPU process: %s" % err}
            box["reply"] = reply
            box["done"].set()

    def command(self, req):
        cmd = req.get("cmd")
        st = self.state
        if cmd == "pause":
            if not st["paused"]:
                st["paused"] = True
                self.log("paused")
                self.emit("pause")
                self.save_state()
            return {"paused": True}
        if cmd == "resume":
            if st["paused"]:
                st["paused"] = False
                self.log("resumed")
                self.emit("resume")
                self.save_state()
            return {"paused": False}
        if cmd == "stop":
            self.request_stop("rmtrain stop")
            return {"stopping": True}
        if cmd == "set":
            key, value = C.parse_live(req["key"], str(req["value"]))
            old = C.get_key(self.cfg, key)
            self.cfg = C.set_key(self.cfg, key, value)
            if key in C.GPU_LIVE and self.gpu is not None:
                self.gpu.call("set", key=key, value=value)
            self.pool.settings = self.cfg["actors"]
            write_json(self.run.config, self.cfg)
            self.log("set %s = %s (was %s)" % (key, json.dumps(value), json.dumps(old)))
            self.emit("set", key=key, value=value, old=old)
            return {"key": key, "value": value, "old": old}
        if cmd == "eval":
            if self.gpu is None or not self.gpu_model:
                raise ValueError("no policy to evaluate yet (phase %s)" % st["phase"])
            label = "manual%d" % (st["eval"]["seq"] + 1 + len(st["eval"]["queue"]))
            evaluators = req.get("evaluators") or self.cfg["eval"]["evaluators"]
            self.queue_eval(label, self.eval_options(), evaluators)
            return {"queued": label, "running": self.eval_job.label if self.eval_job else None}
        if cmd == "phase":
            what = req.get("what", "status")
            if what == "hold":
                st["hold"] = True
                self.log("transitions held")
                self.emit("hold")
            elif what == "release":
                st["hold"] = False
                st["held_reason"] = ""
                self.log("transitions released")
                self.emit("release")
            elif what == "next":
                self.force_next = True
                self.log("phase next requested (in %s/%s)" % (st["phase"], st["stage"]))
            elif what != "status":
                raise ValueError("phase takes hold, release, next or status")
            self.save_state()
            return {"phase": st["phase"], "stage": st["stage"], "hold": st["hold"], "held_reason": st["held_reason"],
                    "progress": self.progress()}
        if cmd == "gpu_crash":            # tests: make the GPU process die
            self.gpu.proc.stdin.write(json.dumps({"id": 0, "cmd": "crash"}) + "\n")
            self.gpu.proc.stdin.flush()
            return {}
        raise ValueError("unknown command %r" % cmd)

    # ── status ──────────────────────────────────────────────────────────────────────────────

    def progress(self):
        st = self.state
        p, s = st["phase"], st["stage"]
        c = self.cfg
        if p == "B0":
            return {"decisions": self.counters.of("b0")["decisions"], "target": c["b0"]["decisions"],
                    "games": self.counters.of("b0")["games"]}
        if p == "B1":
            return {"stage": s, "bc_step": st.get("bc_step", 0), "min_steps": c["b1"]["min_steps"],
                    "max_steps": c["b1"]["max_steps"], "best_agree": st["bc"]["best_agree"],
                    "evals_since_best": st["bc"]["evals_since_best"], "patience": c["b1"]["patience"]}
        if p == "B2":
            r = st["b2"]["round"]
            return {"stage": s, "round": r, "rounds_max": c["b2"]["rounds_max"], "teacher_mix": self.dagger_mix(r),
                    "decisions": self.counters.of("b2-r%d" % r)["decisions"],
                    "decisions_per_round": c["b2"]["decisions_per_round"],
                    "bc_steps_in_round": st.get("bc_step", 0) - st["b2"]["round_start_step"],
                    "best_win_rate": st["b2"]["best"], "history": st["b2"]["history"]}
        if p in ("B4", "B5"):
            return {"learner_steps": st["rnad"]["learner_steps"], "iterations": st["rnad"]["iterations"],
                    "b4_iterations": c["b4"]["iterations"], "b4_max_learner_steps": c["b4"]["max_learner_steps"],
                    "b5_max_learner_steps": c["b5"]["max_learner_steps"],
                    "selfplay_games": self.counters.of(st["selfplay_version"])["games"]}
        return {}

    def write_status(self, running=True):
        if self.state is None:
            return
        now = time.time()
        if running and self.gpu is not None and self.gpu.alive() and now - self.gpu_stats_at > 5:
            try:
                self.gpu_stats = self.gpu.call("stats", timeout=10)
                self.gpu_stats.pop("re", None)
                self.gpu_stats.pop("ok", None)
            except GpuError as err:
                self.gpu_stats = {"error": str(err)}
            self.gpu_stats_at = now
            if self.gpu_model == "rnad" and self.gpu_stats.get("learner_steps") is not None:
                rn = self.state["rnad"]
                rn["learner_steps"] = max(rn["learner_steps"], int(self.gpu_stats["learner_steps"]))
        st = self.state
        health = self.pool.health() if self.pool else []
        doc = {"run": self.run.name, "path": self.run.path, "running": running, "pid": os.getpid(),
               "updated_at": now, "started_at": self.started, "uptime_s": now - self.started,
               "phase": st["phase"], "stage": st["stage"], "hold": st["hold"], "held_reason": st.get("held_reason"),
               "paused": st["paused"], "stopping": self.stopping, "phase_started": st["phase_started"],
               "progress": self.progress(), "totals": dict(self.counters.total, **self.counters.rates()),
               "actors": {"wanted": self.pool.count if self.pool else 0, "running": self.pool.running() if self.pool else 0,
                          "restarts": sum(a["restarts"] for a in health), "health": health},
               "gpu": self.gpu_stats if running else {}, "metrics": self.last_metrics,
               "eval": {"running": self.eval_job.label if self.eval_job else None,
                        "queue": [x["label"] for x in st["eval"]["queue"]], "last": st["eval"]["last"],
                        "ratings": st["eval"].get("ratings"), "exploiter_last": st["eval"].get("exploiter_last"),
                        "exploiter": getattr(self.eval_job, "status", None) if isinstance(
                            self.eval_job, EVAL.ExploitJob) else None},
               "exports": {"running": self.export_job[1] if self.export_proc else None,
                           "queue": [x[1] for x in self.export_queue], "done": len(st["exports"])},
               "checkpoint": st["checkpoint"], "transitions": st["transitions"][-6:], "findings": st.get("findings"),
               "config": {k: C.get_key(self.cfg, k) for k in sorted(C.LIVE)}, "sockets": {
                   "control": self.run.control_sock, "rmip": self.run.rmip_sock}}
        with self.status_lock:
            self.status_cache = doc
        write_json(self.run.status, doc)

    # ── shutdown ────────────────────────────────────────────────────────────────────────────

    def shutdown(self, reason, checkpoint=True):
        self.log("stopping: %s" % reason)
        self.stopping = self.stopping or reason
        try:
            if self.eval_job is not None:
                self.eval_job.cancel()
                self.state["eval"]["queue"].insert(0, self.eval_job.request)
                self.state["eval"].pop("running", None)
                self.eval_job = None
            if self.pool is not None:
                self.pool.stop_all()
            if checkpoint:
                while self.export_proc is not None or self.export_queue:     # the queued exports finish
                    self.export_tick()
                    time.sleep(0.5)
            elif self.export_proc is not None:
                try:
                    self.export_proc.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    self.export_proc.kill()
            if self.gpu is not None and self.gpu.alive():
                if checkpoint and self.gpu_model and self.state["phase"] not in ("B3", "done"):
                    try:
                        self.checkpoint()
                    except GpuError as err:
                        self.log("final checkpoint failed: %s" % err)
                self.gpu.stop()
            elif self.gpu is not None:
                self.gpu.kill()
        finally:
            self.ingest_results()
            self.save_state()
            self.emit("stop", reason=reason)


def _f(x):
    return "%.3f" % x if isinstance(x, (int, float)) else "-"


def _iteration_of(label):
    """k of a fixed point's label ``m<k>`` (0 for another label)."""
    return int(label[1:]) if label[:1] == "m" and label[1:].isdigit() else 0


def status_from_files(run):
    """The last status.json with the state's phase (for a run that is not running)."""
    doc = read_json(run.status) or {}
    st = read_json(run.state) or {}
    if st:
        doc.setdefault("phase", st.get("phase"))
        doc["phase"] = st.get("phase", doc.get("phase"))
        doc["stage"] = st.get("stage", doc.get("stage"))
        doc["hold"] = st.get("hold")
        doc["paused"] = st.get("paused")
        doc["checkpoint"] = st.get("checkpoint")
        doc["transitions"] = st.get("transitions", [])[-6:]
        doc.setdefault("eval", {})["last"] = (st.get("eval") or {}).get("last")
        doc["eval"]["ratings"] = (st.get("eval") or {}).get("ratings")
        doc["findings"] = st.get("findings")
    doc["running"] = False
    doc.pop("stopping", None)
    if st.get("phase") == "done":
        doc["finished"] = True
    doc["recent_metrics"] = read_jsonl(run.metrics, last=5)
    return doc
