#!/usr/bin/env python3
"""Integration checks and benches of rmlearn against the engine's C++ actor (needs torch, numpy and
a built ``robomage_nn``; never run by ``make``; not part of ``check.py``, which is CPU-only and
needs no engine).

    python learn/integration.py actor  --tables DIR --work DIR [--engine PATH] [--bin-dir DIR]
    python learn/integration.py server --tables DIR --work DIR [--device cuda] [...]
    python learn/integration.py bench  --tables DIR --work DIR [--device cuda] [--actors 8,24] [...]

- ``actor``: a seeded model of the default config exported as TorchScript and ``.pt2``; each
  export plays a bo3 match (argmax) against explore in ``robomage_nn`` (the two play the same
  match), a sampled nn-vs-nn bo3 match; the C++ in-process evaluator's logits and values on every
  decision it evaluated equal the Python model's (batch 1 and padded batches) within 1e-4.
- ``server``: the inference server (``rmlearn.server``) in this process on ``--device``: argmax
  bo1 games through ``nn:remote:`` by concurrent actors are the in-process games decision for
  decision, the evaluations within 1e-5 of the in-process C++ evaluator's (policy 0; the
  regularisation policies' within 1e-4 of the Python model on the CPU); concurrent sampled bo3
  matches (nn vs explore, nn vs nn) with a hot policy swap mid-match; handshake refusals; then a
  standalone CPU server killed mid-game makes its actor fail loudly.
- ``bench``: the default model's forward on ``--device`` at batch 64 / 256 / 512 (rows/s,
  latency, peak memory; fp32 and the fp16 / bf16 autocast deviations), then the end-to-end
  decisions/s through the server with N concurrent nn-vs-nn actors.

Engine processes run with the GPU hidden (HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES empty): only
this process may touch it. Run from anywhere; ``--bin-dir`` is the engine's working directory (its
``resources/`` holds the card scripts and decks), ``--engine`` the ``robomage_nn`` binary. Unix
socket paths must be short (108 bytes): sockets go under ``--sock-dir`` (default /tmp/rmlearn-<pid>).
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rmlearn import batch as B  # noqa: E402
from rmlearn import client as CL  # noqa: E402
from rmlearn import export as E  # noqa: E402
from rmlearn import model as M  # noqa: E402
from rmlearn import rmtb  # noqa: E402
from rmlearn import server as SV  # noqa: E402
from rmlearn import spec as S  # noqa: E402

REPO_ROOT = os.path.dirname(HERE)
CPU_ENV = dict(os.environ, HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
FAILURES = []
CONFIG = [M.ModelConfig()]   # the model config of every built model (--config)


def say(msg):
    print("integration: " + msg, flush=True)


def fail(msg):
    FAILURES.append(msg)
    print("FAIL: " + msg, flush=True)


def close(a, b, tol):
    return np.abs(a - b) <= tol + tol * np.abs(b)


# ── engine processes ─────────────────────────────────────────────────────────────────────────

class Engine:
    def __init__(self, args):
        self.engine = args.engine
        self.bin_dir = args.bin_dir
        self.work = args.work

    def start(self, argv, name):
        out = open(os.path.join(self.work, name + ".txt"), "w")
        p = subprocess.Popen([self.engine] + argv, cwd=self.bin_dir, env=CPU_ENV, stdout=out,
                             stderr=subprocess.STDOUT)
        p.name, p.log, p.started = name, out.name, time.time()
        return p

    @staticmethod
    def finish(p, timeout=3600):
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        with open(p.log) as fp:
            p.text = fp.read()
        return p


def problem(p, bo3):
    """"" when the process ran to a result with no FATAL / ERROR line, else what went wrong."""
    if p.returncode != 0:
        return "%s exited %d (see %s)" % (p.name, p.returncode, p.log)
    lines = p.text.splitlines()
    bad = [ln for ln in lines if ln.startswith("FATAL") or ln.startswith("ERROR") or "WARNING" in ln]
    if bad:
        return "%s: %s (see %s)" % (p.name, bad[0], p.log)
    key = "MATCH_RESULT: " if bo3 else "GAME_RESULT: "
    if not any(ln.startswith(key) for ln in lines):
        return "%s: no %s line (see %s)" % (p.name, key.strip(), p.log)
    if any(ln.startswith("GAME_RESULT: ") and ln.endswith(" draw") for ln in lines):
        return "%s: a drawn game (see %s)" % (p.name, p.log)
    return ""


def result_lines(p):
    return [ln for ln in p.text.splitlines() if ln.startswith("MATCH_RESULT") or ln.startswith("GAME_RESULT")]


def decisions_of(rmlog):
    with open(rmlog) as fp:
        text = fp.read()
    at = text.find("DECISIONS\n")
    return text[at + 10:] if at >= 0 else ""


def read_evals(path):
    meta, a = rmtb.read(path)
    if meta.get("kind") != "evaluations":
        raise ValueError("%s is not an evaluations file" % path)
    return a


# ── shared setup ─────────────────────────────────────────────────────────────────────────────

def build_model(tables, seed, cfg=None):
    torch.manual_seed(seed)
    model = M.RMNet(cfg or CONFIG[0], tables)
    # Every parameter perturbed so the zero-initialised relation biases and LayerNorms are not trivial.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    return model.eval()


def python_eval(policy, decs, indices, batch, tables):
    """[(logits over real rows, value)] of `decs` at `indices` from `policy` (CPU), in batches."""
    out = []
    for s in range(0, len(indices), batch):
        sub = decs.select(indices[s:s + batch])
        x = B.to_torch(B.pad(sub, tables.spec))
        with torch.no_grad():
            lg, v = policy(*x)
        for i in range(len(sub)):
            a = int(sub.sizes["a"][i])
            out.append((lg[i, :a].numpy().astype(np.float64), float(v[i])))
    return out


def compare_with_evals(name, ref, ev, column, tol):
    """Worst |logit| / |log pi| / |value| differences of evals file `ev` (policy `column`) against
    `ref` ([(logits, value)] per decision); a failure past `tol`."""
    off = ev["menu_offsets"]
    d = len(off) - 1
    if d != len(ref):
        fail("%s: %d evaluations, %d reference decisions" % (name, d, len(ref)))
        return
    worst = [0.0, 0.0, 0.0]
    ok = True
    for i in range(d):
        lg = ev["logits"][off[i]:off[i + 1], column].astype(np.float64)
        lp = ev["log_pi"][off[i]:off[i + 1], column].astype(np.float64)
        rl, rv = ref[i]
        rlp = rl - (rl.max() + np.log(np.exp(rl - rl.max()).sum()))
        v = float(ev["value"][i, column])
        worst[0] = max(worst[0], float(np.abs(lg - rl).max()))
        worst[1] = max(worst[1], float(np.abs(lp - rlp).max()))
        worst[2] = max(worst[2], abs(v - rv))
        ok = ok and bool(close(lg, rl, tol).all()) and bool(close(lp, rlp, tol).all()) and \
            bool(close(np.array(v), np.array(rv), tol))
    say("%s: %d decisions, max |logit| %.3g, |log pi| %.3g, |value| %.3g (tolerance %g)"
        % (name, d, worst[0], worst[1], worst[2], tol))
    if not ok:
        fail("%s past the tolerance %g" % (name, tol))


def nn_spec(model_path, **opts):
    return "nn:" + model_path + "".join(":" + (k if v is True else "%s=%s" % (k, v)) for k, v in opts.items())


def remote_spec(sock, **opts):
    return "nn:remote:" + sock + "".join(":" + (k if v is True else "%s=%s" % (k, v)) for k, v in opts.items())


def match_argv(spec_a, spec_b, deck_a, deck_b, seed, bo3, rmlog=None):
    argv = ["--agent-a", spec_a, "--agent-b", spec_b, "--deck-a", deck_a, "--deck-b", deck_b, "--game-seed",
            str(seed)]
    if bo3:
        argv.append("--bo3")
    if rmlog:
        argv += ["--decision-log", rmlog]
    return argv


# ── actor ────────────────────────────────────────────────────────────────────────────────────

def cmd_actor(args, tables):
    eng = Engine(args)
    model = build_model(tables, args.seed)
    say("model: %d parameters (%s)" % (model.param_count(), json.dumps(model.cfg.to_dict())))
    policy = model.inference_module(version=5)
    prefix = os.path.join(args.work, "actor_policy")
    t = time.time()
    paths = E.export_policy(policy, prefix, 5, formats=("ts", "pt2"))
    say("exported %s in %.1f s" % (", ".join(sorted(paths.values())), time.time() - t))
    for f, p in paths.items():
        meta = E.read_metadata(p)
        want = E.metadata(policy)
        if meta != want:
            fail("%s metadata %r, expected %r" % (f, meta, want))
    w = args.work
    decks = ("league/ur_delver", "league/burn")
    runs = {
        "ts": eng.start(match_argv(nn_spec(paths["ts"], argmax=True, dump=w + "/ts_A.rmtb", evals=w + "/ts_A_ev.rmtb"),
                                   "explore", *decks, args.seed, True, w + "/ts.rmlog"), "actor_ts"),
        "pt2": eng.start(match_argv(nn_spec(paths["pt2"], argmax=True, evals=w + "/pt2_A_ev.rmtb"), "explore", *decks,
                                    args.seed, True, w + "/pt2.rmlog"), "actor_pt2"),
        "nn_vs_nn": eng.start(match_argv(nn_spec(paths["ts"], temp=1), nn_spec(paths["pt2"], temp=1),
                                         "league/bw_dnt", "league/reanimator", args.seed + 1, True), "actor_nn_vs_nn"),
    }
    for k, p in runs.items():
        Engine.finish(p)
        pr = problem(p, True)
        if pr:
            fail(pr)
        else:
            say("%s: %s in %.0f s" % (k, " / ".join(result_lines(p)), time.time() - p.started))
    if FAILURES:
        return
    if decisions_of(w + "/ts.rmlog") != decisions_of(w + "/pt2.rmlog"):
        fail("the TorchScript and .pt2 exports did not play the same argmax match")
    else:
        say("argmax: the TorchScript and .pt2 exports play the same bo3 match (%d decisions)"
            % len(decisions_of(w + "/ts.rmlog").splitlines()))
    decs = B.Decisions.from_file(w + "/ts_A.rmtb")
    idx = list(range(len(decs)))
    ref1 = python_eval(policy, decs, idx, 1, tables)
    ref16 = python_eval(policy, decs, idx, 16, tables)
    eager = model.cpu().eval()

    class Eager(torch.nn.Module):
        def forward(self, *x):
            o = eager(*x)
            return o["logits"], o["value"]
    ref_train = python_eval(Eager(), decs, idx, 16, tables)
    compare_with_evals("C++ in-process TorchScript vs Python (batch 1)", ref1, read_evals(w + "/ts_A_ev.rmtb"), 0, 1e-4)
    compare_with_evals("C++ in-process TorchScript vs Python (padded batches of 16)", ref16,
                       read_evals(w + "/ts_A_ev.rmtb"), 0, 1e-4)
    compare_with_evals("C++ in-process TorchScript vs the training model (unfolded, eval mode)", ref_train,
                       read_evals(w + "/ts_A_ev.rmtb"), 0, 1e-4)
    compare_with_evals("C++ in-process .pt2 vs Python (batch 1)", ref1, read_evals(w + "/pt2_A_ev.rmtb"), 0, 1e-4)


# ── server ───────────────────────────────────────────────────────────────────────────────────

def start_server(args, tables, policies, max_batch=256, deadline_us=2000, precision="fp32"):
    sock = os.path.join(args.sock_dir, "srv.sock")
    srv = SV.InferenceServer(sock, device=args.device, max_batch=max_batch, deadline_us=deadline_us,
                             precision=precision)
    for pid, (m, ver) in policies.items():
        srv.set_policy(pid, m, ver)
    srv.start()
    return srv, sock


def cmd_server(args, tables):
    eng = Engine(args)
    w = args.work
    cur = build_model(tables, args.seed)
    reg = build_model(tables, args.seed + 100)
    ts = E.export_policy(cur, os.path.join(w, "server_cur"), 3, formats=("ts",))["ts"]
    srv, sock = start_server(args, tables, {0: (cur, 3), 1: (reg, 7), 2: (cur, 8)})
    say("server on %s: policies %s" % (args.device, srv.policies()))

    # 1. Argmax bo1 games, in process and through the server, concurrently.
    games = [("league/bug", "league/burn"), ("league/ur_delver", "league/uw_control"),
             ("league/reanimator", "league/bw_dnt"), ("league/gw_maverick", "league/wrb_energy")]
    procs = []
    for g, (da, db) in enumerate(games):
        seed = args.seed + 10 + g
        local = nn_spec(ts, argmax=True, evals="%s/local_%d_%%s.rmtb" % (w, g))
        procs.append(("local", g, eng.start(match_argv(local % "A", local % "B", da, db, seed, False,
                                                       "%s/local_%d.rmlog" % (w, g)), "local_%d" % g)))
        ra = remote_spec(sock, argmax=True, stats=True, timeout=300, policies="0,1,2",
                         dump="%s/remote_%d_A_dump.rmtb" % (w, g), evals="%s/remote_%d_A.rmtb" % (w, g))
        rb = remote_spec(sock, argmax=True, stats=True, timeout=300, evals="%s/remote_%d_B.rmtb" % (w, g))
        procs.append(("remote", g, eng.start(match_argv(ra, rb, da, db, seed, False, "%s/remote_%d.rmlog" % (w, g)),
                                             "remote_%d" % g)))
    t0 = time.time()
    for _, _, p in procs:
        Engine.finish(p)
    say("%d argmax bo1 games (%d in process, %d through the server) in %.0f s"
        % (len(procs), len(games), len(games), time.time() - t0))
    say("server after the argmax games: %s" % short_stats(srv.stats()))
    reg_policy = reg.inference_module(7)
    for kind, g, p in procs:
        pr = problem(p, False)
        if pr:
            fail(pr)
    if not FAILURES:
        n_dec = 0
        for g in range(len(games)):
            want = decisions_of("%s/local_%d.rmlog" % (w, g))
            if want != decisions_of("%s/remote_%d.rmlog" % (w, g)):
                fail("remote game %d is not the in-process game" % g)
                continue
            n_dec += len(want.splitlines())
            for seat in "AB":
                loc = read_evals("%s/local_%d_%s.rmtb" % (w, g, seat))
                rem = read_evals("%s/remote_%d_%s.rmtb" % (w, g, seat))
                ref = [(loc["logits"][loc["menu_offsets"][i]:loc["menu_offsets"][i + 1], 0].astype(np.float64),
                        float(loc["value"][i, 0])) for i in range(len(loc["menu_offsets"]) - 1)]
                compare_with_evals("game %d seat %s: server (%s) vs C++ in-process, policy 0" % (g, seat, args.device),
                                   ref, rem, 0, 1e-5)
                vers = sorted(set(rem["version"][:, 0].tolist()))
                if vers != [3]:
                    fail("game %d seat %s: policy 0 versions %s, expected [3]" % (g, seat, vers))
                if seat == "A":
                    if sorted(set(rem["version"][:, 1].tolist())) != [7] or \
                            sorted(set(rem["version"][:, 2].tolist())) != [8]:
                        fail("game %d: policies 1, 2 versions are not 7, 8" % g)
                    dump = B.Decisions.from_file("%s/remote_%d_A_dump.rmtb" % (w, g))
                    idx = list(range(len(dump)))
                    compare_with_evals("game %d seat A: server policy 1 (reg) vs Python CPU" % g,
                                       python_eval(reg_policy, dump, idx, 16, tables), rem, 1, 1e-4)
                    compare_with_evals("game %d seat A: server policy 2 (= cur) vs its policy 0" % g,
                                       [(rem["logits"][rem["menu_offsets"][i]:rem["menu_offsets"][i + 1], 0]
                                         .astype(np.float64), float(rem["value"][i, 0])) for i in idx], rem, 2, 1e-6)
        say("argmax games through the server equal the in-process ones (%d decisions)" % n_dec)

    # 2. Concurrent sampled bo3 matches, with a hot swap of policy 0 mid-way.
    matches = [("nn_vs_explore_%d" % i, remote_spec(sock, temp=1, timeout=300, evals="%s/m%d_A.rmtb" % (w, i)),
                "explore", games[i % 4][0], games[i % 4][1], args.seed + 20 + i) for i in range(4)]
    matches += [("nn_vs_nn_%d" % i, remote_spec(sock, temp=1, timeout=300, policies="0,1,2",
                                                evals="%s/n%d_A.rmtb" % (w, i)),
                 remote_spec(sock, temp=1, timeout=300), games[(i + 1) % 4][1], games[i % 4][0], args.seed + 30 + i)
                for i in range(2)]
    served = srv.stats()["decisions"]
    procs = [(m[0], eng.start(match_argv(m[1], m[2], m[3], m[4], m[5], True), m[0])) for m in matches]
    t = time.time()
    while srv.stats()["decisions"] < served + args.swap_after and time.time() - t < 600:
        time.sleep(0.01)
    swapped = build_model(tables, args.seed + 200)
    t = time.perf_counter()
    srv.set_policy(0, swapped, 4)
    say("set_policy(0, version 4) while %d matches run: %.1f ms" % (len(procs), (time.perf_counter() - t) * 1e3))
    for name, p in procs:
        Engine.finish(p)
        pr = problem(p, True)
        if pr:
            fail(pr)
        else:
            say("%s: %s (%.0f s)" % (name, " / ".join(result_lines(p)), time.time() - p.started))
    seen = set()
    for name, _ in procs:
        f = "%s/%s_A.rmtb" % (w, name.replace("nn_vs_explore_", "m").replace("nn_vs_nn_", "n"))
        if os.path.exists(f):
            seen |= set(read_evals(f)["version"][:, 0].tolist())
    say("policy 0 versions the matches saw: %s" % sorted(seen))
    if not {3, 4} <= seen:
        fail("the hot swap was not seen by the matches (versions %s)" % sorted(seen))
    say("server after the matches: %s" % short_stats(srv.stats()))

    # 3. Handshake refusals.
    base = {"protocol": 1, "client": "integration", "spec_hash": tables.hashes["spec_hash"],
            "vocab_hash": tables.hashes["vocab_hash"], "text_model": tables.hashes["text_model"],
            "dims": tables.spec.dims(), "shm": "/rmip-none", "request_bytes": 64, "response_bytes": 64,
            "policies": [0]}
    cases = [("spec_hash", dict(base, spec_hash="0123456789abcdef")),
             ("vocab_hash", dict(base, vocab_hash="0123456789abcdef")),
             ("text_model", dict(base, text_model="other@0")), ("protocol", dict(base, protocol=2)),
             ("policy", dict(base, policies=[9])), ("shm", base)]
    for code, hello in cases:
        got = CL.hello_only(sock, hello)
        if got != ("ERROR", code):
            fail("a HELLO expecting ERROR %s got %r" % (code, got))
    p = Engine.finish(eng.start(["--agent-a", remote_spec(sock, policies=9), "--agent-b", "explore", "--game-seed", "1"],
                                "refused_policy"))
    if p.returncode == 0 or "not served" not in p.text or sock not in p.text:
        fail("nn:remote:...:policies=9 did not fail naming the endpoint (see %s)" % p.log)
    say("handshake: refused %s; nn:remote: asking for policy 9 exits naming the endpoint"
        % ", ".join(c for c, _ in cases))
    srv.stop()
    kill_check(args, eng, ts)


def short_stats(st):
    keys = ("batches", "decisions", "requests", "refusals", "errors", "rows_per_s", "batch_mean", "batch_max",
            "queue_ms_mean", "queue_ms_p99", "compute_ms_mean", "vram_peak_mb")
    return ", ".join("%s %s" % (k, ("%.1f" % st[k]) if isinstance(st[k], float) else st[k]) for k in keys if k in st)


def kill_check(args, eng, ts):
    """A standalone CPU server that stops answering after 40 requests is SIGKILLed: the actor fails
    loudly, naming the endpoint, with the answered decisions logged."""
    sock = os.path.join(args.sock_dir, "kill.sock")
    log = open(os.path.join(args.work, "kill_server.txt"), "w")
    srv = subprocess.Popen([sys.executable, "-m", "rmlearn.server", "--socket", sock, "--policy", "0=" + ts,
                            "--device", "cpu", "--pause-after", "40"], cwd=HERE, env=CPU_ENV, stdout=log,
                           stderr=subprocess.STDOUT)
    if not wait_for(log.name, "listening on", 60):
        fail("the standalone server did not start (see %s)" % log.name)
        srv.kill()
        return
    spec = remote_spec(sock, argmax=True, timeout=300)
    rmlog = os.path.join(args.work, "kill.rmlog")
    actor = eng.start(match_argv(spec, spec, "league/ur_delver", "league/uw_control", args.seed + 12, False, rmlog),
                      "kill_actor")
    paused = wait_for(log.name, "paused after", 300)
    srv.send_signal(signal.SIGKILL)
    srv.wait()
    t0 = time.time()
    Engine.finish(actor, timeout=120)
    waited = time.time() - t0
    fatal = [ln for ln in actor.text.splitlines() if ln.startswith("FATAL")]
    n = len(decisions_of(rmlog).splitlines()) if os.path.exists(rmlog) else -1
    if not paused or actor.returncode == 0 or not fatal or sock not in fatal[-1] or n != 40:
        fail("kill: after the server was killed the actor exited %s with %d logged decisions and %r (see %s)"
             % (actor.returncode, n, fatal[-1:] or "", actor.log))
    else:
        say("kill: server SIGKILLed after 40 answers: the actor exited %d %.1f s later, %d decisions logged, %r"
            % (actor.returncode, waited, n, fatal[-1]))


def wait_for(path, text, timeout):
    t = time.time()
    while time.time() - t < timeout:
        with open(path) as fp:
            if text in fp.read():
                return True
        time.sleep(0.1)
    return False


# ── bench ────────────────────────────────────────────────────────────────────────────────────

def cmd_bench(args, tables):
    dev = torch.device(args.device)
    model = build_model(tables, args.seed)
    say("model: %d parameters (%s)" % (model.param_count(), json.dumps(model.cfg.to_dict())))
    policy = model.inference_module(1).to(dev).eval()
    pool = []
    for f in args.samples:
        pool.append(B.Decisions.from_file(f))
    all_idx = [(k, i) for k, d in enumerate(pool) for i in range(len(d))]
    rng = np.random.default_rng(0)
    for bsz in args.batches:
        pick = [all_idx[j] for j in rng.integers(0, len(all_idx), bsz)]
        decs = SV._concat([pool[k].select([i]) for k, i in pick])
        t = time.perf_counter()
        x = B.pad(decs, tables.spec)
        t_pad = time.perf_counter() - t
        inputs = [torch.from_numpy(x[k]).to(dev) for k in B.INPUTS]
        res = {}
        for prec in ("fp32", "fp16", "bf16"):
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            with torch.inference_mode():
                def run():
                    if prec == "fp32":
                        return policy(*inputs)
                    dt = torch.float16 if prec == "fp16" else torch.bfloat16
                    with torch.autocast(device_type=dev.type, dtype=dt):
                        return policy(*inputs)
                for _ in range(3):
                    out = run()
                sync(dev)
                iters = 10
                t = time.perf_counter()
                for _ in range(iters):
                    out = run()
                sync(dev)
                dt_s = (time.perf_counter() - t) / iters
            lg, v = out[0].float(), out[1].float()
            peak = torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else 0
            res[prec] = (lg, v)
            line = "batch %d (N %d, R %d, A %d) %s: %.1f ms per forward, %.0f rows/s, peak %.0f MiB" % (
                bsz, x["tok_type"].shape[1], x["rel"].shape[1], x["act_tok"].shape[1], prec, dt_s * 1e3, bsz / dt_s, peak)
            if prec != "fp32":
                m = inputs[9]
                ref_lg, ref_v = res["fp32"]
                dl = (lg - ref_lg)[m].abs().max().item()
                dv = (v - ref_v).abs().max().item()
                agree = ((lg.masked_fill(~m, -1e30).argmax(1) == ref_lg.masked_fill(~m, -1e30).argmax(1))
                         .float().mean().item())
                line += "; vs fp32 max |logit| %.3g, |value| %.3g, argmax agreement %.1f%%" % (dl, dv, 100 * agree)
            say(line)
        say("batch %d: padding on the CPU %.1f ms" % (bsz, t_pad * 1e3))
    del policy
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    eng = Engine(args)
    for n in args.actors:
        e2e(args, tables, model, eng, n)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def e2e(args, tables, model, eng, n_actors):
    """N concurrent actors (nn:remote both seats, sampled, bo1 games back to back) for --seconds."""
    srv, sock = start_server(args, tables, {0: (model, 1)}, max_batch=512, deadline_us=args.deadline_us,
                             precision=args.precision)
    decks = ["league/bug", "league/burn", "league/ur_delver", "league/uw_control", "league/reanimator",
             "league/bw_dnt", "league/gw_maverick", "league/wrb_energy", "league/lands", "league/wubg_doomsday"]
    spec = remote_spec(sock, temp=1, timeout=300)
    procs = {}
    games = [0]

    def launch(k):
        g = games[0]
        games[0] += 1
        argv = match_argv(spec, spec, decks[g % len(decks)], decks[(g * 3 + 1) % len(decks)], 1000 + g, False)
        procs[k] = eng.start(argv + ["--max-decisions", "3000"], "e2e_%d_%d" % (n_actors, k))

    for k in range(n_actors):
        launch(k)
    time.sleep(args.warmup)
    st0 = srv.stats()
    t0 = time.time()
    finished = 0
    while time.time() - t0 < args.seconds:
        time.sleep(0.2)
        for k, p in list(procs.items()):
            if p.poll() is not None:
                Engine.finish(p)
                pr = problem(p, False)
                if pr and "MAX_DECISIONS" not in p.text:
                    fail("e2e: " + pr)
                finished += 1
                launch(k)
    st1 = srv.stats()
    dt = time.time() - t0
    for p in procs.values():
        p.send_signal(signal.SIGKILL)
        p.wait()
    srv.stop()
    rows = st1["decisions"] - st0["decisions"]
    batches = st1["batches"] - st0["batches"]
    say("e2e %d actors (%d clients, nn:remote both seats, %s): %.0f decisions/s over %.0f s, mean batch %.1f, "
        "%d games finished; server window: %s" % (n_actors, 2 * n_actors, args.precision, rows / dt, dt,
                                                rows / max(batches, 1), finished, short_stats(st1)))


# ── main ─────────────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["actor", "server", "bench"])
    ap.add_argument("--tables", required=True, help="rmfeat --export-tables DIR")
    ap.add_argument("--work", required=True, help="output directory")
    ap.add_argument("--bin-dir", default=os.path.join(REPO_ROOT, "bin"), help="the engine's working directory")
    ap.add_argument("--engine", help="robomage_nn (default <bin-dir>/debug/robomage_nn)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sock-dir", default="/tmp/rmlearn-%d" % os.getpid())
    ap.add_argument("--swap-after", type=int, default=300, help="server: decisions served before the hot swap")
    ap.add_argument("--samples", nargs="*", default=[], help="bench: tensors files to draw batches from")
    ap.add_argument("--batches", default="64,256,512")
    ap.add_argument("--actors", default="8,24")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--warmup", type=float, default=15.0)
    ap.add_argument("--deadline-us", type=int, default=2000)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--config", help="ModelConfig fields as JSON (default: the default config)")
    args = ap.parse_args()
    if args.config:
        CONFIG[0] = M.ModelConfig.from_dict(json.loads(args.config))
    args.engine = args.engine or os.path.join(args.bin_dir, "debug", "robomage_nn")
    args.batches = [int(x) for x in args.batches.split(",") if x]
    args.actors = [int(x) for x in args.actors.split(",") if x]
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.sock_dir, exist_ok=True)
    args.work = os.path.abspath(args.work)
    torch.set_num_threads(4)
    tables = S.load_tables(args.tables)
    {"actor": cmd_actor, "server": cmd_server, "bench": cmd_bench}[args.cmd](args, tables)
    print("INTEGRATION: %s (%d failure%s)" % ("ok" if not FAILURES else "FAIL", len(FAILURES),
                                             "" if len(FAILURES) == 1 else "s"), flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
