#!/usr/bin/env python3
"""rmlearn's CPU-only correctness checks (torch + numpy; the GPU is hidden from this process).

    learn/.venv/bin/python learn/check.py --tables DIR --tensors A.rmtb [--omniscient O.rmtb] [--targets T.rmtb]
    learn/.venv/bin/python learn/check.py --engine-dir bin/debug --bin-dir bin      # make the inputs first

Inputs: the exported tables (``rmfeat --export-tables``: ``card.rmtb``, ``ability.rmtb``,
``part.rmtb``, ``spec.json``; the text table beside them or the committed one), a seat's tensors file
(``rmfeat --featurize ... --viewer A``) and optionally the omniscient tensors and the seat's targets
of the same dump. With ``--engine-dir`` (``robomage`` and ``rmfeat``) and ``--bin-dir`` (the
engine's working directory) the inputs are made from one short explore game into ``--work``.

Checks: the numpy RMTB reader against ``python/rmtb.py`` and its writer's round trip; the tables'
hashes; the model's forward (shapes, finite outputs, padded rows), padding invariance (a decision's
outputs do not depend on its batch or its padding), the training model against its folded
inference module, auxiliary losses and a backward pass, the omniscient critic on paired decisions,
the smoothed policy; the TorchScript and ``.pt2`` exports' metadata and outputs; the inference
server on the CPU through the Python client (results against the model, several policies and
versions, a swap, batching across concurrent clients, handshake and request refusals, shutdown).
Prints ``RMLEARN-CHECK: ok`` and exits 0, or the failures and exits 1.
"""

import os

os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from rmlearn import batch as B  # noqa: E402
from rmlearn import client as CL  # noqa: E402
from rmlearn import export as E  # noqa: E402
from rmlearn import model as M  # noqa: E402
from rmlearn import rmtb  # noqa: E402
from rmlearn import server as SV  # noqa: E402
from rmlearn import spec as S  # noqa: E402

FAILURES = []
TOL = 1e-5


def ok(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print("FAIL: " + msg, flush=True)
    return cond


def info(msg):
    print("  " + msg, flush=True)


def maxdiff(a, b):
    return float((a - b).detach().abs().max()) if a.numel() else 0.0


# ── inputs ───────────────────────────────────────────────────────────────────────────────────

def make_inputs(args):
    """Runs the engine and rmfeat to make the tables and one explore game's tensors in --work."""
    w = args.work
    text = os.path.join(args.bin_dir, "resources", "embeddings", "oracle_text.rmtb")
    eng = os.path.join(args.engine_dir, "robomage")
    rmfeat = os.path.join(args.engine_dir, "rmfeat")
    cards, tables, dump = os.path.join(w, "cards.rmstate"), os.path.join(w, "tables"), os.path.join(w, "game.dump")
    steps = [
        [eng, "--cards-frame", cards],
        [rmfeat, "--export-tables", cards, tables, "--text-table", text],
        [eng, "--agent-a", "explore", "--agent-b", "explore", "--bo3", "--game-seed", "1", "--deck-a",
         "league/ur_delver", "--deck-b", "league/burn", "--max-decisions", "300", "--rmstate-dump", dump],
        [rmfeat, "--featurize", dump, "--viewer", "A", "--out", os.path.join(w, "A.rmtb"), "--cards", cards,
         "--tables", tables, "--text-table", text],
        [rmfeat, "--featurize", dump, "--viewer", "omniscient", "--out", os.path.join(w, "omniscient.rmtb"),
         "--cards", cards, "--tables", tables, "--text-table", text],
        [rmfeat, "--targets", dump, "--seat", "A", "--out", os.path.join(w, "targets_A.rmtb"), "--cards", cards,
         "--text-table", text],
    ]
    os.makedirs(tables, exist_ok=True)
    for argv in steps:
        r = subprocess.run(argv, cwd=args.bin_dir, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("check: %s failed (%d): %s" % (" ".join(argv[:2]), r.returncode, (r.stderr or r.stdout)[-400:]))
    args.tables, args.tensors = tables, os.path.join(w, "A.rmtb")
    args.omniscient, args.targets = os.path.join(w, "omniscient.rmtb"), os.path.join(w, "targets_A.rmtb")


# ── checks ───────────────────────────────────────────────────────────────────────────────────

def check_readers(args, tables):
    print("readers", flush=True)
    import rmtb as stdlib_rmtb
    for path in [os.path.join(args.tables, n) for n in ("card.rmtb", "ability.rmtb", "part.rmtb")] + [args.tensors]:
        meta, arrs = rmtb.read(path)
        smeta, sarrs = stdlib_rmtb.read(path)
        same = meta == smeta and set(arrs) == set(sarrs)
        for name, a in arrs.items():
            s = sarrs[name]
            same = same and list(a.shape) == s.shape and np.array_equal(a.reshape(-1), np.array(s.values, a.dtype))
        ok(same, "%s: the numpy reader differs from python/rmtb.py" % path)
        cols = {a["name"]: a["columns"] for a in meta["arrays"] if "columns" in a}
        data = rmtb.write_bytes({k: v for k, v in meta.items() if k != "arrays"}, arrs, cols)
        m2, a2 = rmtb.read_bytes(data)
        ok(m2 == meta and all(np.array_equal(a2[k], arrs[k]) for k in arrs), "%s: write/read round trip" % path)
        with open(path, "rb") as fp:
            same_bytes = fp.read() == data
        info("%s: %d arrays equal to python/rmtb.py; rewritten %s" % (os.path.basename(path), len(arrs),
                                                                      "byte for byte" if same_bytes else "(same arrays)"))
    ok(tables.hashes["spec_hash"] == tables.spec.hash, "the tables' spec hash")
    decs = B.Decisions.from_file(args.tensors)
    ok(decs.meta.get("spec_hash") == tables.spec.hash, "the tensors file is of another spec")
    info("tables: card %s, text %s, ability %s, part %s; spec %s vocab %s"
         % (tables.card.shape, tables.text.shape, tables.ability.shape, tables.part.shape, tables.hashes["spec_hash"],
            tables.hashes["vocab_hash"]))
    return decs


def outputs(model, decs, idx, min_dims=None):
    sub = decs.select(idx)
    x = B.to_torch(B.pad(sub, model.spec, min_dims))
    with torch.no_grad():
        out = model(*x)
    return out, x


def check_model(args, tables, decs):
    print("model", flush=True)
    torch.manual_seed(3)
    cfg = M.ModelConfig(d=64, heads=4, layers=2, omniscient_critic=True, critic_layers=1)
    model = M.RMNet(cfg, tables)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    model.eval()
    n = min(len(decs), 24)
    idx = list(range(n))
    out, x = outputs(model, decs, idx)
    a = decs.select(idx).sizes["a"]
    ok(out["logits"].shape == x[8].shape and out["value"].shape == (n,), "output shapes")
    ok(all(torch.isfinite(v).all() for v in out.values()), "non-finite outputs")
    pad_rows = ~x[9]
    ok(bool((out["logits"][pad_rows] == -1e9).all()), "padded menu rows are not -1e9")
    # Padding invariance: alone, in the batch, and padded much further.
    worst = 0.0
    big = {"N": int(x[0].shape[1]) + 40, "R": int(x[6].shape[1]) + 24, "A": int(x[8].shape[1]) + 16}
    far, _ = outputs(model, decs, idx, big)
    rev, _ = outputs(model, decs, idx[::-1])
    for i in idx:
        one, _ = outputs(model, decs, [i])
        k = int(a[i])
        for other in (out, far):
            worst = max(worst, maxdiff(one["logits"][0, :k], other["logits"][i, :k]),
                        maxdiff(one["value"][:1], other["value"][i:i + 1]))
        worst = max(worst, maxdiff(one["logits"][0, :k], rev["logits"][n - 1 - i, :k]))
    ok(worst <= TOL, "padding invariance: %.3g" % worst)
    info("forward on %d decisions; padding invariance (alone / in a batch / reversed / padded +40 tokens): max %.3g"
         % (n, worst))
    pol = model.inference_module(2)
    with torch.no_grad():
        lg, v = pol(*x)
    d = max(maxdiff(lg, out["logits"]), maxdiff(v, out["value"]))
    ok(d <= TOL, "the folded inference module differs from the training model: %.3g" % d)
    info("folded inference module vs training model: %.3g; %d parameters (%d without the critic)"
         % (d, model.param_count(), model.param_count() - sum(p.numel() for p in model.critic.parameters())))
    # Training: id-dropout changes the forward, aux losses and a backward pass.
    model.train()
    out_t = model(*x)
    ok(maxdiff(out_t["logits"][x[9]], out["logits"][x[9]]) > 0, "id-dropout has no effect in training")
    loss = (out_t["value"] ** 2).mean() - (M.masked_log_softmax(out_t["logits"], x[9]).exp() *
                                           out_t["logits"]).sum(-1).mean() * 0
    if args.targets:
        tg = B.Targets(args.targets)
        rows = [tg.key_index[(int(g), int(dd))] for g, dd in zip(decs.fields["game"][:n], decs.fields["decision"][:n])]
        t = tg.batch(rows, model.cards.n_cards, tables.spec.library_depth)
        al = M.aux_losses(out_t, t)
        ok(all(torch.isfinite(v) for v in al.values()), "aux losses not finite")
        loss = loss + sum(al.values())
        info("aux losses: %s" % ", ".join("%s %.3f" % (k, float(v.detach())) for k, v in al.items()))
    if args.omniscient:
        omni = B.Decisions.from_file(args.omniscient)
        key = {(int(g), int(dd)): i for i, (g, dd) in enumerate(zip(omni.fields["game"], omni.fields["decision"]))}
        pairs = [key[(int(g), int(dd))] for g, dd in zip(decs.fields["game"][:n], decs.fields["decision"][:n])]
        ox = B.to_torch(B.pad(omni.select(pairs), tables.spec))
        seat = torch.from_numpy(decs.fields["viewer"][:n].copy())
        out_c = model(*x, omni=ox, seat=seat)
        ok(out_c["critic_value"].shape == (n,) and bool(torch.isfinite(out_c["critic_value"]).all()),
           "critic value")
        loss = loss + (out_c["critic_value"] ** 2).mean()
        info("omniscient critic on %d paired decisions (omniscient N %d vs seat N %d)"
             % (n, ox[0].shape[1], x[0].shape[1]))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    ok(len(grads) > 0 and all(torch.isfinite(g).all() for g in grads), "gradients")
    info("backward: %d parameter tensors with finite gradients" % len(grads))
    model.eval()
    check_smoothing(out["logits"], x[9])
    return model, pol, x


def check_smoothing(logits, mask):
    print("smoothed policy", flush=True)
    lp0 = M.smoothed_log_policy(logits, mask, 0.0)
    ok(maxdiff(lp0[mask], M.masked_log_softmax(logits, mask)[mask]) < 1e-6, "eps 0 is log softmax")
    b = logits.shape[0]
    sb = torch.zeros(b, dtype=torch.bool)
    sb[::2] = True
    lp = M.smoothed_log_policy(logits, mask, 0.1, sideboarding=sb, eps_sideboarding=0.3)
    p = lp.exp()
    ok(bool((p.sum(-1) - 1).abs().max() < 1e-5), "smoothed policy sums to 1")
    nl = mask.sum(-1).float()
    floor = torch.where(sb, 0.3, 0.1) / nl
    ok(bool((p + 1e-7 >= floor.unsqueeze(-1))[mask].all()), "smoothed policy floor eps / legal")
    ok(bool((p[~mask] == 0).all()), "smoothed policy is 0 at padded rows")
    info("eps 0.1 (sideboarding 0.3): sums to 1, every legal row ≥ eps / |legal|")


def check_exports(args, model, pol, x):
    print("exports", flush=True)
    prefix = os.path.join(args.work, "check_policy")
    formats = ("ts",) if args.no_pt2 else ("ts", "pt2")
    t = time.time()
    paths = E.export_policy(pol, prefix, 2, formats=formats)
    info("exported %s in %.1f s" % (", ".join(sorted(paths)), time.time() - t))
    with torch.no_grad():
        ref = pol(*x)
    for f, p in paths.items():
        meta = E.read_metadata(p)
        ok(meta == E.metadata(pol) and meta["policy_version"] == "2", "%s metadata %r" % (f, meta))
        with torch.no_grad():
            r = E.load(p)(*x)
        d = max(maxdiff(r[0], ref[0]), maxdiff(r[1], ref[1]))
        ok(d <= 1e-4, "%s export differs from eager: %.3g" % (f, d))
        info("%s: metadata %s; outputs vs eager %.3g" % (f, meta, d))


def check_server(args, tables, model, decs):
    print("server", flush=True)
    sock = os.path.join(tempfile.mkdtemp(prefix="rmlc-", dir="/tmp"), "s.sock")
    srv = SV.InferenceServer(sock, device="cpu", max_batch=64, deadline_us=20000, log=lambda m: None)
    other = M.RMNet(model.cfg, tables)
    srv.set_policy(0, model, 5)
    srv.set_policy(1, other, 9)
    srv.start()
    ref0 = model.inference_module(5)
    ref1 = other.inference_module(9)
    n = min(len(decs), 16)
    try:
        cl = CL.Client(sock, tables.hashes, tables.spec.dims(), policies=[1, 0])
        ok([p["version"] for p in cl.welcome["policies"]] == [9, 5], "WELCOME versions")
        worst = 0.0
        for s in range(0, n, 5):
            idx = list(range(s, min(n, s + 5)))
            sub = decs.select(idx)
            r = cl.evaluate(sub)
            x = B.to_torch(B.pad(sub, tables.spec))
            for j, ref in ((0, ref1), (1, ref0)):
                with torch.no_grad():
                    lg, v = ref(*x)
                real = lg[x[9]]
                lp = M.masked_log_softmax(lg, x[9])[x[9]]
                worst = max(worst, maxdiff(torch.from_numpy(r["logits"][j]), real),
                            maxdiff(torch.from_numpy(r["log_pi"][j]), lp), maxdiff(torch.from_numpy(r["value"][j]), v))
            ok(r["ids"] == [1, 0] and r["versions"] == [9, 5], "response ids / versions %r %r" % (r["ids"],
                                                                                                r["versions"]))
        ok(worst <= TOL, "server results vs the model: %.3g" % worst)
        info("round trip: %d decisions, 2 policies, results within %.3g of the model" % (n, worst))
        srv.set_policy(0, model, 6)
        r = cl.evaluate(decs.select([0]))
        ok(r["versions"] == [9, 6], "after set_policy the version is %r" % r["versions"])
        # Request refusals (each ends the connection), on a decision with relations to tamper with.
        with_rel = next((i for i in range(len(decs)) if decs.sizes["r"][i] > 0), 0)
        bad = [("act_tok", lambda name, a: a + 100000 if name == "act_tok" else a, "request"),
               ("rel", lambda name, a: a + 100000 if name == "rel" else a, "request"),
               ("tok_card", lambda name, a: a + 100000 if name == "tok_card" else a, "request")]
        for what, tamper, code in bad:
            c2 = CL.Client(sock, tables.hashes, tables.spec.dims(), policies=[0])
            try:
                c2.evaluate(decs.select([with_rel]), tamper=tamper)
                ok(False, "a request with a bad %s was answered" % what)
            except CL.RmipError as e:
                ok(e.code == code, "a bad %s refused with %s, expected %s" % (what, e.code, code))
            c2.close()
        c2 = CL.Client(sock, tables.hashes, tables.spec.dims(), policies=[0])
        c2.spec_hash = "0123456789abcdef"
        try:
            c2.evaluate(decs.select([0]))
            ok(False, "a request of another spec hash was answered")
        except CL.RmipError as e:
            ok(e.code == "spec_hash", "a request of another spec hash refused with %s" % e.code)
        c2.close()
        base = {"protocol": 1, "client": "check", "spec_hash": tables.hashes["spec_hash"],
                "vocab_hash": tables.hashes["vocab_hash"], "text_model": tables.hashes["text_model"],
                "dims": tables.spec.dims(), "shm": "/rmip-none", "request_bytes": 64, "response_bytes": 64,
                "policies": [0]}
        cases = [("spec_hash", dict(base, spec_hash="0123456789abcdef")),
                 ("vocab_hash", dict(base, vocab_hash="0123456789abcdef")),
                 ("text_model", dict(base, text_model="x@0")), ("protocol", dict(base, protocol=2)),
                 ("policy", dict(base, policies=[7])), ("shm", base)]
        for code, hello in cases:
            got = CL.hello_only(sock, hello)
            ok(got == ("ERROR", code), "HELLO expecting %s got %r" % (code, got))
        info("refusals: requests (act_tok, rel, tok_card out of range; spec hash) and handshakes (%s)"
             % ", ".join(c for c, _ in cases))
        # Batching across clients: 6 clients evaluate at once; batches hold several requests.
        results = {}

        def worker(k):
            c = CL.Client(sock, tables.hashes, tables.spec.dims(), policies=[0])
            got = []
            for i in range(4):
                got.append(c.evaluate(decs.select([(k + i) % len(decs)]))["batch"])
            c.close()
            results[k] = got

        before = srv.stats()
        threads = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        st = srv.stats()
        batches = st["batches"] - before["batches"]
        biggest = max(max(v) for v in results.values())
        ok(len(results) == 6 and biggest > 1, "no batch held requests of several clients (%r)" % results)
        info("batching: 6 concurrent clients × 4 requests in %d batches, largest %d decisions" % (batches, biggest))
    finally:
        try:
            cl
        except NameError:
            cl = None
        srv.stop()
    if cl is not None:
        try:
            kind, body = CL.recv_message(cl.sock)
            code = json.loads(body)["code"] if kind == SV.ERROR else "message %d" % kind
        except (CL.RmipError, OSError) as e:
            code = getattr(e, "code", "closed")
        ok(code == "shutdown", "after stop the client got %s, not ERROR shutdown" % code)
        info("stop: the connected client gets ERROR %s, the socket is removed" % code)
        cl.close()
    ok(not os.path.exists(sock), "the socket is removed at stop")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", help="rmfeat --export-tables DIR")
    ap.add_argument("--tensors", help="a seat's tensors file")
    ap.add_argument("--omniscient", help="the omniscient tensors of the same dump")
    ap.add_argument("--targets", help="the seat's targets of the same dump")
    ap.add_argument("--engine-dir", help="bin/<config> with robomage and rmfeat: make the inputs")
    ap.add_argument("--bin-dir", default=os.path.join(REPO_ROOT, "bin"), help="the engine's working directory")
    ap.add_argument("--work", help="output directory (default: a temporary one)")
    ap.add_argument("--no-pt2", action="store_true", help="skip the AOTInductor export (slow to compile)")
    args = ap.parse_args()
    args.work = os.path.abspath(args.work or tempfile.mkdtemp(prefix="rmlearn-check-"))
    os.makedirs(args.work, exist_ok=True)
    if args.engine_dir:
        make_inputs(args)
    if not args.tables or not args.tensors:
        ap.error("give --tables and --tensors, or --engine-dir")
    torch.set_num_threads(2)
    tables = S.load_tables(args.tables)
    decs = check_readers(args, tables)
    model, pol, x = check_model(args, tables, decs)
    check_exports(args, model, pol, x)
    check_server(args, tables, model, decs)
    print("RMLEARN-CHECK: %s" % ("ok" if not FAILURES else "FAIL (%d)" % len(FAILURES)), flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
