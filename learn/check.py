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
the smoothed policy; the TorchScript and ``.pt2`` exports' metadata and outputs; the seen-card mask
(an unseen card's folded row is its features plus ``unknown_id``, marking it seen changes that row
alone, training's marking moves no output, the mask through a checkpoint and the exports); model
compatibility (``spec.check_compatible``: a checkpoint loads on tables with a card appended and
carries their manifest, a removed or renamed card, another layout, text model or a moved registry
entry refused; an export of the old tables refused on the new ones with the rebuild message; the same
verdicts as the C++ check, ``rmfeat --compat``, case by case, when the engine is given); the checkpoint
migration (``rmlearn.migrate``) from a synthetic old layout and from a synthetic earlier registry
layout (a layout bump, the seen mask carried), exact; the inference server on the CPU
through the Python client (results against the model, several policies and versions, a swap, batching
across concurrent clients, handshake and request refusals, shutdown); BC's event dropout; the replay
window's splits by game and by match and its probe sample.
Prints ``RMLEARN-CHECK: ok`` and exits 0, or the failures and exits 1.
"""

import os

os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import copy  # noqa: E402
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
from rmlearn import bc as BC  # noqa: E402
from rmlearn import client as CL  # noqa: E402
from rmlearn import episodes as EP  # noqa: E402
from rmlearn import export as E  # noqa: E402
from rmlearn import migrate as MG  # noqa: E402
from rmlearn import model as M  # noqa: E402
from rmlearn import replay as RP  # noqa: E402
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
    model.cards.set_seen(range(model.cards.n_cards - 1))     # a trained network's: every card seen
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
    # With gradients the attention takes the other formulation (the additive bias GEMM): the same
    # function, padding invariant too.
    xg = B.to_torch(B.pad(decs.select(idx), model.spec, big))
    out_g = model(*x)
    far_g = model(*xg)
    worst_g = 0.0
    for i in idx:
        k = int(a[i])
        for other in (out_g, far_g):
            worst_g = max(worst_g, maxdiff(other["logits"][i, :k].detach(), out["logits"][i, :k]),
                          maxdiff(other["value"][i:i + 1].detach(), out["value"][i:i + 1]))
    ok(worst_g <= TOL, "the forward with gradients differs from the one without: %.3g" % worst_g)
    info("forward with gradients (in the batch / padded +40 tokens) vs without: max %.3g" % worst_g)
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


def check_seen(args, tables, model, x):
    """The seen-card mask (docs/open_vocab_plan.md S4): a never-seen card's folded export row is its
    features plus ``unknown_id`` and marking it seen changes that row only; training's marking moves
    no output; the mask round-trips through a checkpoint, and a checkpoint without it is refused unless
    given a seen set; the TorchScript and ``.pt2`` exports of a network with an unseen card agree with
    eager."""
    print("seen cards", flush=True)
    cards_in = sorted(set(int(c) for c in x[3].flatten().tolist() if c >= 0))
    u = cards_in[len(cards_in) // 2]
    every = list(range(model.cards.n_cards - 1))
    m = copy.deepcopy(model).eval()
    m.cards.set_seen([i for i in every if i != u])
    c = m.cards
    ok(not bool(c.card_seen[u]) and int(c.card_seen.sum()) == len(every) - 1 and not bool(c.card_id.weight[u].any()),
       "set_seen: card %d unseen with a zero id row, every other card seen" % u)
    with torch.no_grad():
        feat = c.card_feature_rows().float()
        folded = m.inference_module(0).folded_rows
        n = c.n_cards
        ok(torch.equal(folded[u], feat[u] + c.unknown_id), "the unseen card's folded row is not its features plus "
           "unknown_id")
        seen_rows = feat + c.card_id.weight * c.card_real
        others = [i for i in range(n) if i != u]
        ok(torch.equal(folded[others], seen_rows[others]), "a seen card's folded row is not its features plus its id "
           "row")
        table = c.token_table()
        K = max(model.spec.K, 1)
        ok(all(torch.equal(table[k * n + u], feat[u] + c.unknown_id + c.card_slot[k]) for k in range(K)),
           "the training token table's unseen row")
        drop = c.token_table(True)
        ok(all(torch.equal(drop[k * 2 * n + n + u], feat[u] + c.unknown_id + c.card_slot[k]) for k in range(K)),
           "the id-dropout copy of the unseen row")
        ref_lg, ref_v = m.inference_module(0)(*x)
        # Marking it seen (its id row zero) changes its row and nothing else.
        m2 = copy.deepcopy(m)
        m2.cards.set_seen(every)
        f2 = m2.inference_module(0).folded_rows
        changed = torch.nonzero((f2 != folded).any(1)).squeeze(1).tolist()
        ok(changed == [u], "marking card %d seen changed folded rows %s" % (u, changed[:8]))
        lg2, _ = m2.inference_module(0)(*x)
        ok(maxdiff(lg2, ref_lg) > 0, "marking a card of the batch seen moved no output")
    # The aux heads score every id: the unseen one reads unknown_id there too, but trains neither it nor
    # its own id row.
    if model.cfg.aux_heads:
        ma = copy.deepcopy(m).train()
        lse = ma.aux(torch.randn(3, ma.cfg.d))["aux_hand"].logsumexp(-1).sum()
        lse.backward()
        g_unk, g_id = ma.cards.unknown_id.grad, ma.cards.card_id.weight.grad
        ok((g_unk is None or not bool(g_unk.any())) and not bool(g_id[u].any()) and bool(g_id[others[0]].any()),
           "the aux heads' gradient reaches unknown_id or the unseen card's id row")
    # Training's marking: the batch's cards seen, the newly seen card's id row unknown_id, no output moved.
    m3 = copy.deepcopy(m).train()
    m3.cards.mark_seen(x[3])
    ok(set(m3.cards.seen_ids()) == set(every), "mark_seen left cards of the batch unseen or marked others")
    ok(torch.equal(m3.cards.card_id.weight[u], m3.cards.unknown_id.detach()),
       "a newly seen card's id row is not unknown_id")
    other = [i for i in range(n) if i != u]
    ok(torch.equal(m3.cards.card_id.weight[other], m.cards.card_id.weight[other]), "mark_seen moved another id row")
    with torch.no_grad():
        lg3, v3 = m3.eval().inference_module(0)(*x)
    d = max(maxdiff(lg3, ref_lg), maxdiff(v3, ref_v))
    ok(d == 0.0, "training's first marking moved the outputs: %.3g" % d)
    # The checkpoint round trip, and a checkpoint made before the mask.
    path = os.path.join(args.work, "seen_check.pt")
    M.save_checkpoint(m, path)
    back = M.load_checkpoint(path, tables)
    ok(torch.equal(back.cards.card_seen, m.cards.card_seen) and
       all(torch.equal(a, b) for a, b in zip(back.state_dict().values(), m.state_dict().values())),
       "the seen mask's checkpoint round trip")
    data = torch.load(path, map_location="cpu", weights_only=True)
    del data["state_dict"]["cards.card_seen"]
    old = os.path.join(args.work, "seen_check_old.pt")
    torch.save(data, old)
    try:
        M.load_checkpoint(old, tables)
        ok(False, "a checkpoint without card_seen loaded")
    except RuntimeError as e:
        ok("card_seen is missing" in str(e) and "S4" in str(e), "the refusal of a checkpoint without the mask: %s" % e)
    given = [i for i in every if i != u]
    m4 = M.load_checkpoint(old, tables, seen=given)
    ok(m4.cards.seen_ids() == given and not bool(m4.cards.card_id.weight[u].any()),
       "load_checkpoint(seen=...) on a checkpoint without the mask")
    try:
        M.load_checkpoint(path, tables, seen=given)
        ok(False, "seen= accepted for a checkpoint that has the mask")
    except ValueError:
        pass
    # The exports of the network with an unseen card.
    formats = ("ts",) if args.no_pt2 else ("ts", "pt2")
    paths = E.export_policy(m.inference_module(5), os.path.join(args.work, "seen_policy"), 5, formats=formats)
    worst = {}
    for f, p in paths.items():
        loaded = E.load(p)
        with torch.no_grad():
            r = loaded(*x)
        worst[f] = max(maxdiff(r[0], ref_lg), maxdiff(r[1], ref_v))
        ok(worst[f] <= 1e-4, "%s export with an unseen card differs from eager: %.3g" % (f, worst[f]))
        if f == "ts":
            ok(torch.equal(loaded.folded_rows[u], folded[u]), "the TorchScript export's row of the unseen card")
    info("card %d unseen of %d in the batch: its folded row = features + unknown_id (training table and id-dropout "
         "copy too); marking it seen changes that row alone; training's marking sets its id row to unknown_id and "
         "moves no output; the aux heads train neither unknown_id nor its row; checkpoint round trip; a checkpoint "
         "without the mask refused, loaded with seen=; exports vs eager %s"
         % (u, len(cards_in), ", ".join("%s %.3g" % kv for kv in sorted(worst.items()))))


def manifest_variants(m):
    """Named variants of manifest `m` (what the compatibility checks run on): a card appended (new
    spec and vocab hashes), an id renamed, an id removed, another layout or text model, two ability
    keys' rows swapped in the registry, another tables' digest (a known card's rows changed), a model
    made before manifests."""
    a, b = json.loads(m["registry"])["ab_keys"][:2]
    line = lambda row, key: '{"row":%d,"key":%s}' % (row, json.dumps(key))  # noqa: E731
    swapped = m["registry"].replace(line(a["row"], a["key"]), "\x01") \
        .replace(line(b["row"], b["key"]), line(a["row"], b["key"])).replace("\x01", line(b["row"], a["key"]))
    v = m["vocab"]
    return {
        "same": dict(m),
        "appended": dict(m, vocab=v + [[899, "A Card Appended"]], spec_hash="0123456789abcdef",
                         vocab_hash="00000000000000aa"),
        "renamed": dict(m, vocab=[[i, n + " Renamed" if k == 3 else n] for k, (i, n) in enumerate(v)],
                        vocab_hash="00000000000000bb"),
        "removed": dict(m, vocab=v[:3] + v[4:], vocab_hash="00000000000000cc"),
        "layout": dict(m, layout_hash="0123456789abcdef"),
        "text": dict(m, text_model="other/model@0"),
        "registry_moved": dict(m, registry=swapped),
        "drifted": dict(m, tables_digest="0123456789abcdef"),
        "legacy": {k: m[k] for k in ("spec_hash", "vocab_hash", "text_model")},
    }


def check_compat(args, tables, model):
    """spec.check_compatible on checkpoints and exports across manifest variants, and against the C++
    rule (rmfeat --compat) when the engine is given."""
    print("compatibility", flush=True)
    m = S.manifest_of(tables.hashes)
    var = manifest_variants(m)
    ok(var["registry_moved"]["registry"] != m["registry"], "the registry variant moved nothing")
    want = {  # (model, now, exact) -> the refusal code (None: compatible)
        ("same", "same", False): None, ("same", "same", True): None,
        ("same", "appended", False): None, ("same", "appended", True): "vocab_hash",
        ("same", "renamed", False): "vocab_hash", ("same", "removed", False): "vocab_hash",
        ("same", "layout", False): "layout_hash", ("same", "text", False): "text_model",
        ("same", "registry_moved", False): "layout_hash", ("appended", "same", False): "vocab_hash",
        ("legacy", "same", False): None, ("legacy", "appended", False): "spec_hash",
        ("legacy", "renamed", True): "vocab_hash",
        ("same", "drifted", False): None, ("same", "drifted", True): "tables_digest",
    }
    for (a, b, exact), code in want.items():
        r = S.check_compatible(var[a], var[b], exact)
        ok((r and r[0]) == code, "check_compatible(%s, %s, exact=%s): %r, expected %s" % (a, b, exact, r, code))
        if r and exact:
            ok("rebuild it from its checkpoint" in r[1], "an exact refusal does not say to rebuild: %s" % r[1])
    r = S.check_compatible(var["same"], var["removed"])
    ok(r and "card %d" % m["vocab"][3][0] in r[1], "a removed card is named: %r" % (r,))
    # A checkpoint of these tables loads on tables with a card appended and carries their manifest.
    path = os.path.join(args.work, "compat_check.pt")
    M.save_checkpoint(model, path)
    grown = S.Tables(tables.spec, tables.card, tables.text, tables.ability, tables.part, tables.part_ability,
                     var["appended"])
    m2 = M.load_checkpoint(path, grown)
    ok(m2.hashes["vocab"] == var["appended"]["vocab"], "the loaded model carries the new tables' manifest")
    ok(all(torch.equal(a, b) for a, b in zip(m2.state_dict().values(), model.state_dict().values())),
       "the checkpoint's parameters across the expansion")
    for bad in ("renamed", "removed", "layout", "text", "registry_moved"):
        t = S.Tables(tables.spec, tables.card, tables.text, tables.ability, tables.part, tables.part_ability,
                     var[bad])
        try:
            M.load_checkpoint(path, t)
            ok(False, "a checkpoint loaded on the %s tables" % bad)
        except ValueError as e:
            ok(path in str(e), "the refusal names the checkpoint: %s" % e)
    # An export of these tables runs on them only.
    prefix = os.path.join(args.work, "compat_policy")
    ts = E.export_policy(model, prefix, 4, formats=("ts",))["ts"]
    ok(E.export_refusal(ts, tables.hashes) is None, "the export refused on its own tables: %s"
       % E.export_refusal(ts, tables.hashes))
    why = E.export_refusal(ts, var["appended"])
    ok(why is not None and "A Card Appended" in why and "rebuild it from its checkpoint" in why,
       "the export on the grown vocab: %s" % why)
    meta = E.read_metadata(ts)
    ok(meta["layout_hash"] == m["layout_hash"] and json.loads(meta["vocab"]) == m["vocab"],
       "the export's layout_hash and vocab metadata")
    parity = 0
    if args.engine_dir:
        rmfeat = os.path.join(args.engine_dir, "rmfeat")
        files = {}
        for name, mm in var.items():
            files[name] = os.path.join(args.work, "manifest_%s.json" % name)
            with open(files[name], "w") as fp:
                json.dump(mm, fp)
        for (a, b, exact), code in want.items():
            r = subprocess.run([rmfeat, "--compat", files[a], files[b]] + (["--exact"] if exact else []),
                               capture_output=True, text=True)
            got = None if r.returncode == 0 else (r.stdout.split("COMPAT-REFUSED: ", 1)[-1].split(":", 1)[0]
                                                   if r.returncode == 1 else "error %d: %s" % (r.returncode, r.stderr))
            ok(got == code, "rmfeat --compat %s %s%s: %r, Python %s" % (a, b, " --exact" if exact else "", got, code))
            parity += 1
    info("%d manifest cases (%s); a checkpoint loads across a card appended and is refused on a removed or renamed "
         "card, another layout, text model or a moved registry entry; its export runs on its own tables only (the "
         "rebuild message)%s" % (len(want), ", ".join(sorted(var)),
                                 "; rmfeat --compat agrees on all %d" % parity if parity else ""))


# ── the checkpoint migration ─────────────────────────────────────────────────────────────────

def synthetic_old_layout(doc, rng):
    """A layout before the registry (ids in order, running offsets, no reserved positions) holding the
    same names as the spec document `doc` in other orders: token types, numeric slots, enum values,
    relations and table columns permuted, plus what S1 removed (a ``lists`` type, the knowledge level
    ``EXPERT_BELIEF``); (the spec document, its enums)."""
    enums = {k: list(v)[::-1] for k, v in doc["enums"].items()}
    enums["knowledge_level"] = enums["knowledge_level"] + ["EXPERT_BELIEF"]
    types, cat = [], 1
    for i, ti in enumerate(rng.permutation(len(doc["token_types"]))):
        t = doc["token_types"][int(ti)]
        cats = []
        for c in t["categorical"]:
            size = len(enums[c["source"][5:]]) + 1 if c["source"].startswith("enum:") else c["size"]
            cats.append({"name": c["name"], "source": c["source"], "offset": cat, "size": size})
            cat += size
        types.append({"id": i, "name": t["name"], "numeric": [{"name": n["name"]} for n in t["numeric"]][::-1],
                      "categorical": cats,
                      "cards": [{"name": c["name"]} for c in sorted(t["cards"], key=lambda c: c["col"])],
                      "abilities": [{"name": c["name"]} for c in sorted(t["abilities"], key=lambda c: c["col"])]})
    types.append({"id": len(types), "name": "lists", "numeric": [{"name": "a"}, {"name": "b"}],
                  "categorical": [{"name": "player_rel", "source": "player-rel", "offset": cat, "size": 4}],
                  "cards": [], "abilities": []})
    rels = [doc["relation_types"][int(i)] for i in rng.permutation(len(doc["relation_types"]))]
    tables = {}
    for k in ("card", "ability", "part"):
        cols = [c for c in doc["card_tables"][k]["columns"] if not c.startswith("#")]
        tables[k] = {"columns": [cols[int(i)] for i in rng.permutation(len(cols))]}
    return ({"token_types": types, "relation_types": [dict(r, id=j) for j, r in enumerate(rels)],
             "card_tables": tables, "categorical_size": cat + 4, "K": doc["K"], "Q": doc["Q"]}, enums)


def synthetic_registry_layout(doc, rng):
    """A layout of the registry form (positions, capacities, reserved columns) holding the same names as
    the spec document `doc` elsewhere, as a layout bump moves them: type ids and numeric blocks, each
    type's slot columns, categorical ranges, relation ids and table columns permuted within larger
    capacities, reserved table columns (``#<i>``) between them, and every enum's values in another
    order (the registry's, which a categorical id follows); the spec document."""
    enums = {k: list(v)[::-1] for k, v in doc["enums"].items()}
    n = len(doc["token_types"])
    ids = rng.permutation(doc["type_capacity"] + 8)[:n]
    types, at, cat = [], 0, 1
    for j, ti in enumerate(rng.permutation(n)):
        t = doc["token_types"][int(ti)]
        cap = len(t["numeric"]) + 8
        cols = rng.permutation(cap)[:len(t["numeric"])]
        cats = []
        for c in t["categorical"]:
            size = c["size"] + 3 if c["source"].startswith("enum:") else c["size"]
            cats.append(dict(c, offset=cat, size=size))
            cat += size + 2
        types.append(dict(t, id=int(ids[j]), num_at=at, num_capacity=cap, categorical=cats,
                          numeric=[dict(sl, at=int(cols[i])) for i, sl in enumerate(t["numeric"])]))
        at += cap
    rel_ids = rng.permutation(doc["relation_capacity"] + 8)[:len(doc["relation_types"])]
    tables = {}
    for k in ("card", "ability", "part"):
        cols = [c for c in doc["card_tables"][k]["columns"] if not c.startswith("#")]
        cols = [cols[int(i)] for i in rng.permutation(len(cols))]
        for r in range(0, len(cols), 7):
            cols.insert(r, "#%d" % r)
        tables[k] = {"columns": cols}
    return dict(doc, layout_hash="synthetic", token_types=types, enums=enums, G=at, categorical_size=cat + 4,
                type_capacity=doc["type_capacity"] + 8, relation_capacity=doc["relation_capacity"] + 8,
                relation_types=[dict(r, id=int(rel_ids[j])) for j, r in enumerate(doc["relation_types"])],
                card_tables=tables)


def old_state_of(lm, sd, rng, keep_seen=False):
    """`sd` (a state_dict of the current layout) laid out in `lm`'s old layout: each layout-indexed
    tensor at its old positions (random where the current one has none), the aux heads S1 removed
    added (unless `keep_seen`: a network of a later layout, which keeps its seen mask)."""
    g = torch.Generator().manual_seed(int(rng.integers(1 << 30)))
    out = {}
    for k, v in sd.items():
        if k == "cards.card_seen":
            if keep_seen:
                out[k] = v.clone()
            continue
        p = lm.pairs(k)
        if p is None:
            out[k] = v.clone()
            continue
        kind, pairs, n_old = p
        shape = (n_old,) + tuple(v.shape[1:]) if kind == "rows" else (v.shape[0], n_old)
        o = torch.randn(shape, generator=g)
        ij = torch.tensor(pairs, dtype=torch.int64).reshape(-1, 2)
        if kind == "rows":
            o[ij[:, 0]] = v[ij[:, 1]]
        else:
            o[:, ij[:, 0]] = v[:, ij[:, 1]]
        out[k] = o
    if keep_seen:
        return out
    d = sd["cards.card_proj.weight"].shape[0]
    out["aux_list.weight"], out["aux_list.bias"] = torch.randn(13, d, generator=g), torch.zeros(13)
    out["aux_archetype.weight"], out["aux_archetype.bias"] = torch.randn(8, d, generator=g), torch.zeros(8)
    return out


def zero_unmapped(lm, sd):
    """`sd` with every layout-indexed position the old layout of `lm` lacks set to zero (what a migration
    gives there)."""
    want = {k: v.clone() for k, v in sd.items()}
    for k, v in want.items():
        p = lm.pairs(k)
        if p is not None:
            kind, pairs, _ = p
            keep = torch.zeros(v.shape[0] if kind == "rows" else v.shape[1], dtype=torch.bool)
            keep[torch.tensor([b for _, b in pairs], dtype=torch.int64)] = True
            v.mul_((keep.view(-1, *([1] * (v.dim() - 1))) if kind == "rows" else keep.view(1, -1)).to(v.dtype))
    return want


def check_migration_bump(doc, tables, model, x, rng):
    """A layout bump's migration (registry layout to registry layout): a network of a synthetic earlier
    registry layout, carrying its seen mask, migrates back exactly, mask, ``card_id`` rows and Adam's
    moments untouched; a --seen set that differs from the mask refused."""
    old_doc = synthetic_registry_layout(doc, rng)
    lm = MG.LayoutMap(MG.Layout(old_doc, old_doc["enums"]), MG.Layout(doc, doc["enums"]))
    ok(lm.dropped == [], "migration (bump): dropped %s" % lm.dropped)
    ref = copy.deepcopy(model).cpu().eval()
    seen = sorted(set(range(1, 900, 5)) | {901})
    ref.cards.set_seen(seen)
    with torch.no_grad():
        ref.cards.card_id.weight[2] = 0.25           # an unseen row that is not zero stays as it is
    ref.load_state_dict(zero_unmapped(lm, ref.state_dict()))
    want = ref.state_dict()
    old = old_state_of(lm, want, rng, keep_seen=True)
    got = MG.build_network(lm, ref.cfg.to_dict(), old, tables, None)
    gsd = got.state_dict()
    bad = [k for k in want if not torch.equal(gsd[k], want[k])]
    ok(not bad and set(gsd) == set(want), "migration (bump): the migrated network differs at %s" % bad[:4])
    ok(got.cards.seen_ids() == seen, "migration (bump): the carried seen mask was not kept")
    with torch.no_grad():
        a, b = ref(*x), got(*x)
    ok(maxdiff(a["logits"], b["logits"]) == 0 and maxdiff(a["value"], b["value"]) == 0,
       "migration (bump): the migrated network's outputs differ")
    names = [k for k in old if k != "cards.card_seen"]
    adam = {"t": 3, "m": [old[k] for k in names], "v": [old[k] * 2 for k in names]}
    mig = MG.migrate_adam(lm, adam, old, got)
    params = dict(got.named_parameters())
    bad = [k for (k, _), m, v in zip(got.named_parameters(), mig["m"], mig["v"])
           if not (torch.equal(m, params[k].detach()) and torch.equal(v, params[k].detach() * 2))]
    ok(not bad, "migration (bump): Adam's moments do not follow their parameters at %s" % bad[:4])
    try:
        MG.build_network(lm, ref.cfg.to_dict(), old, tables, seen[1:])
        ok(False, "migration (bump): a --seen set other than the carried mask was not refused")
    except MG.MigrationError as err:
        ok("seen mask" in str(err), "migration (bump): the refusal does not name the mask: %s" % err)
    info("migration: a synthetic earlier registry layout (ids, blocks, slots, ranges and columns permuted, "
         "reserved columns, enums reordered) carrying its seen mask migrates back exactly")


def check_migration(args, tables, model, x):
    """rmlearn.migrate on a synthetic old layout (the module's mapping by name, independent of any run):
    a network laid out in it migrates back to the original exactly, but the positions the old layout
    lacks (zero), with the given seen set (set_seen) and the same outputs; Adam's moments like their
    parameters; an old name the current layout lacks refused, naming it."""
    print("migration", flush=True)
    rng = np.random.default_rng(5)
    with open(os.path.join(args.tables, "spec.json")) as fp:
        doc = json.load(fp)
    old_doc, enums = synthetic_old_layout(doc, rng)
    lm = MG.LayoutMap(MG.Layout(old_doc, enums), MG.Layout(doc, doc["enums"]))
    ok(sorted(lm.dropped) == ["token type lists", "value EXPERT_BELIEF of header.knowledge",
                              "value EXPERT_BELIEF of k_meta.level"], "migration: dropped %s" % lm.dropped)
    ref = copy.deepcopy(model).cpu().eval()
    seen = sorted(set(range(0, 900, 3)) | {900, 1023})
    ref.cards.set_seen(seen)
    want = zero_unmapped(lm, ref.state_dict())       # the positions the old layout lacks read zero
    ref.load_state_dict(want)
    old = old_state_of(lm, ref.state_dict(), rng)
    got = MG.build_network(lm, ref.cfg.to_dict(), old, tables, seen)
    gsd = got.state_dict()
    bad = [k for k in want if not torch.equal(gsd[k], want[k])]
    ok(not bad and set(gsd) == set(want), "migration: the migrated network differs from the original at %s" % bad[:4])
    ok(got.cards.seen_ids() == seen, "migration: the seen set is not the one given")
    with torch.no_grad():
        a, b = ref(*x), got(*x)
    ok(maxdiff(a["logits"], b["logits"]) == 0 and maxdiff(a["value"], b["value"]) == 0,
       "migration: the migrated network's outputs differ")
    names = list(old)
    adam = {"t": 7, "m": [old[k] for k in names], "v": [old[k] * 2 for k in names]}
    mig = MG.migrate_adam(lm, adam, old, got)
    params = dict(got.named_parameters())
    bad = [k for (k, _), m, v in zip(got.named_parameters(), mig["m"], mig["v"])
           if not (torch.equal(m, params[k].detach()) and torch.equal(v, params[k].detach() * 2))]
    ok(not bad and mig["t"] == 7, "migration: Adam's moments do not follow their parameters at %s" % bad[:4])
    broken = copy.deepcopy(old_doc)
    broken["token_types"][0]["numeric"].append({"name": "no_such_slot"})
    try:
        MG.LayoutMap(MG.Layout(broken, enums), MG.Layout(doc, doc["enums"]))
        ok(False, "migration: an old numeric slot the current layout lacks was not refused")
    except MG.MigrationError as err:
        ok("no_such_slot" in str(err), "migration: the refusal does not name the slot: %s" % err)
    info("migration: a synthetic old layout (%d types, every order permuted, S1's removals) migrates back exactly, "
         "outputs equal, Adam's moments like their parameters, an unknown name refused" % len(old_doc["token_types"]))
    check_migration_bump(doc, tables, model, x, rng)


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
        base = {"protocol": SV.PROTOCOL, "client": "check", "spec_hash": tables.hashes["spec_hash"],
                "vocab_hash": tables.hashes["vocab_hash"], "text_model": tables.hashes["text_model"],
                "layout_hash": tables.hashes["layout_hash"], "tables_digest": tables.hashes["tables_digest"],
                "vocab": tables.hashes["vocab"],
                "dims": tables.spec.dims(), "shm": "/rmip-none", "request_bytes": 64, "response_bytes": 64,
                "policies": [0]}
        cases = [("spec_hash", dict(base, spec_hash="0123456789abcdef")),
                 ("vocab_hash", dict(base, vocab_hash="0123456789abcdef")),
                 ("text_model", dict(base, text_model="x@0")), ("protocol", dict(base, protocol=SV.PROTOCOL + 1)),
                 ("layout_hash", dict(base, layout_hash="0123456789abcdef")),
                 ("tables_digest", dict(base, tables_digest="0123456789abcdef")),
                 ("vocab_hash", dict(base, vocab=base["vocab"] + [[899, "A Card Appended"]])),
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

# ── episodes, the replay window, batches, behaviour cloning ──────────────────────────────────

def make_episodes(args):
    """Two short bo3 matches with the debug engine (explore vs random, the scripted teacher on seat
    A, omniscient episodes) into --work/episodes; the directory."""
    out = os.path.join(args.work, "episodes")
    if os.path.isdir(out) and any(n.endswith(".rmtraj") for n in os.listdir(out)):
        return out
    eng = os.path.join(args.engine_dir, "robomage")
    for seed, da, db, cap in ((5, "league/ur_delver", "league/burn", "600"), (6, "league/bug", "league/lands", "400")):
        argv = [eng, "--agent-a", "explore", "--agent-b", "random", "--bo3", "--game-seed", str(seed), "--deck-a", da,
                "--deck-b", db, "--teacher-a", "python3 ../python/teacher.py", "--trajectory-out", out,
                "--trajectory-omniscient", "--max-decisions", cap]
        r = subprocess.run(argv, cwd=args.bin_dir, capture_output=True, text=True, env=dict(os.environ))
        if r.returncode != 0:
            sys.exit("check: the episode game failed (%d): %s" % (r.returncode, (r.stderr or r.stdout)[-400:]))
    return out


def episode_files(d):
    return sorted(os.path.join(d, n) for n in os.listdir(d) if n.endswith(".rmtraj"))


def rmtraj_decisions(steps, viewer, dims):
    """A batch.Decisions of python/rmtraj.py steps' tensors (the reference for expand == pad)."""
    F, C, K, Q = dims
    ts = [s.tensors if viewer == "seat" else s.omniscient for s in steps]
    cat = lambda k, w, dt: np.concatenate([np.asarray(t[k], dt).reshape(-1, w) if w else  # noqa: E731
                                           np.asarray(t[k], dt) for t in ts])
    arrays = {"tok_type": cat("tok_type", 0, np.int64), "tok_num": cat("tok_num", F, np.float32),
              "tok_cat": cat("tok_cat", C, np.int64), "tok_card": cat("tok_card", K, np.int64),
              "tok_ability": cat("tok_ability", Q, np.int64), "tok_mask": cat("tok_mask", 0, np.uint8),
              "rel": cat("rel", 3, np.int64), "act_tok": cat("act_tok", 0, np.int64),
              "act_mask": cat("act_mask", 0, np.uint8)}
    sizes = {"n": np.array([len(t["tok_type"]) for t in ts], np.int64),
             "r": np.array([len(t["rel"]) // 3 for t in ts], np.int64),
             "a": np.array([len(t["act_tok"]) for t in ts], np.int64)}
    return B.Decisions(arrays, sizes)


def check_episode_reader(files):
    print("episode reader", flush=True)
    import rmtraj
    steps = 0
    bad = []
    for f in files:
        ep = EP.read_episode(f)
        ref = rmtraj.read(f)
        ok(ep.attrs == ref.attrs and ep.steps == len(ref.steps), "%s: attrs / steps" % f)
        for d, s in enumerate(ref.steps):
            same = (int(ep.seat[d]) == s.seat and int(ep.game[d]) == s.game and int(ep.decision[d]) == s.decision and
                    bool(ep.sideboarding[d]) == s.sideboarding and int(ep.chosen[d]) == s.chosen and
                    int(ep.teacher[d]) == s.teacher and int(ep.policy_version[d]) == s.policy_version and
                    bool(ep.has_mu[d]) == (s.mu is not None))
            if s.mu is not None:
                same = same and np.array_equal(ep.mu[ep.menu_off[d]:ep.menu_off[d + 1]], np.array(s.mu, np.float32))
            if s.targets is not None:
                t = EP.targets([(ep, d)], 1025, 10)
                same = same and float(t["opp_hand"].sum()) == len(s.targets["opp_hand"]) and \
                    list(t["own_library"][0][:len(s.targets["own_library"])]) == list(s.targets["own_library"][:10])
            if not same:
                bad.append((f, d))
        steps += len(ref.steps)
        for viewer in ("seat", "omni") if ep.omniscient else ("seat",):
            items = [(ep, d) for d in range(ep.steps)]
            x = EP.expand(items, viewer)
            x.pop("sizes")
            ref_x = B.pad(rmtraj_decisions(ref.steps, viewer, ep.dims), dict(zip("FCKQ", ep.dims)))
            for k in EP.INPUTS:
                if x[k].shape != ref_x[k].shape or x[k].dtype != ref_x[k].dtype or not np.array_equal(x[k], ref_x[k]):
                    bad.append((f, viewer + ":" + k))
    ok(not bad, "the numpy episode reader differs from python/rmtraj.py: %r" % bad[:5])
    info("%d episodes, %d steps: per-step fields, μ, targets and both featurizations' padded batches equal "
         "python/rmtraj.py expanded and batch.pad-ed" % (len(files), steps))


def _place(src, dst_dir, mtime):
    """Copies an episode into dst_dir as the engine does: a temporary name, then a rename."""
    name = os.path.basename(src)
    tmp = os.path.join(dst_dir, name + ".tmp")
    with open(src, "rb") as a, open(tmp, "wb") as b:
        b.write(a.read())
    os.utime(tmp, (mtime, mtime))
    os.replace(tmp, os.path.join(dst_dir, name))


def check_window(args, files):
    print("replay window", flush=True)
    eps = [EP.read_episode(f) for f in files]
    wdir = os.path.join(args.work, "window")
    if os.path.isdir(wdir):
        for n in os.listdir(wdir):
            os.unlink(os.path.join(wdir, n))
    os.makedirs(wdir, exist_ok=True)
    rmlog = os.path.join(wdir, "keep.rmlog")
    with open(rmlog, "w") as fp:
        fp.write("RMLOG\n")
    with open(os.path.join(wdir, "partial.rmtraj.tmp"), "wb") as fp:
        fp.write(b"RMTB")
    cap = eps[0].steps + eps[1].steps
    win = RP.ReplayWindow(RP.WindowConfig(dirs=[wdir], max_decisions=cap, delete_evicted=True))
    ok(win.poll() == 0 and len(win) == 0, "an empty directory (a .tmp, an .rmlog) ingested something")
    order = []
    base = time.time() - 1000
    for i, f in enumerate(files):
        _place(f, wdir, base + i)
        got = win.poll()
        order.append(os.path.basename(f))
        kept = [e.name for e in win.episodes()]
        ok(got == 1, "a new episode was not ingested (%d)" % got)
        ok(kept == order[len(order) - len(kept):], "the window is not the newest episodes in order: %r" % kept)
        ok(win.decisions <= cap or len(kept) == 1, "the window holds %d decisions, the cap is %d" % (win.decisions, cap))
        ok(win.decisions == sum(e.steps for e in win.episodes()), "the window's decision count")
    present = sorted(n for n in os.listdir(wdir) if n.endswith(".rmtraj"))
    ok(present == sorted(e.name for e in win.episodes()), "evicted files not deleted / kept files deleted: %r" % present)
    ok(os.path.exists(rmlog) and os.path.exists(os.path.join(wdir, "partial.rmtraj.tmp")),
       "the window deleted an .rmlog or a temporary file")
    st = win.stats()
    ok(st["evicted"] == len(files) - len(win) and st["deleted"] == st["evicted"], "eviction counts %r" % st)
    info("%d episodes ingested one by one under a %d-decision cap: %d evicted oldest first and deleted, the .rmlog "
         "and a .tmp untouched" % (len(files), cap, st["evicted"]))
    # Bytes cap, no deletion.
    w2 = RP.ReplayWindow(RP.WindowConfig(dirs=[os.path.dirname(files[0])], max_bytes=max(e.nbytes for e in eps) + 1))
    w2.poll()
    ok(len(w2) >= 1 and (w2.bytes <= max(e.nbytes for e in eps) + 1), "the bytes cap")
    ok(all(os.path.exists(f) for f in files), "a window without delete_evicted deleted a file")
    # Another spec is skipped; the poller thread ingests files appearing while it runs.
    tdir = os.path.join(args.work, "window_thread")
    if os.path.isdir(tdir):
        for n in os.listdir(tdir):
            os.unlink(os.path.join(tdir, n))
    os.makedirs(tdir, exist_ok=True)
    w3 = RP.ReplayWindow(RP.WindowConfig(dirs=[tdir], poll_interval=0.05, spec_hash=eps[0].spec_hash))
    w3.start()
    try:
        meta, arrs = rmtb.read(files[0])
        bad_meta = {k: v for k, v in meta.items() if k != "arrays"}
        bad_meta["spec_hash"] = "0123456789abcdef"
        cols = {a["name"]: a["columns"] for a in meta["arrays"] if "columns" in a}
        with open(os.path.join(tdir, "other_spec.rmtraj"), "wb") as fp:
            fp.write(rmtb.write_bytes(bad_meta, arrs, cols))
        for i, f in enumerate(files[:3]):
            _place(f, tdir, time.time())
        end = time.time() + 10
        while time.time() < end and (w3.counts["ingested"] < 3 or w3.counts["skipped_spec"] < 1):
            time.sleep(0.05)
    finally:
        w3.stop()
    ok(w3.counts["ingested"] == 3 and w3.counts["skipped_spec"] == 1,
       "the polling thread: %r" % dict(w3.counts))
    info("polling thread: 3 episodes renamed in while it ran ingested; one of another spec skipped")


def check_decision_batches(files):
    print("decision batches", flush=True)
    win = RP.ReplayWindow(RP.WindowConfig(dirs=[os.path.dirname(files[0])]))
    win.poll()
    eps = win.episodes()
    labelled = 0
    for ep in eps:
        a = ep.seat == 0
        want = np.where(a & (ep.teacher >= 0), ep.teacher, -1)
        ok(np.array_equal(ep.labels, want), "%s: labels are not seat A's teacher labels" % ep.name)
        labelled += int((ep.labels >= 0).sum())
        alt = RP.step_labels(ep, RP.LabelConfig(teacher=False, exclude_agents=()))
        ok(np.array_equal(alt, np.where(a & (ep.chosen >= 0), ep.chosen, -1)),
           "%s: chosen labels without μ (explore's seat, the random seat has μ)" % ep.name)
        none = RP.step_labels(ep, RP.LabelConfig(teacher=False))
        ok(bool((none < 0).all()), "%s: an excluded agent's replies became labels" % ep.name)
    ok(labelled > 0, "no labelled decisions")
    rng = np.random.default_rng(1)
    items = win.sample_decisions(48, rng)
    db = RP.decision_batch(items, 1025, 10, omniscient=True)
    a = db.sizes["a"]
    ok(bool((db.label >= 0).all() and (db.label < a).all()), "sampled labels outside their menus")
    ok(np.array_equal(db.outcome, [ep.result[ep.seat[d]] for ep, d in items]), "outcomes")
    ok(db.targets is not None and bool(db.targets["present"].all()), "targets")
    ok(db.omni["tok_type"].shape[0] == 48 and db.x["tok_num"].shape[2] == eps[0].dims[0], "batch shapes")
    tb = db.to_torch("cpu")
    ok(len(tb.inputs) == 10 and tb.inputs[0].dtype == torch.int64 and tb.inputs[5].dtype == torch.bool and
       tb.inputs[1].dtype == torch.float32, "torch dtypes")
    # Held-out split by game.
    wh = RP.ReplayWindow(RP.WindowConfig(dirs=[os.path.dirname(files[0])], heldout_fraction=0.5, split_seed=3))
    wh.poll()
    splits = {e.name: e.split for e in wh.episodes()}
    ok(all((e.split == "heldout") == RP.heldout(e, 0.5, 3) for e in wh.episodes()), "the split is the game hash")
    if "train" in splits.values():
        got = {ep.split for ep, _ in wh.sample_decisions(200, rng)}
        ok(got == {"train"}, "train sampling drew held-out games")
    ev = [it for batch in wh.iter_decisions("heldout", True, 32) for it in batch]
    ok(all(ep.split == "heldout" and ep.labels[d] >= 0 for ep, d in ev), "held-out iteration")
    # Held-out split by match, and the probe sample of the training games.
    wm = RP.ReplayWindow(RP.WindowConfig(dirs=[os.path.dirname(files[0])], heldout_fraction=0.5, split_seed=3,
                                         heldout_by="match", probe_fraction=0.5))
    wm.poll()
    sides = {}
    for e in wm.episodes():
        sides.setdefault(e.match_key(), set()).add(e.split)
        ok((e.split == "heldout") == RP.heldout(e, 0.5, 3, "match"), "the split is the match hash")
    ok(all(len(s) == 1 for s in sides.values()), "a match's games on both sides of the split")
    ok(not any(e.probe and e.split != "train" for e in wm.episodes()), "a held-out game is a probe game")
    pv = [ep for batch in wm.iter_decisions("probe", True, 32) for ep, _ in batch]
    ok(all(ep.probe for ep in pv) and len({ep.name for ep in pv}) == sum(
        1 for e in wm.episodes() if e.probe and (e.labels >= 0).any()), "probe iteration")
    info("labels (teacher on A; chosen only without μ; excluded agents), %d labelled of %d; a batch of 48 with "
         "outcomes, targets and the omniscient inputs; split %r" % (labelled, win.decisions,
                                                                   sorted(collections_counter(splits.values()))))
    return win


def collections_counter(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out.items()


def with_evals(ep, policies=(1, 2)):
    """A copy of episode `ep` with synthesized evaluation arrays (P = 3, the last column a pad)."""
    meta, arrs = rmtb.read_bytes(open(ep.path, "rb").read())
    D, A = ep.steps, len(ep.mu)
    P = len(policies) + 1
    pol = np.full((D, P), -1, np.int64)
    pol[:, :len(policies)] = policies
    pol[::3, 0] = -1                                     # some steps lack the first policy
    ver = np.where(pol >= 0, 40 + pol, -1).astype(np.int64)
    val = np.random.default_rng(0).uniform(-1, 1, (D, P)).astype(np.float32)
    lp = np.random.default_rng(1).normal(size=(A, P)).astype(np.float32)
    new = dict(arrs)
    new.update({"eval_policy": pol, "eval_version": ver, "eval_value": val, "eval_log_pi": lp})
    cols = {a["name"]: a["columns"] for a in meta["arrays"] if "columns" in a}
    data = rmtb.write_bytes({k: v for k, v in meta.items() if k != "arrays"}, new, cols)
    out = EP.Episode(data, ep.path)
    return out, pol, ver, val, lp


def check_trajectory_batches(win):
    print("trajectory batches", flush=True)
    eps = win.episodes()
    ev_ep, pol, ver, val, lp = with_evals(eps[0])
    games = [ev_ep] + eps[1:]
    chunk = 16
    tb = RP.TrajectoryBatch(games, chunk_len=chunk, policies=(1, 2))
    L = np.array([e.steps for e in games])
    ok(tb.T % chunk == 0 and tb.T >= L.max() and tb.T - L.max() < chunk, "T is the longest game rounded up")
    ok(np.array_equal(tb.valid.sum(0), L), "valid steps per game")
    bad = 0
    for g, ep in enumerate(games):
        n = ep.steps
        bad += not np.array_equal(tb.seat[:n, g], ep.seat)
        bad += not np.array_equal(tb.chosen[:n, g], np.where(ep.chosen >= 0, ep.chosen, -1))
        bad += not np.array_equal(tb.teacher[:n, g], ep.teacher)
        bad += not np.array_equal(tb.has_mu[:n, g], ep.has_mu.astype(bool))
        bad += not np.array_equal(tb.legal[:n, g].sum(-1), np.diff(ep.menu_off))
        mu_sum = tb.mu[:n, g].sum(-1)
        bad += not np.allclose(mu_sum[ep.has_mu.astype(bool)], 1.0, atol=1e-5)
        bad += not (tb.mu[:n, g][~tb.legal[:n, g]] == 0).all()
        rw = np.zeros((tb.T, 2), np.float32)
        rw[n - 1] = ep.result
        bad += not np.array_equal(tb.reward[:, g], rw)
        bad += bool(tb.valid[n:, g].any() or tb.legal[n:, g].any())
    ok(bad == 0, "trajectory grids differ from their episodes (%d)" % bad)
    # The recorded evaluations of policies 1 and 2.
    D = ev_ep.steps
    for k, p in enumerate((1, 2)):
        e = tb.evals[p]
        has = pol[:, k] == p
        same = np.array_equal(e["present"][:D, 0], has) and np.array_equal(e["version"][:D, 0][has], ver[has, k]) \
            and np.array_equal(e["value"][:D, 0][has], val[has, k])
        for d in np.nonzero(has)[0][:50]:
            m0, m1 = ev_ep.menu_off[d], ev_ep.menu_off[d + 1]
            same = same and np.array_equal(e["log_pi"][d, 0, :m1 - m0], lp[m0:m1, k])
        ok(same and not e["present"][:, 1:].any(), "policy %d's recorded evaluations" % p)
    # Chunks: every valid step once, in time-major order, their inputs those of expand.
    flat = []
    for c in range(tb.chunks):
        x, idx = tb.chunk_inputs(c)
        if x is None:
            continue
        flat.extend(idx.tolist())
        items, _ = tb.chunk_items(c)
        ref = EP.expand(items, "seat", {"A": tb.A})
        ok(all(np.array_equal(x[k], ref[k]) for k in EP.INPUTS) and x["act_tok"].shape[1] == tb.A,
           "chunk %d inputs" % c)
        t, g = np.divmod(idx, tb.G)
        ok(bool(tb.valid[t, g].all()) and bool((t >= c * chunk).all() and (t < (c + 1) * chunk).all()),
           "chunk %d indices" % c)
    ok(flat == sorted(flat) and len(flat) == int(L.sum()) and len(set(flat)) == len(flat), "chunks cover every step")
    info("%d games, T %d (chunks of %d), %d valid steps: seats, replies, μ, legal rows, terminal rewards, recorded "
         "evaluations (synthesized) and chunk inputs" % (tb.G, tb.T, chunk, int(L.sum())))
    # The prefetcher.
    pf = RP.Prefetcher(lambda rng: int(rng.integers(1000)), depth=2, workers=2, seed=4)
    got = [pf.get() for _ in range(6)]
    pf.stop()
    ok(len(got) == 6, "prefetcher")

    def boom(rng):
        raise RuntimeError("boom")

    pf = RP.Prefetcher(boom, depth=1, workers=1)
    try:
        pf.get()
        ok(False, "the prefetcher swallowed an exception")
    except RuntimeError:
        pass
    pf.stop()


def check_event_dropout():
    """bc.event_dropout on synthetic inputs: event tokens 1, 2 (type 1), card tokens 3, 4 of event 1
    and 5 of event 2 (relation 5), token 0 related to 3 otherwise, token 6 padding."""
    print("event dropout", flush=True)

    class Hist:
        hist_event_type, hist_card_rel = 1, 5

    class NoHist:
        hist_event_type = hist_card_rel = None

    tok_type = torch.tensor([[0, 1, 1, 2, 2, 2, 0]] * 3)
    tok_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]] * 3, dtype=torch.bool)
    rel = torch.tensor([[[3, 1, 5], [4, 1, 5], [5, 2, 5], [0, 3, 7], [0, 0, 0]]] * 3)
    rel_mask = torch.tensor([[1, 1, 1, 1, 0]] * 3, dtype=torch.bool)
    inputs = [tok_type, None, None, None, None, tok_mask, rel, rel_mask, None, None]
    ok(BC.event_dropout(inputs, 0.0, Hist) is inputs and BC.event_dropout(inputs, 0.5, NoHist) is inputs,
       "event dropout at p 0 or without history changed the inputs")
    ok(torch.equal(BC.event_dropout(inputs, 1.0, Hist)[5],
                   torch.tensor([[1, 0, 0, 0, 0, 0, 0]] * 3, dtype=torch.bool)),
       "event dropout at p 1 must leave only the non-history tokens")
    torch.manual_seed(0)
    kept = 0
    for _ in range(40):
        m = BC.event_dropout(inputs, 0.5, Hist)[5]
        ok(bool(m[:, 0].all()) and not bool(m[:, 6].any()) and torch.equal(m[:, 3], m[:, 1]) and
           torch.equal(m[:, 4], m[:, 1]) and torch.equal(m[:, 5], m[:, 2]),
           "event dropout: a card token kept or dropped apart from its event, or another token touched")
        kept += int(m[:, 1:3].sum())
    ok(0.3 < kept / 240.0 < 0.7, "event dropout at p 0.5 kept %d of 240 event tokens" % kept)
    info("p 0 and no history: unchanged; p 1: only the non-history tokens; p 0.5: cards follow their events, "
         "%d of 240 events kept" % kept)


def check_bc(args, tables, win):
    print("behaviour cloning", flush=True)
    torch.manual_seed(0)
    one = RP.ReplayWindow(RP.WindowConfig(dirs=[]))
    ep = max(win.episodes(), key=lambda e: int((e.labels >= 0).sum()))
    one.add(EP.read_episode(ep.path))
    model = M.RMNet(M.ModelConfig(d=32, heads=2, layers=1, id_dropout=0.0), tables)
    cfg = BC.BCConfig(batch_size=24, lr=2e-3, warmup=1, amp="off", loader_workers=1, prefetch=2, eval_limit=200)
    tr = BC.BCTrainer(model, one, cfg, "cpu")
    try:
        losses = []
        for _ in range(30):
            m = tr.train(1)
            losses.append(m["train_policy"])
            ok(all(np.isfinite(v) for k, v in m.items() if k.startswith("train_")), "a non-finite BC loss %r" % m)
        first, last = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
        ok(last < first, "the BC policy loss did not decrease: %.3f -> %.3f" % (first, last))
        seen = set(model.cards.seen_ids())
        held = set(EP.read_episode(ep.path).seat_feat.card_ids().tolist())
        ok(seen and seen <= held, "BC marked cards %s seen; the episode holds %s" % (sorted(seen - held), len(held)))
        e = tr.evaluate("train", 200)
        ok(e.get("eval_decisions", 0) > 0 and 0 <= e["eval_agree_all"] <= 1 and np.isfinite(e["eval_value_mse"]),
           "BC evaluation %r" % e)
        path = os.path.join(args.work, "bc_check.pt")
        tr.save(path, version=3)
        m2 = M.load_checkpoint(path, tables)
        ok(all(torch.equal(a, b) for a, b in zip(m2.state_dict().values(), model.state_dict().values())),
           "BC checkpoint round trip")
        tr.restore(path)
        ok(tr.step == 30, "BC optimizer state restored")
    finally:
        tr.close()
    info("30 steps on one episode (%d labelled decisions): policy loss %.3f -> %.3f, finite; %d of the episode's %d "
         "cards marked seen; evaluation, checkpoint and optimizer state round trip"
         % (int((ep.labels >= 0).sum()), first, last, len(seen), len(held)))


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
    ap.add_argument("--episodes", help="a trajectory directory for the episode checks (made with --engine-dir)")
    args = ap.parse_args()
    args.work = os.path.abspath(args.work or tempfile.mkdtemp(prefix="rmlearn-check-"))
    os.makedirs(args.work, exist_ok=True)
    if args.engine_dir:
        args.engine_dir, args.bin_dir = os.path.abspath(args.engine_dir), os.path.abspath(args.bin_dir)
        make_inputs(args)
    if not args.tables or not args.tensors:
        ap.error("give --tables and --tensors, or --engine-dir")
    torch.set_num_threads(2)
    tables = S.load_tables(args.tables)
    decs = check_readers(args, tables)
    model, pol, x = check_model(args, tables, decs)
    check_event_dropout()
    check_exports(args, model, pol, x)
    check_seen(args, tables, model, x)
    check_compat(args, tables, model)
    check_migration(args, tables, model, x)
    check_server(args, tables, model, decs)
    if args.engine_dir and not args.episodes:
        args.episodes = make_episodes(args)
    if args.episodes:
        files = episode_files(args.episodes)
        ok(len(files) >= 4, "fewer than 4 episodes in %s" % args.episodes)
        check_episode_reader(files)
        check_window(args, files)
        win = check_decision_batches(files)
        check_trajectory_batches(win)
        check_bc(args, tables, win)
    else:
        info("episode checks skipped (no --episodes or --engine-dir)")
    print("RMLEARN-CHECK: %s" % ("ok" if not FAILURES else "FAIL (%d)" % len(FAILURES)), flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
