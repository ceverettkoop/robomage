#!/usr/bin/env python3
"""The open-vocabulary refactor's baseline fixtures (docs/open_vocab_plan.md S0; read by S5's
acceptance test): every runnable checkpoint's outputs on a fixed decision set, under today's tables.

    learn/.venv/bin/python learn/open_vocab_baseline.py generate [--out tests/open_vocab] [--work runs/open_vocab_s0]
    learn/.venv/bin/python learn/open_vocab_baseline.py select   # once: copy the decision sources in

**Decision sources** (committed under ``OUT/sources/``, so the fixture does not depend on a later
corpus re-recording or on a run directory): the replay corpus's decision logs
(``tests/corpus/*.rmlog``, copied to ``sources/corpus/``) and the decision logs of five complete
self-play bo3 matches from the end of ``runs/league24h-sb`` (``sources/selfplay/``; ``select``
copies them and checks that their replay featurizes to the run's own episodes, step for step).
Each log is replayed with ``--machine --rmstate-dump`` and featurized by ``rmfeat --featurize`` for
seats A and B and the omniscient viewer, over the tables exported from the engine's
``--cards-frame`` (all under ``WORK``). The decision set is every seat decision of the corpus logs
and 200 of the self-play logs' seat decisions (evenly spaced over them in order); a decision is
(source, viewer, game, sideboarding, decision), its omniscient frame the omniscient viewer's frame of
the same (game, sideboarding, decision).

**Networks.** Every checkpoint in a ``ckpt/`` directory under ``runs/`` (``runs/**/ckpt/**/*.pt``)
whose hashes are the current tables' is run; the others are listed in the manifest as not runnable,
with the reason. A ``save_checkpoint`` file is one network (``model``); an R-NaD learner state
(``{"learner", "window"}``) is ``model``, ``target`` and each distinct regularisation policy
(``reg0``, ``reg1``). Networks are identified by a hash of their config and parameters, so one
stored many times (a fixed point that is also a later learner's regularisation policy, a seed that
is the BC policy) is evaluated and stored once. No checkpoint of a run has an omniscient critic, so
``generate`` first builds one (``runs/open_vocab_s0/ckpt/m11-critic.pt``, ``build_critic_fixture``;
listed with the runs' checkpoints): league24h-sb's ``m11`` loaded through
``rnad_learner._load_with_config`` with ``omniscient_critic`` on, its critic initialised under a
fixed seed and the critic layers' (zero-initialised) relation biases drawn from a second seeded
generator, so the critic's every embedding and bias is non-trivial.

**Outputs** (``OUT/baseline/``): ``decisions.rmtb`` (per decision its source, viewer, game, sideboarding,
decision and its menu rows' offset) and per network ``<id>.rmtb``: ``logits`` (the decisions' real
menu rows, concatenated) and ``value``; ``logits_nolists`` / ``value_nolists``, the same with the
``lists`` and ``b_lists`` tokens removed from every decision (the inputs S1 deletes; S5 compares a
migrated checkpoint with these); with a critic ``critic_value`` / ``critic_value_nolists`` on the
decisions' omniscient frames, the deciding seat as ``seat``. All fp32 on the CPU, eval mode, without
gradients (the export's path), in fixed batches (decisions sorted by token count, at most
``BATCH_ROWS`` a forward) on ``--threads`` intra-op threads. ``OUT/manifest.json`` records the
tables' hashes, the torch / numpy versions, the threads, every source's sha256 and decision counts,
every checkpoint (path, sha256, kind, runnable or why not, its networks) and every network (its
file's sha256). Nothing in it depends on the time or the commit, so a regeneration reproduces every
file bit for bit (``generate --out`` elsewhere, then ``diff -r``).
"""

import os

os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import concurrent.futures as CF  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as MP  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from rmlearn import batch as B  # noqa: E402
from rmlearn import episodes as EP  # noqa: E402
from rmlearn import model as M  # noqa: E402
from rmlearn import rmtb  # noqa: E402
from rmlearn import rnad_learner as RL  # noqa: E402
from rmlearn import spec as S  # noqa: E402

FORMAT = 1
BATCH_ROWS = 32
SELFPLAY_DECISIONS = 200
REMOVED_TYPES = ("lists", "b_lists")
VIEWERS = ("A", "B", "omniscient")
CRITIC_SOURCE = "runs/league24h-sb/ckpt/fixed/m11.pt"
CRITIC_FIXTURE = "runs/open_vocab_s0/ckpt/m11-critic.pt"  # a checkpoint like the runs', not an intermediate
CRITIC_CONFIG = {"omniscient_critic": True, "critic_layers": 2}
CRITIC_SEED = 20261024
CRITIC_REL_BIAS_STD = 0.1
# The self-play matches `select` copies: complete bo3 matches near the end of league24h-sb (nn vs nn,
# both seats recorded), seven of the ten league decks between them.
SELFPLAY_RUN = "runs/league24h-sb"
SELFPLAY_MATCHES = ["a15-m18000", "a9-m18276", "a3-m18391", "a2-m17912", "a14-m18217"]


# ── helpers ───────────────────────────────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    return os.path.relpath(os.path.abspath(path), REPO_ROOT)


def run(argv, cwd):
    p = subprocess.run(argv, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        sys.exit("open_vocab_baseline: %s exited %d: %s" % (" ".join(argv), p.returncode, p.stderr[-2000:]))


def write_json(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w") as fp:
        json.dump(doc, fp, indent=1)
        fp.write("\n")
    os.replace(tmp, path)


def write_rmtb(path, meta, arrays):
    tmp = path + ".tmp"
    rmtb.write(tmp, meta, arrays)
    os.replace(tmp, path)


# ── the decision sources, replayed and featurized ─────────────────────────────────────────────

class Paths:
    def __init__(self, args):
        self.out = os.path.abspath(args.out)
        self.work = os.path.abspath(args.work)
        self.bin_dir = os.path.abspath(args.bin_dir)
        self.engine = os.path.abspath(args.engine)
        self.rmfeat = os.path.abspath(args.rmfeat)
        self.cards = os.path.join(self.work, "cards.rmstate")
        self.tables = os.path.join(self.work, "tables")


def export_tables(p):
    os.makedirs(p.tables, exist_ok=True)
    run([p.engine, "--cards-frame", p.cards], p.bin_dir)
    run([p.rmfeat, "--export-tables", p.cards, p.tables], p.bin_dir)


def sources(out_dir):
    """[(kind, name, log path)]: the committed decision logs, corpus first, each sorted by name."""
    out = []
    for kind in ("corpus", "selfplay"):
        for f in sorted(glob.glob(os.path.join(out_dir, "sources", kind, "*.rmlog"))):
            out.append((kind, os.path.basename(f)[:-len(".rmlog")], f))
    return out


def featurize_source(p, kind, name, log):
    """Replays `log` into a dump and featurizes it per viewer; {viewer: tensors path}."""
    d = os.path.join(p.work, "feat", kind)
    os.makedirs(d, exist_ok=True)
    dump = os.path.join(d, name + ".rms")
    run([p.engine, "--replay", log, "--machine", "--rmstate-dump", dump], p.bin_dir)
    out = {}
    for v in VIEWERS:
        t = os.path.join(d, "%s.%s.rmtb" % (name, v))
        run([p.rmfeat, "--featurize", dump, "--viewer", v, "--out", t, "--cards", p.cards, "--tables", p.tables],
            p.bin_dir)
        out[v] = t
    return out


def featurize_all(p, srcs, jobs):
    with CF.ThreadPoolExecutor(jobs) as ex:
        futs = [ex.submit(featurize_source, p, k, n, f) for k, n, f in srcs]
        return [f.result() for f in futs]


def decision_key(decs, i):
    """A decision's place in its match: (game, the sideboarding phase before the game's play, decision);
    a sideboarding phase numbers its decisions from 1 again."""
    return int(decs.fields["game"][i]), 1 - int(decs.fields["sideboarding"][i]), int(decs.fields["decision"][i])


def seat_decisions(files):
    """(seat Decisions, omniscient Decisions) of one source, aligned: every seat decision in match
    order (``decision_key``), and the omniscient viewer's decision of the same key."""
    a, b, o = (B.Decisions.from_file(files[v]) for v in VIEWERS)
    keys = [decision_key(a, i) + (0, i) for i in range(len(a))] + [decision_key(b, i) + (1, i) for i in range(len(b))]
    keys.sort()
    if len({k[:3] for k in keys}) != len(keys):
        sys.exit("open_vocab_baseline: %s: two seat decisions share a key" % files["A"])
    omni_of = {decision_key(o, i): i for i in range(len(o))}
    if len(omni_of) != len(o) or set(omni_of) != {k[:3] for k in keys}:
        sys.exit("open_vocab_baseline: %s: the omniscient decisions are not the seats'" % files["A"])
    seat = concat([a, b]).select([k[4] if k[3] == 0 else len(a) + k[4] for k in keys])
    omni = o.select([omni_of[k[:3]] for k in keys])
    return seat, omni


def concat(parts):
    arrays = {name: np.concatenate([p.arrays[name] for p in parts]) for name, _ in B.TENSORS}
    sizes = {k: np.concatenate([p.sizes[k] for p in parts]) for k in ("n", "r", "a")}
    fields = {k: np.concatenate([p.fields[k] for p in parts]) for k in B.DECISION_FIELDS}
    return B.Decisions(arrays, sizes, fields)


def decision_set(srcs, feats):
    """(seat Decisions, omniscient Decisions, source index [D]) of the fixture's decisions."""
    seat_parts, omni_parts, src_parts = [], [], []
    sp_seat, sp_omni, sp_src = [], [], []
    for i, ((kind, _, _), files) in enumerate(zip(srcs, feats)):
        s, o = seat_decisions(files)
        tgt = (seat_parts, omni_parts, src_parts) if kind == "corpus" else (sp_seat, sp_omni, sp_src)
        tgt[0].append(s)
        tgt[1].append(o)
        tgt[2].append(np.full(len(s), i, np.int64))
    if sp_seat:
        s, o, src = concat(sp_seat), concat(sp_omni), np.concatenate(sp_src)
        n = len(s)
        if n < SELFPLAY_DECISIONS:
            sys.exit("open_vocab_baseline: the self-play sources have %d decisions, fewer than %d"
                     % (n, SELFPLAY_DECISIONS))
        pick = [i * n // SELFPLAY_DECISIONS for i in range(SELFPLAY_DECISIONS)]
        seat_parts.append(s.select(pick))
        omni_parts.append(o.select(pick))
        src_parts.append(src[pick])
    return concat(seat_parts), concat(omni_parts), np.concatenate(src_parts)


def drop_types(decs, type_ids):
    """`decs` with every token of the types `type_ids` removed: the later tokens' indices in the
    relations and menu rows renumbered; a relation touching a removed token is an error (the
    removed types have none)."""
    starts = {k: np.concatenate([[0], np.cumsum(v)]) for k, v in decs.sizes.items()}
    out = {name: [] for name, _ in B.TENSORS}
    sizes = {"n": [], "r": [], "a": []}
    drop = np.isin(decs.arrays["tok_type"], np.asarray(type_ids, np.int64))
    for i in range(len(decs)):
        t0, t1 = starts["n"][i], starts["n"][i + 1]
        keep = ~drop[t0:t1]
        new_index = np.cumsum(keep) - 1
        for name, key in B.TENSORS:
            if key == "n":
                out[name].append(decs.arrays[name][t0:t1][keep])
        r = decs.arrays["rel"][starts["r"][i]:starts["r"][i + 1]].copy()
        if r.size and not (keep[r[:, 0]].all() and keep[r[:, 1]].all()):
            sys.exit("open_vocab_baseline: a relation touches a removed token")
        if r.size:
            r[:, 0] = new_index[r[:, 0]]
            r[:, 1] = new_index[r[:, 1]]
        out["rel"].append(r)
        at = decs.arrays["act_tok"][starts["a"][i]:starts["a"][i + 1]]
        if not keep[at].all():
            sys.exit("open_vocab_baseline: a menu row is a removed token")
        out["act_tok"].append(new_index[at])
        out["act_mask"].append(decs.arrays["act_mask"][starts["a"][i]:starts["a"][i + 1]])
        sizes["n"].append(int(keep.sum()))
        sizes["r"].append(len(r))
        sizes["a"].append(len(at))
    arrays = {name: np.concatenate(v) for name, v in out.items()}
    arrays["rel"] = arrays["rel"].reshape(-1, 3)
    return B.Decisions(arrays, {k: np.asarray(v, np.int64) for k, v in sizes.items()}, dict(decs.fields))


# ── the networks ──────────────────────────────────────────────────────────────────────────────

def checkpoint_networks(data):
    """(kind, hashes, model config, {role: state_dict}) of a loaded checkpoint."""
    if "learner" in data:
        L = data["learner"]
        nets = {"model": L["model"], "target": L["target"]}
        for i, r in enumerate(L["regs"]):
            if r.get("same_as") is None:
                nets["reg%d" % i] = r["state"]
        return "learner_state", L["hashes"], L["model_config"], nets
    return "save_checkpoint", data["hashes"], data["config"], {"model": data["state_dict"]}


def network_id(cfg, sd):
    h = hashlib.sha256(json.dumps(M.ModelConfig.from_dict(cfg).to_dict(), sort_keys=True).encode())
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(json.dumps(list(t.shape)).encode())
        h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.view(torch.int16).numpy().tobytes())
    return h.hexdigest()[:16]


def build_critic_fixture(tables, out_path):
    """m11 with an omniscient critic (see the module docstring); a save_checkpoint file."""
    src = os.path.join(REPO_ROOT, CRITIC_SOURCE)
    torch.manual_seed(CRITIC_SEED)
    model = RL._load_with_config(src, tables, CRITIC_CONFIG)
    g = torch.Generator().manual_seed(CRITIC_SEED + 1)
    with torch.no_grad():
        for layer in model.critic.encoder.layers:
            layer.rel_bias.weight.copy_(torch.randn(layer.rel_bias.weight.shape, generator=g) * CRITIC_REL_BIAS_STD)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    M.save_checkpoint(model, out_path, extra={"built_by": "learn/open_vocab_baseline.py build_critic_fixture",
                                             "source": CRITIC_SOURCE, "source_sha256": sha256_file(src),
                                             "config": CRITIC_CONFIG, "seed": CRITIC_SEED,
                                             "rel_bias_std": CRITIC_REL_BIAS_STD})
    return {"path": rel(out_path), "source": CRITIC_SOURCE, "source_sha256": sha256_file(src),
            "config_override": CRITIC_CONFIG, "torch_seed": CRITIC_SEED, "rel_bias_seed": CRITIC_SEED + 1,
            "rel_bias_std": CRITIC_REL_BIAS_STD,
            "how": "torch.manual_seed(torch_seed); rnad_learner._load_with_config(source, tables, config_override); "
                   "each critic layer's rel_bias.weight = randn(generator seeded rel_bias_seed) * rel_bias_std; "
                   "model.save_checkpoint"}


# ── evaluation (worker processes) ─────────────────────────────────────────────────────────────

_W = {}


def _worker_init(tables_dir, work, threads):
    torch.set_num_threads(threads)
    tables = S.load_tables(tables_dir)
    d = {}
    for name in ("seat", "omni", "seat_nolists", "omni_nolists"):
        d[name] = load_decisions(os.path.join(work, "decisions", name + ".rmtb"))
    _W.update(tables=tables, decs=d)


def save_decisions(path, decs):
    arrays = {}
    for name, key in B.TENSORS:
        arrays[name] = decs.arrays[name]
        arrays[name + "_count"] = decs.sizes[key]
    for k in B.DECISION_FIELDS:
        arrays[k] = decs.fields[k]
    write_rmtb(path, {"kind": "open_vocab_decisions"}, arrays)


def load_decisions(path):
    _, arr = rmtb.read(path)
    key_of = dict(B.TENSORS)
    arrays = {name: np.array(arr[name]) for name, _ in B.TENSORS}
    sizes = {key_of[name]: np.array(arr[name + "_count"]) for name in ("tok_type", "rel", "act_tok")}
    fields = {k: np.array(arr[k]) for k in B.DECISION_FIELDS}
    return B.Decisions(arrays, sizes, fields)


def forward_all(fn, decs, omni_only=False):
    """[D] values (and, unless `omni_only`, the real menu rows' logits concatenated) of `fn` over
    `decs` in fixed batches: the decisions sorted (stably) by token count, BATCH_ROWS a forward."""
    order = np.argsort(decs.sizes["n"], kind="stable")
    groups = B.forward_groups(decs.sizes["n"], order, max_rows=BATCH_ROWS)
    D = len(decs)
    value = np.zeros(D, np.float32)
    logits = [None] * D
    spec = _W["tables"].spec
    for g in groups:
        sub = decs.select(g)
        x = B.pad(sub, spec)
        inputs = B.to_torch(x)
        lg, v = fn(inputs, sub.fields["viewer"])
        value[g] = v.numpy()
        if not omni_only:
            a = sub.sizes["a"]
            for j, i in enumerate(g):
                logits[i] = lg[j, :a[j]].numpy().copy()
    return value, (np.concatenate(logits).astype(np.float32) if not omni_only else None)


def evaluate_network(job):
    ckpt, role, net_id, out_path, meta = job
    tables = _W["tables"]
    data = torch.load(ckpt, map_location="cpu", weights_only=True)
    _, _, cfg, nets = checkpoint_networks(data)
    model = M.RMNet(M.ModelConfig.from_dict(cfg), tables)
    model.load_state_dict(nets[role])
    model.eval()
    del data, nets

    def policy(inputs, _viewer):
        out = model(*inputs, aux=False)
        return out["logits"], out["value"]

    def critic(inputs, viewer):
        seat = torch.from_numpy(np.asarray(viewer, np.int64))
        x = model.critic(model.cards, 0.0, *inputs[:8], seat)
        return None, x

    arrays = {}
    with torch.no_grad():
        for suffix in ("", "_nolists"):
            v, lg = forward_all(policy, _W["decs"]["seat" + suffix])
            arrays["logits" + suffix] = lg
            arrays["value" + suffix] = v
        if model.critic is not None:
            # The omniscient decisions carry the seat decisions' viewer as their seat (see main).
            for suffix in ("", "_nolists"):
                v, _ = forward_all(critic, _W["decs"]["omni" + suffix], omni_only=True)
                arrays["critic_value" + suffix] = v
    write_rmtb(out_path, meta, arrays)
    return net_id


# ── generate ──────────────────────────────────────────────────────────────────────────────────

def list_checkpoints():
    paths = sorted(glob.glob(os.path.join(REPO_ROOT, "runs", "**", "ckpt", "**", "*.pt"), recursive=True))
    return [p for p in paths if not p.endswith(".ts.pt")]


def generate(args):
    p = Paths(args)
    torch.set_num_threads(args.threads)
    os.makedirs(p.work, exist_ok=True)
    export_tables(p)
    tables = S.load_tables(p.tables)
    spec = tables.spec
    removed = [spec.type_ids[t] for t in REMOVED_TYPES if t in spec.type_ids]

    srcs = sources(p.out)
    if not srcs:
        sys.exit("open_vocab_baseline: no decision sources under %s/sources (run select)" % p.out)
    feats = featurize_all(p, srcs, args.jobs)
    seat, omni, src_index = decision_set(srcs, feats)
    # The critic's seat is the deciding seat: the omniscient decisions take the seat decisions' viewer.
    omni.fields["viewer"] = seat.fields["viewer"].copy()
    ddir = os.path.join(p.work, "decisions")
    os.makedirs(ddir, exist_ok=True)
    save_decisions(os.path.join(ddir, "seat.rmtb"), seat)
    save_decisions(os.path.join(ddir, "omni.rmtb"), omni)
    seat_nl, omni_nl = drop_types(seat, removed), drop_types(omni, removed)
    save_decisions(os.path.join(ddir, "seat_nolists.rmtb"), seat_nl)
    save_decisions(os.path.join(ddir, "omni_nolists.rmtb"), omni_nl)

    bdir = os.path.join(p.out, "baseline")
    if os.path.isdir(bdir):
        shutil.rmtree(bdir)
    os.makedirs(bdir)
    menu_off = np.concatenate([[0], np.cumsum(seat.sizes["a"])]).astype(np.int64)
    write_rmtb(os.path.join(bdir, "decisions.rmtb"),
               {"kind": "open_vocab_decisions_index", "vocab_hash": spec.vocab_hash, "spec_hash": spec.hash},
               {"source": src_index, "viewer": seat.fields["viewer"], "game": seat.fields["game"],
                "decision": seat.fields["decision"], "sideboarding": seat.fields["sideboarding"],
                "menu_offsets": menu_off})

    critic = build_critic_fixture(tables, os.path.join(REPO_ROOT, CRITIC_FIXTURE))

    checkpoints, networks, jobs = [], {}, []
    for ck in list_checkpoints():
        entry = {"path": rel(ck), "sha256": sha256_file(ck), "bytes": os.path.getsize(ck)}
        data = torch.load(ck, map_location="cpu", weights_only=True)
        kind, hashes, cfg, nets = checkpoint_networks(data)
        entry["kind"] = kind
        entry["hashes"] = {k: hashes.get(k) for k in ("spec_hash", "vocab_hash", "text_model")}
        bad = [k for k in ("spec_hash", "vocab_hash", "text_model") if hashes.get(k) != tables.hashes[k]]
        if bad:
            entry["runnable"] = False
            entry["reason"] = "; ".join("%s %s is not the current tables' %s" % (k, hashes.get(k), tables.hashes[k])
                                        for k in bad)
            checkpoints.append(entry)
            continue
        entry["runnable"] = True
        entry["networks"] = {}
        for role in sorted(nets):
            nid = network_id(cfg, nets[role])
            entry["networks"][role] = nid
            if nid not in networks:
                networks[nid] = {"file": "baseline/%s.rmtb" % nid, "first_seen": "%s:%s" % (rel(ck), role),
                                 "config": M.ModelConfig.from_dict(cfg).to_dict(),
                                 "critic": bool(M.ModelConfig.from_dict(cfg).omniscient_critic)}
                meta = {"kind": "open_vocab_baseline", "vocab_hash": spec.vocab_hash, "spec_hash": spec.hash,
                        "network": nid}
                jobs.append((ck, role, nid, os.path.join(bdir, nid + ".rmtb"), meta))
        checkpoints.append(entry)
        del data, nets

    ctx = MP.get_context("spawn")
    with CF.ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_worker_init,
                                initargs=(p.tables, p.work, args.threads)) as ex:
        for nid in ex.map(evaluate_network, jobs):
            print("open_vocab_baseline: network %s done" % nid, flush=True)
    for nid, n in networks.items():
        n["sha256"] = sha256_file(os.path.join(p.out, n["file"]))

    src_docs = []
    for i, (kind, name, log) in enumerate(srcs):
        src_docs.append({"index": i, "kind": kind, "name": name, "log": os.path.relpath(log, p.out),
                         "sha256": sha256_file(log),
                         "decisions": int((src_index == i).sum())})
    manifest = {
        "format": FORMAT,
        "plan": "docs/open_vocab_plan.md S0",
        "generator": "learn/open_vocab_baseline.py generate",
        "tables": {"spec_hash": spec.hash, "vocab_hash": spec.vocab_hash, "text_model": tables.hashes["text_model"],
                   "spec_json_sha256": sha256_file(os.path.join(p.tables, "spec.json")),
                   "card_rmtb_sha256": sha256_file(os.path.join(p.tables, "card.rmtb")),
                   "ability_rmtb_sha256": sha256_file(os.path.join(p.tables, "ability.rmtb")),
                   "part_rmtb_sha256": sha256_file(os.path.join(p.tables, "part.rmtb")),
                   "text_table_sha256": sha256_file(S.COMMITTED_TEXT_TABLE)},
        "torch": torch.__version__, "numpy": np.__version__, "threads": args.threads,
        "evaluation": {"precision": "fp32", "device": "cpu", "mode": "eval, torch.no_grad (the export's path)",
                       "batching": "decisions stably sorted by token count, batch.forward_groups max_rows %d, "
                                   "batch.pad" % BATCH_ROWS,
                       "removed_types": list(REMOVED_TYPES), "removed_type_ids": removed,
                       "critic_seat": "the deciding seat (0 = A)"},
        "decision_set": {"corpus": "every seat decision of sources/corpus/*.rmlog",
                         "selfplay": "%d of the seat decisions of sources/selfplay/*.rmlog (all of them in source "
                                     "then match order, index i * n // %d)"
                                     % (SELFPLAY_DECISIONS, SELFPLAY_DECISIONS),
                         "decisions": len(seat), "menu_rows": int(menu_off[-1]),
                         "replay": "robomage --replay LOG --machine --rmstate-dump DUMP (cwd bin/); "
                                   "rmfeat --featurize DUMP --viewer A|B|omniscient --cards CARDS --tables TABLES"},
        "sources": src_docs,
        "critic_fixture": critic,
        "best_policy": {"checkpoint": "runs/league24h-sb/ckpt/fixed/m11.pt",
                        "why": "runs/league24h-sb/bench/end/summary.md: m10 and m11 0.499 vs scripted:hard, "
                               "+0.041 paired over bc-final (final 0.495, +0.037); bench/end-vs-hard_sb: m10 "
                               "+0.036, m11 +0.034, final +0.037: tied within their SEs; m11 is the latest fixed "
                               "point and pilotA's warm start; the pilots have no benchmark"},
        "checkpoints": checkpoints,
        "networks": dict(sorted(networks.items())),
    }
    write_json(os.path.join(p.out, "manifest.json"), manifest)
    n_run = sum(1 for c in checkpoints if c["runnable"])
    print("open_vocab_baseline: %d decisions (%d menu rows), %d checkpoints (%d runnable), %d networks in %s"
          % (len(seat), int(menu_off[-1]), len(checkpoints), n_run, len(networks), p.out))


# ── select ────────────────────────────────────────────────────────────────────────────────────

def select(args):
    """Copies the corpus logs and the self-play matches' logs into OUT/sources and checks that each
    self-play log's replay featurizes to its episodes' seat tensors (tok_num at the episodes' f16)."""
    p = Paths(args)
    os.makedirs(p.work, exist_ok=True)
    export_tables(p)
    cdir = os.path.join(p.out, "sources", "corpus")
    sdir = os.path.join(p.out, "sources", "selfplay")
    os.makedirs(cdir, exist_ok=True)
    os.makedirs(sdir, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(REPO_ROOT, "tests", "corpus", "*.rmlog"))):
        shutil.copyfile(f, os.path.join(cdir, os.path.basename(f)))
    run_dir = os.path.join(REPO_ROOT, SELFPLAY_RUN)
    for m in SELFPLAY_MATCHES:
        log = os.path.join(run_dir, "rmlog", "selfplay", m + ".rmlog")
        files = featurize_source(p, "select", m, log)
        seat, _ = seat_decisions(files)
        eps = sorted(glob.glob(os.path.join(run_dir, "episodes", "selfplay", m + ".g*.rmtraj")))
        items = [(ep, d) for ep in map(EP.read_episode, eps) for d in range(ep.steps)]
        if len(items) != len(seat):
            sys.exit("open_vocab_baseline: %s: %d replayed decisions, %d episode steps" % (m, len(seat), len(items)))
        spec = S.load_spec(os.path.join(p.tables, "spec.json"))
        for lo in range(0, len(items), 64):
            chunk = list(range(lo, min(lo + 64, len(items))))
            x = EP.expand([items[i] for i in chunk])
            y = B.pad(seat.select(chunk), spec)
            for k in B.INPUTS:
                a, b = x[k], y[k]
                if k == "tok_num":
                    b = b.astype(np.float16).astype(np.float32)
                if a.shape != b.shape or not np.array_equal(a, b):
                    sys.exit("open_vocab_baseline: %s: decisions %d-%d: %s differs from the episodes'"
                             % (m, chunk[0], chunk[-1], k))
        shutil.copyfile(log, os.path.join(sdir, m + ".rmlog"))
        print("open_vocab_baseline: %s: %d decisions, the replay featurizes to the episodes" % (m, len(seat)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["generate", "select"])
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "tests", "open_vocab"))
    ap.add_argument("--work", default=os.path.join(REPO_ROOT, "runs", "open_vocab_s0"))
    ap.add_argument("--bin-dir", default=os.path.join(REPO_ROOT, "bin"))
    ap.add_argument("--engine", default=os.path.join(REPO_ROOT, "bin", "debug", "robomage"))
    ap.add_argument("--rmfeat", default=os.path.join(REPO_ROOT, "bin", "debug", "rmfeat"))
    ap.add_argument("--jobs", type=int, default=16, help="engine / rmfeat processes at once")
    ap.add_argument("--workers", type=int, default=8, help="network evaluation processes")
    ap.add_argument("--threads", type=int, default=4, help="torch intra-op threads per process (recorded)")
    args = ap.parse_args(argv)
    if args.command == "generate":
        generate(args)
    else:
        select(args)


if __name__ == "__main__":
    main()
