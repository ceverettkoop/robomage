#!/usr/bin/env python3
"""The reference policy of the network contract (plan step 5d, docs/featurizer.md §6).

NEEDS torch (and, for the ONNX export and --check, onnx, onnxscript and onnxruntime), e.g.
/root/torchenv/bin/python; `make` never runs it (the gate's `actor` tier does, when asked).

  build   reads the exported card tables (rmfeat --export-tables DIR: card.rmtb, ability.rmtb,
          part.rmtb, spec.json, the FEATURE_SPEC document) and the text table, builds the recommended
          model family with a seeded random initialisation and exports it three ways:
            PREFIX.ts.pt   TorchScript (torch.jit.script), the hashes as string attributes
            PREFIX.pt2     an AOTInductor package (torch.export + aoti_compile_and_package, CPU), the
                           hashes in the package's metadata (inductor config aot_inductor.metadata)
            PREFIX.onnx    ONNX (the dynamo exporter), the hashes in the model's metadata_props
          Each carries spec_hash, vocab_hash, text_model and policy_version (strings), what the
          actor's evaluator (src/actor/evaluator.h) checks on load.
  check   loads the three exports (torch.jit.load, torch._inductor.aoti_load_package, an
          onnxruntime session) and evaluates the decisions of a tensors file (rmfeat --featurize),
          one at a time and as padded batches; prints the largest logit and value differences of
          each export against the TorchScript one at batch 1 (and TorchScript's own batched against
          unbatched) and exits 1 past the tolerance (1e-4, absolute plus relative).

    python make_reference_policy.py build --tables DIR --out PREFIX [--text-table FILE] [--seed N]
    python make_reference_policy.py check --out PREFIX --tables DIR --tensors FILE [--decisions N] [--batch B]

The model (featurizer.md §6): per token type a linear projection of tok_num, plus the type's
embedding and the summed embeddings of tok_cat (one table over the global categorical id space,
id 0 the pad), plus projections of the gathered card-table and text-table rows of each tok_card slot
and of the ability rows (each with its parts' projections averaged in) of each tok_ability slot
(the tables' projections are folded into buffers before export, constants of the weights, so a
forward gathers rows instead of projecting the whole vocab);
an entity transformer (pre-LayerNorm) whose attention adds, per head, a learned bias for each
relation type at (src, dst) and another at (dst, src), scattered into [B, H, N, N]; padded tokens
are masked as keys. A pointer head (an MLP over each act_tok token's output) gives the logits
(padded menu rows -1e9), a value head (an MLP over the mean of the real tokens, tanh) the value.
Export-friendly: no control flow that depends on the data; gathers and index_add only.

Inputs as the actor's padding batcher (src/featurize/batch.h) lays them out: every decision padded
to the batch's largest N, R, A rounded up to a multiple of 8; pads tok_card / tok_ability -1, the
masks 0, everything else 0.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import rmtb  # noqa: E402

TEXT_TABLE = os.path.join(REPO_ROOT, "bin", "resources", "embeddings", "oracle_text.rmtb")
INPUTS = ["tok_type", "tok_num", "tok_cat", "tok_card", "tok_ability", "tok_mask", "rel", "rel_mask", "act_tok",
          "act_mask"]
PAD_MULTIPLE = 8
TOLERANCE = 1e-4


def fnv1a64(data):
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def table_tensor(arrays, name):
    a = arrays[name]
    return torch.tensor(a.values, dtype=torch.float32 if a.dtype in ("f32", "f16") else torch.int64).view(*a.shape)


class Layer(nn.Module):
    def __init__(self, d, heads, n_bias):
        super().__init__()
        self.heads = heads
        self.dh = d // heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff1 = nn.Linear(d, 2 * d)
        self.ff2 = nn.Linear(2 * d, d)
        self.rel_bias = nn.Embedding(n_bias, heads)

    def forward(self, x, bias_idx_fwd, bias_idx_rev, rel_type, rel_w, n_rel: int, key_bias):
        b, n, d = x.shape
        h = self.heads
        # Relation biases: per head, the type's bias at (src, dst) and the reverse type's at (dst, src).
        vf = self.rel_bias(rel_type) * rel_w
        vr = self.rel_bias(rel_type + n_rel) * rel_w
        bias = torch.zeros(b * n * n, h, dtype=x.dtype, device=x.device)
        bias = bias.index_add(0, bias_idx_fwd, vf.reshape(-1, h)).index_add(0, bias_idx_rev, vr.reshape(-1, h))
        bias = bias.view(b, n, n, h).permute(0, 3, 1, 2)
        y = self.ln1(x)
        qkv = self.qkv(y).view(b, n, 3, h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh) + bias + key_bias
        att = torch.softmax(att, dim=-1)
        y = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(b, n, d)
        x = x + self.out(y)
        return x + self.ff2(torch.nn.functional.gelu(self.ff1(self.ln2(x))))


class Policy(nn.Module):
    def __init__(self, spec, card, text, ability, part_width, hashes, d=64, heads=4, layers=2):
        super().__init__()
        self.spec_hash = hashes["spec_hash"]
        self.vocab_hash = hashes["vocab_hash"]
        self.text_model = hashes["text_model"]
        self.policy_version = hashes["policy_version"]
        self.n_types = len(spec["token_types"])
        self.n_rel = len(spec["relation_types"])
        self.d = d
        self.F, self.C, self.K, self.Q = spec["F"], spec["C"], spec["K"], spec["Q"]
        # The tables' rows as the model reads them (fold): per card id its card-table and text-table
        # rows' projections (a -1 slot reads the none row, masked out), per ability row its
        # projection plus the mean of its parts', and a zero row for -1.
        self.register_buffer("card_rows", torch.zeros(card.shape[0], d))
        self.register_buffer("ability_rows", torch.zeros(ability.shape[0] + 1, d))
        self.none_card = card.shape[0] - 1
        self.none_ability = ability.shape[0]
        self.num_w = nn.Parameter(torch.randn(self.n_types, self.F, d) / math.sqrt(self.F))
        self.type_emb = nn.Embedding(self.n_types, d)
        self.cat_emb = nn.Embedding(spec["categorical_size"], d, padding_idx=0)
        self.card_proj = nn.Linear(card.shape[1], d)
        self.text_proj = nn.Linear(text.shape[1], d)
        self.card_slot = nn.Parameter(torch.randn(max(self.K, 1), d) * 0.1)
        self.ability_proj = nn.Linear(ability.shape[1], d)
        self.part_proj = nn.Linear(part_width, d)
        self.ability_slot = nn.Parameter(torch.randn(max(self.Q, 1), d) * 0.1)
        self.layers = nn.ModuleList([Layer(d, heads, 2 * max(self.n_rel, 1)) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.ptr1 = nn.Linear(d, d)
        self.ptr2 = nn.Linear(d, 1)
        self.val1 = nn.Linear(d, d)
        self.val2 = nn.Linear(d, 1)

    @torch.jit.ignore
    def fold(self, card, text, ability, part, part_ability):
        """The frozen tables' projections into card_rows / ability_rows: constants of the weights for
        an inference export (a learner computes them in its forward)."""
        with torch.no_grad():
            self.card_rows.copy_(self.card_proj(card) + self.text_proj(text))
            parts = self.part_proj(part)
            na = ability.shape[0]
            part_sum = torch.zeros(na, self.d).index_add(0, part_ability, parts)
            part_n = torch.zeros(na, 1).index_add(0, part_ability, torch.ones(parts.shape[0], 1))
            self.ability_rows[:na].copy_(self.ability_proj(ability) + part_sum / part_n.clamp(min=1.0))
            self.ability_rows[na:].zero_()

    def forward(self, tok_type, tok_num, tok_cat, tok_card, tok_ability, tok_mask, rel, rel_mask, act_tok,
                act_mask):
        b = tok_type.shape[0]
        n = tok_type.shape[1]
        d = self.d
        # Each token's own type's projection, its weights gathered per token.
        w = torch.nn.functional.embedding(tok_type, self.num_w.view(self.n_types, self.F * d)).view(b, n, self.F, d)
        x = torch.matmul(tok_num.unsqueeze(-2), w).squeeze(-2)
        x = x + self.type_emb(tok_type) + self.cat_emb(tok_cat).sum(2)
        # Card slots: the folded card rows (-1 → the none row).
        card_idx = torch.where(tok_card < 0, torch.full_like(tok_card, self.none_card), tok_card)
        card_w = (tok_card >= 0).to(x.dtype).unsqueeze(-1)
        x = x + ((torch.nn.functional.embedding(card_idx, self.card_rows) + self.card_slot) * card_w).sum(2)
        # Ability slots: the folded ability rows (-1 → the zero row).
        ab_idx = torch.where(tok_ability < 0, torch.full_like(tok_ability, self.none_ability), tok_ability)
        ab_w = (tok_ability >= 0).to(x.dtype).unsqueeze(-1)
        x = x + ((torch.nn.functional.embedding(ab_idx, self.ability_rows) + self.ability_slot) * ab_w).sum(2)
        tok_w = tok_mask.to(x.dtype).unsqueeze(-1)
        x = x * tok_w
        # Relations as flat indices into [B, N, N].
        base = torch.arange(b, device=rel.device).view(b, 1) * (n * n)
        src = rel[:, :, 0]
        dst = rel[:, :, 1]
        idx_fwd = (base + src * n + dst).reshape(-1)
        idx_rev = (base + dst * n + src).reshape(-1)
        rel_type = rel[:, :, 2]
        rel_w = rel_mask.to(x.dtype).unsqueeze(-1)
        key_bias = (1.0 - tok_mask.to(x.dtype)).view(b, 1, 1, n) * -1e9
        for layer in self.layers:
            x = layer(x, idx_fwd, idx_rev, rel_type, rel_w, self.n_rel, key_bias)
        x = self.ln_f(x)
        a = act_tok.shape[1]
        acts = x.gather(1, act_tok.unsqueeze(-1).expand(b, a, d))
        logits = self.ptr2(torch.nn.functional.gelu(self.ptr1(acts))).squeeze(-1)
        logits = torch.where(act_mask, logits, torch.full_like(logits, -1e9))
        mean = (x * tok_w).sum(1) / tok_w.sum(1).clamp(min=1.0)
        value = torch.tanh(self.val2(torch.nn.functional.gelu(self.val1(mean)))).squeeze(-1)
        return logits, value


# ── inputs ────────────────────────────────────────────────────────────────────────────────────

def read_np(path):
    """(meta, {name: np.ndarray}) of an RMTB file, by the layout of featurizer.md §4.4."""
    with open(path, "rb") as fp:
        data = fp.read()
    meta_len = int.from_bytes(data[8:12], "little")
    meta = json.loads(data[12:12 + meta_len].decode("utf-8"))
    dt = {"f32": np.float32, "f16": np.float16, "i64": np.int64, "i32": np.int32, "u16": np.uint16, "u8": np.uint8}
    out = {}
    for a in meta["arrays"]:
        count = int(np.prod(a["shape"])) if a["shape"] else 1
        out[a["name"]] = np.frombuffer(data, dt[a["dtype"]], count, 12 + meta_len + a["offset"]).reshape(a["shape"])
    return meta, out


def decisions(tensors_path, limit):
    """The decisions of a tensors file as dicts of unpadded arrays."""
    _, arr = read_np(tensors_path)
    n = len(arr["viewer"])
    out = []
    for i in range(min(n, limit)):
        dec = {}
        for name in ["tok_type", "tok_num", "tok_cat", "tok_card", "tok_ability", "tok_mask", "rel", "act_tok",
                     "act_mask"]:
            off = arr[name + "_offsets"]
            dec[name] = arr[name][off[i]:off[i + 1]]
        out.append(dec)
    return out


def padded(size):
    return max(PAD_MULTIPLE, (size + PAD_MULTIPLE - 1) // PAD_MULTIPLE * PAD_MULTIPLE)


def pad_batch(batch, spec):
    """The contract's padded inputs (numpy) of a batch of decisions, as src/featurize/batch.cpp pads."""
    b = len(batch)
    n = padded(max(len(d["tok_type"]) for d in batch))
    r = padded(max(len(d["rel"]) for d in batch))
    a = padded(max(len(d["act_tok"]) for d in batch))
    F, C, K, Q = spec["F"], spec["C"], spec["K"], spec["Q"]
    x = {
        "tok_type": np.zeros((b, n), np.int64), "tok_num": np.zeros((b, n, F), np.float32),
        "tok_cat": np.zeros((b, n, C), np.int64), "tok_card": np.full((b, n, K), -1, np.int64),
        "tok_ability": np.full((b, n, Q), -1, np.int64), "tok_mask": np.zeros((b, n), np.bool_),
        "rel": np.zeros((b, r, 3), np.int64), "rel_mask": np.zeros((b, r), np.bool_),
        "act_tok": np.zeros((b, a), np.int64), "act_mask": np.zeros((b, a), np.bool_),
    }
    for i, d in enumerate(batch):
        t, rr, aa = len(d["tok_type"]), len(d["rel"]), len(d["act_tok"])
        x["tok_type"][i, :t] = d["tok_type"]
        x["tok_num"][i, :t] = d["tok_num"]
        x["tok_cat"][i, :t] = d["tok_cat"]
        x["tok_card"][i, :t] = d["tok_card"]
        x["tok_ability"][i, :t] = d["tok_ability"]
        x["tok_mask"][i, :t] = d["tok_mask"] != 0
        x["rel"][i, :rr] = d["rel"]
        x["rel_mask"][i, :rr] = True
        x["act_tok"][i, :aa] = d["act_tok"]
        x["act_mask"][i, :aa] = d["act_mask"] != 0
    return [x[k] for k in INPUTS]


def example_inputs(spec, card_rows, ability_rows, n_rel, b=2, n=16, r=24, a=8, seed=0):
    """Random inputs of the contract's shapes (for tracing and export)."""
    g = torch.Generator().manual_seed(seed)
    T = len(spec["token_types"])
    tok_card = torch.randint(-1, card_rows - 1, (b, n, spec["K"]), generator=g)
    tok_ability = torch.randint(-1, ability_rows, (b, n, spec["Q"]), generator=g)
    tok_mask = torch.ones(b, n, dtype=torch.bool)
    tok_mask[:, n - 3:] = False
    rel = torch.stack([torch.randint(0, n - 3, (b, r), generator=g), torch.randint(0, n - 3, (b, r), generator=g),
                       torch.randint(0, max(n_rel, 1), (b, r), generator=g)], -1)
    rel_mask = torch.ones(b, r, dtype=torch.bool)
    rel_mask[:, r - 4:] = False
    act_mask = torch.ones(b, a, dtype=torch.bool)
    act_mask[:, a - 2:] = False
    return (torch.randint(0, T, (b, n), generator=g), torch.randn(b, n, spec["F"], generator=g),
            torch.randint(0, spec["categorical_size"], (b, n, spec["C"]), generator=g), tok_card, tok_ability,
            tok_mask, rel, rel_mask, torch.randint(0, n - 3, (b, a), generator=g), act_mask)


# ── build ─────────────────────────────────────────────────────────────────────────────────────

def load_inputs(tables_dir, text_table):
    with open(os.path.join(tables_dir, "spec.json"), "rb") as fp:
        spec_bytes = fp.read().rstrip(b"\n")
    spec = json.loads(spec_bytes)
    card_meta, card = rmtb.read(os.path.join(tables_dir, "card.rmtb"))
    ab_meta, ability = rmtb.read(os.path.join(tables_dir, "ability.rmtb"))
    part_meta, part = rmtb.read(os.path.join(tables_dir, "part.rmtb"))
    text_meta, text = rmtb.read(text_table)
    spec_hash = "%016x" % fnv1a64(spec_bytes)
    for m in (card_meta, ab_meta, part_meta):
        if m.get("spec_hash") != spec_hash:
            raise SystemExit("make_reference_policy: %s's tables carry spec hash %s, spec.json hashes to %s"
                             % (tables_dir, m.get("spec_hash"), spec_hash))
    if text_meta.get("vocab_hash") != card_meta.get("vocab_hash"):
        raise SystemExit("make_reference_policy: the text table is of another vocab")
    hashes = {"spec_hash": spec_hash, "vocab_hash": card_meta["vocab_hash"], "text_model": text_meta["model"]}
    tables = (table_tensor(card, "features"), table_tensor(text, "vectors"), table_tensor(ability, "features"),
              table_tensor(part, "features"), table_tensor(part, "ability"))
    return spec, tables, hashes


def dynamic_shapes():
    from torch.export import Dim
    B, N, R, A = Dim("B", min=1, max=4096), Dim("N", min=2, max=65536), Dim("R", min=2, max=1 << 20), \
        Dim("A", min=2, max=65536)
    tok = {0: B, 1: N}
    return {"tok_type": tok, "tok_num": tok, "tok_cat": tok, "tok_card": tok, "tok_ability": tok, "tok_mask": tok,
            "rel": {0: B, 1: R}, "rel_mask": {0: B, 1: R}, "act_tok": {0: B, 1: A}, "act_mask": {0: B, 1: A}}


def build(args):
    torch.manual_seed(args.seed)
    spec, tables, hashes = load_inputs(args.tables, args.text_table)
    hashes["policy_version"] = str(args.policy_version)
    model = Policy(spec, tables[0], tables[1], tables[2], tables[3].shape[1], hashes, d=args.dim, heads=args.heads,
                   layers=args.layers).eval()
    # Every parameter perturbed so biases and LayerNorms are not trivial.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    model.fold(*tables)
    ex = example_inputs(spec, tables[0].shape[0], tables[2].shape[0], model.n_rel)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        ref_logits, ref_value = model(*ex)

    scripted = torch.jit.script(model)
    scripted.save(args.out + ".ts.pt")
    print("make_reference_policy: wrote %s.ts.pt" % args.out)

    ep = torch.export.export(model, ex, dynamic_shapes=dynamic_shapes(), strict=False)
    metadata = dict(hashes)
    torch._inductor.aoti_compile_and_package(ep, package_path=args.out + ".pt2",
                                             inductor_configs={"aot_inductor.metadata": metadata})
    print("make_reference_policy: wrote %s.pt2" % args.out)

    import onnx
    onnx_path = args.out + ".onnx"
    prog = torch.onnx.export(model, ex, dynamo=True, dynamic_shapes=dynamic_shapes(), input_names=INPUTS,
                             output_names=["logits", "value"], external_data=False)
    prog.save(onnx_path)
    m = onnx.load(onnx_path)
    for k, v in hashes.items():
        m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    onnx.save(m, onnx_path)
    print("make_reference_policy: wrote %s" % onnx_path)

    # The exports reproduce the eager model on the example inputs.
    with torch.no_grad():
        for name, (lg, vl) in (("ts", torch.jit.load(args.out + ".ts.pt")(*ex)),
                               ("pt2", torch._inductor.aoti_load_package(args.out + ".pt2")(*ex))):
            dl = (lg - ref_logits).abs().max().item()
            dv = (vl - ref_value).abs().max().item()
            print("make_reference_policy: %s vs eager on the example inputs: logits %.3g, value %.3g" % (name, dl, dv))
    print("make_reference_policy: spec %s vocab %s text %s policy_version %s" % (
        hashes["spec_hash"], hashes["vocab_hash"], hashes["text_model"], hashes["policy_version"]))
    return 0


# ── check ─────────────────────────────────────────────────────────────────────────────────────

def close(a, b):
    return np.abs(a - b) <= TOLERANCE + TOLERANCE * np.abs(b)


def check(args):
    import onnx
    import onnxruntime as ort
    # aoti_load_package reads torch._inductor.codecache, which torch 2.10 no longer imports with
    # torch._inductor (build imports it by compiling first).
    import torch._inductor.codecache  # noqa: F401
    ts = torch.jit.load(args.out + ".ts.pt")
    pt2 = torch._inductor.aoti_load_package(args.out + ".pt2")
    sess = ort.InferenceSession(args.out + ".onnx", providers=["CPUExecutionProvider"])
    metas = {
        "ts": {k: getattr(ts, k) for k in ("spec_hash", "vocab_hash", "text_model", "policy_version")},
        "pt2": {k: v for k, v in pt2.get_metadata().items()
                if k in ("spec_hash", "vocab_hash", "text_model", "policy_version")},
        "onnx": {p.key: p.value for p in onnx.load(args.out + ".onnx").metadata_props},
    }
    ok = True
    for name, m in metas.items():
        if m != metas["ts"]:
            print("FAIL: %s metadata %r differs from the TorchScript export's %r" % (name, m, metas["ts"]))
            ok = False
    with open(os.path.join(args.tables, "spec.json")) as fp:
        spec = json.load(fp)
    decs = decisions(args.tensors, args.decisions)
    if not decs:
        print("FAIL: %s has no decisions" % args.tensors)
        return 1

    def run_all(batch):
        x = pad_batch(batch, spec)
        tx = [torch.from_numpy(np.ascontiguousarray(v)) for v in x]
        with torch.no_grad():
            r_ts = [t.numpy() for t in ts(*tx)]
            r_pt2 = [t.numpy() for t in pt2(*tx)]
        r_onnx = sess.run(["logits", "value"], {k: v for k, v in zip(INPUTS, x)})
        return {"ts": r_ts, "pt2": r_pt2, "onnx": r_onnx}

    worst = {"pt2": [0.0, 0.0], "onnx": [0.0, 0.0], "ts-batched": [0.0, 0.0]}
    single = []
    for d in decs:
        r = run_all([d])
        a = len(d["act_tok"])
        single.append((r["ts"][0][0, :a], r["ts"][1][0]))
        for name in ("pt2", "onnx"):
            lg, vl = r[name][0][0, :a], r[name][1][0]
            worst[name][0] = max(worst[name][0], float(np.abs(lg - single[-1][0]).max()))
            worst[name][1] = max(worst[name][1], float(abs(vl - single[-1][1])))
            ok = ok and bool(close(lg, single[-1][0]).all()) and bool(close(np.array(vl), np.array(single[-1][1])))
    for start in range(0, len(decs), args.batch):
        batch = decs[start:start + args.batch]
        r = run_all(batch)
        for i, d in enumerate(batch):
            a = len(d["act_tok"])
            ref_l, ref_v = single[start + i]
            for name, key in (("ts-batched", "ts"), ("pt2", "pt2"), ("onnx", "onnx")):
                lg, vl = r[key][0][i, :a], r[key][1][i]
                worst[name][0] = max(worst[name][0], float(np.abs(lg - ref_l).max()))
                worst[name][1] = max(worst[name][1], float(abs(vl - ref_v)))
                ok = ok and bool(close(lg, ref_l).all()) and bool(close(np.array(vl), np.array(ref_v)))
    for name, (dl, dv) in worst.items():
        print("NN-CHECK: %s vs torchscript: max |logit diff| %.3g, max |value diff| %.3g over %d decisions"
              % (name, dl, dv, len(decs)))
    print("NN-CHECK: %s (tolerance %g abs + %g rel)" % ("ok" if ok else "FAIL", TOLERANCE, TOLERANCE))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tables", required=True, help="rmfeat --export-tables DIR")
    b.add_argument("--text-table", default=TEXT_TABLE)
    b.add_argument("--out", required=True, help="the exports' path prefix")
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--dim", type=int, default=64)
    b.add_argument("--heads", type=int, default=4)
    b.add_argument("--layers", type=int, default=2)
    b.add_argument("--policy-version", type=int, default=0)
    c = sub.add_parser("check")
    c.add_argument("--out", required=True, help="the exports' path prefix")
    c.add_argument("--tensors", required=True, help="rmfeat --featurize FILE")
    c.add_argument("--tables", required=True, help="rmfeat --export-tables DIR (its spec.json)")
    c.add_argument("--decisions", type=int, default=64)
    c.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(1)
    return build(args) if args.cmd == "build" else check(args)


if __name__ == "__main__":
    sys.exit(main())
