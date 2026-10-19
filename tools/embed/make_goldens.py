#!/usr/bin/env python3
"""Reference outputs for the oracle-text generator (plan step 5c, docs/featurizer.md §5).

NEEDS torch, transformers and safetensors (e.g. /root/torchenv/bin/python); `make` never runs it.
It produces, from the reference implementation (Hugging Face transformers), what the C++ side is
checked against:

  tokens    tests/embed/tokens_golden.jsonl: for every text entry (the engine's
            --text-table-inputs) and a set of hard strings (accents, dashes, symbols, every
            character the C++ tokenizer's Unicode tables cover), the token ids of the pinned
            model's tokenizer (tokenizer.json, fetched at the pinned commit when missing, checked
            to agree with BertTokenizerFast over vocab.txt + tokenizer_config.json). Committed; the
            gate's vocab tier checks src/featurize/wordpiece.cpp against it.
  encoder   the encoder's correctness check: builds a randomly initialised BERT from the pinned
            config.json with small size overrides (seeded; every parameter perturbed so biases and
            LayerNorms are not trivial), saved as model.safetensors (F32, or F16 with "bert."-
            prefixed names under --dtype f16) beside its config.json and the pinned tokenizer files,
            or takes the real pinned model (--model DIR); computes reference pooled, normalised
            vectors with BertModel (eval, float32, eager attention) for card texts in both pooling
            modes; runs `rmembed --embed-test` on the same texts (batch 1 and padded batches) and
            prints the minimum cosine and maximum absolute difference. Exit 1 below cosine 0.99999.
  table     compares a generated text table's rows with the reference vectors of their texts
            (the real pinned model).

    /root/torchenv/bin/python tools/embed/make_goldens.py tokens
    /root/torchenv/bin/python tools/embed/make_goldens.py encoder [--dtype f16] [--model DIR]
    /root/torchenv/bin/python tools/embed/make_goldens.py table

Options: --binary (the engine, default bin/debug/robomage), --rmembed (default bin/debug/rmembed),
--work DIR (scratch, default a temporary directory).
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import fetch_model  # noqa: E402
import rmtb  # noqa: E402

PIN = os.path.join(HERE, "MODEL_PIN")
GOLDEN = os.path.join(REPO_ROOT, "tests", "embed", "tokens_golden.jsonl")
TABLE = os.path.join(REPO_ROOT, "bin", "resources", "embeddings", "oracle_text.rmtb")
MIN_COSINE = 0.99999
# The random encoder's size (the pinned config's other fields are kept).
SMALL = {"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4, "intermediate_size": 128}

# Strings the card texts may not exercise: accents, dashes, symbols, digits, apostrophes, CJK.
HARD = [
    "Lim-Dûl the Necromancer",
    "Jötun Grunt — cumulative upkeep—pay 2 life.",
    "{T}: Add {C}{C}. {2/W}{W/P} {X}{X}",
    "+1/+1 counter, -2/-0, +X/+0 and −1: loyalty",
    "Thalia's ability; it's ‘quoted’ and “double”; don't",
    "Deal 3 damage. Draw 10 cards. 1,000 life 0.5 ½",
    "• Choose one • Mode two… (Reminder text.)",
    "Æther Vial, Dandân, Séance, Ifh-Bíff Efreet, Márton Stromgald, Ærathi Berserker",
    "naïve café résumé İstanbul ŁÓDŹ ß ĳ ŉ ſ",
    "中文卡牌 　ideographic space",
    "tab\there nbsp thin​zero-width­soft-hyphen",
    "é à combining marks ñ",
    "supercalifragilisticexpialidocious" * 4,
    "[CLS] [SEP] [UNK] ##ing CARDNAME CARDNAME's",
]


def coverage_strings():
    """Every character the C++ tokenizer's tables cover, a few per string: Latin-1 Supplement,
    Latin Extended-A, combining marks after a letter, General Punctuation, Mathematical Operators,
    the ideographic space."""
    cps = list(range(0xA0, 0x180)) + list(range(0x2000, 0x2070)) + list(range(0x2200, 0x2300)) + [0x3000]
    out = []
    for i in range(0, len(cps), 16):
        out.append("x" + " ".join("a%cb" % chr(c) for c in cps[i:i + 16]))
    out.append(" ".join("o" + chr(c) for c in range(0x300, 0x370)))
    return out


def read_pin():
    pin = fetch_model.read_pin(PIN)
    pin["prefix"] = json.loads(pin.get("prefix", '""'))
    return pin


def model_dir(pin):
    return os.path.join(REPO_ROOT, ".cache", "embed", pin["repo"].replace("/", "__"), pin["commit"])


def engine_entries(binary, work):
    """The engine's text entries (every entry: --text-table-inputs with no table)."""
    path = os.path.join(work, "inputs.jsonl")
    subprocess.run([os.path.abspath(binary), "--text-table-inputs", path, "--model-pin", PIN],
                   cwd=os.path.join(REPO_ROOT, "bin"), check=True, stdout=subprocess.DEVNULL)
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return lines[0], lines[1:]


def fetch_tokenizer_json(pin, mdir):
    path = os.path.join(mdir, "tokenizer.json")
    if not os.path.isfile(path):
        url = "%s/%s/resolve/%s/tokenizer.json" % (fetch_model.HF, pin["repo"], pin["commit"])
        with urllib.request.urlopen(url, timeout=120) as r, open(path + ".part", "wb") as out:
            out.write(r.read())
        os.replace(path + ".part", path)
    return path


def cmd_tokens(args):
    from transformers import BertTokenizerFast, PreTrainedTokenizerFast
    pin = read_pin()
    mdir = model_dir(pin)
    if not os.path.isfile(os.path.join(mdir, "vocab.txt")):
        sys.exit("make_goldens: no %s/vocab.txt: run python3 tools/embed/fetch_model.py" % mdir)
    work = args.work or tempfile.mkdtemp()
    _, entries = engine_entries(args.binary, work)
    ref = PreTrainedTokenizerFast(tokenizer_file=fetch_tokenizer_json(pin, mdir))
    slow_cfg = BertTokenizerFast.from_pretrained(mdir)
    with open(os.path.join(mdir, "config.json"), encoding="utf-8") as f:
        max_pos = json.load(f)["max_position_embeddings"]
    rows = [("%d %s" % (e["id"], e["name"]), e["text"]) for e in entries]
    rows += [("hard:%d" % i, s) for i, s in enumerate(HARD)]
    rows += [("coverage:%d" % i, s) for i, s in enumerate(coverage_strings())]
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    longest = 0
    with open(GOLDEN, "w", encoding="utf-8") as out:
        out.write(json.dumps({"model": "%s@%s" % (pin["repo"], pin["commit"]), "reference": "tokenizer.json",
                              "transformers": __import__("transformers").__version__}) + "\n")
        for name, text in rows:
            ids = ref(text)["input_ids"]
            other = slow_cfg(text)["input_ids"]
            if ids != other:
                sys.exit("make_goldens: tokenizer.json and vocab.txt + tokenizer_config.json disagree on %r" % name)
            if not name.startswith(("hard:", "coverage:")):
                longest = max(longest, len(ids))
                if len(ids) > max_pos:
                    sys.exit("make_goldens: %s is %d tokens, past %d positions" % (name, len(ids), max_pos))
            out.write(json.dumps({"name": name, "text": text, "ids": ids}, ensure_ascii=False) + "\n")
    print("make_goldens: %s: %d entries, %d hard/coverage strings; longest card text %d tokens (max %d)" % (
        os.path.relpath(GOLDEN, REPO_ROOT), len(entries), len(rows) - len(entries), longest, max_pos))


def build_random_model(pin, out_dir, dtype, seed):
    import torch
    from safetensors.torch import save_file
    from transformers import BertConfig, BertModel
    src = model_dir(pin)
    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.update(SMALL)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    for name in ("vocab.txt", "tokenizer_config.json"):
        shutil.copy(os.path.join(src, name), os.path.join(out_dir, name))
    torch.manual_seed(seed)
    model = BertModel(BertConfig(**cfg), add_pooling_layer=False)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    if dtype == "f16":
        state = {"bert." + k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
    save_file(state, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    return out_dir


def reference_vectors(mdir, texts, pooling):
    """Pooled, normalised BertModel vectors of `texts`, one at a time, float32."""
    import torch
    import torch.nn.functional as F
    from transformers import BertModel, BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained(mdir)
    model = BertModel.from_pretrained(mdir, add_pooling_layer=False, attn_implementation="eager",
                                      dtype=torch.float32).eval()
    out = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt")
            h = model(**enc).last_hidden_state
            if pooling == "cls":
                v = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(F.normalize(v, dim=-1)[0].tolist())
    return out


def compare(ref, got, label):
    """(min cosine, max abs diff) of two lists of vectors; prints them."""
    if len(ref) != len(got):
        sys.exit("make_goldens: %s: %d reference vectors, %d from rmembed" % (label, len(ref), len(got)))
    worst_cos, worst_abs = 1.0, 0.0
    for a, b in zip(ref, got):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        worst_cos = min(worst_cos, dot / (na * nb))
        worst_abs = max(worst_abs, max(abs(x - y) for x, y in zip(a, b)))
    print("make_goldens: %-34s %4d vectors  min cosine %.9f  max |diff| %.3g" % (label, len(ref), worst_cos,
                                                                                worst_abs))
    return worst_cos, worst_abs


def run_rmembed(args, mdir, texts_path, pooling, batch, work):
    out = os.path.join(work, "rmembed_%s_b%d.rmtb" % (pooling, batch))
    subprocess.run([os.path.abspath(args.rmembed), "--embed-test", "--model", mdir, "--texts", texts_path,
                    "--out", out, "--pooling", pooling, "--batch", str(batch)], check=True)
    meta, arrays = rmtb.read(out)
    assert meta["kind"] == "tensors"
    return arrays["vectors"].rows()


def cmd_encoder(args):
    pin = read_pin()
    work = args.work or tempfile.mkdtemp()
    _, entries = engine_entries(args.binary, work)
    if args.model:
        mdir, label = args.model, "pinned model"
        picked = entries if args.n <= 0 else entries[:: max(1, len(entries) // args.n)][: args.n]
    else:
        mdir = build_random_model(pin, os.path.join(work, "random_" + args.dtype), args.dtype, args.seed)
        label = "random %s" % args.dtype
        picked = entries[:: max(1, len(entries) // args.n)][: args.n]
    texts = [e["text"] for e in picked]
    texts_path = os.path.join(work, "texts.txt")
    with open(texts_path, "w", encoding="utf-8") as f:
        f.write("".join(t + "\n" for t in texts))
    ok = True
    for pooling in ("cls", "mean"):
        ref = reference_vectors(mdir, texts, pooling)
        for batch in (1, 7):
            cos, _ = compare(ref, run_rmembed(args, mdir, texts_path, pooling, batch, work),
                             "%s %s batch %d" % (label, pooling, batch))
            ok = ok and cos >= MIN_COSINE
    sys.exit(0 if ok else 1)


def cmd_table(args):
    pin = read_pin()
    work = args.work or tempfile.mkdtemp()
    _, entries = engine_entries(args.binary, work)
    meta, arrays = rmtb.read(args.table)
    rows = arrays["vectors"].rows()
    by_id = {r["id"]: r["hash"] for r in meta["rows_meta"]}
    texts = [e["text"] for e in entries]
    ref = reference_vectors(model_dir(pin), texts, pin["pooling"])
    missing = [e["name"] for e in entries if by_id.get(e["id"]) != e["hash"]]
    if missing:
        sys.exit("make_goldens: the table lacks current rows for %s" % ", ".join(missing))
    cos, _ = compare(ref, [rows[e["id"]] for e in entries], "table vs transformers")
    present = set(by_id)
    zero = all(not any(rows[i]) for i in range(len(rows)) if i not in present)
    print("make_goldens: rows outside rows_meta all zero: %s" % zero)
    sys.exit(0 if cos >= MIN_COSINE and zero else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("tokens", "encoder", "table"))
    ap.add_argument("--binary", default=os.path.join(REPO_ROOT, "bin", "debug", "robomage"))
    ap.add_argument("--rmembed", default=os.path.join(REPO_ROOT, "bin", "debug", "rmembed"))
    ap.add_argument("--work", default="")
    ap.add_argument("--model", default="", help="encoder: the real model directory instead of a random one")
    ap.add_argument("--dtype", choices=("f32", "f16"), default="f32", help="encoder: the random model's weights")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20, help="encoder: how many card texts (0: all, real model only)")
    ap.add_argument("--table", default=TABLE)
    args = ap.parse_args()
    {"tokens": cmd_tokens, "encoder": cmd_encoder, "table": cmd_table}[args.command](args)


if __name__ == "__main__":
    main()
