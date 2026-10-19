#!/usr/bin/env python3
"""Compare candidate text encoders for the oracle-text table (plan step 5c, open decision 7).

Needs torch and transformers (NOT stdlib; `make` never runs this). Embeds every vocab text the
engine would embed and reports, per model:

- mAP: for each card of a hand-made functional group (burn, cantrips, counterspells, fetches,
  duals, …), the average precision of its group-mates among its nearest neighbours, averaged;
- per-group mAP, the longest tokenized text, the width and parameter count;
- the five nearest neighbours of a dozen probe cards, to eyeball.

The texts come from the engine, so they are normalised exactly as the table's rows are:

    cd bin && ./release/robomage --text-table-inputs /tmp/entries.jsonl \\
        --model-pin ../tools/embed/MODEL_PIN
    python3 tools/embed/compare_models.py --inputs /tmp/entries.jsonl --out report.json
    python3 tools/embed/compare_models.py --inputs /tmp/entries.jsonl \\
        --model /models/bge-small-en-v1.5,cls,"" --model /models/all-MiniLM-L12-v2,mean,""

Each --model is `<local dir or Hugging Face repo>,<cls|mean>,<prefix>` (a hand-downloaded
directory needs config.json, the tokenizer files and the weights); without --model the
candidate list below is used. The prefix is put before every text (e5 wants "query: ").
"""

import argparse
import json
import sys
import time

import torch
from transformers import AutoModel, AutoTokenizer

CANDIDATES = [
    ("sentence-transformers/all-MiniLM-L6-v2", "mean", ""),
    ("sentence-transformers/all-MiniLM-L12-v2", "mean", ""),
    ("BAAI/bge-small-en-v1.5", "cls", ""),
    ("thenlper/gte-small", "mean", ""),
    ("intfloat/e5-small-v2", "mean", "query: "),
    ("Snowflake/snowflake-arctic-embed-s", "cls", ""),
    ("BAAI/bge-base-en-v1.5", "cls", ""),
    ("sentence-transformers/all-mpnet-base-v2", "mean", ""),
    ("thenlper/gte-base", "mean", ""),
]

GROUPS = {
    "burn": ["Lightning Bolt", "Chain Lightning", "Lightning Strike", "Rift Bolt", "Lava Spike", "Fireblast",
             "Skewer the Critics", "Exquisite Firecraft"],
    "cantrip": ["Ponder", "Preordain", "Brainstorm", "Consider"],
    "counter": ["Counterspell", "Daze", "Force of Will", "Force of Negation", "Spell Pierce", "Flusterstorm",
                "Mystical Dispute", "Metallic Rebuke", "Spell Snare"],
    "fetch": ["Scalding Tarn", "Flooded Strand", "Polluted Delta", "Wooded Foothills", "Misty Rainforest",
              "Windswept Heath", "Bloodstained Mire", "Verdant Catacombs", "Arid Mesa", "Marsh Flats"],
    "dual": ["Volcanic Island", "Tundra", "Underground Sea", "Savannah", "Plateau", "Bayou", "Scrubland",
             "Badlands", "Tropical Island"],
    "discard": ["Thoughtseize", "Duress", "Cabal Therapy", "Hymn to Tourach"],
    "removal": ["Swords to Plowshares", "Fatal Push", "Dismember", "Snuff Out", "Unholy Heat",
                "Sheoldred's Edict", "Prismatic Ending", "Council's Judgment"],
    "blast": ["Hydroblast", "Pyroblast", "Red Elemental Blast", "Blue Elemental Blast"],
    "dork": ["Birds of Paradise", "Noble Hierarch", "Ignoble Hierarch"],
    "gy_hate": ["Surgical Extraction", "Tormod's Crypt", "Leyline of the Void", "Faerie Macabre", "Bojuka Bog",
                "Grafdigger's Cage"],
    "ritual": ["Dark Ritual", "Cabal Ritual"],
    "fast_mana": ["Mox Opal", "Mox Amber", "Mox Diamond", "Lotus Petal"],
    "tax": ["Chalice of the Void", "Sphere of Resistance", "Trinisphere", "Thalia, Guardian of Thraben",
            "Damping Sphere"],
    "art_kill": ["Smash to Smithereens", "Meltdown", "Force of Vigor", "Abrade", "Pick Your Poison"],
    "tron": ["Urza's Mine", "Urza's Power Plant", "Urza's Tower"],
    "basic": ["Mountain", "Forest", "Island", "Plains", "Swamp", "Snow-Covered Island", "Wastes"],
}
PROBES = ["Lightning Bolt", "Brainstorm", "Counterspell", "Swords to Plowshares", "Thoughtseize",
          "Dark Ritual", "Delver of Secrets", "Murktide Regent", "Orcish Bowmasters", "Force of Will",
          "Wasteland", "Chalice of the Void"]


def read_entries(path):
    """(name, text) per engine entry, in file order; lines without a text (a header) are skipped."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict) and "text" in row:
                out.append((row["name"], row["text"]))
    return out


def parse_model(spec):
    parts = spec.split(",", 2)
    if len(parts) != 3 or parts[1] not in ("cls", "mean"):
        raise SystemExit("--model wants <dir or repo>,<cls|mean>,<prefix>: %r" % spec)
    prefix = parts[2]
    if len(prefix) >= 2 and prefix[0] == prefix[-1] == '"':
        prefix = prefix[1:-1]
    return parts[0], parts[1], prefix


@torch.no_grad()
def embed(source, pooling, prefix, texts):
    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(source).eval()
    vecs, longest = [], 0
    for i in range(0, len(texts), 32):
        batch = tok([prefix + t for t in texts[i:i + 32]], padding=True, truncation=False, return_tensors="pt")
        longest = max(longest, batch["input_ids"].shape[1])
        h = model(**batch).last_hidden_state
        if pooling == "cls":
            v = h[:, 0]
        else:
            m = batch["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
        vecs.append(torch.nn.functional.normalize(v, dim=-1))
    return torch.cat(vecs), longest, sum(p.numel() for p in model.parameters())


def average_precision(sims, i, relevant):
    hits, total = 0, 0.0
    order = [j for j in torch.argsort(sims[i], descending=True).tolist() if j != i]
    for rank, j in enumerate(order, 1):
        if j in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def evaluate(names, vectors):
    idx = {n: k for k, n in enumerate(names)}
    sims = vectors @ vectors.T
    all_ap, per_group = [], {}
    for group, members in GROUPS.items():
        ids = [idx[n] for n in members if n in idx]
        if len(ids) < 2:
            continue
        aps = [average_precision(sims, i, set(ids) - {i}) for i in ids]
        per_group[group] = round(sum(aps) / len(aps), 3)
        all_ap += aps
    neighbours = {}
    for probe in PROBES:
        if probe not in idx:
            continue
        i = idx[probe]
        order = [j for j in torch.argsort(sims[i], descending=True).tolist() if j != i][:5]
        neighbours[probe] = [[names[j], round(sims[i, j].item(), 3)] for j in order]
    return sum(all_ap) / max(1, len(all_ap)), per_group, neighbours


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", required=True, help="robomage --text-table-inputs output (every row)")
    ap.add_argument("--model", action="append", default=[], help="<dir or repo>,<cls|mean>,<prefix>")
    ap.add_argument("--out", default="compare_models.json")
    args = ap.parse_args()
    entries = read_entries(args.inputs)
    names = [n for n, _ in entries]
    texts = [t for _, t in entries]
    missing = sorted({n for g in GROUPS.values() for n in g} - set(names))
    print("entries: %d; group cards missing: %s" % (len(names), missing or "none"), flush=True)
    models = [parse_model(m) for m in args.model] if args.model else CANDIDATES
    report = {}
    for source, pooling, prefix in models:
        t0 = time.time()
        try:
            vectors, longest, n_params = embed(source, pooling, prefix, texts)
        except (OSError, ValueError, RuntimeError) as e:
            print("%s: FAILED %s" % (source, e), flush=True)
            continue
        m_ap, per_group, neighbours = evaluate(names, vectors)
        report[source] = {"pooling": pooling, "prefix": prefix, "mAP": round(m_ap, 4), "groups": per_group,
                          "dim": vectors.shape[1], "params_M": round(n_params / 1e6, 1),
                          "longest_tokens": longest, "neighbours": neighbours}
        print("%-45s mAP %.3f  dim %4d  params %5.1fM  longest %3d tokens  %3.0fs" % (
            source, m_ap, vectors.shape[1], n_params / 1e6, longest, time.time() - t0), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
