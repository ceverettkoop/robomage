#!/usr/bin/env python3
"""
Run from repo root after updating src/card_vocab.h:
    python train/gen_card_costs.py
Writes train/card_costs.py with _CARD_COST_MATRIX and _CARD_ABILITY_COST_MATRIX.
"""
import re, os, unicodedata

REPO_ROOT   = os.path.dirname(os.path.abspath(__file__)) + "/.."
VOCAB_H     = os.path.join(REPO_ROOT, "src/card_vocab.h")
MACHINE_IO_H = os.path.join(REPO_ROOT, "src/machine_io.h")
CARDS_DIR   = os.path.join(REPO_ROOT, "bin/resources/cardsfolder")
OUT_FILE    = os.path.join(REPO_ROOT, "train/card_costs.py")
N_FEATS     = 7   # W U B R G C generic

COLOR_MAP = {'W': 0, 'U': 1, 'B': 2, 'R': 3, 'G': 4, 'C': 5}

def parse_n_card_types(path):
    """Read N_CARD_TYPES (embedding vocab size) from machine_io.h."""
    m = re.search(r'N_CARD_TYPES\s*=\s*(\d+)', open(path).read())
    if not m:
        raise RuntimeError(f"could not find N_CARD_TYPES in {path}")
    return int(m.group(1))

N_TYPES = parse_n_card_types(MACHINE_IO_H)

def parse_vocab(path):
    """Return {card_name: index} from card_vocab.h."""
    text = open(path).read()
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r'"([^"]+)"\s*,\s*(\d+)', text)}

def parse_mana_cost(cost_str):
    """Parse a ManaCost field value into a 7-int list [W,U,B,R,G,C,generic]."""
    counts = [0] * N_FEATS
    if cost_str.strip() == "no cost":
        return counts
    for token in cost_str.split():
        for ch in token:
            if ch.isdigit():
                counts[6] += int(ch)
            elif ch in COLOR_MAP:
                counts[COLOR_MAP[ch]] += 1
    return counts

def find_card_file(name):
    """Find the card script file for a given card name.

    Normalization matches parse.cpp name_to_uid: lowercase, spaces→underscores,
    non-alpha/non-underscore chars removed.  For double-faced cards the file is
    named after the combined front//back name (e.g. delver_of_secrets_insectile_aberration.txt),
    so we also try a prefix match when an exact match isn't found.
    """
    stem = re.sub(r'[^a-z0-9_]', '', name.lower().replace(' ', '_').replace('-', '_'))
    # Accented names: the C++ name_to_uid strips non-ASCII bytes (an accented letter is
    # dropped, not transliterated), but Forge's filename transliterates the accent (e.g. to
    # "lorien_revealed"). Try an NFKD-decomposed ASCII stem as well so an accented card name
    # resolves to its on-disk script.
    translit = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    translit_stem = re.sub(r'[^a-z0-9_]', '', translit.lower().replace(' ', '_').replace('-', '_'))
    stems = [stem] if translit_stem == stem else [stem, translit_stem]
    for s in stems:
        filename = s + '.txt'
        first_letter = s[0]
        # 1. Exact match in expected subdirectory
        candidate = os.path.join(CARDS_DIR, first_letter, filename)
        if os.path.exists(candidate):
            return candidate
        # 2. Prefix match in expected subdirectory (catches DFC combined files)
        subdir_path = os.path.join(CARDS_DIR, first_letter)
        if os.path.isdir(subdir_path):
            for f in sorted(os.listdir(subdir_path)):
                if f.lower().startswith(s) and f.lower().endswith('.txt'):
                    return os.path.join(subdir_path, f)
    filename = stem + '.txt'
    # 3. Case-insensitive exact fallback across all subdirectories
    for subdir in sorted(os.listdir(CARDS_DIR)):
        subdir_path = os.path.join(CARDS_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for f in os.listdir(subdir_path):
            if f.lower() == filename:
                return os.path.join(subdir_path, f)
    return None

def get_mana_cost(card_name):
    """Return raw int list for a card's cast cost."""
    path = find_card_file(card_name)
    if path is None:
        print(f"  WARNING: no card file found for '{card_name}', defaulting to zero cost")
        return [0] * N_FEATS
    for line in open(path):
        if line.startswith("ManaCost:"):
            return parse_mana_cost(line[len("ManaCost:"):].strip())
    print(f"  WARNING: no ManaCost field in '{path}', defaulting to zero cost")
    return [0] * N_FEATS

def get_is_land(card_name):
    """Return True if the card's Types line includes the Land type."""
    path = find_card_file(card_name)
    if path is None:
        return False
    for line in open(path):
        if line.startswith("Types:"):
            return "Land" in line[len("Types:"):].split()
    return False

def main():
    vocab = parse_vocab(VOCAB_H)
    print(f"Found {len(vocab)} cards in vocab: {list(vocab)}")

    cast_matrix  = [[0] * N_FEATS for _ in range(N_TYPES)]
    is_land      = [False] * N_TYPES
    # _CARD_ABILITY_COST_MATRIX stays all-zeros until non-mana activated
    # abilities are parsed from card scripts (Cost$ field is not yet used by C++).

    for name, idx in vocab.items():
        cost = get_mana_cost(name)
        cast_matrix[idx] = cost
        is_land[idx] = get_is_land(name)
        print(f"  [{idx}] {name}: {cost}{' (land)' if is_land[idx] else ''}")

    # Emit card_costs.py
    lines = [
        "# AUTO-GENERATED by train/gen_card_costs.py — do not edit manually.",
        "# Re-run after updating src/card_vocab.h or card script files.",
        "import numpy as np",
        "",
        f"N_CARD_TYPES = {N_TYPES}",
        "_N_COST_FEATS = 7  # W, U, B, R, G, C, generic (pip counts / 10.0)",
        "",
        "_CARD_COST_MATRIX = np.array([",
    ]
    for idx, row in enumerate(cast_matrix):
        name_comment = next((n for n, i in vocab.items() if i == idx), "")
        normalized = [f"{v/10.0:.1f}" for v in row]
        lines.append(f"    [{', '.join(normalized)}],  # {idx}: {name_comment}")
    lines += [
        "], dtype=np.float32)",
        "",
        "# Populated manually when non-mana activated abilities are added.",
        "_CARD_ABILITY_COST_MATRIX = np.zeros((N_CARD_TYPES, _N_COST_FEATS), dtype=np.float32)",
        "",
        "# Index → card name, mirrors card_vocab.h",
        "_VOCAB_NAMES = [",
    ]
    vocab_by_idx = {i: n for n, i in vocab.items()}
    for idx in range(N_TYPES):
        name = vocab_by_idx.get(idx, "")
        lines.append(f'    "{name}",  # {idx}')
    lines += ["]", ""]

    # Vocab indices whose card is a Land (used by the scripted agent's mulligan).
    land_ids = sorted(i for i in range(N_TYPES) if is_land[i])
    lines += [
        "# Vocab indices that are Land cards (Types line includes 'Land').",
        f"_LAND_VOCAB_IDS = frozenset({{{', '.join(str(i) for i in land_ids)}}})",
    ]
    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_FILE}")

if __name__ == "__main__":
    main()
