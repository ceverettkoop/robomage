#!/usr/bin/env python3
"""
Run from repo root after updating src/card_vocab.h:
    python train/gen_card_props.py
Writes train/card_props.py with _CARD_PROP_MATRIX — the FROZEN per-card printed-
property block consumed by the network's card representation (concatenated with
the trainable identity embedding inside CardGameExtractor._embed_ids).

The column layout is a HARD-CODED constant here — never derived from the current
card pool — so adding vocab cards can never change the matrix width (which would
be a network-shape break). Unknown keywords/ability-categories/subtypes in the
scripts are simply not encoded; a summary of what was ignored is printed so
column-set drift is a visible, deliberate decision rather than a silent one.

Values are stored normalized like card_costs.py: pip counts / 10, cmc / 10,
P/T / 10, and binary flags. No C++ mirror header is emitted — unlike the cost
matrices this block is consumed inside the Python network only, never
serialized into the observation.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_card_costs import (REPO_ROOT, VOCAB_H, CARDS_DIR, N_TYPES,
                            parse_vocab, parse_token_vocab_base,
                            parse_mana_cost, find_card_file)
from _enums import _OBS_KEYWORDS

TOKENS_DIR = os.path.join(REPO_ROOT, "bin/resources/tokenscripts")
OUT_FILE = os.path.join(REPO_ROOT, "train/card_props.py")

# ---------------------------------------------------------------------------
# Fixed column layout (P = 96). Order is load-bearing: consumers address slices
# by these names via card_props._PROP_NAMES. Widening past the reserved columns
# is a deliberate network-shape break (every checkpoint invalidated).
# ---------------------------------------------------------------------------

CARD_TYPES = ["Creature", "Instant", "Sorcery", "Land", "Artifact",
              "Enchantment", "Planeswalker", "Battle"]
LAND_SUBTYPES = ["Plains", "Island", "Swamp", "Mountain", "Forest", "Urza's"]
# The 16 obs keywords (imported so these columns stay in sync with the
# effective-keyword multi-hot in src/machine_io.h) + 8 printed mechanics.
EXTRA_KEYWORDS = ["Equip", "Flashback", "Cycling", "Delve",
                  "Storm", "Evoke", "Affinity", "Prowess"]
KEYWORDS = list(_OBS_KEYWORDS) + EXTRA_KEYWORDS
# SP$/AB$/DB$ ability-category heads, fixed by vocab frequency.
ABILITY_CATS = ["ChangeZone", "Mana", "Draw", "DealDamage", "Destroy",
                "Counter", "Token", "PutCounter", "Tap", "Pump",
                "Charm", "Surveil", "Sacrifice", "Dig", "GainLife",
                "LoseLife", "Discard", "Animate", "Mill", "Untap",
                "Attach", "DelayedTrigger", "NameCard", "WinsGame"]
# Tribal + strategically notable non-creature subtypes.
SUBTYPES = ["Human", "Eldrazi", "Wizard", "Elemental", "Goblin", "Warrior",
            "Cat", "Soldier", "Spirit", "Equipment", "Aura", "Saga"]
N_RESERVED = 3

COLOR_WORDS = {"white": 0, "blue": 1, "black": 2, "red": 3, "green": 4}

PROP_NAMES = (
    ["pip_w", "pip_u", "pip_b", "pip_r", "pip_g", "pip_c", "pip_generic",
     "cmc", "x_cost",
     "color_w", "color_u", "color_b", "color_r", "color_g"]
    + ["type_" + t.lower() for t in CARD_TYPES]
    + ["legendary", "basic"]
    + ["land_" + re.sub(r"[^a-z]", "", s.lower()) for s in LAND_SUBTYPES]
    + ["power", "toughness", "has_pt"]
    + ["kw_" + re.sub(r"[^a-z]", "_", k.lower()).strip("_") for k in KEYWORDS]
    + ["cat_" + c.lower() for c in ABILITY_CATS]
    + ["sub_" + s.lower() for s in SUBTYPES]
    + [f"reserved_{i}" for i in range(N_RESERVED)]
)
N_PROPS = len(PROP_NAMES)
COL = {n: i for i, n in enumerate(PROP_NAMES)}

DFC_SEPARATOR = "ALTERNATE"

# Ignored-but-seen script attributes, reported once at the end (not per-card:
# the scripts legitimately carry far more keywords/categories/subtypes than the
# fixed columns encode, so per-occurrence warnings would drown real problems).
_ignored_keywords = set()
_ignored_cats = set()
_ignored_subtypes = set()


def parse_vocab_with_stems(path):
    """Return {index: script_stem} for the token band's 3-tuple entries
    {"script_stem", "Display Name", N}. The shared parse_vocab deliberately
    captures only the display name; the stem is what resolves the on-disk
    token script under bin/resources/tokenscripts/."""
    text = open(path).read()
    return {int(m.group(3)): m.group(1)
            for m in re.finditer(r'\{\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d+)', text)}


def find_back_face_file(name):
    """Resolve a DFC BACK-face vocab entry to its combined <front>_<back>.txt.

    find_card_file matches by front stem (exact, then prefix), so a back-face
    name like "Insectile Aberration" misses entirely; the combined filename
    ends with the back face's uid instead."""
    stem = re.sub(r'[^a-z0-9_]', '',
                  name.lower().replace(' ', '_').replace('-', '_').replace('/', '_'))
    stem = re.sub(r'_+', '_', stem)
    suffix = "_" + stem + ".txt"
    for subdir in sorted(os.listdir(CARDS_DIR)):
        subdir_path = os.path.join(CARDS_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for f in sorted(os.listdir(subdir_path)):
            if f.lower().endswith(suffix):
                return os.path.join(subdir_path, f)
    return None


def split_faces(text):
    """Split a script into (front_lines, back_lines) on the ALTERNATE
    separator line. A single-faced script has back_lines == None."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == DFC_SEPARATOR:
            return lines[:i], lines[i + 1:]
    return lines, None


def face_lines_for(name):
    """The script lines for the vocab entry's own face: the front section of
    the resolved script, or the back section when the name only resolves as a
    combined-file suffix (a DFC back-face vocab entry). None if no script."""
    path = find_card_file(name)
    if path is not None:
        front, _back = split_faces(open(path).read())
        return front
    path = find_back_face_file(name)
    if path is not None:
        _front, back = split_faces(open(path).read())
        if back is not None:
            return back
        print(f"  WARNING: '{name}' matched combined file '{path}' "
              f"but it has no {DFC_SEPARATOR} section")
    return None


def field(lines, key):
    """First 'Key:value' line's value, or None."""
    prefix = key + ":"
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def parse_face(lines, row):
    """Fill a property row from one face's script lines."""
    # Cost, CMC, X, colors-from-pips
    cost_str = field(lines, "ManaCost") or "no cost"
    pips = parse_mana_cost(cost_str)
    for i in range(7):
        row[i] = pips[i] / 10.0
    row[COL["cmc"]] = (sum(pips[:6]) + pips[6]) / 10.0
    if any("X" in tok for tok in cost_str.split()):
        row[COL["x_cost"]] = 1.0
    for ci in range(5):
        if pips[ci] > 0:
            row[COL["color_w"] + ci] = 1.0
    # Colors: line (tokens and DFC back faces have "no cost" ManaCost)
    colors = field(lines, "Colors")
    if colors:
        for word in colors.replace(",", " ").split():
            ci = COLOR_WORDS.get(word.lower())
            if ci is not None:
                row[COL["color_w"] + ci] = 1.0
    # Types line: supertypes, card types, subtypes
    types_line = field(lines, "Types")
    if types_line:
        tokens = types_line.split()
        for t in CARD_TYPES:
            if t in tokens:
                row[COL["type_" + t.lower()]] = 1.0
        if "Legendary" in tokens:
            row[COL["legendary"]] = 1.0
        if "Basic" in tokens:
            row[COL["basic"]] = 1.0
        for s in LAND_SUBTYPES:
            if s in tokens:
                row[COL["land_" + re.sub(r"[^a-z]", "", s.lower())]] = 1.0
        known = set(CARD_TYPES) | {"Legendary", "Basic", "Snow", "World",
                                   "Kindred", "Tribal"} | set(LAND_SUBTYPES)
        for t in tokens:
            if t in SUBTYPES:
                row[COL["sub_" + t.lower()]] = 1.0
            elif t not in known:
                _ignored_subtypes.add(t)
    # P/T ('*' and friends parse as 0 but still set has_pt)
    pt = field(lines, "PT")
    if pt and "/" in pt:
        p_str, t_str = pt.split("/", 1)
        try:
            row[COL["power"]] = int(p_str) / 10.0
        except ValueError:
            pass
        try:
            row[COL["toughness"]] = int(t_str) / 10.0
        except ValueError:
            pass
        row[COL["has_pt"]] = 1.0
    # Printed keywords: head token before the first ':' of the K-value
    kw_lookup = {k.lower(): "kw_" + re.sub(r"[^a-z]", "_", k.lower()).strip("_")
                 for k in KEYWORDS}
    for line in lines:
        if line.startswith("K:"):
            head = line[2:].split(":")[0].split("|")[0].strip()
            col = kw_lookup.get(head.lower())
            if col is not None:
                row[COL[col]] = 1.0
            else:
                _ignored_keywords.add(head)
    # Ability-category heads across A:/T:/S:/SVar: lines of this face
    for m in re.finditer(r"(?:SP|AB|DB)\$\s*([A-Za-z]+)", "\n".join(lines)):
        cat = m.group(1)
        if cat in ABILITY_CATS:
            row[COL["cat_" + cat.lower()]] = 1.0
        else:
            _ignored_cats.add(cat)


def main():
    vocab = parse_vocab(VOCAB_H)
    token_base = parse_token_vocab_base(VOCAB_H)
    token_stems = parse_vocab_with_stems(VOCAB_H)
    print(f"Building {N_PROPS}-column property matrix "
          f"for {len(vocab)} vocab cards")

    matrix = [[0.0] * N_PROPS for _ in range(N_TYPES)]
    for name, idx in vocab.items():
        if idx >= token_base:
            stem = token_stems.get(idx)
            path = os.path.join(TOKENS_DIR, stem + ".txt") if stem else None
            if path is None or not os.path.exists(path):
                print(f"  WARNING: no token script for '{name}' "
                      f"(stem {stem!r}), zero property row")
                continue
            lines, _ = split_faces(open(path).read())
        else:
            lines = face_lines_for(name)
            if lines is None:
                print(f"  WARNING: no card file found for '{name}', "
                      f"zero property row")
                continue
        parse_face(lines, matrix[idx])

    for label, ignored in (("keywords", _ignored_keywords),
                           ("ability categories", _ignored_cats),
                           ("subtypes", _ignored_subtypes)):
        if ignored:
            print(f"  not encoded ({label}): {', '.join(sorted(ignored))}")

    vocab_by_idx = {i: n for n, i in vocab.items()}
    lines = [
        "# AUTO-GENERATED by train/gen_card_props.py — do not edit manually.",
        "# Re-run after updating src/card_vocab.h or card script files.",
        "# Frozen printed-property block of the network's card representation;",
        "# concatenated with the trainable identity embedding in _embed_ids.",
        "import numpy as np",
        "",
        f"N_CARD_TYPES = {N_TYPES}",
        "",
        "# Column names (pips/cmc/pt are /10.0; the rest are binary flags).",
        "_PROP_NAMES = [",
    ]
    for name in PROP_NAMES:
        lines.append(f'    "{name}",')
    lines += [
        "]",
        "N_CARD_PROPS = len(_PROP_NAMES)",
        f"assert N_CARD_PROPS == {N_PROPS}",
        "",
        "_CARD_PROP_MATRIX = np.array([",
    ]
    for idx, row in enumerate(matrix):
        vals = ", ".join(f"{v:.1f}" for v in row)
        lines.append(f"    [{vals}],  # {idx}: {vocab_by_idx.get(idx, '')}")
    lines += ["], dtype=np.float32)", ""]
    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
