#!/usr/bin/env python3
"""
Run from repo root after updating src/card_vocab.h:
    python train/gen_card_costs.py
Writes train/card_costs.py with _CARD_COST_MATRIX and _CARD_ABILITY_COST_MATRIX.

Two matrices, both [N_CARD_TYPES][7] with the SAME column layout
(W, U, B, R, G, C pip counts then generic, each divided by 10):

* _CARD_COST_MATRIX      — the card's printed ManaCost (what it costs to CAST).
                           Gathered per hand slot by env.py.
* _CARD_ABILITY_COST_MATRIX — the MANA part of the cost of ACTIVATING one of the
                           card's abilities while it is on the battlefield.
                           Gathered per self-battlefield permanent slot by env.py.

Ability-cost selection rule (see get_ability_cost):
  candidates = the card face's `A:AB$ <Category>` activated abilities, plus the
  battlefield-activated keyword abilities in KEYWORD_ABILITY_COST (Equip,
  Reconfigure, Monstrosity, ...), MINUS
    - mana abilities (`AB$ Mana` / `AddMana` / `ManaReflected`): "{T}: add {G}"
      is not an activation cost worth encoding, and every land would otherwise
      collapse to the same all-zero row anyway;
    - abilities that are not activatable from the battlefield (an explicit
      `ActivationZone$` that does not list Battlefield — channel/cycling-style
      abilities activated from hand or graveyard); this block describes
      permanents already in play;
    - candidates whose parsed MANA component is zero (tap-only, sac-only,
      pay-life-only, loyalty).
  Of the remaining candidates the CHEAPEST by total converted mana wins; ties go
  to the first-listed ability.

Accepted ambiguities (documented, not bugs):
  * A card with only free activations ("{T}: ..." like Wasteland or Maze of Ith)
    gets an all-zero row, indistinguishable from a card with no activated
    ability at all. The trainable card-identity embedding disambiguates.
  * Only the cheapest qualifying ability is encoded; a card's other (pricier)
    abilities are invisible in this block.
  * `X` in an activation cost contributes 0, exactly as `X` in a ManaCost does
    for the cast matrix (there is no X column in this 7-feature layout). So an
    ability whose only mana is X (Blast Zone's `X X T`, Pernicious Deed's
    `X Sac<...>`) is NOT a candidate, and a `X R` cost encodes as just {R}.
  * Non-mana cost components (T, Q, PayLife<N>, Sac<...>, Discard<...>,
    Return<...>, AddCounter/SubCounter<N/LOYALTY>, ...) are dropped entirely —
    this block is about mana only.
"""
import re, os, unicodedata

REPO_ROOT   = os.path.dirname(os.path.abspath(__file__)) + "/.."
VOCAB_H     = os.path.join(REPO_ROOT, "src/card_vocab.h")
MACHINE_IO_H = os.path.join(REPO_ROOT, "src/machine_io.h")
CARDS_DIR   = os.path.join(REPO_ROOT, "bin/resources/cardsfolder")
TOKENS_DIR  = os.path.join(REPO_ROOT, "bin/resources/tokenscripts")
OUT_FILE    = os.path.join(REPO_ROOT, "train/card_costs.py")
OUT_HEADER  = os.path.join(REPO_ROOT, "src/gen/card_costs_gen.h")
N_FEATS     = 7   # W U B R G C generic

COLOR_MAP = {'W': 0, 'U': 1, 'B': 2, 'R': 3, 'G': 4, 'C': 5}

DFC_SEPARATOR = "ALTERNATE"

# Ability categories that ARE mana abilities — excluded from the ability-cost
# matrix (parse.cpp normalizes the script's `Mana` to `AddMana`).
MANA_ABILITY_CATS = {"Mana", "AddMana", "ManaReflected"}

# Keyword-granted activated abilities that are activated from the BATTLEFIELD,
# mapped to the 0-based field of the `K:` line holding their mana cost:
#   K:Equip:2                 -> field 1 = "2"
#   K:Equip:1 R               -> field 1 = "1 R"
#   K:Equip:1:Creature.Knight+YouCtrl:Knight -> field 1 (rest is the target spec)
#   K:Monstrosity:4:5 B G     -> field 2 ("4" is the counter count)
# Keyword abilities activated from OTHER zones (Cycling/TypeCycling from hand,
# Unearth/Escape/Flashback from the graveyard, Ninjutsu from hand, ...) are
# deliberately absent: this matrix indexes battlefield permanents.
KEYWORD_ABILITY_COST = {
    "equip": 1,
    "reconfigure": 1,
    "fortify": 1,
    "outlast": 1,
    "level up": 1,
    "monstrosity": 2,
}

# Whitespace tokens of a Cost$ value that carry mana: pips, generic numbers,
# hybrid/phyrexian pips (W/U, 2/W, P/R) and the X-family placeholders (which
# parse to zero, like X in a ManaCost). Anything else (T, Q, and the
# Keyword<...> components, whose <> groups are stripped first) is not mana.
_MANA_TOKEN_RE = re.compile(r'^[0-9WUBRGCXYZSP/]+$')
# `Keyword<...>` cost components; the <> group may contain spaces
# (Sac<1/CARDNAME/this creature>), so strip whole groups before tokenizing.
_COST_GROUP_RE = re.compile(r'[A-Za-z]\w*<[^>]*>')

# Cost$ tokens that were neither recognized as mana nor as a known non-mana
# component; reported once at the end of a run rather than per occurrence.
_unknown_cost_tokens = set()
_KNOWN_NON_MANA_TOKENS = {"T", "Q"}

def parse_n_card_types(path):
    """Read N_CARD_TYPES (embedding vocab size) from machine_io.h."""
    m = re.search(r'N_CARD_TYPES\s*=\s*(\d+)', open(path).read())
    if not m:
        raise RuntimeError(f"could not find N_CARD_TYPES in {path}")
    return int(m.group(1))

N_TYPES = parse_n_card_types(MACHINE_IO_H)

def parse_vocab(path):
    """Return {card_name: index} from card_vocab.h.

    Matches both real card entries {"Name", N} and the token band's
    {"script_stem", "Display Name", N} (only the display name precedes the
    index, so the stem is never captured).
    """
    text = open(path).read()
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r'"([^"]+)"\s*,\s*(\d+)', text)}

def parse_token_vocab_base(path):
    """Read TOKEN_VOCAB_BASE (first index of the token identity band) from
    card_vocab.h. Token band entries are script-less by design (tokens have no
    cast cost), so indices >= this get silent zero-cost rows instead of the
    missing-script warning; N_TYPES (no band) if the constant is absent."""
    m = re.search(r'TOKEN_VOCAB_BASE\s*=\s*(\d+)', open(path).read())
    return int(m.group(1)) if m else N_TYPES

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

    Normalization matches parse.cpp name_to_uid: lowercase, space/hyphen/slash→underscore,
    non-alpha/non-underscore chars removed.  For double-faced cards the file is
    named after the combined front//back name (e.g. delver_of_secrets_insectile_aberration.txt),
    so we also try a prefix match when an exact match isn't found.

    '/' is a separator, not punctuation (CR 709 split cards), so a combined "Front/Back"
    name resolves EXACTLY to Forge's underscore-joined script (e.g. "Dead/Gone" ->
    dead_gone.txt) instead of falling through to the unverified prefix match below,
    which would otherwise pick whichever "dead*.txt" happens to sort first on this
    machine — making the generated costs depend on which scripts are provisioned.
    """
    stem = re.sub(r'[^a-z0-9_]', '',
                  name.lower().replace(' ', '_').replace('-', '_').replace('/', '_'))
    # Collapse consecutive underscores to one, matching parse.cpp name_to_uid — a name whose
    # "& "/punctuation excision leaves a doubled "__" (e.g. "Raph & Mikey, Troublemakers")
    # must resolve to Forge's single-underscore filename (raph_mikey_troublemakers.txt).
    stem = re.sub(r'_+', '_', stem)
    # Accented names: the C++ name_to_uid strips non-ASCII bytes (an accented letter is
    # dropped, not transliterated), but Forge's filename transliterates the accent (e.g. to
    # "lorien_revealed"). Try an NFKD-decomposed ASCII stem as well so an accented card name
    # resolves to its on-disk script.
    translit = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    translit_stem = re.sub(r'[^a-z0-9_]', '',
                           translit.lower().replace(' ', '_').replace('-', '_').replace('/', '_'))
    translit_stem = re.sub(r'_+', '_', translit_stem)
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
    """The script lines for the vocab entry's own face, plus DFC context.

    Returns ``(lines, front_lines, is_back)``: the front section of the
    resolved script, or the back section (with the front carried alongside)
    when the name only resolves as a combined-file suffix — a DFC back-face
    vocab entry. ``(None, None, False)`` if no script."""
    path = find_card_file(name)
    if path is not None:
        front, _back = split_faces(open(path).read())
        return front, front, False
    path = find_back_face_file(name)
    if path is not None:
        front, back = split_faces(open(path).read())
        if back is not None:
            return back, front, True
        print(f"  WARNING: '{name}' matched combined file '{path}' "
              f"but it has no {DFC_SEPARATOR} section")
    return None, None, False


def parse_ability_cost(cost_str):
    """Parse the MANA component of an ability Cost$ value into [W,U,B,R,G,C,generic].

    Non-mana components are dropped: the `Keyword<...>` groups (Sac, PayLife,
    Discard, Return, AddCounter/SubCounter, PayEnergy, Exert, ...) are stripped
    whole — their <> body may contain spaces — and the remaining tokens are kept
    only when they look like mana. Recognized mana tokens are handed to the same
    parse_mana_cost used for printed cast costs, so pips/generic/hybrid are
    counted identically in both matrices (and X contributes nothing)."""
    counts = [0] * N_FEATS
    for token in _COST_GROUP_RE.sub(' ', cost_str).split():
        if _MANA_TOKEN_RE.match(token):
            pips = parse_mana_cost(token)
            for i in range(N_FEATS):
                counts[i] += pips[i]
        elif token not in _KNOWN_NON_MANA_TOKENS:
            _unknown_cost_tokens.add(token)
    return counts


def ability_cost_candidates(lines):
    """Every battlefield-activatable, non-mana activated ability of one face, as
    (label, mana_cost_list) pairs in script order.

    Covers `A:AB$ <Category>` lines and the keyword-granted activated abilities
    in KEYWORD_ABILITY_COST. See the module docstring for what is excluded."""
    out = []
    for line in lines:
        line = line.strip()
        if line.startswith("A:AB$"):
            params = [p.strip() for p in line[len("A:"):].split("|")]
            category = params[0][len("AB$"):].strip()
            if category in MANA_ABILITY_CATS:
                continue
            zone = next((p[len("ActivationZone$"):].strip() for p in params
                         if p.startswith("ActivationZone$")), None)
            if zone is not None and "Battlefield" not in zone:
                continue  # channel/cycling-style: activated from hand/graveyard
            cost = next((p[len("Cost$"):].strip() for p in params
                         if p.startswith("Cost$")), "")
            out.append((category, parse_ability_cost(cost)))
        elif line.startswith("K:"):
            fields = line[len("K:"):].split(":")
            field_idx = KEYWORD_ABILITY_COST.get(fields[0].strip().lower())
            if field_idx is not None and len(fields) > field_idx:
                out.append((fields[0].strip(),
                            parse_ability_cost(fields[field_idx])))
    return out


def get_ability_cost(lines):
    """Return (mana_cost_list, label) for the cheapest qualifying activated
    ability of one face, or ([0]*N_FEATS, None) when the face has none.

    Candidates with a zero mana component (tap-only, sac-only, loyalty, X-only)
    are not selectable — they carry no information this block can hold; ties on
    total converted mana go to the first-listed ability."""
    best, best_label, best_total = [0] * N_FEATS, None, None
    for label, cost in ability_cost_candidates(lines):
        total = sum(cost)
        if total == 0:
            continue
        if best_total is None or total < best_total:
            best, best_label, best_total = cost, label, total
    return best, best_label


def main():
    vocab = parse_vocab(VOCAB_H)
    token_base = parse_token_vocab_base(VOCAB_H)
    print(f"Found {len(vocab)} cards in vocab: {list(vocab)}")

    cast_matrix  = [[0] * N_FEATS for _ in range(N_TYPES)]
    ability_matrix = [[0] * N_FEATS for _ in range(N_TYPES)]
    is_land      = [False] * N_TYPES
    token_stems  = parse_vocab_with_stems(VOCAB_H)

    n_tokens = 0
    for name, idx in vocab.items():
        if idx >= token_base:
            # Token identity band: no CARD script exists (and a name like "Cat"
            # would prefix-match an unrelated card file). Tokens have no cast
            # cost — the cast row and land status stay zero/false — but a token
            # permanent CAN have an activated ability ("{2}, Sacrifice this
            # token: Draw a card"), so its ability row is parsed from the token
            # script under bin/resources/tokenscripts/.
            n_tokens += 1
            stem = token_stems.get(idx)
            path = os.path.join(TOKENS_DIR, stem + ".txt") if stem else None
            if path is None or not os.path.exists(path):
                print(f"  WARNING: no token script for '{name}' (stem {stem!r}), "
                      f"zero ability-cost row")
                continue
            lines, _back = split_faces(open(path).read())
            acost, alabel = get_ability_cost(lines)
            ability_matrix[idx] = acost
            if alabel:
                print(f"  [{idx}] {name} (token): ability {alabel} {acost}")
            continue
        cost = get_mana_cost(name)
        cast_matrix[idx] = cost
        is_land[idx] = get_is_land(name)
        # Activated-ability cost: parsed from THIS vocab entry's own face, so a
        # DFC back-face entry describes the back face's abilities (mirroring
        # gen_card_props.py) rather than the front's.
        lines, _front, _is_back = face_lines_for(name)
        alabel = None
        if lines is not None:
            ability_matrix[idx], alabel = get_ability_cost(lines)
        print(f"  [{idx}] {name}: {cost}{' (land)' if is_land[idx] else ''}"
              f"{f'  ability {alabel} {ability_matrix[idx]}' if alabel else ''}")
    if n_tokens:
        print(f"  [{token_base}+] token band: {n_tokens} zero cast-cost rows")
    n_ability = sum(1 for row in ability_matrix if any(row))
    print(f"  activated-ability mana costs: {n_ability} nonzero rows")
    if _unknown_cost_tokens:
        print("  not recognized as mana (Cost$ tokens, ignored): "
              + ", ".join(sorted(_unknown_cost_tokens)))

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
        "# Mana cost of ACTIVATING one of the card's abilities from the battlefield —",
        "# the cheapest qualifying activated ability, same 7-column layout as above.",
        "# Excluded from selection: mana abilities ({T}: add {G}), abilities activated",
        "# from another zone (channel/cycling), and candidates with no mana in their",
        "# cost. So an all-zero row means EITHER 'no activated ability' OR 'its",
        "# activations are mana-free' (tap-only/sac-only/loyalty/X-only) — an accepted",
        "# ambiguity the trainable card-identity embedding disambiguates. Only the",
        "# cheapest ability is encoded; X contributes 0, as it does in the cast matrix.",
        "# See train/gen_card_costs.py's module docstring for the full rule.",
        "_CARD_ABILITY_COST_MATRIX = np.array([",
    ]
    for idx, row in enumerate(ability_matrix):
        name_comment = next((n for n, i in vocab.items() if i == idx), "")
        normalized = [f"{v/10.0:.1f}" for v in row]
        lines.append(f"    [{', '.join(normalized)}],  # {idx}: {name_comment}")
    lines += [
        "], dtype=np.float32)",
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

    # Emit src/gen/card_costs_gen.h — the C++ mirror of the two cost matrices, so
    # a future C++ actor reproduces the obs cost blocks bit-exactly. The float
    # literals are derived from the SAME `v/10.0` source expression and the SAME
    # ".1f" formatting used for card_costs.py above (identity by construction);
    # C++ float32 conversion of "0.1f" equals np.float32(0.1) for these n/10 values.
    emit_cpp_header(cast_matrix, ability_matrix, vocab_by_idx)


def emit_cpp_header(cast_matrix, ability_matrix, vocab_by_idx):
    """Write src/gen/card_costs_gen.h with CARD_COST_MATRIX (from cast_matrix)
    and CARD_ABILITY_COST_MATRIX (from ability_matrix), mirroring card_costs.py
    exactly."""
    def row_literal(row):
        return "{" + ", ".join(f"{v/10.0:.1f}f" for v in row) + "}"

    hdr = [
        "// AUTO-GENERATED by train/gen_card_costs.py — do not edit manually.",
        "// Re-run after updating src/card_vocab.h or card script files.",
        "// Inputs: src/card_vocab.h, src/machine_io.h, bin/resources/cardsfolder/*",
        "// C++ mirror of train/card_costs.py's _CARD_COST_MATRIX /",
        "// _CARD_ABILITY_COST_MATRIX (identical np.float32 values by construction).",
        "#pragma once",
        "",
        f"static const int N_COST_FEATS = {N_FEATS};",
        "",
        f"static const float CARD_COST_MATRIX[{N_TYPES}][{N_FEATS}] = {{",
    ]
    for idx, row in enumerate(cast_matrix):
        name = vocab_by_idx.get(idx, "")
        hdr.append(f"    {row_literal(row)},  // {idx}: {name}")
    hdr += [
        "};",
        "",
        "// Mana cost of activating the card's cheapest qualifying battlefield",
        "// ability, mirroring card_costs.py's _CARD_ABILITY_COST_MATRIX.",
        f"static const float CARD_ABILITY_COST_MATRIX[{N_TYPES}][{N_FEATS}] = {{",
    ]
    for idx, row in enumerate(ability_matrix):
        name = vocab_by_idx.get(idx, "")
        hdr.append(f"    {row_literal(row)},  // {idx}: {name}")
    hdr += ["};", ""]

    os.makedirs(os.path.dirname(OUT_HEADER), exist_ok=True)
    with open(OUT_HEADER, "w") as f:
        f.write("\n".join(hdr) + "\n")
    print(f"Wrote {OUT_HEADER}")

if __name__ == "__main__":
    main()
