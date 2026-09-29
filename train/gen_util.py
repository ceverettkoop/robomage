"""Shared helpers for the codegen scripts (gen_enums / gen_card_costs /
gen_card_props / gen_archetypes / gen_sb_rules): write-if-changed output and
the card-script face resolver.

Stdlib-only, like the generators themselves."""
import functools
import os
import re
import unicodedata
from typing import List, NamedTuple, Optional


def write_if_changed(path, content):
    """Write `content` to `path` only when it differs from the file already there.

    Returns True if the file was created or its content changed, False if it was
    already up to date. Leaving an unchanged file untouched preserves its mtime,
    which matters because `make` regenerates the codegen on EVERY build: the C++
    mirror headers (src/gen/*.h) are #included by the engine, so rewriting them
    with identical bytes would bump their mtime and trigger a needless recompile
    cascade. write-if-changed keeps a no-op regeneration a true no-op."""
    try:
        with open(path, "r") as f:
            if f.read() == content:
                return False
    except FileNotFoundError:
        pass
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return True


# ---------------------------------------------------------------------------
# Card-script resolution — THE one place the generators (gen_card_costs,
# gen_card_props, and gen_sb_rules through gen_card_props) and decode.py map a
# vocab card name to the Forge script face that describes it.
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CARDS_DIR = os.path.join(REPO_ROOT, "bin", "resources", "cardsfolder")

# Line separating the two faces (DFC front/back, split-card halves) of one script.
FACE_SEPARATOR = "ALTERNATE"


class CardFace(NamedTuple):
    """One face of a resolved card script.

    `lines` are this face's own script lines; `front_lines` the front face's
    (the same list for a front face). `layout` is the front's `AlternateMode:`
    (DoubleFaced = transforming DFC, Modal = MDFC, Split), None for a
    single-faced script."""
    path: str
    lines: List[str]
    front_lines: List[str]
    is_back: bool
    layout: Optional[str]


def card_uid_stems(name):
    """Script-filename stems for a card name, mirroring src/parse.cpp name_to_uid:
    lowercase, space/hyphen/slash -> '_' ('/' is the split-card separator, CR 709,
    so "Dead/Gone" -> "dead_gone"), other punctuation dropped, '__' collapsed.

    name_to_uid drops a non-ASCII byte, but Forge's filename transliterates the
    accent (an accented "o" becomes "o"), so an NFKD-decomposed ASCII stem is
    also returned when it differs — the same accent folding the engine applies
    to names via ascii_fold_card_name (src/card_vocab.h)."""
    stems = []
    for variant in (name, unicodedata.normalize('NFKD', name)
                    .encode('ascii', 'ignore').decode('ascii')):
        stem = re.sub(r'[^a-z0-9_]', '', variant.lower().replace(' ', '_')
                      .replace('-', '_').replace('/', '_'))
        stem = re.sub(r'_+', '_', stem)
        if stem and stem not in stems:
            stems.append(stem)
    return stems


def script_field(lines, key):
    """Value of the first `Key:value` line in `lines`, stripped, or None."""
    prefix = key + ":"
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def split_faces(text):
    """Split a script into (front_lines, back_lines) on the FACE_SEPARATOR
    line. A single-faced script has back_lines == None."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == FACE_SEPARATOR:
            return lines[:i], lines[i + 1:]
    return lines, None


def resolve_card_face(name):
    """The CardFace a vocab card name denotes, or None if no script has it.

    Mirrors load_card (src/card_db.cpp), which loads the exact `<uid>.txt`, else
    the combined `<uid>_<back>.txt` of a double-faced card, and aliases the
    back face's name to that same script. In order, per stem of the name:
      1. the exact `<letter>/<stem>.txt` — its front face (a whole split-card
         name such as "Dead/Gone" lands here too);
      2. a combined `<letter>/<stem>_*.txt` whose FRONT face is named `name`;
      3. a combined `*/*_<stem>.txt` whose BACK face is named `name` (a DFC
         back face or the second half of a split card).
    Steps 2 and 3 match on the face boundary: the filename must split at an
    underscore exactly at the stem, and the face's own `Name:` must fold to the
    same stem, so "Dead" never resolves to an unrelated "dead_weight.txt"."""
    stems = card_uid_stems(name)
    for stem in stems:
        letter = stem[0]
        exact = os.path.join(CARDS_DIR, letter, stem + ".txt")
        if os.path.exists(exact):
            return _script_face(exact, back=False)
        for fname in _letter_scripts(letter):
            if fname.startswith(stem + "_"):
                face = _script_face(os.path.join(CARDS_DIR, letter, fname), back=False)
                if _face_named(face, stem):
                    return face
    for stem in stems:
        suffix = "_" + stem + ".txt"
        for letter in _letter_dirs():
            for fname in _letter_scripts(letter):
                if fname.endswith(suffix):
                    face = _script_face(os.path.join(CARDS_DIR, letter, fname), back=True)
                    if face is not None and _face_named(face, stem):
                        return face
    return None


def mana_value_lines(face):
    """Script lines whose `ManaCost:` gives this face's mana value.

    A transforming (nonmodal) DFC's back face has no mana cost of its own, but
    its mana value is calculated from the FRONT face's mana cost (CR 712.8e). A
    modal DFC's back face (CR 712.8f) and a split card's half (CR 709.3b) have
    only their own characteristics, so their own cost stands."""
    if face.is_back and face.layout == "DoubleFaced":
        return face.front_lines
    return face.lines


def card_row_lines(face):
    """The script lines a vocab card's cost and property rows are read from.

    A vocab id names a whole card. The two halves of a split card
    (`AlternateMode:Split`) share ONE id, and off the stack the card has the
    combined characteristics of both halves (CR 709.4): its mana cost is the two
    costs combined (Dead // Gone: {R} + {2}{R} = {2}{R}{R}, mana value 4, CR
    709.4b), and its colors, types and abilities are both halves'. So for a face
    of a split card — whichever half the vocab name resolved to — this returns
    one line set with a combined `ManaCost:`, `Types:` and `Colors:` and both
    halves' other lines. Any other face's rows describe that face alone: its own
    lines."""
    if face.layout != "Split":
        return face.lines
    with open(face.path) as f:
        front, back = split_faces(f.read())
    halves = [front, back or []]
    combined = ["ManaCost:" + _combined_mana_cost(
        [script_field(h, "ManaCost") or "no cost" for h in halves])]
    for key in ("Types", "Colors"):
        words = []
        for h in halves:
            for w in (script_field(h, key) or "").replace(",", " ").split():
                if w not in words:
                    words.append(w)
        if words:
            combined.append(f"{key}:{' '.join(words)}")
    single = ("Name:", "ManaCost:", "Types:", "Colors:")
    for h in halves:
        combined.extend(line for line in h if not line.startswith(single))
    return combined


def _combined_mana_cost(costs):
    """Forge ManaCost values added together (CR 709.4b): the generic amounts
    summed, every other symbol kept. "no cost" when there is nothing to pay."""
    generic = 0
    symbols = []
    for cost in costs:
        if cost.strip() == "no cost":
            continue
        for tok in cost.split():
            if tok.isdigit():
                generic += int(tok)
            else:
                symbols.append(tok)
    parts = ([str(generic)] if generic else []) + symbols
    return " ".join(parts) if parts else "no cost"


def _script_face(path, back):
    """The front (back=False) or back (back=True) CardFace of the script at
    `path`; None when a back face is asked of a single-faced script."""
    with open(path) as f:
        front, back_lines = split_faces(f.read())
    layout = script_field(front, "AlternateMode")
    if not back:
        return CardFace(path, front, front, False, layout)
    if back_lines is None:
        return None
    return CardFace(path, back_lines, front, True, layout)


def _face_named(face, stem):
    """True if the face's own `Name:` folds to `stem`."""
    face_name = script_field(face.lines, "Name")
    return face_name is not None and stem in card_uid_stems(face_name)


@functools.lru_cache(maxsize=None)
def _letter_dirs():
    """Sorted cardsfolder subdirectory names."""
    if not os.path.isdir(CARDS_DIR):
        return ()
    return tuple(sorted(d for d in os.listdir(CARDS_DIR)
                        if os.path.isdir(os.path.join(CARDS_DIR, d))))


@functools.lru_cache(maxsize=None)
def _letter_scripts(letter):
    """Sorted `.txt` script filenames in one cardsfolder subdirectory."""
    path = os.path.join(CARDS_DIR, letter)
    if not os.path.isdir(path):
        return ()
    return tuple(sorted(f for f in os.listdir(path) if f.endswith(".txt")))
