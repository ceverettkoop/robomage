#!/usr/bin/env python3
"""List cards referenced by deck files that are not yet in src/card_vocab.h.

Drives the card-implementation workflow: it diffs the cards used by a set of
deck files against the registered vocabulary and reports, for each missing card,
whether a Forge script already exists locally and a suggested next vocab index.

    train/.venv/bin/python train/missing_cards.py                 # scan decks/meta
    train/.venv/bin/python train/missing_cards.py --decks-dir bin/resources/decks
    train/.venv/bin/python train/missing_cards.py --json

Output is sorted by cross-deck frequency (most-played missing cards first), so
high-impact cards get implemented first.

Double-faced / adventure cards are matched the way the engine understands them: a
deck entry that concatenates both face names (e.g. "Delver of Secrets Insectile
Aberration") is resolved through its script to its front-face name and checked
against the vocab by that name, so an already-implemented DFC is not reported missing
merely because of its combined deck-file name.
"""

import argparse
import json
import os
import re
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VOCAB_H = os.path.join(_REPO_ROOT, "src", "card_vocab.h")
CARDS_DIR = os.path.join(_REPO_ROOT, "bin", "resources", "cardsfolder")
DEFAULT_DECKS = os.path.join(_REPO_ROOT, "bin", "resources", "decks", "meta")


def name_to_uid(name):
    """Mirror src/parse.cpp name_to_uid: lowercase, space/hyphen/slash -> '_', drop other
    punct ('/' is a split-card separator, so "Dead/Gone" -> "dead_gone")."""
    return re.sub(r"[^a-z0-9_]", "",
                  name.lower().replace(" ", "_").replace("-", "_").replace("/", "_"))


def parse_vocab(path):
    """Return (uid->index dict, highest_index) from card_vocab.h entries."""
    text = open(path).read()
    by_uid, highest = {}, -1
    for m in re.finditer(r'"([^"]+)"\s*,\s*(\d+)', text):
        idx = int(m.group(2))
        by_uid[name_to_uid(m.group(1))] = idx
        highest = max(highest, idx)
    return by_uid, highest


def parse_deck(path):
    """Yield (qty, card_name) for each mainboard/sideboard line in a .dk file."""
    for raw in open(path):
        line = raw.strip()
        if not line or line.upper().startswith("SIDEBOARD"):
            continue
        m = re.match(r"(\d+)\s+(.+)", line)
        if m:
            yield int(m.group(1)), m.group(2).strip()


def resolve_script_path(uid):
    """The script file the engine would load for this uid, mirroring src/card_db.cpp.

    Returns "<uid>.txt" when it exists, else a double-faced card's combined
    "<uid>_<back>.txt" file (the engine's DFC fallback when a front-face uid has no
    script of its own), else None."""
    if not uid:
        return None
    direct = os.path.join(CARDS_DIR, uid[0], f"{uid}.txt")
    if os.path.exists(direct):
        return direct
    letter_dir = os.path.join(CARDS_DIR, uid[0])
    if os.path.isdir(letter_dir):
        prefix = uid + "_"
        for fn in sorted(os.listdir(letter_dir)):
            if fn.startswith(prefix) and fn.endswith(".txt"):
                return os.path.join(letter_dir, fn)
    return None


def front_face_uid(script_path):
    """UID of a script's FRONT face (its first `Name:` line), or None.

    For a double-faced/adventure card the combined deck name (e.g. "Delver of Secrets
    Insectile Aberration") is not how the engine identifies the card: the engine
    registers and one-hot-encodes it by its front-face name ("Delver of Secrets").
    This returns that front-face uid so a DFC can be matched against the vocab the
    same way the engine understands it."""
    try:
        with open(script_path) as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("Name:"):
                    return name_to_uid(line[len("Name:"):].strip())
    except OSError:
        pass
    return None


def vocab_identity(uid, vocab):
    """The vocab index by which the engine understands this deck card, or None.

    A card referenced directly by a registered name matches immediately. A
    double-faced card referenced by its combined "<front> <back>" deck name (whose
    concatenated uid is not itself a vocab key) is resolved through its script to the
    FRONT-FACE name — which is the identity the engine registers (src/card_db.cpp
    load_card) and the vocab encodes — so an already-implemented DFC is not reported
    missing just because of its combined deck-file name."""
    if uid in vocab:
        return vocab[uid]
    path = resolve_script_path(uid)
    if path:
        front = front_face_uid(path)
        if front and front in vocab:
            return vocab[front]
    return None


def has_local_script(uid):
    return resolve_script_path(uid) is not None


def collect(decks_dir):
    """Scan all .dk files; return {uid: {name, deck_count, copies}}."""
    refs = {}
    dk_files = sorted(f for f in os.listdir(decks_dir) if f.endswith(".dk"))
    for fn in dk_files:
        seen_in_deck = set()
        for qty, name in parse_deck(os.path.join(decks_dir, fn)):
            uid = name_to_uid(name)
            r = refs.setdefault(uid, {"name": name, "deck_count": 0, "copies": 0})
            r["copies"] += qty
            if uid not in seen_in_deck:
                r["deck_count"] += 1
                seen_in_deck.add(uid)
    return refs, dk_files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decks-dir", default=DEFAULT_DECKS,
                    help="directory of .dk files to scan (default bin/resources/decks/meta)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.decks_dir):
        print(f"ERROR: no such directory: {args.decks_dir}", file=sys.stderr)
        return 2

    vocab, highest = parse_vocab(VOCAB_H)
    refs, dk_files = collect(args.decks_dir)

    # A card is "missing" only if the engine has no vocab identity for it. This
    # resolves double-faced cards by their front-face name (see vocab_identity), so an
    # implemented DFC referenced by its combined deck-file name is not falsely flagged.
    missing = [
        {"name": r["name"], "uid": uid, "in_vocab": False,
         "has_local_script": has_local_script(uid),
         "deck_count": r["deck_count"], "copies": r["copies"]}
        for uid, r in refs.items() if vocab_identity(uid, vocab) is None
    ]
    # Most-played first: by deck presence, then total copies, then name.
    missing.sort(key=lambda d: (-d["deck_count"], -d["copies"], d["name"].lower()))
    for i, d in enumerate(missing):
        d["suggested_index"] = highest + 1 + i

    if args.json:
        print(json.dumps({"decks_dir": args.decks_dir, "deck_files": dk_files,
                          "vocab_size": len(vocab), "highest_index": highest,
                          "missing": missing}, indent=2))
        return 0

    print(f"Scanned {len(dk_files)} deck(s) in {os.path.relpath(args.decks_dir, _REPO_ROOT)}")
    print(f"Vocab: {len(vocab)} cards (highest index {highest}); "
          f"{len(missing)} unique missing card(s)\n")
    if not missing:
        print("All referenced cards are already in the vocab.")
        return 0
    print(f"  {'idx':>4}  {'scr':>3}  {'decks':>5}  {'copies':>6}  name")
    print(f"  {'-'*4}  {'-'*3}  {'-'*5}  {'-'*6}  {'-'*30}")
    for d in missing:
        scr = "yes" if d["has_local_script"] else " no"
        print(f"  {d['suggested_index']:>4}  {scr:>3}  {d['deck_count']:>5}  "
              f"{d['copies']:>6}  {d['name']}")
    n_no_script = sum(1 for d in missing if not d["has_local_script"])
    print(f"\n{len(missing)} missing; {n_no_script} without a local Forge script "
          f"(need fetch or hand-authoring).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
