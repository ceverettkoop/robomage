#!/usr/bin/env python3
"""Scrape the top Legacy metagame decks from MTGTop8 into robomage .dk files.

Standalone tool — depends only on the Python standard library and does NOT import
anything from the robomage engine or train/. Run it from anywhere:

    python tools/meta_scraper/scrape_meta.py --top 10
    python tools/meta_scraper/scrape_meta.py --top 10 --dry-run

It writes one `<archetype>.dk` per archetype into bin/resources/decks/meta/ (by
default), in the same format as the hand-built decks in bin/resources/decks/.

Source endpoints (MTGTop8):
  - metagame page : https://mtgtop8.com/format?f=LE
        anchors `<a href=archetype?a=ID&meta=M&f=LE>NAME</a>` listed in
        descending meta-share order, each followed by a `... %` div.
  - archetype page: https://mtgtop8.com/archetype?a=ID&f=LE
        deck rows `<a href=/event?e=E&d=DECKID&f=LE>...</a>`; the first row is
        the most recent/representative list.
  - deck export   : https://mtgtop8.com/mtgo?d=DECKID
        plain MTGO text: `<qty> <name>` (CRLF) lines, a `Sideboard` line, then
        the sideboard `<qty> <name>` lines.
"""

import argparse
import html
import os
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "https://mtgtop8.com"
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# tools/meta_scraper/scrape_meta.py -> repo root is two levels up.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUT = os.path.join(_REPO_ROOT, "bin", "resources", "decks", "meta")
DEFAULT_CARDS = os.path.join(_REPO_ROOT, "bin", "resources", "cardsfolder")


def name_to_uid(name):
    """Mirror src/parse.cpp name_to_uid: lowercase, space/hyphen -> '_', drop other punct."""
    return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_").replace("-", "_"))


def _script_face_names(path):
    """Return the list of `Name:` values in a card script (front first, back after ALTERNATE)."""
    names = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("Name:"):
                names.append(line[len("Name:"):].strip())
    return names


def expand_dfc_name(name, cards_dir):
    """Expand a single-face DFC/adventure name to the combined `Front Back` form.

    Multi-face cards (DFCs, adventures, split cards) are stored under a combined
    `front_back` filename (e.g. ajani_nacatl_pariah_ajani_nacatl_avenger.txt), and
    decks must reference that combined name so name_to_uid resolves the script
    (mirrors how delver.dk writes "Delver of Secrets Insectile Aberration").

    MTGTop8 lists only the front face, so when there is no exact single-face
    script but a `<uid>_*.txt` combined script exists whose front Name matches,
    return "Front Back". Otherwise return the name unchanged (single-faced, or
    no local script — left for the implement-missing-cards workflow).
    """
    uid = name_to_uid(name)
    if not uid or not cards_dir:
        return name
    letter_dir = os.path.join(cards_dir, uid[0])
    if not os.path.isdir(letter_dir):
        return name
    if os.path.exists(os.path.join(letter_dir, f"{uid}.txt")):
        return name  # single-faced card (exact script present)
    for fn in sorted(os.listdir(letter_dir)):
        if not (fn.startswith(uid + "_") and fn.endswith(".txt")):
            continue
        faces = _script_face_names(os.path.join(letter_dir, fn))
        if len(faces) >= 2 and name_to_uid(faces[0]) == uid:
            return f"{faces[0]} {faces[1]}"
    return name


def fetch(url, retries=3, timeout=25):
    """GET a URL with a browser User-Agent; retry on transient failures."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def parse_metagame(page_html, top):
    """Return up to `top` (archetype_id, name, pct_str) tuples in meta order."""
    # Anchor immediately followed (within the same card block) by a `<n> %` div.
    pat = re.compile(
        r'archetype\?a=(\d+)[^>]*>(.*?)</a>.*?'
        r'<div[^>]*>\s*([\d.]+)\s*%\s*</div>',
        re.S)
    out, seen = [], set()
    for m in pat.finditer(page_html):
        aid = m.group(1)
        if aid in seen:
            continue
        seen.add(aid)
        name = html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
        out.append((aid, name, m.group(3)))
        if len(out) >= top:
            break
    return out


def parse_archetype_decks(page_html):
    """Return deck ids (`event?e=E&d=DECKID`) in page order (first = newest)."""
    ids, seen = [], set()
    for m in re.finditer(r'event\?e=\d+&d=(\d+)', page_html):
        did = m.group(1)
        if did not in seen:
            seen.add(did)
            ids.append(did)
    return ids


def fetch_deck(deck_id):
    """Return (mainboard, sideboard) as lists of (qty, name) from a deck export."""
    text = fetch(f"{BASE}/mtgo?d={deck_id}")
    main, side, in_side = [], [], False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower() == "sideboard":
            in_side = True
            continue
        m = re.match(r"(\d+)\s+(.+)", line)
        if not m:
            continue
        entry = (int(m.group(1)), m.group(2).strip())
        (side if in_side else main).append(entry)
    return main, side


def to_dk(main, side):
    """Render mainboard/sideboard entries as robomage .dk text (tab-separated)."""
    lines = [f"{qty}\t{name}" for qty, name in main]
    if side:
        lines.append("")
        lines.append("SIDEBOARD:")
        lines += [f"{qty}\t{name}" for qty, name in side]
    return "\n".join(lines) + "\n"


def sanitize_filename(name):
    """Archetype name -> safe .dk stem (lowercase, ascii, underscore-separated)."""
    s = name.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "deck"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=10,
                    help="number of top archetypes to fetch (default 10)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output directory for .dk files (default bin/resources/decks/meta)")
    ap.add_argument("--format", default="LE",
                    help="MTGTop8 format code (default LE = Legacy)")
    ap.add_argument("--cardsfolder", default=DEFAULT_CARDS,
                    help="cardsfolder used to expand DFC names to Front Back "
                         "(default bin/resources/cardsfolder; '' to disable)")
    ap.add_argument("--delay", type=float, default=0.6,
                    help="seconds to wait between requests (be polite; default 0.6)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be written without creating files")
    args = ap.parse_args(argv)

    meta_url = f"{BASE}/format?f={args.format}"
    print(f"Fetching metagame: {meta_url}")
    archetypes = parse_metagame(fetch(meta_url), args.top)
    if not archetypes:
        print("ERROR: no archetypes parsed (page structure may have changed).",
              file=sys.stderr)
        return 1

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)

    written, total_cards = [], set()
    for rank, (aid, name, pct) in enumerate(archetypes, 1):
        time.sleep(args.delay)
        try:
            deck_ids = parse_archetype_decks(fetch(f"{BASE}/archetype?a={aid}&f={args.format}"))
        except RuntimeError as e:
            print(f"  [{rank:2}] {name}: SKIP ({e})", file=sys.stderr)
            continue
        if not deck_ids:
            print(f"  [{rank:2}] {name}: SKIP (no decklists found)", file=sys.stderr)
            continue
        time.sleep(args.delay)
        try:
            main_b, side_b = fetch_deck(deck_ids[0])
        except RuntimeError as e:
            print(f"  [{rank:2}] {name}: SKIP ({e})", file=sys.stderr)
            continue
        if not main_b:
            print(f"  [{rank:2}] {name}: SKIP (empty mainboard)", file=sys.stderr)
            continue

        # Expand multi-face cards (DFC/adventure/split) to their combined name so
        # the engine resolves the script (e.g. "Brazen Borrower" -> "Brazen Borrower Petty Theft").
        main_b = [(q, expand_dfc_name(cn, args.cardsfolder)) for q, cn in main_b]
        side_b = [(q, expand_dfc_name(cn, args.cardsfolder)) for q, cn in side_b]

        fname = f"{sanitize_filename(name)}.dk"
        path = os.path.join(args.out, fname)
        dk = to_dk(main_b, side_b)
        for _, cn in main_b + side_b:
            total_cards.add(cn)
        n_main = sum(q for q, _ in main_b)
        n_side = sum(q for q, _ in side_b)
        print(f"  [{rank:2}] {pct:>5}%  {name:<22} -> {fname}  ({n_main} main, {n_side} side)")
        if args.dry_run:
            continue
        with open(path, "w") as f:
            f.write(dk)
        written.append(fname)

    print(f"\n{'Would write' if args.dry_run else 'Wrote'} {len(written) or len(archetypes)} "
          f"deck(s); {len(total_cards)} unique card names"
          f"{' (dry run)' if args.dry_run else f' -> {args.out}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
