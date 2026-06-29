#!/usr/bin/env python3
"""Download a card (or token) script from the Card-Forge/forge repo.

Standalone tool — Python standard library only, imports nothing from the
robomage engine. By default fetches a card's `<name>.txt` from Forge's
cardsfolder and writes it to `bin/resources/cardsfolder/<letter>/<name>.txt`.
With --token, fetches a token script from Forge's flat tokenscripts directory
and writes it to `bin/resources/tokenscripts/<name>.txt`.

This is ADD-ONLY: it never overwrites an existing local script (unless --force),
honouring the project rule "DO NOT MODIFY CARD SCRIPTS".

Double-faced cards (DFCs / MDFCs / transform / Rooms): Forge stores these under
a single COMBINED "<front>_<back>.txt" filename, NOT under the front face alone.
The engine mirrors this — src/card_db.cpp load_card() resolves "<uid>.txt" first
and only falls back to scanning for a combined "<uid>_*.txt" when no exact file
exists. So this tool MUST NOT write a front-name-only "<uid>.txt" when the
combined script is already present (or is what Forge actually serves): doing so
adds the card to the engine TWICE and the duplicate front-name file shadows the
correct combined script. To honour that, the tool:
  * treats a card as "already present" if EITHER "<uid>.txt" OR a combined
    "<uid>_<back>.txt" (whose front face normalizes back to <uid>) exists locally;
  * on a front-name miss, discovers the combined "<uid>_*.txt" Forge serves (via
    the GitHub contents API) and fetches THAT under its combined name.

    python tools/forge_fetch/fetch_script.py "Brainstorm"
    python tools/forge_fetch/fetch_script.py "Lion's Eye Diamond" "Swords to Plowshares"
    python tools/forge_fetch/fetch_script.py "Tamiyo, Inquisitive Student"   # DFC
    python tools/forge_fetch/fetch_script.py "Some Card" --dry-run
    python tools/forge_fetch/fetch_script.py --token w_2_1_cat_warrior

Exit code is the number of requested scripts that could NOT be obtained (0 = all
present/fetched), so a caller can branch on "needs hand-authoring".

Known gap (not handled): accented card names in the FILENAME/uid. name_to_uid
mirrors the C++ engine, which drops non-ASCII bytes rather than transliterating
them (an accented "o" becomes nothing, not "o"), so an accented name won't match
Forge's transliterated filename. Such cards must be fetched/named by their
ASCII-transliterated stem by hand. (The engine's ML vocab match is separately
accent-insensitive, so an accented CardData::name still maps to its ASCII entry.)
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

# Forge serves resources under forge-gui/res/. Cards live under
# cardsfolder/<first-letter>/<name>.txt; token scripts live flat under
# tokenscripts/<name>.txt (the same stem the engine reads from a card's
# TokenScript$ field, e.g. "w_2_1_cat_warrior").
RES_BASE = "https://raw.githubusercontent.com/Card-Forge/forge/{branch}/forge-gui/res"
# Contents API used to discover a DFC's combined "<front>_<back>.txt" filename
# (the raw endpoint can't list a directory). Unauthenticated -> 60 req/hr and a
# 1000-entry listing cap per directory; both are fine for the occasional DFC and
# any failure degrades gracefully to "NOT FOUND" (hand-authoring).
API_DIR = ("https://api.github.com/repos/Card-Forge/forge/contents/"
           "forge-gui/res/cardsfolder/{letter}?ref={branch}")
BRANCHES = ("master", "main")
USER_AGENT = "robomage-forge-fetch/1.0"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CARDS_DIR = os.path.join(_REPO_ROOT, "bin", "resources", "cardsfolder")
TOKENS_DIR = os.path.join(_REPO_ROOT, "bin", "resources", "tokenscripts")

# Per-run cache of contents-API directory listings, keyed by (letter, branch);
# value is the list of filenames or None when the listing was unavailable.
_DIR_CACHE = {}


def name_to_uid(name):
    """Mirror src/parse.cpp name_to_uid: lowercase, space/hyphen -> '_', drop other punct."""
    return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_").replace("-", "_"))


def _resolve(uid, is_token):
    """Return (url_subpath, dest_path) for the direct (front-name) card or token uid."""
    if is_token:
        # tokenscripts are a flat directory, keyed by the script stem.
        return f"tokenscripts/{uid}.txt", os.path.join(TOKENS_DIR, f"{uid}.txt")
    # cards are bucketed by first letter.
    letter = uid[0]
    return f"cardsfolder/{letter}/{uid}.txt", os.path.join(CARDS_DIR, letter, f"{uid}.txt")


def fetch_text(url, timeout=25):
    """Return the body for a 200 response, or None for 404; raise on other errors."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _first_name(body):
    """The card name from a script's first 'Name:' line, or '' if none."""
    for line in body.splitlines():
        if line.startswith("Name:"):
            return line[len("Name:"):].strip()
    return ""


def _front_face_is(path_or_body, uid, is_path):
    """True if the script's front face (first 'Name:' line) normalizes to uid.

    Guards the combined-name match against a prefix collision — a single-faced
    card whose uid is a prefix of an unrelated DFC's combined filename (e.g.
    'cloak_and_dagger' vs 'cloak_and_dagger_entwined.txt')."""
    try:
        body = path_or_body
        if is_path:
            with open(path_or_body, encoding="utf-8", errors="replace") as f:
                body = f.read()
    except OSError:
        return False
    return name_to_uid(_first_name(body)) == uid


def _local_combined(letter_dir, uid):
    """A local combined DFC script '<uid>_<back>.txt' whose front face is uid, else None."""
    if not os.path.isdir(letter_dir):
        return None
    prefix = uid + "_"
    for fn in sorted(os.listdir(letter_dir)):
        if fn.startswith(prefix) and fn.endswith(".txt"):
            cand = os.path.join(letter_dir, fn)
            if _front_face_is(cand, uid, is_path=True):
                return cand
    return None


def _local_present(uid, is_token):
    """Existing local script path for this uid (exact, or verified DFC-combined), else None.

    Mirrors the engine's load_card() resolution (src/card_db.cpp) so the tool's
    'already present' decision matches what the engine would actually load — which
    is what keeps it from writing a second, shadowing copy of a DFC."""
    if is_token:
        p = os.path.join(TOKENS_DIR, f"{uid}.txt")
        return p if os.path.exists(p) else None
    letter_dir = os.path.join(CARDS_DIR, uid[0])
    exact = os.path.join(letter_dir, f"{uid}.txt")
    if os.path.exists(exact):
        return exact
    return _local_combined(letter_dir, uid)


def _list_forge_dir(letter, branch):
    """Cached contents-API listing of a cardsfolder letter dir, or None if unavailable."""
    key = (letter, branch)
    if key in _DIR_CACHE:
        return _DIR_CACHE[key]
    names = None
    try:
        body = fetch_text(API_DIR.format(letter=letter, branch=branch))
        if body is not None:
            names = [e.get("name", "") for e in json.loads(body)]
    except (urllib.error.URLError, ValueError, OSError):
        names = None  # rate-limited / offline / malformed: degrade gracefully
    _DIR_CACHE[key] = names
    return names


def _discover_forge_dfc(uid, branch):
    """The combined '<uid>_<back>.txt' filename Forge serves for a DFC, or None."""
    names = _list_forge_dir(uid[0], branch)
    if not names:
        return None
    prefix = uid + "_"
    for nm in sorted(names):
        if nm.startswith(prefix) and nm.endswith(".txt"):
            return nm
    return None


def _write(name, dest, body, branch):
    """Write a fetched script to dest and report it. Returns True."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        f.write(body)
    print(f"  {name}: wrote {os.path.relpath(dest, _REPO_ROOT)} (Forge@{branch})")
    return True


def fetch_script(name, force=False, dry_run=False, is_token=False):
    """Fetch one card or token Forge script. Returns True if present/written, else False."""
    uid = name_to_uid(name)
    if not uid:
        print(f"  {name}: SKIP (empty normalized name)", file=sys.stderr)
        return False
    kind = "token" if is_token else "card"

    # ADD-ONLY + no-duplicate: skip if already present locally, INCLUDING a DFC
    # stored under its combined "<front>_<back>.txt" name. Re-fetching such a card
    # under the front-name-only file would add it to the engine twice and shadow
    # the combined script (see module docstring / src/card_db.cpp).
    if not force:
        present = _local_present(uid, is_token)
        if present:
            print(f"  {name}: already present ({os.path.relpath(present, _REPO_ROOT)})")
            return True

    subpath, dest = _resolve(uid, is_token)
    if dry_run:
        # Report the primary (front-name) URL. A DFC's combined filename is resolved
        # live at fetch time (it needs a directory listing), so it isn't shown here.
        url = f"{RES_BASE.format(branch=BRANCHES[0])}/{subpath}"
        print(f"  {name}: would fetch {url} -> {os.path.relpath(dest, _REPO_ROOT)}")
        return True

    # 1. Direct "<uid>.txt" — single-faced cards and tokens.
    for branch in BRANCHES:
        body = fetch_text(f"{RES_BASE.format(branch=branch)}/{subpath}")
        if body is not None:
            return _write(name, dest, body, branch)

    # 2. Front-name miss for a card → likely a double-faced card Forge stores under
    #    a combined "<uid>_<back>.txt" filename. Discover and fetch THAT (verifying
    #    its front face) so it lands under the combined name the engine expects —
    #    never a duplicate front-name file.
    if not is_token:
        for branch in BRANCHES:
            combined = _discover_forge_dfc(uid, branch)
            if not combined:
                continue
            url = f"{RES_BASE.format(branch=branch)}/cardsfolder/{uid[0]}/{combined}"
            body = fetch_text(url)
            if body is None or not _front_face_is(body, uid, is_path=False):
                continue  # gone, or a prefix collision rather than this card's front face
            return _write(name, os.path.join(CARDS_DIR, uid[0], combined), body, branch)

    print(f"  {name}: NOT FOUND on Forge ({kind} {uid}.txt) — hand-authoring required",
          file=sys.stderr)
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cards", nargs="+", metavar="NAME",
                    help="card name(s), or token script stem(s) with --token")
    ap.add_argument("--token", action="store_true",
                    help="fetch token script(s) from Forge's tokenscripts dir "
                         "into bin/resources/tokenscripts/ (NAME is the script "
                         "stem, e.g. w_2_1_cat_warrior)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing local script (default: skip)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved URL/destination without writing")
    args = ap.parse_args(argv)

    missing = 0
    for name in args.cards:
        if not fetch_script(name, force=args.force, dry_run=args.dry_run,
                            is_token=args.token):
            missing += 1
    return missing


if __name__ == "__main__":
    raise SystemExit(main())
