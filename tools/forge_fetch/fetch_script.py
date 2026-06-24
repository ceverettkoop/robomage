#!/usr/bin/env python3
"""Download a card's Forge script from the Card-Forge/forge repo.

Standalone tool — Python standard library only, imports nothing from the
robomage engine. Fetches `<name>.txt` from Forge's cardsfolder and writes it to
this repo's `bin/resources/cardsfolder/<letter>/<name>.txt`.

This is ADD-ONLY: it never overwrites an existing local script (unless --force),
honouring the project rule "DO NOT MODIFY CARD SCRIPTS".

    python tools/forge_fetch/fetch_script.py "Brainstorm"
    python tools/forge_fetch/fetch_script.py "Lion's Eye Diamond" "Swords to Plowshares"
    python tools/forge_fetch/fetch_script.py "Some Card" --dry-run

Exit code is the number of requested cards that could NOT be obtained (0 = all
present/fetched), so a caller can branch on "needs hand-authoring".
"""

import argparse
import os
import re
import sys
import urllib.error
import urllib.request

# Forge stores scripts under res/cardsfolder/<first-letter>/<name>.txt.
RAW_BASE = "https://raw.githubusercontent.com/Card-Forge/forge/{branch}/forge-gui/res/cardsfolder"
BRANCHES = ("master", "main")
USER_AGENT = "robomage-forge-fetch/1.0"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CARDS_DIR = os.path.join(_REPO_ROOT, "bin", "resources", "cardsfolder")


def name_to_uid(name):
    """Mirror src/parse.cpp name_to_uid: lowercase, space/hyphen -> '_', drop other punct."""
    return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_").replace("-", "_"))


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


def fetch_script(name, force=False, dry_run=False):
    """Fetch one card's Forge script. Returns True if present/written, else False."""
    uid = name_to_uid(name)
    if not uid:
        print(f"  {name}: SKIP (empty normalized name)", file=sys.stderr)
        return False
    letter = uid[0]
    dest = os.path.join(CARDS_DIR, letter, f"{uid}.txt")

    if os.path.exists(dest) and not force:
        print(f"  {name}: already present ({os.path.relpath(dest, _REPO_ROOT)})")
        return True

    body = None
    used = None
    for branch in BRANCHES:
        url = f"{RAW_BASE.format(branch=branch)}/{letter}/{uid}.txt"
        if dry_run:
            print(f"  {name}: would fetch {url} -> {os.path.relpath(dest, _REPO_ROOT)}")
            return True
        body = fetch_text(url)
        if body is not None:
            used = branch
            break

    if body is None:
        print(f"  {name}: NOT FOUND on Forge ({uid}.txt) — hand-authoring required",
              file=sys.stderr)
        return False

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        f.write(body)
    print(f"  {name}: wrote {os.path.relpath(dest, _REPO_ROOT)} (Forge@{used})")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cards", nargs="+", help="card name(s) to fetch")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing local script (default: skip)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved URL/destination without writing")
    args = ap.parse_args(argv)

    missing = 0
    for name in args.cards:
        if not fetch_script(name, force=args.force, dry_run=args.dry_run):
            missing += 1
    return missing


if __name__ == "__main__":
    raise SystemExit(main())
