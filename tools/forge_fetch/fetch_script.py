#!/usr/bin/env python3
"""Download a card (or token) script from the Card-Forge/forge repo.

Standalone tool — Python standard library only, imports nothing from the
robomage engine. By default fetches a card's `<name>.txt` from Forge's
cardsfolder and writes it to `bin/resources/cardsfolder/<letter>/<name>.txt`.
With --token, fetches a token script from Forge's flat tokenscripts directory
and writes it to `bin/resources/tokenscripts/<name>.txt`.

This is ADD-ONLY: it never overwrites an existing local script (unless --force),
honouring the project rule "DO NOT MODIFY CARD SCRIPTS".

    python tools/forge_fetch/fetch_script.py "Brainstorm"
    python tools/forge_fetch/fetch_script.py "Lion's Eye Diamond" "Swords to Plowshares"
    python tools/forge_fetch/fetch_script.py "Some Card" --dry-run
    python tools/forge_fetch/fetch_script.py --token w_2_1_cat_warrior

Exit code is the number of requested scripts that could NOT be obtained (0 = all
present/fetched), so a caller can branch on "needs hand-authoring".
"""

import argparse
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
BRANCHES = ("master", "main")
USER_AGENT = "robomage-forge-fetch/1.0"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CARDS_DIR = os.path.join(_REPO_ROOT, "bin", "resources", "cardsfolder")
TOKENS_DIR = os.path.join(_REPO_ROOT, "bin", "resources", "tokenscripts")


def name_to_uid(name):
    """Mirror src/parse.cpp name_to_uid: lowercase, space/hyphen -> '_', drop other punct."""
    return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_").replace("-", "_"))


def _resolve(uid, is_token):
    """Return (url_subpath, dest_path) for a card or token uid."""
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


def fetch_script(name, force=False, dry_run=False, is_token=False):
    """Fetch one card or token Forge script. Returns True if present/written, else False."""
    uid = name_to_uid(name)
    if not uid:
        print(f"  {name}: SKIP (empty normalized name)", file=sys.stderr)
        return False
    kind = "token" if is_token else "card"
    subpath, dest = _resolve(uid, is_token)

    if os.path.exists(dest) and not force:
        print(f"  {name}: already present ({os.path.relpath(dest, _REPO_ROOT)})")
        return True

    body = None
    used = None
    for branch in BRANCHES:
        url = f"{RES_BASE.format(branch=branch)}/{subpath}"
        if dry_run:
            print(f"  {name}: would fetch {url} -> {os.path.relpath(dest, _REPO_ROOT)}")
            return True
        body = fetch_text(url)
        if body is not None:
            used = branch
            break

    if body is None:
        print(f"  {name}: NOT FOUND on Forge ({kind} {uid}.txt) — hand-authoring required",
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
