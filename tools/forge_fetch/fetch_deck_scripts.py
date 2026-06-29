#!/usr/bin/env python3
"""Fetch Forge card scripts for every card referenced by a set of deck files.

Companion to fetch_script.py. Parses .dk deck files, collects the unique card
names (skipping quantities and the SIDEBOARD: marker), and runs each through
fetch_script.fetch_script (add-only; never overwrites without --force).

The SessionStart hook fetches scripts for the top-level and meta/ decks but does
NOT scan decks/league/, so after adding or altering a league deck run this over
that folder to provision its scripts.

Usage:
    # default: every deck in bin/resources/decks/league/
    python tools/forge_fetch/fetch_deck_scripts.py

    # specific deck files or directories (each dir scanned for *.dk)
    python tools/forge_fetch/fetch_deck_scripts.py bin/resources/decks/meta
    python tools/forge_fetch/fetch_deck_scripts.py path/to/a.dk path/to/b.dk

    # pass-through flags
    python tools/forge_fetch/fetch_deck_scripts.py --force --dry-run
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_script

# repo root = three levels up from tools/forge_fetch/this_file
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DECK_DIR = os.path.join(REPO_ROOT, "bin", "resources", "decks", "league")


def collect_dk_files(paths):
    """Expand the given paths (dirs and/or .dk files) into a sorted .dk file list."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "*.dk")))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"  WARNING: no such deck path: {p}", file=sys.stderr)
    return sorted(set(files))


def collect_card_names(dk_files):
    """Parse '<qty> <card name>' deck lines into a de-duped, order-preserving list."""
    names = []
    seen = set()
    for dk in dk_files:
        with open(dk) as f:
            for line in f:
                line = line.strip()
                if not line or line.upper().startswith("SIDEBOARD"):
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2 or not parts[0].isdigit():
                    continue
                name = parts[1].strip()
                key = name.lower()
                if key not in seen:
                    seen.add(key)
                    names.append(name)
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", metavar="PATH",
                    help="deck files or directories of .dk files "
                         "(default: bin/resources/decks/league/)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing local script (default: skip)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved URL/destination without writing")
    args = ap.parse_args(argv)

    dk_files = collect_dk_files(args.paths or [DEFAULT_DECK_DIR])
    if not dk_files:
        print("No .dk deck files found.", file=sys.stderr)
        return 1

    names = collect_card_names(dk_files)
    print(f"{len(names)} unique cards across {len(dk_files)} deck(s)\n")

    missing = 0
    for name in names:
        if not fetch_script.fetch_script(name, force=args.force,
                                         dry_run=args.dry_run):
            missing += 1

    print(f"\n{missing} card(s) could not be fetched")
    return missing


if __name__ == "__main__":
    raise SystemExit(main())
