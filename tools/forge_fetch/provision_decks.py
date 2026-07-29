#!/usr/bin/env python3
"""Provision Forge card + token scripts for every playable deck.

Single provisioning entry point shared by the SessionStart hook
(.claude/hooks/session-start.sh) and CI (.github/actions/setup-robomage). Card
scripts live under bin/resources/cardsfolder/ but are .gitignored and fetched on
demand from Card-Forge/forge, so a fresh clone / CI runner has none and the
engine asserts the moment it tries to load a card.

It collects the union of (a) every card registered in src/card_vocab.h — the
engine's full implemented-card set — and (b) every card named by the
curated/fully-implemented decks (top-level *.dk), the scraped metagame decks
(meta/), and the PFSP league roster (league/), plus the basic lands the harness
examples use. Fetching the whole vocab (not just deck cards) is what keeps the
codegen deterministic: train/gen_card_costs.py reads each vocab card's ManaCost
from its script, so a vocab card whose script is absent would regenerate to a
zero cost — a spurious card_costs.py diff that varies by which decks were
provisioned. Deck cards not yet in the vocab (unimplemented meta cards) are still
fetched so those decks load for testing. It then scans the fetched card scripts
for the token scripts those cards create and fetches those too. The underlying
fetch tool (tools/forge_fetch/fetch_script.py) is add-only and DFC-aware: a card
already present under its combined "<front>_<back>.txt" name is skipped, and a
front-name miss discovers and fetches the combined file (via the GitHub contents
API, or a card_vocab.h back-face fallback when that API is unavailable), so
double-faced cards are never fetched/errored twice.

Non-fatal by design: a card missing from Forge just needs hand-authoring, so a
failed fetch never aborts provisioning (always exits 0). Set GITHUB_TOKEN in the
environment to authenticate the contents-API calls fetch_script.py makes to
resolve DFC filenames (raises the 60 req/hr unauthenticated cap to 5000 in CI).
"""
import glob
import os
import re
import subprocess
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VOCAB_H = os.path.join(_REPO_ROOT, "src", "card_vocab.h")


def name_to_uid(name):
    """Mirror src/parse.cpp name_to_uid (and the fetch tool): lowercase,
    space/hyphen/slash -> '_', drop other punct. '/' is a split-card separator
    (CR 709), so "Dead/Gone" -> "dead_gone", matching Forge's filename."""
    return re.sub(r"[^a-z0-9_]", "",
                  name.lower().replace(" ", "_").replace("-", "_").replace("/", "_"))


def collect_vocab_names(vocab_h):
    """Every card name registered in src/card_vocab.h, as a set."""
    try:
        with open(vocab_h, encoding="utf-8", errors="replace") as f:
            return set(re.findall(r'\{"((?:[^"\\]|\\.)*)"\s*,\s*\d+\}', f.read()))
    except OSError:
        return set()


def collect_deck_names(decks_dir):
    """Every card name referenced by the top-level, meta/, and league/ decks
    (plus the basic lands), as a set. not_used/ is intentionally excluded."""
    deck_paths = []
    for fn in sorted(os.listdir(decks_dir)):
        if fn.endswith(".dk"):
            deck_paths.append(os.path.join(decks_dir, fn))
    for sub in ("meta", "league"):
        sub_dir = os.path.join(decks_dir, sub)
        if os.path.isdir(sub_dir):
            for fn in sorted(os.listdir(sub_dir)):
                if fn.endswith(".dk"):
                    deck_paths.append(os.path.join(sub_dir, fn))

    names = {"Mountain", "Forest", "Island", "Plains", "Swamp"}
    for path in deck_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.upper().startswith("SIDEBOARD"):
                    continue
                m = re.match(r"^\d+\s+(.+)$", line)
                if m:
                    names.add(m.group(1).strip())
    return names


def collect_tokens(names, cards_dir):
    """Scan the (now-fetched) card scripts named by `names` for the token scripts
    they create, returning the set of token script stems to fetch."""
    # A card names each token it creates by script stem in a "TokenScript$ <stem>"
    # field; the engine loads that stem from bin/resources/tokenscripts/<stem>.txt.
    tok_re = re.compile(r"TokenScript\$\s*([a-z0-9_]+)")
    # Amass (DB$ Amass | Type$ <subtype>, rule 701.46) creates/grows an Army token whose
    # script stem the engine SYNTHESIZES as "b_0_0_<subtype>_army" (src/effects/effect_amass.cpp)
    # — there is no TokenScript$ field to scan, so detect the Amass Type$ separately and fetch
    # the matching army token (e.g. Orcish Bowmasters' "Amass Orcs" -> b_0_0_orc_army).
    amass_re = re.compile(r"Amass\b.*?Type\$\s*([A-Za-z]+)")
    # Some tokens are SYNTHESIZED by the engine from a keyword/effect with no TokenScript$
    # field to scan (src/effects/effect_*.cpp): an Investigate effect makes the Clue token
    # "c_a_clue_draw" and a Mobilize effect makes the Warrior token "r_1_1_warrior". Detect
    # those by keyword so their scripts are provisioned too (Amass is handled above).
    keyword_tokens = [
        (re.compile(r"\bInvestigate\b"), "c_a_clue_draw"),
        (re.compile(r"\bMobilize\b"), "r_1_1_warrior"),
    ]
    tokens = set()
    for nm in names:
        uid = name_to_uid(nm)
        if not uid:
            continue
        # Resolve the card script: exact <uid>.txt, else the combined DFC/MDFC/Room
        # filename <uid>_*.txt (Forge stores double-faced cards under one combined file —
        # e.g. tamiyo_inquisitive_student_tamiyo_seasoned_scholar.txt — which a front-name
        # uid alone won't match, so its tokens would otherwise be missed).
        cdir = os.path.join(cards_dir, uid[0])
        cpath = os.path.join(cdir, uid + ".txt")
        if not os.path.exists(cpath):
            combined = sorted(glob.glob(os.path.join(cdir, uid + "_*.txt")))
            cpath = combined[0] if combined else None
        if not cpath or not os.path.exists(cpath):
            continue
        with open(cpath) as f:
            text = f.read()
        tokens.update(tok_re.findall(text))
        for sub in amass_re.findall(text):
            tokens.add("b_0_0_" + sub.lower() + "_army")
        for pat, stem in keyword_tokens:
            if pat.search(text):
                tokens.add(stem)
    return tokens


def main(argv=None):
    root = _REPO_ROOT
    decks_dir = os.path.join(root, "bin", "resources", "decks")
    cards_dir = os.path.join(root, "bin", "resources", "cardsfolder")
    tool = os.path.join(root, "tools", "forge_fetch", "fetch_script.py")

    # Union of the full vocab and every deck's cards (see module docstring).
    names = collect_vocab_names(_VOCAB_H) | collect_deck_names(decks_dir)
    # Non-fatal: a card missing from Forge just needs hand-authoring, don't abort
    # the whole provisioning over it.
    subprocess.run([sys.executable, tool, *sorted(names)], check=False)

    tokens = collect_tokens(names, cards_dir)
    if tokens:
        subprocess.run([sys.executable, tool, "--token", *sorted(tokens)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
