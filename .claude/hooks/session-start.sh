#!/bin/bash
# SessionStart hook — provision the cloud environment so the engine builds and
# the test harness / training env run out of the box.
#
# What it sets up (all idempotent, safe to re-run):
#   1. Python venv (train/.venv) with the harness/env deps (numpy, gymnasium).
#   2. Forge card scripts for the playable decks. These live under
#      bin/resources/cardsfolder/ but are .gitignored and fetched on demand
#      from Card-Forge/forge, so a fresh clone has none and the engine asserts
#      the moment it tries to load a card.
#
# It does NOT build the engine — building is left to the user (`make HEADLESS=TRUE`)
# so the hook can't clobber a bin/robomage that a running training/league job is
# exec'ing (a mid-session rebuild races the workers and fails them with EACCES).
set -euo pipefail

ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

VENV_PY="train/.venv/bin/python"

echo "[session-start] Setting up Python venv + harness deps..."
if [ ! -x "$VENV_PY" ]; then
  python3 -m venv train/.venv
fi
"$VENV_PY" -m pip install --quiet --upgrade pip
# numpy + gymnasium are all the test harness / NarrativeEnv import chain needs.
# (torch / stable-baselines3 are only needed for PPO training; install those
# manually when you actually train — they are large and slow to fetch.)
"$VENV_PY" -m pip install --quiet numpy gymnasium

echo "[session-start] Fetching Forge card + token scripts for the playable decks..."
# Collect every card named by the curated, fully-implemented decks (plus the
# basic lands the harness examples use) and fetch any missing scripts, then
# fetch the token scripts those cards create. The fetch tool is add-only, so
# already-present scripts are skipped.
"$VENV_PY" - "$ROOT" <<'PY'
import os, re, sys, subprocess, glob
root = sys.argv[1]
decks_dir = os.path.join(root, "bin", "resources", "decks")
cards_dir = os.path.join(root, "bin", "resources", "cardsfolder")
tool = os.path.join(root, "tools", "forge_fetch", "fetch_script.py")

def name_to_uid(name):
    # Mirror src/parse.cpp name_to_uid (and the fetch tool).
    return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_").replace("-", "_"))

# Decks to provision card scripts for: the curated, fully-implemented decks the
# engine/training/harness use end to end (delver/doomsday/mav, the top-level
# .dk files), the scraped metagame decks under meta/, and the PFSP league
# roster under league/ (so `train.py league` and league regression runs have
# their card + token scripts). The meta/ decks reference some cards not yet in
# src/card_vocab.h, but fetching their scripts is harmless (add-only / non-fatal)
# and lets those decks load for testing.
# (not_used/ holds throwaway test decks and is intentionally excluded.)
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

# Non-fatal: a card missing from Forge just needs hand-authoring; don't abort
# the whole hook over it.
subprocess.run([sys.executable, tool, *sorted(names)], check=False)

# A card names each token it creates by script stem in a "TokenScript$ <stem>"
# field; the engine loads that stem from bin/resources/tokenscripts/<stem>.txt.
# Scan the (now-fetched) card scripts for those stems and fetch them as tokens
# so token-creating cards don't hit a missing-token-script error at runtime.
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
if tokens:
    subprocess.run([sys.executable, tool, "--token", *sorted(tokens)], check=False)
PY

echo "[session-start] Done."
