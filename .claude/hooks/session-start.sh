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
#   3. A headless build of the engine (bin/robomage). The GUI build needs raylib
#      which isn't vendored in the cloud env, so we build with HEADLESS=TRUE.
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

echo "[session-start] Fetching Forge card scripts for the playable decks..."
# Collect every card named by the curated, fully-implemented decks (plus the
# basic lands the harness examples use) and fetch any missing scripts. The
# fetch tool is add-only, so already-present scripts are skipped.
"$VENV_PY" - "$ROOT" <<'PY'
import os, re, sys, subprocess
root = sys.argv[1]
decks_dir = os.path.join(root, "bin", "resources", "decks")
# Curated decks the engine/training/harness actually use end to end. (The
# meta/ decks reference cards not yet in src/card_vocab.h and are out of scope.)
decks = ["delver.dk", "doomsday.dk", "mav.dk"]
names = {"Mountain", "Forest", "Island", "Plains", "Swamp"}
for dk in decks:
    path = os.path.join(decks_dir, dk)
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.upper().startswith("SIDEBOARD"):
                continue
            m = re.match(r"^\d+\s+(.+)$", line)
            if m:
                names.add(m.group(1).strip())
tool = os.path.join(root, "tools", "forge_fetch", "fetch_script.py")
# Non-fatal: a card missing from Forge just needs hand-authoring; don't abort
# the whole hook over it.
subprocess.run([sys.executable, tool, *sorted(names)], check=False)
PY

echo "[session-start] Building the engine (headless)..."
make HEADLESS=TRUE

echo "[session-start] Done."
