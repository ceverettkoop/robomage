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
# provision_decks.py is the single provisioning entry point (shared with CI): it
# collects every card named by the top-level / meta/ / league/ decks, fetches any
# missing card scripts, then scans them for the token scripts they create (incl.
# DFC-combined scripts and synthesized Amass / Investigate / Mobilize tokens) and
# fetches those too. Add-only and non-fatal — a card missing from Forge just needs
# hand-authoring and never aborts the hook.
"$VENV_PY" tools/forge_fetch/provision_decks.py

echo "[session-start] Done."
