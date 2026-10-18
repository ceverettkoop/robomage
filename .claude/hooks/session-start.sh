#!/bin/bash
# SessionStart hook — provision the cloud environment so the engine builds and
# runs out of the box: the Forge card scripts for the playable decks. These live
# under bin/resources/cardsfolder/ but are .gitignored and fetched on demand from
# Card-Forge/forge, so a fresh clone has none and the engine asserts the moment
# it tries to load a card. Idempotent, safe to re-run; stdlib python3 only.
#
# It does NOT build the engine — building is left to the user (`make` →
# bin/debug/robomage, `make BUILD=RELEASE` → bin/release/robomage) so the hook
# can't clobber a binary that a running job is exec'ing (a mid-session rebuild
# races the workers and fails them with EACCES).
set -euo pipefail

# Only provision when running in a Claude Code remote (cloud) environment.
# CLAUDE_CODE_REMOTE is set by the remote runner and absent on a local CLI
# session — locally the user's own machine already has the scripts (or manages
# them themselves), so skip the (slow) fetch work.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  echo "[session-start] Not a remote environment (CLAUDE_CODE_REMOTE unset) — skipping provisioning."
  exit 0
fi

ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

echo "[session-start] Fetching Forge card + token scripts for the playable decks..."
# provision_decks.py is the single provisioning entry point (shared with CI): it
# collects every card named by the top-level and league/ decks (plus meta/ when
# that directory exists), fetches any missing card scripts, then scans them for
# the token scripts they create (incl. DFC-combined scripts and synthesized
# Amass / Investigate / Mobilize tokens, mapped by its keyword_tokens table) and
# fetches those too. Add-only and non-fatal — a card missing from Forge just needs
# hand-authoring and never aborts the hook.
python3 tools/forge_fetch/provision_decks.py

echo "[session-start] Done."
