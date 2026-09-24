#!/bin/bash
# The GUI app on its welcome pane (File ▸ New Session); extra play.py flags
# pass through, e.g. ./gui.sh --deck-a league/bug --player-b scripted.
train/.venv/bin/python train/play.py --board gui "$@"
