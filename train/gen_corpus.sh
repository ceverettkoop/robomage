#!/usr/bin/env bash
# Generate the T2.1 layer-refactor regression corpus.
#
# Runs every ordered pairing of the three fully-implemented decks across a dozen
# seeds and captures the full `observe --verbose` transcript per game. The corpus
# is generated once on clean `main` (baseline) and again after the refactor; the
# two trees are diffed byte-for-byte to prove the layer-system re-architecture is
# behavior-preserving for the current card vocab.
#
# Usage:  train/gen_corpus.sh <output_dir>
# Must be run from the repo root (resources are resolved relative to cwd).
set -u

OUTDIR="${1:?usage: train/gen_corpus.sh <output_dir>}"
PY=train/.venv/bin/python
DECKS=(delver doomsday mav)
SEEDS=$(seq 1 12)

mkdir -p "$OUTDIR"

for a in "${DECKS[@]}"; do
  for b in "${DECKS[@]}"; do
    for s in $SEEDS; do
      out="$OUTDIR/${a}_vs_${b}_seed${s}.log"
      "$PY" train/train.py observe --deck "$a" --opponent "$b" --seed "$s" --verbose > "$out" 2>&1
      # Flag any game that did not end with a decisive result (draws/errors are unacceptable).
      if ! grep -q "=== Scripted/. wins ===" "$out"; then
        echo "NON-DECISIVE: $out"
      fi
    done
  done
done

echo "corpus written to $OUTDIR ($(ls "$OUTDIR" | wc -l) files)"
