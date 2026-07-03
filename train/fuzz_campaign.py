#!/usr/bin/env python3
"""Batch fuzzing driver: run N scripted games for one matchup and dump the
verbose transcript to a file for bug/rules-deviation review.

Unlike ``test_harness.py`` (one game, stdout) this loops many games in one
process with per-game seed increments, driving BOTH seats with the coverage
``explore`` fuzzer (independent per-seat novelty state). It is the tool behind
the league fuzz campaigns: one invocation per (matchup, mode), the resulting
transcript file is then reviewed for engine bugs / rules deviations.

Modes (``--mode``) map to scripted-agent tiers (see train/scripted_agent.py):
  explore          — default coverage fuzzer (broad, uniform-ish action mix)
  explore:patient  — big-mana profile: develops mana, holds expensive cards
                     until castable (surfaces late-game / expensive-card lines)

Usage:
  train/.venv/bin/python train/fuzz_campaign.py \
      --deck-a ur_delver --deck-b gw_maverick \
      --mode explore --games 100 --seed 1 \
      --out /tmp/fuzz_delver_maverick_explore.txt

Decks are resolved relative to bin/resources/decks/ (e.g. league/ur_delver).
The verbose transcript (all game narrative + decoded state + action menus) goes
to --out; the one-line W/L/D summary is printed to stdout. Draws (step-cap
stalls) are also saved to draw_<n>.txt by run_games and are worth reviewing.
"""
import argparse
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runner
from opponents import make_controller


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deck-a", required=True, help="Player A deck name (relative to decks/)")
    p.add_argument("--deck-b", required=True, help="Player B deck name (relative to decks/)")
    p.add_argument("--mode", default="explore",
                   help="Scripted fuzzer tier: 'explore' (default) or 'explore:patient'")
    p.add_argument("--games", type=int, default=100, help="Number of games to run")
    p.add_argument("--seed", type=int, default=1, help="Base seed (game i uses seed+i)")
    p.add_argument("--max-decisions", type=int, default=None,
                   help="Optional per-game decision cap (default: run to completion)")
    p.add_argument("--out", required=True, help="Transcript output file")
    p.add_argument("--binary", default=str(runner.BINARY), help="Path to robomage binary")
    args = p.parse_args()

    # Independent explore controllers per seat so each seat's novelty set is its
    # own (better coverage than sharing one object across both seats).
    ctrl_a = make_controller(args.mode)
    ctrl_b = make_controller(args.mode)

    with open(args.out, "w") as fh, contextlib.redirect_stdout(fh):
        wins, losses, draws = runner.run_games(
            ctrl_a, ctrl_b,
            label_a=f"A:{args.mode}", label_b=f"B:{args.mode}",
            binary_path=args.binary, deck_a=args.deck_a, deck_b=args.deck_b,
            n_games=args.games, seed=args.seed, verbose=True,
            max_decisions=args.max_decisions)

    total = wins + losses + draws
    print(f"{args.deck_a} vs {args.deck_b} [{args.mode}] "
          f"{args.games} games: {wins}W / {losses}L / {draws}D "
          f"(completed {total}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
