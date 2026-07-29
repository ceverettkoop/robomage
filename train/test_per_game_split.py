#!/usr/bin/env python3
"""Regression for the bo3 per-game-index win-rate split (runner.py).

An aggregate match W/L cannot say whether SIDEBOARDING helped: game 1 is played
on the registered decks and games 2-3 on sideboarded ones, and the match tally
pools them. ``runner`` therefore records a per-game outcome (winner + who was on
the play) and reports a per-index split.

The subtle part is that a raw pre-board vs post-board comparison is CONFOUNDED:
from game 2 the LOSER of the previous game chooses to play first, so whoever wins
game 1 is usually on the draw next. Measured on mirrors with agents that never
sideboard, that alone drags a ~63-75% game-1 rate down to ~46% post-board. The
report therefore also splits post-board results by seat, and these tests pin both
the bookkeeping and that invariant.

Torch-free. The engine case needs bin/robomage built (like test_snapshot.py).

Runnable standalone::

    train/.venv/bin/python train/test_per_game_split.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runner  # noqa: E402
from runner import GameRecord, GameOutcome  # noqa: E402
from cli_spec import BINARY  # noqa: E402

DECK_A, DECK_B = "league/ur_delver", "league/gw_maverick"
N_MATCHES = 6
SEED = 5


class SplitError(AssertionError):
    """A per-game-split invariant was violated."""


def _rec(*outcomes):
    return GameRecord(0.0, 1, False, [], None, list(outcomes))


def check_seat_flip():
    """Both seats must agree on who was on the play, and on who won.

    ``tally_per_game(flip=True)`` scores for Player B, which is what `baseline`
    needs because the model alternates seats between matches. Getting the flip
    wrong would silently invert the on-the-play column — the one number the whole
    report exists to control for."""
    # A wins g1 on the play; the bo3 rule puts the LOSER (B) on the play for g2.
    rec = _rec(GameOutcome("A", True), GameOutcome("B", False))
    a = runner.tally_per_game([rec])
    b = runner.tally_per_game([rec], flip=True)

    for gi in (0, 1):
        if a[gi]["played"] != 1 or b[gi]["played"] != 1:
            raise SplitError(f"game {gi}: both views must count it once")
        if a[gi]["wins"] + b[gi]["wins"] != 1:
            raise SplitError(f"game {gi}: exactly one seat won it, got "
                             f"A={a[gi]['wins']} B={b[gi]['wins']}")
        # Exactly one seat was on the play, and the two views must not agree that
        # it was the same one.
        if a[gi]["play_n"] + b[gi]["play_n"] != 1:
            raise SplitError(
                f"game {gi}: exactly one seat is on the play, got "
                f"A={a[gi]['play_n']} B={b[gi]['play_n']}")
    if a[0]["play_wins"] != 1:
        raise SplitError("A won g1 while on the play; play_wins should be 1")
    if b[1]["play_wins"] != 1:
        raise SplitError("B won g2 while on the play; play_wins should be 1")
    return "seat flip consistent"


def check_merge_and_format():
    """merge_per_game accumulates, and the formatter surfaces the confound."""
    dest = {}
    runner.merge_per_game(dest, runner.tally_per_game([
        _rec(GameOutcome("A", True), GameOutcome("B", False))]))
    runner.merge_per_game(dest, runner.tally_per_game([
        _rec(GameOutcome("A", True), GameOutcome("A", False))]))
    if dest[0]["played"] != 2 or dest[0]["wins"] != 2:
        raise SplitError(f"merge lost games: {dest[0]}")
    if dest[1]["played"] != 2 or dest[1]["wins"] != 1:
        raise SplitError(f"merge lost post-board games: {dest[1]}")

    lines = runner.format_per_game_split(dest)
    if len(lines) != 2:
        raise SplitError(f"expected a per-index line and a by-seat line, got {lines}")
    if "confounded" not in lines[1]:
        raise SplitError("the by-seat line must flag the play/draw confound, or a "
                         "reader will take a post-board drop for a sideboard result")

    # bo1 records carry no per-game breakdown -> nothing to report.
    if runner.format_per_game_split(runner.tally_per_game([_rec()])) != []:
        raise SplitError("a bo1 record must produce no split lines")
    return "merge + format OK"


def check_live_bo3():
    """Drive real bo3 matches and pin the bookkeeping against the engine.

    Game 1's starting player is fixed by play_bo3_match (Player A goes first in
    game 1 of every match), so from A's view game 0 must be ALL on-the-play — a
    cheap, strong check that the on-the-play flag is read from the right
    perspective rather than being constant or inverted."""
    if not os.path.exists(BINARY):
        raise SplitError(f"binary not found at {BINARY} — run `make` first")
    result = runner.run_match("scripted", "scripted", deck_a=DECK_A, deck_b=DECK_B,
                              games=N_MATCHES, bo3=True, seed=SEED,
                              transcript="quiet")
    per_game = runner.tally_per_game(result.records)
    if not per_game:
        raise SplitError("a bo3 run produced no per-game breakdown")
    if per_game[0]["played"] != N_MATCHES:
        raise SplitError(f"every match plays exactly one game 1; got "
                         f"{per_game[0]['played']} over {N_MATCHES} matches")
    if per_game[0]["draw_n"] != 0:
        raise SplitError(
            f"Player A is the starting player in game 1 of every bo3 match, but "
            f"{per_game[0]['draw_n']} game-1 result(s) say A was on the draw — the "
            "on-the-play flag is being read from the wrong perspective")
    total_games = sum(c["played"] for c in per_game.values())
    if not (2 * N_MATCHES <= total_games <= 3 * N_MATCHES):
        raise SplitError(f"{total_games} games over {N_MATCHES} bo3 matches is "
                         "outside the possible 2-3 per match")
    for gi, cell in per_game.items():
        if cell["play_n"] + cell["draw_n"] != cell["played"]:
            raise SplitError(f"game {gi}: play/draw counts {cell['play_n']}+"
                             f"{cell['draw_n']} != played {cell['played']}")
    return (f"{N_MATCHES} matches, {total_games} games, per-index "
            + " ".join(f"g{gi+1}={c['wins']}/{c['played']}"
                       for gi, c in sorted(per_game.items())))


def main():
    checks = (check_seat_flip, check_merge_and_format, check_live_bo3)
    for fn in checks:
        try:
            detail = fn()
        except SplitError as e:
            print(f"FAIL  {fn.__name__}\n  {e}", flush=True)
            return 1
        print(f"ok    {fn.__name__}: {detail}", flush=True)
    print("\nper-game split: all checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
