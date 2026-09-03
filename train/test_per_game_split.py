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

import numpy as np  # noqa: E402

import runner  # noqa: E402
from runner import GameRecord, GameOutcome  # noqa: E402
from cli_spec import BINARY  # noqa: E402
from env import OBS_SIZE, MAX_ACTIONS, _SELF_IS_A_IDX, _IS_SIDEBOARD_IDX  # noqa: E402
from az_selfplay import _backfill_and_pack  # noqa: E402

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


# ----------------------------------------------------------------------
# n-step TD value targets (az_selfplay._backfill_and_pack)
# ----------------------------------------------------------------------
#
# The per-game outcome z is the OTHER half of the value target the shards carry;
# the n-step TD column td_q bootstraps off a later decision's search root value
# instead, sign-flipped when that decision's mover is the other seat. This is a
# synthetic match (no engine): a hand-built sample stream with known q values and
# a known winner, so every branch of the rule has an exactly predictable td_q.

TD_N = 3   # small horizon so a short synthetic game exercises the full window


def _td_sample(game_idx, mover_is_a, q, explored=0, sideboard=False):
    """One packed-sample dict in the shape _backfill_and_pack consumes."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[_SELF_IS_A_IDX] = 1.0 if mover_is_a else 0.0
    obs[_IS_SIDEBOARD_IDX] = 1.0 if sideboard else 0.0
    pi = np.zeros(MAX_ACTIONS, dtype=np.float32)
    pi[0] = 1.0
    mask = np.zeros(MAX_ACTIONS, dtype=bool)
    mask[:2] = True
    s = {"obs": obs, "pi": pi, "mask": mask, "mover_is_a": mover_is_a,
         "game_idx": game_idx, "explored": int(explored)}
    if q is not None:
        s["q"] = float(q)
    return s


def check_td_targets():
    """Every branch of the n-step rule, with exact expected td_q values.

    Game 0 (won by A) is a 9-sample stream with seats alternating A,B,A,... and
    one exploratory move at index 4, covering: full-horizon bootstrap WITH the
    seat flip, the shortened horizon at the explored barrier, the immediate
    block, and terminal-inside-the-horizon. Game 1 is preceded by two SIDEBOARD
    samples (their own chain) and won by B."""
    #        idx: 0    1    2    3    4*   5    6    7    8      (* = explored)
    q0 = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    seats0 = [True, False, True, False, True, False, True, False, True]
    samples = [_td_sample(0, seats0[i], q0[i], explored=(i == 4))
               for i in range(9)]
    # Game 1: two sideboard roots (seat A then B) then three in-game samples.
    samples += [_td_sample(1, True, 0.11, sideboard=True),
                _td_sample(1, False, 0.22, sideboard=True),
                _td_sample(1, True, 0.33),
                _td_sample(1, False, 0.44),
                _td_sample(1, True, 0.55)]
    game_winners = ["A", "B"]

    packed = _backfill_and_pack(samples, game_winners, td_n=TD_N)
    z, td = packed["z"], packed["td_q"]

    # z: mover-perspective outcome of the sample's own game.
    exp_z = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,   # game 0, A won
             -1.0, 1.0, -1.0, 1.0, -1.0]                        # game 1, B won
    if list(map(float, z)) != exp_z:
        raise SplitError(f"z mispriced: {list(map(float, z))} != {exp_z}")

    exp_td = [
        # 0: E=4, j_max=3, 0+3 <= 3 -> bootstrap q[3]; mover(3)=B != A -> NEGATE.
        -0.40,
        # 1: E=4, j_max=3, 1+3=4 > 3 -> shortened to j_max=3; mover(3)=B == B -> +.
        0.40,
        # 2: j_max=3 -> shortened to 3; mover(3)=B != A -> negate.
        -0.40,
        # 3: j_max=3 == t -> immediate block -> z.
        -1.0,
        # 4: E=inf (no later explored), j_max=L=8, 4+3=7 <= 8 -> bootstrap q[7];
        #    mover(7)=B != A -> negate.
        -0.80,
        # 5: 5+3=8 <= 8 -> bootstrap q[8]; mover(8)=A != B -> negate.
        -0.90,
        # 6: 6+3=9 > j_max=L=8 -> window reaches the game end -> z.
        1.0,
        # 7, 8: same, terminal in reach -> z.
        -1.0, 1.0,
        # 9, 10: SIDEBOARD chain -> always z (never bootstraps).
        -1.0, 1.0,
        # 11: game-1 in-game chain, L=13; 11+3=14 > 13 -> z.
        -1.0,
        # 12, 13: likewise z.
        1.0, -1.0,
    ]
    for i, (got, want) in enumerate(zip(map(float, td), exp_td)):
        if abs(got - want) > 1e-6:
            raise SplitError(f"td_q[{i}] = {got}, expected {want}")

    # A chain must never cross the game boundary: game 0's last samples resolve to
    # z, not to game 1's q values (already asserted above by exact value, but pin
    # the intent explicitly for the sideboard rows too).
    if float(td[8]) != float(z[8]) or float(td[9]) != float(z[9]):
        raise SplitError("chains crossed a game / sideboard boundary")

    # Expert-style rows: no q recorded at all -> td_q must fall back to z.
    exp_samples = [_td_sample(0, i % 2 == 0, None) for i in range(6)]
    ep = _backfill_and_pack(exp_samples, ["A"], td_n=TD_N)
    if not np.all(np.isnan(ep["q"])):
        raise SplitError("a sample recorded without a search must store q = NaN")
    if not np.array_equal(ep["td_q"], ep["z"]):
        raise SplitError(f"expert rows must keep td_q == z, got "
                         f"{ep['td_q']} vs {ep['z']}")
    if not np.all(np.isfinite(ep["td_q"])):
        raise SplitError("td_q must never be NaN")

    return (f"9+5 synthetic samples, td_n={TD_N}: seat-flip bootstrap, "
            f"shortened horizon, immediate block, terminal->z, sideboard->z, "
            f"expert->z")


def check_playout_cap():
    """The playout cap's coin and its shard contract.

    A fast-search row records no pi (all zeros) but a REAL root value q — so a
    full row whose bootstrap index lands on it must still bootstrap (a NaN q
    there would silently degrade the neighbour's target to z)."""
    from az_selfplay import playout_cap_full

    # Short-circuits: frac >= 1 is always full, frac <= 0 never.
    for ridx in range(50):
        if not playout_cap_full(123, ridx, 1.0):
            raise SplitError("frac=1.0 must always pick the full search")
        if playout_cap_full(123, ridx, 0.0):
            raise SplitError("frac=0.0 must never pick the full search")
    # Deterministic for a fixed (cap_seed, root_idx) and sensitive to both.
    picks = [playout_cap_full(7, i, 0.25) for i in range(4096)]
    if picks != [playout_cap_full(7, i, 0.25) for i in range(4096)]:
        raise SplitError("the coin must be deterministic per (seed, root)")
    if picks == [playout_cap_full(8, i, 0.25) for i in range(4096)]:
        raise SplitError("the coin must depend on cap_seed")
    # Empirical rate ~ frac (binomial 3-sigma band around 0.25 over 4096).
    rate = sum(picks) / len(picks)
    if not 0.23 <= rate <= 0.27:
        raise SplitError(f"frac=0.25 coin rate {rate:.3f} outside [0.23, 0.27]")

    # td bootstrap THROUGH a fast row: index 0's window (td_n=3) lands on
    # index 3 — a fast row (zero pi, finite q). Same-seat stream so no flip.
    fast = _td_sample(0, True, 0.75)
    fast["pi"] = np.zeros(MAX_ACTIONS, dtype=np.float32)
    samples = [_td_sample(0, True, 0.10), _td_sample(0, True, 0.20),
               _td_sample(0, True, 0.30), fast,
               _td_sample(0, True, 0.50), _td_sample(0, True, 0.60),
               _td_sample(0, True, 0.70)]
    packed = _backfill_and_pack(samples, ["A"], td_n=TD_N)
    if abs(float(packed["td_q"][0]) - 0.75) > 1e-6:
        raise SplitError(f"row 0 must bootstrap off the fast row's q=0.75, "
                         f"got {float(packed['td_q'][0])}")
    if float(packed["pi"][3].sum()) != 0.0:
        raise SplitError("the fast row's zero pi must survive packing")
    if float(packed["q"][3]) != 0.75:
        raise SplitError("the fast row must keep its real root value q")

    return (f"coin: frac 0/1 short-circuit, deterministic, rate {rate:.3f} "
            f"at frac=0.25; td bootstraps through a zero-pi fast row")


def main():
    checks = (check_seat_flip, check_merge_and_format, check_td_targets,
              check_playout_cap, check_live_bo3)
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
