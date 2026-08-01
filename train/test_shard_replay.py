#!/usr/bin/env python3
"""Regression for shard_replay.py: shards on disk -> browsable match records.

Torch-free end-to-end check. Drives two scripted bo3 matches through the real
engine (RoboMageEnv + runner.drive_game — the same loop every game-running tool
uses), recording every decision's full observation, menu width, chosen action,
and per-game winners. The recordings are packed into synthetic ``shard_*.npz``
files that replicate az_selfplay's format exactly (one-hot pi at the chosen
action, so the argmax reconstruction is exact everywhere; z backfilled per game
with the sideboard-rows-price-the-NEXT-game convention). Then asserts:

  1. ``shard_sort_key`` orders equal-mtime files by their numeric n suffix.
  2. ``load_shard_rows`` concatenates in write order with correct file spans.
  3. ``segment_matches`` recovers the exact per-match / per-game row structure
     (game_number constant within a game segment; sideboard rows attached to
     the finished game; segmentation never crosses a file span).
  4. ``build_match_records`` emits the analysis-trace schema: steps are the
     viewpoint seat's rows in order, actions/num_choices match the recording,
     pi round-trips as action_probs, z-based values, match result matches the
     engine's game outcomes, the other seat's rows land in opp_actions.
  5. Every step decodes (decode_game_state + decode_actions_from_obs).
  6. Seat B viewpoint mirrors seat A (complementary steps, negated result).

Run: train/.venv/bin/python train/test_shard_replay.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import decode
import runner
import shard_replay
from env import (MAX_ACTIONS, OBS_SIZE, STATE_SIZE, RoboMageEnv,
                 _MATCH_CTX_START, _IS_SIDEBOARD_IDX, _SELF_IS_A_IDX)
from opponents import make_controller

DECK = "league/ur_delver"       # league deck: has a SIDEBOARD (bo3 matches)
SEEDS = (11, 12)                # one engine seed per recorded match
MAX_DECISIONS = 2000            # safety cap only; a scripted match ends well under


def _record_match(engine_seed):
    """Play one scripted bo3 mirror match; return (rows, game_winners) where
    rows = [(obs_copy, num_choices, action, mover_is_a), ...] in decision order
    and game_winners = ["A"|"B"|None per game]."""
    env = RoboMageEnv(deck_a=DECK, deck_b=DECK, bo3=True)
    rows = []

    def on_query(d):
        rows.append([d.obs.copy(), d.num_choices, None, d.priority_is_a])

    def on_action(d, action):
        rows[-1][2] = int(action)

    try:
        obs, _ = env.reset(options={"engine_seed": engine_seed})
        ctrl = make_controller("scripted")
        rec = runner.drive_game(env, obs, ctrl, ctrl,
                                on_query=on_query, on_action=on_action,
                                max_decisions=MAX_DECISIONS)
    finally:
        env.close()
    assert not rec.capped, "scripted match hit the decision cap"
    winners = [g.winner for g in rec.game_results]
    assert winners, "bo3 match produced no per-game results"
    return rows, winners


def _pack_shard(rows, winners):
    """Replicate az_selfplay's shard arrays from a recorded match: one-hot pi
    at the chosen action; z per row from its game's winner in the MOVER's
    perspective. Row -> game assignment mirrors _play_match's game_idx counter,
    which is simply the obs game_number: the engine stamps sideboard prompts
    with the UPCOMING game's number, the same game whose outcome prices them."""
    n = len(rows)
    obs = np.zeros((n, OBS_SIZE), dtype=np.float32)
    pi = np.zeros((n, MAX_ACTIONS), dtype=np.float32)
    z = np.zeros((n,), dtype=np.float32)
    mask = np.zeros((n, MAX_ACTIONS), dtype=bool)
    for i, (o, nc, action, mover_is_a) in enumerate(rows):
        obs[i] = o
        pi[i, action] = 1.0
        mask[i, :nc] = True
        game_idx = int(round(float(o[_MATCH_CTX_START]) * 3.0))
        winner = winners[game_idx] if game_idx < len(winners) else None
        if winner is not None:
            z[i] = 1.0 if (winner == "A") == mover_is_a else -1.0
    return obs, pi, z, mask


def _check_sort_key(tmp):
    a = os.path.join(tmp, "shard_20260801_000000_1_2.npz")
    b = os.path.join(tmp, "shard_20260801_000000_1_10.npz")
    for p in (a, b):
        np.savez_compressed(p, x=np.zeros(1))
        os.utime(p, (1000, 1000))   # equal mtimes: numeric n must break the tie
    got = sorted([b, a], key=shard_replay.shard_sort_key)
    assert got == [a, b], f"sort key misordered equal-mtime shards: {got}"
    for p in (a, b):
        os.remove(p)
    print("  sort key: OK")


def _check_segments(matches, per_file_rows, obs):
    assert len(matches) == len(per_file_rows), \
        f"expected {len(per_file_rows)} matches, segmented {len(matches)}"
    row = 0
    for mi, (games, match_rows) in enumerate(zip(matches, per_file_rows)):
        n_rows = sum(len(g) for g in games)
        assert n_rows == len(match_rows), \
            f"match {mi}: {n_rows} segmented rows != {len(match_rows)} recorded"
        assert [i for g in games for i in g] == list(range(row, row + n_rows)), \
            f"match {mi}: segmented rows not contiguous in write order"
        row += n_rows
        for gi, g_rows in enumerate(games):
            gns = {int(round(float(obs[i, _MATCH_CTX_START]) * 3.0)) for i in g_rows}
            assert gns == {gi}, \
                f"match {mi} game {gi}: mixed game_numbers {sorted(gns)}"
            sb = [bool(obs[i, _IS_SIDEBOARD_IDX] > 0.5) for i in g_rows]
            n_sb = sum(sb)
            assert all(sb[:n_sb]) and not any(sb[n_sb:]), \
                f"match {mi} game {gi}: sideboard rows not a leading prefix"
            assert gi > 0 or n_sb == 0, "game 0 cannot have sideboard rows"
    print(f"  segmentation: OK ({len(matches)} matches, "
          f"{[len(g) for g in matches]} games)")


def _check_records(records, matches, obs, pi, z, mask, winners_per_match,
                   viewpoint_is_a):
    label = "A" if viewpoint_is_a else "B"
    assert len(records) == len(matches), \
        f"seat {label}: {len(records)} records for {len(matches)} matches"
    for mi, (rec, games) in enumerate(zip(records, matches)):
        all_rows = [i for g in games for i in g]
        vp = [i for i in all_rows
              if bool(obs[i, _SELF_IS_A_IDX] > 0.5) == viewpoint_is_a]
        opp = [i for i in all_rows if i not in set(vp)]
        assert len(rec["observations"]) == len(vp), \
            f"seat {label} match {mi}: {len(rec['observations'])} steps != {len(vp)}"
        assert len(rec["opp_actions"]) == len(opp)
        assert rec["model_is_a"] == viewpoint_is_a
        assert rec["engine_seed"] is None and rec["full_actions"] is None
        for k, i in enumerate(vp):
            assert np.array_equal(rec["observations"][k], obs[i])
            nc = int(mask[i].sum())
            assert rec["num_choices"][k] == nc
            assert rec["actions"][k] == int(np.argmax(pi[i]))
            assert np.allclose(rec["action_probs"][k], pi[i, :nc])
            assert rec["values"][k] == float(z[i])
        before = [oa["before_model_step"] for oa in rec["opp_actions"]]
        assert before == sorted(before), "opp_actions out of step order"
        # Match result: majority of per-game winners, from the viewpoint seat.
        won = sum(1 for w in winners_per_match[mi] if w == label)
        lost = sum(1 for w in winners_per_match[mi] if w and w != label)
        expect = 0.0 if won == lost else (1.0 if won > lost else -1.0)
        assert rec["result"] == expect, \
            f"seat {label} match {mi}: result {rec['result']} != {expect}"
    print(f"  records seat {label}: OK")


def _check_decode(records):
    n_dec = 0
    for rec in records:
        for k, o in enumerate(rec["observations"]):
            gs = decode.decode_game_state(o[:STATE_SIZE])
            assert np.isfinite(gs["self"]["life"]) and gs["turn"] >= 1
            acts = decode.decode_actions_from_obs(o, rec["num_choices"][k])
            assert len(acts) == rec["num_choices"][k]
            assert all(a["description"] for a in acts)
            n_dec += 1
    print(f"  decode: OK ({n_dec} steps)")


def main():
    print(f"Recording {len(SEEDS)} scripted bo3 matches ({DECK} mirror)...")
    recorded = [_record_match(s) for s in SEEDS]
    with tempfile.TemporaryDirectory() as tmp:
        _check_sort_key(tmp)
        for k, (rows, winners) in enumerate(recorded):
            arrays = _pack_shard(rows, winners)
            path = os.path.join(tmp, f"shard_20260801_00000{k}_1_{k}.npz")
            np.savez_compressed(path, obs=arrays[0], pi=arrays[1],
                                z=arrays[2], mask=arrays[3])
            os.utime(path, (2000 + k, 2000 + k))  # pin write order
        obs, pi, z, mask, spans = shard_replay.load_shard_rows(tmp)
        assert obs.shape == (sum(len(r) for r, _ in recorded), OBS_SIZE)
        assert [e - s for _, s, e in spans] == [len(r) for r, _ in recorded]
        matches = shard_replay.segment_matches(obs, spans)
        _check_segments(matches, [r for r, _ in recorded], obs)
        winners_per_match = [w for _, w in recorded]
        for viewpoint_is_a in (True, False):
            records = shard_replay.build_match_records(
                obs, pi, z, mask, matches, viewpoint_is_a=viewpoint_is_a)
            _check_records(records, matches, obs, pi, z, mask,
                           winners_per_match, viewpoint_is_a)
            if viewpoint_is_a:
                _check_decode(records)
        # The one-call front door + limit.
        assert len(shard_replay.load_records(tmp, limit=1)) == 1
    print("PASS: shard_replay round-trip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
