"""Regression for shard_record.ShardRecorder — the play-session shard writer.

Torch-free and engine-free: drives a recorder with synthetic decisions shaped
like a bo3 match (searched opponent rows via the on_result signature, one-hot
human rows via the step observer, sideboard rows between games) and asserts

  * the written file passes the trainer-schema checklist
    (test_actor_shards' rules: exact key set, dtypes/shapes, pi/mask/z/
    explored/td_q invariants),
  * a MID-GAME flush is already a valid, readable shard whose unfinished-game
    rows price z = 0, and the game-boundary rewrite replaces (never
    duplicates) it,
  * both existing readers round-trip it: az_inspect.load_shard_sample and
    shard_replay.load_records (correct match/game segmentation and per-seat
    viewpoints),
  * a second match opens a second file.

Run: train/.venv/bin/python train/test_shard_record.py
"""

import glob
import os
import shutil
import sys
import tempfile

import numpy as np

import az_inspect
import shard_replay
from env import (MAX_ACTIONS, OBS_SIZE, _CUR_TURN_IDX, _IS_SIDEBOARD_IDX,
                 _MATCH_CTX_START, _SELF_IS_A_IDX)
from shard_record import ShardRecorder


class _FakeResult:
    """Duck-typed mcts.SearchResult: just what the recorder reads."""

    def __init__(self, visits, root_value):
        self.visits = np.asarray(visits, dtype=np.float64)
        self.root_value = float(root_value)

    def policy_target(self, temperature):
        return self.visits / self.visits.sum()

    def best_action(self):
        return int(np.argmax(self.visits))


def _obs(mover_is_a, game_number, turn, sideboard=False):
    o = np.zeros(OBS_SIZE, dtype=np.float32)
    o[_SELF_IS_A_IDX] = 1.0 if mover_is_a else 0.0
    o[_MATCH_CTX_START] = game_number / 3.0
    o[_CUR_TURN_IDX] = turn / 50.0
    o[_IS_SIDEBOARD_IDX] = 1.0 if sideboard else 0.0
    return o


def _searched(rec, obs, num, visits, root_value, chosen,
              reward=0.0, game_result=False, done=False):
    rec.on_search_result(obs, num, _FakeResult(visits, root_value), chosen)
    rec.observe_step(obs, num, chosen, reward,
                     {"game_result": game_result}, done)


def _onehot(rec, obs, num, action, reward=0.0, game_result=False, done=False):
    rec.observe_step(obs, num, action, reward,
                     {"game_result": game_result}, done)


def _assert_searched_row_matches_selfplay(rec, obs, num, visits, root_value,
                                          chosen):
    """The recorder's searched row == az_selfplay's own sample builder output,
    field for field (self-play then finalizes `explored` exactly as the
    recorder does). Stashes and discards the row; nothing is committed."""
    from az_selfplay import sample_from_search_result
    result = _FakeResult(visits, root_value)
    rec.on_search_result(obs, num, result, chosen)
    got, rec._pending = rec._pending, None
    want = sample_from_search_result(obs, num, result)
    want["explored"] = int(int(chosen) != int(result.best_action()))
    assert set(got) == set(want), (sorted(got), sorted(want))
    for k, v in want.items():
        g = got[k]
        if isinstance(v, np.ndarray):
            assert g.dtype == v.dtype and np.array_equal(g, v), k
        else:
            assert type(g) is type(v) and g == v, (k, g, v)


def _check_schema(path):
    from az_selfplay import SHARD_KEYS
    d = np.load(path)
    assert set(d.files) == set(SHARD_KEYS), d.files
    n = d["obs"].shape[0]
    assert d["obs"].shape == (n, OBS_SIZE) and d["obs"].dtype == np.float32
    assert d["pi"].shape == (n, MAX_ACTIONS) and d["pi"].dtype == np.float32
    assert d["mask"].shape == (n, MAX_ACTIONS) and d["mask"].dtype == np.bool_
    for k in ("z", "q", "td_q"):
        assert d[k].shape == (n,) and d[k].dtype == np.float32, k
    assert d["explored"].shape == (n,) and d["explored"].dtype == np.uint8
    assert set(np.unique(d["z"])) <= {-1.0, 0.0, 1.0}
    assert set(np.unique(d["explored"])) <= {0, 1}
    assert np.isfinite(d["td_q"]).all() and (np.abs(d["td_q"]) <= 1.0).all()
    for i in range(n):
        m = d["mask"][i]
        assert 2 <= m.sum() <= MAX_ACTIONS
        assert abs(d["pi"][i][m].sum() - 1.0) < 1e-5
        assert (d["pi"][i][~m] == 0.0).all()
    return d


def main():
    tmp = tempfile.mkdtemp(prefix="shard_record_test_")
    try:
        rec = ShardRecorder(tmp, td_n=3)

        # ── Match 0, game 0: A = the search opponent, B = the human ─────────
        # Two searched A rows (the second an EXPLORED off-argmax pick), one
        # human one-hot B row, one trivial (num==1) step that must NOT record.
        # A searched row must be EXACTLY what self-play would build from the
        # same (obs, num_choices, result) — the two share
        # az_selfplay.sample_from_search_result, and this pins that.
        _assert_searched_row_matches_selfplay(
            rec, _obs(True, 0, 1), 3, [8, 1, 1], 0.25, chosen=0)

        _searched(rec, _obs(True, 0, 1), 3, [8, 1, 1], 0.25, chosen=0)
        _onehot(rec, _obs(False, 0, 1), 4, action=2)
        _onehot(rec, _obs(False, 0, 2), 1, action=0)          # no row
        _searched(rec, _obs(True, 0, 2), 2, [5, 3], -0.10, chosen=1)

        # Mid-game flush: valid shard, all rows priced z = 0 (game unfinished).
        p = rec.flush()
        assert p is not None and os.path.basename(p).startswith("shard_")
        d = _check_schema(p)
        assert d["obs"].shape[0] == 3
        assert (d["z"] == 0.0).all()
        assert d["explored"].tolist() == [0, 0, 1]
        assert np.isnan(d["q"][1]) and not np.isnan(d["q"][0])

        # Game 0 ends: A wins (reward is Player-A-perspective).
        _onehot(rec, _obs(False, 0, 3), 2, action=0,
                reward=1.0, game_result=True)
        files = sorted(glob.glob(os.path.join(tmp, "shard_*.npz")))
        assert len(files) == 1, files                # rewritten, not duplicated
        d = _check_schema(files[0])
        assert d["obs"].shape[0] == 4
        # Mover-perspective z: A rows +1, B rows -1.
        assert d["z"].tolist() == [1.0, -1.0, 1.0, -1.0]

        # ── Sideboard rows, then game 1 (B wins), then game 2 (A wins match).
        _onehot(rec, _obs(True, 1, 0, sideboard=True), 3, action=1)
        _searched(rec, _obs(False, 1, 1), 2, [2, 6], 0.4, chosen=1)
        _searched(rec, _obs(True, 1, 2), 2, [6, 2], -0.3, chosen=0,
                  reward=-1.0, game_result=True)     # B wins game 1
        _searched(rec, _obs(True, 2, 1), 2, [7, 1], 0.6, chosen=0)
        _onehot(rec, _obs(False, 2, 2), 2, action=0,
                reward=1.0, game_result=True, done=True)   # A wins match

        files = sorted(glob.glob(os.path.join(tmp, "shard_*.npz")))
        assert len(files) == 1, files
        d = _check_schema(files[0])
        assert d["obs"].shape[0] == 9
        # Sideboard row (idx 4, mover A) is priced on the UPCOMING game 1,
        # which A lost.
        assert d["z"][4] == -1.0
        # Game-1 rows: B row +1, A row -1; game-2 rows: A +1, B -1.
        assert d["z"][5:].tolist() == [1.0, -1.0, 1.0, -1.0]

        # ── Readers ────────────────────────────────────────────────────────
        sample = az_inspect.load_shard_sample(tmp, max_rows=100)
        assert sample["obs"].shape[0] == 9 and "td_q" in sample

        matches = shard_replay.segment_matches(
            *(lambda o, p, z, m, s: (o, s))(*shard_replay.load_shard_rows(tmp)))
        assert len(matches) == 1 and len(matches[0]) == 3, [
            [len(g) for g in m] for m in matches]
        rec_a = shard_replay.load_records(tmp, viewpoint_is_a=True)
        rec_b = shard_replay.load_records(tmp, viewpoint_is_a=False)
        assert len(rec_a) == 1 and len(rec_b) == 1
        assert rec_a[0]["result"] == 1.0        # A took the match 2-1
        assert rec_b[0]["result"] == -1.0
        assert len(rec_a[0]["observations"]) == 5   # A's rows incl. sideboard
        assert len(rec_b[0]["observations"]) == 4

        # Trainer ingestion (skipped when torch isn't installed).
        try:
            import az_train
        except ImportError:
            print("  (az_train unavailable — trainer ingestion skipped)")
        else:
            w = az_train.load_window("gen", window=10, data_dir=tmp)
            assert w["obs"].shape[0] == 9

        # ── A second match opens a second file ─────────────────────────────
        _searched(rec, _obs(True, 0, 1), 2, [3, 3], 0.0, chosen=0,
                  reward=-1.0, done=True)            # bo1-style: no game_result
        rec.close()
        files = sorted(glob.glob(os.path.join(tmp, "shard_*.npz")))
        assert len(files) == 2, files
        d = _check_schema(files[1])
        assert d["obs"].shape[0] == 1 and d["z"][0] == -1.0

        # ── Replay sidecar: replay_meta + seed_fn make each match's shard
        # exactly replayable (engine seed + action log + row_decision_idx),
        # and shard_replay attaches them to the browsable records. ──────────
        tmp2 = tempfile.mkdtemp(prefix="shard_record_test_sidecar_")
        try:
            import json
            rec2 = ShardRecorder(
                tmp2, td_n=3,
                replay_meta={"deck_a": "league/ur_delver",
                             "deck_b": "league/uw_control",
                             "human_deck": "league/uw_control",
                             "opp_deck": "league/ur_delver",
                             "human_is_a": False, "bo3": False,
                             "opponent_spec": "az:gen", "binary": None},
                seed_fn=lambda: 12345)
            _searched(rec2, _obs(True, 0, 1), 3, [8, 1, 1], 0.25, chosen=0)
            _onehot(rec2, _obs(False, 0, 1), 4, action=2)
            _onehot(rec2, _obs(False, 0, 2), 1, action=0)   # 1-choice: no row,
            #                                                 but IS an action
            _searched(rec2, _obs(True, 0, 2), 2, [5, 3], -0.10, chosen=1,
                      reward=1.0, done=True)
            rec2.close()
            sides = sorted(glob.glob(os.path.join(tmp2, "shard_*.rmplay")))
            assert len(sides) == 1, sides
            with open(sides[0]) as f:
                doc = json.load(f)
            assert doc["engine_seed"] == 12345
            assert doc["actions"] == [0, 2, 0, 1]        # 1-choice step included
            assert doc["row_decision_idx"] == [0, 1, 3]  # rows skip it
            assert doc["in_progress"] is False
            recs = shard_replay.load_records(tmp2, viewpoint_is_a=True)
            assert len(recs) == 1
            r = recs[0]
            assert r["engine_seed"] == 12345
            assert r["full_actions"] == [0, 2, 0, 1]
            assert r["prefix_len"] == [0, 3]             # A's two rows
            assert r["replay_decks"] == {"deck_a": "league/ur_delver",
                                         "deck_b": "league/uw_control",
                                         "bo3": False}
            recs_b = shard_replay.load_records(tmp2, viewpoint_is_a=False)
            assert recs_b[0]["prefix_len"] == [1]        # B's one recorded row
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

        print("shard_record OK: schema, mid-game flush, z backfill, "
              "segmentation, both readers, per-match files, replay sidecar")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
