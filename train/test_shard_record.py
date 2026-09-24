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
  * a second match opens a second file,
  * a capped load_records reads only the shards its matches need and equals
    the uncapped load's prefix; an uncapped load of an oversized directory is
    refused,
  * the browser net probes' π is the SEARCH posterior (diag visits / a pool
    shard's search π), never a behavior row or a simulated trace's
    action_probs, and survives an .rmtrace round-trip; az_inspect's own
    shard sample marks the same rows (``pi_valid``),
  * the shared search-vs-net divergence folds duplicate menu actions and
    measures KL(search ‖ net) (test_menu_merge.test_search_net_divergence).

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
from shard_record import (DIAG_KEYS, DIAG_KIND_FOLLOWED, DIAG_KIND_NONE,
                          DIAG_KIND_SEARCH, ShardRecorder, diag_path_for)


class _FakeResult:
    """Duck-typed mcts.SearchResult: just what the recorder reads (the row
    builders' fields plus the .diag sidecar's diagnostics)."""

    def __init__(self, visits, root_value, q=None, worlds=2):
        self.visits = np.asarray(visits, dtype=np.float64)
        self.root_value = float(root_value)
        self.q = None if q is None else np.asarray(q, dtype=np.float64)
        n = len(self.visits)
        self.priors = np.full(n, 1.0 / n)
        self.w_sum = self.visits * self.root_value
        self.sims_run = int(self.visits.sum())
        self.sim_steps = 7 * self.sims_run
        self.reused_visits = 3
        self.stopped_early = True
        self.memo_hits = 2
        self.seeds = [100 + w for w in range(worlds)]
        self.world_visits = [int(self.visits.sum()) // worlds] * worlds
        self.world_values = [self.root_value] * worlds
        self.roots = [object()] * worlds       # non-empty => in-game search

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
                                          chosen, q=None):
    """The recorder's searched row == az_selfplay's own sample builder output,
    field for field (self-play finalizes `explored`/`q` through the same
    :func:`finalize_searched_sample`). Stashes and discards the row; nothing
    is committed."""
    from az_selfplay import finalize_searched_sample, sample_from_search_result
    result = _FakeResult(visits, root_value, q=q)
    rec.on_search_result(obs, num, result, chosen)
    got, rec._pending = rec._pending, None
    want = sample_from_search_result(obs, num, result)
    finalize_searched_sample(want, result, chosen)
    assert set(got) == set(want), (sorted(got), sorted(want))
    for k, v in want.items():
        g = got[k]
        if isinstance(v, np.ndarray):
            assert g.dtype == v.dtype and np.array_equal(g, v), k
        else:
            assert type(g) is type(v) and g == v, (k, g, v)
    if result.q is not None:
        assert got["q"] == float(result.q[int(chosen)])
    else:
        assert got["q"] == float(root_value)


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


def _check_diag(shard_path, n_rows):
    """The .diag sidecar beside a shard: exact key set, fixed dtypes/shapes,
    one row per shard row."""
    p = diag_path_for(shard_path)
    assert os.path.isfile(p), p
    assert not p.endswith(".npz")      # must dodge the shard_*.npz globs
    d = np.load(p)
    assert set(d.files) == set(DIAG_KEYS), d.files
    n = d["kind"].shape[0]
    assert n == n_rows, (n, n_rows)
    w = d["world_seeds"].shape[1]
    want = {
        "kind": (np.int8, (n,)),
        "visits": (np.int32, (n, MAX_ACTIONS)),
        "priors": (np.float32, (n, MAX_ACTIONS)),
        "q_act": (np.float32, (n, MAX_ACTIONS)),
        "w_sum": (np.float32, (n, MAX_ACTIONS)),
        "root_value": (np.float32, (n,)),
        "sims_run": (np.int32, (n,)),
        "sim_steps": (np.int32, (n,)),
        "reused_visits": (np.int32, (n,)),
        "memo_hits": (np.int32, (n,)),
        "stopped_early": (np.uint8, (n,)),
        "time_budget_s": (np.float32, (n,)),
        "time_budget_min_s": (np.float32, (n,)),
        "n_worlds": (np.int16, (n,)),
        "world_seeds": (np.int64, (n, w)),
        "world_visits": (np.int32, (n, w)),
        "world_values": (np.float32, (n, w)),
        "origin_row": (np.int32, (n,)),
        "follow_worlds": (np.uint8, (n, w)),
    }
    for k, (dt, shape) in want.items():
        assert d[k].dtype == dt and d[k].shape == shape, (k, d[k].dtype,
                                                          d[k].shape)
    assert d["follow_path"].dtype == np.int32 and d["follow_path"].ndim == 2 \
        and d["follow_path"].shape[0] == n and d["follow_path"].shape[1] >= 1
    return d


def _close(a, b):
    return a is not None and np.allclose(np.asarray(a), np.asarray(b))


def _check_search_posterior(ra, rb):
    """The net probes' π is the SEARCH posterior (shard_probes.step_search_pi):
    diag visits where a search / tree-follow ran, the recorded π for a
    sidecar-less pool-shard search row, never a behavior row's one-hot and
    never a simulated trace's action_probs (the inspection net's own
    softmax). Rows without one are excluded from the π views with a note."""
    import gui_session_io
    import shard_probes

    # Recorded match (A = searcher, B = human): shard rows 0,2,4,7,8 / 1,3,5,6,9.
    sp = ra["search_pi"]
    assert _close(sp[0], [0.8, 0.1, 0.1]) and _close(sp[1], [5 / 8, 3 / 8])
    assert sp[2] is None                  # one-hot sideboard pick (q = NaN)
    assert _close(sp[3], [0.75, 0.25]) and _close(sp[4], [7 / 8, 1 / 8])
    sp = rb["search_pi"]
    assert sp[0] is None and sp[1] is None and sp[4] is None   # human rows
    assert _close(sp[2], [0.25, 0.75])
    # The followed row's shard pi is the one-hot behavior row; its posterior
    # is the followed subtree's visits from the diag.
    assert _close(sp[3], np.array([40, 2, 1]) / 43.0)

    # A sidecar-less POOL shard: no diags, so a finite-q row's recorded pi IS
    # the posterior; q = NaN (expert BC) and a one-hot sideboard row are not.
    obs = np.stack([_obs(True, 0, 1), _obs(True, 0, 2),
                    _obs(True, 1, 0, sideboard=True), _obs(True, 1, 1)])
    pi = np.zeros((4, MAX_ACTIONS), dtype=np.float32)
    pi[0, :2] = [0.3, 0.7]
    pi[1, 1] = 1.0
    pi[2, 0] = 1.0
    pi[3, :2] = [0.0, 0.0]                # playout-cap fast row
    mask = np.zeros((4, MAX_ACTIONS), dtype=bool)
    mask[:, :2] = True
    z = np.array([1, 1, -1, -1], dtype=np.float32)
    q = np.array([0.2, np.nan, 0.1, 0.0], dtype=np.float32)
    recs = shard_replay.build_match_records(obs, pi, z, mask, [[[0, 1, 2, 3]]],
                                            viewpoint_is_a=True, q=q)
    sp = recs[0]["search_pi"]
    assert _close(sp[0], [0.3, 0.7]) and sp[1] is None and sp[2] is None \
        and sp[3] is None, sp

    # A SIMULATED trace: action_probs are the inspection net's softmax and
    # must not be π. Step 0 searched (merged duplicate: visits live on the
    # representative index 0, index 1 stays 0), step 1 raw policy (no diag),
    # step 2 tree-followed.
    sim = {"observations": [_obs(True, 0, 1), _obs(True, 0, 2), _obs(True, 0, 3)],
           "num_choices": [3, 2, 2],
           "action_probs": [np.array([0.1, 0.1, 0.8]), np.array([0.5, 0.5]),
                            np.array([0.9, 0.1])],
           "diag": [{"kind": DIAG_KIND_SEARCH, "visits": np.array([6, 0, 2])},
                    None,
                    {"kind": DIAG_KIND_FOLLOWED, "visits": np.array([1, 3])}],
           "result": 1.0}
    snap = shard_probes.snapshot([sim, ra, rb], cur_game=0, cur_step=1)
    sample, index, z_valid, pi_valid = shard_probes.build_sample(snap)
    assert pi_valid.tolist() == [True, False, True,
                                 True, True, False, True, True,
                                 False, False, True, True, False]
    assert np.allclose(sample["pi"][0, :3], [0.75, 0.0, 0.25])
    assert sample["pi"][1].sum() == 0.0
    assert np.allclose(sample["pi"][2, :2], [0.25, 0.75])
    assert z_valid.all()                  # calibration keeps every row

    # π-dependent probes say what they skipped. No net needed on these paths.
    raw_only = shard_probes.snapshot([{**sim, "diag": [None, None, None]}],
                                     cur_game=0, cur_step=0)
    lines = shard_probes.run_probe("probe_kl", None, raw_only)
    assert lines[0].startswith("no browsed decision carries a search "
                               "posterior"), lines
    assert "3 of 3 browsed decisions skipped" in lines[-1], lines
    lines = shard_probes.run_probe("probe_state", None, snap)   # step 1: raw
    assert lines[0].startswith("(no search posterior at this decision"), lines
    assert not any("50.0%" in ln for ln in lines), lines
    lines = shard_probes.run_probe("probe_state", None,
                                   {**snap, "sel": (0, 0)})
    assert any(ln.lstrip().startswith("75.0%") for ln in lines), lines

    # The .rmtrace keeps the resolved posterior (the diag dicts don't survive
    # a save/load), so a reloaded session probes the same π.
    tmp = tempfile.mkdtemp(prefix="shard_record_test_trace_")
    try:
        path = os.path.join(tmp, "s.rmtrace")
        gui_session_io.save_traces(path, [sim, ra])
        games, _meta = gui_session_io.load_traces(path)
        _s2, _i2, _z2, pv2 = shard_probes.build_sample(
            shard_probes.snapshot(games))
        assert pv2.tolist() == pi_valid[:8].tolist()
        assert _close(games[0]["search_pi"][0], [0.75, 0.0, 0.25])
        assert games[0]["search_pi"][1] is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _same(a, b):
    """Deep equality over record values (dicts, lists, numpy arrays, NaN-aware
    floats)."""
    if isinstance(a, dict):
        return (isinstance(b, dict) and a.keys() == b.keys()
                and all(_same(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return (isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(_same(x, y) for x, y in zip(a, b)))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return a == b


def _check_bounded_load():
    """shard_replay.load_records(limit=k) reads only the shards its first k
    matches need and returns exactly the uncapped load's records[:k] — over
    recorder files (one match each, with .rmplay/.diag sidecars), a match
    with no viewpoint-A rows, and a pooled file holding two matches — and an
    uncapped load past MAX_UNBOUNDED_SHARD_BYTES is refused."""
    tmp = tempfile.mkdtemp(prefix="shard_record_test_bounded_")
    try:
        rec = ShardRecorder(tmp, td_n=3,
                            replay_meta={"deck_a": "a", "deck_b": "b",
                                         "human_is_a": False, "bo3": True},
                            seed_fn=lambda: 7)
        # Match 0: bo3, both seats.
        _searched(rec, _obs(True, 0, 1), 3, [8, 1, 1], 0.25, chosen=0)
        _onehot(rec, _obs(False, 0, 1), 2, action=1,
                reward=1.0, game_result=True)
        _searched(rec, _obs(False, 1, 1), 2, [2, 6], 0.4, chosen=1)
        _searched(rec, _obs(True, 1, 2), 2, [6, 2], -0.3, chosen=0,
                  reward=1.0, game_result=True, done=True)
        # Match 1: only B moves (no viewpoint-A record).
        _searched(rec, _obs(False, 0, 1), 2, [3, 5], 0.1, chosen=1,
                  reward=-1.0, done=True)
        # Match 2: both seats.
        _onehot(rec, _obs(False, 0, 1), 3, action=2)
        _searched(rec, _obs(True, 0, 2), 2, [1, 7], 0.2, chosen=1,
                  reward=-1.0, done=True)
        rec.close()
        files = sorted(glob.glob(os.path.join(tmp, "shard_*.npz")),
                       key=shard_replay.shard_sort_key)
        assert len(files) == 3, files
        # A pooled (sidecar-less) file holding two matches, written last.
        a, b = np.load(files[0]), np.load(files[2])
        pooled = os.path.join(tmp, "shard_29990101_000000_1_0.npz")
        np.savez_compressed(pooled, **{k: np.concatenate([a[k], b[k]])
                                       for k in a.files})
        for k, f in enumerate(files + [pooled]):
            os.utime(f, (3000 + k, 3000 + k))       # pin write order
        for vp in (True, False):
            full = shard_replay.load_records(tmp, viewpoint_is_a=vp)
            assert len(full) == (4 if vp else 5), len(full)
            for k in range(1, len(full) + 2):
                part = shard_replay.load_records(tmp, viewpoint_is_a=vp,
                                                 limit=k)
                assert _same(part, full[:k]), f"limit {k} seat A={vp}"
        # The cap stops reading at the shard that completes the k-th match.
        spans = shard_replay.load_shard_rows(tmp, max_matches=1)[4]
        assert len(spans) == 1, spans
        spans = shard_replay.load_shard_rows(tmp, max_matches=2)[4]
        assert len(spans) == 3, spans      # match 1 has no viewpoint-A rows
        # Uncapped past the guard: refused, naming --games; capped still loads.
        saved = shard_replay.MAX_UNBOUNDED_SHARD_BYTES
        shard_replay.MAX_UNBOUNDED_SHARD_BYTES = 1
        try:
            for lim in (None, 0):
                try:
                    shard_replay.load_records(tmp, limit=lim)
                except shard_replay.ShardPoolTooLarge as exc:
                    assert "--games" in str(exc), exc
                else:
                    raise AssertionError("uncapped load was not refused")
            assert len(shard_replay.load_records(tmp, limit=2)) == 2
        finally:
            shard_replay.MAX_UNBOUNDED_SHARD_BYTES = saved
        print("  bounded shard load: OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
        # With per-action Q on the result, the row's q is the PLAYED action's
        # Q, not the visit-weighted root value.
        _assert_searched_row_matches_selfplay(
            rec, _obs(True, 0, 1), 3, [8, 1, 1], 0.25, chosen=1,
            q=[0.4, -0.2, 0.1])

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
        # Diag sidecar: searched rows carry the full root facts, the one-hot
        # human row is kind 0 with nothing else.
        g = _check_diag(p, 3)
        assert g["kind"].tolist() == [DIAG_KIND_SEARCH, DIAG_KIND_NONE,
                                      DIAG_KIND_SEARCH]
        assert g["visits"][0][:3].tolist() == [8, 1, 1]
        assert g["visits"][0][3:].sum() == 0 and g["visits"][1].sum() == 0
        assert g["visits"][2][:2].tolist() == [5, 3]
        assert abs(g["root_value"][2] - (-0.10)) < 1e-6
        assert g["sims_run"][0] == 10 and g["sim_steps"][0] == 70
        assert g["reused_visits"][0] == 3 and g["memo_hits"][0] == 2
        assert g["stopped_early"].tolist() == [1, 0, 1]
        assert np.isnan(g["time_budget_s"]).all()      # fake sets none
        assert g["n_worlds"].tolist() == [2, 0, 2]
        assert g["world_seeds"][0].tolist() == [100, 101]
        assert g["world_seeds"][1].tolist() == [-1, -1]
        assert g["world_visits"][0].tolist() == [5, 5]
        assert np.isnan(g["world_values"][1]).all()
        assert (g["origin_row"] == -1).all()
        assert (g["follow_path"] == -1).all()
        assert g["follow_worlds"].sum() == 0
        assert abs(g["priors"][0][:3].sum() - 1.0) < 1e-6

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
        # A tree-FOLLOWED decision: on_followed stashes its diag, the step
        # observer's one-hot row commits it (kind 2, origin = the last
        # searched row, the cumulative path and surviving worlds).
        rec.on_followed(_obs(False, 1, 1), 3, np.array([40.0, 2.0, 1.0]),
                        [1, 0, 2], [1], 0)
        _onehot(rec, _obs(False, 1, 1), 3, action=0)
        _searched(rec, _obs(True, 1, 2), 2, [6, 2], -0.3, chosen=0,
                  reward=-1.0, game_result=True)     # B wins game 1
        _searched(rec, _obs(True, 2, 1), 2, [7, 1], 0.6, chosen=0)
        _onehot(rec, _obs(False, 2, 2), 2, action=0,
                reward=1.0, game_result=True, done=True)   # A wins match

        files = sorted(glob.glob(os.path.join(tmp, "shard_*.npz")))
        assert len(files) == 1, files
        d = _check_schema(files[0])
        assert d["obs"].shape[0] == 10
        g = _check_diag(files[0], 10)
        assert g["kind"][5] == DIAG_KIND_SEARCH
        assert g["kind"][6] == DIAG_KIND_FOLLOWED
        assert g["visits"][6][:3].tolist() == [40, 2, 1]
        assert g["origin_row"][6] == 5
        assert g["follow_path"][6].tolist() == [1, 0, 2]
        assert g["follow_worlds"][6].tolist() == [0, 1]
        assert g["n_worlds"][6] == 0 and g["sims_run"][6] == 0
        # Sideboard row (idx 4, mover A) is priced on the UPCOMING game 1,
        # which A lost.
        assert d["z"][4] == -1.0
        # Game-1 rows: B rows +1, A row -1; game-2 rows: A +1, B -1.
        assert d["z"][5:].tolist() == [1.0, 1.0, -1.0, 1.0, -1.0]

        # ── Readers ────────────────────────────────────────────────────────
        sample = az_inspect.load_shard_sample(tmp, max_rows=100)
        assert sample["obs"].shape[0] == 10 and "td_q" in sample
        # π-dependent views read only search-posterior rows: the searched
        # rows 0,2,5,7,8 — not the human / followed one-hots (q = NaN) or
        # the sideboard one-hot (row 4).
        assert sample["pi_valid"].tolist() == [
            True, False, True, False, False, True, False, True, True,
            False], sample["pi_valid"]
        lines = az_inspect.render_state(sample, 1)
        assert lines[0].startswith("(no search posterior"), lines
        assert not az_inspect.render_state(sample, 0)[0].startswith("(no")

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
        assert len(rec_b[0]["observations"]) == 5
        # Diag sidecar attached per step: A's rows are shard rows 0,2,4,7,8
        # (row 4 the one-hot sideboard pick), B's rows 1,3,5,6,9 (row 5 the
        # searched root, row 6 the decision followed from its trees).
        from browse_session import decision_data, search_line_for
        ra, rb = rec_a[0], rec_b[0]
        assert ra["row_index"] == [0, 2, 4, 7, 8]
        assert rb["row_index"] == [1, 3, 5, 6, 9]
        assert [d["kind"] if d else None for d in ra["diag"]] == [
            DIAG_KIND_SEARCH, DIAG_KIND_SEARCH, None, DIAG_KIND_SEARCH,
            DIAG_KIND_SEARCH]
        assert [d["kind"] if d else None for d in rb["diag"]] == [
            None, None, DIAG_KIND_SEARCH, DIAG_KIND_FOLLOWED, None]
        assert ra["diag"][0]["visits"].tolist() == [8, 1, 1]
        assert ra["diag"][0]["num_choices"] == 3
        assert ra["diag"][0]["world_seeds"] == [100, 101]
        assert ra["diag"][0]["time_budget_s"] is None
        assert ra["diag"][0]["stopped_early"] is True
        assert rb["diag"][2]["visits"].tolist() == [2, 6]
        assert rb["diag"][3]["visits"].tolist() == [40, 2, 1]
        assert rb["diag"][3]["origin_row"] == 5
        assert rb["diag"][3]["follow_path"] == [1, 0, 2]
        assert rb["diag"][3]["follow_worlds"] == [1]
        assert rb["origin_step"] == [None, None, None, 2, None]
        assert ra["origin_step"] == [None] * 5
        assert ra["diag_prov"] is None and ra["shard_stem"] is None  # no .rmplay
        line = search_line_for(ra["diag"][0], None)
        assert line and line.startswith("search: 10 sims"), line
        assert "reused 3" in line and "timed" not in line
        line = search_line_for(rb["diag"][3], rb["origin_step"][3])
        assert line == "followed from decision 2 via 3 action(s) · 43 visits"
        assert search_line_for(rb["diag"][3], None) == \
            "followed (origin not in view)"
        assert search_line_for(None, None) == ""
        # decision_data surfaces the diag columns: searched rows sort by
        # visits, followed rows carry visits only, plain rows nothing.
        dd = decision_data(rb, 2)
        assert dd.search_line.startswith("search: 8 sims · 2 worlds · root V +0.400"), \
            dd.search_line
        assert [r.k for r in dd.rows] == [1, 0]
        assert [r.visits for r in dd.rows] == [6, 2]
        assert all(r.q is not None and r.prior is not None for r in dd.rows)
        dd = decision_data(rb, 3)
        assert dd.search_line.startswith("followed from decision 2")
        assert [r.visits for r in dd.rows] == [40, 2, 1]
        assert all(r.q is None and r.prior is None for r in dd.rows)
        dd = decision_data(rb, 0)
        assert dd.search_line == ""
        assert all(r.visits is None for r in dd.rows)

        _check_search_posterior(ra, rb)
        _check_bounded_load()

        # The one search-vs-net KL / top-1 definition (duplicate-folded net
        # priors, KL(search ‖ net)) the probes and analysis.py share.
        import test_menu_merge
        test_menu_merge.test_search_net_divergence()
        assert not test_menu_merge.FAILURES, test_menu_merge.FAILURES

        # Trainer ingestion (skipped when torch isn't installed).
        try:
            import az_train
        except ImportError:
            print("  (az_train unavailable — trainer ingestion skipped)")
        else:
            w = az_train.load_window("gen", window=10, data_dir=tmp)
            assert w["obs"].shape[0] == 10

        # ── A second match opens a second file ─────────────────────────────
        _searched(rec, _obs(True, 0, 1), 2, [3, 3], 0.0, chosen=0,
                  reward=-1.0, done=True)            # bo1-style: no game_result
        rec.close()
        files = sorted(glob.glob(os.path.join(tmp, "shard_*.npz")))
        assert len(files) == 2, files
        d = _check_schema(files[1])
        assert d["obs"].shape[0] == 1 and d["z"][0] == -1.0
        _check_diag(files[1], 1)                 # diag follows per-match files

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
                             "opponent_spec": "az:gen", "binary": None,
                             "search_provenance": {
                                 "spec": "az:gen", "sims": 128, "worlds": 4,
                                 "checkpoint": None}},
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
            assert doc["search_provenance"] == {
                "spec": "az:gen", "sims": 128, "worlds": 4, "checkpoint": None}
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
            # Provenance + stem ride the .rmplay sidecar into the record.
            assert r["diag_prov"] == doc["search_provenance"]
            assert r["shard_stem"] == os.path.splitext(sides[0])[0]
            assert recs_b[0]["diag_prov"] == doc["search_provenance"]
            assert [d["kind"] for d in r["diag"]] == [DIAG_KIND_SEARCH] * 2
            assert recs_b[0]["diag"] == [None]
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

        print("shard_record OK: schema, mid-game flush, z backfill, "
              "segmentation, both readers, per-match files, replay sidecar, "
              "diag sidecar, probe search posterior")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
