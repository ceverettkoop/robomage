#!/usr/bin/env python
"""Regression for ``tree_rebuild``: a recorded search opponent's trees are
rebuilt bit-for-bit from the recording's diagnostics, cached, and browsable.

Records two short sessions of a scripted seat A vs an ``mcts:uniform`` seat B
(one fixed-sims, one TIMED so ``sims_run`` is arbitrary) through
``ShardRecorder`` with the replay sidecar, loads them back through
``shard_replay.load_records``, and for the searched steps checks that
``TreeSession`` verifies the rebuilt root visits against the recorded ones,
that a second open is a cache hit with identical visits, and that pv/walk/
node_stats work. Uses the debug engine (``cli_spec.BINARY``) and no torch.

Run: train/.venv/bin/python train/test_tree_rebuild.py
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from cli_spec import BINARY  # noqa: E402
from env import _SELF_IS_A_IDX  # noqa: E402
from opponents import make_controller  # noqa: E402
from search_env import SearchRoboMageEnv  # noqa: E402
from shard_record import ShardRecorder  # noqa: E402
import shard_replay  # noqa: E402
from tree_rebuild import (RebuildError, TreeSession, cache_path_for,  # noqa: E402
                          evaluator_spec_for)

DECK_A = "league/ur_delver"
DECK_B = "league/gw_maverick"
SEED = 7
MAX_DECISIONS = 60
MIN_SEARCHED = 3
MAX_REBUILDS = 4    # kind-1 steps rebuilt per session (runtime cap)


class TreeRebuildTestError(Exception):
    pass


def _check(cond, msg):
    if not cond:
        raise TreeRebuildTestError(msg)


def _tmp_dir(name):
    base = os.environ.get("ROBOMAGE_TEST_TMP")
    return tempfile.mkdtemp(prefix=f"tree_rebuild_{name}_",
                            dir=base if base and os.path.isdir(base) else None)


def record_session(out_dir, spec, seed, max_decisions=MAX_DECISIONS):
    """Play scripted (A) vs ``spec`` (B) into ``out_dir`` through the recorder
    until ``max_decisions`` or the game ends. Returns (searched, followed)."""
    ctrl_a = make_controller("scripted")
    ctrl_b = make_controller(spec)
    env = SearchRoboMageEnv(binary_path=BINARY, deck_a=DECK_A, deck_b=DECK_B,
                            bo3=False)
    counts = {"searched": 0, "followed": 0}
    try:
        set_names = getattr(ctrl_a, "set_deck_names", None)
        if set_names is not None:
            set_names(DECK_A, DECK_B)
        ctrl_b.bind_env(env)
        rec = ShardRecorder(
            out_dir,
            replay_meta={"deck_a": DECK_A, "deck_b": DECK_B,
                         "human_deck": DECK_A, "opp_deck": DECK_B,
                         "human_is_a": True, "bo3": False,
                         "opponent_spec": spec, "binary": BINARY,
                         "search_provenance": ctrl_b.search_provenance()},
            seed_fn=lambda: env.last_engine_seed)

        def on_result(obs, n, result, chosen):
            counts["searched"] += 1
            rec.on_search_result(obs, n, result, chosen)

        def on_followed(obs, n, visits, path, worlds, chosen):
            counts["followed"] += 1
            rec.on_followed(obs, n, visits, path, worlds, chosen)

        ctrl_b.on_result = on_result
        ctrl_b.on_followed = on_followed
        obs, _ = env.reset(options={"engine_seed": seed})
        done = False
        decisions = 0
        while not done and decisions < max_decisions:
            n = int(env._num_choices)
            ctrl = ctrl_a if obs[_SELF_IS_A_IDX] > 0.5 else ctrl_b
            action = ctrl.choose(obs, n, action_masks=env.action_masks())
            pre = obs.copy()
            obs, reward, term, trunc, info = env.step(action)
            done = bool(term or trunc)
            rec.observe_step(pre, n, int(action), reward, info, done)
            decisions += 1
        rec.flush()
        rec.close()
    finally:
        env.close()
    return counts["searched"], counts["followed"]


def _load_single_record(out_dir):
    games = shard_replay.load_records(out_dir, viewpoint_is_a=False)
    _check(len(games) == 1, f"expected one match record, got {len(games)}")
    game = games[0]
    _check(game.get("diag") is not None and game.get("shard_stem"),
           "record carries no diag/shard_stem (shard_replay Phase B missing?)")
    _check(game.get("engine_seed") is not None, "record is not replayable")
    return game


def _session(game, step, **kw):
    return TreeSession(game, step, binary=BINARY, deck_a=DECK_A,
                       deck_b=DECK_B, bo3=False, **kw)


def check_searched_step(game, step):
    """Rebuild + verify, then reopen from cache and browse."""
    diag = game["diag"][step]
    with _session(game, step) as s:
        _check(not s.from_cache, f"step {step}: first open should not hit cache")
        _check(s.verified, f"step {step}: {s.summary()}")
        _check(s.root_step == step and s.follow_path == [],
               f"step {step}: kind-1 root bookkeeping wrong")
        first = s.root_stats()
        _check(np.array_equal(first.visits.astype(np.int64), diag["visits"]),
               f"step {step}: root_stats visits != recorded")
        _check(int(first.visits.sum()) == diag["sims_run"] == s.sims_run,
               f"step {step}: visit mass {int(first.visits.sum())} != sims_run "
               f"{diag['sims_run']}")
        labels = s.root_labels()
        _check(len(labels) == diag["num_choices"], f"step {step}: label count")
        cache = s.cache_path
        _check(cache and os.path.exists(cache),
               f"step {step}: tree cache not written ({cache})")
        _check(cache == cache_path_for(game["shard_stem"], game["row_index"][step]),
               f"step {step}: cache path {cache} not beside the shard")
        summary1 = s.summary()
    with _session(game, step) as s:
        _check(s.from_cache, f"step {step}: second open should hit the cache")
        _check(s.verified, f"step {step}: cached verified flag lost")
        second = s.root_stats()
        _check(np.array_equal(first.visits, second.visits),
               f"step {step}: cached visits differ from the rebuilt ones")
        _check(np.allclose(first.w_sum, second.w_sum),
               f"step {step}: cached W differs from the rebuilt one")
        best = int(np.argmax(second.visits))
        # A world where the best root action was visited, so the PV is non-empty.
        for w in range(s.worlds):
            pv = s.pv(best, w)
            if pv:
                break
        _check(pv, f"step {step}: empty PV for the best action in every world")
        _check(pv[0].action == best or pv[0].visits > 0,
               f"step {step}: PV does not start at the best action")
        nodes, labels = s.walk(w, [p.action for p in pv])
        _check(len(nodes) >= 1, f"step {step}: walk returned no nodes")
        if nodes[-1].terminal is None:
            _check(len(labels) == nodes[-1].num_choices,
                   f"step {step}: walk labels {len(labels)} != menu "
                   f"{nodes[-1].num_choices}")
        else:
            _check(labels == [], f"step {step}: labels at a terminal")
        stats = s.node_stats(w, [])
        _check(stats and sum(n for _a, n, _q, _p in stats)
               == int(second.world_visits[w].sum()),
               f"step {step}: root node_stats visit mass mismatch")
        summary2 = s.summary()
    return summary1, summary2


def check_followed_step(game, step):
    diag = game["diag"][step]
    with _session(game, step) as s:
        origin = game["origin_step"][step]
        _check(s.root_step == origin,
               f"followed step {step}: root_step {s.root_step} != origin {origin}")
        _check(s.follow_path == diag["follow_path"]
               and s.follow_worlds == diag["follow_worlds"],
               f"followed step {step}: follow bookkeeping mismatch")
        _check(s.verified, f"followed step {step}: origin tree {s.summary()}")
        found = None
        total = np.zeros(diag["num_choices"], dtype=np.int64)
        for w in s.follow_worlds:
            stats = s.node_stats(w, s.follow_path)
            if stats:
                found = w
                for a, n, _q, _p in stats:
                    total[a] += n
        _check(found is not None,
               f"followed step {step}: no surviving world reaches the path")
        _check(np.array_equal(total, diag["visits"]),
               f"followed step {step}: summed node visits != recorded")
        nodes, labels = s.walk(found, s.follow_path)
        _check(len(nodes) == len(s.follow_path) and nodes[-1].terminal is None
               and len(labels) == diag["num_choices"],
               f"followed step {step}: walk to the followed node failed")
        return s.summary()


def synth_followed_record(game, step):
    """A copy of ``game`` with one extra FOLLOWED step (kind 2) whose origin
    is searched step ``step``, built from that step's (cached) tree: the
    most-visited root action of world 0, followed in every world whose tree
    expanded it. The uniform evaluator never satisfies the controller's
    follow gates (a 66% visit share), so the kind-2 resolution path is
    exercised on a synthesized row shaped exactly like the recorder's."""
    with _session(game, step) as s:
        roots = s._search.roots
        root0 = roots[0]
        order = np.argsort(-root0.N)
        for a in order:
            a = int(a)
            if root0.children.get(a) is None:
                continue
            nc = int(root0.children[a].num_choices)
            worlds, visits = [], np.zeros(nc, dtype=np.int64)
            for w, root in enumerate(roots):
                rep = int(root.rep[a]) if root.rep is not None else a
                child = root.children.get(rep)
                if child is not None and int(child.num_choices) == nc:
                    worlds.append(w)
                    visits += child.N.astype(np.int64)
            break
        else:
            raise TreeRebuildTestError(f"step {step}: root has no expanded child")
    followed = {
        "kind": 2, "num_choices": nc, "visits": visits,
        "origin_row": int(game["row_index"][step]),
        "follow_path": [a], "follow_worlds": worlds,
    }
    game2 = dict(game)
    game2["diag"] = list(game["diag"]) + [followed]
    game2["origin_step"] = list(game["origin_step"]) + [step]
    game2["row_index"] = list(game["row_index"]) + [len(game["row_index"])]
    return game2, len(game["diag"])


def check_refusals(game):
    plain = [i for i, d in enumerate(game["diag"]) if d is None]
    if plain:
        try:
            _session(game, plain[0]).open()
        except RebuildError:
            pass
        else:
            raise TreeRebuildTestError("kind-0 step opened a TreeSession")
    prov = dict(game["diag_prov"] or {})
    prov["procs"] = 2
    timed = [i for i, d in enumerate(game["diag"])
             if d is not None and d["kind"] == 1 and d["time_budget_s"] is not None]
    if timed:
        try:
            _session(game, timed[0], prov=prov).open()
        except RebuildError:
            pass
        else:
            raise TreeRebuildTestError("timed procs>1 recording was rebuilt")


def check_pure_helpers():
    _check(evaluator_spec_for({"spec": "mcts:uniform?sims=64", "checkpoint": None})
           == "uniform", "evaluator_spec_for uniform")
    _check(evaluator_spec_for({"checkpoint": "/x/gen__azfinal.pt"})
           == "az:/x/gen__azfinal.pt", "evaluator_spec_for .pt")
    _check(evaluator_spec_for({"checkpoint": "/x/gen__final.zip"})
           == "mcts:/x/gen__final.zip", "evaluator_spec_for .zip")
    _check(evaluator_spec_for({"spec": "az:gen?sims=8"}) == "az:gen",
           "evaluator_spec_for base")
    _check(cache_path_for("/r/shard_1_2_0", 5) == "/r/trees/shard_1_2_0_row5.npz",
           "cache_path_for")


def run_session(name, spec):
    out_dir = _tmp_dir(name)
    t0 = time.monotonic()
    searched, followed = record_session(out_dir, spec, SEED)
    print(f"[{name}] recorded {searched} searched / {followed} followed "
          f"decisions in {time.monotonic() - t0:.1f}s -> {out_dir}")
    _check(searched >= MIN_SEARCHED,
           f"[{name}] only {searched} searched decisions recorded")
    game = _load_single_record(out_dir)
    kinds = [None if d is None else d["kind"] for d in game["diag"]]
    k1 = [i for i, k in enumerate(kinds) if k == 1]
    k2 = [i for i, k in enumerate(kinds) if k == 2]
    _check(len(k1) >= MIN_SEARCHED, f"[{name}] record has {len(k1)} kind-1 steps")
    for step in k1[:MAX_REBUILDS]:
        t0 = time.monotonic()
        s1, s2 = check_searched_step(game, step)
        print(f"[{name}] step {step}: {s1}")
        print(f"[{name}] step {step}: {s2} ({time.monotonic() - t0:.1f}s)")
    if k2:
        print(f"[{name}] followed step {k2[0]}: "
              f"{check_followed_step(game, k2[0])}")
    else:
        print(f"[{name}] no tree-followed decision within {MAX_DECISIONS} "
              f"decisions; checking a synthesized one")
    game2, fake = synth_followed_record(game, k1[0])
    print(f"[{name}] synthesized followed step {fake}: "
          f"{check_followed_step(game2, fake)}")
    check_refusals(game)
    return len(k1), len(k2)


def main():
    t0 = time.monotonic()
    check_pure_helpers()
    run_session("fixed", "mcts:uniform?sims=64&worlds=4")
    # No sims= knob: the clock alone terminates, so sims_run is arbitrary and
    # the rebuild must reproduce a search that stopped mid round-robin.
    run_session("timed", "mcts:uniform?worlds=4&time=0.3")
    print(f"test_tree_rebuild: OK ({time.monotonic() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TreeRebuildTestError as exc:
        print(f"test_tree_rebuild: FAIL: {exc}")
        sys.exit(1)
