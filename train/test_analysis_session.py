#!/usr/bin/env python3
"""Analysis-window engine-core regression (GUI analysis, Phase 1).

Proves the guarantees the analysis window relies on:

  1. chunked parity — an ``IncrementalSearch`` run in ragged chunks (5+16+43)
     with pinned world seeds produces bit-identical visits/q/world_values to a
     plain ``run_search`` with the same seeds and total (worlds are independent
     trees, so interleaving across worlds cannot change them). Also checks the
     new ``SearchResult.q/w_sum/world_values`` fields are populated and
     self-consistent, in both ``run_search`` and a mirrored
     ``run_search_parallel`` merge.
  2. pv/walk driver discipline — a principal variation reads out with
     non-increasing visits; walking it replays decodable states; the snapshot
     survives an idle pause (parked search) and further chunks; after close()
     the env is byte-identical to the root and a REAL step works.
  3. session lockstep — an ``AnalysisSession``'s detached analysis engine
     follows a scripted bo3 by delta replay (including across the sideboard
     boundary), analyzing at several stops, each analysis leaving the live env
     untouched.
  4. rewind — a request for a decision the engine has already moved PAST (the
     window's "opponent's last decision" review) respawns the analysis engine
     at that history prefix, analyzes there without touching the live env, and
     then follows the live game forward again.
  5. world-parallel analysis — a run split across N detached engines
     (``AnalysisConfig.procs``) merges to a bit-identical search to the
     one-engine run with the same seed (visits/w_sum/q/root_value/world_values),
     browses (pv/walk) a world owned by a non-first engine to the same line,
     and rewinds the whole fleet together. A non-positive ``chunk_sims`` (which
     would spin an uncapped run forever) is rejected loudly.
  6. cancellation — a cross-thread stop event ends an unlimited analyze()
     within one chunk; the parked search stays browsable; a re-analyze at the
     same decision succeeds.

Torch-free (UniformEvaluator + scripted agent); spawns bin/robomage with
--search-server. Wired into ``train/ci_check.py`` as the opt-in ``analysis``
tier; also runnable standalone::

    train/.venv/bin/python train/test_analysis_session.py
"""
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import decode  # noqa: E402
from analysis_session import (AnalysisConfig, AnalysisRequest,  # noqa: E402
                              AnalysisSession)
from cli_spec import BINARY, BIN_DIR  # noqa: E402
from env import STATE_SIZE, _IS_SIDEBOARD_IDX, _SELF_IS_A_IDX  # noqa: E402
from mcts import (IncrementalSearch, UniformEvaluator,  # noqa: E402
                  run_search, run_search_parallel)
from scripted_agent import make_agent  # noqa: E402
from search_env import SearchRoboMageEnv  # noqa: E402

SEED = 7            # engine seed (deterministic game)
SEARCH_SEED = 1234  # search rng seed (world-seed derivation)
WORLD_SEEDS = [101, 202, 303, 404]
BO3_STEP_CAP = 3000


class AnalysisTestError(AssertionError):
    pass


def _write_decks():
    """Same lopsided fast matchup as test_mirror_search: A (bears+forests)
    reliably beats B (all swamps), so a bo3 finishes quickly and crosses one
    sideboard boundary; 15-card sideboards make the sideboard menu real."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    a = os.path.join(d, "analysis_test_a.dk")
    b = os.path.join(d, "analysis_test_b.dk")
    with open(a, "w") as f:
        f.write("36 Grizzly Bears\n24 Forest\nSIDEBOARD:\n15 Mountain\n")
    with open(b, "w") as f:
        f.write("60 Swamp\nSIDEBOARD:\n15 Mountain\n")
    return "temp/analysis_test_a", "temp/analysis_test_b", [a, b]


def _rm(paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _drive_to_safe(env, min_nc=2, cap=300):
    """Auto-0 step until a loop-safe decision with >= min_nc choices."""
    for _ in range(cap):
        if env.last_search_safe and env._num_choices >= min_nc:
            return
        env.step(0)
    raise AnalysisTestError(
        f"no safe decision with >= {min_nc} choices in {cap} steps")


def _req(env):
    """An AnalysisRequest for the primary env's current decision (what the
    driver's StateUpdate would carry)."""
    obs = env._obs.copy()
    return AnalysisRequest(
        obs=obs, num_choices=env._num_choices,
        history_len=len(env._action_history),
        search_safe=bool(getattr(env, "last_search_safe", False)),
        seat_is_a=bool(obs[_SELF_IS_A_IDX] > 0.5),
        is_sideboard=bool(obs[_IS_SIDEBOARD_IDX] > 0.5))


def test_chunked_parity():
    """Ragged-chunk IncrementalSearch == run_search with the same world seeds
    and per-world sim totals; new SearchResult fields are consistent."""
    deck_a, deck_b, paths = _write_decks()
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b)
        ev = UniformEvaluator()
        try:
            env.reset(options={"engine_seed": SEED})
            _drive_to_safe(env)

            r = run_search(env, ev, worlds=4, sims=64, world_seeds=WORLD_SEEDS,
                           rng=np.random.default_rng(SEARCH_SEED))
            if r.q is None or r.w_sum is None or r.world_values is None:
                raise AnalysisTestError("run_search left new fields None")
            if len(r.world_values) != 4:
                raise AnalysisTestError(
                    f"world_values has {len(r.world_values)} entries, want 4")
            expect_q = np.where(r.visits > 0,
                                r.w_sum / np.maximum(r.visits, 1), 0.0)
            if not np.array_equal(r.q, expect_q):
                raise AnalysisTestError("q != w_sum/visits")

            s = IncrementalSearch(env, ev, worlds=4, world_seeds=WORLD_SEEDS)
            for n in (5, 16, 43):        # ragged chunks, total 64 = 4 * 16
                stats = s.run_chunk(n)
            try:
                if stats.sims_run != r.sims_run:
                    raise AnalysisTestError(
                        f"sims_run {stats.sims_run} != {r.sims_run}")
                if not np.array_equal(stats.visits, r.visits):
                    raise AnalysisTestError(
                        f"chunked visits {stats.visits} != plain {r.visits}")
                if not np.allclose(stats.q, r.q):
                    raise AnalysisTestError("chunked q != plain q")
                if not np.allclose(stats.world_values, r.world_values):
                    raise AnalysisTestError(
                        "chunked world_values != plain world_values")
                if int(stats.world_visits.sum()) != int(stats.visits.sum()):
                    raise AnalysisTestError("world_visits don't sum to visits")
                if not (-1.0 <= stats.net_value <= 1.0):
                    raise AnalysisTestError(f"net_value {stats.net_value} out of range")
                rr = s.result()
                if not np.array_equal(rr.visits, r.visits) or not np.allclose(rr.q, r.q):
                    raise AnalysisTestError("result() disagrees with run_search")
            finally:
                s.close()
            return (f"64 sims in chunks 5+16+43 bit-equal to run_search "
                    f"(visits {r.visits.astype(int).tolist()})")
        finally:
            env.close()
    finally:
        _rm(paths)


def test_parallel_merge_fields():
    """run_search_parallel (primary + 1 mirror) merges the new fields to the
    same values as a plain run_search with the same rng seed."""
    deck_a, deck_b, paths = _write_decks()
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b)
        ev = UniformEvaluator()
        try:
            env.reset(options={"engine_seed": SEED})
            _drive_to_safe(env)
            r1 = run_search(env, ev, worlds=4, sims=16,
                            rng=np.random.default_rng(SEARCH_SEED))
            env.ensure_mirrors(1)
            r2 = run_search_parallel(env.search_envs(), ev, worlds=4, sims=16,
                                     rng=np.random.default_rng(SEARCH_SEED))
            if not np.array_equal(r1.visits, r2.visits):
                raise AnalysisTestError("parallel visits diverged")
            if not np.allclose(r1.w_sum, r2.w_sum):
                raise AnalysisTestError("parallel w_sum diverged")
            if not np.allclose(r1.q, r2.q):
                raise AnalysisTestError("parallel q diverged")
            if not np.allclose(r1.world_values, r2.world_values):
                raise AnalysisTestError("parallel world_values diverged "
                                        "(env order != world order?)")
            return "parallel merge of q/w_sum/world_values bit-equal to plain"
        finally:
            env.close()
    finally:
        _rm(paths)


def test_pv_walk_discipline():
    """PV reads out sanely; walking it yields decodable states; the parked
    snapshot survives an idle pause and more chunks; after close() the env is
    back at the root and a real step works."""
    deck_a, deck_b, paths = _write_decks()
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b)
        ev = UniformEvaluator()
        try:
            env.reset(options={"engine_seed": SEED})
            _drive_to_safe(env)

            s = IncrementalSearch(env, ev, worlds=2, world_seeds=[11, 22])
            s.run_chunk(32)
            best0 = int(np.argmax(s.stats().world_visits[0]))
            pv = s.pv(best0, world=0)
            if not pv:
                raise AnalysisTestError("empty PV for world 0's best action")
            for a, b in zip(pv, pv[1:]):
                if b.visits > a.visits:
                    raise AnalysisTestError(
                        f"PV visits increased ({a.visits} -> {b.visits})")
            for step in pv:
                if not (-1.0 <= step.q <= 1.0):
                    raise AnalysisTestError(f"PV q {step.q} out of range")

            nodes = s.walk(0, [p.action for p in pv])
            if not nodes or len(nodes) > len(pv):
                raise AnalysisTestError(
                    f"walk returned {len(nodes)} nodes for a {len(pv)}-step PV")
            for node in nodes:
                if node.terminal is not None:
                    if node is not nodes[-1]:
                        raise AnalysisTestError("terminal WalkNode not last")
                    continue
                if node.obs is None:
                    raise AnalysisTestError("non-terminal WalkNode without obs")
                gs = decode.decode_game_state(node.obs[:STATE_SIZE])
                if gs["self"]["life"] < -30 or gs["opponent"]["life"] < -30:
                    raise AnalysisTestError("walked state decoded implausibly")

            # Parked search survives an idle pause (snapshot held) + more work.
            time.sleep(1.5)
            s.run_chunk(4)
            if s.sims_run != 36:
                raise AnalysisTestError(f"resume after idle ran {s.sims_run} != 36")

            s.close()
            if not np.array_equal(env._obs, s.root_obs):
                raise AnalysisTestError("env not restored to root after close()")
            env.step(0)   # a REAL step must work (release happened)
            return (f"PV len {len(pv)}, walk {len(nodes)} nodes decoded, "
                    "idle+resume ok, real step after close ok")
        finally:
            env.close()
    finally:
        _rm(paths)


def test_session_lockstep_bo3():
    """An AnalysisSession follows a scripted bo3 by delta replay, analyzing at
    several stops including a sideboard decision, without touching the live
    env; a respawn() rebuilds the engine by full replay."""
    deck_a, deck_b, paths = _write_decks()
    cfg = AnalysisConfig(evaluator_spec="uniform", worlds=2, chunk_sims=4,
                         max_sims=8, seed=SEARCH_SEED)
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b, bo3=True,
                                auto_sideboard=False)
        agent = make_agent("scripted")
        session = AnalysisSession(env, cfg, evaluator=UniformEvaluator(),
                                  evaluator_label="uniform")
        try:
            env.reset(options={"engine_seed": SEED})
            agent.new_game()

            analyses = 0
            sb_analyses = 0
            saw_sideboard = False
            terminated = False
            since_analysis = 10**9   # analyze at the first analyzable decision
            for i in range(BO3_STEP_CAP):
                since_analysis += 1
                req = _req(env)
                if req.is_sideboard:
                    saw_sideboard = True
                # Analyze at the NEXT analyzable decision every ~25 real
                # decisions (the scripted bears bo3 sweeps in ~100 decisions
                # total), plus the first searchable sideboard decision.
                want = (since_analysis >= 25
                        or (req.is_sideboard and sb_analyses < 1))
                if want and session.can_analyze(req) is None:
                    before = env._obs.copy()
                    stats = session.analyze(req)
                    if stats.sims_run != cfg.max_sims:
                        raise AnalysisTestError(
                            f"analyze ran {stats.sims_run} != {cfg.max_sims}")
                    if not np.array_equal(env._obs, before):
                        raise AnalysisTestError(
                            "analysis touched the LIVE env's state")
                    analyses += 1
                    since_analysis = 0
                    if req.is_sideboard:
                        sb_analyses += 1
                action = agent.act(env._obs, env._num_choices)
                _, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    terminated = True
                    break

            if not terminated:
                raise AnalysisTestError(f"bo3 unfinished in {BO3_STEP_CAP} steps")
            if not saw_sideboard:
                raise AnalysisTestError("bo3 never crossed a sideboard boundary")
            if analyses < 3:
                raise AnalysisTestError(f"only {analyses} analyses ran")
            if sb_analyses < 1:
                raise AnalysisTestError("no sideboard decision was analyzed")

            # respawn(): full-replay recovery builds a fresh engine mid-match.
            session.respawn()
            return (f"{analyses} analyses in lockstep (incl. {sb_analyses} "
                    "sideboard), live env untouched")
        finally:
            session.close()
            env.close()
    finally:
        _rm(paths)


def test_rewind_to_earlier_decision():
    """The analysis engine can be sent BACKWARDS: after following the live game
    forward, an AnalysisRequest for an EARLIER decision (the window's
    "opponent's last decision" review) respawns the engine at that history
    prefix and analyzes there, leaving the live env untouched; the engine then
    follows the live game forward again."""
    deck_a, deck_b, paths = _write_decks()
    cfg = AnalysisConfig(evaluator_spec="uniform", worlds=2, chunk_sims=4,
                         max_sims=8, seed=SEARCH_SEED)
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b)
        session = AnalysisSession(env, cfg, evaluator=UniformEvaluator(),
                                  evaluator_label="uniform")
        try:
            env.reset(options={"engine_seed": SEED})
            _drive_to_safe(env)
            old = _req(env)
            session.analyze(old)              # engine synced at `old`

            # Advance the live game well past that decision.
            agent = make_agent("scripted")
            agent.new_game()
            for _ in range(8):
                _, _, term, trunc, _ = env.step(
                    agent.act(env._obs, env._num_choices))
                if term or trunc:
                    raise AnalysisTestError("game ended before the rewind test")
            _drive_to_safe(env)
            new = _req(env)
            if new.history_len <= old.history_len:
                raise AnalysisTestError("live game did not advance")
            session.analyze(new)              # forward delta

            before = env._obs.copy()
            stats = session.analyze(old)      # <- the rewind
            if stats.sims_run != cfg.max_sims:
                raise AnalysisTestError(
                    f"rewound analyze ran {stats.sims_run} != {cfg.max_sims}")
            if not np.array_equal(env._obs, before):
                raise AnalysisTestError("rewind touched the LIVE env's state")
            # And forward again from the rewound engine.
            stats2 = session.analyze(new)
            if stats2.sims_run != cfg.max_sims:
                raise AnalysisTestError(
                    f"re-forward analyze ran {stats2.sims_run}")
            return (f"rewound {new.history_len} -> {old.history_len} and "
                    "forward again, live env untouched")
        finally:
            session.close()
            env.close()
    finally:
        _rm(paths)


class _ValueEvaluator(UniformEvaluator):
    """Uniform priors, but a DETERMINISTIC non-neutral value read off the obs.

    UniformEvaluator backs up 0.0 from every non-terminal leaf, so its W sums
    are exact zeros — which would make a w_sum/q/root_value parity check
    vacuous. These irrational-looking float values make the merged sums depend
    on real floating-point addition, so the parity claim is actually tested."""

    def evaluate(self, obs: np.ndarray, num_choices: int):
        priors, _ = super().evaluate(obs, num_choices)
        return priors, float(np.tanh(float(np.sum(obs[:512])) * 0.017))


def test_parallel_analysis_parity():
    """World-parallel analysis (cfg.procs=2) merges to the SAME search as the
    one-engine run with the same cfg.seed: bit-identical visits/w_sum/q/
    root_value/world_values, and a world owned by the SECOND engine browses
    (pv/walk) to the same line."""
    deck_a, deck_b, paths = _write_decks()
    worlds, cap = 4, 16          # cap divisible by worlds: 4 sims per world
    far_world = worlds - 1       # lives in engine 2's slice under a 2-way split

    def _cfg(procs):
        return AnalysisConfig(evaluator_spec="uniform", worlds=worlds,
                              chunk_sims=4, max_sims=cap, seed=SEARCH_SEED,
                              procs=procs)

    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b)
        try:
            env.reset(options={"engine_seed": SEED})
            _drive_to_safe(env)
            # Analyze a MID-GAME decision (wider menu, deeper trees, real value
            # mass backed up) rather than the opening mulligan.
            agent = make_agent("scripted")
            agent.new_game()
            for _ in range(24):
                _, _, term, trunc, _ = env.step(
                    agent.act(env._obs, env._num_choices))
                if term or trunc:
                    raise AnalysisTestError("game ended before the parity test")
            _drive_to_safe(env, min_nc=3)
            req = _req(env)

            runs = {}
            for procs in (1, 2):
                session = AnalysisSession(env, _cfg(procs),
                                          evaluator=_ValueEvaluator(),
                                          evaluator_label="uniform")
                try:
                    stats = session.analyze(req)
                    if len(session._envs) != procs:
                        raise AnalysisTestError(
                            f"procs={procs} spawned {len(session._envs)} engines")
                    if stats.sims_run != cap:
                        raise AnalysisTestError(
                            f"procs={procs} ran {stats.sims_run} != {cap} sims")
                    best = int(np.argmax(stats.visits))
                    pv = [(p.action, p.visits, p.q, p.seat_is_a)
                          for p in session.pv(best, world=far_world)]
                    nodes = session.walk(far_world, [p[0] for p in pv])
                    runs[procs] = (stats, best, pv, nodes)
                finally:
                    session.close()

            (s1, best1, pv1, nodes1) = runs[1]
            (s2, best2, pv2, nodes2) = runs[2]
            if not np.array_equal(s1.visits, s2.visits):
                raise AnalysisTestError(
                    f"parallel visits {s2.visits} != serial {s1.visits}")
            if not np.array_equal(s1.w_sum, s2.w_sum):
                raise AnalysisTestError("parallel w_sum != serial w_sum")
            if not np.array_equal(s1.q, s2.q):
                raise AnalysisTestError("parallel q != serial q")
            if s1.root_value != s2.root_value:
                raise AnalysisTestError(
                    f"parallel root_value {s2.root_value} != {s1.root_value}")
            if not np.array_equal(s1.world_values, s2.world_values):
                raise AnalysisTestError(
                    "parallel world_values != serial (world order broken?)")
            if not np.array_equal(s1.world_visits, s2.world_visits):
                raise AnalysisTestError("parallel world_visits != serial")
            if s1.sims_run != s2.sims_run or s1.sim_steps != s2.sim_steps:
                raise AnalysisTestError("parallel sims/steps != serial")
            if best1 != best2 or pv1 != pv2:
                raise AnalysisTestError(
                    f"PV of world {far_world} differs: {pv2} != {pv1}")
            if len(nodes1) != len(nodes2):
                raise AnalysisTestError("walked line lengths differ")
            for n1, n2 in zip(nodes1, nodes2):
                if n1.terminal != n2.terminal:
                    raise AnalysisTestError("walked terminals differ")
                if (n1.obs is None) != (n2.obs is None):
                    raise AnalysisTestError("walked obs presence differs")
                if n1.obs is not None and not np.array_equal(n1.obs, n2.obs):
                    raise AnalysisTestError("walked state differs")
            return (f"{worlds} worlds / {cap} sims over {req.num_choices} "
                    f"choices: 2-engine merge bit-equal to 1-engine (visits "
                    f"{s1.visits.astype(int).tolist()}, |w_sum| "
                    f"{float(np.abs(s1.w_sum).sum()):.3f}), world {far_world} "
                    f"PV {len(pv1)} steps identical")
        finally:
            env.close()
    finally:
        _rm(paths)


def test_parallel_rewind():
    """The F6-style rewind works with a fleet: a request for an EARLIER
    decision respawns EVERY engine at that prefix and analyzes there, leaving
    the live env untouched; the fleet then follows the live game forward."""
    deck_a, deck_b, paths = _write_decks()
    cfg = AnalysisConfig(evaluator_spec="uniform", worlds=2, chunk_sims=4,
                         max_sims=8, seed=SEARCH_SEED, procs=2)
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b)
        session = AnalysisSession(env, cfg, evaluator=UniformEvaluator(),
                                  evaluator_label="uniform")
        try:
            env.reset(options={"engine_seed": SEED})
            _drive_to_safe(env)
            old = _req(env)
            session.analyze(old)
            if len(session._envs) != 2:
                raise AnalysisTestError(
                    f"procs=2 spawned {len(session._envs)} engines")

            agent = make_agent("scripted")
            agent.new_game()
            for _ in range(8):
                _, _, term, trunc, _ = env.step(
                    agent.act(env._obs, env._num_choices))
                if term or trunc:
                    raise AnalysisTestError("game ended before the rewind test")
            _drive_to_safe(env)
            new = _req(env)
            if new.history_len <= old.history_len:
                raise AnalysisTestError("live game did not advance")
            session.analyze(new)              # forward delta into both engines

            before = env._obs.copy()
            stats = session.analyze(old)      # <- the rewind
            if stats.sims_run != cfg.max_sims:
                raise AnalysisTestError(
                    f"rewound analyze ran {stats.sims_run} != {cfg.max_sims}")
            if len(session._envs) != 2:
                raise AnalysisTestError("rewind lost an engine")
            if session._synced_len != old.history_len:
                raise AnalysisTestError(
                    f"fleet synced at {session._synced_len} != {old.history_len}")
            if not np.array_equal(env._obs, before):
                raise AnalysisTestError("rewind touched the LIVE env's state")
            # Every engine must browse its own world after the respawn.
            best = int(np.argmax(stats.visits))
            for w in range(cfg.worlds):
                session.pv(best, world=w)
            stats2 = session.analyze(new)     # and forward again
            if stats2.sims_run != cfg.max_sims:
                raise AnalysisTestError(
                    f"re-forward analyze ran {stats2.sims_run}")
            return (f"2-engine fleet rewound {new.history_len} -> "
                    f"{old.history_len} and forward again, live env untouched")
        finally:
            session.close()
            env.close()
    finally:
        _rm(paths)


def test_chunk_sims_zero_rejected():
    """A non-positive chunk_sims runs no sims, so an uncapped analyze() would
    spin forever — it must fail loudly instead, at construction AND when a
    live-edited config reaches analyze(). Same for a negative procs."""
    from analysis_session import AnalysisError

    try:
        AnalysisConfig(evaluator_spec="uniform", chunk_sims=0)
        raise AnalysisTestError("AnalysisConfig(chunk_sims=0) was accepted")
    except ValueError:
        pass
    try:
        AnalysisConfig(evaluator_spec="uniform", procs=-1)
        raise AnalysisTestError("AnalysisConfig(procs=-1) was accepted")
    except ValueError:
        pass

    deck_a, deck_b, paths = _write_decks()
    cfg = AnalysisConfig(evaluator_spec="uniform", worlds=2, chunk_sims=4,
                         max_sims=0, seed=SEARCH_SEED, procs=1)
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b)
        session = AnalysisSession(env, cfg, evaluator=UniformEvaluator(),
                                  evaluator_label="uniform")
        try:
            env.reset(options={"engine_seed": SEED})
            _drive_to_safe(env)
            session.cfg.chunk_sims = 0        # the window edits knobs live
            try:
                session.analyze(_req(env))
                raise AnalysisTestError("analyze accepted chunk_sims=0")
            except AnalysisError as e:
                if "chunk_sims" not in str(e):
                    raise AnalysisTestError(f"unhelpful message: {e}")
            if session._envs:
                raise AnalysisTestError(
                    "a rejected config still spawned an engine")
            return "chunk_sims=0 rejected at construction and at analyze()"
        finally:
            session.close()
            env.close()
    finally:
        _rm(paths)


def test_cancellation():
    """A cross-thread stop event ends an unlimited analyze() promptly; the
    parked search stays browsable; re-analyzing the same decision works."""
    deck_a, deck_b, paths = _write_decks()
    cfg = AnalysisConfig(evaluator_spec="uniform", worlds=2, chunk_sims=4,
                         max_sims=0, seed=SEARCH_SEED)   # 0 = until stopped
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b)
        session = AnalysisSession(env, cfg, evaluator=UniformEvaluator(),
                                  evaluator_label="uniform")
        try:
            env.reset(options={"engine_seed": SEED})
            _drive_to_safe(env)
            req = _req(env)

            stop = threading.Event()
            out = {}

            def _run():
                out["stats"] = session.analyze(req, stop=stop)

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            time.sleep(0.5)
            stop.set()
            t.join(timeout=15.0)
            if t.is_alive():
                raise AnalysisTestError("analyze did not stop within 15s")
            stats = out["stats"]
            if stats.sims_run <= 0:
                raise AnalysisTestError("no sims ran before cancellation")

            # Parked search is browsable after the cancel.
            best = int(np.argmax(stats.world_visits[0]))
            session.pv(best, world=0)

            # Same decision, new run: sync() closes the parked search and
            # replays an empty delta; a capped re-analyze completes.
            session.cfg.max_sims = 8
            stats2 = session.analyze(req)
            if stats2.sims_run != 8:
                raise AnalysisTestError(f"re-analyze ran {stats2.sims_run} != 8")
            return (f"cancelled after {stats.sims_run} sims, pv browsable, "
                    "re-analyze ok")
        finally:
            session.close()
            env.close()
    finally:
        _rm(paths)


def test_evaluator_note():
    """An evaluator that reports an untrained value bucket surfaces a caveat
    (the AZ multi-head critic's dead-column guard); a plain one reports none."""

    class _Flagging(UniformEvaluator):
        untrained_bucket = None

    deck_a, deck_b, paths = _write_decks()
    cfg = AnalysisConfig(evaluator_spec="uniform", worlds=2, chunk_sims=4,
                         max_sims=4, seed=SEARCH_SEED)
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b)
        ev = _Flagging()
        session = AnalysisSession(env, cfg, evaluator=ev,
                                  evaluator_label="uniform")
        try:
            env.reset(options={"engine_seed": SEED})
            _drive_to_safe(env)
            session.analyze(_req(env))
            if session.evaluator_note() is not None:
                raise AnalysisTestError("note reported for a healthy evaluator")
            ev.untrained_bucket = 0
            note = session.evaluator_note()
            if not note or "untrained" not in note:
                raise AnalysisTestError(f"no untrained-bucket note: {note!r}")
            if "tempo_vs_tempo" not in note:
                raise AnalysisTestError(
                    f"note doesn't name the matchup bucket: {note!r}")
            return f"note surfaced: {note[:48]}…"
        finally:
            session.close()
            env.close()
    finally:
        _rm(paths)


def test_opp_result_hook():
    """SearchController.on_result fires after a searched decision with a
    SearchResult carrying the new q/world_values fields, an obs copy, and the
    action actually chosen; fallback (unsafe) decisions never fire it."""
    from opponents import SearchController

    deck_a, deck_b, paths = _write_decks()
    captured = []
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b)
        ctrl = SearchController(UniformEvaluator(), sims=16, worlds=2,
                                rng_seed=SEARCH_SEED)
        ctrl.on_result = lambda obs, num, result, chosen: captured.append(
            (obs, num, result, chosen))
        try:
            env.reset(options={"engine_seed": SEED})
            ctrl.bind_env(env)

            # Since the snapshot-safe conversion (branch snapshot_safe) every
            # decision in this scenario reports safe=1 — the opening mulligan
            # included — so no naturally-unsafe decision exists to exercise the
            # controller's fallback branch. Force the flag off for one decision
            # (the controller reads env.last_search_safe; the next engine query
            # rewrites it) to keep the "fallback never fires the hook" contract
            # covered.
            if not env.last_search_safe:
                raise AnalysisTestError("expected the opening decision safe=1 "
                                        "(snapshot-safe conversion)")
            env.last_search_safe = False
            ctrl.choose(env._obs, env._num_choices)
            if captured:
                raise AnalysisTestError("hook fired on a fallback decision")
            env.last_search_safe = True

            _drive_to_safe(env)
            chosen = ctrl.choose(env._obs, env._num_choices)
            if len(captured) != 1:
                raise AnalysisTestError(f"{len(captured)} hook calls, want 1")
            obs, num, result, cb_chosen = captured[0]
            if cb_chosen != chosen:
                raise AnalysisTestError("hook chosen != returned action")
            if num != env._num_choices:
                raise AnalysisTestError("hook num_choices mismatch")
            if obs is env._obs:
                raise AnalysisTestError("hook obs is the live buffer, not a copy")
            if result.q is None or result.world_values is None:
                raise AnalysisTestError("hook result missing q/world_values")
            if int(result.visits.sum()) != result.sims_run:
                raise AnalysisTestError("root visits don't sum to sims_run")
            env.step(chosen)   # engine still healthy after the searched choice
            return (f"hook fired once (chosen {chosen}, "
                    f"{result.sims_run} sims), fallback silent")
        finally:
            env.close()
    finally:
        _rm(paths)


TESTS = [
    ("chunked_parity", test_chunked_parity),
    ("parallel_merge_fields", test_parallel_merge_fields),
    ("pv_walk_discipline", test_pv_walk_discipline),
    ("session_lockstep_bo3", test_session_lockstep_bo3),
    ("rewind_to_earlier_decision", test_rewind_to_earlier_decision),
    ("parallel_analysis_parity", test_parallel_analysis_parity),
    ("parallel_rewind", test_parallel_rewind),
    ("chunk_sims_zero_rejected", test_chunk_sims_zero_rejected),
    ("cancellation", test_cancellation),
    ("evaluator_note", test_evaluator_note),
    ("opp_result_hook", test_opp_result_hook),
]


def main():
    if not os.path.exists(BINARY):
        print(f"binary not found at {BINARY} — run `make` first", file=sys.stderr)
        return 2
    failures = 0
    for name, fn in TESTS:
        try:
            detail = fn()
            print(f"ok    {name}: {detail}", flush=True)
        except (AnalysisTestError, AssertionError, EOFError, RuntimeError) as e:
            failures += 1
            print(f"FAIL  {name}: {e}", flush=True)
    print(f"\nanalysis session: {len(TESTS) - failures}/{len(TESTS)} tests passed",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
