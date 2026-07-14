#!/usr/bin/env python3
"""World-parallel mirror-pool search regression (Stage 10).

The MCTS worlds of a determinized search are independent, so they can be split
across N engine processes kept in lockstep with the primary game. This test
proves the three guarantees the interactive mirror pool relies on:

  1. bit-exact merge — a plain ``run_search`` over W worlds and a
     ``run_search_parallel`` over (primary + mirror) with the SAME rng seed
     (identical world-seed derivation, worlds split 2+2) produce identical summed
     visit counts, sims_run, and num_choices. Fanning the worlds across processes
     changes nothing about the result.
  2. lockstep — with a mirror created at game start, driving a full bo3 (scripted
     agent) forwarding every real action keeps the mirror byte-identical to the
     primary at every decision, including across the GAME_RESULT / sideboard
     boundary (the pool never trips its own obs-equality guard).
  3. fallback on drift — a deliberately desynced mirror is detected on the next
     real step; the pool disables (one stderr warning), the primary plays on, and
     a subsequent ``run_search_parallel`` over the now single-env list delegates
     to a plain ``run_search`` (identical result).

Also checks the ``procs=`` spec knob parses on the torch-free ``mcts:uniform``
grammar.

Torch-free: uses ``mcts.UniformEvaluator`` and the torch-free scripted agent; the
engine is exercised through ``SearchRoboMageEnv`` (spawns bin/robomage with
--search-server). Needs bin/robomage built.

Wired into ``train/ci_check.py`` as the ``mirror`` tier (so ``make check`` runs
it); also runnable standalone::

    train/.venv/bin/python train/test_mirror_search.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import _IS_SIDEBOARD_IDX  # noqa: E402
from cli_spec import BINARY, BIN_DIR  # noqa: E402
from search_env import SearchRoboMageEnv  # noqa: E402
from mcts import UniformEvaluator, run_search, run_search_parallel  # noqa: E402
from scripted_agent import make_agent  # noqa: E402

SEED = 7          # engine seed (deterministic game)
SEARCH_SEED = 1234  # search rng seed (world-seed derivation)
BO3_STEP_CAP = 3000


class MirrorError(AssertionError):
    pass


def _write_decks():
    """A lopsided aggro-vs-lands mirror-test matchup (like the self-play test):
    A (36 Grizzly Bears + 24 Forest) reliably beats B (60 Swamp does nothing)
    fast, so a bo3 finishes cleanly and crosses one sideboard boundary. Each has
    a 15-card sideboard so the sideboard IN menu is a real (searchable) choice."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    a = os.path.join(d, "mirror_test_a.dk")
    b = os.path.join(d, "mirror_test_b.dk")
    with open(a, "w") as f:
        f.write("36 Grizzly Bears\n24 Forest\nSIDEBOARD:\n15 Mountain\n")
    with open(b, "w") as f:
        f.write("60 Swamp\nSIDEBOARD:\n15 Mountain\n")
    return "temp/mirror_test_a", "temp/mirror_test_b", [a, b]


def _rm(paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _drive_to_safe(env, min_nc=2, cap=300):
    """Auto-0 step until a loop-safe decision with >= min_nc choices; the env is
    left parked at that decision."""
    for _ in range(cap):
        if env.last_search_safe and env._num_choices >= min_nc:
            return
        env.step(0)
    raise MirrorError(f"no safe decision with >= {min_nc} choices in {cap} steps")


def test_bitexact_merge():
    """Plain run_search(worlds=4) == run_search_parallel over primary+1 mirror
    (worlds split 2+2), same rng seed => same world seeds => identical visits."""
    deck_a, deck_b, paths = _write_decks()
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b)
        ev = UniformEvaluator()
        try:
            env.reset(options={"engine_seed": SEED})
            _drive_to_safe(env)

            # Single-process reference; run_search restores+releases back to root.
            r1 = run_search(env, ev, worlds=4, sims=16,
                            rng=np.random.default_rng(SEARCH_SEED))

            # Build one mirror (replays the recorded history to the same root).
            env.ensure_mirrors(1)
            envs = env.search_envs()
            if len(envs) != 2:
                raise MirrorError(f"expected primary+1 mirror, got {len(envs)} env(s)")

            r2 = run_search_parallel(envs, ev, worlds=4, sims=16,
                                     rng=np.random.default_rng(SEARCH_SEED))

            if not np.array_equal(r1.visits, r2.visits):
                raise MirrorError(f"visit merge diverged: single {r1.visits} vs "
                                  f"parallel {r2.visits}")
            if r1.sims_run != r2.sims_run:
                raise MirrorError(f"sims_run {r1.sims_run} != {r2.sims_run}")
            if r1.num_choices != r2.num_choices:
                raise MirrorError(f"num_choices {r1.num_choices} != {r2.num_choices}")
            return (f"visits identical ({r1.visits.astype(int).tolist()}), "
                    f"sims_run={r1.sims_run}, worlds 4 split 2+2")
        finally:
            env.close()
    finally:
        _rm(paths)


def test_lockstep_bo3():
    """A mirror created at game start stays byte-identical through a full bo3
    driven by the scripted agent, across the sideboard boundary."""
    deck_a, deck_b, paths = _write_decks()
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b, bo3=True,
                                auto_sideboard=False)
        agent = make_agent("scripted")
        try:
            env.reset(options={"engine_seed": SEED})
            agent.new_game()
            env.ensure_mirrors(1)
            if len(env.search_envs()) != 2:
                raise MirrorError("mirror not created at game start")

            saw_sideboard = False
            decisions = 0
            terminated = False
            for _ in range(BO3_STEP_CAP):
                if env._mirror_pool_disabled:
                    raise MirrorError(f"mirror drifted after {decisions} decisions "
                                      "(pool disabled during a clean lockstep run)")
                obs = env._obs
                if obs[_IS_SIDEBOARD_IDX] > 0.5:
                    saw_sideboard = True
                action = agent.act(obs, env._num_choices)
                _, _, term, trunc, _ = env.step(action)
                decisions += 1
                if term or trunc:
                    terminated = term or trunc
                    break

            if not terminated:
                raise MirrorError(f"bo3 did not finish within {BO3_STEP_CAP} decisions")
            if env._mirror_pool_disabled or len(env._mirrors) != 1:
                raise MirrorError("mirror pool disabled by the end of a clean match")
            if not np.array_equal(env._mirrors[0]._obs, env._obs):
                raise MirrorError("final mirror obs != primary obs")
            if not saw_sideboard:
                raise MirrorError("bo3 never crossed a sideboard boundary — the "
                                  "cross-boundary lockstep was not exercised")
            return (f"{decisions} decisions in lockstep across the sideboard "
                    "boundary, final obs identical")
        finally:
            env.close()
    finally:
        _rm(paths)


def test_fallback_on_drift():
    """A desynced mirror disables the pool on the next real step; the primary is
    unaffected and a parallel search over the single-env list == plain search."""
    deck_a, deck_b, paths = _write_decks()
    try:
        env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b)
        ev = UniformEvaluator()
        try:
            env.reset(options={"engine_seed": SEED})
            _drive_to_safe(env)
            env.ensure_mirrors(1)
            if len(env._mirrors) != 1:
                raise MirrorError("mirror not created")

            # Corrupt: advance the mirror one extra action out of lockstep. The
            # next real step forwards to it and the obs-equality guard trips.
            env._mirrors[0].step(0)
            env.step(0)  # prints one "[search-mirror] pool disabled" line to stderr

            if not env._mirror_pool_disabled:
                raise MirrorError("pool did not disable after a mirror drifted")
            if env._mirrors:
                raise MirrorError("mirrors not closed after drift")
            if env.search_envs() != [env]:
                raise MirrorError("search_envs() should be [primary] after disable")

            # A parallel search over the now single-env list delegates to a plain
            # single-env run_search — identical result.
            _drive_to_safe(env)
            r_par = run_search_parallel(env.search_envs(), ev, worlds=3, sims=12,
                                        rng=np.random.default_rng(SEARCH_SEED))
            r_plain = run_search(env, ev, worlds=3, sims=12,
                                 rng=np.random.default_rng(SEARCH_SEED))
            if not np.array_equal(r_par.visits, r_plain.visits):
                raise MirrorError(f"single-env delegate diverged: {r_par.visits} vs "
                                  f"{r_plain.visits}")
            return "pool disabled on drift, primary intact, single-env delegate exact"
        finally:
            env.close()
    finally:
        _rm(paths)


def test_procs_spec_parsing():
    """The ``procs=`` knob parses on the torch-free mcts:uniform grammar; default
    is 1; a bad value is rejected loudly."""
    from opponents import make_controller

    c = make_controller("mcts:uniform?procs=3")
    if c._procs != 3 or c.stats.get("pool_procs") != 3:
        raise MirrorError(f"procs=3 not applied (procs={c._procs}, "
                          f"pool_procs={c.stats.get('pool_procs')})")
    c1 = make_controller("mcts:uniform")
    if c1._procs != 1:
        raise MirrorError(f"default procs should be 1, got {c1._procs}")
    try:
        make_controller("mcts:uniform?procs=0")
    except ValueError:
        pass
    else:
        raise MirrorError("procs=0 should be rejected with ValueError")
    return "procs= parses (3), defaults to 1, rejects 0"


TESTS = [
    ("procs_spec_parsing", test_procs_spec_parsing),
    ("bitexact_merge", test_bitexact_merge),
    ("lockstep_bo3", test_lockstep_bo3),
    ("fallback_on_drift", test_fallback_on_drift),
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
        except (MirrorError, AssertionError, EOFError, RuntimeError) as e:
            failures += 1
            print(f"FAIL  {name}: {e}", flush=True)
    print(f"\nmirror search: {len(TESTS) - failures}/{len(TESTS)} tests passed",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
