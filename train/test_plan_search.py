#!/usr/bin/env python3
"""Sideboard plan-search regression (mcts.run_plan_search / IncrementalPlanSearch).

A bo3 sideboard root is searched as a flat set of complete sideboard
configurations ("plans") — see the plan-search section of train/mcts.py. This
test drives a real bo3 match to its first sideboard prompt and pins the
contract every consumer relies on:

  (1) coverage — every legal root action (Done included) gets a variant-0 plan;
  (2) accounting — each plan is priced on EVERY world, plan value = the world
      mean, Q per first pick = its best plan's value, pi = softmax(Q/SB_PI_TAU),
      visits = pi * evaluations, best_action == argmax Q, root_value = pi @ Q;
  (3) determinism — a repeat under the same pinned world seeds is bit-identical;
  (4) memo — re-searching with the boundary-shared memo prices every plan from
      cache (no rollout engine steps beyond generation), and the memo carries
      across the boundary's consecutive picks (memo_picks);
  (5) extras — deterministic alternate completions: at most `branches` novel
      configurations beyond coverage, never re-priced duplicates counted;
  (6) IncrementalPlanSearch (the analysis window's chunked twin) reaches the
      same plans, Q and visits as the one-shot function under the same seeds;
  (7) rollout_memo_eligible excludes multisets holding one card in BOTH pick
      directions (pure-function check).

Torch-free (a deterministic non-uniform fake evaluator; priors depend only on
the menu size, values on the observation). Needs bin/robomage built.

Wired into ``train/ci_check.py`` as the ``plansearch`` tier (so ``make check``
runs it); also runnable standalone::

    train/.venv/bin/python train/test_plan_search.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import _IS_SIDEBOARD_IDX  # noqa: E402
from _enums import CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT  # noqa: E402
from cli_spec import BINARY  # noqa: E402
from search_env import SearchRoboMageEnv  # noqa: E402
from mcts import (SB_PI_TAU, IncrementalPlanSearch, _plan_softmax,  # noqa: E402
                  rollout_memo_eligible, run_plan_search, sb_pick_descriptor)

DECK_A = "league/ur_delver"
DECK_B = "league/burn"
SEED = 7
WORLDS = 2
BRANCHES = 2
# Deep enough that games played with DIFFERENT sideboard configurations
# actually diverge before the horizon (a 1-turn horizon can land on states
# where the plan's effect is still hidden information to the mover, collapsing
# every plan to one value); still small enough for a CI tier.
ROLLOUT_TURNS = 3


class FakeEvaluator:
    """Deterministic, torch-free, NON-uniform.

    Priors put the max on index 0 (Done / Keep / Pass in the engine's menus, so
    greedy completions terminate instead of saturating the swap cap) with a
    varied tail (so the extras' second-best deviation is meaningful). The value
    is a whole-obs hash, so any state divergence separates plan values."""

    def evaluate(self, obs, num_choices):
        idx = np.arange(num_choices, dtype=np.float64)
        p = 1.0 + ((idx * 7.0) % 5.0) / 10.0
        p[0] = 2.0
        p /= p.sum()
        v = float(np.tanh(float(np.sum(obs)) * 0.003))
        return p, v


def _drive_to_sideboard(env):
    obs, _ = env.reset(options={"engine_seed": SEED})
    for _ in range(5000):
        if obs[_IS_SIDEBOARD_IDX] > 0.5:
            return obs
        obs, _r, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    raise AssertionError("never reached a sideboard prompt")


def main() -> int:
    if not os.path.exists(BINARY):
        print(f"FAIL: engine binary not found at {BINARY} — build it first")
        return 1
    failures = 0

    def check(cond, msg):
        nonlocal failures
        if cond:
            print(f"ok  {msg}")
        else:
            failures += 1
            print(f"FAIL {msg}")

    # (7) pure-function: a card picked in BOTH directions (the forced OUT of a
    # stranded IN can cut the name just sided in) is never memoizable.
    good = ((CAT_SIDEBOARD_IN, 5, 1), (CAT_SIDEBOARD_OUT, 9, 1))
    both_ways = good + ((CAT_SIDEBOARD_OUT, 5, 1),)
    check(rollout_memo_eligible(good)
          and not rollout_memo_eligible(both_ways),
          "(7) rollout_memo_eligible admits swaps, excludes both-direction picks")

    env = SearchRoboMageEnv(deck_a=DECK_A, deck_b=DECK_B, bo3=True)
    ev = FakeEvaluator()
    try:
        _drive_to_sideboard(env)
        root_obs = env._obs.copy()
        n = env._num_choices
        memo: dict = {}
        plans: list = []
        res = run_plan_search(env, ev, worlds=WORLDS, branches=BRANCHES,
                              rollout_turns=ROLLOUT_TURNS,
                              rng=np.random.default_rng(1), plan_memo=memo,
                              plans_out=plans)

        # (1) coverage
        covered = {p.first_action for p in plans if p.variant == 0}
        check(covered == set(range(n)),
              f"(1) coverage: every root action has a variant-0 plan ({n})")

        # (2) accounting
        ok_vals = all(p.values.shape == (WORLDS,)
                      and p.value == float(p.values.mean()) for p in plans)
        q_re = np.full(n, -np.inf)
        for p in plans:
            q_re[p.first_action] = max(q_re[p.first_action], p.value)
        pi = _plan_softmax(q_re, SB_PI_TAU)
        n_evals = len(plans) * WORLDS
        check(ok_vals and np.array_equal(res.q, q_re),
              "(2a) plan value = world mean; Q = best plan per first pick")
        check(np.array_equal(res.visits, pi * float(n_evals))
              and res.sims_run == n_evals,
              "(2b) visits = softmax(Q/tau) * evaluations; sims_run = evals")
        check(res.best_action() == int(np.argmax(res.q))
              and res.root_value == float(pi @ q_re),
              "(2c) best_action = argmax Q; root_value = pi @ Q")
        check(float(np.ptp(res.q)) > 0.0,
              "(2d) the fake evaluator actually separates first-pick Q values")

        # (3) determinism under pinned seeds
        res2 = run_plan_search(env, ev, worlds=WORLDS, branches=BRANCHES,
                               rollout_turns=ROLLOUT_TURNS,
                               world_seeds=res.seeds, plan_memo={})
        check(np.array_equal(res.visits, res2.visits)
              and np.array_equal(res.q, res2.q)
              and res.root_value == res2.root_value,
              "(3) repeat under the same world seeds is bit-identical")

        # (4a) full-memo re-search: every MEMOIZABLE (plan, world) prices from
        # cache (an extras plan holding one card in both directions is never
        # memoized — its greedy completion genuinely depends on pick order).
        plans3: list = []
        res3 = run_plan_search(env, ev, worlds=WORLDS, branches=BRANCHES,
                               rollout_turns=ROLLOUT_TURNS,
                               world_seeds=res.seeds, plan_memo=memo,
                               plans_out=plans3)
        want_hits = WORLDS * sum(
            1 for p in plans3 if rollout_memo_eligible(p.multiset))
        check(res3.memo_hits == want_hits and want_hits > 0
              and np.array_equal(res3.q, res.q)
              and res3.sim_steps < res.sim_steps,
              f"(4a) memo re-search: {res3.memo_hits}/{want_hits} memoizable "
              f"evals from cache, Q identical, steps "
              f"{res.sim_steps}->{res3.sim_steps}")

        # (5) extras: at most `branches` novel configurations beyond coverage
        extras = [p for p in plans if p.variant > 0]
        novel = {p.multiset for p in extras} - {
            p.multiset for p in plans if p.variant == 0}
        check(len(novel) <= BRANCHES,
              f"(5) extras: {len(novel)} novel configuration(s) "
              f"<= branches ({BRANCHES}); {len(extras)} attempted")

        # (6) IncrementalPlanSearch equivalence under the same seeds
        inc = IncrementalPlanSearch(env, ev, worlds=WORLDS, branches=BRANCHES,
                                    rollout_turns=ROLLOUT_TURNS,
                                    world_seeds=res.seeds)
        try:
            while not inc.done:
                inc.run_chunk(16)
            s = inc.stats()
            check(np.array_equal(s.q, res.q)
                  and np.array_equal(s.visits, res.visits)
                  and len(inc.plans) == len(plans),
                  "(6a) IncrementalPlanSearch reaches the one-shot Q/visits")
            best = res.best_action()
            pv = inc.pv(best, 0)
            check(len(pv) >= 1 and pv[0].action == best,
                  "(6b) pv(best) returns the best plan's pick sequence")
            walked = inc.walk(0, [st.action for st in pv])
            check(len(walked) >= 1,
                  "(6c) walk replays the plan's picks in a world")
        finally:
            inc.close()

        # (4b) boundary carry: play the best pick, search the next root with
        # the shared memo + latched descriptor.
        best = res.best_action()
        d = sb_pick_descriptor(root_obs, best)
        obs, _r, _t, _tr, _ = env.step(best)
        if obs[_IS_SIDEBOARD_IDX] > 0.5 and env._num_choices > 1:
            res4 = run_plan_search(env, ev, worlds=WORLDS, branches=BRANCHES,
                                   rollout_turns=ROLLOUT_TURNS,
                                   world_seeds=res.seeds, plan_memo=memo,
                                   memo_picks=(d,) if d is not None else ())
            check(res4.memo_hits > 0,
                  f"(4b) boundary carry: next pick hit the memo "
                  f"{res4.memo_hits} time(s)")
        else:
            print("ok  (4b) skipped — best first pick ended the phase")
    finally:
        env.close()

    if failures:
        print(f"\nplan search: FAILED ({failures} failure(s))")
        return 1
    print("\nplan search: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
