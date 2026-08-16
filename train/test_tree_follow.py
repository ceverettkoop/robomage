"""Standalone checks for search-tree reuse: the revealed-hidden-info
fingerprint (decode.hidden_info_fingerprint) and the SearchController
_try_follow_tree gates (opponent-action categories, reveal ratchet, visit
threshold, winning-action share threshold, menu/seat match).

Pure Python on synthetic observation vectors and hand-built _Node trees — no
engine binary, no torch. Run standalone (not wired into a ci_check tier,
matching test_match_clock.py):
    train/.venv/bin/python train/test_tree_follow.py
"""

import sys

import numpy as np

from env import (OBS_SIZE, STATE_SIZE, _SELF_IS_A_IDX,
                 _HAND_START, _HAND_SLOT_SIZE, MAX_HAND_SLOTS,
                 _SELF_PERM_START, _OPP_PERM_START, _PERM_SLOT_SIZE,
                 _PERM_SLOTS, _PERM_CARD_OFF,
                 _GY_START, _EXILE_START, _GY_SLOT_SIZE, MAX_GY_SLOTS,
                 _KNOWN_TOP_LIB_START, _KNOWN_TOP_LIB_SLOTS,
                 _KNOWN_TOP_LIB_SLOT_SIZE,
                 _OPP_KNOWN_HAND_START, _OPP_KNOWN_HAND_SLOTS,
                 _OPP_KNOWN_HAND_SLOT_SIZE,
                 _STACK_START, _STACK_SLOT_SIZE, _STACK_SLOTS)
from _enums import (CAT_CAST_SPELL, CAT_PASS_PRIORITY, CAT_ACTIVATE_ABILITY,
                    CAT_SELECT_ATTACKER)
from card_costs import N_CARD_TYPES
from decode import hidden_info_fingerprint, _OPP_GY_START, _OPP_EXILE_START
from mcts import _Node
from opponents import SearchController

FAILURES: list[str] = []

_SENT = -1.0 / N_CARD_TYPES
_TOKEN_IDX = N_CARD_TYPES - 1


def check(cond: bool, msg: str) -> None:
    if cond:
        return
    FAILURES.append(msg)
    print(f"FAIL: {msg}", file=sys.stderr)


def blank_obs():
    """A zeroed obs with every fingerprint-read card-id position set to the
    empty sentinel (0.0 would decode as vocab index 0, not empty)."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    for i in range(MAX_HAND_SLOTS):
        obs[_HAND_START + i * _HAND_SLOT_SIZE] = _SENT
    for start in (_SELF_PERM_START, _OPP_PERM_START):
        for s in range(_PERM_SLOTS):
            obs[start + s * _PERM_SLOT_SIZE + _PERM_CARD_OFF] = _SENT
    for start in (_GY_START, _OPP_GY_START, _EXILE_START, _OPP_EXILE_START):
        for i in range(MAX_GY_SLOTS):
            obs[start + i * _GY_SLOT_SIZE] = _SENT
    for i in range(_KNOWN_TOP_LIB_SLOTS):
        obs[_KNOWN_TOP_LIB_START + i * _KNOWN_TOP_LIB_SLOT_SIZE] = _SENT
    for i in range(_OPP_KNOWN_HAND_SLOTS):
        obs[_OPP_KNOWN_HAND_START + i * _OPP_KNOWN_HAND_SLOT_SIZE] = _SENT
    for s in range(_STACK_SLOTS):
        obs[_STACK_START + s * _STACK_SLOT_SIZE + 1] = _SENT
    obs[_SELF_IS_A_IDX] = 1.0
    return obs


def enc(idx):
    return idx / N_CARD_TYPES


def gained(fp_then, fp_now):
    return bool(np.any(fp_now[0] > fp_then[0]) or np.any(fp_now[1] > fp_then[1]))


def test_fingerprint() -> None:
    # Movement between visible zones conserves counts: cast from hand -> stack
    # -> graveyard is never a "reveal".
    a = blank_obs()
    a[_HAND_START] = enc(7)
    b = blank_obs()
    b[_GY_START] = enc(7)
    check(not gained(hidden_info_fingerprint(a[:STATE_SIZE]),
                     hidden_info_fingerprint(b[:STATE_SIZE])),
          "hand -> graveyard move must not read as a reveal")

    # A draw (hand gains an identity from nowhere visible) is a reveal.
    c = blank_obs()
    c[_HAND_START] = enc(7)
    c[_HAND_START + _HAND_SLOT_SIZE] = enc(9)
    check(gained(hidden_info_fingerprint(a[:STATE_SIZE]),
                 hidden_info_fingerprint(c[:STATE_SIZE])),
          "a drawn card must read as a reveal")

    # Drawing a KNOWN top-of-library card conserves counts (no new info).
    top = blank_obs()
    top[_KNOWN_TOP_LIB_START] = enc(7)
    drawn = blank_obs()
    drawn[_HAND_START] = enc(7)
    check(not gained(hidden_info_fingerprint(top[:STATE_SIZE]),
                     hidden_info_fingerprint(drawn[:STATE_SIZE])),
          "drawing a known top card must not read as a reveal")

    # A new known-top id (Ponder/scry writing the block) IS a reveal.
    check(gained(hidden_info_fingerprint(blank_obs()[:STATE_SIZE]),
                 hidden_info_fingerprint(top[:STATE_SIZE])),
          "a newly known top-of-library card must read as a reveal")

    # An opponent card appearing on the stack (ctrl float 0.0 = opp) is a
    # reveal on the opponent side.
    opp = blank_obs()
    opp[_STACK_START + 1] = enc(12)
    check(gained(hidden_info_fingerprint(blank_obs()[:STATE_SIZE]),
                 hidden_info_fingerprint(opp[:STATE_SIZE])),
          "an opponent stack object must read as a reveal")

    # Token permanents are excluded — token creation is fully public.
    tok = blank_obs()
    tok[_SELF_PERM_START + _PERM_CARD_OFF] = enc(_TOKEN_IDX)
    check(not gained(hidden_info_fingerprint(blank_obs()[:STATE_SIZE]),
                     hidden_info_fingerprint(tok[:STATE_SIZE])),
          "a token permanent must not read as a reveal")

    if not FAILURES:
        print("PASS: hidden-info fingerprint semantics")


class StubEvaluator:
    def evaluate(self, obs, num_choices):
        return np.full(num_choices, 1.0 / num_choices, dtype=np.float64), 0.0


class FakeEnv:
    def __init__(self):
        self._action_history = []
        self._action_meta = []


def arm(ctrl, env, tree_roots, obs):
    ctrl.bind_env(env)
    ctrl._followed_trees = tree_roots
    ctrl._followed_hist_len = len(env._action_history)
    ctrl._followed_fp = hidden_info_fingerprint(obs[:STATE_SIZE])


def make_chain(*, leaf_visits=(30, 10)):
    """root --a=1--> mid --a=0--> leaf (2 choices, our seat)."""
    root = _Node(3, np.full(3, 1.0 / 3.0), True)
    mid = _Node(2, np.full(2, 0.5), False)
    leaf = _Node(2, np.full(2, 0.5), True)
    leaf.N = np.array(leaf_visits, dtype=np.int64)
    root.children[1] = mid
    mid.children[0] = leaf
    return root


def test_follow_gates() -> None:
    base = blank_obs()

    # Happy path: own action then a safe opponent pass -> follow to the leaf,
    # answer argmax visits, and advance the follow roots.
    ctrl = SearchController(StubEvaluator(), rng_seed=1)
    env = FakeEnv()
    root = make_chain()
    arm(ctrl, env, [root], base)
    env._action_history += [1, 0]
    env._action_meta += [(True, CAT_CAST_SPELL), (False, CAT_PASS_PRIORITY)]
    a = ctrl._try_follow_tree(base, 2)
    check(a == 0, f"expected follow to answer 0, got {a}")
    check(ctrl._followed_trees == [root.children[1].children[0]],
          "follow should advance the stored roots to the reached node")

    # Safe opponent combat declaration also follows.
    ctrl2 = SearchController(StubEvaluator(), rng_seed=1)
    env2 = FakeEnv()
    root2 = make_chain()
    arm(ctrl2, env2, [root2], base)
    env2._action_history += [1, 0]
    env2._action_meta += [(True, CAT_CAST_SPELL), (False, CAT_SELECT_ATTACKER)]
    check(ctrl2._try_follow_tree(base, 2) == 0,
          "safe opponent combat action should be followed through")

    # Gate 1: an unsafe opponent action (activate — hand-dependent menu
    # indices) forces a re-search even though the path exists.
    ctrl3 = SearchController(StubEvaluator(), rng_seed=1)
    env3 = FakeEnv()
    arm(ctrl3, env3, [make_chain()], base)
    env3._action_history += [1, 0]
    env3._action_meta += [(True, CAT_CAST_SPELL), (False, CAT_ACTIVATE_ABILITY)]
    check(ctrl3._try_follow_tree(base, 2) is None,
          "unsafe opponent action must force a re-search")
    check(ctrl3._followed_trees is None, "unsafe opp action should drop the trees")

    # Gate 2: a reveal since the search (drawn card) forces a re-search even
    # on a fully explored path.
    ctrl4 = SearchController(StubEvaluator(), rng_seed=1)
    env4 = FakeEnv()
    arm(ctrl4, env4, [make_chain()], base)
    env4._action_history += [1, 0]
    env4._action_meta += [(True, CAT_CAST_SPELL), (False, CAT_PASS_PRIORITY)]
    revealed = base.copy()
    revealed[_HAND_START] = enc(7)
    check(ctrl4._try_follow_tree(revealed, 2) is None,
          "a revealed card must force a re-search")
    check(ctrl4._followed_trees is None, "a reveal should drop the trees")

    # Thin visits at the reached node fall through to a search.
    ctrl5 = SearchController(StubEvaluator(), rng_seed=1)
    env5 = FakeEnv()
    arm(ctrl5, env5, [make_chain(leaf_visits=(3, 2))], base)
    env5._action_history += [1, 0]
    env5._action_meta += [(True, CAT_CAST_SPELL), (False, CAT_PASS_PRIORITY)]
    check(ctrl5._try_follow_tree(base, 2) is None,
          "a thin-visit node must not be followed")

    # A contested node — enough total visits but the winning action below the
    # _FOLLOW_MIN_SHARE cut — also falls through to a search.
    ctrl5b = SearchController(StubEvaluator(), rng_seed=1)
    env5b = FakeEnv()
    arm(ctrl5b, env5b, [make_chain(leaf_visits=(18, 14))], base)
    env5b._action_history += [1, 0]
    env5b._action_meta += [(True, CAT_CAST_SPELL), (False, CAT_PASS_PRIORITY)]
    check(ctrl5b._try_follow_tree(base, 2) is None,
          "a contested node (winner below the share cut) must not be followed")
    check(ctrl5b._followed_trees is None,
          "a contested node should drop the trees")

    # A share exactly at the cut is trusted (the gate is strictly-below).
    ctrl5c = SearchController(StubEvaluator(), rng_seed=1)
    env5c = FakeEnv()
    arm(ctrl5c, env5c, [make_chain(leaf_visits=(33, 17))], base)
    env5c._action_history += [1, 0]
    env5c._action_meta += [(True, CAT_CAST_SPELL), (False, CAT_PASS_PRIORITY)]
    check(ctrl5c._try_follow_tree(base, 2) == 0,
          "a winner holding exactly the share cut should be followed")

    # Menu-size mismatch at the reached node is a divergence.
    ctrl6 = SearchController(StubEvaluator(), rng_seed=1)
    env6 = FakeEnv()
    arm(ctrl6, env6, [make_chain()], base)
    env6._action_history += [1, 0]
    env6._action_meta += [(True, CAT_CAST_SPELL), (False, CAT_PASS_PRIORITY)]
    check(ctrl6._try_follow_tree(base, 4) is None,
          "a menu-size mismatch must not be followed")

    # An unexplored real action (no child) is a divergence.
    ctrl7 = SearchController(StubEvaluator(), rng_seed=1)
    env7 = FakeEnv()
    arm(ctrl7, env7, [make_chain()], base)
    env7._action_history += [2]
    env7._action_meta += [(True, CAT_CAST_SPELL)]
    check(ctrl7._try_follow_tree(base, 2) is None,
          "an unexplored real action must not be followed")

    if not FAILURES:
        print("PASS: tree-follow gates")


def main() -> int:
    test_fingerprint()
    test_follow_gates()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s)", file=sys.stderr)
        return 1
    print("tree follow: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
