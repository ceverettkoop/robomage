"""Standalone checks for duplicate-edge merging (decode.menu_merge_reps and the
_Node-level fold/select/walk behavior in mcts.py).

Pure Python on synthetic observation vectors — no engine binary, no torch.
The cross-language contract (the C++ twin in src/actor/menu_merge.h producing
the same partition, hence bit-identical visits) is covered by
train/test_mcts_parity.py; this file pins the Python-side semantics.

Run standalone (not wired into a ci_check tier, matching test_trivial_menu.py):
    train/.venv/bin/python train/test_menu_merge.py
"""

import sys

import numpy as np

from env import (OBS_SIZE, ACTION_CATEGORY_MAX,
                 ACT_CATS_START, ACT_IDS_START, ACT_CTRL_START,
                 ACT_ZONE_START, ACT_REFS_START, ACT_ORDS_START,
                 N_ENTITY_REF_SLOTS, _ACTION_CTRL_NULL)
from _enums import (REF_ZONE_MAX, OPTION_ORDINAL_MAX, CAT_CAST_SPELL,
                    CAT_PLAY_LAND, CAT_PASS_PRIORITY, CAT_SELECT_TARGET,
                    CAT_OTHER_CHOICE, CAT_SEARCH_LIBRARY, CAT_TOP_LIBRARY,
                    CAT_DISCARD, CAT_BOTTOM_DECK_CARD, CAT_PAYING_COSTS,
                    CAT_CHOOSE_CARD, CAT_EXILE_FROM_YARD)
from card_costs import N_CARD_TYPES
from decode import menu_merge_reps
from mcts import _Node, walk_reuse_root

FAILURES: list[str] = []

# ActionRefZone values (see src/classes/gamestate.h / _enums._REF_NAMES).
Z_NONE, Z_SELF_HAND = 0, 3
Z_SELF_GY, Z_OPP_GY, Z_SELF_EX, Z_OPP_EX = 6, 7, 8, 9


def check(cond: bool, msg: str) -> None:
    if cond:
        return
    FAILURES.append(msg)
    print(f"FAIL: {msg}", file=sys.stderr)


def make_obs(actions):
    """Synthetic obs: zero state + the six per-action metadata blocks (same
    encoding as test_trivial_menu.make_obs)."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    for i, a in enumerate(actions):
        ctrl = a.get("ctrl", 1.0)
        obs[ACT_CATS_START + i] = a.get("cat", 0) / ACTION_CATEGORY_MAX
        obs[ACT_IDS_START + i] = a.get("id", -1) / N_CARD_TYPES
        obs[ACT_CTRL_START + i] = (
            _ACTION_CTRL_NULL if ctrl is None else ctrl)
        obs[ACT_ZONE_START + i] = a.get("zone", 0) / REF_ZONE_MAX
        obs[ACT_REFS_START + i] = (
            (a.get("ref", -1) + 1) / N_ENTITY_REF_SLOTS)
        obs[ACT_ORDS_START + i] = (
            (a.get("ord", -1) + 1) / (OPTION_ORDINAL_MAX + 1))
    return obs


def reps(actions):
    return menu_merge_reps(make_obs(actions), len(actions)).tolist()


def test_partition() -> None:
    # Hand-duplicate casts inside a mixed menu: dupes fold to the lowest
    # index, everything else keeps its own edge.
    got = reps([{"cat": CAT_PASS_PRIORITY},
                {"cat": CAT_CAST_SPELL, "id": 7, "zone": Z_SELF_HAND, "ord": 0},
                {"cat": CAT_CAST_SPELL, "id": 7, "zone": Z_SELF_HAND, "ord": 0},
                {"cat": CAT_CAST_SPELL, "id": 9, "zone": Z_SELF_HAND, "ord": 0}])
    check(got == [0, 1, 1, 3], f"mixed-menu hand casts: {got}")

    # Whole-menu duplicates (four Play Mountain) all fold onto index 0.
    land = {"cat": CAT_PLAY_LAND, "id": 7, "zone": Z_SELF_HAND}
    check(reps([land] * 4) == [0, 0, 0, 0], "play-land dupes should all fold")

    # Non-contiguous duplicates map to the LOWEST group member.
    got = reps([{"cat": CAT_CAST_SPELL, "id": 7, "zone": Z_SELF_HAND, "ord": 0},
                {"cat": CAT_CAST_SPELL, "id": 9, "zone": Z_SELF_HAND, "ord": 0},
                {"cat": CAT_CAST_SPELL, "id": 7, "zone": Z_SELF_HAND, "ord": 0}])
    check(got == [0, 1, 0], f"non-contiguous dupes: {got}")

    # Every other whitelisted hand pick merges too.
    for cat in (CAT_DISCARD, CAT_BOTTOM_DECK_CARD, CAT_PAYING_COSTS):
        a = {"cat": cat, "id": 5, "zone": Z_SELF_HAND}
        check(reps([a, a]) == [0, 0], f"hand cat {cat} dupes should merge")

    # Library search picks (zone REF_NONE, null ctrl sentinel).
    for cat in (CAT_SEARCH_LIBRARY, CAT_TOP_LIBRARY):
        a = {"cat": cat, "id": 5, "zone": Z_NONE, "ctrl": None}
        check(reps([a, a]) == [0, 0], f"library cat {cat} dupes should merge")
    # Belt-and-braces: a library pick with a NON-null ctrl does not merge.
    a = {"cat": CAT_SEARCH_LIBRARY, "id": 5, "zone": Z_NONE, "ctrl": 1.0}
    check(reps([a, a]) == [0, 1], "library pick with real ctrl must not merge")

    # Graveyard picks: targets, zone-change picks, escape-cost exiles, and
    # flashback-variant casts (ord distinguishes them from normal casts).
    for cat, ordv in ((CAT_SELECT_TARGET, -1), (CAT_CHOOSE_CARD, -1),
                      (CAT_EXILE_FROM_YARD, -1), (CAT_CAST_SPELL, 4)):
        for zone in (Z_SELF_GY, Z_OPP_GY):
            a = {"cat": cat, "id": 5, "zone": zone, "ord": ordv}
            check(reps([a, a]) == [0, 0],
                  f"GY cat {cat} zone {zone} dupes should merge")

    # Exile: CHOOSE_CARD picks merge; cast-from-exile (ord 7 impulse) does NOT
    # (hidden ImpulseCastPermissions can differ between same-name copies).
    for zone in (Z_SELF_EX, Z_OPP_EX):
        a = {"cat": CAT_CHOOSE_CARD, "id": 5, "zone": zone}
        check(reps([a, a]) == [0, 0], f"exile pick zone {zone} should merge")
    a = {"cat": CAT_CAST_SPELL, "id": 5, "zone": Z_SELF_EX, "ord": 7}
    check(reps([a, a]) == [0, 1], "cast-from-exile must not merge")

    # OTHER_CHOICE and null-id actions never merge.
    a = {"cat": CAT_OTHER_CHOICE, "id": 5, "zone": Z_SELF_HAND}
    check(reps([a, a]) == [0, 1], "OTHER_CHOICE must not merge")
    a = {"cat": CAT_CAST_SPELL, "id": -1, "zone": Z_SELF_HAND}
    check(reps([a, a]) == [0, 1], "null card id must not merge")

    # A slot_ref-carrying action never merges (battlefield/stack picks keep
    # their per-entity edges even if a key collided).
    a = {"cat": CAT_SELECT_TARGET, "id": 5, "zone": Z_SELF_GY, "ref": 3}
    check(reps([a, a]) == [0, 1], "slot_ref-carrying action must not merge")

    # Differing key fields split groups.
    base = {"cat": CAT_CAST_SPELL, "id": 7, "zone": Z_SELF_HAND, "ord": 0}
    for variant in ({"ord": 1}, {"id": 8}, {"zone": Z_SELF_GY},
                    {"ctrl": 0.0}):
        got = reps([base, {**base, **variant}])
        check(got == [0, 1], f"key variant {variant} must split: {got}")

    # rep[0] == 0 always (lowest-index representative invariant).
    for menu in ([{"cat": CAT_PASS_PRIORITY}], [land] * 3):
        check(reps(menu)[0] == 0, "rep[0] must be 0")


def test_node_fold_and_select() -> None:
    priors = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    rep = np.array([0, 1, 1, 3], dtype=np.int16)
    node = _Node(4, priors, True, rep=rep)
    check(np.allclose(node.P, [0.1, 0.5, 0.0, 0.4]),
          f"fold onto representatives: {node.P}")
    check(abs(float(node.P.sum()) - 1.0) < 1e-12, "fold preserves total mass")
    check(np.allclose(priors, [0.1, 0.2, 0.3, 0.4]),
          "fold must not mutate the caller's priors array (shared across "
          "worlds and with SearchResult.priors)")

    # select never returns a merged-away duplicate, even when every
    # representative's Q is negative (the -inf mask, not P=0, must exclude it).
    node.N[:] = [5, 5, 0, 5]
    node.W[:] = [-4.0, -4.0, 0.0, -4.0]
    a = node.select(1.5)
    check(int(rep[a]) == a, f"select returned non-representative {a}")

    # Disabled merging keeps the legacy behavior: P is the caller's array,
    # no mask, select may pick any index.
    legacy = _Node(4, priors, True)
    check(legacy.P is priors and legacy.sel_mask is None and legacy.rep is None,
          "rep=None must leave _Node bit-identical to the legacy layout")


def test_walk_canonicalization() -> None:
    # A real played action may be a merged-away duplicate; the walk must
    # canonicalize it onto the representative before the child lookup.
    priors = np.array([0.5, 0.25, 0.25], dtype=np.float64)
    rep = np.array([0, 1, 1], dtype=np.int16)
    root = _Node(3, priors, True, rep=rep)
    child = _Node(2, np.array([0.5, 0.5]), False)
    root.children[1] = child   # tree only keeps the representative edge
    got = walk_reuse_root(root, [2], 2, False)
    check(got is child, "walk must canonicalize dupe index 2 -> rep 1")
    check(walk_reuse_root(root, [2], 5, False) is None,
          "menu-size mismatch must still fail the walk")


def main() -> int:
    test_partition()
    test_node_fold_and_select()
    test_walk_canonicalization()
    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) FAILED", file=sys.stderr)
        return 1
    print("test_menu_merge: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
