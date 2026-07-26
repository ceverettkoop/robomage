"""Standalone checks for interchangeable-menu collapse (decode.menu_is_interchangeable)
and the SearchController trivial fast path.

Pure Python on synthetic observation vectors — no engine binary, no torch.
Run standalone (not wired into a ci_check tier, matching test_match_clock.py):
    train/.venv/bin/python train/test_trivial_menu.py
"""

import sys

import numpy as np

from env import (OBS_SIZE, STATE_SIZE, MAX_ACTIONS, ACTION_CATEGORY_MAX,
                 ACT_CATS_START, ACT_IDS_START, ACT_CTRL_START,
                 ACT_ZONE_START, ACT_REFS_START, ACT_ORDS_START,
                 N_ENTITY_REF_SLOTS, _STACK_START, _STACK_TGT_START,
                 _SELF_PERM_START, _PERM_SLOT_SIZE, _OFF_ATTACHED_BY,
                 _ACTION_CTRL_NULL)
from _enums import (REF_ZONE_MAX, OPTION_ORDINAL_MAX, CAT_CAST_SPELL,
                    CAT_PLAY_LAND, CAT_PASS_PRIORITY, CAT_SELECT_TARGET,
                    CAT_ACTIVATE_ABILITY, CAT_MANA_W, CAT_OTHER_CHOICE)
from card_costs import N_CARD_TYPES
from decode import (menu_is_interchangeable, entity_ref_features,
                    state_ref_targets)
from opponents import SearchController

FAILURES: list[str] = []

CAT_MANA_G = CAT_MANA_W + 4  # W,U,B,R,G,C serialized order


def check(cond: bool, msg: str) -> None:
    if cond:
        return
    FAILURES.append(msg)
    print(f"FAIL: {msg}", file=sys.stderr)


def make_obs(actions):
    """Synthetic obs: zero state + the six per-action metadata blocks.

    Each action is a dict with keys cat, id (vocab index, -1 null), ctrl
    (1.0/0.0/None), zone, ref, ord — encoded exactly like env.py unpacks them.
    """
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


def set_perm_block(obs, ref, pattern):
    """Write a perm-slot feature block via the shared ref-space resolver."""
    entity_ref_features(obs[:STATE_SIZE], ref)[:] = pattern


def test_metadata_collapse() -> None:
    # Single-choice menus are trivially interchangeable.
    check(menu_is_interchangeable(make_obs([{"cat": CAT_PASS_PRIORITY}]), 1),
          "single-choice menu should be interchangeable")

    # Four "Play Mountain" from a hand of duplicates.
    land = {"cat": CAT_PLAY_LAND, "id": 7, "zone": 3}
    check(menu_is_interchangeable(make_obs([land] * 4), 4),
          "duplicate play-land menu should collapse")

    # Mixed categories (pass + cast) never collapse.
    obs = make_obs([{"cat": CAT_PASS_PRIORITY},
                    {"cat": CAT_CAST_SPELL, "id": 7, "zone": 3}])
    check(not menu_is_interchangeable(obs, 2),
          "mixed-category menu must not collapse")

    # Different card ids never collapse.
    obs = make_obs([{"cat": CAT_CAST_SPELL, "id": 7, "zone": 3},
                    {"cat": CAT_CAST_SPELL, "id": 9, "zone": 3}])
    check(not menu_is_interchangeable(obs, 2),
          "distinct-card menu must not collapse")

    # Same card, different ordinals (modes / X values) never collapse.
    obs = make_obs([{"cat": CAT_CAST_SPELL, "id": 7, "zone": 3, "ord": 0},
                    {"cat": CAT_CAST_SPELL, "id": 7, "zone": 3, "ord": 1}])
    check(not menu_is_interchangeable(obs, 2),
          "distinct-ordinal menu must not collapse")

    # CAT_OTHER_CHOICE is excluded outright (text-only distinctions).
    other = {"cat": CAT_OTHER_CHOICE, "id": 7, "zone": 0}
    check(not menu_is_interchangeable(make_obs([other] * 2), 2),
          "OTHER_CHOICE menus must never collapse")

    # Null card ids (player targets, text prompts) never collapse.
    ptgt = {"cat": CAT_SELECT_TARGET, "id": -1, "ctrl": 0.0}
    check(not menu_is_interchangeable(make_obs([ptgt] * 2), 2),
          "null-card-id menu must not collapse")

    if not FAILURES:
        print("PASS: metadata collapse rules")


def test_entity_refs() -> None:
    pattern = np.arange(1, _PERM_SLOT_SIZE + 1, dtype=np.float32) / 100.0
    tap = lambda ref: {"cat": CAT_MANA_G, "id": 12, "zone": 1, "ref": ref}  # noqa: E731

    # Two untapped identical basics: identical blocks -> collapse.
    obs = make_obs([tap(3), tap(5)])
    set_perm_block(obs, 3, pattern)
    set_perm_block(obs, 5, pattern)
    check(menu_is_interchangeable(obs, 2),
          "identical perm blocks should collapse")

    # A one-float difference (damage, a counter) blocks the collapse.
    obs = make_obs([tap(3), tap(5)])
    set_perm_block(obs, 3, pattern)
    dented = pattern.copy()
    dented[6] += 0.1
    set_perm_block(obs, 5, dented)
    check(not menu_is_interchangeable(obs, 2),
          "differing perm blocks must not collapse")

    # Identical blocks, but a stack announced target points at one of them
    # (opp Bolt targets bear A): the choice is real.
    obs = make_obs([tap(3), tap(5)])
    set_perm_block(obs, 3, pattern)
    set_perm_block(obs, 5, pattern)
    obs[_STACK_START + _STACK_TGT_START + 3] = (3 + 1) / N_ENTITY_REF_SLOTS
    check(3 in state_ref_targets(obs[:STATE_SIZE]),
          "state_ref_targets should see the announced stack target")
    check(not menu_is_interchangeable(obs, 2),
          "an announced stack target on a candidate must block the collapse")

    # Identical blocks, but a third permanent's attachment ref points at one.
    obs = make_obs([tap(3), tap(5)])
    set_perm_block(obs, 3, pattern)
    set_perm_block(obs, 5, pattern)
    obs[_SELF_PERM_START + 10 * _PERM_SLOT_SIZE + _OFF_ATTACHED_BY] = (
        (3 + 1) / N_ENTITY_REF_SLOTS)
    check(not menu_is_interchangeable(obs, 2),
          "an attachment ref at a candidate must block the collapse")

    # Two actions on the SAME entity (two abilities of one permanent) are
    # distinct choices even with identical metadata.
    act = {"cat": CAT_ACTIVATE_ABILITY, "id": 12, "zone": 1, "ref": 3}
    obs = make_obs([act, act])
    set_perm_block(obs, 3, pattern)
    check(not menu_is_interchangeable(obs, 2),
          "same-slot actions must not collapse")

    if not FAILURES:
        print("PASS: entity-ref collapse rules")


class CountingEvaluator:
    """Uniform priors; counts evaluate() calls."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, obs, num_choices):
        self.calls += 1
        return np.full(num_choices, 1.0 / num_choices, dtype=np.float64), 0.0


def test_controller_fast_path() -> None:
    ev = CountingEvaluator()
    ctrl = SearchController(ev, rng_seed=1)

    # Trivial menu: instant first action, no evaluator call, no search.
    obs = make_obs([{"cat": CAT_PLAY_LAND, "id": 7, "zone": 3}] * 4)
    a = ctrl.choose(obs, 4)
    check(a == 0, f"trivial menu should return action 0, got {a}")
    check(ev.calls == 0, "trivial menu must not invoke the evaluator")
    check(ctrl.stats["trivial"] == 1 and ctrl.stats["fallback"] == 0,
          f"trivial tally wrong: {ctrl.stats}")

    # Non-trivial menu with no bound env: the usual evaluator fallback.
    obs = make_obs([{"cat": CAT_CAST_SPELL, "id": 7, "zone": 3},
                    {"cat": CAT_CAST_SPELL, "id": 9, "zone": 3}])
    ctrl.choose(obs, 2)
    check(ev.calls == 1 and ctrl.stats["fallback"] == 1,
          f"non-trivial menu should fall back to the evaluator: {ctrl.stats}")

    if not FAILURES:
        print("PASS: SearchController trivial fast path")


def main() -> int:
    test_metadata_collapse()
    test_entity_refs()
    test_controller_fast_path()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s)", file=sys.stderr)
        return 1
    print("trivial menu: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
