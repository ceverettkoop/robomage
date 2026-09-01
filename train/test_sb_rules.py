#!/usr/bin/env python3
"""Dead-sideboard-card rules regression (sb_dead_rules.json -> sb_rules.py ->
mcts.sb_dead_mask).

Engine-free and torch-free: builds synthetic observation vectors (layout
constants imported from env/_enums, no magic offsets) and pins:

  (1) table integrity — every generated rule keys a real vocab card with a
      nonzero condition mask; the fact table covers the whole vocab; spot
      facts (a dual land carries both land-subtype bits);
  (2) masking — an IN of a ruled card is dead against an opponent 75 lacking
      the condition (Choke / Red Elemental Blast vs a mono-green list) and
      live against one providing it (a list with Volcanic Island);
  (3) only SIDEBOARD_IN actions are ever masked — Done and OUT picks stay
      live whatever the opponent's list;
  (4) outside the sideboard phase the mask is all-false.

The C++ twin (src/actor/sb_rules.h) is asserted bit-identical by the actor
parity gate (test_mcts_parity.py bo3 legs / ci_check --tier actor); this test
pins the Python semantics the twin mirrors.

Wired into ``train/ci_check.py`` as the ``sbrules`` tier (default ``make
check``); also runnable standalone::

    train/.venv/bin/python train/test_sb_rules.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import env  # noqa: E402
from _enums import (ACTION_CATEGORY_MAX, N_CARD_TYPES,  # noqa: E402
                    CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT, CAT_SIDEBOARD_DONE)
from card_costs import _VOCAB_NAMES  # noqa: E402
from mcts import sb_dead_mask  # noqa: E402
from sb_rules import SB_CARD_FACTS, SB_DEAD_RULES, SB_FACT_COLS  # noqa: E402

NAME_TO_IDX = {n: i for i, n in enumerate(_VOCAB_NAMES)}
COL_BIT = {c: i for i, c in enumerate(SB_FACT_COLS)}

failures = 0


def check(ok, label):
    global failures
    if ok:
        print(f"ok  {label}")
    else:
        failures += 1
        print(f"FAIL {label}")


def cid(name):
    idx = NAME_TO_IDX.get(name)
    assert idx is not None, f"{name!r} not in vocab"
    return idx


def make_obs(*, sideboard, opp_cards, actions):
    """Synthetic obs: opp registered maindeck = ``opp_cards`` (list of names),
    menu = ``actions`` (list of (category, card_name_or_None))."""
    obs = np.zeros(env.OBS_SIZE, dtype=np.float32)
    obs[env._IS_SIDEBOARD_IDX] = 1.0 if sideboard else 0.0
    # Empty-slot sentinels for both opp deck blocks, then pack the maindeck.
    for j in range(env._OPP_DECK_MAIN_START, env._OPP_DECK_SIDE_END, 2):
        obs[j] = -1.0 / N_CARD_TYPES
    for k, name in enumerate(sorted(opp_cards, key=cid)):
        obs[env._OPP_DECK_MAIN_START + 2 * k] = cid(name) / N_CARD_TYPES
        obs[env._OPP_DECK_MAIN_START + 2 * k + 1] = 4 / 4.0
    for a, (cat, name) in enumerate(actions):
        obs[env.ACT_CATS_START + a] = cat / ACTION_CATEGORY_MAX
        obs[env.ACT_IDS_START + a] = ((cid(name) if name else -1)
                                      / N_CARD_TYPES)
    return obs


def main():
    # (1) table integrity.
    check(len(SB_CARD_FACTS) == N_CARD_TYPES,
          f"(1a) fact table covers the whole vocab ({N_CARD_TYPES})")
    check(SB_DEAD_RULES and all(
        0 <= c < N_CARD_TYPES and m > 0 for c, m in SB_DEAD_RULES.items()),
        f"(1b) {len(SB_DEAD_RULES)} rules, all on vocab cards with nonzero "
        f"condition masks")
    volc = SB_CARD_FACTS[cid("Volcanic Island")]
    check(bool(volc >> COL_BIT["land_island"] & 1)
          and bool(volc >> COL_BIT["land_mountain"] & 1),
          "(1c) Volcanic Island carries both Island and Mountain fact bits")

    menu = [(CAT_SIDEBOARD_DONE, None),
            (CAT_SIDEBOARD_IN, "Choke"),
            (CAT_SIDEBOARD_IN, "Red Elemental Blast"),
            (CAT_SIDEBOARD_IN, "Grafdigger's Cage"),   # no rule -> never dead
            (CAT_SIDEBOARD_OUT, "Choke")]
    mono_green = ["Forest", "Grizzly Bears", "Green Sun's Zenith"]
    blue_lists = ["Volcanic Island", "Brainstorm", "Lightning Bolt"]

    # (2) masking per opponent list.
    obs = make_obs(sideboard=True, opp_cards=mono_green, actions=menu)
    dead = sb_dead_mask(obs, len(menu))
    check(bool(dead[1]) and bool(dead[2]),
          "(2a) Choke + REB INs dead vs a mono-green 75")
    check(not dead[3],
          "(2b) an unruled card's IN is never dead")
    obs = make_obs(sideboard=True, opp_cards=blue_lists, actions=menu)
    dead = sb_dead_mask(obs, len(menu))
    check(not dead[1] and not dead[2],
          "(2c) Choke + REB live vs a list with Volcanic Island + Brainstorm")

    # (3) Done / OUT actions are never masked.
    obs = make_obs(sideboard=True, opp_cards=mono_green, actions=menu)
    dead = sb_dead_mask(obs, len(menu))
    check(not dead[0] and not dead[4],
          "(3) Done and OUT picks stay live whatever the opponent's list")

    # (4) not a sideboard prompt -> all-false.
    obs = make_obs(sideboard=False, opp_cards=mono_green, actions=menu)
    check(not sb_dead_mask(obs, len(menu)).any(),
          "(4) mask is all-false outside the sideboard phase")

    if failures:
        print(f"\nsb rules: FAILED ({failures} failure(s))")
        return 1
    print("\nsb rules: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
