#!/usr/bin/env python3
"""Self-play data-path regression for SEARCHED bo3 sideboard decisions (Stage 4).

Stages 1-3 made a bo3 sideboard prompt a valid, loop-safe MCTS search root. This
test proves the Python self-play collector (:func:`az_selfplay._play_match` +
:func:`az_selfplay._backfill_and_pack`) turns those roots into training samples
that are priced by the UPCOMING game's winner — the whole point of learned
sideboarding.

It runs ONE small bo3 match through the real ``_play_match`` (uniform torch-free
evaluator, tiny sim/world budgets, a lopsided aggro-vs-lands matchup so the match
finishes fast and cleanly) and asserts:

  (i)   at least one collected sample is a sideboard decision (its observation has
        the ``is_sideboard_phase`` flag set);
  (ii)  every sideboard sample's ``game_idx`` equals the number of games completed
        when it was appended — i.e. it gates the UPCOMING game, not the one that
        just ended (verified independently from the ordering of in-game samples);
  (iii) after ``_backfill_and_pack`` each sideboard sample's z is exactly +/-1 per
        that upcoming game's winner vs the sample's ``mover_is_a`` (recomputed
        here straight from the recorded ``game_winners``);
  (iv)  sanity: in-game samples still price by their OWN game's winner.

Torch-free: ``_play_match``/``_backfill_and_pack`` (az_selfplay), ``run_search``
(mcts) and ``SearchRoboMageEnv`` (search_env) all import torch only lazily inside
worker/evaluator code paths this test never enters; the evaluator here is
:class:`mcts.UniformEvaluator`. Every layout constant is imported (no magic
offsets). Needs bin/robomage built (like train/test_snapshot.py).

Wired into ``train/ci_check.py`` as the ``sbselfplay`` tier (so ``make check``
runs it); also runnable standalone::

    train/.venv/bin/python train/test_sideboard_selfplay.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import _MATCH_CTX_START  # noqa: E402
from cli_spec import BINARY, BIN_DIR  # noqa: E402
from search_env import SearchRoboMageEnv  # noqa: E402
from mcts import UniformEvaluator  # noqa: E402
from az_selfplay import _play_match, _backfill_and_pack  # noqa: E402

_IS_SIDEBOARD_IDX = _MATCH_CTX_START + 3

# Small budgets so the match finishes fast; sideboard roots get their own budget.
SIMS = 8
WORLDS = 2
SB_SIMS = 8
SB_WORLDS = 2
SB_MAX_DEPTH = 200
# Pinned small (not the shipped DEFAULT_SB_ROLLOUT_TURNS) to bound the tier's
# runtime while still exercising the leaf-rollout path at every searched
# sideboard root (uniform-evaluator rollouts argmax to action 0 and are bounded
# by the 40/turn step cap).
SB_ROLLOUT_TURNS = 2
TEMP_MOVES = 20
SEED = 7


def _write_decks():
    """Two stacked bo3 decks with sideboards. A is aggressive (Grizzly Bears +
    Forest) so it reliably beats B (60 Swamp does nothing) within a few turns,
    making game 1 end and a real sideboard prompt appear. Each has a 15-card
    sideboard so the sideboard IN menu has >1 choice (searchable)."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    a = os.path.join(d, "az_sb_a.dk")
    b = os.path.join(d, "az_sb_b.dk")
    with open(a, "w") as f:
        f.write("36 Grizzly Bears\n24 Forest\nSIDEBOARD:\n15 Mountain\n")
    with open(b, "w") as f:
        f.write("60 Swamp\nSIDEBOARD:\n15 Mountain\n")
    return "temp/az_sb_a", "temp/az_sb_b", [a, b]


def _is_sideboard(sample) -> bool:
    return bool(sample["obs"][_IS_SIDEBOARD_IDX] > 0.5)


def _expected_z(winner, mover_is_a) -> float:
    if winner is None:
        return 0.0
    return 1.0 if ((winner == "A") == mover_is_a) else -1.0


def main() -> int:
    if not os.path.exists(BINARY):
        print(f"binary not found at {BINARY} — run `make` first", file=sys.stderr)
        return 2

    deck_a, deck_b, paths = _write_decks()
    env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b, bo3=True,
                            auto_sideboard=False)
    rng = np.random.default_rng(SEED)
    failures = 0
    try:
        samples, game_winners, searched, fallback, dropped = _play_match(
            env, UniformEvaluator(), rng, sims=SIMS, worlds=WORLDS,
            temp_moves=TEMP_MOVES, root_noise_eps=0.0, root_noise_alpha=1.0,
            seed=SEED, sb_sims=SB_SIMS, sb_worlds=SB_WORLDS,
            sb_max_depth=SB_MAX_DEPTH, sb_rollout_turns=SB_ROLLOUT_TURNS)
    finally:
        env.close()

    print(f"match: games={len(game_winners)} winners={game_winners} "
          f"samples={len(samples)} searched={searched} fallback={fallback} "
          f"dropped={dropped}")

    # The match must have finished cleanly (>=2 games in a bo3; the sideboard
    # sample's upcoming game must have completed, else it would have been dropped).
    if len(game_winners) < 2:
        print(f"FAIL setup: expected the bo3 to reach >=2 games, got "
              f"{len(game_winners)} ({game_winners}); no sideboard prompt occurs "
              f"before game 2")
        return 1

    sb_idx = [i for i, s in enumerate(samples) if _is_sideboard(s)]

    # (i) at least one sideboard decision was collected.
    if not sb_idx:
        print("FAIL (i): no sideboard sample collected (is_sideboard_phase flag "
              "never set on a searched decision)")
        return 1
    print(f"ok  (i): {len(sb_idx)} sideboard sample(s) collected "
          f"(at positions {sb_idx})")

    # (ii) each sideboard sample gates the UPCOMING game: its game_idx equals the
    # number of games completed when it was appended. Verify independently from the
    # ordering of in-game samples — walking in append order, the distinct in-game
    # game indices seen before a sideboard sample are exactly {0..k-1}, and the
    # sideboard sample belongs to game k (not yet started -> not among them).
    seen_ingame = set()
    for i, s in enumerate(samples):
        if _is_sideboard(s):
            gi = s["game_idx"]
            completed = len(seen_ingame)
            ok = (gi == completed
                  and gi not in seen_ingame
                  and (not seen_ingame or gi == max(seen_ingame) + 1)
                  and seen_ingame == set(range(completed)))
            if not ok:
                failures += 1
                print(f"FAIL (ii): sideboard sample at pos {i} game_idx={gi} but "
                      f"{completed} game(s) completed before it "
                      f"(in-game games seen so far: {sorted(seen_ingame)})")
        else:
            seen_ingame.add(s["game_idx"])
    if not failures:
        print(f"ok  (ii): every sideboard sample's game_idx == completed-games "
              f"count (gates the upcoming game)")

    # Backfill z from each sample's game vs its mover.
    obs, pi, z, mask = _backfill_and_pack(samples, game_winners)

    # (iii) each sideboard sample's z == +/-1 per the UPCOMING game's winner.
    for i in sb_idx:
        s = samples[i]
        gi = s["game_idx"]
        winner = game_winners[gi]
        if winner is None:
            failures += 1
            print(f"FAIL (iii): sideboard sample pos {i} gates game {gi} which "
                  f"drew (winner None) — a searched sideboard sample must gate a "
                  f"decided game")
            continue
        exp = _expected_z(winner, s["mover_is_a"])
        if z[i] != exp or abs(z[i]) != 1.0:
            failures += 1
            print(f"FAIL (iii): sideboard sample pos {i} game_idx={gi} "
                  f"winner={winner} mover_is_a={s['mover_is_a']}: z={z[i]} "
                  f"expected {exp}")
    if not failures:
        print(f"ok  (iii): every sideboard sample's z == +/-1 per its upcoming "
              f"game's winner")

    # (iv) sanity: in-game samples price by their OWN game's winner.
    for i, s in enumerate(samples):
        if _is_sideboard(s):
            continue
        exp = _expected_z(game_winners[s["game_idx"]], s["mover_is_a"])
        if z[i] != exp:
            failures += 1
            print(f"FAIL (iv): in-game sample pos {i} game_idx={s['game_idx']}: "
                  f"z={z[i]} expected {exp}")
            break
    else:
        print("ok  (iv): in-game samples price by their own game's winner")

    # Housekeeping: remove the temp decks we created.
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass

    if failures:
        print(f"\nsideboard self-play: FAILED ({failures} assertion failure(s))")
        return 1
    print("\nsideboard self-play: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
