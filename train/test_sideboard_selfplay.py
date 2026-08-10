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

It also covers az_selfplay's exhaustive SCHEDULE building — specifically the
rotating vs-scripted slice an ``--exhaustive-selfplay`` slot adds
(:func:`check_scripted_cell_rotation`): slot ``s`` takes the K consecutive
ordered (focus, opponent) pairs at offset ``(s * K) % n_ordered`` with
wraparound, ceil(n_ordered / K) slots cover every pair, K=0 adds nothing, and a
full ``--exhaustive`` matrix ignores the knob. That part is engine-free.

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

from env import (_MATCH_CTX_START, RoboMageEnv, scripted_action,  # noqa: E402
                 ACT_CATS_START, ACT_IDS_START)
from _enums import (ACTION_CATEGORY_MAX, N_CARD_TYPES,  # noqa: E402
                    CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT, CAT_SIDEBOARD_DONE)
from cli_spec import BINARY, BIN_DIR  # noqa: E402
from search_env import SearchRoboMageEnv  # noqa: E402
from mcts import UniformEvaluator  # noqa: E402
from az_selfplay import (_play_match, _backfill_and_pack,  # noqa: E402
                         build_exhaustive_schedule_ex, ordered_matchup_pairs,
                         rotating_scripted_pairs)

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


def check_scripted_cell_rotation() -> int:
    """The rotating vs-scripted slice of an --exhaustive-selfplay schedule.

    Engine-free (pure schedule building): slot s takes the K consecutive ordered
    (focus, opponent) pairs starting at (s * K) % n_ordered, wrapping, so
    successive az-league slots tile the whole ordered list. Returns a failure
    count."""
    failures = 0
    roster = [f"league/d{i}" for i in range(10)]
    pairs = ordered_matchup_pairs(roster, roster)
    n_ordered = len(pairs)          # 100 on a 10-deck roster
    k = 20
    if n_ordered != 100:
        failures += 1
        print(f"FAIL (r0): expected 100 ordered pairs on a 10-deck roster, "
              f"got {n_ordered}")

    # (r1) slot 0 takes pairs 0..K-1; slot 5 wraps back to offset 0 ((5*20)%100).
    if rotating_scripted_pairs(roster, roster, k, slot=0) != pairs[:k]:
        failures += 1
        print("FAIL (r1): slot 0 must take the first K ordered pairs")
    if rotating_scripted_pairs(roster, roster, k, slot=5) != pairs[:k]:
        failures += 1
        print("FAIL (r1): slot 5 with K=20 must wrap to offset 0 "
              "((5*20) % 100 == 0)")
    # A genuinely wrapping slice: K=30, slot 3 -> offset 90 -> 90..99 then 0..19.
    wrapped = rotating_scripted_pairs(roster, roster, 30, slot=3)
    if wrapped != pairs[90:] + pairs[:20]:
        failures += 1
        print(f"FAIL (r1): K=30 slot 3 must wrap 90..99 + 0..19, got "
              f"{wrapped[:3]}...{wrapped[-3:]}")
    if not failures:
        print("ok  (r1): rotation offset (slot * K) % n_ordered wraps correctly")

    # (r2) ceil(n_ordered / K) consecutive slots cover every ordered pair.
    slots = -(-n_ordered // k)
    seen = set()
    for s in range(slots):
        seen.update(rotating_scripted_pairs(roster, roster, k, slot=s))
    if seen != set(pairs):
        failures += 1
        print(f"FAIL (r2): {slots} slots covered {len(seen)}/{n_ordered} "
              f"ordered pairs")
    else:
        print(f"ok  (r2): {slots} slots cover all {n_ordered} ordered pairs")

    # (r3) the cells land in the schedule, marked exactly like --exhaustive's
    # scripted cells (a non-None net_is_a seat), on top of the 55 unordered
    # self-play cells x repeats.
    sched, seats = build_exhaustive_schedule_ex(
        roster, roster, seed=1, include_scripted=False, repeats=2,
        scripted_cells=k, slot=0)
    n_scr = sum(1 for s in seats if s is not None)
    n_self = len(seats) - n_scr
    if n_scr != k or n_self != 55 * 2 or len(sched) != len(seats):
        failures += 1
        print(f"FAIL (r3): expected {k} scripted + 110 self-play cells, got "
              f"{n_scr} + {n_self} (sched {len(sched)}, seats {len(seats)})")
    else:
        print(f"ok  (r3): schedule = 110 self-play + {k} scripted cells "
              f"(K is per-slot, not multiplied by repeats)")
    # The scripted cells are exactly this slot's pair slice (either seat order).
    want = {tuple(sorted(p)) for p in pairs[:k]}
    got = {tuple(sorted((a, b))) for (a, b, _), s in zip(sched, seats)
           if s is not None}
    if not got <= want:
        failures += 1
        print(f"FAIL (r3): scripted cells outside this slot's slice: {got - want}")

    # (r4) K=0 adds nothing.
    _s0, seats0 = build_exhaustive_schedule_ex(
        roster, roster, seed=1, include_scripted=False, repeats=1,
        scripted_cells=0, slot=0)
    if any(s is not None for s in seats0) or len(seats0) != 55:
        failures += 1
        print(f"FAIL (r4): scripted_cells=0 must leave a pure 55-cell self-play "
              f"matrix, got {len(seats0)} cells "
              f"({sum(1 for s in seats0 if s is not None)} scripted)")
    else:
        print("ok  (r4): scripted_cells=0 leaves the pure self-play matrix")

    # (r5) full --exhaustive ignores the knob (it already plays every ordered
    # pair): the schedule is identical with and without it.
    full_a = build_exhaustive_schedule_ex(roster, roster, seed=3,
                                          include_scripted=True, repeats=1,
                                          scripted_cells=0, slot=0)
    full_b = build_exhaustive_schedule_ex(roster, roster, seed=3,
                                          include_scripted=True, repeats=1,
                                          scripted_cells=k, slot=7)
    if full_a != full_b or len(full_a[0]) != 155:
        failures += 1
        print(f"FAIL (r5): full exhaustive must ignore scripted_cells "
              f"(155 cells; got {len(full_a[0])} vs {len(full_b[0])})")
    else:
        print("ok  (r5): full exhaustive ignores scripted_cells (155 cells)")
    return failures


def _menu(obs, num):
    """Decode (categories, card ids) for the current menu from the raw obs."""
    cats = np.round(obs[ACT_CATS_START:ACT_CATS_START + num]
                    * ACTION_CATEGORY_MAX).astype(int)
    ids = np.round(obs[ACT_IDS_START:ACT_IDS_START + num]
                   * N_CARD_TYPES).astype(int)
    return cats, ids


def _drive_to_sideboard(env, seed, max_decisions=3000):
    """Play scripted until the first sideboard prompt; return its obs."""
    obs, _ = env.reset(seed=seed)
    for _ in range(max_decisions):
        if obs[_IS_SIDEBOARD_IDX] > 0.5:
            return obs
        obs, _r, term, trunc, _i = env.step(scripted_action(obs, env._num_choices))
        if term or trunc:
            raise AssertionError("match ended before any sideboard prompt")
    raise AssertionError("no sideboard prompt within the decision cap")


def check_takeback_only_when_stranded() -> int:
    """The reverse of an outstanding half-move (the TAKEBACK) is offered ONLY
    when the balancing direction has no genuine completion.

    Case 1 (genuine completions exist): after opening a swap with a cut, the
    follow-up menu must offer sideboard-in choices but NOT the cut card itself —
    an out-X-then-in-X no-op is unexpressible.

    Case 2 (stranded, sideboard-less deck): a cut's follow-up menu is exactly the
    lone takeback, so the deck can still never be stuck off-size."""
    failures = 0

    def fail(msg):
        nonlocal failures
        failures += 1
        print(f"FAIL takeback: {msg}")

    # ── Case 1: real sideboard, so genuine completions exist ─────────────────
    deck_a, deck_b, paths = _write_decks()
    env = RoboMageEnv(deck_a=deck_a, deck_b=deck_b, bo3=True,
                      auto_sideboard=False)
    try:
        obs = _drive_to_sideboard(env, seed=SEED)
        num = env._num_choices
        cats, ids = _menu(obs, num)
        if CAT_SIDEBOARD_DONE not in cats:
            fail(f"balanced menu missing Done (cats={cats.tolist()})")
        outs = [i for i in range(num) if cats[i] == CAT_SIDEBOARD_OUT]
        if not outs:
            fail("balanced menu offers no cut")
        else:
            cut_id = ids[outs[0]]
            obs, _r, _t, _tr, _i = env.step(outs[0])
            num = env._num_choices
            cats, ids = _menu(obs, num)
            if obs[_IS_SIDEBOARD_IDX] <= 0.5 or num < 1:
                fail("no follow-up menu after opening a cut")
            ins = [i for i in range(num) if cats[i] == CAT_SIDEBOARD_IN]
            if not ins:
                fail("follow-up menu offers no genuine completion")
            if any(ids[i] == cut_id for i in ins):
                fail(f"avoidable takeback offered (cut card id {cut_id} back "
                     f"in the IN menu alongside {len(ins)} completions)")
            elif ins:
                # Complete the swap; the balanced menu must re-offer Done.
                obs, _r, _t, _tr, _i = env.step(ins[0])
                cats, _ = _menu(obs, env._num_choices)
                if CAT_SIDEBOARD_DONE not in cats:
                    fail("Done absent after completing the swap")
                else:
                    print(f"  ok  avoidable takeback absent ({len(ins)} genuine "
                          f"completions offered; swap completed, Done back)")
    finally:
        env.close()
        for p in paths:
            if os.path.exists(p):
                os.remove(p)

    # ── Case 2: sideboard-less deck — a cut strands, the takeback appears ────
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    a = os.path.join(d, "az_sb_nosb_a.dk")
    b = os.path.join(d, "az_sb_nosb_b.dk")
    with open(a, "w") as f:
        f.write("36 Grizzly Bears\n24 Forest\n")
    with open(b, "w") as f:
        f.write("60 Swamp\n")
    env = RoboMageEnv(deck_a="temp/az_sb_nosb_a", deck_b="temp/az_sb_nosb_b",
                      bo3=True, auto_sideboard=False)
    try:
        obs = _drive_to_sideboard(env, seed=SEED)
        num = env._num_choices
        cats, ids = _menu(obs, num)
        outs = [i for i in range(num) if cats[i] == CAT_SIDEBOARD_OUT]
        if not outs:
            fail("sideboard-less balanced menu offers no cut")
        else:
            cut_id = ids[outs[0]]
            obs, _r, _t, _tr, _i = env.step(outs[0])
            num = env._num_choices
            cats, ids = _menu(obs, num)
            if num != 1 or cats[0] != CAT_SIDEBOARD_IN or ids[0] != cut_id:
                fail(f"stranded menu should be the lone takeback of card "
                     f"{cut_id}, got cats={cats.tolist()} ids={ids.tolist()}")
            else:
                # Taking it must restore the balanced menu (Done available).
                obs, _r, _t, _tr, _i = env.step(0)
                cats, _ = _menu(obs, env._num_choices)
                if CAT_SIDEBOARD_DONE not in cats:
                    fail("Done absent after the forced takeback")
                else:
                    print("  ok  stranded cut offers exactly the lone takeback; "
                          "deck restored to balance")
    finally:
        env.close()
        for p in (a, b):
            if os.path.exists(p):
                os.remove(p)
    return failures


def main() -> int:
    sched_failures = check_scripted_cell_rotation()
    if sched_failures:
        print(f"\nexhaustive scripted-cell rotation: FAILED "
              f"({sched_failures} assertion failure(s))")
        return 1

    if not os.path.exists(BINARY):
        print(f"binary not found at {BINARY} — run `make` first", file=sys.stderr)
        return 2

    tb_failures = check_takeback_only_when_stranded()
    if tb_failures:
        print(f"\ntakeback-only-when-stranded: FAILED ({tb_failures} failure(s))")
        return 1

    deck_a, deck_b, paths = _write_decks()
    env = SearchRoboMageEnv(deck_a=deck_a, deck_b=deck_b, bo3=True,
                            auto_sideboard=False)
    rng = np.random.default_rng(SEED)
    failures = 0
    try:
        samples, game_winners, searched, fallback, dropped, sb_stats = _play_match(
            env, UniformEvaluator(), rng, sims=SIMS, worlds=WORLDS,
            temp_moves=TEMP_MOVES, root_noise_eps=0.0, root_noise_alpha=1.0,
            seed=SEED, sb_sims=SB_SIMS, sb_worlds=SB_WORLDS,
            sb_max_depth=SB_MAX_DEPTH, sb_rollout_turns=SB_ROLLOUT_TURNS,
            sb_persist=True)
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
    packed = _backfill_and_pack(samples, game_winners)
    z = packed["z"]

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

    # (v) n-step TD targets: a sideboard-root sample never bootstraps (its td_q is
    # its own z), and every td_q is finite and inside [-1, 1].
    td_q = packed["td_q"]
    if not np.all(np.isfinite(td_q)) or np.any(np.abs(td_q) > 1.0 + 1e-6):
        failures += 1
        print(f"FAIL (v): td_q must be finite and within [-1,1]; got "
              f"min={td_q.min()} max={td_q.max()}")
    for i in sb_idx:
        if td_q[i] != z[i]:
            failures += 1
            print(f"FAIL (v): sideboard sample pos {i}: td_q={td_q[i]} != "
                  f"z={z[i]} — a sideboard root must not bootstrap")
            break
    else:
        print(f"ok  (v): every sideboard sample's td_q == z; td_q finite in "
              f"[-1,1] ({int((td_q != z).sum())}/{len(z)} rows bootstrapped)")

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
