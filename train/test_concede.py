#!/usr/bin/env python3
"""Concession regression: the engine's CR 104.3a concede sentinels end-to-end.

A player may concede at any time and loses the game immediately (CR 104.3a).
The engine takes a concession as an out-of-band decision INPUT — the sentinels
``env.CONCEDE_GAME`` (-2) and ``env.CONCEDE_MATCH`` (-3), accepted wherever an
action index is read (-1 is already the confirm slot) and attributed to the
seat the pending decision belongs to. This test asserts the four behaviours
everything else is built on:

  1. bo1 — CONCEDE_GAME at any decision ends the game with the CONCEDING seat
     as the loser (the reward, Player-A perspective, carries that sign).
  2. bo3 — CONCEDE_GAME is an ordinary game loss: GAME_RESULT lands, the match
     does NOT end, and play continues into the between-games sideboard phase
     and on into game 2.
  3. bo3 — CONCEDE_MATCH mid-game-1 loses that game AND ends the match, with
     MATCH_RESULT naming the conceder's OPPONENT as the match winner (whatever
     the game score is).
  4. determinism — the same seed plus the same action sequence INCLUDING the
     sentinel reproduces the game byte-for-byte, and the sentinel is recorded
     in the RMLOG decision log, so ``--replay`` re-plays the concession and the
     replayed game ends with the same winner.

Mirror replay of a concession (a SearchRoboMageEnv spawning a detached mirror
from a history that contains one) is deliberately NOT covered: a concession
ends the game it is played in, so replaying a prefix that contains one only
ever reconstructs a decision of a LATER game of a bo3 — replaying to a terminal
is a non-goal (see search_env.spawn_detached_mirror).

Torch-free: drives ``env.RoboMageEnv`` (which spawns bin/robomage in --machine
mode) plus one raw ``--replay`` subprocess. Wired into ``train/ci_check.py`` as
the ``concede`` tier (so ``make check`` runs it); also runnable standalone::

    train/.venv/bin/python train/test_concede.py
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import decode
from env import (CONCEDE_GAME, CONCEDE_MATCH, GAME_LOSS_REWARD, GAME_WIN_REWARD,
                 NarrativeEnv, STATE_SIZE, _SELF_IS_A_IDX)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN_DIR = os.path.join(_REPO_ROOT, "bin")
_BINARY = os.path.join(_BIN_DIR, "robomage")

DECK_A = "delver"
DECK_B = "mav"

_failures = []


def check(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        _failures.append(msg)


def make_env(bo3=False, log_decisions=False):
    return NarrativeEnv(binary_path=_BINARY, deck_a=DECK_A, deck_b=DECK_B,
                        bo3=bo3, log_decisions=log_decisions)


def priority_is_a(obs):
    """True when the seat facing this decision (the one a concession here is
    attributed to) is Player A."""
    return bool(obs[_SELF_IS_A_IDX] > 0.5)


def loss_reward_for(conceder_is_a):
    """The env reward (Player-A perspective) for that seat losing the game."""
    return GAME_LOSS_REWARD if conceder_is_a else GAME_WIN_REWARD


def advance(env, obs, n, actions=None):
    """Step `n` decisions, taking choice 0 (pass / first option) each time.
    Returns (obs, done, played) — `played` is the action list actually taken,
    so it can be replayed verbatim. Stops early if the game ends."""
    played = list(actions or [])
    done = False
    for _ in range(n):
        obs, _r, term, trunc, _info = env.step(0)
        played.append(0)
        done = term or trunc
        if done:
            break
    return obs, done, played


# ── 1. bo1: CONCEDE_GAME loses the game for the conceding seat ───────────────

def test_bo1_concede_game():
    print("[1] bo1 CONCEDE_GAME ends the game, conceder loses")
    env = make_env()
    try:
        obs, _ = env.reset(options={"engine_seed": 4242})
        obs, done, _played = advance(env, obs, 12)
        check(not done, "reached a mid-game decision without the game ending")
        conceder_is_a = priority_is_a(obs)
        obs, reward, terminated, truncated, info = env.step(CONCEDE_GAME)
        check(terminated, "CONCEDE_GAME terminated the episode")
        check(not truncated, "termination was a real game end, not a truncation")
        check(reward == loss_reward_for(conceder_is_a),
              f"reward {reward:+.1f} == loss for the conceding seat "
              f"(Player {'A' if conceder_is_a else 'B'})")
        check(info.get("done", False), "info reported the game as done")
        text = "\n".join(env.flush_lines())
        who = "Player A" if conceder_is_a else "Player B"
        check(f"{who} concedes" in text,
              f"narrative announced the concession ('{who} concedes')")
    finally:
        env.close()


# ── 2. bo3: CONCEDE_GAME is an ordinary game loss; the match continues ───────

def test_bo3_concede_game_continues_match():
    print("[2] bo3 CONCEDE_GAME: game 1 lost, match continues (sideboard, game 2)")
    env = make_env(bo3=True)
    try:
        obs, _ = env.reset(options={"engine_seed": 909})
        obs, done, _played = advance(env, obs, 12)
        check(not done, "reached a mid-game-1 decision")
        conceder_is_a = priority_is_a(obs)
        obs, reward, terminated, truncated, info = env.step(CONCEDE_GAME)
        check(not (terminated or truncated),
              "the MATCH did not end — a conceded game is just a game loss")
        check(bool(info.get("game_result")),
              "the step carried a GAME_RESULT boundary")
        check(reward == loss_reward_for(conceder_is_a),
              f"game reward {reward:+.1f} == loss for the conceding seat")
        # The next decisions are the between-games sideboard phase, then game 2.
        mctx = decode._decode_match_context(obs[:STATE_SIZE])
        check(bool(mctx["is_sideboard"]),
              "the next decision is the sideboard phase")
        check(mctx["self_wins"] + mctx["opp_wins"] == 1,
              "the match score records exactly one finished game")
        # Play the sideboard "Done" (choice 0) for both seats and land in game 2.
        for _ in range(40):
            obs, _r, term, trunc, _i = env.step(0)
            if term or trunc:
                break
            mctx = decode._decode_match_context(obs[:STATE_SIZE])
            if not mctx["is_sideboard"]:
                break
        check(not mctx["is_sideboard"], "sideboarding completed")
        check(bool(mctx["is_post_board"]) and mctx["game_number"] >= 1,
              "game 2 of the match started (post-board)")
    finally:
        env.close()


# ── 3. bo3: CONCEDE_MATCH ends the match for the conceder ────────────────────

def test_bo3_concede_match():
    print("[3] bo3 CONCEDE_MATCH: match ends, opponent wins")
    env = make_env(bo3=True)
    try:
        obs, _ = env.reset(options={"engine_seed": 909})
        obs, done, _played = advance(env, obs, 12)
        check(not done, "reached a mid-game-1 decision")
        conceder_is_a = priority_is_a(obs)
        obs, reward, terminated, truncated, info = env.step(CONCEDE_MATCH)
        check(terminated, "CONCEDE_MATCH terminated the episode in game 1")
        check(not truncated, "termination was a real match end")
        check(reward == loss_reward_for(conceder_is_a),
              f"the conceded game still scored {reward:+.1f} for the conceder's loss")
        text = "\n".join(env.flush_lines())
        winner = "B" if conceder_is_a else "A"
        check(f"MATCH_RESULT: Player {winner} wins" in text,
              f"MATCH_RESULT named the opponent (Player {winner}) the match winner")
        check("concedes the match" in text, "narrative announced the match concession")
        check("MATCH GAME 2" not in text, "no second game was played")
    finally:
        env.close()


# ── 3b. bo3: the sentinels between games (guard rail) ───────────────────────

def test_bo3_concede_between_games():
    print("[3b] bo3 sentinels during the between-games sideboard phase")
    env = make_env(bo3=True)
    try:
        obs, _ = env.reset(options={"engine_seed": 909})
        obs, done, _played = advance(env, obs, 12)
        obs, _r, _t, _tr, _i = env.step(CONCEDE_GAME)          # lose game 1
        mctx = decode._decode_match_context(obs[:STATE_SIZE])
        check(bool(mctx["is_sideboard"]), "parked at a sideboard decision")
        # -2 here has no live game to concede: it must be ignored safely (taken
        # as the first menu choice), never crash and never end the match.
        obs, _r, terminated, truncated, _i = env.step(CONCEDE_GAME)
        check(not (terminated or truncated),
              "CONCEDE_GAME with no live game was ignored safely")
        text = "\n".join(env.flush_lines())
        check("no live game to concede" in text,
              "the engine said so rather than inventing a loser")
        # -3 still ends the match from between games, crediting the opponent.
        sideboarder_is_a = priority_is_a(obs)
        obs, _r, terminated, _tr, _i = env.step(CONCEDE_MATCH)
        check(terminated, "CONCEDE_MATCH between games ended the match")
        text = "\n".join(env.flush_lines())
        winner = "B" if sideboarder_is_a else "A"
        check(f"MATCH_RESULT: Player {winner} wins" in text,
              f"MATCH_RESULT credited the opponent (Player {winner})")
    finally:
        env.close()


# ── 4. determinism: same seed + same actions (sentinel included) ─────────────

def _play_conceded_game(seed, log_decisions=False):
    """Play a fixed line ending in CONCEDE_GAME; return (actions, reward, obs)."""
    env = make_env(log_decisions=log_decisions)
    try:
        obs, _ = env.reset(options={"engine_seed": seed})
        obs, done, played = advance(env, obs, 12)
        assert not done, "fixture line ended the game before the concession"
        obs, reward, _term, _trunc, _info = env.step(CONCEDE_GAME)
        played.append(CONCEDE_GAME)
        return played, reward, obs.copy()
    finally:
        env.close()


def test_determinism_and_replay():
    print("[4] determinism: the sentinel replays identically (env + --replay)")
    seed = 31337
    actions_a, reward_a, obs_a = _play_conceded_game(seed)
    actions_b, reward_b, obs_b = _play_conceded_game(seed, log_decisions=True)
    check(actions_a == actions_b, "the same seed produced the same action line")
    check(reward_a == reward_b, f"the same reward ({reward_a:+.1f}) both runs")
    check(np.array_equal(obs_a, obs_b),
          "the final observation is byte-identical across runs")

    # The --log-decisions run wrote an RMLOG; the sentinel must be IN it (that
    # is what makes a concession replayable) and --replay must reproduce it.
    log_path = os.path.join(_BIN_DIR, "resources", "logs", f"game_{seed}.log")
    check(os.path.exists(log_path), f"a decision log was written ({log_path})")
    if not os.path.exists(log_path):
        return
    with open(log_path, encoding="utf-8") as f:
        logged = f.read()
    decisions = logged.split("DECISIONS\n", 1)[-1].split()
    check(decisions and decisions[-1] == str(CONCEDE_GAME),
          f"the log's last decision is the sentinel ({CONCEDE_GAME})")
    r = subprocess.run([_BINARY, "--replay", log_path],
                       cwd=_BIN_DIR, capture_output=True, text=True, timeout=120)
    check(r.returncode == 0, f"--replay exited cleanly (rc={r.returncode})")
    conceder_is_a = reward_a == GAME_LOSS_REWARD
    who = "Player A" if conceder_is_a else "Player B"
    other = "Player B" if conceder_is_a else "Player A"
    check(f"{who} concedes - {other} wins!" in r.stdout,
          f"the replayed game ended on the same concession ({who} concedes)")
    check("FATAL" not in r.stdout and "FATAL" not in r.stderr,
          "the replay produced no fatal error")


# ── 5. GameDriver: the concede API and hard-timeout enforcement ──────────────

class _StubSink:
    def __init__(self):
        self.logs = []
        self.timeouts = []

    def on_state(self, u):
        pass

    def on_log(self, lines):
        self.logs.extend(lines)

    def on_opp_thinking(self, active):
        pass

    def on_game_over(self, text):
        pass

    def on_timeout(self, seat, reason):
        self.timeouts.append((seat, reason))


class _StubClock:
    """The shape GameDriver reads off a controller: opponents.MatchClock."""

    def __init__(self, remaining):
        self.remaining = float(remaining)


class _StubController:
    def __init__(self, remaining):
        self._clock = _StubClock(remaining)


def _driver(sink, **kw):
    from game_driver import GameDriver
    return GameDriver(env=None, opp_act=None, opp_is_a=True, is_model=True,
                      opp_label="Model", bo3=False, sink=sink, **kw)


def test_driver_concede_and_timeout():
    print("[5] GameDriver: concede API + hard timeout (engine-free)")
    sink = _StubSink()

    # Defaults: no clocks, nothing ever times out.
    d = _driver(sink)
    check(d.human_clock_remaining() is None, "default driver leaves the human untimed")
    check(d._hard_timeout_seat(opp_turn=False, human_must_act=True) is None
          and d._hard_timeout_seat(opp_turn=True, human_must_act=False) is None,
          "default driver never times a seat out")
    d._debit_human(5.0)
    check(d.human_clock_remaining() is None, "debiting an untimed human is a no-op")

    # concede() queues the sentinel on the human-action queue.
    d.concede()
    check(d._human_q.get() == CONCEDE_GAME, "concede() queues CONCEDE_GAME")
    check(d._forfeit is None, "a GAME concede leaves the match banner to the reward")
    d.concede(match=True)
    check(d._human_q.get() == CONCEDE_MATCH, "concede(match=True) queues CONCEDE_MATCH")
    check(d._forfeit is None,
          "concede(match=True) records no forfeit until the loop consumes the sentinel")

    # Hard timeout on the human seat: the bank is debited per decision and an
    # empty bank at the human's own decision flags a loss on time.
    d = _driver(sink, human_clock_s=2.0, hard_timeout=True)
    check(d.human_clock_remaining() == 2.0, "the human bank starts at human_clock_s")
    d._debit_human(0.5)
    check(abs(d.human_clock_remaining() - 1.5) < 1e-9, "a decision debits the bank")
    check(d._hard_timeout_seat(opp_turn=False, human_must_act=True) is None,
          "a bank with time left does not time out")
    d._debit_human(99.0)
    check(d.human_clock_remaining() == 0.0, "the bank floors at zero")
    check(d._hard_timeout_seat(opp_turn=False, human_must_act=False) is None,
          "an autopass window (nobody on the clock) never times out")
    check(d._hard_timeout_seat(opp_turn=False, human_must_act=True) == "you",
          "an empty bank at the human's own decision times them out")
    d._report_timeout("you")
    check(sink.timeouts and sink.timeouts[-1][0] == "you",
          "the sink's optional on_timeout hook was called")
    check("You lost on time." in sink.timeouts[-1][1], "the reason names the loser")
    check(d._hard_timeout_seat(opp_turn=False, human_must_act=True) is None,
          "a seat times out only once")
    check("Model wins — lost on time." in d._winner_text(),
          f"the banner reports the timeout loss ({d._winner_text()})")

    # Opponent seat: the bank is read duck-typed off the controller's MatchClock.
    d = _driver(sink, controller=_StubController(3.0), hard_timeout=True)
    check(d._opp_clock_remaining() == 3.0, "the opponent bank is read off controller._clock")
    check(d._hard_timeout_seat(opp_turn=True, human_must_act=False) is None,
          "an opponent with time left does not time out")
    d._controller._clock.remaining = 0.0
    check(d._hard_timeout_seat(opp_turn=True, human_must_act=False) == "opponent",
          "an opponent whose bank is empty times out at its decision")
    d._report_timeout("opponent")
    check("You win! — lost on time" in d._winner_text(),
          f"the banner credits the human the timeout win ({d._winner_text()})")

    # Soft mode (hard_timeout off) is unchanged even with banks present.
    d = _driver(sink, controller=_StubController(0.0), human_clock_s=0.0)
    check(d._hard_timeout_seat(opp_turn=True, human_must_act=False) is None
          and d._hard_timeout_seat(opp_turn=False, human_must_act=True) is None,
          "an exhausted bank never concedes while hard_timeout is off")


def test_driver_drain_pending_input():
    print("[6] GameDriver: a stale queued concession is dropped at a game boundary")
    sink = _StubSink()

    # A concession queued while the human is off the clock but never consumed
    # (its game ended by other means first) must not survive into the next game,
    # where it would be read at the first decision and auto-concede a game the
    # human never chose to.
    d = _driver(sink)
    d.concede()                          # queues CONCEDE_GAME out of turn
    check(not d._human_q.empty(), "the concession is queued")
    d._drain_pending_input()
    check(d._human_q.empty(),
          "the stale concession is drained at the game boundary")

    # A pending quit (None) survives the drain so shutdown still fires.
    d = _driver(sink)
    d.request_quit()                     # queues None (quit)
    d._drain_pending_input()
    check(d._human_q.get_nowait() is None, "a pending quit survives the drain")


def main():
    for t in (test_bo1_concede_game,
              test_bo3_concede_game_continues_match,
              test_bo3_concede_match,
              test_bo3_concede_between_games,
              test_determinism_and_replay,
              test_driver_concede_and_timeout,
              test_driver_drain_pending_input):
        t()
    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s)")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("All concede checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
