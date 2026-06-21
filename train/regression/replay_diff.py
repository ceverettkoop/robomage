#!/usr/bin/env python3
"""Determinism safety net for the effect-dispatch refactor.

Plays a fixed set of scripted games against the engine and snapshots the full
game narrative (every `game_log` line the engine emits, plus the integer
decision the scripted agent makes at each point) into a transcript per
scenario. The scripted agent is a pure function of game state and the engine is
deterministic given a seed + `--no-shuffle`, so an unchanged engine reproduces
each transcript byte-for-byte.

  record  — play each scenario, write train/regression/corpus/<name>.txt
  check   — replay each scenario, diff against the recorded transcript

Run `record` ONCE on a known-good build (main, before the refactor). Run
`check` after every refactor batch. A passing `check` proves the refactor did
not change a single narrative line or scripted decision across these games —
which is exactly what the effect-dispatch rewrite must preserve.

We drive the game in the engine's real machine mode (not `--replay`) on
purpose: replaying a machine-mode decision log in CLI mode diverges and is not a
faithful reproduction of the game.

Usage (from repo root):
  train/.venv/bin/python train/regression/replay_diff.py record
  train/.venv/bin/python train/regression/replay_diff.py check
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TRAIN = _HERE.parent
_REPO = _TRAIN.parent
_BINARY = _REPO / "bin" / "robomage"
_CORPUS = _HERE / "corpus"

# test_harness owns the BQUERY binary protocol + scripted agent.
sys.path.insert(0, str(_TRAIN))

# Single seed governing the engine RNG for every scenario. Drawn once at random
# (secrets.randbelow) and then frozen here so the corpus is reproducible but not
# cherry-picked. To re-roll, pick a new random value, update this, and re-record.
SEED = 1641108517

# (name, deck_a, deck_b) — deck names are relative to resources/decks/. Each
# scenario is a natural deck-vs-deck scripted game: NO hand/battlefield sculpting,
# no forced outcomes. Pairings cover the three decks in both seatings.
SCENARIOS = [
    ("delver_mirror",      "delver",   "delver"),
    ("doomsday_mirror",    "doomsday", "doomsday"),
    ("mav_mirror",         "mav",      "mav"),
    ("delver_vs_doomsday", "delver",   "doomsday"),
    ("doomsday_vs_delver", "doomsday", "delver"),
    ("delver_vs_mav",      "delver",   "mav"),
    ("mav_vs_delver",      "mav",      "delver"),
    ("doomsday_vs_mav",    "doomsday", "mav"),
    ("mav_vs_doomsday",    "mav",      "doomsday"),
]


def _play(deck_a: str, deck_b: str, seed: int, actions=None):
    """Drive one game in machine mode. If `actions` is None, the (nondeterministic)
    scripted agent chooses and we record what it picked; otherwise the given
    action ints are fed back verbatim. Returns (transcript, chosen_actions).

    The transcript is all engine narrative plus a marker per decision, so it is
    a deterministic function of (seed, decks, actions) alone."""
    from test_harness import (TestHarness, get_scripted_action,  # noqa: E402
                              _MANDATORY_CATS)

    th = TestHarness(binary=str(_BINARY))
    # Shuffle each library with the seeded RNG (no --no-shuffle): natural games,
    # deterministic given the frozen SEED.
    th.start(deck_a, deck_b, seed, no_shuffle=False)
    lines: list[str] = []
    chosen: list[int] = []
    decision = 0
    try:
        while True:
            narrative, query = th.read_until_query()
            lines.extend(narrative)
            if query is None:
                lines.append(f"=== GAME OVER: winner={th.winner} ===")
                break
            if actions is not None and decision >= len(actions):
                # Refactor needs more decisions than recorded — drift; stop here
                # (the narrative diff has already diverged above).
                lines.append("<<RAN OUT OF RECORDED ACTIONS>>")
                break
            state, cats_int, card_ids, ctrl, num_choices = query
            if actions is None:
                choice = get_scripted_action(
                    state, cats_int, card_ids, ctrl, num_choices)
                has_confirm = any(int(cats_int[i]) in _MANDATORY_CATS
                                  for i in range(num_choices))
                game_action = -1 if (has_confirm and choice == num_choices - 1) else choice
            else:
                game_action = actions[decision]
            decision += 1
            chosen.append(int(game_action))
            lines.append(f"<<DECISION {decision}: action={game_action}>>")
            th._send(game_action)
    finally:
        th.kill()
    return "\n".join(lines) + "\n", chosen


def cmd_record() -> int:
    _CORPUS.mkdir(parents=True, exist_ok=True)
    for name, deck_a, deck_b in SCENARIOS:
        transcript, actions = _play(deck_a, deck_b, SEED, actions=None)
        (_CORPUS / f"{name}.txt").write_text(transcript)
        (_CORPUS / f"{name}.actions").write_text(
            ",".join(str(a) for a in actions))
        print(f"  recorded {name}: {len(actions)} decisions, "
              f"{len(transcript)} bytes")
    print(f"Recorded {len(SCENARIOS)} scenarios (seed {SEED}) to {_CORPUS}")
    return 0


def cmd_check() -> int:
    failures = []
    for name, deck_a, deck_b in SCENARIOS:
        expected_path = _CORPUS / f"{name}.txt"
        actions_path = _CORPUS / f"{name}.actions"
        if not expected_path.exists() or not actions_path.exists():
            print(f"  MISSING corpus for {name} — run `record` first")
            failures.append(name)
            continue
        expected = expected_path.read_text()
        raw = actions_path.read_text().strip()
        actions = [int(x) for x in raw.split(",")] if raw else []
        actual, _ = _play(deck_a, deck_b, SEED, actions=actions)
        if actual == expected:
            print(f"  OK   {name}")
        else:
            print(f"  DRIFT {name}")
            _print_first_diff(expected, actual)
            failures.append(name)

    if failures:
        print(f"\nFAILED: {len(failures)}/{len(SCENARIOS)} scenarios drifted: "
              f"{', '.join(failures)}")
        return 1
    print(f"\nPASS: all {len(SCENARIOS)} scenarios byte-identical")
    return 0


def _print_first_diff(expected: str, actual: str) -> None:
    exp_lines = expected.splitlines()
    act_lines = actual.splitlines()
    for i in range(max(len(exp_lines), len(act_lines))):
        e = exp_lines[i] if i < len(exp_lines) else "<EOF>"
        a = act_lines[i] if i < len(act_lines) else "<EOF>"
        if e != a:
            print(f"     first diff at line {i + 1}:")
            print(f"       expected: {e!r}")
            print(f"       actual:   {a!r}")
            return


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("record", "check"):
        print(__doc__)
        return 2
    if not _BINARY.exists():
        print(f"binary not found at {_BINARY} — run `make` first")
        return 2
    return cmd_record() if sys.argv[1] == "record" else cmd_check()


if __name__ == "__main__":
    raise SystemExit(main())
