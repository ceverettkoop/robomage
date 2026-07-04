#!/usr/bin/env python3
"""Byte-identical determinism safety net for the curated decks.

Plays a fixed set of scripted deck-vs-deck games (the three fully-implemented
curated decks — delver/doomsday/mav — in every seating) and snapshots the full
scripted transcript (every `game_log` line the engine emits plus the runner's
per-decision markers) into one file per scenario. The scripted agent is a pure
function of game state and the engine is deterministic given a seed, so an
unchanged engine + agent + card data reproduces each transcript byte-for-byte.

  record  — play each scenario, write train/regression/corpus/<name>.txt
  check   — replay each scenario, diff against the recorded transcript (exit 1
            on any drift). Wired as the `replay` tier of train/ci_check.py.

Run `record` only when a behavior change is INTENTIONAL (an engine/effect change,
a scripted-agent change, or a card-data change that legitimately alters these
games); commit the regenerated corpus as part of that change's review. A passing
`check` otherwise proves the change did not alter a single narrative line or
scripted decision across these games.

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


def _play(deck_a: str, deck_b: str, seed: int) -> str:
    """Drive one scripted deck-vs-deck game and return its full transcript.

    Uses the shared deterministic game loop (runner.run_games) with the scripted
    (hard) agent on both seats. The scripted agent is a pure function of game
    state and the engine is deterministic given a seed, so an unchanged engine +
    agent + card data reproduces the transcript byte-for-byte. Libraries shuffle
    with the seeded RNG (natural games, deterministic given the frozen SEED)."""
    import contextlib
    import io

    import runner  # noqa: E402
    from opponents import make_controller  # noqa: E402

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runner.run_games(
            make_controller("scripted"), make_controller("scripted"),
            label_a="A", label_b="B", binary_path=str(_BINARY),
            deck_a=deck_a, deck_b=deck_b, n_games=1, seed=seed, verbose=True)
    return buf.getvalue()


def cmd_record() -> int:
    _CORPUS.mkdir(parents=True, exist_ok=True)
    # Drop any stale ".actions" files from the previous (forced-replay) format —
    # the deterministic scripted transcript no longer needs them.
    for stale in _CORPUS.glob("*.actions"):
        stale.unlink()
    for name, deck_a, deck_b in SCENARIOS:
        transcript = _play(deck_a, deck_b, SEED)
        (_CORPUS / f"{name}.txt").write_text(transcript)
        print(f"  recorded {name}: {len(transcript)} bytes")
    print(f"Recorded {len(SCENARIOS)} scenarios (seed {SEED}) to {_CORPUS}")
    return 0


def cmd_check() -> int:
    failures = []
    for name, deck_a, deck_b in SCENARIOS:
        expected_path = _CORPUS / f"{name}.txt"
        if not expected_path.exists():
            print(f"  MISSING corpus for {name} — run `record` first")
            failures.append(name)
            continue
        expected = expected_path.read_text()
        actual = _play(deck_a, deck_b, SEED)
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
