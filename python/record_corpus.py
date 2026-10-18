#!/usr/bin/env python3
"""Records the replay corpus (``tests/corpus/``) with the live scripted agent.

Plays a fixed set of scripted deck-vs-deck games (the three fully-implemented curated decks —
delver/doomsday/mav — in every seating, and every league pairing): the scripted agent (hard, both
seats) on its RMSTATE frame view (``rm_agent.py``), driven by the RMSTATE driver
(``rm_driver.py``) over ``--machine-format rmstate``. Each game writes its decision log
``<name>.rmlog`` (the engine's ``--decision-log``: seed, setup flags, both decklists, every
decision); then ``robomage_check --tier replay --record`` writes each log's replay baseline
``<name>.replay``, which the gate's ``replay`` tier checks byte for byte (no Python needed).

The scripted agent is a pure function of game state and the engine is deterministic given a
seed, so an unchanged engine + agent + card data records the same logs. Run it only when a
behavior change is INTENTIONAL (an engine/effect change, a scripted-agent change, or a card-data
change that legitimately alters these games), and commit the regenerated corpus as part of that
change's review. ``make regen`` runs it.

Usage (from the repo root; stdlib only)::

  python3 python/record_corpus.py [--binary bin/debug/robomage]
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_CORPUS = _REPO / "tests" / "corpus"

sys.path.insert(0, str(_HERE))

from rm_driver import DEFAULT_BINARY  # noqa: E402

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

# Full league-matchup coverage: every deck of the league roster against every
# other (each unordered pair once, the lexically-earlier deck seated as Player
# A) plus each deck's mirror — 55 scenarios named lg_<a>__<b>. One game per
# matchup keeps the corpus and the CI replay tier tractable; the curated
# scenarios above already exercise both seatings for the deepest decks.
LEAGUE_DECKS = [
    "bug", "burn", "bw_dnt", "gw_maverick", "lands", "reanimator",
    "ur_delver", "uw_control", "wrb_energy", "wubg_doomsday",
]
for _i, _a in enumerate(LEAGUE_DECKS):
    for _b in LEAGUE_DECKS[_i:]:
        SCENARIOS.append((f"lg_{_a}__{_b}", f"league/{_a}", f"league/{_b}"))


def _record_scenario(task) -> str:
    """Plays one SCENARIOS entry with its decision log written into the corpus (a picklable
    process-pool task); returns its name."""
    (name, deck_a, deck_b), binary = task
    import rm_driver  # noqa: E402

    rm_driver.run_games("scripted", "scripted", label_a="A", label_b="B", binary=binary, deck_a=deck_a,
                        deck_b=deck_b, n_games=1, seed=SEED, transcript="quiet",
                        engine_flags=["--decision-log", str(_CORPUS / f"{name}.rmlog")])
    return name


def record(binary: str) -> int:
    _CORPUS.mkdir(parents=True, exist_ok=True)
    names = {name for name, _a, _b in SCENARIOS}
    for stale in list(_CORPUS.glob("*.rmlog")) + list(_CORPUS.glob("*.replay")):
        if stale.stem not in names:
            stale.unlink()
    # The games are independent (each its own engine process, seeded by SEED), so they run in a
    # process pool; a decision log does not depend on which worker played it.
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as ex:
        for name in ex.map(_record_scenario, [(sc, binary) for sc in SCENARIOS]):
            log = _CORPUS / f"{name}.rmlog"
            if not log.exists():
                print(f"  {name}: no decision log written")
                return 1
            print(f"  recorded {name}: {log.stat().st_size} bytes")
    print(f"Recorded {len(SCENARIOS)} decision logs (seed {SEED}) to {_CORPUS}")
    # The decision logs' replay baselines, written by the C++ gate's replay tier.
    gate = Path(binary).parent / "robomage_check"
    r = subprocess.run([str(gate), "--tier", "replay", "--record", "--binary", binary])
    if r.returncode != 0:
        print(f"robomage_check --tier replay --record failed (exit {r.returncode})")
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="record_corpus", description="Record the replay corpus (tests/corpus/) "
                                 "with the live scripted agent.")
    ap.add_argument("--binary", default=DEFAULT_BINARY, help="the engine (default %(default)s)")
    args = ap.parse_args(argv)
    binary = os.path.abspath(args.binary)
    if not os.path.exists(binary):
        print(f"binary not found at {binary} — run `make` first")
        return 2
    return record(binary)


if __name__ == "__main__":
    raise SystemExit(main())
