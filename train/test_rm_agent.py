#!/usr/bin/env python3
"""The ``rmagent`` gate tier: the scripted agent's RMSTATE view (plan step 2, pass 2).

The scripted agent's decisions (``scripted_core``) read, on ``--machine-format rmstate``, the
deciding seat's decoded frame (``rm_agent.FrameView``) and the card knowledge of the match's
``CARDS`` frame's structured columns (``rm_agent.FrameCards``). The replay corpus pins its games
(``regression/replay_diff.py``, the ``replay`` tier); this tier checks what the corpus cannot:

  1. names    every card the decision rules name (the name constants of ``scripted_core``) is a
              card of a ``--closed-decklist`` CARDS frame (every vocab card), so a misspelt or
              renamed card cannot silently disable a rule.
  2. cards    ``FrameCards`` against the vocab tables the BQUERY agent reads
              (``scripted_agent.VocabCardKnowledge``, by name) on every vocab card: they answer
              alike except the differences listed in ``EXPECTED_CARD_DIFFERENCES``, each one where
              the structured columns follow the rules and the generated tables do not.
  3. closed   scripted-hard bo3 matches under ``--closed-decklist`` (the opponent's list
              withheld; the agent reads the lists it can be on from ``k_template``), played by the
              RMSTATE driver (``rm_driver.py``), finish with a winner and no engine error.
  4. driver   the RMSTATE driver itself: its engine seeds are the gymnasium env's (pinned values,
              and numpy's draw when numpy is installed), its hand-line card names are the card
              vocabulary's for every card of a ``--closed-decklist`` CARDS frame, its verbose
              transcripts of bo3 matches (scripted hard, explore, explore:patient) equal
              ``runner.run_games``' on ``--machine-format rmstate`` byte for byte, a game imports
              nothing of the RL stack (``DRIVER_FORBIDDEN``), and concessions (``actions:`` with
              ``-2`` / ``-3``) end games and matches as the engine scores them.

  train/.venv/bin/python train/test_rm_agent.py               # the tier
  train/.venv/bin/python train/test_rm_agent.py --part cards  # one part
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from cli_spec import BINARY  # noqa: E402

PARTS = ("names", "cards", "closed", "driver")

# Where FrameCards (the CARDS frame's structured columns) and the generated vocab tables answer
# a card question differently: (card, question) -> why. Every other card and question must agree.
_MV_HYBRID = "mana value: a hybrid pip counts once (CR 202.3e); the cost table counts both colours"
EXPECTED_CARD_DIFFERENCES = {
    ("Yorion, Sky Nomad", "mana_value"): _MV_HYBRID,
    ("Raph & Mikey, Troublemakers", "mana_value"): _MV_HYBRID,
    ("Emrakul, the Aeons Torn", "mana_value"): "mana value 15: the cost table reads the "
                                               "two-digit generic pip as 1 + 5",
    ("Dead", "mana_value"): "a split card's half has its own mana value (CR 709.3); the vocab "
                            "has one row for Dead/Gone",
    ("Gone", "mana_value"): "a split card's half has its own mana value (CR 709.3)",
    ("Price of Progress", "deals_damage"): "damage to each player, no chosen target (the cost "
                                           "table's flag is any DealDamage in the script)",
    ("Ancient Tomb", "deals_damage"): "a mana ability's damage to its controller, no chosen target",
    ("Smash to Smithereens", "deals_damage"): "damage to the destroyed artifact's controller, no "
                                              "chosen player target",
    ("Gone", "deals_damage"): "a split card's half: Gone returns a creature; the vocab's one "
                              "Dead/Gone row has Dead's damage",
}
_QUESTIONS = ("mana_value", "cost_colors", "is_land", "power", "is_delve", "is_equipment",
              "has_miracle", "is_legendary", "land_colors", "deals_damage", "has_nonmana_activation",
              "is_artifact_or_enchantment")
# The core's module constants that are not card names.
_NOT_CARDS = ("_PRESETS", "_MANA_CATS",
              "_SEARCH_MENU_CATS", "_HARD_CONFIG", "_FILTERED")

CLOSED_GAMES = (("league/wubg_doomsday", "league/burn", 41), ("league/bw_dnt", "league/wrb_energy", 42),
                ("league/reanimator", "league/uw_control", 43))


def closed_cards_frame(binary):
    """(schema, CARDS frame) of a --closed-decklist engine (every vocab card)."""
    import rmstate
    proc = subprocess.Popen([os.path.realpath(binary), "--machine-format", "rmstate", "--closed-decklist",
                             "--seed", "1", "--deck-a", "league/bug", "--deck-b", "league/burn"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            cwd=os.path.join(_REPO_ROOT, "bin"))
    dec = rmstate.StreamDecoder()
    try:
        for payload in rmstate.records(proc.stdout):
            frame = dec.feed(payload)
            if frame.kind == "CARDS":
                return dec.schema, frame
    finally:
        proc.kill()
        proc.wait()
    return None, None


def core_card_names():
    """Every card name the decision rules name (the core's name constants)."""
    import scripted_core as core
    out = set()
    for name, value in vars(core).items():
        if (not name.startswith("_") or name.startswith(("__", "_CAT_", "_REF_", "_STEP_"))
                or name in _NOT_CARDS or callable(value)):
            continue
        if isinstance(value, str):
            out.add(value)
        elif isinstance(value, (frozenset, set, tuple, dict)):
            out |= {x for x in value if isinstance(x, str)}
    return out


def check_names(binary):
    import rm_agent
    schema, cards = closed_cards_frame(binary)
    if cards is None:
        return ["names: the engine wrote no CARDS frame"]
    known = {rm_agent.fold_name(r["name"]) for r in cards.tables["card"]}
    names = core_card_names()
    print(f"  {len(names)} card names checked", flush=True)
    return [f"names: scripted_core names {n!r}, which no CARDS row has" for n in sorted(names - known)]


def check_cards(binary):
    import rm_agent
    import scripted_agent
    schema, cards = closed_cards_frame(binary)
    if cards is None:
        return ["cards: the engine wrote no CARDS frame"]
    frame_k = rm_agent.FrameCards(schema, cards)
    vocab_k = scripted_agent._NameCardKnowledge(scripted_agent.VocabCardKnowledge())
    fails, seen = [], set()
    n = 0
    for r in cards.tables["card"]:
        if r["token"] or r["face_card"] is None or r["face_card"] == 0xFFFF:
            continue
        name = rm_agent.fold_name(r["name"])
        vname = scripted_agent._id_name(r["face_card"])
        n += 1
        for q in _QUESTIONS:
            if q == "has_nonmana_activation" and not frame_k.is_land(name):
                continue          # asked of lands only
            a, b = getattr(frame_k, q)(name), getattr(vocab_k, q)(vname)
            key = (name, q)
            if a != b:
                seen.add(key)
                if key not in EXPECTED_CARD_DIFFERENCES:
                    fails.append(f"cards: {name}: {q} {a!r} from CARDS, {b!r} from the vocab tables "
                                 "(list it in EXPECTED_CARD_DIFFERENCES with the reason, or fix it)")
    for key in sorted(set(EXPECTED_CARD_DIFFERENCES) - seen):
        fails.append(f"cards: {key[0]}: {key[1]} is listed as differing but agrees")
    print(f"  {n} card faces compared, {len(seen)} listed differences", flush=True)
    return fails


def _closed_task(task):
    deck_a, deck_b, seed, binary = task
    import rm_driver
    buf = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        w, l, d = rm_driver.run_games("scripted", "scripted", deck_a=deck_a, deck_b=deck_b, n_games=1,
                                      seed=seed, bo3=True, transcript="compact", out=buf, binary=binary,
                                      engine_flags=["--closed-decklist"])
    return deck_a, deck_b, seed, (w, l, d), buf.getvalue() + err.getvalue()


def check_closed(binary):
    fails = []
    tasks = [(a, b, s, binary) for a, b, s in CLOSED_GAMES]
    with ProcessPoolExecutor(max_workers=min(len(tasks), os.cpu_count() or 4)) as ex:
        for deck_a, deck_b, seed, (w, l, d), text in ex.map(_closed_task, tasks):
            label = f"closed: {deck_a} vs {deck_b} seed {seed}"
            bad = [ln for ln in text.splitlines() if ln.startswith(("ERROR:", "FATAL", "Traceback"))]
            if w + l != 1 or d:
                fails.append(f"{label}: no winner ({w}W {l}L {d}D)")
            if bad:
                fails.append(f"{label}: {bad[0]!r}")
            print(f"  {'FAIL' if (bad or w + l != 1) else 'ok  '} {label}: {'A' if w else 'B'} wins", flush=True)
    return fails


# Game seed -> engine seed, as numpy's PCG64(SeedSequence(seed)).integers(0, 2**31 - 1) draws it.
PINNED_ENGINE_SEEDS = ((0, 1826701614), (1, 1016164991), (2, 1798679647), (7, 2029167940),
                       (1001, 1940026651), (2147483647, 1751933748), (4294967295, 409763535),
                       (4294967296, 1121323793), (1099511627781, 30406389), (13532794167, 596834746),
                       (1641108517, 1187322509))
# Modules of the RL stack (deleted in plan step 3b) the driver must not import.
DRIVER_FORBIDDEN = ("numpy", "gymnasium", "torch", "env", "runner", "opponents", "decode", "cli_spec",
                    "_enums", "card_costs", "card_props", "scripted_agent", "rmstate_env", "rmstate_adapter")
# bo3 matches whose transcripts the driver and runner.run_games must write alike: (agent, deck A,
# deck B, seed) — the explore ones hold a split card (Dead/Gone) in hand.
TRANSCRIPT_MATCHES = (("scripted", "league/ur_delver", "league/reanimator", 133),
                      ("explore", "league/burn", "league/lands", 107),
                      ("explore:patient", "league/wrb_energy", "league/burn", 156))


def _transcript_task(task):
    spec, deck_a, deck_b, seed, binary = task
    import rm_driver
    import runner
    from opponents import make_controller
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runner.run_games(make_controller(spec), make_controller(spec), label_a=f"A:{spec}", label_b=f"B:{spec}",
                         binary_path=binary, deck_a=deck_a, deck_b=deck_b, n_games=1, seed=seed, bo3=True,
                         verbose=True, machine_format="rmstate")
    out = io.StringIO()
    rm_driver.run_games(spec, spec, label_a=f"A:{spec}", label_b=f"B:{spec}", binary=binary, deck_a=deck_a,
                        deck_b=deck_b, n_games=1, seed=seed, bo3=True, transcript="verbose", out=out)
    want, got = buf.getvalue().splitlines(), out.getvalue().splitlines()
    for i in range(max(len(want), len(got))):
        w = want[i] if i < len(want) else "<end>"
        g = got[i] if i < len(got) else "<end>"
        if w != g:
            return task, f"line {i + 1}: runner {w!r}, driver {g!r}", len(want)
    return task, None, len(want)


def check_driver(binary):
    """The driver part; its games run in a scratch directory, so a draw's log stays out of the
    tree (a draw fails the part: these games are chosen to have a winner)."""
    import tempfile
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="rmagent_driver_") as scratch:
        os.chdir(scratch)
        try:
            fails = _check_driver(os.path.realpath(os.path.join(cwd, binary)))
        finally:
            os.chdir(cwd)
        draws = sorted(f for f in os.listdir(scratch) if f.startswith("draw_"))
        for f in draws:
            with open(os.path.join(scratch, f)) as fh:
                tail = fh.read().strip().splitlines()[-2:]
            fails.append(f"driver: a game drew (choose games with a winner): {' | '.join(tail)}")
    return fails


def _check_driver(binary):
    import rm_driver
    fails = []
    for seed, want in PINNED_ENGINE_SEEDS:
        if rm_driver.engine_seed(seed) != want:
            fails.append(f"driver: engine_seed({seed}) = {rm_driver.engine_seed(seed)}, pinned {want}")
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        import random
        rng = random.Random(11)
        for seed in [rng.randrange(2 ** 40) for _ in range(300)]:
            want = int(np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed))).integers(0, 2 ** 31 - 1))
            if rm_driver.engine_seed(seed) != want:
                fails.append(f"driver: engine_seed({seed}) = {rm_driver.engine_seed(seed)}, numpy {want}")
                break
    import decode
    schema, cards = closed_cards_frame(binary)
    names = rm_driver._CardNames(schema, cards)
    tokens = {r["card"] for r in cards.tables["card"] if r["token"]}
    n = 0
    for cid, name in sorted(names.display.items()):
        if cid in tokens:
            continue
        n += 1
        if name != decode.card_index_to_name(cid):
            fails.append(f"driver: card {cid} reads {name!r}, the vocab names it {decode.card_index_to_name(cid)!r}")
    print(f"  engine seeds checked, {n} card names compared", flush=True)
    tasks = [(*m, binary) for m in TRANSCRIPT_MATCHES]
    with ProcessPoolExecutor(max_workers=min(len(tasks), os.cpu_count() or 4)) as ex:
        for task, fail, lines in ex.map(_transcript_task, tasks):
            label = f"driver: {task[0]} {task[1]} vs {task[2]} seed {task[3]} (bo3)"
            print(f"  {'FAIL' if fail else 'ok  '} {label}: {lines} transcript lines", flush=True)
            if fail:
                fails.append(f"{label}: {fail}")
    # The driver plays without the RL stack: none of its modules is imported.
    probe = ("import sys; import rm_driver; rm_driver.run_games('scripted', 'explore', binary=sys.argv[1], "
             "deck_a='league/burn', seed=3, transcript='quiet'); print(' '.join(m for m in DRIVER_FORBIDDEN "
             "if m in sys.modules))").replace("DRIVER_FORBIDDEN", repr(DRIVER_FORBIDDEN))
    r = subprocess.run([sys.executable, "-c", probe, binary], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=_HERE))
    if r.returncode != 0 or r.stdout.strip():
        fails.append(f"driver: a game imported {r.stdout.strip() or '?'} (exit {r.returncode}): {r.stderr[-300:]}")
    # Concessions: a bo1 game conceded at A's third decision, a bo3 match conceded at once, a bo3
    # game conceded (the match plays on).
    for spec, bo3, want in (("actions:0,0,-2", False, (0, 1, 0)), ("actions:-3", True, (0, 1, 0)),
                            ("actions:-2", True, None)):
        recs = []
        wld = rm_driver.run_games(spec, "scripted", binary=binary, deck_a="league/burn", deck_b="league/bug",
                                  seed=5, bo3=bo3, transcript="quiet", on_game_end=recs.append)
        label = f"driver: concession {spec} ({'bo3' if bo3 else 'bo1'})"
        results = recs[0].game_results
        fail = None
        if want is not None and wld != want:
            fail = f"{label}: {wld}, expected {want}"
        if want is None and (results[:1] != [("B", True)] or len(results) < 2):
            fail = f"{label}: game 1 should go to B and the match play on: {results}"
        print(f"  {'FAIL' if fail else 'ok  '} {label}: {results}", flush=True)
        if fail:
            fails.append(fail)
    return fails


def run(binary=BINARY, parts=None):
    """The tier; returns failure messages (empty = pass)."""
    fails, times = [], []
    for part in PARTS:
        if parts and part not in parts:
            continue
        t0 = time.monotonic()
        print(f"rmagent: {part}", flush=True)
        fails += {"names": check_names, "cards": check_cards, "closed": check_closed,
                  "driver": check_driver}[part](binary)
        times.append((part, time.monotonic() - t0))
    print("rmagent: " + ", ".join(f"{p} {dt:.0f}s" for p, dt in times), flush=True)
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--binary", default=BINARY, help="engine binary (default: the debug tier)")
    ap.add_argument("--part", action="append", choices=PARTS, help="run only this part (repeatable)")
    args = ap.parse_args(argv)
    fails = run(args.binary, args.part)
    for msg in fails:
        print(f"FAIL {msg}", flush=True)
    print("rmagent: " + ("FAIL" if fails else "PASS"), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
