#!/usr/bin/env python3
"""The C++ in-process agents against their Python originals (plan step 3a, part 1).

Two comparisons, each run on the engine binary (``--binary``, default the debug build):

``explore``: the C++ explore tier (``--agent-a explore --agent-b explore``, ``src/agents/explore.*``)
against the Python one (``explore_agent.py`` through ``rm_driver.play_game``) on the same games:
game seed s is ``--game-seed s`` for the engine (the drivers' engine seed, and the agents'
generator seeded with s) and ``random.seed(s)`` + ``--seed engine_seed(s)`` for the driver. A game
matches when both play the same replies at every decision (the C++ run's ``--trace-decisions``
choice lines against the driver's record) and print the same narrative and results.

``scenarios``: every ``regression/scenarios/*.json`` run through the Python harness
(``test_harness.py --scenario``, as ``test_scenarios.py`` runs it) and through the engine alone
(``--play`` with the scenario's presets as engine flags, ``--trace-decisions``). A scenario matches
when both print the same ``resolved --actions`` list; the scenario's assertions are checked on
both outputs (``test_scenarios.check_output`` on the harness's, the same checks on the engine's
narrative and trace menus; the knowledge-dump assertions on both runs' ``--rmstate-dump``).

Usage:
  train/.venv/bin/python train/compare_cpp_agents.py explore --games 50 [--format bo3] [--matchups ...]
  train/.venv/bin/python train/compare_cpp_agents.py scenarios [names/globs ...]
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

_TRAIN = os.path.dirname(os.path.abspath(__file__))
if _TRAIN not in sys.path:
    sys.path.insert(0, _TRAIN)

import rm_driver  # noqa: E402

REPO_ROOT = os.path.dirname(_TRAIN)
BIN_DIR = os.path.join(REPO_ROOT, "bin")
DEFAULT_BINARY = os.path.join(BIN_DIR, os.environ.get("ROBOMAGE_BUILD", "debug"), "robomage")
DEFAULT_MATCHUPS = ("league/ur_delver:league/gw_maverick", "league/bug:league/ur_delver",
                    "league/gw_maverick:league/bug")
_TRACE = "| "
_CHOICE_RE = re.compile(r"^\| >> Player [AB]: (-?\d+)")
_MENU_RE = re.compile(r"^\|   \d+: (.*)$")
# Engine lines that are neither narrative nor trace (the driver files them as diagnostics or drops
# them as training signals).
_DIAGNOSTIC_PREFIXES = tuple(p.decode() for p in rm_driver._DIAGNOSTIC_PREFIXES + rm_driver._SIGNAL_PREFIXES)


# ── explore ────────────────────────────────────────────────────────────────────────────────

def _engine_lines(text):
    """The engine output's narrative lines (blank lines, trace lines and diagnostics dropped)."""
    return [ln for ln in text.splitlines()
            if ln.strip() and not ln.startswith(_TRACE) and not ln.startswith(_DIAGNOSTIC_PREFIXES)]


def run_cpp_explore(binary, deck_a, deck_b, seed, bo3, spec="explore"):
    """(narrative lines, replies, seconds) of one C++ explore game."""
    cmd = [binary, "--agent-a", spec, "--agent-b", spec, "--game-seed", str(seed), "--narrative",
           "--trace-decisions", "--deck-a", deck_a, "--deck-b", deck_b] + (["--bo3"] if bo3 else [])
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=BIN_DIR, capture_output=True, text=True, errors="replace")
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"engine exited {r.returncode}: {r.stderr[-400:]}")
    out = r.stdout
    replies = [int(m.group(1)) for m in map(_CHOICE_RE.match, out.splitlines()) if m]
    return _engine_lines(out), replies, dt


def run_py_explore(binary, deck_a, deck_b, seed, bo3, spec="explore"):
    """(narrative lines, replies, seconds) of one Python explore game over the driver."""
    lines = []
    t0 = time.perf_counter()
    rec = rm_driver.play_game(rm_driver.make_agent(spec), rm_driver.make_agent(spec), binary=binary,
                              deck_a=deck_a, deck_b=deck_b, bo3=bo3, seed=seed, narrative=True,
                              on_narrative=lambda ls: lines.extend(ls))
    dt = time.perf_counter() - t0
    if rec.diagnostics:
        raise RuntimeError("driver diagnostics: " + " | ".join(rec.diagnostics[-3:]))
    return [ln for ln in lines if ln.strip()], list(rec.actions), dt


def explore_job(job):
    """One game on both sides (a process of its own: the driver's explore tier draws from Python's
    global random, which concurrent games in one process would share)."""
    binary, a, b, s, bo3, spec = job
    c_lines, c_replies, c_dt = run_cpp_explore(binary, a, b, s, bo3, spec)
    p_lines, p_replies, p_dt = run_py_explore(binary, a, b, s, bo3, spec)
    diff = None
    if c_replies != p_replies:
        k = next((i for i, (x, y) in enumerate(zip(c_replies, p_replies)) if x != y),
                 min(len(c_replies), len(p_replies)))
        diff = f"replies differ at decision {k + 1} (C++ {len(c_replies)}, Python {len(p_replies)})"
    elif c_lines != p_lines:
        k = next((i for i, (x, y) in enumerate(zip(c_lines, p_lines)) if x != y),
                 min(len(c_lines), len(p_lines)))
        diff = (f"narrative differs at line {k + 1}: "
                f"{c_lines[k] if k < len(c_lines) else None!r} vs {p_lines[k] if k < len(p_lines) else None!r}")
    return (a, b, s), diff, len(c_replies), c_dt, p_dt


def compare_explore(args):
    pairs = [m.split(":") for m in (args.matchups or DEFAULT_MATCHUPS)]
    bo3 = args.format == "bo3"
    jobs = [(args.binary, a, b, args.seed + i, bo3, args.spec) for a, b in pairs for i in range(args.games)]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(explore_job, jobs))
    bad = [r for r in results if r[1]]
    for (a, b, s), diff, n, _c, _p in results:
        if diff or args.verbose:
            print(f"  {'FAIL' if diff else 'ok  '} {a} vs {b} seed {s}: {n} decisions"
                  + (f": {diff}" if diff else ""), flush=True)
    decisions = sum(r[2] for r in results)
    c_time, p_time = sum(r[3] for r in results), sum(r[4] for r in results)
    print(f"explore ({args.spec}, {args.format}): {len(results) - len(bad)}/{len(results)} games identical, "
          f"{decisions} decisions; serial time C++ {c_time:.1f}s ({decisions / max(c_time, 1e-9):.0f} decisions/s), "
          f"Python driver {p_time:.1f}s ({decisions / max(p_time, 1e-9):.0f} decisions/s)", flush=True)
    return 1 if bad else 0


# ── scenarios ──────────────────────────────────────────────────────────────────────────────

def _flag_list(cards):
    return ",".join(c.replace("'", "").replace("’", "") for c in cards)


def engine_scenario_cmd(scn, binary, dump_path=None):
    """The engine command line that plays a scenario the way test_harness.py does."""
    import test_scenarios
    cmd = [binary, "--game-seed", str(scn.get("seed", 1)), "--narrative", "--trace-decisions"]
    if scn.get("format", "bo1") == "bo3":
        cmd.append("--bo3")
    cmd += ["--max-decisions", str(scn.get("max_decisions", test_scenarios.DEFAULT_MAX_DECISIONS))]
    for seat in ("a", "b"):
        hand = scn.get(f"hand_{seat}", [])
        if hand:
            cmd += [f"--hand-{seat}", ",".join(hand)]
            if scn.get(f"library_{seat}"):
                cmd += [f"--library-{seat}", ",".join(scn[f"library_{seat}"])]
        else:
            cmd += [f"--deck-{seat}", scn.get(f"deck_{seat}", "delver")]
        for zone in ("battlefield", "graveyard", "exile", "sideboard"):
            cards = scn.get(f"{zone}_{seat}", [])
            if cards:
                cmd += [f"--{zone}-{seat}", _flag_list(cards)]
        if scn.get(f"life_{seat}") is not None:
            cmd += [f"--life-{seat}", str(scn[f"life_{seat}"])]
    play = scn.get("play")
    if play:
        cmd += ["--play", ",".join(str(p).strip() for p in play if str(p).strip())]
    elif scn.get("actions"):
        cmd += ["--actions", ",".join(str(a) for a in scn["actions"])]
    else:
        cmd += ["--agent-a", "first", "--agent-b", "first"]
    if scn.get("offer_cancel"):
        cmd.append("--offer-cancel")
    cmd += list(scn.get("engine_flags", []))
    if dump_path:
        cmd += ["--rmstate-dump", dump_path, "--state-dump", "json"]
    return cmd


def split_engine_output(text):
    """(narrative lines from the first turn banner on, offered-action descriptions) of an engine
    run under --trace-decisions."""
    import test_scenarios
    narrative, actions = [], []
    for line in text.splitlines():
        if line.startswith(_TRACE):
            m = _MENU_RE.match(line)
            if m:
                actions.append(m.group(1))
        elif line.strip():
            narrative.append(line)
    first_turn = next((i for i, ln in enumerate(narrative) if test_scenarios._TURN_RE.search(ln)), 0)
    return narrative[first_turn:], actions


def check_engine_output(scn, text):
    """test_scenarios.check_output's assertions on an engine run's output."""
    import test_scenarios
    saved = test_scenarios.split_output
    test_scenarios.split_output = split_engine_output
    try:
        return test_scenarios.check_output(scn, text)
    finally:
        test_scenarios.split_output = saved


_RESOLVED_RE = re.compile(r"^resolved --actions: (.*)$", re.M)


def resolved_list(text):
    m = _RESOLVED_RE.search(text)
    return m.group(1).strip() if m else None


def run_engine_scenario(scn, binary, dump):
    """(output, knowledge frames) of one scenario on the engine alone."""
    import test_scenarios
    dump_dir = tempfile.mkdtemp(prefix="cmp_scenario_") if dump else None
    dump_path = os.path.join(dump_dir, "frames.rms") if dump_dir else None
    try:
        r = subprocess.run(engine_scenario_cmd(scn, binary, dump_path), cwd=BIN_DIR, capture_output=True,
                           text=True, errors="replace", timeout=test_scenarios._TIMEOUT_S)
        out = r.stdout + r.stderr
        if r.returncode != 0:
            out += f"\nERROR: engine exited with code {r.returncode}\n"
        frames = []
        if dump_path and os.path.exists(dump_path + ".jsonl"):
            frames = test_scenarios.knowledge_frames(dump_path + ".jsonl")
        return out, frames
    finally:
        if dump_dir:
            shutil.rmtree(dump_dir, ignore_errors=True)


def compare_one_scenario(path, binary):
    import test_scenarios
    name = os.path.basename(path)[:-5]
    scn = test_scenarios.load_scenario(path)
    dump = test_scenarios.wants_dump(scn)
    _n, py_fails, py_out, _py_frames = test_scenarios.run_one(path, binary)
    c_out, c_frames = run_engine_scenario(scn, binary, dump)
    c_fails = check_engine_output(scn, c_out)
    if dump:
        c_fails += test_scenarios.check_dump(scn, c_frames)
    for i, variant in enumerate(scn.get("variants", [])):
        merged = {k: v for k, v in scn.items() if k != "variants"}
        merged.update({k: v for k, v in variant.items() if k != "expect_same"})
        v_out, _ = run_engine_scenario(merged, binary, False)
        c_fails += [f"variant {i}: {f}" for f in check_engine_output(
            {k: v for k, v in scn.items() if k not in ("expect_same", "variants")}, v_out)]
        patterns = variant.get("expect_same", scn.get("expect_same", []))
        rxs = [re.compile(rx) for rx in patterns]
        same = [[ln for ln in split_engine_output(t)[0] if any(rx.search(ln) for rx in rxs)] for t in (c_out, v_out)]
        if not same[0] or same[0] != same[1]:
            c_fails.append(f"variant {i}: expect_same lines differ")
    py_resolved, c_resolved = resolved_list(py_out), resolved_list(c_out)
    return name, py_resolved, c_resolved, py_fails, c_fails


def compare_scenarios(args):
    import test_scenarios
    paths = test_scenarios.scenario_paths(args.patterns)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(lambda p: compare_one_scenario(p, args.binary), paths))
    same = 0
    bad = 0
    for name, py_r, c_r, py_f, c_f in results:
        match = py_r == c_r
        same += match
        if not match or py_f or c_f:
            bad += 1
            print(f"  FAIL {name}", flush=True)
            if not match:
                print(f"         resolved --actions differ:\n           Python {py_r}\n           C++    {c_r}")
            for f in py_f:
                print(f"         Python: {f}")
            for f in c_f:
                print(f"         C++: {f}")
        elif args.verbose:
            print(f"  ok   {name}")
    n_py = sum(1 for r in results if not r[3])
    n_c = sum(1 for r in results if not r[4])
    print(f"scenarios: {same}/{len(results)} resolve to the same --actions; assertions pass "
          f"Python {n_py}/{len(results)}, C++ {n_c}/{len(results)}", flush=True)
    return 1 if bad else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary", default=DEFAULT_BINARY)
    p.add_argument("-j", "--jobs", type=int, default=min(16, os.cpu_count() or 4))
    p.add_argument("--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("explore", help="C++ explore vs Python explore on the same games")
    e.add_argument("--games", type=int, default=10, help="games per matchup (default 10)")
    e.add_argument("--seed", type=int, default=1)
    e.add_argument("--format", choices=("bo1", "bo3"), default="bo1")
    e.add_argument("--spec", default="explore", choices=("explore", "explore:patient"))
    e.add_argument("--matchups", nargs="*", help="deck_a:deck_b pairs")
    s = sub.add_parser("scenarios", help="the rules scenarios on the C++ --play resolver vs the harness")
    s.add_argument("patterns", nargs="*")
    args = p.parse_args(argv)
    args.binary = os.path.realpath(args.binary)
    return compare_explore(args) if args.cmd == "explore" else compare_scenarios(args)


if __name__ == "__main__":
    sys.exit(main())
