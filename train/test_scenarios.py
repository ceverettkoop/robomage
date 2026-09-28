#!/usr/bin/env python3
"""Rules-regression scenarios: sculpted test_harness games with asserted outcomes.

Each ``train/regression/scenarios/*.json`` file is a test_harness ``--scenario``
(zone presets, hands, a ``play`` spec list, seed; see CLAUDE.md "JSON
scenarios") extended with assertions about the RULES OUTCOME of the line it
plays. Every scenario runs as a bo1 harness process (in parallel), and a
scenario fails when an assertion misses, the engine or harness prints an error
(``ERROR:`` / ``FATAL:`` / a traceback / a failed ``--play`` spec), or the
``play`` script ends before all of its specs were applied.

Scenario keys beyond the harness's own:

  why            one line: the CR rule and the commit that fixed the behavior
  expect         regexes that must each match some narrative line
  expect_not     regexes that must match no narrative line
  expect_order   regexes that must match narrative lines in this order
  expect_menu    regexes that must each match some offered action (any decision)
  expect_menu_not  regexes that must match no offered action
  expect_result  "A" | "B" | "draw": the GAME_RESULT the game must end with
  player_a / player_b  the seats' agents once the script runs out (default auto)
  deck_a / deck_b      a deck for a seat with no hand preset
  offer_cancel   true: run with --offer-cancel

The narrative is the engine's game log (the harness's "--- Narrative ---"
blocks) from the first turn banner on (the pregame's opening draws are left
out), so assertions stay independent of the decoded board / menu layout.
``max_decisions`` defaults to 60: keep it just past the asserted event.

Usage:
  train/.venv/bin/python train/test_scenarios.py              # run all
  train/.venv/bin/python train/test_scenarios.py ward*        # a subset (globs)
  train/.venv/bin/python train/test_scenarios.py --show NAME  # print one run's output
  train/.venv/bin/python train/test_scenarios.py --narrative NAME  # just its narrative
  train/.venv/bin/python train/test_scenarios.py --binary PATH  # another engine build
"""
import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from test_harness import SCRIPT_INCOMPLETE

_TRAIN = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TRAIN)
SCENARIO_DIR = os.path.join(_TRAIN, "regression", "scenarios")
_HARNESS = os.path.join(_TRAIN, "test_harness.py")

DEFAULT_MAX_DECISIONS = 60
_TIMEOUT_S = 120

# Keys test_harness reads from a scenario file.
_HARNESS_KEYS = {"name", "hand_a", "hand_b", "library_a", "library_b",
                 "battlefield_a", "battlefield_b", "graveyard_a", "graveyard_b",
                 "exile_a", "exile_b", "sideboard_a", "sideboard_b",
                 "life_a", "life_b", "play", "actions", "seed", "max_decisions"}
_ASSERT_KEYS = {"expect", "expect_not", "expect_order", "expect_menu",
                "expect_menu_not", "expect_result"}
_RUN_KEYS = {"why", "player_a", "player_b", "deck_a", "deck_b", "offer_cancel"}
_KNOWN_KEYS = _HARNESS_KEYS | _ASSERT_KEYS | _RUN_KEYS

_ERROR_RE = re.compile(
    r"^\s*(ERROR|FATAL):|Traceback \(most recent call last\)|Segmentation fault|"
    r"Assertion .* failed|\bAborted\b|core dumped|PlayResolveError|"
    + re.escape(SCRIPT_INCOMPLETE))
_ACTION_RE = re.compile(r"^\s+\d+: (.*)$")
_TURN_RE = re.compile(r"^-+ TURN \d+ ")
_RESULT_RE = re.compile(r"GAME_RESULT: \d+ (Player (A|B) wins|draw)")


def scenario_paths(patterns=None):
    """Scenario files, sorted, optionally filtered by name globs."""
    names = sorted(f for f in os.listdir(SCENARIO_DIR) if f.endswith(".json"))
    if patterns:
        names = [n for n in names
                 if any(fnmatch.fnmatch(n[:-5], p) or fnmatch.fnmatch(n, p)
                        for p in patterns)]
    return [os.path.join(SCENARIO_DIR, n) for n in names]


def split_output(text):
    """(narrative lines, offered-action descriptions) of one harness run."""
    narrative, actions = [], []
    section = None
    for line in text.splitlines():
        if line.startswith("--- Narrative ---"):
            section = "narrative"
            continue
        if line.startswith("--- ") or line.startswith("=== "):
            section = None
            continue
        if line.strip() == "Actions:":
            section = "actions"
            continue
        if section == "narrative":
            if line.strip():
                narrative.append(line[2:] if line.startswith("  ") else line)
        elif section == "actions":
            m = _ACTION_RE.match(line)
            if m:
                actions.append(re.sub(r"\s+\[#\d+\]$", "", m.group(1)))
            else:
                section = None
    # The pregame (opening draws and hands) precedes the first turn banner and
    # is never what a scenario asserts on.
    first_turn = next((i for i, ln in enumerate(narrative) if _TURN_RE.search(ln)), 0)
    return narrative[first_turn:], actions


def harness_cmd(scn, path, binary=None):
    """The test_harness command line for one scenario."""
    cmd = [sys.executable, _HARNESS, "--scenario", path, "--format", "bo1",
           "--player-a", scn.get("player_a", "auto"),
           "--player-b", scn.get("player_b", "auto")]
    if "max_decisions" not in scn:
        cmd += ["--max-decisions", str(DEFAULT_MAX_DECISIONS)]
    for key in ("deck_a", "deck_b"):
        if key in scn:
            cmd += ["--" + key.replace("_", "-"), scn[key]]
    if scn.get("offer_cancel"):
        cmd.append("--offer-cancel")
    if binary:
        cmd += ["--binary", binary]
    return cmd


def check_output(scn, text):
    """Failure messages (empty = pass) for one scenario's harness output."""
    fails = []
    for line in text.splitlines():
        if _ERROR_RE.search(line):
            fails.append(f"error line: {line.strip()}")
            break
    narrative, actions = split_output(text)
    for rx in scn.get("expect", []):
        if not any(re.search(rx, ln) for ln in narrative):
            fails.append(f"missing narrative line /{rx}/")
    for rx in scn.get("expect_not", []):
        hit = next((ln for ln in narrative if re.search(rx, ln)), None)
        if hit is not None:
            fails.append(f"forbidden narrative line /{rx}/: {hit!r}")
    pos = 0
    for rx in scn.get("expect_order", []):
        k = next((i for i in range(pos, len(narrative))
                  if re.search(rx, narrative[i])), None)
        if k is None:
            fails.append(f"out of order or missing narrative line /{rx}/ "
                         f"(searched from line {pos})")
            break
        pos = k + 1
    for rx in scn.get("expect_menu", []):
        if not any(re.search(rx, a) for a in actions):
            fails.append(f"no offered action matches /{rx}/")
    for rx in scn.get("expect_menu_not", []):
        hit = next((a for a in actions if re.search(rx, a)), None)
        if hit is not None:
            fails.append(f"forbidden action offered /{rx}/: {hit!r}")
    want = scn.get("expect_result")
    if want is not None:
        m = _RESULT_RE.search(text)
        got = None if m is None else (m.group(2) or "draw")
        if got != want:
            fails.append(f"expected result {want!r}, got {got!r}")
    return fails


def load_scenario(path):
    """The parsed scenario, or raise ValueError naming a malformed field."""
    with open(path) as f:
        scn = json.load(f)
    unknown = set(scn) - _KNOWN_KEYS
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)}")
    if not scn.get("why"):
        raise ValueError("missing 'why'")
    if not (_ASSERT_KEYS & set(scn)):
        raise ValueError("no assertion (expect / expect_not / expect_order / "
                         "expect_menu / expect_menu_not / expect_result)")
    for key in _ASSERT_KEYS - {"expect_result"}:
        for rx in scn.get(key, []):
            re.compile(rx)
    return scn


def run_one(path, binary=None):
    """(name, failure messages, harness output) for one scenario file."""
    name = os.path.basename(path)[:-5]
    try:
        scn = load_scenario(path)
    except (ValueError, re.error, json.JSONDecodeError) as e:
        return name, [f"bad scenario file: {e}"], ""
    try:
        r = subprocess.run(harness_cmd(scn, path, binary), cwd=_REPO_ROOT,
                           capture_output=True, text=True, timeout=_TIMEOUT_S)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        return name, [f"timed out after {_TIMEOUT_S}s"], str(e.stdout or "")
    return name, check_output(scn, out), out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("patterns", nargs="*", help="scenario name globs (default: all)")
    p.add_argument("--show", action="store_true",
                   help="print each selected scenario's full harness output")
    p.add_argument("--narrative", action="store_true",
                   help="print each selected scenario's narrative (what expect* match)")
    p.add_argument("--binary", default=None, help="engine binary to run against")
    p.add_argument("-j", "--jobs", type=int, default=min(16, os.cpu_count() or 4))
    args = p.parse_args(argv)

    paths = scenario_paths(args.patterns)
    if not paths:
        print("no scenarios matched", file=sys.stderr)
        return 2
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(lambda q: run_one(q, args.binary), paths))
    failed = 0
    for name, fails, out in results:
        if args.show:
            print(f"##### {name}\n{out}")
        elif args.narrative:
            print(f"##### {name}")
            print("\n".join(split_output(out)[0]))
        if fails:
            failed += 1
            print(f"  FAIL {name}", flush=True)
            for f in fails:
                print(f"         {f}", flush=True)
            print(f"         reproduce: train/.venv/bin/python train/test_scenarios.py "
                  f"--show {name}", flush=True)
    print(f"scenarios: {len(results) - failed}/{len(results)} passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
