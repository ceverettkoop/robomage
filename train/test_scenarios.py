#!/usr/bin/env python3
"""Rules-regression scenarios: sculpted test_harness games with asserted outcomes.

Each ``train/regression/scenarios/*.json`` file is a test_harness ``--scenario``
(zone presets, hands, a ``play`` spec list, seed; see CLAUDE.md "JSON
scenarios") extended with assertions about the RULES OUTCOME of the line it
plays. Every scenario runs as a harness process (in parallel; bo1 unless its
``format`` says bo3), and a
scenario fails when an assertion misses, the engine or harness prints an error
(``ERROR:`` / ``FATAL:`` / a traceback / a failed ``--play`` spec), or the
``play`` script ends before all of its specs were applied.

Scenario keys beyond the harness's own:

  why            one line: the CR rule and the commit that fixed the behavior
  expect         regexes that must each match some narrative line
  expect_not     regexes that must match no narrative line
  expect_order   regexes that must match narrative lines in this order
  expect_count   {regex: n}: each regex must match exactly n narrative lines
  expect_menu    regexes that must each match some offered action (any decision)
  expect_menu_not  regexes that must match no offered action
  expect_result  "A" | "B" | "draw": the GAME_RESULT the game must end with
  expect_unique  regexes with capture groups: over all matching narrative lines, the matched
                 groups (joined) must be distinct (e.g. no handle issued twice)
  variants       a list of {harness key: value} overrides: the scenario is also run once per
                 variant (e.g. the same position built from a different deck order); a variant
                 may carry its own expect_same, which replaces the scenario's for that variant
  expect_same    regexes: the narrative lines matching any of them must be identical, in
                 order, between the scenario's run and each variant's run (needs variants)
  format         "bo1" (default) or "bo3": a bo3 scenario plays on through sideboarding and the
                 later games once its script runs out (presets apply to game 1)
  player_a / player_b  the seats' agents once the script runs out (default auto)
  deck_a / deck_b      a deck for a seat with no hand preset
  offer_cancel   true: run with --offer-cancel
  engine_flags   extra engine flags (a list, e.g. ["--trace-visibility"]), passed to the
                 engine through $ROBOMAGE_ENGINE_FLAGS (train/env.py)
  expect_dump    regexes that must each match some knowledge line (below), any frame
  expect_dump_not  regexes that must match no knowledge line
  expect_dump_final  regexes that must each match a knowledge line of a seat's last frame
  expect_dump_final_not  regexes that must match no knowledge line of either seat's last frame

Knowledge lines: with any expect_dump* key the scenario runs with ``--rmstate-dump FILE
--state-dump json`` (a temporary file), and every seat STATE frame's knowledge tables (``k_*``,
docs/knowledge_store.md §6) are rendered one fact per line, prefixed by the seat and the frame::

  <A|B> d=<decision> g<game> <t<turn>|sb> <fact>        (sb: a sideboarding frame, between games)

  meta level=<OFF|EXPERT> exact=<0|1>
  seen <self|opp> <card> game=<n> match=<n>
  pool <self|opp> <hand|library|sideboard|face_down|set_aside>[ <range>] size=<n>: <card> <min>-<max>, ...
  position <self|opp> <at> = <card>                 (the card there is known)
  position <self|opp> <at> one of: <card>, ...      (its candidates)
  known_hand <card> x<copies>
  list <self|opp> <reference list> <consistent|inconsistent>[ unproven][ registered]
  archetype <self|opp> <label> <possible|impossible>[ exact]
  stack_target <pos> <card> <order> = <card>|player <self|opp>|departed <card>|none
                                                     (a stack object's announced targets; a target
                                                      gone from its zone shows its last-known card)

Owners are relative to the seat; a library range or position is ``top<pos>`` (one position) or
``top<pos>+<span>``; a pool's cards (every card with max > 0) and a position's candidates are
sorted by name. A seat's last frame is its last STATE frame in the dump (the decision at
``max_decisions``, or the terminal frame of a finished game). Variants do not check these keys.

The narrative is the engine's game log (the harness's "--- Narrative ---"
blocks) from the first turn banner on (the pregame's opening draws are left
out), so assertions stay independent of the decoded board / menu layout.
``max_decisions`` defaults to 60: keep it just past the asserted event.

Usage:
  train/.venv/bin/python train/test_scenarios.py              # run all
  train/.venv/bin/python train/test_scenarios.py ward*        # a subset (globs)
  train/.venv/bin/python train/test_scenarios.py --show NAME  # print one run's output
  train/.venv/bin/python train/test_scenarios.py --narrative NAME  # just its narrative
  train/.venv/bin/python train/test_scenarios.py --dump NAME  # its knowledge lines
  train/.venv/bin/python train/test_scenarios.py --binary PATH  # another engine build
"""
import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from env import ENGINE_FLAGS_ENV
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
_DUMP_KEYS = {"expect_dump", "expect_dump_not", "expect_dump_final", "expect_dump_final_not"}
_ASSERT_KEYS = {"expect", "expect_not", "expect_order", "expect_count", "expect_menu",
                "expect_menu_not", "expect_result", "expect_unique", "expect_same"} | _DUMP_KEYS
_RUN_KEYS = {"why", "format", "player_a", "player_b", "deck_a", "deck_b", "offer_cancel", "engine_flags",
             "variants"}
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
    cmd = [sys.executable, _HARNESS, "--scenario", path, "--format", scn.get("format", "bo1"),
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


def harness_env(scn, dump_path=None, extra_flags=()):
    """The environment for one scenario's harness process: ours, plus its engine_flags and
    `extra_flags` (and, with `dump_path`, the flags that write the RMSTATE dump the knowledge
    lines are read from)."""
    env = dict(os.environ)
    flags = list(scn.get("engine_flags", [])) + list(extra_flags)
    if dump_path:
        flags += ["--rmstate-dump", dump_path, "--state-dump", "json"]
    if flags:
        env[ENGINE_FLAGS_ENV] = " ".join([env.get(ENGINE_FLAGS_ENV, "")] + flags).strip()
    return env


def wants_dump(scn):
    return bool(_DUMP_KEYS & set(scn))


def _libpos_text(anchor, pos, span):
    if anchor is None:
        return None
    end = "top" if anchor == "TOP" else "bottom"
    return f"{end}{pos}" + (f"+{span}" if span and span > 1 else "")


def _knowledge_facts(tables, names, metagame=None):
    """The knowledge lines (without the seat/frame prefix) of one decoded STATE frame; `metagame`
    is (list names, archetype labels) from the seat's METAGAME frame."""
    lists, labels = metagame or ([], [])
    def card(cid):
        return names.get(cid, f"#{cid}")

    def rel(r):
        return "self" if r == 1 else "opp"

    out = []
    for m in tables.get("k_meta", []):
        out.append(f"meta level={m['level']} exact={int(bool(m['exact']))}")
    for r in tables.get("k_seen", []):
        out.append(f"seen {rel(r['player_rel'])} {card(r['card'])} game={r['game']} match={r['match']}")
    pool_cards = {}
    for r in tables.get("k_pool_card", []):
        pool_cards.setdefault(r["pool"], []).append((card(r["card"]), r["min"], r["max"]))
    pools = tables.get("k_pool", [])
    for i, r in enumerate(pools):
        rng = _libpos_text(r["range_anchor"], r["range_pos"], r["range_span"])
        kind = {"LIBRARY_RANGE": "library"}.get(r["kind"], r["kind"].lower())
        cards = ", ".join(f"{n} {lo}-{hi}" for n, lo, hi in sorted(pool_cards.get(i, [])))
        out.append(f"pool {rel(r['owner_rel'])} {kind}{' ' + rng if rng else ''} size={r['size']}: {cards}")
    for r in tables.get("k_position", []):
        at = _libpos_text(r["at_anchor"], r["at_pos"], r["at_span"])
        if r["card"] is not None:
            out.append(f"position {rel(r['owner_rel'])} {at} = {card(r['card'])}")
        else:
            cands = ", ".join(sorted(n for n, _lo, hi in pool_cards.get(r["pool"], []) if hi > 0))
            out.append(f"position {rel(r['owner_rel'])} {at} one of: {cands}")
    for r in tables.get("k_known_hand", []):
        out.append(f"known_hand {card(r['card'])} x{r['copies']}")
    for r in tables.get("k_template", []):
        name = lists[r["list"]] if r["list"] < len(lists) else f"#{r['list']}"
        out.append(f"list {rel(r['player_rel'])} {name} {'consistent' if r['consistent'] else 'inconsistent'}"
                   f"{'' if r['exact'] else ' unproven'}{' registered' if r['registered'] else ''}")
    for r in tables.get("k_archetype", []):
        label = labels[r["archetype"]] if r["archetype"] < len(labels) else f"#{r['archetype']}"
        out.append(f"archetype {rel(r['player_rel'])} {label} {'possible' if r['possible'] else 'impossible'}"
                   f"{' exact' if r['exact'] else ''}")
    out += _stack_target_facts(tables, card)
    return out


def _stack_target_facts(tables, card):
    """The stack_target lines of one decoded STATE frame (module docstring)."""
    objects = {o["handle"]: o for o in tables.get("object", [])}
    lki = tables.get("lki", [])

    def obj_card(handle):
        o = objects.get(handle)
        return card(o["card"]) if o is not None and o["card"] is not None else "?"
    out = []
    pos_of = {r["handle"]: (r["pos"], r["source_card"]) for r in tables.get("stack", [])}
    for r in tables.get("stack_target", []):
        pos, src = pos_of.get(r["stack"], (None, None))
        if r["kind"] == "PLAYER":
            what = f"player {'self' if r['target_player_rel'] == 1 else 'opp'}"
        elif r["target"]:
            what = obj_card(r["target"])
        elif r["lki"] is not None and r["lki"] < len(lki):
            what = f"departed {card(lki[r['lki']]['card']) if lki[r['lki']]['card'] is not None else '?'}"
        else:
            what = "none"
        out.append(f"stack_target {pos} {card(src) if src is not None else '?'} {r['order']} = {what}")
    return out


def knowledge_frames(jsonl_path):
    """[(seat, lines)] for every seat STATE frame of a --state-dump json file, in order: each
    line is one knowledge fact prefixed by the seat and the frame (see the module docstring)."""
    names = {}
    metagames = {}
    frames = []
    with open(jsonl_path) as fh:
        for raw in fh:
            rec = json.loads(raw)
            viewer = rec.get("viewer")
            if viewer not in ("A", "B"):
                continue
            tables = rec.get("tables", {})
            if rec.get("frame") == "CARDS":
                seat_names = names.setdefault(viewer, {})
                for row in tables.get("card", []):
                    seat_names.setdefault(row["card"], row["name"])
            elif rec.get("frame") == "METAGAME":
                metagames[viewer] = ([r["name"] for r in tables.get("meta_list", [])],
                                     [r["name"] for r in tables.get("meta_archetype", [])])
            elif rec.get("frame") == "STATE":
                head = tables["header"][0]
                turn = tables["turn"][0]["turn"] if tables.get("turn") else 0
                when = "sb" if head.get("sideboarding") else f"t{turn}"
                prefix = f"{viewer} d={head['decision']} g{head['game']} {when} "
                facts = _knowledge_facts(tables, names.get(viewer, {}), metagames.get(viewer))
                frames.append((viewer, [prefix + f for f in facts]))
    return frames


def check_dump(scn, frames):
    """Failure messages for the expect_dump* assertions over `frames` (knowledge_frames)."""
    if not frames:
        return ["no seat STATE frame in the --state-dump json output"]
    fails = []
    lines = [ln for _seat, fl in frames for ln in fl]
    final = []
    for seat in ("A", "B"):
        last = [fl for s, fl in frames if s == seat]
        if last:
            final += last[-1]
    for rx in scn.get("expect_dump", []):
        if not any(re.search(rx, ln) for ln in lines):
            fails.append(f"missing knowledge line /{rx}/")
    for rx in scn.get("expect_dump_not", []):
        hit = next((ln for ln in lines if re.search(rx, ln)), None)
        if hit is not None:
            fails.append(f"forbidden knowledge line /{rx}/: {hit!r}")
    for rx in scn.get("expect_dump_final", []):
        if not any(re.search(rx, ln) for ln in final):
            fails.append(f"missing knowledge line in a seat's last frame /{rx}/")
    for rx in scn.get("expect_dump_final_not", []):
        hit = next((ln for ln in final if re.search(rx, ln)), None)
        if hit is not None:
            fails.append(f"forbidden knowledge line in a seat's last frame /{rx}/: {hit!r}")
    return fails


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
    for rx, n in scn.get("expect_count", {}).items():
        got = sum(1 for ln in narrative if re.search(rx, ln))
        if got != n:
            fails.append(f"narrative line /{rx}/ matched {got} time(s), expected {n}")
    for rx in scn.get("expect_menu", []):
        if not any(re.search(rx, a) for a in actions):
            fails.append(f"no offered action matches /{rx}/")
    for rx in scn.get("expect_menu_not", []):
        hit = next((a for a in actions if re.search(rx, a)), None)
        if hit is not None:
            fails.append(f"forbidden action offered /{rx}/: {hit!r}")
    for rx in scn.get("expect_unique", []):
        seen = {}
        for ln in narrative:
            m = re.search(rx, ln)
            if m is None:
                continue
            key = " ".join(g for g in m.groups() if g is not None)
            if key in seen:
                fails.append(f"/{rx}/ captured {key!r} twice: {seen[key]!r} and {ln!r}")
                break
            seen[key] = ln
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
    if scn.get("format", "bo1") not in ("bo1", "bo3"):
        raise ValueError(f"format must be bo1 or bo3, got {scn['format']!r}")
    if not (_ASSERT_KEYS & set(scn)):
        raise ValueError("no assertion (expect / expect_not / expect_order / expect_count / "
                         "expect_menu / expect_menu_not / expect_result)")
    for key in _ASSERT_KEYS - {"expect_result"}:
        for rx in scn.get(key, []):
            re.compile(rx)
    for rx in scn.get("expect_unique", []):
        if re.compile(rx).groups < 1:
            raise ValueError(f"expect_unique /{rx}/ needs a capture group")
    if scn.get("expect_same") and not scn.get("variants"):
        raise ValueError("expect_same needs variants")
    for variant in scn.get("variants", []):
        bad = set(variant) - (_HARNESS_KEYS - {"name"}) - {"expect_same"}
        if bad:
            raise ValueError(f"variant overrides non-harness key(s) {sorted(bad)}")
        if not variant.get("expect_same", scn.get("expect_same")):
            raise ValueError("a variant needs an expect_same (its own or the scenario's)")
        for rx in variant.get("expect_same", []):
            re.compile(rx)
    return scn


def same_lines(patterns, text):
    """The narrative lines of one run that match any of the expect_same `patterns`."""
    rxs = [re.compile(rx) for rx in patterns]
    return [ln for ln in split_output(text)[0] if any(rx.search(ln) for rx in rxs)]


def run_variant(scn, variant, binary=None):
    """The harness output of `scn` with `variant`'s overrides applied."""
    merged = {k: v for k, v in scn.items() if k != "variants"}
    merged.update({k: v for k, v in variant.items() if k != "expect_same"})
    fd, path = tempfile.mkstemp(suffix=".json", prefix="scenario_variant_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(merged, f)
        r = subprocess.run(harness_cmd(merged, path, binary), cwd=_REPO_ROOT,
                           env=harness_env(merged), capture_output=True, text=True,
                           timeout=_TIMEOUT_S)
        return r.stdout + r.stderr
    finally:
        os.unlink(path)


def check_variants(scn, text, binary=None):
    """Failure messages for the expect_same comparison of each variant against `text`."""
    fails = []
    for i, variant in enumerate(scn.get("variants", [])):
        patterns = variant.get("expect_same", scn.get("expect_same", []))
        base = same_lines(patterns, text)
        if not base:
            fails.append(f"variant {i}: expect_same matched no narrative line")
            continue
        try:
            out = run_variant(scn, variant, binary)
        except subprocess.TimeoutExpired:
            fails.append(f"variant {i} timed out after {_TIMEOUT_S}s")
            continue
        fails += [f"variant {i}: {f}" for f in check_output(
            {k: v for k, v in scn.items() if k not in ("expect_same", "variants")}, out)]
        got = same_lines(patterns, out)
        if got != base:
            k = next((j for j, (a, b) in enumerate(zip(base, got)) if a != b),
                     min(len(base), len(got)))
            fails.append(f"variant {i}: expect_same line {k} differs: "
                         f"{base[k] if k < len(base) else None!r} vs "
                         f"{got[k] if k < len(got) else None!r}")
    return fails


def run_one(path, binary=None, dump=False, extra_flags=()):
    """(name, failure messages, harness output, knowledge frames) for one scenario file, its engine
    run with `extra_flags` as well; the frames (knowledge_frames) are read when the scenario has an
    expect_dump* key or `dump` is set, else they are empty."""
    name = os.path.basename(path)[:-5]
    try:
        scn = load_scenario(path)
    except (ValueError, re.error, json.JSONDecodeError) as e:
        return name, [f"bad scenario file: {e}"], "", []
    dump_dir = tempfile.mkdtemp(prefix=f"scenario_{name}_") if dump or wants_dump(scn) else None
    dump_path = os.path.join(dump_dir, "frames.rms") if dump_dir else None
    frames = []
    try:
        try:
            r = subprocess.run(harness_cmd(scn, path, binary), cwd=_REPO_ROOT,
                               env=harness_env(scn, dump_path, extra_flags),
                               capture_output=True, text=True, timeout=_TIMEOUT_S)
            out = r.stdout + r.stderr
        except subprocess.TimeoutExpired as e:
            return name, [f"timed out after {_TIMEOUT_S}s"], str(e.stdout or ""), []
        fails = check_output(scn, out)
        if dump_path:
            jsonl = dump_path + ".jsonl"
            frames = knowledge_frames(jsonl) if os.path.exists(jsonl) else []
            if wants_dump(scn):
                fails += check_dump(scn, frames)
    finally:
        if dump_dir:
            shutil.rmtree(dump_dir, ignore_errors=True)
    if scn.get("variants"):
        fails += check_variants(scn, out, binary)
    return name, fails, out, frames


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("patterns", nargs="*", help="scenario name globs (default: all)")
    p.add_argument("--show", action="store_true",
                   help="print each selected scenario's full harness output")
    p.add_argument("--narrative", action="store_true",
                   help="print each selected scenario's narrative (what expect* match)")
    p.add_argument("--dump", action="store_true",
                   help="print each selected scenario's knowledge lines (what expect_dump* match)")
    p.add_argument("--binary", default=None, help="engine binary to run against")
    p.add_argument("-j", "--jobs", type=int, default=min(16, os.cpu_count() or 4))
    args = p.parse_args(argv)

    paths = scenario_paths(args.patterns)
    if not paths:
        print("no scenarios matched", file=sys.stderr)
        return 2
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(lambda q: run_one(q, args.binary, args.dump), paths))
    failed = 0
    for name, fails, out, frames in results:
        if args.show:
            print(f"##### {name}\n{out}")
        elif args.narrative:
            print(f"##### {name}")
            print("\n".join(split_output(out)[0]))
        elif args.dump:
            print(f"##### {name}")
            print("\n".join(ln for _seat, lines in frames for ln in lines))
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
