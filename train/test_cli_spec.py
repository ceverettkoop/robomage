#!/usr/bin/env python3
"""Regression tests for the shared CLI vocabulary in train/cli_spec.py.

* Removed flags (``cli_spec.REMOVED_FLAGS``) error with their replacement hint on
  every parser built from cli_spec and on the standalone parsers, never show in
  ``--help``, and never render in the TUI forms (which are built from the Sub
  items, not the removed table).
* ``--format`` exists on every game-playing subcommand and defaults to bo3.
* Seat vocabulary: single-seat decks are ``--deck-a``/``--deck-b`` and seat
  agents ``--player-a``/``--player-b`` (no ``--opponent``/``--deck`` anywhere;
  a stray positional model names ``--player-a``); play needs exactly one
  ``human`` seat.
* Removed environment variables fail at startup with their hint.
* Every game/match count is ``--games`` and the PUCT constant ``--c-puct``;
  ``--seed`` defaults to 1 on test/eval/inspection tools and to None (random,
  printed) on long training runs (``SEED_DEFAULTS``).
* The GUI launcher dialogs mirror the CLI: every field of
  ``launcher_config``'s Play / Analysis tables is a play.py / browser flag
  with the same dest and default (and every flag is a field or listed as
  CLI-only); the one settings file ignores unknown / ill-typed keys.
* The search-knob fold (``cli_spec.search_knob_pairs`` / play.py's
  ``search_values``) and play's per-board option errors.
* ``analysis.py browse``: the ``--source`` dispatch table (simulate / shard
  directory / .rmtrace), per-source flag applicability, the removed
  ``--shards``, and the retired tui_analysis.py entry point.

The cli_spec parsers are built in-process exactly as the scripts build them
(``apply_to_parser``); the standalone scripts are exercised as subprocesses.
A script whose import chain needs a package this venv lacks (torch and the SB3
stack for train.py, textual for the TUI stubs — the per-push CI image has
neither) has its legs skipped; the nightly full-check job installs them.

    train/.venv/bin/python train/test_cli_spec.py

Wired into ci_check.py as the 'clispec' tier, so `make check` runs it.
"""

import argparse
import importlib.util
import io
import os
import re
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cli_spec
from cli_spec import (ALL_TOOLS, RemovedFlag, apply_to_parser, is_bo3,
                      iter_args)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES = []

# Every (tool key, sub name) that plays games and so must carry --format.
FORMAT_SUBS = {
    ("train", s) for s in ("train", "league", "exploiter", "sweep",
                           "fixed-model", "alternate", "observe", "baseline",
                           "az-selfplay", "az-eval", "az", "az-league",
                           "bench-nenvs")
} | {("analysis", s) for s in ("browse", "report")} | {
    ("play", "play"), ("harness", "harness")}


# Scripts that import a package the per-push CI image omits before they parse.
TORCH_DEPS = ("torch", "stable_baselines3", "sb3_contrib")
SCRIPT_DEPS = {
    "train/train.py": TORCH_DEPS,
    "train/bench_actor.py": TORCH_DEPS,
    "train/tui_az_inspect.py": ("textual",),
    "train/tui_analysis.py": ("textual", "rich"),
}


def need(what, modules):
    """True when every module in ``modules`` is importable; else print a skip."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        print(f"  [skip] {what}: {', '.join(missing)} not installed")
    return not missing


def can_run(script):
    return need(script, SCRIPT_DEPS.get(script, ()))


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print(f"  FAIL: {msg}")
    return cond


def build(sub):
    p = argparse.ArgumentParser(prog=f"{sub.tool} {sub.name}")
    apply_to_parser(p, sub)
    return p


def parse_error(parser, argv):
    """(exit code, stderr) of parsing ``argv``; code None if it parsed."""
    err = io.StringIO()
    try:
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code, err.getvalue()
    return None, err.getvalue()


def all_subs():
    for tool in ALL_TOOLS:
        for sub in tool.subs:
            yield tool, sub


# ── 1. cli_spec-built parsers ────────────────────────────────────────────────

def test_removed_flags_error():
    print("removed flags error with their hint on every cli_spec parser")
    for tool, sub in all_subs():
        p = build(sub)
        flags = cli_spec.removed_flags_for(*sub.scopes)
        check(len({r.flag for r in flags}) == len(flags),
              f"{tool.key}/{sub.name}: one removed entry per flag "
              f"(a scoped entry must shadow the global one)")
        for r in flags:
            code, err = parse_error(p, [r.flag])
            check(code == 2 and cli_spec.removed_flag_message(r) in err,
                  f"{tool.key}/{sub.name}: {r.flag} should error with its hint "
                  f"(code={code}, stderr={err!r})")
            if r.is_positional:
                continue
            # The value form must hit the same error, not "unrecognized".
            code, err = parse_error(p, [f"{r.flag}=x"])
            check(code == 2 and "was removed" in err,
                  f"{tool.key}/{sub.name}: {r.flag}=x should error with the hint")


def _mentions(text, flag):
    """True if ``flag`` appears in ``text`` as a whole option token (so
    '--deck' does not match '--deck-a' or '--decks')."""
    return re.search(r"(?<![\w-])" + re.escape(flag) + r"(?![\w-])", text) is not None


def test_removed_flags_hidden():
    print("removed flags are absent from --help and the TUI form items")
    for tool, sub in all_subs():
        help_text = build(sub).format_help()
        live = {a.name for a in iter_args(sub)}
        for r in cli_spec.removed_flags_for(*sub.scopes):
            if not r.is_positional:
                check(not _mentions(help_text, r.flag),
                      f"{tool.key}/{sub.name}: {r.flag} leaks into --help")
            check(r.flag not in live,
                  f"{tool.key}/{sub.name}: {r.flag} is still a live Arg (the "
                  f"TUI form would render it)")


# Scripts folded into a train.py / analysis.py / az_inspect.py subcommand (each
# leaves a stub that exits with its replacement) and front-end modules that are
# no longer entry points: help text must point at the replacement instead.
_RETIRED_SCRIPTS = ("fuzz_campaign", "bench_engine", "eval_search_gate",
                    "az_embed_viz", "sb_shard_report", "tui_analysis.py",
                    "tui_az_inspect.py")


def _help_texts():
    """(where, text) for every help string cli_spec defines: each Sub's help
    and each Arg's help, across every tool."""
    for tool, sub in all_subs():
        yield f"{tool.key}/{sub.name}", sub.help or ""
        for a in iter_args(sub):
            yield f"{tool.key}/{sub.name} {a.name}", a.help or ""


def test_help_names_nothing_removed():
    print("help text never names a removed flag, subcommand or script")
    unscoped = [r.flag for r in cli_spec.REMOVED_FLAGS
                if not r.scopes and not r.is_positional]
    scripts = {t.key: os.path.basename(t.script) for t in ALL_TOOLS}
    removed_cmds = [f"{scripts.get(tool, tool)} {name}"
                    for tool, name in cli_spec.REMOVED_SUBCOMMANDS]
    for where, text in _help_texts():
        for flag in unscoped:
            check(not _mentions(text, flag),
                  f"{where}: help names the removed {flag}")
        for cmd in removed_cmds:
            check(cmd not in text,
                  f"{where}: help names the removed command `{cmd}`")
        for name in _RETIRED_SCRIPTS:
            check(name not in text,
                  f"{where}: help names the retired script {name}")


def test_seat_vocabulary():
    print("seat decks are --deck-a/--deck-b and seat agents --player-a/--player-b")
    for tool, sub in all_subs():
        where = f"{tool.key}/{sub.name}"
        for a in iter_args(sub):
            check(a.name not in ("--opponent", "--deck", "--human-deck",
                                 "--model-deck"),
                  f"{where}: {a.name} is a live flag (seat decks are "
                  f"--deck-a/--deck-b, pools --decks/--opponents)")
            if a.suggest == "deck" and not a.multi:
                check(a.name in ("--deck-a", "--deck-b"),
                      f"{where}: single-seat deck flag {a.name} should be "
                      f"--deck-a or --deck-b")
            if a.suggest == "agent":
                check(a.name in ("--player-a", "--player-b"),
                      f"{where}: seat agent {a.name} should be --player-a or "
                      f"--player-b")
    subs = {(t.key, s.name): s for t, s in all_subs()}
    # A scoped entry shadows the global one with its own hint.
    check(cli_spec.removed_flag_hint("--deck", "train", "train/az")
          == "--deck was removed; use --decks (the comma-separated focus deck pool)",
          "az's --deck hint should point at --decks")
    check("use --deck-a" in (cli_spec.removed_flag_hint(
        "--deck", "train", "train/observe") or ""),
          "observe's --deck hint should point at --deck-a")
    check("use --player-b" in (cli_spec.removed_flag_hint(
        "--opponent", "analysis", "analysis/report") or ""),
          "analysis --opponent hint should point at --player-b")
    # A stray positional model names its replacement flag.
    for key in (("train", "baseline"), ("analysis", "report"),
                ("analysis", "browse")):
        code, err = parse_error(build(subs[key]), ["gen"])
        check(code == 2 and "positional MODEL argument was removed; use "
              "--player-a" in err,
              f"{'/'.join(key)}: a positional model should name --player-a "
              f"({err!r})")
    code, _ = parse_error(build(subs[("train", "baseline")]), [])
    check(code is None, "baseline parses with no positional")
    p = build(subs[("train", "baseline")])
    check(p.parse_args([]).player_a == cli_spec.DEFAULT_BASELINE_MODEL,
          "baseline --player-a defaults to the baseline model")


def test_play_seats():
    print("play seat resolution: exactly one human")
    rps = cli_spec.resolve_play_seats
    check(rps(None, None) == ("A", None), "default: human on A vs the default opponent")
    check(rps("human", None) == ("A", None), "human A, default opponent on B")
    check(rps(None, "human") == ("B", None), "human B, default opponent on A")
    check(rps("az:gen", None) == ("B", "az:gen"), "an opponent on A puts the human on B")
    check(rps(None, "scripted") == ("A", "scripted"), "an opponent on B puts the human on A")
    check(rps("Human", "gen") == ("A", "gen"), "'human' is case-insensitive")
    for a, b in (("human", "human"), ("gen", "scripted")):
        try:
            rps(a, b)
            check(False, f"({a!r}, {b!r}) should be rejected")
        except ValueError:
            pass


def test_on_the_play():
    print("play --on-the-play: side swap, seeded coin flip, bad values")
    rotp = cli_spec.resolve_on_the_play
    order = cli_spec.order_play_sides
    sides = ("human", "az:gen", "league/bug", "league/ur_delver")
    check(order(*sides, "A") == sides, "a: the sides stay on their seats")
    check(order(*sides, "B") == ("az:gen", "human", "league/ur_delver",
                                 "league/bug"),
          "b: player B's agent AND deck move to engine seat A")
    check(rotp("a") == "A" and rotp("B") == "B" and rotp(None) == "A",
          "a/b resolve directly (case-insensitive); unset is the default a")
    picks = {seed: rotp("random", seed) for seed in range(40)}
    check(all(rotp("random", seed) == side for seed, side in picks.items()),
          "random with a seed is reproducible")
    check(set(picks.values()) == {"A", "B"},
          f"random reaches both sides across seeds: {set(picks.values())}")
    check(rotp("random") in ("A", "B"), "unseeded random picks a side")
    for bad in ("c", "", "first"):
        try:
            rotp(bad)
            check(False, f"--on-the-play {bad!r} should be rejected")
        except ValueError:
            pass
    parser = build(cli_spec.PLAY_TOOL.subs[0])
    check(parser.parse_args([]).on_the_play == cli_spec.DEFAULT_ON_THE_PLAY
          == "a", "--on-the-play defaults to a")
    code, err = parse_error(parser, ["--on-the-play", "c"])
    check(code == 2 and "invalid choice" in err,
          f"--on-the-play c is rejected by the parser ({code}, {err!r})")
    # The human's engine seat and decks after resolving seats + the sides.
    for pa, pb, otp, want in ((None, None, "a", ("A", "x", "y")),
                              (None, None, "b", ("B", "x", "y")),
                              ("az:gen", "human", "b", ("A", "y", "x")),
                              ("human", "scripted", "b", ("B", "x", "y"))):
        human, opp = cli_spec.resolve_play_seats(pa, pb)
        got = cli_spec.seat_play_sides(human, opp, "x", "y", rotp(otp))
        check(got == want, f"{pa}/{pb} --on-the-play {otp}: (human seat, "
                           f"human deck, opp deck) {got} != {want}")


def test_format_default():
    print("--format defaults to bo3 on every game-playing subcommand")
    for tool, sub in all_subs():
        has = any(a.name == "--format" for a in iter_args(sub))
        want = (tool.key, sub.name) in FORMAT_SUBS
        check(has == want, f"{tool.key}/{sub.name}: --format present={has}, "
                           f"expected {want}")
        if not has:
            continue
        p = build(sub)
        check(p.get_default("format") == "bo3",
              f"{tool.key}/{sub.name}: --format default "
              f"{p.get_default('format')!r} != 'bo3'")
        check(is_bo3(argparse.Namespace(format=p.get_default("format"))),
              f"{tool.key}/{sub.name}: default format should read as bo3")
        code, err = parse_error(p, ["--format", "bo2"])
        check(code == 2 and "invalid choice" in err,
              f"{tool.key}/{sub.name}: --format bo2 should be rejected")
    check(is_bo3(argparse.Namespace()), "a namespace without format is bo3")
    check(not is_bo3({"format": "bo1"}), "an opts dict with format bo1 is bo1")


# --seed default per (tool, sub): 1 on deterministic test/eval/inspection
# tools, None (randomly drawn at launch and printed) on long training runs and
# interactive play. The harness's None means "1, or the scenario's seed".
SEED_DEFAULTS = {
    ("train", "observe"): 1, ("train", "baseline"): 1, ("train", "az-eval"): 1,
    ("analysis", "report"): 1, ("analysis", "browse"): 1,
    ("harness", "harness"): None,
    ("play", "play"): None,
    ("train", "az-selfplay"): None, ("train", "az-train"): None,
    ("train", "az"): None, ("train", "az-league"): None,
    ("train", "bench-actor"): 1, ("train", "bench-workers"): 1,
    # az_inspect: every view that samples shards (or seeds k-means / t-SNE).
    **{("az-inspect", s): 1 for s in (
        "tui", "overview", "neighbors", "structure", "clusters", "project",
        "occur", "buckets", "calib", "divergence", "state", "blocks",
        "readout", "swap", "sweep")},
}

# Count flags other than --games, each naming a DIFFERENT count.
_OTHER_COUNT_FLAGS = {"--eval-games", "--expert-games"}


def test_count_puct_seed_vocabulary():
    print("--games / --c-puct / --seed vocabulary and seed defaults")
    subs = {(t.key, s.name): s for t, s in all_subs()}
    for (key, sub) in subs.items():
        where = "/".join(key)
        names = {a.name for a in iter_args(sub)}
        for n in names:
            if re.search(r"games|matches", n) and n != "--games":
                check(n in _OTHER_COUNT_FLAGS,
                      f"{where}: game-count flag {n} should be --games")
        check("--c" not in names and "--n-games" not in names,
              f"{where}: --c / --n-games are live (use --c-puct / --games)")
        if "--seed" in names:
            check(key in SEED_DEFAULTS,
                  f"{where}: --seed default is not pinned in SEED_DEFAULTS")
            if key in SEED_DEFAULTS:
                got = build(sub).get_default("seed")
                check(got == SEED_DEFAULTS[key],
                      f"{where}: --seed default {got!r} != "
                      f"{SEED_DEFAULTS[key]!r}")
        else:
            check(key not in SEED_DEFAULTS, f"{where}: expected a --seed flag")
    # Removed spellings error with their hint.
    for key, argv, needle in (
            (("analysis", "report"), ["--n-games", "5"],
             "--n-games was removed; use --games"),
            (("analysis", "browse"), ["--n-games", "5"],
             "--n-games was removed; use --games")):
        code, err = parse_error(build(subs[key]), argv)
        check(code == 2 and needle in err,
              f"{'/'.join(key)} {argv[0]} should error with {needle!r} ({err!r})")
    p = build(subs[("analysis", "report")])
    ns = p.parse_args(["--player-a", "az:gen", "--sims", "8", "--games", "2",
                       "--workers", "2"])
    check(ns.sims == 8 and ns.games == 2 and ns.workers == 2,
          "report --sims/--games/--workers parse")
    # The inspector's one --seed drives sampling and k-means alike.
    import az_inspect
    ns = az_inspect.build_parser().parse_args(["clusters"])
    check(ns.seed == 1 and not hasattr(ns, "cluster_seed"),
          f"az_inspect clusters --seed defaults to 1 ({ns})")


def test_scoped_removal():
    print("scoped removals register only in their scope")
    saved = cli_spec.REMOVED_FLAGS
    cli_spec.REMOVED_FLAGS = saved + (
        RemovedFlag("--zz-old", "use --zz-new", scopes=("train/observe",)),
        RemovedFlag("--yy-old", "use --yy-new", scopes=("play",)))
    try:
        subs = {(t.key, s.name): s for t, s in all_subs()}
        code, err = parse_error(build(subs[("train", "observe")]), ["--zz-old"])
        check(code == 2 and "--zz-old was removed; use --zz-new" in err,
              f"sub-scoped removal should fire on train/observe: {err!r}")
        code, err = parse_error(build(subs[("train", "league")]), ["--zz-old"])
        check(code == 2 and "unrecognized arguments" in err,
              f"sub-scoped removal must not register on train/league: {err!r}")
        code, err = parse_error(build(subs[("play", "play")]), ["--yy-old"])
        check(code == 2 and "--yy-old was removed; use --yy-new" in err,
              f"tool-scoped removal should fire on play: {err!r}")
        check(cli_spec.removed_flag_hint("--zz-old", "train", "train/observe")
              == "--zz-old was removed; use --zz-new",
              "removed_flag_hint should find a scoped entry")
        check(cli_spec.removed_flag_hint("--zz-old", "train", "train/az") is None,
              "removed_flag_hint must respect scope")
    finally:
        cli_spec.REMOVED_FLAGS = saved


def test_removed_subcommands():
    print("removed subcommands error with their hint and stay out of --help")
    for (tool_key, name), _hint in cli_spec.REMOVED_SUBCOMMANDS.items():
        tool = next(t for t in ALL_TOOLS if t.key == tool_key)
        check(all(s.name != name for s in tool.subs),
              f"{tool_key}: removed subcommand {name!r} is still a live Sub")
        p = argparse.ArgumentParser(prog=tool_key)
        sp = p.add_subparsers(dest="command", required=True)
        for s in tool.subs:
            apply_to_parser(sp.add_parser(s.name, help=s.help), s)
        cli_spec.add_removed_subcommands(sp, tool_key)
        msg = cli_spec.removed_subcommand_message(tool_key, name)
        for argv in ([name], [name, "--player-a", "gen"], [name, "-h"],
                     [name, "x", "--y"]):
            code, err = parse_error(p, argv)
            check(code == 2 and msg in err,
                  f"{tool_key} {' '.join(argv)} should error with {msg!r} "
                  f"(code={code}, stderr={err!r})")
        out = io.StringIO()
        with redirect_stdout(out):
            try:
                p.parse_args(["--help"])
            except SystemExit:
                pass
        listed = re.findall(r"\{([^}]*)\}", out.getvalue())
        check(listed and all(name not in c.split(",") for c in listed)
              # a subcommand's own help row is indented exactly four
              and not re.search(rf"^ {{4}}{re.escape(name)}\s", out.getvalue(),
                                re.M),
              f"{tool_key} --help should not list removed {name!r}")
    check(cli_spec.removed_subcommand_message("analysis", "browse") is None,
          "a live subcommand has no removal message")
    rc, out = run_script("train/analysis.py", "interactive", "--player-a", "gen")
    check(rc == 2 and "`interactive` was removed; use `analysis.py browse`" in out,
          f"analysis.py interactive should error naming browse (rc={rc}):\n"
          f"{out[-600:]}")
    rc, out = run_script("train/analysis.py", "search", "--player-a", "gen",
                         "--workers", "4")
    check(rc == 2 and "`search` was removed; use `analysis.py report" in out,
          f"analysis.py search should error naming report (rc={rc}):\n"
          f"{out[-600:]}")


def test_removed_env():
    print("removed env vars fail at startup")
    saved = dict(cli_spec.REMOVED_ENV_VARS)
    cli_spec.REMOVED_ENV_VARS["ROBOMAGE_ZZ_TEST"] = "use --zz"
    try:
        try:
            cli_spec.check_removed_env(environ={"ROBOMAGE_ZZ_TEST": "1"})
            check(False, "a set removed env var should exit")
        except SystemExit as exc:
            check("ROBOMAGE_ZZ_TEST was removed; use --zz" in str(exc.code),
                  f"env error should name the replacement: {exc.code!r}")
        cli_spec.check_removed_env(environ={})       # unset: no error
    finally:
        cli_spec.REMOVED_ENV_VARS.clear()
        cli_spec.REMOVED_ENV_VARS.update(saved)


def test_removed_env_real():
    print("the removed smoke / PopArt / head env vars error naming the flag")
    for name, want in (("ROBOMAGE_POPART", "--popart / --no-popart"),
                       ("ROBOMAGE_PER_ACTION_HEAD", "--stock-head"),
                       ("ROBOMAGE_GUI_SMOKE", "ROBOMAGE_SMOKE=play:N"),
                       ("ROBOMAGE_ANALYSIS_SMOKE", "ROBOMAGE_SMOKE=play:N,analysis"),
                       ("ROBOMAGE_GUI_SESSION_SMOKE", "ROBOMAGE_SMOKE=session"),
                       ("ROBOMAGE_GUI_TRACE_SMOKE", "ROBOMAGE_SMOKE=trace"),
                       ("ROBOMAGE_BROWSER_SMOKE", "ROBOMAGE_SMOKE=browser"),
                       ("ROBOMAGE_BROWSER_SMOKE_SHARDS", "ROBOMAGE_SMOKE=browser:DIR"),
                       ("ROBOMAGE_TREE_SMOKE", "ROBOMAGE_SMOKE=tree:DIR")):
        try:
            cli_spec.smoke_legs(environ={name: "1"})
            check(False, f"{name} set should exit")
        except SystemExit as exc:
            check(f"{name} was removed; use" in str(exc.code)
                  and want in str(exc.code),
                  f"{name} error should name {want!r}: {exc.code!r}")
    if not can_run("train/train.py"):
        return
    env = dict(os.environ, ROBOMAGE_POPART="1")
    r = subprocess.run([sys.executable, "train/train.py", "league", "--help"],
                       cwd=REPO, capture_output=True, text=True, timeout=300,
                       env=env)
    out = r.stdout + r.stderr
    check(r.returncode == 2 and "ROBOMAGE_POPART was removed" in out,
          f"train.py should refuse ROBOMAGE_POPART (rc={r.returncode}):\n"
          f"{out[-600:]}")


def test_smoke_legs():
    print("ROBOMAGE_SMOKE parses into one leg map")
    parse = cli_spec.parse_smoke
    check(parse("") == {} and cli_spec.smoke_legs(environ={}) == {},
          "unset smoke = no legs")
    check(parse("play:8,analysis") == {"play": 8, "analysis": True},
          f"play:8,analysis ({parse('play:8,analysis')})")
    check(parse("play") == {"play": 1}, "bare play = 1 decision")
    check(parse("session") == {"session": True}, "session")
    check(parse("browser:/tmp/a:b, tree:/x") == {"browser": "/tmp/a:b",
                                                  "tree": "/x"},
          "browser/tree carry their dir (split on the first colon)")
    check(parse("browser") == {"browser": True}, "bare browser")
    for bad in ("bogus", "play:0", "play:x", "session:1", "analysis:on"):
        try:
            parse(bad)
            check(False, f"smoke {bad!r} should be rejected")
        except ValueError:
            pass
    try:
        cli_spec.smoke_legs(environ={"ROBOMAGE_SMOKE": "nope"})
        check(False, "a malformed ROBOMAGE_SMOKE should exit")
    except SystemExit as exc:
        check("ROBOMAGE_SMOKE" in str(exc.code) and "nope" in str(exc.code),
              f"malformed smoke error: {exc.code!r}")
    check(cli_spec.smoke_leg("play", environ={"ROBOMAGE_SMOKE": "play:3"}) == 3
          and cli_spec.smoke_leg("tree", environ={}) is None, "smoke_leg")


def test_popart_default():
    print("PopArt is on by default for PPO training; --no-popart / --stock-head")
    import curriculum
    subs = {s.name: s for s in cli_spec.TRAIN_TOOL.subs}
    for name in ("train", "sweep", "fixed-model", "alternate", "league",
                 "exploiter", "bench-nenvs"):
        sub = subs[name]
        req = []
        for a in iter_args(sub):
            if a.required or a.is_positional:
                val = (a.choices[0] if a.choices
                       else "1" if a.kind in ("int", "float") else "x")
                req += [val] if a.is_positional else [a.name, val]
        p = build(sub)
        on = cli_spec.resolve_popart(p.parse_args(req))
        off = cli_spec.resolve_popart(p.parse_args(req + ["--no-popart"]))
        check(on is True and off is False,
              f"{name}: popart default on, --no-popart off ({on}, {off})")
        if any(a.dest == "stock_head" for a in iter_args(sub)):
            check(cli_spec.resolve_popart(p.parse_args(req + ["--stock-head"]))
                  is False, f"{name}: --stock-head implies --no-popart")
            code, err = parse_error(p, req + ["--popart", "--stock-head"])
            check(code is None, "argparse accepts the pair itself")
            try:
                cli_spec.resolve_popart(p.parse_args(
                    req + ["--popart", "--stock-head"]))
                check(False, f"{name}: --popart --stock-head should error")
            except SystemExit as exc:
                check("--stock-head" in str(exc.code),
                      f"{name}: popart/stock-head error ({exc.code!r})")
    arg = curriculum.phase_args("league")["popart"]
    check(curriculum._format_value(arg, False, "t") == "--no-popart"
          and curriculum._format_value(arg, True, "t") == "--popart",
          "a curriculum plan's popart true/false composes --popart/--no-popart")


def _visible_options(parser):
    """Every help-visible option string of ``parser`` except -h/--help."""
    out = set()
    for act in parser._actions:
        if act.help == argparse.SUPPRESS:
            continue
        out.update(s for s in act.option_strings if s not in ("-h", "--help"))
    return out


def test_harness_parity():
    print("test_harness's parser is HARNESS_TOOL (every option from cli_spec)")
    import test_harness
    sub = cli_spec.HARNESS_TOOL.subs[0]
    parser = test_harness.build_parser()
    spec = {a.name for a in iter_args(sub)}
    got = _visible_options(parser)
    check(got == spec, f"harness options differ from HARNESS_TOOL: only in the "
                       f"parser {sorted(got - spec)}, only in cli_spec "
                       f"{sorted(spec - got)}")
    for name in ("--play", "--actions", "--player-a", "--player-b",
                 "--graveyard-a", "--exile-b", "--sideboard-a", "--life-a",
                 "--merge-sideboard", "--coverage-json", "--log-decisions"):
        check(name in spec, f"HARNESS_TOOL lacks {name}")
    ns = parser.parse_args([])
    check(ns.player_a == ns.player_b == cli_spec.HARNESS_DEFAULT_PLAYER,
          "harness seats default to the auto player")
    for flag, needle in (("--scripted", "use --player-a scripted --player-b scripted"),
                         ("--scripted-spec", "use --player-a / --player-b"),
                         ("--interactive", "use --player-a human")):
        code, err = parse_error(parser, [flag])
        check(code == 2 and f"{flag} was removed; {needle}" in err,
              f"harness {flag} should error with {needle!r} ({err!r})")
    subs = {(t.key, s.name): s for t, s in all_subs()}
    for flag in ("--play-a", "--play-b"):
        code, err = parse_error(build(subs[("train", "observe")]), [flag, "pass"])
        check(code == 2 and f"{flag} was removed; use --player-" in err
              and "play:" in err,
              f"observe {flag} should point at a play: player spec ({err!r})")


def test_harness_script_then_players():
    print("harness script precedence: the script first, then the seat players")
    import test_harness
    from env import _SELF_IS_A_IDX, STATE_SIZE
    from opponents import ActionListController, PlayController

    class Fixed:
        def __init__(self, idx):
            self.idx, self.calls = idx, 0

        def choose(self, obs, num_choices, action_masks=None, decoded_actions=None):
            self.calls += 1
            return self.idx

    def obs_for(seat):
        obs = [0.0] * STATE_SIZE
        obs[_SELF_IS_A_IDX] = 1.0 if seat == "A" else 0.0
        return obs

    pass_menu = [{"index": 0, "category": 0, "card": None, "controller": None,
                  "description": "Pass priority"},
                 {"index": 1, "category": 9, "card": "Mountain",
                  "controller": "own", "description": "Play Mountain"}]
    discard_menu = [{"index": 0, "category": 30, "card": "Island",
                     "controller": "own", "description": "Discard Island"},
                    {"index": 1, "category": 30, "card": "Forest",
                     "controller": "own", "description": "Discard Forest"}]

    pa, pb = Fixed(1), Fixed(1)
    play = PlayController("A:land:Mountain", players=(pa, pb))
    # B lacks the next (A-keyed) spec: passes a priority window, but a
    # mandatory choice of B's goes to B's player.
    check(play.choose(obs_for("B"), 2, decoded_actions=pass_menu) == 0,
          "the seat without the keyed spec passes a priority window")
    check(play.choose(obs_for("B"), 2, decoded_actions=discard_menu) == 1
          and pb.calls == 1, "that seat's mandatory choice goes to its player")
    check(play.choose(obs_for("A"), 2, decoded_actions=pass_menu) == 1
          and pa.calls == 0, "the keyed spec is played by the script")
    check(play.choose(obs_for("A"), 2, decoded_actions=pass_menu) == 1
          and pa.calls == 1, "once the script runs out, A's player decides")
    check(play.resolved == [0, 1, 1, 1], f"resolved {play.resolved}")

    acts = ActionListController([0], players=(pa, pb))
    check(acts.choose(obs_for("B"), 2) == 0 and pb.calls == 1,
          "an --actions script decides first")
    check(acts.choose(obs_for("B"), 2) == 1 and pb.calls == 2,
          "then the seat's player decides")

    ca, cb, la, lb = test_harness._build_controllers("auto", "auto", None, None)
    check(ca is cb and la == lb == "Auto", "default seats share one auto player")
    ca, cb, la, lb = test_harness._build_controllers("auto", "scripted", "pass", None)
    check(ca is cb and isinstance(ca, PlayController)
          and (la, lb) == ("Play", "Play+Scripted"),
          f"a --play script drives both seats ({la!r}, {lb!r})")


# ── 2. the scripts themselves (standalone parsers + dispatch) ───────────────

def run_script(*argv):
    r = subprocess.run([sys.executable, *argv], cwd=REPO, capture_output=True,
                       text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


def test_scripts():
    print("scripts reject --bo1/--bo3 with the hint and hide them from --help")
    cases = [
        ("train/test_harness.py",),
        ("train/train.py", "observe"), ("train/train.py", "league"),
        ("train/play.py",),
    ]
    for prefix in cases:
        if not can_run(prefix[0]):
            continue
        name = " ".join(prefix)
        rc, out = run_script(*prefix, "--bo3")
        check(rc == 2 and "--bo3 was removed; use --format bo3" in out,
              f"{name} --bo3 should error with the hint (rc={rc}):\n{out[-600:]}")
        rc, out = run_script(*prefix, "--bo1")
        check(rc == 2 and "--bo1 was removed; use --format bo1" in out,
              f"{name} --bo1 should error with the hint (rc={rc}):\n{out[-600:]}")
        rc, out = run_script(*prefix, "--help")
        check(rc == 0 and "--format" in out and "--bo1" not in out
              and "--bo3" not in out,
              f"{name} --help should list --format and not --bo1/--bo3 "
              f"(rc={rc})")

    print("scripts reject the pre-seat-vocabulary spellings with their hint")
    seat_cases = [
        (("train/train.py", "observe", "--deck", "delver"),
         "--deck was removed; use --deck-a"),
        (("train/train.py", "observe", "--opponent", "mav"),
         "--opponent was removed; use --deck-b"),
        (("train/train.py", "--deck-b", "mav", "--opponent", "mav"),
         "--opponent was removed; use --deck-b"),
        (("train/train.py", "baseline", "gen"),
         "positional MODEL argument was removed; use --player-a"),
        (("train/train.py", "az", "--deck", "league/bug"),
         "--deck was removed; use --decks"),
        (("train/train.py", "az-train", "--deck", "delver"),
         "--deck was removed; az-train"),
        (("train/play.py", "--human-deck", "delver"),
         "--human-deck was removed; use --deck-a (with --player-a human)"),
        (("train/play.py", "--model-deck", "mav"),
         "--model-deck was removed; use --deck-b"),
        (("train/play.py", "--model", "gen"), "--model was removed; use --player-b"),
        (("train/play.py", "--scripted"), "--scripted was removed; use --player-b scripted"),
        (("train/play.py", "--player", "B"), "--player was removed"),
        (("train/test_harness.py", "--scripted"),
         "--scripted was removed; use --player-a scripted --player-b scripted"),
        (("train/test_harness.py", "--interactive"),
         "--interactive was removed; use --player-a human"),
        (("train/train.py", "observe", "--play-a", "pass"),
         "--play-a was removed; use --player-a \"play:"),
        (("train/play.py", "--deck-a", "d", "--deck-b", "d", "--player-a", "gen",
          "--player-b", "scripted"), "exactly one of --player-a / --player-b"),
        (("train/play.py", "--deck-a", "d", "--deck-b", "d", "--player-a", "human",
          "--player-b", "human"), "both 'human'"),
    ]
    for argv, needle in seat_cases:
        if not can_run(argv[0]):
            continue
        rc, out = run_script(*argv)
        check(rc == 2 and needle in out,
              f"{' '.join(argv)} should error with {needle!r} (rc={rc}):\n"
              f"{out[-600:]}")

    print("scripts reject --n-games with its hint")
    vocab_cases = [
        (("train/analysis.py", "report", "--player-a", "gen", "--n-games", "3"),
         "--n-games was removed; use --games"),
        (("train/train.py", "observe", "--n-games", "3"),
         "--n-games was removed; use --games"),
    ]
    for argv, needle in vocab_cases:
        if not can_run(argv[0]):
            continue
        rc, out = run_script(*argv)
        check(rc == 2 and needle in out,
              f"{' '.join(argv)} should error with {needle!r} (rc={rc}):\n"
              f"{out[-600:]}")


# ── GUI launcher mirrors the CLI ──────────────────────────────────────────────

def test_launcher_mirror():
    print("GUI launcher fields mirror the CLI flags (dest + default)")
    import tempfile
    import launcher_config as lc
    from cli_spec import arg_default, sub_defaults
    for section, sub, cli_only in (
            (lc.PLAY_SECTION, cli_spec.PLAY_TOOL.subs[0], lc.PLAY_CLI_ONLY),
            (lc.ANALYSIS_SECTION, cli_spec.ANALYSIS_BROWSE_SUB,
             lc.ANALYSIS_CLI_ONLY)):
        flags = {a.dest: a for a in iter_args(sub)}
        fields = set(lc.section_args(section))
        check(fields <= set(flags),
              f"{section}: fields that are no flag: {fields - set(flags)}")
        check(not fields & set(cli_only),
              f"{section}: CLI-only dests that are fields: "
              f"{fields & set(cli_only)}")
        check(fields | set(cli_only) == set(flags),
              f"{section}: flags neither a field nor CLI-only: "
              f"{set(flags) - fields - set(cli_only)}")
        defaults = lc.section_defaults(section)
        for dest in fields:
            want = arg_default(flags[dest])
            if section == lc.PLAY_SECTION and dest in ("player_a", "player_b"):
                human, opp = cli_spec.resolve_play_seats(
                    flags["player_a"].default, flags["player_b"].default)
                opp = opp or cli_spec.DEFAULT_PLAY_OPPONENT
                want = ({"A": cli_spec.HUMAN_SPEC, "B": opp} if human == "A"
                        else {"A": opp, "B": cli_spec.HUMAN_SPEC})[dest[-1].upper()]
            check(defaults[dest] == want,
                  f"{section}.{dest}: launcher default {defaults[dest]!r} != "
                  f"CLI default {want!r}")
        check(sub_defaults(sub)["format"] == "bo3", f"{section}: bo3 default")

    print("the launcher settings file: sections, junk keys ignored")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "gui_launcher.json")
        check(lc.load_section(lc.PLAY_SECTION, path)
              == lc.section_defaults(lc.PLAY_SECTION), "no file -> defaults")
        values = dict(lc.section_defaults(lc.PLAY_SECTION), worlds=16,
                      match_clock=0.0, sims=200, analysis=False,
                      search_device="cuda", player_b="scripted")
        lc.save_section(lc.PLAY_SECTION, values, path)
        lc.save_section(lc.ANALYSIS_SECTION,
                        dict(lc.section_defaults(lc.ANALYSIS_SECTION),
                             games=3), path)
        check(lc.load_section(lc.PLAY_SECTION, path) == values,
              "play section round-trips")
        check(lc.load_section(lc.ANALYSIS_SECTION, path)["games"] == 3,
              "analysis section kept alongside play")
        import json
        with open(path, "w") as f:
            json.dump({"play": {"worlds": "eight", "search_device": "tpu",
                                "analysis_enabled": True, "deck_a": 5,
                                "player_b": None, "analysis_cap": None,
                                "sims": None, "paced": True},
                       "stale_section": {"x": 1}}, f)
        got = lc.load_section(lc.PLAY_SECTION, path)
        base = lc.section_defaults(lc.PLAY_SECTION)
        check(got == dict(base, paced=True),
              f"ill-typed / unknown / None-on-set keys ignored: {got}")
        with open(path, "w") as f:
            f.write("not json")
        check(lc.load_section(lc.PLAY_SECTION, path) == base,
              "an unreadable file -> defaults")


# ── Search knobs and play's boards ────────────────────────────────────────────

def test_search_knobs():
    print("search-knob fold: pairs, auto procs, paced, clock 0")
    pairs = cli_spec.search_knob_pairs
    check(pairs({"sims": 64, "search_procs": 2}) == [("sims", 64), ("procs", 2)],
          "set knobs fold in SEARCH_KNOB_KEYS order")
    got = dict(pairs({"worlds": 1}))
    check(got.get("procs") == 1, f"auto procs is capped at the worlds: {got}")
    check("procs" not in dict(pairs({}, auto_procs=False)),
          "no auto procs when the tool has no --search-procs")
    check(dict(pairs({"search_xw": False, "search_procs": 1}))["xw"] == 0
          and "xw" not in dict(pairs({"search_xw": True, "search_procs": 1})),
          "only --no-search-xw folds")
    check(dict(pairs({"match_clock": 1500.0, "paced": None,
                      "search_procs": 1}))["paced"] == 1,
          "unset paced turns on with a clock")
    got = dict(pairs({"match_clock": 0.0, "paced": None, "search_procs": 1}))
    check("clock" not in got and got["paced"] == 0,
          f"--match-clock 0 is no clock (and no pacing): {got}")
    check(cli_spec.apply_search_knobs("scripted", {"sims": 5}) == "scripted",
          "a non-search spec passes through")
    check(cli_spec.apply_search_knobs("az:gen?sims=2", {"sims": 5,
                                                         "search_procs": 1})
          == "az:gen?sims=2&sims=5&procs=1", "knobs append last")
    check(cli_spec.spec_query_keys("az:gen?Sims=2&worlds=3") == {"sims", "worlds"},
          "spec_query_keys")
    check(cli_spec.is_search_spec("mcts:uniform")
          and not cli_spec.is_search_spec("azraw:gen"), "is_search_spec")

    print("play.py: explicit flags, spec knobs, and defaults")
    import play
    parser = play.build_parser()

    def values(argv, spec):
        args = parser.parse_args(argv)
        return play.search_values(parser, args,
                                  cli_spec.explicit_dests(parser, argv), spec)

    v = values([], "az:gen")
    check(v["worlds"] == cli_spec.DEFAULT_PLAY_WORLDS
          and v["match_clock"] == cli_spec.DEFAULT_PLAY_MATCH_CLOCK,
          f"play defaults fill a bare search spec: {v}")
    v = values([], "az:gen?worlds=2&sims=32")
    check("worlds" not in v and v["match_clock"] is None,
          f"a spec's own knobs beat the defaults; its sims drop the clock: {v}")
    v = values(["--worlds", "3"], "az:gen?worlds=2")
    check(v["worlds"] == 3, "an explicit flag beats the spec's knob")
    v = values(["--sims", "50"], "az:gen")
    check(v["sims"] == 50 and v["match_clock"] is None,
          "an explicit --sims drops the default clock")
    check(values([], "scripted") == {}, "no knobs for a non-search opponent")
    code, err = parse_error_fn(lambda: values(["--sims", "5"], "scripted"))
    check(code == 2 and "only apply to a search opponent" in err,
          f"an explicit knob on a non-search opponent errors ({err!r})")
    code, err = parse_error_fn(lambda: values(
        ["--sims", "5", "--match-clock", "60"], "az:gen"))
    check(code == 2 and "mutually exclusive" in err, "--sims + --match-clock")
    check(cli_spec.explicit_dests(parser, ["--no-paced", "--deck-a", "x"])
          == {"paced", "deck_a"}, "explicit_dests")


def parse_error_fn(fn):
    """(exit code, stderr) of calling ``fn``; code None if it returned."""
    err = io.StringIO()
    try:
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            fn()
    except SystemExit as exc:
        return exc.code, err.getvalue()
    return None, err.getvalue()


def test_play_boards():
    print("play.py rejects options its board cannot honour")
    base = ("train/play.py", "--player-b", "scripted")
    cases = [
        (("train/play.py", "--gui"), "--gui was removed; use --board gui"),
        (("train/play.py", "--tui"), "--tui was removed; use --board tui"),
        ((*base, "--board", "text", "--record-shards"),
         "--record-shards need the gui or tui board"),
        ((*base, "--board", "text", "--human-clock", "60"),
         "--human-clock need the gui or tui board"),
        ((*base, "--board", "tui", "--analysis"), "--analysis need the gui board"),
        ((*base, "--board", "tui", "--analysis-worlds", "2"),
         "--analysis-worlds need the gui board"),
        ((*base, "--board", "tui", "--sims", "4"),
         "only apply to a search opponent"),
    ]
    try:
        import PySide6  # noqa: F401  (without it --board gui falls back to tui)
        cases.append((("train/play.py", "--board", "gui", "--format", "bo1"),
                      "--format need a session"))
    except ImportError:
        pass
    for argv, needle in cases:
        rc, out = run_script(*argv)
        check(rc == 2 and needle in out,
              f"{' '.join(argv)} should error with {needle!r} (rc={rc}):\n"
              f"{out[-600:]}")


def test_observe_fuzz_bench():
    print("observe carries the fuzz/bench flags; the old scripts point at it")
    sub = next(s for _, s in all_subs() if (s.tool, s.name) == ("train", "observe"))
    args = build(sub).parse_args(
        ["--player-a", "explore", "--player-b", "explore:patient", "--out", "f.txt",
         "--max-decisions", "50", "--quiet", "--timing"])
    check((args.player_a, args.player_b, args.out, args.max_decisions,
           args.quiet, args.timing)
          == ("explore", "explore:patient", "f.txt", 50, True, True),
          f"observe fuzz/bench flags parse ({args})")
    args = build(sub).parse_args([])
    check((args.out, args.max_decisions, args.quiet, args.timing, args.seed)
          == (None, None, False, False, 1),
          f"observe fuzz/bench flag defaults ({args})")
    if can_run("train/train.py"):
        rc, out = run_script("train/train.py", "observe", "--verbose", "--quiet")
        check(rc == 2 and "--verbose and --quiet are mutually exclusive" in out,
              f"observe --verbose --quiet should error (rc={rc}):\n{out[-600:]}")
    for script, needle in (
            ("train/fuzz_campaign.py", "use `train.py observe --player-a explore"),
            ("train/bench_engine.py", "--quiet --timing")):
        rc, out = run_script(script, "--games", "3")
        check(rc == 1 and "was removed" in out and needle in out,
              f"{script} should exit 1 naming observe (rc={rc}):\n{out[-600:]}")


def test_baseline_players():
    print("baseline seats any agent pair; eval_search_gate points at it")
    import az_baseline
    sub = next(s for _, s in all_subs() if (s.tool, s.name) == ("train", "baseline"))
    args = build(sub).parse_args([])
    check((args.player_a, args.player_b)
          == (cli_spec.DEFAULT_BASELINE_MODEL, cli_spec.DEFAULT_BASELINE_OPPONENT),
          f"baseline seat defaults ({args.player_a}, {args.player_b})")
    for spec, oracle in (("scripted", True), ("scripted:hard", True),
                         ("hard", True), ("scripted:easy", False),
                         ("gen", False), ("mcts:gen", False)):
        check(az_baseline.is_oracle_opponent(spec) == oracle,
              f"is_oracle_opponent({spec!r}) should be {oracle}")
    rc, out = run_script("train/eval_search_gate.py", "--games", "3")
    check(rc == 1 and "was removed" in out
          and "train.py baseline --player-a mcts:gen --player-b gen" in out,
          f"eval_search_gate.py should exit 1 naming baseline (rc={rc}):\n"
          f"{out[-600:]}")
    if not can_run("train/train.py"):
        return
    args = build(sub).parse_args(["--sims", "16", "--worlds", "2"])
    kind, _ckpt, _base, params = az_baseline.classify_model("mcts:gen?worlds=3")
    budget = az_baseline.seat_budget(args, "mcts:gen?worlds=3", params)
    spec = az_baseline.python_spec_with_budget("mcts:gen?worlds=3", kind, budget)
    from opponents import _parse_spec_query
    knobs = _parse_spec_query(spec)[1]
    check(kind == "python" and knobs["sims"] == "16" and knobs["worlds"] == "3",
          f"mcts: seat takes the flag budget, its own knobs winning ({spec})")
    check(az_baseline.python_spec_with_budget("gen", "python", budget) == "gen",
          "a non-search seat takes no budget knobs")
    rc, out = run_script("train/train.py", "baseline", "--actor",
                         "--player-b", "gen")
    check(rc != 0 and "needs the Python backend" in out,
          f"baseline --actor with a non-scripted --player-b should refuse "
          f"(rc={rc}):\n{out[-600:]}")
    from train import _baseline_units
    units = _baseline_units([("d", "o")], 5, 4)
    check([u[2:] for u in units] == [(0, 2), (2, 1), (3, 1), (4, 1)],
          f"one matchup over 4 workers splits into contiguous chunks ({units})")
    check(len(_baseline_units([("d", "o")] * 3, 5, 1)) == 3,
          "workers=1 plays one unit per matchup")


def test_benches():
    print("train.py bench-* carry the bench scripts in the az-* vocabulary")
    subs = {s.name: s for _, s in all_subs() if s.tool == "train"}
    args = build(subs["bench-actor"]).parse_args([])
    check((args.player_b, args.no_cross_world, args.eval_server,
           args.no_eval_server, args.actor_device, args.batch, args.sims)
          == (cli_spec.BENCH_PLAYER_SELF, False, False, False, "cpu", "1",
              cli_spec.DEFAULT_AZ_FAST_SIMS),
          f"bench-actor defaults: self-play, cross-world on, eval server AUTO "
          f"({args})")
    args = build(subs["bench-workers"]).parse_args([])
    check((args.sims, args.worlds, args.c_puct, args.td_n,
           args.exhaustive_repeats, args.scripted_cells, args.batches,
           args.eval_games)
          == (cli_spec.DEFAULT_AZ_SIMS, cli_spec.DEFAULT_AZ_WORLDS,
              cli_spec.DEFAULT_AZ_C_PUCT, cli_spec.DEFAULT_AZ_TD_N,
              cli_spec.DEFAULT_AZ_EXHAUSTIVE_REPEATS,
              cli_spec.DEFAULT_AZ_SCRIPTED_CELLS,
              cli_spec.DEFAULT_AZ_CYCLE_BATCHES, cli_spec.DEFAULT_AZ_EVAL_GAMES),
          f"bench-workers takes the az-* defaults ({args})")
    train_help = next(a.help for a in iter_args(subs["bench-workers"])
                      if a.name == "--train")
    check(f"--epoch-frac {cli_spec.DEFAULT_AZ_EPOCH_FRAC}" in train_help
          and f"--q-mix {cli_spec.DEFAULT_AZ_Q_MIX}" in train_help,
          f"bench-workers --train help quotes the live az-train defaults "
          f"({train_help!r})")
    args = build(subs["bench-nenvs"]).parse_args(["--n-envs", "4,8", "--popart"])
    check((args.n_envs, args.popart, args.format)
          == ("4,8", True, cli_spec.DEFAULT_FORMAT),
          f"bench-nenvs parses the training flags ({args})")
    check(cli_spec.parse_int_list("32, 48,64", "--workers") == [32, 48, 64],
          "parse_int_list")
    for bad in ("", "4,x", "0"):
        try:
            cli_spec.parse_int_list(bad, "--workers")
            check(False, f"parse_int_list({bad!r}) should exit")
        except SystemExit as exc:
            check("--workers" in str(exc), f"parse_int_list({bad!r}): {exc}")
    for name, argv, needle in (
            ("bench-actor", ["--scripted"], "--scripted was removed; use --player-b scripted"),
            ("bench-actor", ["--device", "cuda"], "--device was removed; use --actor-device"),
            ("bench-actor", ["--cross"], "--cross was removed; the cross-world leg"),
            ("bench-workers", ["--counts", "4"], "--counts was removed; use --workers"),
            ("bench-workers", ["--repeats", "1"], "--repeats was removed; use --exhaustive-repeats"),
            ("bench-workers", ["--train-window", "5"], "--train-window was removed; use --window"),
            ("bench-workers", ["--train-batches", "5"], "--train-batches was removed; use --batches"),
            ("bench-workers", ["--train-deck", "x"], "--train-deck was removed; use --deck-a"),
            ("bench-nenvs", ["--envs", "4"], "--envs was removed; use --n-envs")):
        code, err = parse_error(build(subs[name]), argv)
        check(code == 2 and needle in err,
              f"{name} {argv[0]} should error with {needle!r} ({err!r})")
    for script, cmd in (("train/bench_actor.py", "bench-actor"),
                        ("train/bench_az_workers.py", "bench-workers"),
                        ("train/bench_nenvs.py", "bench-nenvs")):
        if not can_run(script):
            continue
        rc, out = run_script(script, "--games", "3")
        check(rc == 1 and "was removed" in out and f"train.py {cmd}" in out,
              f"{script} should exit 1 naming {cmd} (rc={rc}):\n{out[-600:]}")
    if can_run("train/train.py"):
        rc, out = run_script("train/train.py", "bench-workers", "--workers", "2",
                             "--decks", "delver", "--dry-run")
        check(rc == 0 and "leg 1: workers=2" in out and "dry run" in out,
              f"bench-workers --dry-run plans a leg (rc={rc}):\n{out[-600:]}")


def test_az_inspect_entry():
    print("az_inspect is one command; the folded scripts point at it")
    for script, needle in (
            ("train/tui_az_inspect.py", "use `az_inspect.py tui`"),
            ("train/az_embed_viz.py", "use `az_inspect.py project --chart`"),
            ("train/sb_shard_report.py", "use `az_inspect.py sbreport`")):
        if not can_run(script):
            continue
        rc, out = run_script(script, "--help")
        check(rc == 1 and "was removed" in out and needle in out,
              f"{script} should exit 1 naming az_inspect (rc={rc}):\n{out[-600:]}")
    rc, out = run_script("train/az_inspect.py", "tui", "--with-shards")
    check(rc == 2 and "--with-shards was removed; use --shards DIR" in out,
          f"az_inspect tui --with-shards should error (rc={rc}):\n{out[-600:]}")


def test_browse_source():
    print("analysis.py browse: one --source, per-source flags, tui_analysis stub")
    import tempfile
    kind = cli_spec.browse_source_kind
    with tempfile.TemporaryDirectory() as tmp:
        shard_dir = os.path.join(tmp, "rec")
        empty_dir = os.path.join(tmp, "empty")
        os.makedirs(shard_dir)
        os.makedirs(empty_dir)
        open(os.path.join(shard_dir, "shard_000.npz"), "wb").close()
        trace = os.path.join(tmp, "s.rmtrace")
        open(trace, "wb").close()
        for source, want in ((None, cli_spec.BROWSE_KIND_SIMULATE),
                             ("simulate", cli_spec.BROWSE_KIND_SIMULATE),
                             (shard_dir, cli_spec.BROWSE_KIND_SHARDS),
                             (trace, cli_spec.BROWSE_KIND_TRACE)):
            check(kind(source) == want, f"browse_source_kind({source!r}) != {want}")
        for bad, needle in ((empty_dir, "holds no shard_*.npz"),
                            (os.path.join(tmp, "gone.rmtrace"), "no such"),
                            ("gen", "the model to inspect is --player-a")):
            try:
                kind(bad)
                check(False, f"browse_source_kind({bad!r}) should raise")
            except ValueError as exc:
                check(needle in str(exc), f"{bad!r}: {exc}")
        # Every flag with a source restriction is a browse flag, and each
        # source kind accepts the flags that describe it.
        sub = cli_spec.ANALYSIS_BROWSE_SUB
        dests = {a.dest for a in iter_args(sub)}
        check(set(cli_spec.BROWSE_SOURCE_DESTS) <= dests,
              f"BROWSE_SOURCE_DESTS names non-flags "
              f"{set(cli_spec.BROWSE_SOURCE_DESTS) - dests}")
        bad = cli_spec.browse_inapplicable_dests
        check(bad(cli_spec.BROWSE_KIND_SIMULATE,
                  {"player_a", "player_b", "deck_a", "sims", "games", "board"})
              == [], "simulate takes the sim / search flags")
        check(bad(cli_spec.BROWSE_KIND_SHARDS,
                  {"player_a", "games", "seat", "no_net", "deck_b", "sims"})
              == ["deck_b", "sims"], "shards reject the sim flags")
        check(bad(cli_spec.BROWSE_KIND_TRACE, {"player_a", "games", "seat"})
              == ["games", "seat"], "a trace takes only --player-a")
        p = build(sub)
        ns = p.parse_args([])
        check((ns.source, ns.board, ns.player_a)
              == ("simulate", cli_spec.BOARD_TUI, "gen"),
              f"browse defaults: simulate on the tui board ({ns})")
        code, err = parse_error(p, ["--shards", shard_dir])
        check(code == 2 and "--shards was removed; use --source DIR" in err,
              f"browse --shards should error ({err!r})")
        code, err = parse_error(p, ["--board", "text"])
        check(code == 2 and "invalid choice" in err, "browse has no text board")
        for argv, needle in (
                (("--source", shard_dir, "--deck-b", "mav"),
                 "--deck-b do not apply to a shards --source"),
                (("--source", trace, "--games", "3"),
                 "--games do not apply to a trace --source"),
                (("--seat", "B"), "--seat do not apply to a simulate --source"),
                (("--source", "gen"), "the model to inspect is --player-a")):
            rc, out = run_script("train/analysis.py", "browse", *argv)
            check(rc == 2 and needle in out,
                  f"browse {' '.join(argv)} should error with {needle!r} "
                  f"(rc={rc}):\n{out[-600:]}")
    if can_run("train/tui_analysis.py"):
        rc, out = run_script("train/tui_analysis.py", "--player-a", "gen")
        check(rc == 1 and "was removed" in out and "use `analysis.py browse`" in out,
              f"tui_analysis.py should exit 1 naming analysis.py browse (rc={rc}):\n"
              f"{out[-600:]}")
    check(not any(t.key == "analysis-tui" for t in ALL_TOOLS),
          "the analysis-tui tool is folded into analysis browse")


def main():
    test_removed_flags_error()
    test_removed_flags_hidden()
    test_help_names_nothing_removed()
    test_format_default()
    test_scoped_removal()
    test_seat_vocabulary()
    test_play_seats()
    test_on_the_play()
    test_removed_env()
    test_removed_env_real()
    test_smoke_legs()
    test_popart_default()
    test_removed_subcommands()
    test_count_puct_seed_vocabulary()
    test_harness_parity()
    test_harness_script_then_players()
    test_launcher_mirror()
    test_search_knobs()
    test_scripts()
    test_observe_fuzz_bench()
    test_baseline_players()
    test_benches()
    test_play_boards()
    test_az_inspect_entry()
    test_browse_source()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S)")
        return 1
    print("\nclispec: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
