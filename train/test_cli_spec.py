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

The cli_spec parsers are built in-process exactly as the scripts build them
(``apply_to_parser``); the standalone scripts are exercised as subprocesses.

    train/.venv/bin/python train/test_cli_spec.py

Wired into ci_check.py as the 'clispec' tier, so `make check` runs it.
"""

import argparse
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
                           "az-selfplay", "az-eval", "az", "az-league")
} | {("analysis", s) for s in ("report", "interactive", "search")} | {
    ("analysis-tui", "browse"), ("play", "play"), ("harness", "harness")}


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
                ("analysis-tui", "browse")):
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
    ("analysis", "report"): 1, ("analysis", "interactive"): 1,
    ("analysis", "search"): 1, ("analysis-tui", "browse"): 1,
    ("az-inspect", "inspect"): 1, ("harness", "harness"): None,
    ("play", "play"): None,
    ("train", "az-selfplay"): None, ("train", "az-train"): None,
    ("train", "az"): None, ("train", "az-league"): None,
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
    # Removed spellings error with their hint (an exact removed entry also
    # stops argparse prefix-matching --c onto --c-puct).
    for key, argv, needle in (
            (("analysis", "report"), ["--n-games", "5"],
             "--n-games was removed; use --games"),
            (("analysis-tui", "browse"), ["--n-games", "5"],
             "--n-games was removed; use --games"),
            (("analysis", "search"), ["--c", "2.0"],
             "--c was removed; use --c-puct")):
        code, err = parse_error(build(subs[key]), argv)
        check(code == 2 and needle in err,
              f"{'/'.join(key)} {argv[0]} should error with {needle!r} ({err!r})")
    p = build(subs[("analysis", "search")])
    ns = p.parse_args(["--player-a", "gen", "--c-puct", "3.5", "--games", "2"])
    check(ns.c_puct == 3.5 and ns.games == 2, "search --c-puct/--games parse")
    # Standalone inspector: sampling and k-means seeds default to 1.
    import az_inspect
    ns = az_inspect.build_parser().parse_args(["clusters"])
    check(ns.seed == 1 and ns.cluster_seed == 1,
          f"az_inspect clusters seeds {ns.seed}/{ns.cluster_seed} != 1/1")


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
        ("train/test_harness.py",), ("train/fuzz_campaign.py",),
        ("train/train.py", "observe"), ("train/train.py", "league"),
        ("train/play.py",),
    ]
    for prefix in cases:
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
        rc, out = run_script(*argv)
        check(rc == 2 and needle in out,
              f"{' '.join(argv)} should error with {needle!r} (rc={rc}):\n"
              f"{out[-600:]}")

    print("scripts reject --n-games / --c with their hint")
    vocab_cases = [
        (("train/analysis.py", "report", "--player-a", "gen", "--n-games", "3"),
         "--n-games was removed; use --games"),
        (("train/analysis.py", "search", "--player-a", "gen", "--c", "2"),
         "--c was removed; use --c-puct"),
        (("train/eval_search_gate.py", "--checkpoint", "gen", "--deck-a", "d",
          "--c", "2"), "--c was removed; use --c-puct"),
        (("train/bench_engine.py", "--n-games", "3"),
         "--n-games was removed; use --games"),
    ]
    for argv, needle in vocab_cases:
        rc, out = run_script(*argv)
        check(rc == 2 and needle in out,
              f"{' '.join(argv)} should error with {needle!r} (rc={rc}):\n"
              f"{out[-600:]}")
    rc, out = run_script("train/eval_search_gate.py", "--help")
    check(rc == 0 and "--c-puct" in out and not _mentions(out, "--c"),
          f"eval_search_gate --help should list --c-puct only (rc={rc})")


def main():
    test_removed_flags_error()
    test_removed_flags_hidden()
    test_format_default()
    test_scoped_removal()
    test_seat_vocabulary()
    test_play_seats()
    test_removed_env()
    test_count_puct_seed_vocabulary()
    test_harness_parity()
    test_harness_script_then_players()
    test_scripts()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S)")
        return 1
    print("\nclispec: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
