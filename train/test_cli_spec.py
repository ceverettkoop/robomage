#!/usr/bin/env python3
"""Regression tests for the shared CLI vocabulary in train/cli_spec.py.

* Removed flags (``cli_spec.REMOVED_FLAGS``) error with their replacement hint on
  every parser built from cli_spec and on the standalone parsers, never show in
  ``--help``, and never render in the TUI forms (which are built from the Sub
  items, not the removed table).
* ``--format`` exists on every game-playing subcommand and defaults to bo3.
* Removed environment variables fail at startup with their hint.

The cli_spec parsers are built in-process exactly as the scripts build them
(``apply_to_parser``); the standalone scripts are exercised as subprocesses.

    train/.venv/bin/python train/test_cli_spec.py

Wired into ci_check.py as the 'clispec' tier, so `make check` runs it.
"""

import argparse
import io
import os
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
        for r in cli_spec.removed_flags_for(*sub.scopes):
            code, err = parse_error(p, [r.flag])
            check(code == 2 and f"{r.flag} was removed; {r.hint}" in err,
                  f"{tool.key}/{sub.name}: {r.flag} should error with its hint "
                  f"(code={code}, stderr={err!r})")
            # The value form must hit the same error, not "unrecognized".
            code, err = parse_error(p, [f"{r.flag}=x"])
            check(code == 2 and "was removed" in err,
                  f"{tool.key}/{sub.name}: {r.flag}=x should error with the hint")


def test_removed_flags_hidden():
    print("removed flags are absent from --help and the TUI form items")
    for tool, sub in all_subs():
        help_text = build(sub).format_help()
        live = {a.name for a in iter_args(sub)}
        for r in cli_spec.REMOVED_FLAGS:
            check(r.flag not in help_text,
                  f"{tool.key}/{sub.name}: {r.flag} leaks into --help")
            check(r.flag not in live,
                  f"{tool.key}/{sub.name}: {r.flag} is still a live Arg (the "
                  f"TUI form would render it)")


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


def main():
    test_removed_flags_error()
    test_removed_flags_hidden()
    test_format_default()
    test_scoped_removal()
    test_removed_env()
    test_scripts()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S)")
        return 1
    print("\nclispec: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
