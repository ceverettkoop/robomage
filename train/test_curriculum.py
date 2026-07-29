#!/usr/bin/env python3
"""Regression tests for the curriculum plan schema, argv composition, and the
progress/resume bookkeeping (train/curriculum.py).

The composed argv IS the contract between a plan file and train.py's
subcommands, so every phase kind's composition is asserted here — a renamed or
retyped cli_spec argument that would silently change what a curriculum runs
fails this test instead. Stdlib-only (curriculum.py imports cli_spec +
progress_io, never torch), so it runs anywhere:

    train/.venv/bin/python train/test_curriculum.py

Wired into ci_check.py as the 'curriculum' tier, so `make check` runs it.
"""

import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import curriculum as cur

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print(f"  FAIL: {msg}")
    return cond


def expect_error(fn, needle, msg):
    """The call must raise PlanError whose message mentions ``needle``."""
    try:
        fn()
    except cur.PlanError as exc:
        return check(needle in str(exc),
                     f"{msg}: expected {needle!r} in error, got {exc}")
    FAILURES.append(f"{msg}: expected a PlanError, none raised")
    print(f"  FAIL: {msg}: expected a PlanError, none raised")
    return False


def argv_of(phase, resume=False):
    """Composed argv with the interpreter/script prefix stripped."""
    argv = cur.compose_phase_argv(phase, resume=resume, python="PY",
                                  script="train/train.py")
    check(argv[:2] == ["PY", "train/train.py"], f"argv prefix: {argv[:2]}")
    return argv[2:]


# ── 1. argv composition, one case per phase kind ─────────────────────────────

def test_compose_league():
    print("league phase")
    got = argv_of({"kind": "league",
                   "decks": ["league/ur_delver", "league/bug"],
                   "steps": 2_000_000,
                   "overrides": {"self_play_frac": 0.2, "promote_margin": 0.05,
                                 "bo3": True, "tally": False}})
    check(got == ["league", "--decks", "league/ur_delver,league/bug",
                  "--self-play-frac", "0.2", "--promote-margin", "0.05",
                  "--total-timesteps", "2000000", "--bo3"],
          f"league argv: {got}")
    # 'all' decks means the subcommand's own default roster: no --decks at all.
    got = argv_of({"kind": "league", "decks": "all", "steps": 20000})
    check(got == ["league", "--total-timesteps", "20000"],
          f"league all-decks argv: {got}")
    # A comma string is accepted alongside a list.
    got = argv_of({"kind": "league", "decks": "league/a,league/b"})
    check(got == ["league", "--decks", "league/a,league/b"],
          f"league string-decks argv: {got}")


def test_compose_exploiter():
    print("exploiter phase")
    got = argv_of({"kind": "exploiter", "archetype": "burn", "steps": 500_000,
                   "overrides": {"chunk_steps": 50_000, "fresh": True}})
    check(got == ["exploiter", "--archetype", "burn", "--steps", "500000",
                  "--chunk-steps", "50000", "--fresh"],
          f"exploiter argv: {got}")


def test_compose_az():
    print("az phase")
    got = argv_of({"kind": "az", "decks": ["league/bug"], "games": 40,
                   "overrides": {"sims": 32, "bo1": True}})
    check(got == ["az", "--deck", "league/bug", "--games", "40", "--sims", "32",
                  "--bo1"], f"az argv: {got}")


def test_compose_az_league():
    print("az-league phase")
    got = argv_of({"kind": "az-league", "rotations": 1,
                   "overrides": {"games": 400}})
    check(got == ["az-league", "--rotations", "1", "--games", "400"],
          f"az-league argv: {got}")


def test_compose_az_selfplay():
    print("az-selfplay phase")
    got = argv_of({"kind": "az-selfplay", "decks": ["league/wubg_doomsday"],
                   "games": 32, "overrides": {"expert": True}})
    check(got == ["az-selfplay", "--deck", "league/wubg_doomsday", "--games",
                  "32", "--expert"], f"az-selfplay argv: {got}")


def test_compose_baseline():
    print("baseline phase")
    got = argv_of({"kind": "baseline", "model": "gen", "deck": "league/bug",
                   "games": 100})
    check(got == ["baseline", "gen", "--games", "100", "--deck", "league/bug"],
          f"baseline argv (positional model first): {got}")
    got = argv_of({"kind": "baseline", "games": 50, "overrides": {"all": True}})
    check(got == ["baseline", "--games", "50", "--all"],
          f"baseline --all argv: {got}")


def test_override_beats_alias():
    print("override beats convenience alias")
    got = argv_of({"kind": "league", "steps": 10,
                   "overrides": {"total_timesteps": 99}})
    check(got == ["league", "--total-timesteps", "99"], f"override argv: {got}")


# ── 2. loud failures ─────────────────────────────────────────────────────────

def test_validation_errors():
    print("validation")
    expect_error(lambda: cur.validate_plan({"version": 1, "phases": [
        {"kind": "trainn"}]}), "unknown phase kind", "unknown kind")
    expect_error(lambda: cur.validate_plan({"version": 2, "phases": [
        {"kind": "league"}]}), "unsupported plan version", "version")
    expect_error(lambda: cur.validate_plan({"version": 1, "phases": []}),
                 "non-empty list", "empty phases")
    expect_error(lambda: cur.validate_plan({"version": 1, "phases": [
        {"kind": "league", "overrides": {"self_play_fraq": 0.2}}]}),
        "unknown override", "unknown override key")
    expect_error(lambda: cur.validate_plan({"version": 1, "phases": [
        {"kind": "league", "overrides": {"resume": True}}]}),
        "owned by the curriculum runner", "runner-owned override")
    expect_error(lambda: cur.validate_plan({"version": 1, "phases": [
        {"kind": "league", "stepz": 10}]}), "unknown field", "unknown field")
    expect_error(lambda: cur.validate_plan({"version": 1, "phases": [
        {"kind": "league", "steps": "lots"}]}), "expects an integer", "bad int")
    expect_error(lambda: cur.validate_plan({"version": 1, "phases": [
        {"kind": "exploiter", "archetype": "nonsense"}]}),
        "must be one of", "bad choice")
    expect_error(lambda: cur.validate_plan({"version": 1, "phases": [
        {"kind": "league", "overrides": {"bo3": "yes"}}]}),
        "must be true or false", "flag needs a bool")
    expect_error(lambda: cur.validate_plan({"version": 1, "phases": [
        {"kind": "az", "overrides": {"actor": True, "no_actor": True}}]}),
        "mutually exclusive", "mutex group")
    # A valid plan passes untouched.
    plan = {"version": 1, "name": "ok", "phases": [
        {"kind": "league", "decks": "all", "steps": 10, "label": "warmup"},
        {"kind": "baseline", "model": "gen", "deck": "delver", "games": 4}]}
    check(cur.validate_plan(plan) is plan, "valid plan should validate")


# ── 3. resume argv ───────────────────────────────────────────────────────────

def test_resume_argv():
    print("resume argv")
    league = {"kind": "league", "decks": ["league/a"], "steps": 5}
    check(argv_of(league, resume=True) == ["league", "--resume"],
          "league resume argv should be just --resume")
    shard = {"kind": "league", "steps": 5, "overrides": {"shard": "0/2"}}
    check(argv_of(shard, resume=True) == ["league", "--shard", "0/2", "--resume"],
          "sharded league resume must keep --shard (it selects the sidecar)")
    exp = {"kind": "exploiter", "archetype": "burn", "steps": 5}
    check(argv_of(exp, resume=True) == ["exploiter", "--archetype", "burn",
                                        "--resume"],
          "exploiter resume must keep --archetype (it selects the sidecar)")
    check(argv_of({"kind": "az-league", "rotations": 2}, resume=True)
          == ["az-league", "--resume"], "az-league resume argv")
    # Non-resumable kinds restart from the phase start: full argv, no --resume.
    base = {"kind": "baseline", "model": "gen", "deck": "d", "games": 4}
    check(argv_of(base, resume=True) == argv_of(base),
          "baseline is not resumable — argv must be the full restart command")


# ── 4. plan/progress round trip + hash prefix check ──────────────────────────

def test_plan_io_and_hashes(tmp):
    print("plan io + hash prefix check")
    path = os.path.join(tmp, "toy.plan.json")
    plan = {"version": 1, "name": "toy", "phases": [
        {"kind": "league", "decks": "all", "steps": 20},
        {"kind": "baseline", "model": "gen", "deck": "delver", "games": 4}]}
    cur.save_plan(path, plan)
    check(cur.plan_name(path) == "toy", "plan_name")
    check(cur.progress_path(path) == os.path.join(tmp, "toy.progress.json"),
          "progress path sits next to the plan")
    loaded = cur.load_plan(path)
    check(loaded["phases"] == plan["phases"], "plan round trip")
    check(cur.plan_hash(loaded) == cur.plan_hash(plan), "hash is stable")

    state = cur.new_progress(loaded, path)
    check([p["status"] for p in state["phases"]] == ["pending", "pending"],
          "fresh progress is all pending")
    state["phases"][0].update(status="done", exit_code=0)
    state["phase_index"] = 1

    # Editing a FUTURE phase is allowed...
    edited = json.loads(json.dumps(loaded))
    edited["phases"][1]["games"] = 8
    try:
        cur.check_plan_unchanged(edited, state)
    except cur.PlanError as exc:
        check(False, f"editing a future phase must be allowed: {exc}")
    # ...appending one too...
    appended = json.loads(json.dumps(loaded))
    appended["phases"].append({"kind": "league", "steps": 5})
    try:
        cur.check_plan_unchanged(appended, state)
    except cur.PlanError as exc:
        check(False, f"appending a phase must be allowed: {exc}")
    # ...but editing a COMPLETED phase is refused.
    broken = json.loads(json.dumps(loaded))
    broken["phases"][0]["steps"] = 99
    expect_error(lambda: cur.check_plan_unchanged(broken, state),
                 "phase 1", "edited completed phase must refuse resume")
    # A phase that is in flight is immutable too (its driver sidecar came from
    # the old definition).
    running = json.loads(json.dumps(loaded))
    running["phases"][1]["games"] = 8
    state2 = cur.new_progress(loaded, path)
    state2["phases"][1]["status"] = "running"
    expect_error(lambda: cur.check_plan_unchanged(running, state2),
                 "phase 2", "edited in-flight phase must refuse resume")


# ── 5. runner surfaces that don't spawn training ─────────────────────────────

def test_dry_run_and_status(tmp):
    print("--dry-run / --status")
    path = os.path.join(tmp, "dry.plan.json")
    cur.save_plan(path, {"version": 1, "name": "dry", "phases": [
        {"kind": "league", "decks": ["league/x"], "steps": 20},
        {"kind": "exploiter", "archetype": "burn", "steps": 30},
        {"kind": "baseline", "model": "gen", "deck": "delver", "games": 4}]})
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cur.run_curriculum(path, dry_run=True)
    out = buf.getvalue()
    check(rc == 0, "dry-run exit code")
    for needle in ("league --decks league/x --total-timesteps 20",
                   "exploiter --archetype burn --steps 30",
                   "baseline gen --games 4 --deck delver"):
        check(needle in out, f"dry-run output missing {needle!r}:\n{out}")
    check("train/train.py" in out, "dry-run shows a repo-relative script path")

    # With progress, --dry-run reflects what a --resume would actually do.
    state = cur.new_progress(cur.load_plan(path), path)
    state["phases"][0].update(status="done", exit_code=0)
    state["phases"][1]["status"] = "running"
    state["phase_index"] = 1
    cur.write_progress(path, state)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cur.run_curriculum(path, resume=True, dry_run=True)
    out = buf.getvalue()
    check("skip (done)" in out, f"resume dry-run should skip phase 1:\n{out}")
    check("exploiter --archetype burn --resume" in out,
          f"resume dry-run should relaunch phase 2 with --resume:\n{out}")

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cur.run_curriculum(path, status=True)
    out = buf.getvalue()
    check(rc == 0 and "[1/3] league     done" in out and
          "[2/3] exploiter  running" in out and "[3/3] baseline   pending" in out,
          f"status view:\n{out}")

    # Without --resume, an in-progress curriculum refuses to start over.
    buf, err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf):
        sys.stderr, real = err, sys.stderr
        try:
            rc = cur.run_curriculum(path)
        finally:
            sys.stderr = real
    check(rc == 2 and "--resume" in err.getvalue(),
          f"re-running without --resume must refuse: rc={rc} {err.getvalue()}")


def test_form_schema():
    print("form schema helpers")
    for kind in cur.PHASE_KINDS:
        items = cur.phase_form_args(kind)
        check(items, f"{kind}: form args should be non-empty")
        dests = {getattr(i, "dest", None) for i in items}
        check("resume" not in dests, f"{kind}: --resume must not be offered")
    # make_phase / phase_values round trip through the convenience aliases.
    phase = cur.make_phase("league", {"decks": ["a", "b"], "total_timesteps": 7,
                                      "self_play_frac": 0.3})
    check(phase == {"kind": "league", "decks": ["a", "b"], "steps": 7,
                    "overrides": {"self_play_frac": 0.3}},
          f"make_phase routing: {phase}")
    check(cur.phase_values(phase) == {"decks": ["a", "b"], "total_timesteps": 7,
                                      "self_play_frac": 0.3},
          "phase_values round trip")
    cur.validate_plan({"version": 1, "phases": [phase]})


def main():
    tmp = tempfile.mkdtemp(prefix="curriculum_test_")
    try:
        test_compose_league()
        test_compose_exploiter()
        test_compose_az()
        test_compose_az_league()
        test_compose_az_selfplay()
        test_compose_baseline()
        test_override_beats_alias()
        test_validation_errors()
        test_resume_argv()
        test_plan_io_and_hashes(tmp)
        test_dry_run_and_status(tmp)
        test_form_schema()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S)")
        return 1
    print("\ncurriculum: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
