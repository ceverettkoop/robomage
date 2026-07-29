"""Multi-phase training curricula: plan files, progress sidecar, and the runner
behind ``train.py curriculum``.

A *curriculum* is an ordered list of training phases — "league for 2M steps,
then a burn exploiter for 500k, then league again, then an AZ league rotation,
then a baseline report" — written down once as JSON and executed (and resumed)
by one command instead of a hand-driven sequence of ad-hoc sessions.

Two files per curriculum, both under ``train/checkpoints/curricula/``:

* ``<name>.plan.json`` — the plan (what to run). Versioned::

      {"version": 1, "name": "q3_archetypes",
       "phases": [
         {"kind": "league", "decks": ["league/ur_delver", "league/bug"],
          "steps": 2000000, "overrides": {"self_play_frac": 0.2}},
         {"kind": "exploiter", "archetype": "burn", "steps": 500000},
         {"kind": "az-league", "rotations": 1, "overrides": {"games": 400}},
         {"kind": "baseline", "model": "gen", "deck": "league/bug", "games": 100}
       ]}

  A phase ``kind`` maps 1:1 onto a ``train.py`` subcommand. The handful of
  top-level convenience fields (``PHASE_FIELDS``) are aliases for that
  subcommand's most-used arguments; everything else goes in ``overrides``, whose
  keys are that subcommand's ``cli_spec`` argument *dests*. The cli_spec IS the
  schema — there is no second source of truth here, and an unknown kind or
  override key fails loudly at load time rather than at phase launch.

* ``<name>.progress.json`` — the progress sidecar (how far it got), written
  through the same atomic writer as every other resumable driver
  (``progress_io``). It records a per-phase hash of the plan, so a resume can
  verify that nothing *already executed* was edited while still allowing the
  user to rewrite the phases still ahead.

Phases run as SUBPROCESSES (``sys.executable train/train.py <kind> …``): memory
isolation between PPO and AZ phases, crash containment, and each phase reuses
its own ``--resume`` machinery unchanged.

This module is stdlib-only (cli_spec + progress_io) so the Textual launcher can
import it for the plan builder / status view without pulling in torch.
"""

import hashlib
import json
import os
import subprocess
import sys
import time

from cli_spec import (TRAIN_TOOL, MutexGroup, REPO_ROOT, iter_args, shard_tag)
from progress_io import read_progress_state, write_progress_state

TRAIN_SCRIPT = os.path.join(REPO_ROOT, "train", "train.py")
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "train", "checkpoints")
CURRICULA_DIR = os.path.join(CHECKPOINT_DIR, "curricula")

PLAN_VERSION = 1
PROGRESS_VERSION = 1
PLAN_SUFFIX = ".plan.json"
PROGRESS_SUFFIX = ".progress.json"

# Phase kinds == train.py subcommands. Adding one is a single entry here plus
# its convenience-field aliases below; its flags come from cli_spec.
PHASE_KINDS = ("league", "exploiter", "az", "az-league", "az-selfplay", "baseline")

# Kinds whose subcommand has its own progress sidecar and --resume: a phase of
# one of these that was interrupted mid-way is RELAUNCHED with --resume (it then
# restores its own budget/rotation from its sidecar). The others restart from the
# phase start — safe, since checkpoints accumulate.
RESUMABLE_KINDS = frozenset({"league", "exploiter", "az-league"})

# On a --resume relaunch the driver restores its whole configuration from its own
# sidecar and ignores the other flags, so the command is composed from just these
# dests (the ones that still select WHICH run is being resumed) plus --resume.
RESUME_SELECTOR_DESTS = {"exploiter": ("archetype",), "league": ("shard",)}

# Top-level plan fields that alias a subcommand argument, per kind: the common
# knobs read straight off a phase line (the plan doc's shape) instead of being
# buried in "overrides". Everything else is an override keyed by arg dest.
PHASE_FIELDS = {
    "league":    {"decks": "decks", "steps": "total_timesteps"},
    "exploiter": {"archetype": "archetype", "steps": "steps", "decks": "decks"},
    "az":        {"decks": "deck", "games": "games"},
    "az-league": {"decks": "decks", "rotations": "rotations", "games": "games"},
    "az-selfplay": {"decks": "deck", "games": "games"},
    "baseline":  {"model": "model", "deck": "deck", "games": "games"},
}

# The runner owns --resume (it decides per phase whether a relaunch resumes), so
# a plan may never set it.
RUNNER_OWNED_DESTS = frozenset({"resume"})

# A deck field set to this means "the subcommand's own default roster" (every
# deck in decks/league/) — the flag is simply omitted.
ALL_DECKS = "all"
_DECK_DESTS = frozenset({"decks", "deck", "opponents", "train_decks"})

# Optional free-form annotations a phase / plan may carry (ignored by the runner).
_PHASE_ANNOTATIONS = ("label", "note")
_PLAN_ANNOTATIONS = ("description", "note")

PHASE_STATUSES = ("pending", "running", "done", "failed")

_LABEL = "curriculum"


class PlanError(ValueError):
    """A curriculum plan (or its progress sidecar) is malformed."""


# ── plan schema, derived from cli_spec ────────────────────────────────────────

def phase_sub(kind: str):
    """The ``cli_spec`` Sub backing a phase kind."""
    if kind not in PHASE_KINDS:
        raise PlanError(f"unknown phase kind {kind!r}; valid kinds: "
                        f"{', '.join(PHASE_KINDS)}")
    for s in TRAIN_TOOL.subs:
        if s.name == kind:
            return s
    raise PlanError(f"phase kind {kind!r} has no train.py subcommand "
                    f"(cli_spec.TRAIN_TOOL out of sync with PHASE_KINDS)")


def phase_args(kind: str) -> dict:
    """``{dest: Arg}`` for a phase kind, in cli_spec declaration order."""
    return {a.dest: a for a in iter_args(phase_sub(kind))}


def phase_form_args(kind: str) -> list:
    """The Sub items a plan may set for this kind, for form generation.

    Mirrors ``Sub.items`` (MutexGroups intact, so a form builder renders them as
    one picker) minus the dests the runner owns."""
    out = []
    for item in phase_sub(kind).items:
        if isinstance(item, MutexGroup):
            args = [a for a in item.args if a.dest not in RUNNER_OWNED_DESTS]
            if args:
                out.append(MutexGroup(args, item.required, item.label, item.help))
        elif item.dest not in RUNNER_OWNED_DESTS:
            out.append(item)
    return out


def phase_field_dest(kind: str, field: str):
    """The arg dest a phase's top-level convenience field aliases, or None."""
    return PHASE_FIELDS.get(kind, {}).get(field)


def dest_phase_field(kind: str, dest: str):
    """Inverse of ``phase_field_dest``: the convenience field for an arg dest."""
    for field, d in PHASE_FIELDS.get(kind, {}).items():
        if d == dest:
            return field
    return None


def make_phase(kind: str, values: dict) -> dict:
    """Build a phase dict from ``{dest: value}``, routing the convenience fields
    to the top level and everything else into ``overrides``.

    The inverse of ``phase_values``; used by the TUI plan builder so a
    hand-written plan and a screen-built one have the same shape."""
    phase = {"kind": kind}
    overrides = {}
    for dest, val in values.items():
        field = dest_phase_field(kind, dest)
        if field is not None:
            phase[field] = val
        else:
            overrides[dest] = val
    if overrides:
        phase["overrides"] = overrides
    return phase


def phase_values(phase: dict) -> dict:
    """``{dest: value}`` for a phase: convenience fields resolved to their arg
    dests, then ``overrides`` applied on top (an override wins over the alias)."""
    kind = phase["kind"]
    values = {}
    for field, dest in PHASE_FIELDS.get(kind, {}).items():
        if field in phase:
            values[dest] = phase[field]
    values.update(phase.get("overrides") or {})
    return values


def coerce_value(arg, value):
    """Best-effort cast of a form/text value to the arg's JSON type.

    Plans are hand-readable JSON, so a numeric argument should land as a number
    rather than a quoted string. An uncastable value is passed through unchanged
    so ``validate_plan`` reports it with its real message."""
    if value is None or isinstance(value, (bool, list, tuple)):
        return value
    try:
        if arg.kind == "int":
            return int(str(value).strip())
        if arg.kind == "float":
            return float(str(value).strip())
    except ValueError:
        return value
    return value


def describe_phase(phase: dict) -> str:
    """One-line summary of a phase for a list view / log line."""
    kind = phase.get("kind", "?")
    bits = []
    for field in PHASE_FIELDS.get(kind, {}):
        if field in phase:
            val = phase[field]
            if isinstance(val, (list, tuple)):
                val = ",".join(str(v) for v in val)
            bits.append(f"{field}={val}")
    n = len(phase.get("overrides") or {})
    if n:
        bits.append(f"+{n} override{'s' if n > 1 else ''}")
    return f"{kind}  " + "  ".join(bits) if bits else kind


def validate_plan(plan, source: str = "<plan>") -> dict:
    """Validate a decoded plan against the cli_spec schema; return it.

    Fails loudly (PlanError) on: a wrong/missing version, an empty or non-list
    phases array, an unknown phase kind, an unknown top-level phase field, an
    unknown override key, a runner-owned key, a value of the wrong type for its
    arg, or two flags of one mutually-exclusive group set at once."""
    if not isinstance(plan, dict):
        raise PlanError(f"{source}: plan must be a JSON object")
    version = plan.get("version")
    if version != PLAN_VERSION:
        raise PlanError(f"{source}: unsupported plan version {version!r} "
                        f"(this build reads version {PLAN_VERSION})")
    allowed = {"version", "name", "phases", *_PLAN_ANNOTATIONS}
    unknown = sorted(set(plan) - allowed)
    if unknown:
        raise PlanError(f"{source}: unknown plan field(s) {unknown}; "
                        f"valid: {sorted(allowed)}")
    phases = plan.get("phases")
    if not isinstance(phases, list) or not phases:
        raise PlanError(f"{source}: 'phases' must be a non-empty list")
    for i, phase in enumerate(phases):
        _validate_phase(phase, f"{source}: phase {i + 1}")
    return plan


def _validate_phase(phase, where: str) -> None:
    if not isinstance(phase, dict):
        raise PlanError(f"{where}: each phase must be a JSON object")
    kind = phase.get("kind")
    if not isinstance(kind, str):
        raise PlanError(f"{where}: missing 'kind' (one of {', '.join(PHASE_KINDS)})")
    try:
        args = phase_args(kind)
    except PlanError as exc:
        raise PlanError(f"{where}: {exc}")
    allowed = {"kind", "overrides", *_PHASE_ANNOTATIONS,
               *PHASE_FIELDS.get(kind, {})}
    unknown = sorted(set(phase) - allowed)
    if unknown:
        raise PlanError(
            f"{where} ({kind}): unknown field(s) {unknown}. Valid top-level "
            f"fields: {sorted(allowed)} — every other knob goes in 'overrides' "
            f"keyed by the subcommand's argument dest.")
    overrides = phase.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise PlanError(f"{where} ({kind}): 'overrides' must be a JSON object")
    for dest in overrides:
        if dest in RUNNER_OWNED_DESTS:
            raise PlanError(
                f"{where} ({kind}): override {dest!r} is owned by the curriculum "
                f"runner (it decides per phase whether a relaunch resumes) and "
                f"may not be set in a plan")
        if dest not in args:
            near = sorted(d for d in args if d.startswith(dest[:3]))
            raise PlanError(
                f"{where} ({kind}): unknown override {dest!r} — "
                f"'train.py {kind}' has no such argument"
                + (f" (did you mean: {', '.join(near)}?)" if near else ""))
    # Every value must survive argv formatting (type / choice checks).
    for dest, val in phase_values(phase).items():
        _format_value(args[dest], val, f"{where} ({kind})")
    # Mutually-exclusive flag pairs (e.g. az's --actor / --no-actor).
    values = phase_values(phase)
    for item in phase_sub(kind).items:
        if isinstance(item, MutexGroup):
            on = [a.name for a in item.args if values.get(a.dest)]
            if len(on) > 1:
                raise PlanError(f"{where} ({kind}): {' and '.join(on)} are "
                                f"mutually exclusive")


def _format_value(arg, value, where: str):
    """Normalize one plan value into the argv text for ``arg``.

    Returns None when the value means "leave the flag off" (False for a flag,
    an empty/blank value, or the ``all`` sentinel on a deck-roster arg)."""
    if value is None:
        return None
    if arg.kind == "flag":
        if not isinstance(value, bool):
            raise PlanError(f"{where}: {arg.name} is a flag — its value must be "
                            f"true or false (got {value!r})")
        return arg.name if value else None
    if isinstance(value, bool):
        raise PlanError(f"{where}: {arg.name} takes a value, not a boolean")
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value if str(v).strip()]
        text = ",".join(parts)
    else:
        text = str(value).strip()
    if not text:
        return None
    if arg.dest in _DECK_DESTS and text == ALL_DECKS:
        return None          # the subcommand's own default roster
    if arg.kind == "int":
        try:
            text = str(int(text))
        except ValueError:
            raise PlanError(f"{where}: {arg.name} expects an integer (got {value!r})")
    elif arg.kind == "float":
        try:
            text = str(float(text))
        except ValueError:
            raise PlanError(f"{where}: {arg.name} expects a number (got {value!r})")
    elif arg.kind == "choice" and text not in arg.choices:
        raise PlanError(f"{where}: {arg.name} must be one of "
                        f"{', '.join(arg.choices)} (got {value!r})")
    return text


# ── argv composition ─────────────────────────────────────────────────────────

def compose_phase_argv(phase: dict, resume: bool = False,
                       python: str = None, script: str = None) -> list:
    """The subprocess argv for one phase.

    Arguments are emitted in cli_spec declaration order, so the composed command
    is stable and diffable. With ``resume`` (only meaningful for a
    ``RESUMABLE_KINDS`` phase that was interrupted mid-way) the driver restores
    its configuration from its own sidecar and ignores the rest of the command
    line, so only the run-selecting flags are emitted alongside ``--resume``."""
    kind = phase["kind"]
    args = phase_args(kind)
    argv = [python or sys.executable, script or TRAIN_SCRIPT, kind]
    values = phase_values(phase)
    where = f"phase ({kind})"
    if resume and kind in RESUMABLE_KINDS:
        for dest in RESUME_SELECTOR_DESTS.get(kind, ()):
            if dest in values:
                text = _format_value(args[dest], values[dest], where)
                if text is not None:
                    argv += [args[dest].name, text]
        argv.append("--resume")
        return argv
    for dest, arg in args.items():
        if dest not in values:
            continue
        text = _format_value(arg, values[dest], where)
        if text is None:
            continue
        if arg.kind == "flag":
            argv.append(text)
        elif arg.is_positional:
            argv.append(text)
        else:
            argv += [arg.name, text]
    return argv


def display_argv(argv: list) -> str:
    """A composed argv as a copy-pasteable repo-relative command string."""
    out = ["python"]
    for a in argv[1:]:
        out.append(os.path.relpath(a, REPO_ROOT) if a.startswith(REPO_ROOT) else a)
    return " ".join(out)


# ── plan / progress files ────────────────────────────────────────────────────

def plan_path(name_or_path: str) -> str:
    """Resolve a plan reference to a path.

    A bare name ('q3') resolves to ``checkpoints/curricula/q3.plan.json``; a
    path (or anything ending in .json) is taken as-is."""
    if os.sep in name_or_path or name_or_path.endswith(".json"):
        return os.path.abspath(name_or_path)
    return os.path.join(CURRICULA_DIR, name_or_path + PLAN_SUFFIX)


def plan_name(path: str) -> str:
    """The curriculum name behind a plan path (its filename stem)."""
    base = os.path.basename(path)
    if base.endswith(PLAN_SUFFIX):
        return base[:-len(PLAN_SUFFIX)]
    return os.path.splitext(base)[0]


def progress_path(path: str) -> str:
    """The progress sidecar path next to a plan file."""
    return os.path.join(os.path.dirname(os.path.abspath(path)),
                        plan_name(path) + PROGRESS_SUFFIX)


def list_plans(directory: str = None) -> list:
    """Curriculum names with a plan file in ``directory`` (curricula/ default)."""
    directory = directory or CURRICULA_DIR
    if not os.path.isdir(directory):
        return []
    return sorted(f[:-len(PLAN_SUFFIX)] for f in os.listdir(directory)
                  if f.endswith(PLAN_SUFFIX))


def load_plan(path: str) -> dict:
    """Read and validate a plan file."""
    try:
        with open(path) as fh:
            plan = json.load(fh)
    except FileNotFoundError:
        raise PlanError(f"no curriculum plan at {path}")
    except (OSError, ValueError) as exc:
        raise PlanError(f"{path}: cannot read plan ({exc})")
    plan.setdefault("name", plan_name(path))
    return validate_plan(plan, path)


def save_plan(path: str, plan: dict) -> str:
    """Validate and write a plan file (creating its directory)."""
    plan = dict(plan)
    plan.setdefault("version", PLAN_VERSION)
    plan.setdefault("name", plan_name(path))
    validate_plan(plan, path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(plan, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def phase_hash(phase: dict) -> str:
    """Content hash of one phase definition (canonical JSON, order-insensitive)."""
    blob = json.dumps(phase, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def plan_hash(plan: dict) -> str:
    """Content hash of a whole plan's phase list."""
    return hashlib.sha256(
        "".join(phase_hash(p) for p in plan["phases"]).encode()).hexdigest()[:16]


def new_progress(plan: dict, path: str) -> dict:
    """A fresh progress state for a plan (every phase pending)."""
    return {
        "version": PROGRESS_VERSION,
        "name": plan.get("name", plan_name(path)),
        "plan": os.path.relpath(os.path.abspath(path), REPO_ROOT),
        "plan_hash": plan_hash(plan),
        "phase_hashes": [phase_hash(p) for p in plan["phases"]],
        "phase_index": 0,
        "phases": [{"kind": p["kind"], "status": "pending", "started": None,
                    "finished": None, "exit_code": None, "summary": None}
                   for p in plan["phases"]],
    }


def read_progress(path: str) -> dict | None:
    """The progress sidecar for a plan path, or None when the run never started."""
    return read_progress_state(progress_path(path), _LABEL)


def write_progress(path: str, state: dict) -> None:
    write_progress_state(progress_path(path), state, _LABEL)


def check_plan_unchanged(plan: dict, state: dict) -> None:
    """Refuse to resume when an already-executed phase was edited.

    Phase hashes are compared position by position for every phase the previous
    run finished or was inside (``done`` / ``running``): those already produced
    checkpoints and driver sidecars, so redefining them would make the recorded
    progress a lie. Phases that are still ``pending`` — or that ``failed`` and
    will be re-run from their start — may be freely rewritten, which is the whole
    point of allowing edits to the future of a plan."""
    old = state.get("phase_hashes") or []
    new = [phase_hash(p) for p in plan["phases"]]
    for i, ph in enumerate(state.get("phases", [])):
        if ph.get("status") not in ("done", "running"):
            continue
        if i >= len(new) or i >= len(old) or new[i] != old[i]:
            raise PlanError(
                f"cannot resume: phase {i + 1} ({ph.get('kind')}, status "
                f"{ph.get('status')}) has changed since it ran. Completed and "
                f"in-flight phases are immutable; edit only the phases still "
                f"ahead, or delete the progress file to start the curriculum "
                f"over.")


# ── per-phase driver sidecars (status view) ──────────────────────────────────

def phase_sidecar_path(phase: dict, checkpoint_dir: str = None) -> str | None:
    """The driver progress sidecar a phase's subcommand writes, if it has one."""
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    kind = phase["kind"]
    values = phase_values(phase)
    if kind == "league":
        return os.path.join(checkpoint_dir,
                            f"_league_progress{shard_tag(values.get('shard'))}.json")
    if kind == "exploiter":
        arch = values.get("archetype")
        return os.path.join(checkpoint_dir, f"_exploiter_{arch}_progress.json")
    if kind == "az-league":
        return os.path.join(checkpoint_dir, "_az_league_progress.json")
    return None


def phase_detail(phase: dict, checkpoint_dir: str = None) -> str | None:
    """One-line progress read out of a phase's own driver sidecar (or None).

    This is what the runner records as a completed phase's ``summary`` and what
    the TUI status view shows for the phase in flight — the curriculum never
    duplicates the drivers' own bookkeeping, it just reads it."""
    path = phase_sidecar_path(phase, checkpoint_dir)
    if not path or not os.path.exists(path):
        return None
    state = read_progress_state(path, _LABEL)
    if not state:
        return None
    kind = phase["kind"]
    if kind in ("league", "exploiter"):
        done, total = state.get("steps_done"), state.get("total_timesteps")
        if done is None:
            return None
        txt = f"{done:,} steps" if total is None else f"{done:,}/{total:,} steps"
        if kind == "league" and state.get("rotation") is not None:
            txt += f", rotation {state['rotation']}"
        return txt
    if kind == "az-league":
        slot = state.get("slot_index")
        results = state.get("results") or []
        txt = f"slot {slot}"
        if results:
            last = results[-1]
            txt += (f", last gate {last.get('deck')} wr="
                    f"{last.get('gate_win_rate')}"
                    f"{' PROMOTED' if last.get('promoted') else ''}")
        return txt
    return None


# ── runner ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def format_status(plan: dict, state: dict | None, path: str) -> str:
    """The human-readable status view (also used by the TUI status pane)."""
    lines = [f"curriculum '{plan.get('name', plan_name(path))}'  "
             f"({os.path.relpath(os.path.abspath(path), REPO_ROOT)})"]
    if state is None:
        lines.append("  not started — no progress file "
                     f"({os.path.basename(progress_path(path))})")
    phases = plan["phases"]
    for i, phase in enumerate(phases):
        ph = (state or {}).get("phases", [])
        rec = ph[i] if i < len(ph) else {}
        status = rec.get("status", "pending")
        bits = [f"  [{i + 1}/{len(phases)}] {phase['kind']:<10} {status:<8}"]
        if rec.get("exit_code") not in (None, 0):
            bits.append(f"exit {rec['exit_code']}")
        if rec.get("started"):
            bits.append(f"started {rec['started']}")
        if rec.get("finished"):
            bits.append(f"finished {rec['finished']}")
        detail = rec.get("summary")
        if status == "running":
            detail = phase_detail(phase) or detail
        if detail:
            bits.append(str(detail))
        lines.append("  ".join(bits))
    return "\n".join(lines)


def run_curriculum(plan_ref: str, resume: bool = False, status: bool = False,
                   dry_run: bool = False, checkpoint_dir: str = None) -> int:
    """Execute (or inspect) a curriculum plan. Returns a process exit code.

    Phases run one at a time as subprocesses. Progress is written BEFORE a phase
    launches (status ``running``) and again the moment it exits (``done`` /
    ``failed``) — so a crash always leaves the sidecar pointing at the phase that
    was in flight, never at a phase that merely might have started.
    """
    path = plan_path(plan_ref)
    plan = load_plan(path)
    phases = plan["phases"]
    state = read_progress(path)

    if status:
        print(format_status(plan, state, path))
        return 0

    if dry_run:
        print(f"curriculum '{plan.get('name')}' — {len(phases)} phase(s), "
              f"dry run{' (resume)' if resume else ''}:")
        rows = (state or {}).get("phases", [])
        for i, phase in enumerate(phases):
            rec = rows[i] if i < len(rows) else {}
            st = rec.get("status", "pending")
            if resume and st == "done":
                print(f"  [{i + 1}/{len(phases)}] {phase['kind']:<10} skip (done)")
                continue
            use_resume = resume and st == "running"
            argv = compose_phase_argv(phase, resume=use_resume)
            tag = " (--resume)" if use_resume and phase["kind"] in RESUMABLE_KINDS \
                else (" (restart)" if use_resume else "")
            print(f"  [{i + 1}/{len(phases)}] {phase['kind']:<10}{tag} "
                  f"{display_argv(argv)}")
        return 0

    if state is None:
        state = new_progress(plan, path)
    elif resume:
        check_plan_unchanged(plan, state)
        # Future phases may have been rewritten (or appended) — re-sync the
        # recorded hashes/rows so the plan and the sidecar stay in step.
        state["phase_hashes"] = [phase_hash(p) for p in phases]
        state["plan_hash"] = plan_hash(plan)
        rows = state.get("phases", [])
        rows = rows[:len(phases)] + [{"kind": p["kind"], "status": "pending",
                                      "started": None, "finished": None,
                                      "exit_code": None, "summary": None}
                                     for p in phases[len(rows):]]
        for row, phase in zip(rows, phases):
            row["kind"] = phase["kind"]
        state["phases"] = rows
    elif any(r.get("status") != "pending" for r in state.get("phases", [])):
        print(f"[{_LABEL}] {progress_path(path)} already records progress "
              f"(phase {state.get('phase_index', 0) + 1}/{len(phases)}). Pass "
              f"--resume to continue it, --status to inspect it, or delete the "
              f"file to run the curriculum from the start.", file=sys.stderr)
        return 2

    print(f"[{_LABEL}] '{plan.get('name')}': {len(phases)} phase(s) from "
          f"{os.path.relpath(path, REPO_ROOT)}")
    for i, phase in enumerate(phases):
        rec = state["phases"][i]
        if rec.get("status") == "done":
            print(f"[{_LABEL}] phase {i + 1}/{len(phases)} ({phase['kind']}): "
                  f"already done — skipping")
            continue
        use_resume = (resume and rec.get("status") == "running"
                      and phase["kind"] in RESUMABLE_KINDS)
        argv = compose_phase_argv(phase, resume=use_resume)
        rec.update(status="running", started=_now(), finished=None,
                   exit_code=None)
        state["phase_index"] = i
        write_progress(path, state)
        print(f"\n{'=' * 70}")
        print(f"[{_LABEL}] phase {i + 1}/{len(phases)}: {phase['kind']}"
              f"{' (resuming)' if use_resume else ''}")
        print(f"  $ {display_argv(argv)}")
        print(f"{'=' * 70}", flush=True)
        try:
            rc = subprocess.run(argv, cwd=REPO_ROOT).returncode
        except KeyboardInterrupt:
            rc = 130
        if rc == 0:
            rec.update(status="done", finished=_now(), exit_code=0,
                       summary=phase_detail(phase, checkpoint_dir))
            state["phase_index"] = i + 1
            write_progress(path, state)
            print(f"[{_LABEL}] phase {i + 1}/{len(phases)} ({phase['kind']}) done"
                  + (f": {rec['summary']}" if rec["summary"] else ""))
            continue
        # SIGINT reaches the child as 130 (its own handler / shell convention) or
        # as a negative signal number when it died on the signal itself.
        interrupted = rc in (130, -2)
        rec.update(exit_code=rc, finished=None if interrupted else _now(),
                   status="running" if interrupted else "failed",
                   summary=phase_detail(phase, checkpoint_dir))
        write_progress(path, state)
        if interrupted:
            print(f"\n[{_LABEL}] interrupted during phase {i + 1}/{len(phases)} "
                  f"({phase['kind']}). Resume with: python train/train.py "
                  f"curriculum --plan {os.path.relpath(path, REPO_ROOT)} --resume")
            return 130
        print(f"\n[{_LABEL}] phase {i + 1}/{len(phases)} ({phase['kind']}) FAILED "
              f"(exit {rc}). Fix it, then resume with: python train/train.py "
              f"curriculum --plan {os.path.relpath(path, REPO_ROOT)} --resume",
              file=sys.stderr)
        return rc

    state["phase_index"] = len(phases)
    write_progress(path, state)
    print(f"\n[{_LABEL}] '{plan.get('name')}' complete: "
          f"{len(phases)} phase(s) finished.")
    return 0
