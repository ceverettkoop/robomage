#!/usr/bin/env python3
"""The ``knowledge`` gate tier: the knowledge sidecar's soundness, exactness and isolation.

Every seat's RMSTATE STATE frame ends with the ``k_*`` tables of its writer's own knowledge
instance (``src/state_format/knowledge_feed.h``, ``docs/knowledge_store.md`` §7, §9). The tier runs:

  1. selftest   ``rmknow --selftest`` (the update rules on synthetic event streams) and
                ``rmknow --exactness EXACTNESS_N`` (the exact solver and compaction against an
                independent enumerator of the worlds);
  2. fuzz       explore-vs-explore games (``FUZZ_GAMES``) with ``--selfcheck-knowledge`` (which implies
                ``--selfcheck-state``): every ``k_*`` fact of both seats against the true world built
                from the engine state at every decision, and the knowledge state across the
                self-check's snapshot restores. Decks for the sidecar's hard cases: Brainstorm and
                Ponder put-backs (ur_delver, bug, uw_control), tutors (wubg_doomsday, reanimator),
                Thoughtseize and Cabal Therapy (bug, wrb_energy, reanimator), Sylvan Library (mav),
                reanimator's face-down exile; bo1 and bo3 with sideboarding; one game sampled at every
                decision. Two write ``--rmstate-dump`` files. A finished game needs one errors=0
                ``SELFCHECK-STATE`` and one ``SELFCHECK-KNOWLEDGE`` summary;
  3. isolation  ``--machine-format rmstate`` sessions (``STREAM_GAMES``: bo1, bo3 with sideboarding,
                a bo3 under ``--closed-decklist``), each with a ``--rmstate-dump`` alongside and
                ``--selfcheck-knowledge``: every stream STATE frame equals the dump's frame for that
                seat and decision, ``k_*`` tables included (``test_rmstate.check_stream``: the model
                follows the seat's events, not when they were drained), and ``rmknow --isolation``
                recomputes the ``k_*`` tables of every seat frame of each writer from that writer's
                own frames byte for byte (each dump, the fuzz dumps included, and each saved stream
                ``stream_<seed>.stream.rms``; ``test_rmstate.check_knowledge_isolation``);
  4. truth      ``rmknow --truth`` for both seats on every dump (fuzz and stream sessions): the same
                soundness check (``src/knowledge/truth.h``) against the dump's omniscient frames;
  5. shadow     the shadow comparison (``src/state_format/shadow_knowledge.h``, part of
                ``--selfcheck-knowledge``): every ``SHADOW:`` line the fuzz and stream sessions'
                engines printed (a place where the ``k_*`` tables and the old BQUERY vector's memory
                disagree: known library tops, known opponent hand cards, cards seen this match) must
                match a signature of ``train/regression/knowledge_divergences.json`` (``classify``);
                and every signature's regression scenario, run with ``--selfcheck-knowledge``, must
                pass and print a ``SHADOW:`` line that signature classifies.

The replay corpus runs under ``--selfcheck-knowledge`` in the ``rmstate`` tier's corpus part
(``train/test_selfcheck_state.py``), which shares those engine runs and classifies their ``SHADOW:``
lines with ``shadow_log_failures``.

**knowledge_divergences.json** is a list of signatures, tried in order (the first that matches
classifies a line): ``name``; ``kind`` (``top`` / ``hand`` / ``seen``, or a list of them) and
``direction`` (``old_more``: the old vector reports something the ``k_*`` tables do not;
``new_more``: the other way; ``conflict``), which must equal the line's; optionally ``owner``
(``self`` / ``opp``); ``pattern``, a regex searched in the whole ``SHADOW:`` line (its ``events=``
field names the frame's events, a hand line's ``left=`` / ``entered=`` / ``slots=`` the card's moves
out of and into the hand and its copies in the vector's ``opp_known_hand`` slots);
``classification`` (``old-vector leak``: the old vector shows something the player cannot know;
``old-vector quirk``: a legitimate difference, such as a deduction the old vector does not make);
``why`` (the CR or format rule); ``scenario`` (a ``train/regression/scenarios`` name that
demonstrates it). A mismatch where the ``k_*`` tables know less than a leak-free old vector is a
sidecar or event bug to fix, never a signature.

Entry points: ``run(binary=…, out_dir=…)`` returns the list of failure messages (empty = pass), for
``train/ci_check.py``; ``main()`` is the CLI (exit 1 on failure)::

    train/.venv/bin/python train/test_knowledge.py                    # everything
    train/.venv/bin/python train/test_knowledge.py --part fuzz        # one part (repeatable)
    train/.venv/bin/python train/test_knowledge.py --part fuzz --fuzz-seed 203   # one game
    train/.venv/bin/python train/test_knowledge.py --classify FILE…   # classify SHADOW lines of logs

The truth and shadow parts need the dumps and logs the fuzz and isolation parts write, so they run
them too. Engine stderr goes to ``--out-dir`` (default ``ci_out/knowledge``); the dumps and saved
streams are deleted once checked. Torch-free; needs the engine and rmknow (debug tier by default, ``--binary``).
"""
import argparse
import collections
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rmstate  # noqa: E402
import test_rmstate  # noqa: E402
import test_scenarios  # noqa: E402
import test_selfcheck_state  # noqa: E402
from cli_spec import BINARY  # noqa: E402

_TRAIN = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TRAIN)
_TIMEOUT_S = 600

SELFCHECK_KNOWLEDGE = ["--selfcheck-knowledge"]
SELFCHECK_KNOWLEDGE_EVERY_DECISION = ["--selfcheck-knowledge", "--selfcheck-state-sample", "1"]
SUMMARY_KINDS = ("STATE", "KNOWLEDGE")

EXACTNESS_N = 40

# Fuzz games: (deck_a, deck_b, bo3, seed, extra engine flags, write an --rmstate-dump).
FUZZ_GAMES = [
    ("league/reanimator", "league/uw_control", True, 201, SELFCHECK_KNOWLEDGE, True),
    ("mav", "league/ur_delver", False, 202, SELFCHECK_KNOWLEDGE, True),
    ("league/bug", "league/wrb_energy", True, 203, SELFCHECK_KNOWLEDGE, False),
    ("league/wubg_doomsday", "league/bw_dnt", False, 204, SELFCHECK_KNOWLEDGE_EVERY_DECISION, False),
    ("mav", "league/reanimator", True, 205, SELFCHECK_KNOWLEDGE, False),
    ("league/ur_delver", "league/bug", False, 206, SELFCHECK_KNOWLEDGE, False),
]

# --machine-format rmstate sessions, each with a dump: (deck_a, deck_b, bo3, seed, extra engine flags).
STREAM_GAMES = [
    ("league/ur_delver", "league/bug", False, 31, SELFCHECK_KNOWLEDGE),
    ("league/reanimator", "league/uw_control", True, 32, SELFCHECK_KNOWLEDGE),
    ("league/bug", "league/wubg_doomsday", True, 33, SELFCHECK_KNOWLEDGE + ["--closed-decklist"]),
]

PARTS = ["selftest", "fuzz", "isolation", "truth", "shadow"]

DIVERGENCES_PATH = os.path.join(_TRAIN, "regression", "knowledge_divergences.json")
CLASSIFICATIONS = ("old-vector leak", "old-vector quirk")
_SIGNATURE_KEYS = {"name", "kind", "direction", "owner", "pattern", "classification", "why", "scenario"}
_SHADOW_RE = re.compile(r"^SHADOW: (top|hand|seen) (old_more|new_more|conflict) viewer=[AB] owner=(self|opp) ")
_SHADOW_SUMMARY = "SHADOW-KNOWLEDGE:"
# Unclassified SHADOW lines reported per log (the rest are counted).
MAX_UNCLASSIFIED_SHOWN = 10


def _workers(n):
    return max(1, min(n, os.cpu_count() or 4))


def _rmknow(binary):
    return test_rmstate.rmknow_binary(binary)


# ── 1. selftest ────────────────────────────────────────────────────────────────────────────────

def check_selftest(binary):
    fails = []
    for args, summary in ((["--selftest"], "RMKNOW_SELFTEST:"), (["--exactness", str(EXACTNESS_N)], "RMKNOW_EXACTNESS:")):
        label = "rmknow " + " ".join(args)
        r = subprocess.run([_rmknow(binary)] + args, capture_output=True, text=True, timeout=_TIMEOUT_S)
        lines = (r.stdout + r.stderr).splitlines()
        last = [ln for ln in lines if ln.startswith(summary)]
        bad = [ln for ln in lines if "ERROR" in ln or "FAIL" in ln]
        if r.returncode != 0 or not last or bad:
            fails.append(f"{label}: exit {r.returncode}: {(bad or lines)[:3]}")
        else:
            print(f"  OK   {label}: {last[-1]}", flush=True)
    return fails


# ── 2. fuzz ────────────────────────────────────────────────────────────────────────────────────

def play_fuzz(binary, out_dir, seeds=None):
    """Plays FUZZ_GAMES (those with a seed in `seeds`, if given) in parallel; returns (failure
    messages, dump paths)."""
    games = [g for g in FUZZ_GAMES if not seeds or g[3] in seeds]
    if not games:
        return [f"fuzz: no game has a seed in {sorted(seeds)}"], []
    tasks = [(g, binary, out_dir) for g in games]
    with ProcessPoolExecutor(max_workers=_workers(len(tasks))) as ex:
        results = list(ex.map(test_selfcheck_state._play_fuzz_game, tasks))
    fails, dumps = [], []
    for game, (err_path, game_dumps, finished, crashed) in zip(games, results):
        label = "fuzz " + test_selfcheck_state._fuzz_label(game)
        got = []
        if crashed is not None:
            got.append(f"{label}: engine crashed: {crashed} (see {err_path})")
        complete = finished and crashed is None
        got += test_selfcheck_state.scan_engine_log(err_path, label, summaries=1 if complete else None,
                                                    kinds=SUMMARY_KINDS)
        if game[5] and not game_dumps:
            got.append(f"{label}: no --rmstate-dump file was written")
        if not got:
            print(f"  OK   {label}{'' if finished else ' (stopped at the decision cap)'}: "
                  f"{_knowledge_summary(err_path)}", flush=True)
        fails += got
        dumps += game_dumps
    return fails, dumps


def _knowledge_summary(err_path):
    with open(err_path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh if ln.startswith("SELFCHECK-KNOWLEDGE:")]
    return lines[-1] if lines else "no SELFCHECK-KNOWLEDGE summary"


# ── 3. isolation ───────────────────────────────────────────────────────────────────────────────

def _has_sideboarding_frame(payloads):
    dec = rmstate.StreamDecoder()
    for p in payloads:
        f = dec.feed(p)
        if f.kind == "STATE" and f.tables["header"][0]["sideboarding"]:
            return True
    return False


def _stream_session(task):
    """One stream session with its dump (a process-pool task): returns (failure messages, the dump,
    the saved stream)."""
    game, binary, out_dir = task
    deck_a, deck_b, bo3, seed, flags = game
    label = f"stream {os.path.basename(deck_a)} vs {os.path.basename(deck_b)} {'bo3' if bo3 else 'bo1'} seed {seed}"
    dump = os.path.join(out_dir, f"stream_{seed}.rms")
    err = os.path.join(out_dir, f"stream_{seed}.err")
    try:
        payloads = test_rmstate.run_stream(binary, deck_a, deck_b, bo3, seed, dump=dump, err_path=err, flags=flags)
    except (rmstate.RMStateError, subprocess.TimeoutExpired) as exc:
        return [f"{label}: {exc} (see {err})"], None, None
    fails = test_rmstate.check_stream(payloads, dump, label)
    saved = os.path.join(out_dir, f"stream_{seed}.stream.rms")
    test_rmstate.write_stream(payloads, saved)
    fails += test_selfcheck_state.scan_engine_log(err, label, summaries=1, kinds=SUMMARY_KINDS)
    if bo3 and not _has_sideboarding_frame(payloads):
        fails.append(f"{label}: no sideboarding frame (the session should cover sideboarding)")
    if not fails:
        print(f"  OK   {label}: {len(payloads)} records, stream frames equal the dump's (k_* included); "
              f"{_knowledge_summary(err)}", flush=True)
    return fails, dump, saved


def play_streams(binary, out_dir):
    """The stream sessions in parallel; returns (failure messages, dump paths, saved stream paths)."""
    tasks = [(g, binary, out_dir) for g in STREAM_GAMES]
    with ProcessPoolExecutor(max_workers=_workers(len(tasks))) as ex:
        results = list(ex.map(_stream_session, tasks))
    fails, dumps, streams = [], [], []
    for got, dump, saved in results:
        fails += got
        if dump:
            dumps.append(dump)
            streams.append(saved)
    return fails, dumps, streams


def check_isolation(binary, files):
    """rmknow --isolation for both seats on each of `files` (dumps and saved streams)."""
    fails = []
    with ThreadPoolExecutor(max_workers=_workers(len(files))) as ex:
        for got in ex.map(lambda path: test_rmstate.check_knowledge_isolation(path, binary), files):
            fails += got
    if not files:
        fails.append("isolation: no dump or saved stream to check")
    return fails


# ── 4. truth ───────────────────────────────────────────────────────────────────────────────────

def _truth_one(task):
    path, seat, binary = task
    r = subprocess.run([_rmknow(binary), "--viewer", seat, "--truth", "--quiet", "--in", path],
                       capture_output=True, text=True, timeout=_TIMEOUT_S)
    lines = r.stderr.splitlines()
    summary = [ln for ln in lines if ln.startswith("RMKNOW:")]
    label = f"truth {os.path.basename(path)} viewer {seat}"
    if r.returncode != 0 or not summary or " truth_frames=0 " in summary[0]:
        bad = [ln for ln in lines if "ERROR:" in ln]
        return [f"{label}: rmknow --truth exit {r.returncode}: {bad[:3]} {summary}"], None
    return [], f"{label}: {summary[0]}"


def check_truth(binary, dumps):
    tasks = [(path, seat, binary) for path in dumps for seat in ("A", "B")]
    fails = []
    with ThreadPoolExecutor(max_workers=_workers(len(tasks))) as ex:
        for got, ok in ex.map(_truth_one, tasks):
            fails += got
            if ok:
                print(f"  OK   {ok}", flush=True)
    if not dumps:
        fails.append("truth: no dump to check")
    return fails


# ── 5. shadow ──────────────────────────────────────────────────────────────────────────────────

def load_divergences(path=DIVERGENCES_PATH):
    """The signatures of knowledge_divergences.json, each with its pattern compiled (`regex`);
    raises ValueError naming a malformed entry."""
    with open(path) as fh:
        sigs = json.load(fh)
    if not isinstance(sigs, list) or not sigs:
        raise ValueError(f"{path}: expected a non-empty list of signatures")
    names = set()
    for i, sig in enumerate(sigs):
        where = f"{path}: signature {i} ({sig.get('name', '?')})"
        missing = _SIGNATURE_KEYS - {"owner"} - set(sig)
        unknown = set(sig) - _SIGNATURE_KEYS
        if missing or unknown:
            raise ValueError(f"{where}: missing {sorted(missing)}, unknown {sorted(unknown)}")
        if sig["name"] in names:
            raise ValueError(f"{where}: duplicate name")
        names.add(sig["name"])
        kinds = sig["kind"] if isinstance(sig["kind"], list) else [sig["kind"]]
        if not kinds or set(kinds) - {"top", "hand", "seen"}:
            raise ValueError(f"{where}: kind must be top, hand, seen or a list of them")
        sig["kinds"] = set(kinds)
        if sig["direction"] not in ("old_more", "new_more", "conflict"):
            raise ValueError(f"{where}: direction must be old_more, new_more or conflict")
        if sig.get("owner", "self") not in ("self", "opp"):
            raise ValueError(f"{where}: owner must be self or opp")
        if sig["classification"] not in CLASSIFICATIONS:
            raise ValueError(f"{where}: classification must be one of {CLASSIFICATIONS}")
        if not os.path.exists(os.path.join(test_scenarios.SCENARIO_DIR, sig["scenario"] + ".json")):
            raise ValueError(f"{where}: no scenario {sig['scenario']}")
        sig["regex"] = re.compile(sig["pattern"])
    return sigs


def classify(line, sigs):
    """The name of the first signature that matches the SHADOW line `line`, or None."""
    m = _SHADOW_RE.match(line)
    if not m:
        return None
    kind, direction, owner = m.groups()
    for sig in sigs:
        if kind in sig["kinds"] and direction == sig["direction"] and sig.get("owner", owner) == owner \
                and sig["regex"].search(line):
            return sig["name"]
    return None


def shadow_log_failures(path, label, summaries=None, sigs=None, counts=None):
    """Failure messages for one engine stderr log: every SHADOW line must be classified, and with
    `summaries` there must be exactly that many SHADOW-KNOWLEDGE summary lines. `counts` (a
    Counter), when given, gets the lines per signature."""
    try:
        sigs = sigs if sigs is not None else load_divergences()
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        return [f"{label}: {exc}"]
    fails, unclassified, found = [], 0, 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if line.startswith(_SHADOW_SUMMARY):
                found += 1
            if not line.startswith("SHADOW:"):
                continue
            name = classify(line, sigs)
            if name is None:
                unclassified += 1
                if unclassified <= MAX_UNCLASSIFIED_SHOWN:
                    fails.append(f"{label}: unclassified shadow mismatch {path}:{n}: {line}")
            elif counts is not None:
                counts[name] += 1
    if unclassified > MAX_UNCLASSIFIED_SHOWN:
        fails.append(f"{label}: {unclassified - MAX_UNCLASSIFIED_SHOWN} more unclassified shadow mismatch(es) in {path}")
    if summaries is not None and found != summaries:
        fails.append(f"{label}: {found} {_SHADOW_SUMMARY} summary line(s) in {path}, expected {summaries}")
    return fails


def _shadow_scenario(task):
    """One divergence scenario under --selfcheck-knowledge (a thread-pool task): (name, failures,
    the signature names its SHADOW lines classify to)."""
    name, binary, sigs = task
    path = os.path.join(test_scenarios.SCENARIO_DIR, name + ".json")
    _n, fails, out, _frames = test_scenarios.run_one(path, binary, extra_flags=SELFCHECK_KNOWLEDGE)
    classes = set()
    for line in out.splitlines():
        if line.startswith("SHADOW:"):
            got = classify(line, sigs)
            if got is None:
                fails.append(f"unclassified shadow mismatch: {line}")
            else:
                classes.add(got)
    return name, fails, classes


def check_shadow(binary, logs):
    """The shadow part: the SHADOW lines of `logs` (engine stderr files) classified, and every
    signature demonstrated by its scenario."""
    try:
        sigs = load_divergences()
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        return [f"shadow: {exc}"]
    fails = []
    counts = collections.Counter()
    for path in logs:
        fails += shadow_log_failures(path, "shadow " + os.path.basename(path), sigs=sigs, counts=counts)
    if not logs:
        fails.append("shadow: no engine log to classify")
    scenarios = sorted({sig["scenario"] for sig in sigs})
    tasks = [(name, binary, sigs) for name in scenarios]
    shown = {}
    with ThreadPoolExecutor(max_workers=_workers(len(tasks))) as ex:
        for name, got, classes in ex.map(_shadow_scenario, tasks):
            fails += [f"shadow scenario {name}: {f}" for f in got]
            shown[name] = classes
    for sig in sigs:
        if sig["name"] not in shown.get(sig["scenario"], set()):
            fails.append(f"shadow: scenario {sig['scenario']} prints no SHADOW line classified "
                         f"{sig['name']} (with {' '.join(SELFCHECK_KNOWLEDGE)})")
    if not fails:
        tally = ", ".join(f"{n} {counts[n]}" for n in (sig["name"] for sig in sigs) if counts[n])
        print(f"  OK   shadow: {sum(counts.values())} mismatch line(s) in {len(logs)} log(s), all classified "
              f"({tally}); {len(sigs)} signatures demonstrated by {len(scenarios)} scenarios", flush=True)
    return fails


def run(binary=BINARY, out_dir=None, parts=None, fuzz_seeds=None):
    """The tier; returns failure messages (empty = pass). `parts` (default PARTS) selects parts; the
    truth and shadow parts reuse the fuzz and isolation parts' dumps and logs (and run those parts
    if they are not selected). `fuzz_seeds` limits the fuzz part to the games with those seeds."""
    parts = list(parts or PARTS)
    out_dir = os.path.abspath(out_dir or os.path.join(_REPO_ROOT, "ci_out", "knowledge"))
    os.makedirs(out_dir, exist_ok=True)
    fails = []
    fuzz_dumps, stream_dumps, streams = [], [], []
    times = []
    for part in PARTS:
        needed = part in parts or (part in ("fuzz", "isolation") and ("truth" in parts or "shadow" in parts))
        if not needed:
            continue
        t0 = time.monotonic()
        print(f"knowledge: {part}", flush=True)
        if part == "selftest":
            fails += check_selftest(binary)
        elif part == "fuzz":
            got, fuzz_dumps = play_fuzz(binary, out_dir, fuzz_seeds)
            fails += got
        elif part == "isolation":
            got, stream_dumps, streams = play_streams(binary, out_dir)
            fails += got
            fails += check_isolation(binary, fuzz_dumps + stream_dumps + streams)
        elif part == "truth":
            fails += check_truth(binary, fuzz_dumps + stream_dumps)
        elif part == "shadow":
            logs = sorted(glob.glob(os.path.join(out_dir, "fuzz_*.err")) + glob.glob(os.path.join(out_dir, "stream_*.err")))
            fails += check_shadow(binary, logs)
        times.append((part, time.monotonic() - t0))
    print("knowledge: " + ", ".join(f"{p} {dt:.0f}s" for p, dt in times), flush=True)
    # The dumps and saved streams are large; once checked (pass or fail: the failure names the game
    # to replay) they go.
    for path in glob.glob(os.path.join(out_dir, "fuzz_*_dumps")) + glob.glob(os.path.join(out_dir, "*.rms*")):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--binary", default=BINARY, help="engine binary (default: the debug tier); rmknow beside it")
    ap.add_argument("--out-dir", help="engine stderr and dumps (default ci_out/knowledge)")
    ap.add_argument("--part", action="append", choices=PARTS, help="run only this part (repeatable)")
    ap.add_argument("--fuzz-seed", type=int, action="append",
                    help="play only the fuzz game with this seed (repeatable; its label names it)")
    ap.add_argument("--classify", nargs="+", metavar="FILE",
                    help="only classify the SHADOW lines of these engine logs (counts per signature)")
    args = ap.parse_args(argv)
    if args.classify:
        counts = collections.Counter()
        fails = []
        for path in args.classify:
            fails += shadow_log_failures(path, os.path.basename(path), counts=counts)
        for name, n in counts.most_common():
            print(f"  {n:7d}  {name}", flush=True)
    else:
        fails = run(args.binary, args.out_dir, args.part, args.fuzz_seed)
    for f in fails:
        print(f"  FAIL {f}", flush=True)
    print(f"knowledge: {'FAIL' if fails else 'OK'} ({len(fails)} failure(s))", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
