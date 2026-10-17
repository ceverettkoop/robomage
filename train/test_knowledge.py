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
                soundness check (``src/knowledge/truth.h``) against the dump's omniscient frames.

The replay corpus runs under ``--selfcheck-knowledge`` in the ``rmstate`` tier's corpus part
(``train/test_selfcheck_state.py``), which shares those engine runs.

Entry points: ``run(binary=…, out_dir=…)`` returns the list of failure messages (empty = pass), for
``train/ci_check.py``; ``main()`` is the CLI (exit 1 on failure)::

    train/.venv/bin/python train/test_knowledge.py                    # everything
    train/.venv/bin/python train/test_knowledge.py --part fuzz        # one part (repeatable)
    train/.venv/bin/python train/test_knowledge.py --part fuzz --fuzz-seed 203   # one game

The truth part needs the dumps the fuzz and isolation parts write, so it runs them too. Engine
stderr goes to ``--out-dir`` (default ``ci_out/knowledge``); the dumps and saved streams are deleted
once checked. Torch-free; needs the engine and rmknow (debug tier by default, ``--binary``).
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rmstate  # noqa: E402
import test_rmstate  # noqa: E402
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

PARTS = ["selftest", "fuzz", "isolation", "truth"]


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


def run(binary=BINARY, out_dir=None, parts=None, fuzz_seeds=None):
    """The tier; returns failure messages (empty = pass). `parts` (default PARTS) selects parts; the
    truth part reuses the fuzz and isolation parts' dumps (and runs those parts if they are not
    selected). `fuzz_seeds` limits the fuzz part to the games with those seeds."""
    parts = list(parts or PARTS)
    out_dir = os.path.abspath(out_dir or os.path.join(_REPO_ROOT, "ci_out", "knowledge"))
    os.makedirs(out_dir, exist_ok=True)
    fails = []
    fuzz_dumps, stream_dumps, streams = [], [], []
    times = []
    for part in PARTS:
        needed = part in parts or (part in ("fuzz", "isolation") and "truth" in parts)
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
    args = ap.parse_args(argv)
    fails = run(args.binary, args.out_dir, args.part, args.fuzz_seed)
    for f in fails:
        print(f"  FAIL {f}", flush=True)
    print(f"knowledge: {'FAIL' if fails else 'OK'} ({len(fails)} failure(s))", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
