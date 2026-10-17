#!/usr/bin/env python3
"""The ``knowledge`` gate tier: the knowledge sidecar's isolation over real games.

Every seat's RMSTATE STATE frame ends with the ``k_*`` tables of its writer's own knowledge
instance (``src/state_format/knowledge_feed.h``, ``docs/knowledge_store.md`` §9.6). The tier runs:

  1. isolation  ``--machine-format rmstate`` sessions (``test_rmstate.run_stream``: bo1, bo3 with
                sideboarding, a bo3 under ``--closed-decklist``), each with a ``--rmstate-dump``
                alongside and the self-checks on: every stream STATE frame equals the dump's frame
                for that seat and decision, ``k_*`` tables included (``test_rmstate.check_stream``:
                the model follows the seat's events, not when they were drained), and
                ``rmknow --isolation`` recomputes the ``k_*`` tables of every seat frame of each
                writer from that writer's own frames byte for byte (the dump's, and the saved
                stream ``stream_<seed>.stream.rms``; ``test_rmstate.check_knowledge_isolation``).

Entry points: ``run(binary=…, out_dir=…)`` returns the list of failure messages (empty = pass), for
``train/ci_check.py``; ``main()`` is the CLI (exit 1 on failure)::

    train/.venv/bin/python train/test_knowledge.py                    # everything
    train/.venv/bin/python train/test_knowledge.py --part isolation   # one part (repeatable)

Engine stderr goes to ``--out-dir`` (default ``ci_out/knowledge``); the dumps and saved streams are
deleted once checked. Torch-free; needs the engine and rmknow (debug tier by default, ``--binary``).
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

SELFCHECK = ["--selfcheck-state"]

# --machine-format rmstate sessions, each with a dump: (deck_a, deck_b, bo3, seed, extra engine flags).
STREAM_GAMES = [
    ("league/ur_delver", "league/bug", False, 31, SELFCHECK),
    ("league/reanimator", "league/uw_control", True, 32, SELFCHECK),
    ("league/bug", "league/wubg_doomsday", True, 33, SELFCHECK + ["--closed-decklist"]),
]

PARTS = ["isolation"]


def _workers(n):
    return max(1, min(n, os.cpu_count() or 4))


def _has_sideboarding_frame(payloads):
    dec = rmstate.StreamDecoder()
    for p in payloads:
        f = dec.feed(p)
        if f.kind == "STATE" and f.tables["header"][0]["sideboarding"]:
            return True
    return False


def _stream_session(task):
    """One stream session with its dump (a process-pool task): returns (failure messages, the dump
    and saved stream paths)."""
    game, binary, out_dir = task
    deck_a, deck_b, bo3, seed, flags = game
    label = f"stream {os.path.basename(deck_a)} vs {os.path.basename(deck_b)} {'bo3' if bo3 else 'bo1'} seed {seed}"
    dump = os.path.join(out_dir, f"stream_{seed}.rms")
    err = os.path.join(out_dir, f"stream_{seed}.err")
    try:
        payloads = test_rmstate.run_stream(binary, deck_a, deck_b, bo3, seed, dump=dump, err_path=err, flags=flags)
    except (rmstate.RMStateError, subprocess.TimeoutExpired) as exc:
        return [f"{label}: {exc} (see {err})"], []
    fails = test_rmstate.check_stream(payloads, dump, label)
    saved = os.path.join(out_dir, f"stream_{seed}.stream.rms")
    test_rmstate.write_stream(payloads, saved)
    fails += test_selfcheck_state.scan_engine_log(err, label, summaries=1)
    if bo3 and not _has_sideboarding_frame(payloads):
        fails.append(f"{label}: no sideboarding frame (the session should cover sideboarding)")
    if not fails:
        print(f"  OK   {label}: {len(payloads)} records, stream frames equal the dump's (k_* included)", flush=True)
    return fails, [dump, saved]


def check_isolation(binary, out_dir):
    """Part 1: the stream sessions, then rmknow --isolation on each writer's frames."""
    tasks = [(g, binary, out_dir) for g in STREAM_GAMES]
    with ProcessPoolExecutor(max_workers=_workers(len(tasks))) as ex:
        results = list(ex.map(_stream_session, tasks))
    fails, files = [], []
    for got, paths in results:
        fails += got
        files += paths
    with ThreadPoolExecutor(max_workers=_workers(len(files))) as ex:
        for got in ex.map(lambda path: test_rmstate.check_knowledge_isolation(path, binary), files):
            fails += got
    return fails


def run(binary=BINARY, out_dir=None, parts=None):
    """The tier; returns failure messages (empty = pass). `parts` (default PARTS) selects parts."""
    parts = list(parts or PARTS)
    out_dir = os.path.abspath(out_dir or os.path.join(_REPO_ROOT, "ci_out", "knowledge"))
    os.makedirs(out_dir, exist_ok=True)
    fails = []
    times = []
    for part in PARTS:
        if part not in parts:
            continue
        t0 = time.monotonic()
        print(f"knowledge: {part}", flush=True)
        if part == "isolation":
            fails += check_isolation(binary, out_dir)
        times.append((part, time.monotonic() - t0))
    print("knowledge: " + ", ".join(f"{p} {dt:.0f}s" for p, dt in times), flush=True)
    # The dumps and saved streams are large; once checked (pass or fail: the failure names the game
    # to replay) they go.
    for path in glob.glob(os.path.join(out_dir, "*.rms*")):
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
    args = ap.parse_args(argv)
    fails = run(args.binary, args.out_dir, args.part)
    for f in fails:
        print(f"  FAIL {f}", flush=True)
    print(f"knowledge: {'FAIL' if fails else 'OK'} ({len(fails)} failure(s))", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
