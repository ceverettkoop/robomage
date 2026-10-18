#!/usr/bin/env python3
"""The kept Python's own checks: the second step of ``robomage_check``'s ``smoke`` tier.

The smoke games test the scripted agent and the RMSTATE driver by playing; these checks test what
playing does not:

  decode    ``python/rmstate.py``'s decode of three ``--rmstate-dump`` files renders exactly the
            JSON lines the engine renders for them (``--rmstate-read DUMP --state-dump json``),
            byte for byte, and both equal the ``DUMP.jsonl`` the engine wrote live with
            ``--state-dump json``. The dumps: a bo1 game and a bo3 match of the engine's
            in-process explore agents, and a bo3 ``--machine-format rmstate`` session under
            ``--closed-decklist`` played by ``rm_driver.py``'s Python explore agents.
  refusals  each mutation of a dump's first STATE frame (another format version, another schema
            hash, an unknown frame kind, a known table at another row width) is refused by both
            decoders; an unknown table block is skipped by both, with the same JSON (listed under
            ``"skipped"``).
  driver    ``rm_driver.py`` derives the engine seeds it always has (pinned values), and a game it
            plays (scripted against explore) on an interpreter without site-packages (``-S``)
            imports no module from outside the standard library and this directory: the kept
            Python is stdlib only.

Stdlib only. Prints one ``  OK   …`` line per passing check and one ``FAIL: …`` line per finding;
exit 1 on any finding::

    python3 python/pycheck.py [--binary bin/debug/robomage] [--out-dir DIR] [--part decode,driver]
"""
from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import rm_driver  # noqa: E402
import rmstate  # noqa: E402

PARTS = ("decode", "refusals", "driver")
_TIMEOUT_S = 900
# The in-process agents' dumps: (deck_a, deck_b, bo3, game seed).
ENGINE_DUMPS = (("league/ur_delver", "league/gw_maverick", False, 41),
                ("league/reanimator", "league/uw_control", True, 42))
# The driver's closed-decklist session: (deck_a, deck_b, bo3, seed).
STREAM_DUMP = ("league/wrb_energy", "league/bw_dnt", True, 43)
# rm_driver.engine_seed's values (numpy's SeedSequence → PCG64 derivation, computed without numpy).
PINNED_ENGINE_SEEDS = ((0, 1826701614), (1, 1016164991), (2, 1798679647), (7, 2029167940),
                       (1001, 1940026651), (2147483647, 1751933748), (4294967295, 409763535),
                       (4294967296, 1121323793), (1099511627781, 30406389), (13532794167, 596834746),
                       (1641108517, 1187322509))
# Run in the probe after a driven game: every imported module with a file outside the standard
# library and the kept Python's directory (argv[2]).
_FOREIGN_MODULES = ("import os, sysconfig; "
                    "roots = {os.path.realpath(sysconfig.get_paths()[k]) for k in ('stdlib', 'platstdlib')}; "
                    "roots.add(os.path.realpath(sys.argv[2])); "
                    "files = {n: getattr(m, '__file__', None) for n, m in list(sys.modules.items())}; "
                    "print(' '.join(sorted(n for n, f in files.items() if f and not any("
                    "os.path.realpath(f).startswith(r + os.sep) for r in roots))))")


def _engine_json(binary, path):
    """(exit code, stdout, stderr) of the engine's --rmstate-read PATH --state-dump json."""
    r = subprocess.run([binary, "--rmstate-read", path, "--state-dump", "json"], cwd=rm_driver.BIN_DIR,
                       capture_output=True, timeout=_TIMEOUT_S)
    return r.returncode, r.stdout, r.stderr.decode(errors="replace")


def _py_json(path):
    with open(path, "rb") as fp:
        return b"".join(line.encode("utf-8", "surrogateescape") + b"\n" for line in rmstate.json_lines(fp))


def _first_diff_line(a, b):
    la, lb = a.split(b"\n"), b.split(b"\n")
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            return i + 1
    return min(len(la), len(lb)) + 1


def check_dump(binary, path, label):
    """The decode check on one dump; returns failure messages."""
    rc, cpp, err = _engine_json(binary, path)
    summary = [ln for ln in err.splitlines() if ln.startswith("RMSTATE_READ:")]
    fails = []
    if rc != 0 or "ERROR:" in err:
        errs = [ln for ln in err.splitlines() if "ERROR:" in ln]
        fails.append(f"{label}: --rmstate-read exit {rc}: {errs[:3]} {summary}")
    try:
        py = _py_json(path)
    except rmstate.RMStateError as exc:
        fails.append(f"{label}: the Python decoder refused the dump: {exc}")
        py = None
    if py is not None and py != cpp:
        fails.append(f"{label}: the Python JSON differs from the engine's at line {_first_diff_line(py, cpp)}")
    live = path + ".jsonl"
    if not os.path.exists(live):
        fails.append(f"{label}: no live --state-dump json file {live}")
    else:
        with open(live, "rb") as fp:
            if fp.read() != cpp:
                fails.append(f"{label}: the live --state-dump json file differs from --rmstate-read's JSON")
    if not fails:
        print(f"  OK   decode {label}: {summary[0] if summary else ''}", flush=True)
    return fails


def _engine_dump(binary, work, deck_a, deck_b, bo3, seed):
    """Plays one in-process explore game (match) with a dump; returns (dump path, failures)."""
    path = os.path.join(work, f"engine_{seed}.rms")
    cmd = [binary, "--agent-a", "explore", "--agent-b", "explore", "--game-seed", str(seed), "--deck-a", deck_a,
           "--deck-b", deck_b, "--rmstate-dump", path, "--state-dump", "json"]
    if bo3:
        cmd.append("--bo3")
    r = subprocess.run(cmd, cwd=rm_driver.BIN_DIR, capture_output=True, timeout=_TIMEOUT_S)
    if r.returncode != 0 or not os.path.exists(path):
        return None, [f"engine {deck_a} vs {deck_b} seed {seed}: exit {r.returncode}: "
                      f"{r.stderr.decode(errors='replace')[-300:]}"]
    return path, []


def _stream_dump(binary, work, deck_a, deck_b, bo3, seed):
    """Plays the driver's closed-decklist session with a dump; returns (dump path, failures)."""
    path = os.path.join(work, f"stream_{seed}.rms")
    cwd = os.getcwd()
    os.chdir(work)  # a draw's saved log stays in the scratch directory
    try:
        wld = rm_driver.run_games("explore", "explore", binary=binary, deck_a=deck_a, deck_b=deck_b, bo3=bo3,
                                  seed=seed, transcript="quiet", dump=path,
                                  engine_flags=["--closed-decklist", "--state-dump", "json"])
    finally:
        os.chdir(cwd)
    if not os.path.exists(path):
        return None, [f"driver {deck_a} vs {deck_b} seed {seed}: no dump written ({wld})"]
    return path, []


def check_decode(binary, work):
    """The decode part; returns (failures, the first dump for the refusals part)."""
    fails = []
    dumps = []
    for deck_a, deck_b, bo3, seed in ENGINE_DUMPS:
        path, f = _engine_dump(binary, work, deck_a, deck_b, bo3, seed)
        fails += f
        if path:
            dumps.append((path, f"engine {os.path.basename(deck_a)} vs {os.path.basename(deck_b)} "
                                f"{'bo3' if bo3 else 'bo1'} seed {seed}"))
    deck_a, deck_b, bo3, seed = STREAM_DUMP
    path, f = _stream_dump(binary, work, deck_a, deck_b, bo3, seed)
    fails += f
    if path:
        dumps.append((path, f"driver {os.path.basename(deck_a)} vs {os.path.basename(deck_b)} "
                            f"{'bo3' if bo3 else 'bo1'} --closed-decklist seed {seed}"))
    for path, label in dumps:
        fails += check_dump(binary, path, label)
    return fails, dumps[0][0] if dumps else None


# ── refusals: frames both decoders must refuse or skip ────────────────────────────────────────

def _rebuild(header, blocks):
    """A frame payload from (version, kind, viewer, hash) and [(id, width, rows, body)]."""
    version, kind, viewer, h = header
    out = bytearray(struct.pack("<4sHBBQII", b"RMSF", version, kind, viewer, h, len(blocks), 0))
    for tid, width, rows, body in blocks:
        out += struct.pack("<HHIII", tid, 0, width, rows, 0) + body
        out += b"\0" * ((-len(out)) % 8)
    return bytes(out)


def _mutants(state_payload):
    """(label, payload, what the engine reports, whether the decoders refuse it) per mutation of
    one STATE frame."""
    header, blocks = rmstate._blocks(state_payload)
    version, kind, viewer, h = header
    widen = next(i for i, b in enumerate(blocks) if b[0] != 1)
    tid, width, rows, body = blocks[widen]
    wide = b"".join(body[r * width:(r + 1) * width] + b"\0" for r in range(rows))
    wider = blocks[:widen] + [(tid, width + 1, rows, wide)] + blocks[widen + 1:]
    return [
        ("version", _rebuild((version + 1, kind, viewer, h), blocks), "format version", True),
        ("hash", _rebuild((version, kind, viewer, h ^ 1), blocks), "schema hash differs", True),
        ("kind", _rebuild((version, 200, viewer, h), blocks), "unknown frame kind", True),
        ("row width", _rebuild(header, wider), "has row width", True),
        ("unknown table", _rebuild(header, blocks + [(0xFFFE, 8, 1, b"\1" * 8)]),
         "blocks the decoding schema does not know", False),
    ]


def check_refusals(binary, dump_path, work):
    """Each mutation of the dump's first STATE frame against both decoders; returns failures."""
    with open(dump_path, "rb") as fp:
        payloads = list(rmstate.records(fp))
    dec = rmstate.StreamDecoder()
    head = []
    state = None
    for p in payloads:
        if dec.feed(p).kind == "STATE":
            state = p
            break
        head.append(p)
    if state is None:
        return [f"refusals: {os.path.basename(dump_path)} has no STATE frame to mutate"]
    fails = []
    mutants = _mutants(state)
    for label, mutant, needle, refused in mutants:
        path = os.path.join(work, f"mutant_{label.replace(' ', '_')}.rms")
        with open(path, "wb") as fp:
            for p in head + [mutant]:
                fp.write(b"RMSTATE %d\n" % len(p) + p)
        rc, out, err = _engine_json(binary, path)
        if rc != 1 or needle not in err:
            fails.append(f"mutant {label}: the engine should report {needle!r} (exit {rc}): {err[-300:]}")
        try:
            py = _py_json(path)
            if refused:
                fails.append(f"mutant {label}: the Python decoder accepted it")
            elif py != out or b'"skipped":[65534]' not in py:
                fails.append(f"mutant {label}: the Python and engine JSON differ, or the block is not listed as skipped")
        except rmstate.RMStateError:
            if not refused:
                fails.append(f"mutant {label}: the Python decoder refused it")
    if not fails:
        print(f"  OK   refusals ({len(mutants)} mutations)", flush=True)
    return fails


# ── the driver ────────────────────────────────────────────────────────────────────────────────

def check_driver(binary, work):
    """The driver part; returns failures."""
    fails = []
    for seed, want in PINNED_ENGINE_SEEDS:
        if rm_driver.engine_seed(seed) != want:
            fails.append(f"driver: engine_seed({seed}) = {rm_driver.engine_seed(seed)}, pinned {want}")
    probe = ("import sys; import rm_driver; rm_driver.run_games('scripted', 'explore', binary=sys.argv[1], "
             "deck_a='league/burn', seed=3, transcript='quiet'); " + _FOREIGN_MODULES)
    r = subprocess.run([sys.executable, "-S", "-c", probe, binary, _HERE], capture_output=True, text=True,
                       cwd=work, env=dict(os.environ, PYTHONPATH=_HERE), timeout=_TIMEOUT_S)
    if r.returncode != 0 or r.stdout.strip():
        fails.append(f"driver: a game imported {r.stdout.strip() or '?'} (exit {r.returncode}): {r.stderr[-300:]}")
    if not fails:
        print(f"  OK   driver: {len(PINNED_ENGINE_SEEDS)} engine seeds pinned, a game imports only the "
              "standard library and the kept Python", flush=True)
    return fails


def run(binary, out_dir=None, parts=PARTS):
    """The requested parts; returns the failure messages (empty = pass)."""
    binary = os.path.abspath(binary)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="pycheck_", dir=out_dir)
    fails = []
    try:
        first_dump = None
        if "decode" in parts or "refusals" in parts:
            f, first_dump = check_decode(binary, work) if "decode" in parts else ([], None)
            fails += f
            if "refusals" in parts:
                if first_dump is None:
                    path, f = _engine_dump(binary, work, *ENGINE_DUMPS[0])
                    fails += f
                    first_dump = path
                if first_dump:
                    fails += check_refusals(binary, first_dump, work)
        if "driver" in parts:
            fails += check_driver(binary, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pycheck", description="The kept Python's checks (robomage_check's "
                                 "smoke tier runs them).")
    ap.add_argument("--binary", default=rm_driver.DEFAULT_BINARY, help="the engine (default %(default)s)")
    ap.add_argument("--out-dir", default=None, help="the scratch directory's parent (default the system's)")
    ap.add_argument("--part", default=",".join(PARTS), help=f"comma list of {', '.join(PARTS)} (default all)")
    args = ap.parse_args(argv)
    parts = [p for p in args.part.split(",") if p]
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        ap.error(f"unknown part(s) {', '.join(unknown)}")
    fails = run(args.binary, args.out_dir, parts)
    for f in fails:
        print(f"FAIL: {f}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
