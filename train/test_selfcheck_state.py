#!/usr/bin/env python3
"""The ``rmstate`` gate tier: the state format's engine self-checks and decoders over real games.

Every engine runs with ``--selfcheck-state`` (``src/state_format/selfcheck.h``,
``docs/state_format.md`` §10), passed through ``$ROBOMAGE_ENGINE_FLAGS`` (``train/env.py``). A
self-check failure is an ``ERROR:`` line on the engine's stderr, and each engine prints one
``SELFCHECK-STATE: … errors=N`` summary line on stderr when its match ends. The tier runs:

  1. corpus     the replay corpus (``train/regression/replay_diff.py check``) with the self-checks
                on, ``--selfcheck-knowledge`` included (the knowledge sidecar against the truth; the
                ``knowledge`` tier, ``train/test_knowledge.py``, shares these runs): still
                byte-identical (the self-checks change no transcript), no ``ERROR:`` line, one
                ``errors=0`` ``SELFCHECK-STATE`` and one ``SELFCHECK-KNOWLEDGE`` summary per corpus
                game, and every ``SHADOW:`` line of the shadow comparison classified by
                ``train/regression/knowledge_divergences.json`` (``test_knowledge.shadow_log_failures``);
  2. fuzz       explore-vs-explore games (``runner.run_match``) over league pairs chosen for the
                state format's hard cases: face-down exile (reanimator), name-a-card menus
                (wrb_energy's Cabal Therapy, bw_dnt's and wrb_energy's Disruptor Flute), bo1 and
                bo3 (sideboarding frames), a few with every decision sampled
                (``--selfcheck-state-sample 1``). A finished game needs its ``errors=0`` summary;
                one stopped at the decision cap is scanned for ``ERROR:`` lines only;
  3. scenarios  every rules-regression scenario (``train/test_scenarios.py``) with
                ``--selfcheck-state --selfcheck-state-sample 1`` (a scenario stops at its decision
                cap, so only its ``ERROR:`` lines count; the scenario runner fails on them);
  4. decode     the Python-vs-C++ decode comparison (``train/test_rmstate.py``) on ``--rmstate-dump``
                files two of the fuzz games write, the decoders' refusals, and
                ``--machine-format rmstate`` sessions with the self-checks on, one of them under
                ``--closed-decklist`` (whose whole-vocab name-a-card menus only the RMSTATE stream
                can carry), and the name-a-card candidate sets with open and closed decklists
                (``test_rmstate.check_name_card_candidates``);
  5. isolation  the knowledge-side sources (``src/state_format/decode.*``, ``schema.*``, ``json.*``,
                ``recode.*`` and ``src/knowledge/*``) include only the standard library and the
                std-only headers listed in ``ALLOWED_PROJECT_HEADERS`` (step-1 handoff §3: the
                knowledge sidecar cannot reach engine state).

Entry points: ``run(binary=…, out_dir=…)`` returns the list of failure messages (empty = pass),
for ``train/ci_check.py``; ``main()`` is the CLI (exit 1 on failure)::

    train/.venv/bin/python train/test_selfcheck_state.py                 # everything
    train/.venv/bin/python train/test_selfcheck_state.py --part fuzz     # one part (repeatable)
    train/.venv/bin/python train/test_selfcheck_state.py --part fuzz --fuzz-seed 102   # one game

Engine stderr and the dumps go to ``--out-dir`` (default ``ci_out/rmstate``); the dumps are
deleted once checked. Torch-free; needs the engine (debug tier by default, ``--binary``).
"""
import argparse
import contextlib
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli_spec import BINARY  # noqa: E402
from env import ENGINE_FLAGS_ENV  # noqa: E402

_TRAIN = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TRAIN)

SELFCHECK = ["--selfcheck-state"]
SELFCHECK_EVERY_DECISION = ["--selfcheck-state", "--selfcheck-state-sample", "1"]
# The corpus part also checks the knowledge sidecar against the truth (--selfcheck-knowledge, which
# implies --selfcheck-state): the knowledge tier (train/test_knowledge.py) shares these engine runs.
SELFCHECK_CORPUS = ["--selfcheck-knowledge"]

# Fuzz games: (deck_a, deck_b, bo3, seed, extra engine flags, write an --rmstate-dump).
FUZZ_GAMES = [
    ("league/reanimator", "league/wrb_energy", False, 101, SELFCHECK, True),
    ("league/wrb_energy", "league/bw_dnt", True, 102, SELFCHECK, True),
    ("league/bw_dnt", "league/wubg_doomsday", False, 103, SELFCHECK, False),
    ("league/reanimator", "league/uw_control", True, 104, SELFCHECK, False),
    ("league/wubg_doomsday", "league/wrb_energy", False, 105, SELFCHECK, False),
    ("league/burn", "league/reanimator", False, 106, SELFCHECK, False),
    ("league/ur_delver", "league/gw_maverick", False, 107, SELFCHECK_EVERY_DECISION, False),
    ("league/lands", "league/bw_dnt", False, 108, SELFCHECK_EVERY_DECISION, False),
    ("league/bug", "league/reanimator", True, 109, SELFCHECK, False),
    ("league/wrb_energy", "league/lands", True, 110, SELFCHECK, False),
    ("league/uw_control", "league/bw_dnt", False, 111, SELFCHECK, False),
    ("league/gw_maverick", "league/wrb_energy", False, 112, SELFCHECK, False),
    ("league/reanimator", "league/reanimator", False, 113, SELFCHECK_EVERY_DECISION, False),
    ("league/wubg_doomsday", "league/burn", True, 114, SELFCHECK, False),
    ("league/bw_dnt", "league/ur_delver", True, 115, SELFCHECK, False),
    ("league/wrb_energy", "league/wrb_energy", False, 116, SELFCHECK_EVERY_DECISION, False),
]
FUZZ_MAX_DECISIONS = 2500

# --machine-format rmstate sessions (test_rmstate.run_stream): (deck_a, deck_b, bo3, seed, flags).
STREAM_GAMES = [
    ("league/ur_delver", "league/bug", False, 21, SELFCHECK + ["--narrative"]),
    ("league/wrb_energy", "league/bw_dnt", True, 23, SELFCHECK + ["--closed-decklist"]),
]

# The knowledge-side sources and what they may include. A quoted include must name one of
# ALLOWED_PROJECT_HEADERS, the file's own header, or (from src/knowledge/) a src/knowledge/ header;
# an angle-bracket include must be a standard library header.
ISOLATED_SOURCES = ["src/state_format/decode.*", "src/state_format/schema.*", "src/state_format/json.*",
                    "src/state_format/recode.*", "src/knowledge/*"]
ALLOWED_PROJECT_HEADERS = {"src/state_format/decode.h", "src/state_format/schema.h", "src/state_format/json.h"}
STD_HEADERS = {
    "algorithm", "any", "array", "atomic", "bitset", "cassert", "cctype", "cerrno", "cfloat", "charconv",
    "chrono", "cinttypes", "climits", "cmath", "complex", "cstdarg", "cstddef", "cstdint", "cstdio",
    "cstdlib", "cstring", "ctime", "deque", "functional", "initializer_list", "iomanip", "ios", "iosfwd",
    "iostream", "istream", "iterator", "limits", "list", "map", "memory", "mutex", "new", "numeric",
    "optional", "ostream", "queue", "random", "ratio", "set", "sstream", "stack", "stdexcept",
    "string", "string_view", "system_error", "tuple", "type_traits", "typeinfo", "unordered_map",
    "unordered_set", "utility", "variant", "vector",
}

PARTS = ["corpus", "fuzz", "scenarios", "decode", "isolation"]

_SUMMARY_RE = re.compile(r"^SELFCHECK-(STATE|KNOWLEDGE): decisions=\d+ .*\berrors=(\d+)\s*$")
_ERROR_RE = re.compile(r"^(ERROR|FATAL):|Traceback \(most recent call last\)|Segmentation fault|"
                       r"Assertion .* failed|\bAborted\b|core dumped")
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')


def _flags_env(flags):
    env = dict(os.environ)
    env[ENGINE_FLAGS_ENV] = " ".join([env.get(ENGINE_FLAGS_ENV, "")] + list(flags)).strip()
    return env


def scan_engine_log(path, label, summaries=None, kinds=("STATE",)):
    """Failure messages for one captured engine stderr: every ERROR: (or crash) line and every
    SELFCHECK-STATE / SELFCHECK-KNOWLEDGE summary without errors=0, and, with `summaries`, that
    exactly that many summary lines of each of `kinds` are there."""
    fails = []
    found = {kind: 0 for kind in kinds}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if _ERROR_RE.search(line):
                fails.append(f"{label}: {path}:{n}: {line}")
            m = _SUMMARY_RE.match(line)
            if m:
                if m.group(1) in found:
                    found[m.group(1)] += 1
                if m.group(2) != "0":
                    fails.append(f"{label}: {path}:{n}: {line}")
    for kind, count in found.items():
        if summaries is not None and count != summaries:
            fails.append(f"{label}: {count} SELFCHECK-{kind} summary line(s) in {path}, expected {summaries}")
    return fails[:20]


# ── 1. corpus ──────────────────────────────────────────────────────────────────────────────────

def check_corpus(out_dir):
    sys.path.insert(0, os.path.join(_TRAIN, "regression"))
    import replay_diff
    out_path = os.path.join(out_dir, "corpus.out")
    err_path = os.path.join(out_dir, "corpus.err")
    with open(out_path, "w") as out, open(err_path, "w") as err:
        r = subprocess.run([sys.executable, os.path.join(_TRAIN, "regression", "replay_diff.py"), "check"],
                           cwd=_REPO_ROOT, env=_flags_env(SELFCHECK_CORPUS), stdout=out, stderr=err)
    fails = []
    if r.returncode != 0:
        fails.append(f"corpus: the replay corpus drifts with {' '.join(SELFCHECK_CORPUS)} on (exit {r.returncode}; "
                     f"see {out_path})")
    fails += scan_engine_log(err_path, "corpus", summaries=len(replay_diff.SCENARIOS), kinds=("STATE", "KNOWLEDGE"))
    # The knowledge self-check's shadow comparison: every SHADOW line classified (the knowledge tier's
    # classifier; imported here, as that module imports this one).
    import test_knowledge
    fails += test_knowledge.shadow_log_failures(err_path, "corpus", summaries=len(replay_diff.SCENARIOS))
    if not fails:
        print(f"  OK   corpus: {len(replay_diff.SCENARIOS)} games byte-identical, self-checks clean "
              f"(state and knowledge, shadow mismatches classified)", flush=True)
    return fails


# ── 2. fuzz ────────────────────────────────────────────────────────────────────────────────────

def _fuzz_label(game):
    deck_a, deck_b, bo3, seed, flags, _dump = game
    sample = " sample=1" if "--selfcheck-state-sample" in flags else ""
    return f"{os.path.basename(deck_a)} vs {os.path.basename(deck_b)} {'bo3' if bo3 else 'bo1'} seed {seed}{sample}"


def _play_fuzz_game(task):
    """One fuzz game (a process-pool task): its engine's stderr goes to a file. It runs in a
    working directory of its own, so a drawn game's log (runner's draw_<stamp>.txt) is moved into
    `out_dir` under the game's name. Returns (stderr path, dump paths, whether every game of the
    match finished, crash message or None)."""
    import runner
    game, binary, out_dir = task
    deck_a, deck_b, bo3, seed, flags, dump = game
    stem = f"fuzz_{seed}"
    err_path = os.path.join(out_dir, stem + ".err")
    flags = list(flags)
    dump_dir = os.path.join(out_dir, stem + "_dumps")
    if dump:
        os.makedirs(dump_dir, exist_ok=True)
        flags += ["--rmstate-dump", os.path.join(dump_dir, "d_%p.rms")]
    os.environ[ENGINE_FLAGS_ENV] = " ".join(flags)
    work_dir = tempfile.mkdtemp(prefix=f".{stem}_", dir=out_dir)
    os.chdir(work_dir)
    crashed = None
    finished = False
    with open(err_path, "w") as err:
        saved = os.dup(2)
        os.dup2(err.fileno(), 2)
        try:
            with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):
                res = runner.run_match("explore", "explore", deck_a=deck_a, deck_b=deck_b, games=1, bo3=bo3,
                                       seed=seed, transcript="quiet", binary_path=binary,
                                       max_decisions=FUZZ_MAX_DECISIONS)
            finished = bool(res.records) and not any(rec.capped for rec in res.records)
        except Exception as exc:  # an engine crash: nonzero exit / EOF mid-game
            crashed = str(exc) or type(exc).__name__
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            os.chdir(_REPO_ROOT)
            for log in glob.glob(os.path.join(work_dir, "draw_*.txt")):
                shutil.move(log, os.path.join(out_dir, f"{stem}_{os.path.basename(log)}"))
            shutil.rmtree(work_dir, ignore_errors=True)
    dumps = sorted(glob.glob(os.path.join(dump_dir, "d_*.rms"))) if dump else []
    return err_path, dumps, finished, crashed


def play_fuzz(binary, out_dir, seeds=None):
    """Plays FUZZ_GAMES (those with a seed in `seeds`, if given) in parallel; returns (failure
    messages, dump paths)."""
    games = [g for g in FUZZ_GAMES if not seeds or g[3] in seeds]
    tasks = [(g, binary, out_dir) for g in games]
    if not tasks:
        return [f"fuzz: no game has a seed in {sorted(seeds)}"], []
    with ProcessPoolExecutor(max_workers=max(1, min(len(tasks), os.cpu_count() or 4))) as ex:
        results = list(ex.map(_play_fuzz_game, tasks))
    fails, dumps = [], []
    for game, (err_path, game_dumps, finished, crashed) in zip(games, results):
        label = "fuzz " + _fuzz_label(game)
        got = []
        if crashed is not None:
            got.append(f"{label}: engine crashed: {crashed} (see {err_path})")
        got += scan_engine_log(err_path, label, summaries=1 if finished and crashed is None else None)
        if game[5] and not game_dumps:
            got.append(f"{label}: no --rmstate-dump file was written")
        if not got:
            print(f"  OK   {label}{'' if finished else ' (stopped at the decision cap)'}", flush=True)
        fails += got
        dumps += game_dumps
    return fails, dumps


# ── 3. scenarios ───────────────────────────────────────────────────────────────────────────────

def check_scenarios(binary, out_dir):
    out_path = os.path.join(out_dir, "scenarios.out")
    cmd = [sys.executable, os.path.join(_TRAIN, "test_scenarios.py"), "--binary", binary]
    with open(out_path, "w") as out:
        r = subprocess.run(cmd, cwd=_REPO_ROOT, env=_flags_env(SELFCHECK_EVERY_DECISION), stdout=out,
                           stderr=subprocess.STDOUT)
    with open(out_path, encoding="utf-8", errors="replace") as fh:
        tail = [ln.rstrip("\n") for ln in fh][-1:]
    if r.returncode != 0:
        return [f"scenarios: test_scenarios.py with {' '.join(SELFCHECK_EVERY_DECISION)} exit {r.returncode} "
                f"(see {out_path})"]
    print(f"  OK   {tail[0] if tail else 'scenarios'}", flush=True)
    return []


# ── 4. decode ──────────────────────────────────────────────────────────────────────────────────

def check_decode(binary, out_dir, dumps):
    import test_rmstate
    fails = []
    if not dumps:
        fails.append("decode: no --rmstate-dump file to check")
    for path in dumps:
        fails += test_rmstate.check_dump(path, binary)
    if dumps:
        fails += test_rmstate.check_refusals(dumps[0], binary, out_dir)
    fails += test_rmstate.stream_checks(binary, out_dir, STREAM_GAMES)
    fails += test_rmstate.check_name_card_candidates(binary, out_dir)
    for _deck_a, _deck_b, _bo3, seed, _flags in STREAM_GAMES:
        err = os.path.join(out_dir, f"stream_{seed}.err")
        if os.path.exists(err):
            fails += scan_engine_log(err, f"stream seed {seed}", summaries=1)
    return fails


# ── 5. isolation ───────────────────────────────────────────────────────────────────────────────

def check_isolation():
    fails = []
    files = sorted({p for pat in ISOLATED_SOURCES for p in glob.glob(os.path.join(_REPO_ROOT, pat))
                    if p.endswith((".h", ".cpp"))})
    for path in files:
        rel = os.path.relpath(path, _REPO_ROOT)
        own = os.path.splitext(rel)[0] + ".h"
        in_knowledge = rel.startswith("src/knowledge/")
        with open(path, encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh, 1):
                m = _INCLUDE_RE.match(line)
                if not m:
                    continue
                kind, name = m.groups()
                if kind == "<":
                    if name not in STD_HEADERS:
                        fails.append(f"isolation: {rel}:{n} includes <{name}>, not a standard library header "
                                     "this check allows (STD_HEADERS)")
                    continue
                target = os.path.normpath(os.path.join(os.path.dirname(rel), name))
                allowed = (target in ALLOWED_PROJECT_HEADERS or target == own or
                           (in_knowledge and target.startswith("src/knowledge/") and target.endswith(".h")))
                if not allowed:
                    fails.append(f'isolation: {rel}:{n} includes "{name}" ({target}): only the standard library, '
                                 f"its own header and {sorted(ALLOWED_PROJECT_HEADERS)} are allowed")
    if not files:
        fails.append("isolation: no source matched " + ", ".join(ISOLATED_SOURCES))
    if not fails:
        print(f"  OK   isolation: {len(files)} file(s) include only the standard library and std-only headers",
              flush=True)
    return fails


def run(binary=BINARY, out_dir=None, parts=None, fuzz_seeds=None):
    """The tier; returns failure messages (empty = pass). `parts` (default PARTS) selects parts;
    the decode part reuses the fuzz part's dumps (and plays the fuzz games if fuzz is not run).
    `fuzz_seeds` limits the fuzz part to the games with those seeds."""
    parts = list(parts or PARTS)
    out_dir = os.path.abspath(out_dir or os.path.join(_REPO_ROOT, "ci_out", "rmstate"))
    os.makedirs(out_dir, exist_ok=True)
    fails = []
    dumps = []
    times = []
    for part in PARTS:
        if part not in parts and not (part == "fuzz" and "decode" in parts):
            continue
        t0 = time.monotonic()
        print(f"rmstate: {part}", flush=True)
        if part == "corpus":
            fails += check_corpus(out_dir)
        elif part == "fuzz":
            got, dumps = play_fuzz(binary, out_dir, fuzz_seeds)
            fails += got
        elif part == "scenarios":
            fails += check_scenarios(binary, out_dir)
        elif part == "decode":
            fails += check_decode(binary, out_dir, dumps)
        elif part == "isolation":
            fails += check_isolation()
        times.append((part, time.monotonic() - t0))
    print("rmstate: " + ", ".join(f"{p} {dt:.0f}s" for p, dt in times), flush=True)
    # The dumps are large; once checked (pass or fail: the failure names the game to replay) they go.
    for path in glob.glob(os.path.join(out_dir, "fuzz_*_dumps")) + glob.glob(os.path.join(out_dir, "*.rms*")):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--binary", default=BINARY, help="engine binary (default: the debug tier)")
    ap.add_argument("--out-dir", help="engine stderr and dumps (default ci_out/rmstate)")
    ap.add_argument("--part", action="append", choices=PARTS, help="run only this part (repeatable)")
    ap.add_argument("--fuzz-seed", type=int, action="append",
                    help="play only the fuzz game with this seed (repeatable; its label names it)")
    args = ap.parse_args(argv)
    fails = run(args.binary, args.out_dir, args.part, args.fuzz_seed)
    for f in fails:
        print(f"  FAIL {f}", flush=True)
    print(f"rmstate: {'FAIL' if fails else 'OK'} ({len(fails)} failure(s))", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
