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
                pass and print a ``SHADOW:`` line that signature classifies;
  6. archetype  archetype inference (``docs/knowledge_store.md`` §10): ``--machine-format rmstate``
                sessions (``ARCHETYPE_GAMES``) whose METAGAME frame must list every deck
                ``archetypes.json`` tags, with the labels of ``archetypes.ARCHETYPES`` plus unknown;
                at every seat frame the true registered list is consistent and no other archetype
                is exact; a list the seat sees (its own, the other's with open decklists) is exactly
                consistent and classified registered from the first frame; under
                ``--closed-decklist`` the other seat's hidden list is classified as nothing, before
                any of its cards is seen every label of a list that can hold the library and
                sideboard the game started with is possible, no consistent list lacks a
                card it was seen to own, the archetype is exact once the cards seen leave one
                label, and it becomes exact during the session;
  7. belief     the belief layer (``docs/knowledge_store.md`` §10.6, level ``expert+belief``):
                structure (``BELIEF_STREAM_GAMES``, with a dump and ``--selfcheck-knowledge``: stream
                frames equal the dump's, ``b_*`` included; ``rmknow --isolation --level
                expert+belief`` on each writer's frames; ``rmknow --truth`` on the dump, which checks
                that a list has probability zero exactly when ``k_template`` excludes it and that
                each player's probabilities sum to the full scale; the self-check's snapshot
                restores); calibration (``BELIEF_CALIBRATION_GAMES``: ``--closed-decklist``
                sessions with every reference list as some seat's opponent, the probability of the
                true list and archetype at every seat frame: mean log-loss must beat the uniform
                prior's, and the true archetype's mean probability must be higher at each game's last
                frame than at its first; reliability bins and Brier scores are printed); and
                ``rmknow --belief-exactness BELIEF_EXACTNESS_N`` (the posterior against the weighted
                enumeration of the hypothesis networks and the model played out directly).

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

# Archetype-inference sessions (--machine-format rmstate, no dump): (deck_a, deck_b, bo3, seed,
# extra engine flags). Under --closed-decklist each seat's view of the other's reference lists
# narrows as cards are seen; with open decklists it is exact from the first frame.
ARCHETYPE_GAMES = [
    ("league/bug", "delver", False, 41, ["--closed-decklist"]),
    ("league/ur_delver", "league/lands", True, 42, ["--closed-decklist"]),
    ("league/burn", "league/bw_dnt", False, 43, ["--closed-decklist"]),
    ("league/uw_control", "league/reanimator", False, 44, []),
]

# Belief-layer sessions with a dump (deck_a, deck_b, bo3, seed, extra engine flags): both seats at
# expert+belief, one under --closed-decklist (the posterior), one with open decklists (explicit).
BELIEF_LEVELS = ["--knowledge-a", "expert+belief", "--knowledge-b", "expert+belief"]
BELIEF_STREAM_GAMES = [
    ("league/reanimator", "league/ur_delver", False, 51, BELIEF_LEVELS + SELFCHECK_KNOWLEDGE + ["--closed-decklist"]),
    ("league/burn", "mav", False, 52, BELIEF_LEVELS),
]
# Calibration sessions: every reference list (archetypes.json's tagged decks, sorted) against the one
# CALIBRATION_STRIDE places on, bo1 under --closed-decklist; both seats' views count.
CALIBRATION_STRIDE = 5
CALIBRATION_SEED = 61
BELIEF_EXACTNESS_N = 10

PARTS = ["selftest", "fuzz", "isolation", "truth", "shadow", "archetype", "belief"]

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


# ── 6. archetype ───────────────────────────────────────────────────────────────────────────────

def _metagame_of(payloads):
    """(list names, list labels, list counts {name: Counter of card → main + side}, archetype labels,
    list sizes [(main, side)]) from the first METAGAME frame of a stream."""
    dec = rmstate.StreamDecoder()
    for p in payloads:
        f = dec.feed(p)
        if f.kind != "METAGAME":
            continue
        labels = [r["name"] for r in f.tables.get("meta_archetype", [])]
        lists = f.tables.get("meta_list", [])
        names = [r["name"] for r in lists]
        counts = {n: collections.Counter() for n in names}
        for r in f.tables.get("meta_list_card", []):
            counts[names[r["list"]]][r["card"]] += r["main"] + r["side"]
        return names, [labels[r["archetype"]] for r in lists], counts, labels, [(r["main"], r["side"]) for r in lists]
    return None


def check_metagame(payloads, label):
    """The METAGAME frame lists every deck archetypes.json tags (all exist and are implemented), with
    its label, and the labels are ARCHETYPES plus unknown in index order."""
    import archetypes
    meta = _metagame_of(payloads)
    if meta is None:
        return [f"{label}: no METAGAME frame"]
    names, list_labels, _counts, labels, _sizes = meta
    fails = []
    if labels != archetypes.ARCHETYPES + [archetypes.UNKNOWN_NAME]:
        fails.append(f"{label}: METAGAME labels {labels}, archetypes.py has {archetypes.ARCHETYPES} + unknown")
    tagged = archetypes.mapping()
    missing = sorted(set(tagged) - set(names))
    if missing:
        fails.append(f"{label}: tagged deck(s) {missing} are not reference lists (missing .dk or unimplemented cards)")
    for n, lab in zip(names, list_labels):
        if tagged.get(n) != lab:
            fails.append(f"{label}: reference list {n} has label {lab}, archetypes.json {tagged.get(n)}")
    return fails


def _archetype_session(task):
    """One archetype session (a process-pool task): (failure messages, summary)."""
    game, binary, out_dir = task
    deck_a, deck_b, bo3, seed, flags = game
    closed = "--closed-decklist" in flags
    label = (f"archetype {os.path.basename(deck_a)} vs {os.path.basename(deck_b)} {'bo3' if bo3 else 'bo1'} "
             f"seed {seed}{' closed' if closed else ''}")
    err = os.path.join(out_dir, f"archetype_{seed}.err")
    try:
        payloads = test_rmstate.run_stream(binary, deck_a, deck_b, bo3, seed, err_path=err, flags=flags)
    except (rmstate.RMStateError, subprocess.TimeoutExpired) as exc:
        return [f"{label}: {exc} (see {err})"], None
    fails = test_selfcheck_state.scan_engine_log(err, label)
    fails += check_metagame(payloads, label)
    meta = _metagame_of(payloads)
    if meta is None:
        return fails, None
    names, list_labels, counts, labels, sizes = meta
    deck = {"A": deck_a, "B": deck_b}
    started = {}  # seat → the other seat's (library, sideboard) at its latest game_start
    dec = rmstate.StreamDecoder()
    frames = {"A": 0, "B": 0}
    exact_at = {}
    first_possible = {}
    game_start = None
    unproven = 0
    for p in payloads:
        f = dec.feed(p)
        if game_start is None:
            game_start = dec.schema.enums["event_kind"].index("GAME_START")
        if f.kind != "STATE" or f.viewer not in deck:
            continue
        seat, other = f.viewer, "B" if f.viewer == "A" else "A"
        frames[seat] += 1
        for r in f.tables.get("event_player", []):
            if r["player_rel"] == 2 and f.tables["event"][r["event"]]["kind"] == game_start:
                started[seat] = (r["library"], r["sideboard"])
        where = f"{label}: {seat} frame {frames[seat]}"
        rows = {(r["player_rel"], r["list"]): r for r in f.tables.get("k_template", [])}
        arch = {(r["player_rel"], r["archetype"]): r for r in f.tables.get("k_archetype", [])}
        if len(rows) != 2 * len(names) or len(arch) != 2 * len(labels):
            fails.append(f"{where}: {len(rows)} k_template / {len(arch)} k_archetype rows")
            continue
        unproven += sum(1 for r in rows.values() if not r["exact"])
        for rel, who in ((1, seat), (2, other)):
            true_list = names.index(deck[who]) if deck[who] in names else None
            true_label = labels.index(list_labels[true_list]) if true_list is not None else None
            consistent = [lst for lst in range(len(names)) if rows[(rel, lst)]["consistent"]]
            registered = [lst for lst in range(len(names)) if rows[(rel, lst)]["registered"]]
            if true_list is not None and true_list not in consistent:
                fails.append(f"{where}: {who}'s true list {deck[who]} is reported inconsistent")
            exact = [a for a in range(len(labels)) if arch[(rel, a)]["exact"]]
            if exact and exact != [true_label]:
                fails.append(f"{where}: {who}'s archetype reported exactly {labels[exact[0]]}")
            visible = rel == 1 or not closed
            if visible:
                # The list is explicit: exactly the true list, classified as itself, from the first frame.
                if consistent != [true_list] or registered != [true_list] or exact != [true_label]:
                    fails.append(f"{where}: {who}'s open list: consistent {consistent}, registered {registered}, "
                                 f"exact archetype {exact}")
                continue
            if registered:
                fails.append(f"{where}: {who}'s hidden list is classified as {registered}")
            # At least as sharp as counting the cards seen: no consistent list lacks a card the
            # opponent was seen to own, and when every list holding them shares a label, it is exact.
            seen = collections.Counter({r["card"]: r["match"] for r in f.tables.get("k_seen", [])
                                        if r["player_rel"] == 2 and r["match"] > 0})
            fits = [lst for lst in range(len(names)) if all(counts[names[lst]][c] >= n for c, n in seen.items())]
            if set(consistent) - set(fits):
                fails.append(f"{where}: lists {sorted(set(consistent) - set(fits))} are consistent but lack a seen card")
            fit_labels = {list_labels[lst] for lst in fits}
            if len(fit_labels) == 1 and true_list in fits and exact != [true_label]:
                fails.append(f"{where}: the cards seen leave only {fit_labels}, not reported exact")
            if frames[seat] == 1 and not seen:
                # Before any card is seen, every label of a list that can hold the library and
                # sideboard the game started with.
                lib, side = started.get(seat, (0, 0))
                want = {list_labels[lst] for lst in range(len(names)) if sizes[lst][0] >= lib and sizes[lst][1] >= side}
                got = {labels[a] for a in range(len(labels)) if arch[(rel, a)]["possible"]}
                if got != want:
                    fails.append(f"{where}: before any card of {who} is seen, possible {sorted(got)}, "
                                 f"the list sizes allow {sorted(want)}")
                first_possible[who] = len(got)
            if exact and who not in exact_at:
                exact_at[who] = (frames[seat], sorted(seen))
    if closed:
        for who in ("A", "B"):
            if who not in exact_at:
                fails.append(f"{label}: {who}'s archetype never became exact for the other seat")
    summary = None
    if not fails:
        when = "; ".join(f"{who}: {first_possible.get(who, '?')} labels possible at first, exact at the other "
                         f"seat's frame {fr}" for who, (fr, _s) in sorted(exact_at.items()))
        summary = (f"{label}: {sum(frames.values())} seat frames, {unproven} unproven row(s)"
                   f"{'; ' + when if when else ''}")
    return fails, summary


def check_archetypes(binary, out_dir):
    tasks = [(g, binary, out_dir) for g in ARCHETYPE_GAMES]
    fails = []
    with ProcessPoolExecutor(max_workers=_workers(len(tasks))) as ex:
        for got, summary in ex.map(_archetype_session, tasks):
            fails += got
            if summary:
                print(f"  OK   {summary}", flush=True)
    return fails


# ── 7. belief ──────────────────────────────────────────────────────────────────────────────────

def _belief_stream_session(task):
    """One belief structure session with its dump (a process-pool task): failure messages."""
    game, binary, out_dir = task
    deck_a, deck_b, bo3, seed, flags = game
    label = (f"belief {os.path.basename(deck_a)} vs {os.path.basename(deck_b)} {'bo3' if bo3 else 'bo1'} seed {seed}"
             f"{' closed' if '--closed-decklist' in flags else ''}")
    dump = os.path.join(out_dir, f"belief_{seed}.rms")
    err = os.path.join(out_dir, f"belief_{seed}.err")
    frames = []
    try:
        payloads = test_rmstate.run_stream(binary, deck_a, deck_b, bo3, seed, dump=dump, err_path=err, flags=flags,
                                           frames=frames)
    except (rmstate.RMStateError, subprocess.TimeoutExpired) as exc:
        return [f"{label}: {exc} (see {err})"]
    fails = test_rmstate.check_stream(payloads, dump, label)
    saved = os.path.join(out_dir, f"belief_{seed}.stream.rms")
    test_rmstate.write_stream(payloads, saved)
    selfcheck = "--selfcheck-knowledge" in flags
    fails += test_selfcheck_state.scan_engine_log(err, label, summaries=1 if selfcheck else None,
                                                  kinds=SUMMARY_KINDS if selfcheck else ())
    levels = ("expert+belief", "expert+belief")
    for path in (dump, saved):
        fails += test_rmstate.check_knowledge_isolation(path, binary, levels)
    for seat in ("A", "B"):
        r = subprocess.run([_rmknow(binary), "--viewer", seat, "--level", "expert+belief", "--truth", "--quiet",
                            "--in", dump], capture_output=True, text=True, timeout=_TIMEOUT_S)
        summary = [ln for ln in r.stderr.splitlines() if ln.startswith("RMKNOW:")]
        if r.returncode != 0 or not summary or " truth_frames=0 " in summary[0]:
            bad = [ln for ln in r.stderr.splitlines() if "ERROR:" in ln]
            fails.append(f"{label}: rmknow --truth viewer {seat} exit {r.returncode}: {bad[:3]} {summary}")
    rows = sum(len(f.tables.get("b_list", [])) for f in frames if f.kind == "STATE")
    if not rows:
        fails.append(f"{label}: no b_list rows at level expert+belief")
    if not fails:
        print(f"  OK   {label}: {len(payloads)} records, {rows} b_list rows; stream frames equal the dump's, "
              f"isolation and truth hold{'; ' + _knowledge_summary(err) if selfcheck else ''}", flush=True)
    return fails


def calibration_games(stride=CALIBRATION_STRIDE, seed=CALIBRATION_SEED, bo3=False):
    """The calibration sessions: every tagged deck against the one `stride` places on (sorted)."""
    import archetypes
    decks = sorted(archetypes.mapping())
    return [(decks[i], decks[(i + stride) % len(decks)], bo3, seed + i,
             BELIEF_LEVELS + ["--closed-decklist"]) for i in range(len(decks))]


def _calibration_session(task):
    """One calibration session (a process-pool task): (failure messages, records). A record is one
    seat frame of a live game: (session label, seat, game, the opponent's list probabilities as
    fractions, the opponent's true list, the list labels, every row exact)."""
    game, binary, out_dir = task
    deck_a, deck_b, bo3, seed, flags = game
    label = f"calibration {deck_a} vs {deck_b} seed {seed}"
    err = os.path.join(out_dir, f"calibration_{seed}.err")
    frames = []
    try:
        test_rmstate.run_stream(binary, deck_a, deck_b, bo3, seed, err_path=err, flags=flags, frames=frames)
    except (rmstate.RMStateError, subprocess.TimeoutExpired) as exc:
        return [f"{label}: {exc} (see {err})"], []
    fails = test_selfcheck_state.scan_engine_log(err, label)
    names, labels = None, None
    deck = {"A": deck_a, "B": deck_b}
    records = []
    for f in frames:
        if f.kind == "METAGAME" and names is None:
            arch = [r["name"] for r in f.tables.get("meta_archetype", [])]
            names = [r["name"] for r in f.tables["meta_list"]]
            labels = [arch[r["archetype"]] for r in f.tables["meta_list"]]
        if f.kind != "STATE" or f.viewer not in deck or names is None:
            continue
        h = f.tables["header"][0]
        if h["terminal"] or h["sideboarding"]:
            continue
        rows = sorted((r for r in f.tables.get("b_list", []) if r["player_rel"] == 2), key=lambda r: r["list"])
        if len(rows) != len(names):
            continue
        other = "B" if f.viewer == "A" else "A"
        if deck[other] not in names:
            continue
        probs = [r["prob"] / 1e9 for r in rows]
        records.append((label, f.viewer, h["game"], probs, names.index(deck[other]), labels,
                        all(r["exact"] for r in rows if r["prob"])))
    if not records:
        fails.append(f"{label}: no seat frame with b_list rows")
    return fails, records


def calibration_report(records, bins=10):
    """Calibration figures of `records` (_calibration_session's): a dict of the means and the
    reliability bins (predicted probability of each list against how often it is the true one)."""
    import math
    eps = 1e-9
    n = len(records)
    out = {"frames": n}
    if not n:
        return out
    n_lists = len(records[0][3])
    ll = ll_cons = ll_arch = ll_arch_uniform = brier = brier_arch = 0.0
    amb = amb_ll = amb_cons = 0.0  # frames with more than one consistent list
    rel = [[0, 0.0, 0] for _ in range(bins)]  # count, sum of p, true count
    exact = 0
    games = collections.OrderedDict()
    for label, seat, game, probs, true, labels, ex in records:
        exact += 1 if ex else 0
        p_true = probs[true]
        support = sum(1 for p in probs if p > 0)
        ll -= math.log(max(p_true, eps))
        ll_cons -= math.log(1.0 / support)
        if support > 1:
            amb += 1
            amb_ll -= math.log(max(p_true, eps))
            amb_cons -= math.log(1.0 / support)
        arch = collections.defaultdict(float)
        for p, lab in zip(probs, labels):
            arch[lab] += p
        tl = labels[true]
        ll_arch -= math.log(max(arch[tl], eps))
        ll_arch_uniform -= math.log(labels.count(tl) / len(labels))
        brier += sum((p - (1.0 if i == true else 0.0)) ** 2 for i, p in enumerate(probs))
        brier_arch += sum((arch.get(a, 0.0) - (1.0 if a == tl else 0.0)) ** 2 for a in set(labels))
        for i, p in enumerate(probs):
            b = min(bins - 1, int(p * bins))
            rel[b][0] += 1
            rel[b][1] += p
            rel[b][2] += 1 if i == true else 0
        games.setdefault((label, seat, game), []).append((p_true, arch[tl]))
    total = n * n_lists
    out.update({
        "exact_frames": exact,
        "logloss_list": ll / n, "logloss_list_uniform": math.log(n_lists), "logloss_list_consistent_uniform": ll_cons / n,
        "logloss_arch": ll_arch / n, "logloss_arch_uniform": ll_arch_uniform / n,
        "brier_list": brier / n, "brier_arch": brier_arch / n,
        "ambiguous_frames": int(amb), "ambiguous_logloss": amb_ll / amb if amb else 0.0,
        "ambiguous_logloss_consistent_uniform": amb_cons / amb if amb else 0.0,
        "first_true_list": sum(v[0][0] for v in games.values()) / len(games),
        "last_true_list": sum(v[-1][0] for v in games.values()) / len(games),
        "first_true_arch": sum(v[0][1] for v in games.values()) / len(games),
        "last_true_arch": sum(v[-1][1] for v in games.values()) / len(games),
        "games": len(games),
        "reliability": [(i / bins, (i + 1) / bins, c, (sp / c if c else 0.0), (t / c if c else 0.0))
                        for i, (c, sp, t) in enumerate(rel)],
        "ece": sum(abs(sp - t) for c, sp, t in rel) / total,
    })
    return out


def format_calibration(rep):
    lines = [f"{rep['frames']} seat frames ({rep.get('exact_frames', 0)} exact), {rep.get('games', 0)} games"]
    if rep["frames"]:
        lines.append(f"list log-loss {rep['logloss_list']:.3f} (uniform {rep['logloss_list_uniform']:.3f}, uniform "
                     f"over the consistent lists {rep['logloss_list_consistent_uniform']:.3f}); archetype log-loss "
                     f"{rep['logloss_arch']:.3f} (uniform prior {rep['logloss_arch_uniform']:.3f}); Brier list "
                     f"{rep['brier_list']:.3f}, archetype {rep['brier_arch']:.3f}; ECE {rep['ece']:.4f}")
        lines.append(f"{rep['ambiguous_frames']} frames with several consistent lists: list log-loss "
                     f"{rep['ambiguous_logloss']:.3f}, uniform over the consistent lists "
                     f"{rep['ambiguous_logloss_consistent_uniform']:.3f}")
        lines.append(f"true list p first/last frame of a game {rep['first_true_list']:.3f}/{rep['last_true_list']:.3f};"
                     f" true archetype {rep['first_true_arch']:.3f}/{rep['last_true_arch']:.3f}")
        lines.append("reliability (bin: pairs, mean predicted, frequency true): " + "; ".join(
            f"[{lo:.1f},{hi:.1f}) {c} {mp:.3f} {fr:.3f}" for lo, hi, c, mp, fr in rep["reliability"] if c))
    return lines


def calibration_failures(rep):
    """The loose calibration gate (the opponent's play is not random, so some miscalibration is
    expected): better than the uniform prior on average, and sharper toward the end of a game."""
    fails = []
    if not rep["frames"]:
        return ["calibration: no frame"]
    if not rep["logloss_list"] < rep["logloss_list_uniform"]:
        fails.append(f"calibration: list log-loss {rep['logloss_list']:.3f} is not below the uniform prior's "
                     f"{rep['logloss_list_uniform']:.3f}")
    if not rep["logloss_arch"] < rep["logloss_arch_uniform"]:
        fails.append(f"calibration: archetype log-loss {rep['logloss_arch']:.3f} is not below the uniform prior's "
                     f"{rep['logloss_arch_uniform']:.3f}")
    if not rep["last_true_arch"] > rep["first_true_arch"]:
        fails.append(f"calibration: the true archetype's probability does not rise over a game "
                     f"({rep['first_true_arch']:.3f} at the first frame, {rep['last_true_arch']:.3f} at the last)")
    return fails


def check_belief(binary, out_dir, games=None):
    fails = []
    tasks = [(g, binary, out_dir) for g in BELIEF_STREAM_GAMES]
    cal = [(g, binary, out_dir) for g in (games or calibration_games())]
    with ProcessPoolExecutor(max_workers=_workers(len(tasks) + len(cal))) as ex:
        structure = [ex.submit(_belief_stream_session, t) for t in tasks]
        calibration = [ex.submit(_calibration_session, t) for t in cal]
        records = []
        for fut in calibration:
            got, recs = fut.result()
            fails += got
            records += recs
        for fut in structure:
            fails += fut.result()
    rep = calibration_report(records)
    for line in format_calibration(rep):
        print(f"  {'OK  ' if not calibration_failures(rep) else 'FAIL'} calibration: {line}", flush=True)
    fails += calibration_failures(rep)
    label = f"rmknow --belief-exactness {BELIEF_EXACTNESS_N}"
    r = subprocess.run([_rmknow(binary), "--belief-exactness", str(BELIEF_EXACTNESS_N)], capture_output=True,
                       text=True, timeout=_TIMEOUT_S)
    lines = (r.stdout + r.stderr).splitlines()
    last = [ln for ln in lines if ln.startswith("RMKNOW_BELIEF_EXACTNESS:")]
    bad = [ln for ln in lines if "ERROR" in ln]
    if r.returncode != 0 or not last or bad:
        fails.append(f"{label}: exit {r.returncode}: {(bad or lines)[:3]}")
    else:
        print(f"  OK   {label}: {last[-1]}", flush=True)
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
        elif part == "archetype":
            fails += check_archetypes(binary, out_dir)
        elif part == "belief":
            fails += check_belief(binary, out_dir)
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
    ap.add_argument("--calibration", type=int, metavar="STRIDES",
                    help="only run calibration sessions: every reference list against the lists 1..STRIDES "
                         "places on (bo1 and bo3), and print the calibration report")
    ap.add_argument("--classify", nargs="+", metavar="FILE",
                    help="only classify the SHADOW lines of these engine logs (counts per signature)")
    args = ap.parse_args(argv)
    if args.calibration:
        out_dir = os.path.abspath(args.out_dir or os.path.join(_REPO_ROOT, "ci_out", "knowledge"))
        os.makedirs(out_dir, exist_ok=True)
        games = []
        for stride in range(1, args.calibration + 1):
            for bo3 in (False, True):
                games += calibration_games(stride, CALIBRATION_SEED + 100 * stride + (50 if bo3 else 0), bo3)
        tasks = [(g, args.binary, out_dir) for g in games]
        records, fails = [], []
        with ProcessPoolExecutor(max_workers=_workers(len(tasks))) as ex:
            for got, recs in ex.map(_calibration_session, tasks):
                fails += got
                records += recs
        rep = calibration_report(records)
        for line in format_calibration(rep):
            print(f"  {line}", flush=True)
        by_game = collections.defaultdict(list)
        for rec in records:
            by_game[min(rec[2], 2)].append(rec)
        for g, recs in sorted(by_game.items()):
            for line in format_calibration(calibration_report(recs)):
                print(f"  game {g + 1}: {line}", flush=True)
        fails += calibration_failures(rep)
    elif args.classify:
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
