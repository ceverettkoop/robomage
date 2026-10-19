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
  selfkill  the scripted agent's self-lethal guard (``scripted_core`` docstring) in sculpted
            positions: at 2 life it does not cast a Price of Progress that would deal it 2 (also
            when it would kill the opponent too, while the opponent's board has it dead next
            turn), nor a Dismember whose Phyrexian pip only life can pay; at 3 life it casts the
            same Price of Progress (the position does tempt it).
  rmtb      ``python/rmtb.py`` reads the RMTB files the C++ writer wrote (``src/featurize/rmtb.cpp``;
            ``robomage_check``'s smoke tier writes two samples, an array of every dtype and a text
            table, and their meta and values, ``--rmtb-samples FILE``) to exactly those values,
            floats bit for bit, the f16 halves equal to ``struct``'s rounding of their sources; and
            the committed text table, when there is one, has zero rows outside its rows_meta.
  featurize ``python/rmtb.py`` reads the featurizer's files ``robomage_check``'s featurize tier wrote
            and lists (``--featurize-expected FILE``: the exported card, ability and part tables, a
            tensors file and a targets file, each array's FNV-1a 64 of the values the C++ side
            built): every array's values repacked hash the same, every file carries the spec's
            hash (FNV-1a 64 of ``spec.json``'s bytes) and vocab hash, and the shapes and indices
            agree with the spec (the tables' rows, widths and column names, the key list, F, C, K
            and Q, every relation, action, card and ability index in range).
  driver    ``rm_driver.py`` derives the engine seeds it always has (pinned values), and a game it
            plays (scripted against explore) on an interpreter without site-packages (``-S``)
            imports no module from outside the standard library and this directory: the kept
            Python is stdlib only.

Stdlib only. Prints one ``  OK   …`` line per passing check and one ``FAIL: …`` line per finding;
exit 1 on any finding::

    python3 python/pycheck.py [--binary bin/debug/robomage] [--out-dir DIR] [--part decode,selfkill,driver]
                              [--rmtb-samples FILE] [--featurize-expected FILE]
"""
from __future__ import annotations

import argparse
import json
import math
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
import rmtb  # noqa: E402

PARTS = ("decode", "refusals", "selfkill", "driver", "rmtb", "featurize")
# The committed oracle-text table (docs/featurizer.md §5).
TEXT_TABLE = os.path.join(os.path.dirname(_HERE), "bin", "resources", "embeddings", "oracle_text.rmtb")
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


# ── selfkill: the scripted agent's self-lethal guard ─────────────────────────────────────────

# A burn hand whose only other spells have no target: Price of Progress deals Player A 2 (its
# Fiery Islet) and Player B 4 (two Underground Seas, at 4 life).
_POP_POSITION = {"--deck-a": "league/burn", "--deck-b": "league/reanimator",
                 "--hand-a": "Price of Progress,Mountain,Mountain,Searing Blood,Searing Blood,"
                             "Smash to Smithereens,Smash to Smithereens",
                 "--library-a": "Mountain", "--battlefield-a": "Mountain,Mountain,Fiery Islet",
                 "--battlefield-b": "Underground Sea,Underground Sea", "--life-b": "4",
                 "--hand-b": "Island,Island,Island,Brainstorm,Ponder,Ponder,Brainstorm", "--library-b": "Island"}
# Dismember ({1}{B/P}{B/P}) with one black source: one pip is 2 life.
_DISMEMBER_POSITION = {"--deck-a": "league/bug", "--deck-b": "league/burn",
                       "--hand-a": "Dismember,Uro Titan of Natures Wrath,Island,Island,Ponder,Ponder,Ponder", "--library-a": "Island",
                       "--battlefield-a": "Island,Swamp", "--battlefield-b": "Goblin Guide,Mountain",
                       "--hand-b": "Mountain,Mountain,Mountain,Lightning Bolt,Lightning Bolt,Lightning Bolt,"
                                   "Lightning Bolt", "--library-b": "Mountain"}
# (label, position, overrides, the narrative line Player A's first turn must (not) show, shown).
SELFKILL_CASES = (
    ("Price of Progress at 2 life", _POP_POSITION, {"--life-a": "2"}, "Player A casts Price of Progress", False),
    ("Price of Progress at 2 life, killing both, dead on board", _POP_POSITION,
     {"--life-a": "2", "--battlefield-b": "Underground Sea,Underground Sea,Griselbrand"},
     "Player A casts Price of Progress", False),
    ("Dismember at 2 life, one black source", _DISMEMBER_POSITION, {"--life-a": "2"}, "Player A casts Dismember",
     False),
    ("Price of Progress at 3 life", _POP_POSITION, {"--life-a": "3"}, "Player A casts Price of Progress", True),
)
_SELFKILL_DECISIONS = 6     # the keeps and Player A's first-turn main phase


def check_selfkill(binary, work):
    """The selfkill part: each SELFKILL_CASES position, Player A's scripted agent deciding;
    returns failures."""
    fails = []
    cwd = os.getcwd()
    os.chdir(work)  # a draw's saved log stays in the scratch directory
    try:
        for label, position, overrides, line, shown in SELFKILL_CASES:
            opts = dict(position, **overrides)
            deck_a, deck_b = opts.pop("--deck-a"), opts.pop("--deck-b")
            flags = [x for kv in opts.items() for x in kv]
            narrative = []
            rec = rm_driver.play_game(rm_driver.make_agent("scripted"), rm_driver.make_agent("scripted"),
                                      binary=binary, deck_a=deck_a, deck_b=deck_b, seed=1,
                                      max_decisions=_SELFKILL_DECISIONS, engine_flags=flags,
                                      on_narrative=narrative.extend)
            seen = any(ln.strip().startswith(line) for ln in narrative)
            if seen != shown:
                fails.append(f"selfkill {label}: the narrative {'lacks' if shown else 'shows'} {line!r} "
                             f"(results {rec.game_results})")
    finally:
        os.chdir(cwd)
    if not fails:
        print(f"  OK   selfkill: {len(SELFKILL_CASES)} positions", flush=True)
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


# ── RMTB ──────────────────────────────────────────────────────────────────────────────────────

def _bits(dtype, x):
    """A float's bytes at its stored width, so -0.0 and the rounding are compared exactly; a value
    past the half range is its infinity (IEEE rounding; ``struct`` raises instead)."""
    try:
        return struct.pack("<e" if dtype == "f16" else "<f", float(x))
    except OverflowError:
        return struct.pack("<e" if dtype == "f16" else "<f", math.copysign(math.inf, float(x)))


def _sample_fails(t):
    """One sample file of robomage_check's against what python/rmtb.py reads from it."""
    name = os.path.basename(t["path"])
    try:
        meta, arrays = rmtb.read(t["path"])
    except (OSError, ValueError) as e:
        return [f"rmtb: {name}: {e}"]
    fails = []
    if meta != t["meta"]:
        fails.append(f"rmtb: {name}: meta {meta!r}, written {t['meta']!r}")
    for aname, want in t["arrays"].items():
        a = arrays.get(aname)
        if a is None:
            fails.append(f"rmtb: {name}: no array {aname}")
            continue
        fl = a.dtype in ("f32", "f16")
        same = len(a.values) == len(want["values"]) and all(
            (_bits(a.dtype, x) == _bits(a.dtype, y)) if fl else x == y for x, y in zip(a.values, want["values"]))
        if not same:
            fails.append(f"rmtb: {name}: {aname} reads {a.values!r}, written {want['values']!r}")
        # struct's 'e' rounds to nearest even like the C++ writer: the stored halves are its packing
        # of the floats they were written from.
        source = [struct.unpack("<f", _bits("f32", x))[0] for x in want.get("source", ())]  # the exact floats
        if "source" in want and [_bits("f16", x) for x in source] != [_bits("f16", x) for x in a.values]:
            fails.append(f"rmtb: {name}: {aname}'s halves differ from struct's rounding of {want['source']!r}")
    return fails


def check_rmtb(samples):
    """The rmtb part; returns failures."""
    fails = []
    checked = []
    if samples:
        with open(samples, encoding="utf-8") as f:
            expected = json.load(f)["files"]
        for t in expected:
            fails += _sample_fails(t)
            checked.append(os.path.basename(t["path"]))
    if os.path.isfile(TEXT_TABLE):
        try:
            meta, arrays = rmtb.read(TEXT_TABLE)
            present = {r["id"] for r in meta["rows_meta"]}
            rows = arrays["vectors"].rows()
            if meta["kind"] != "text" or any(any(rows[i]) for i in range(len(rows)) if i not in present):
                fails.append("rmtb: the text table is not a text table with zero rows outside rows_meta")
            checked.append(os.path.basename(TEXT_TABLE))
        except (OSError, ValueError, KeyError) as e:
            fails.append(f"rmtb: {TEXT_TABLE}: {e}")
    if not fails:
        print(f"  OK   rmtb: {', '.join(checked) or 'nothing to read (no --rmtb-samples, no text table)'}",
              flush=True)
    return fails


# ── The featurizer's files ────────────────────────────────────────────────────────────────────

_PACK = {"f32": "f", "f16": "e", "i64": "q", "i32": "i", "u16": "H", "u8": "B"}


def _fnv1a64(data):
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def _table_fails(kind, arrays, want, spec):
    """An exported table's shapes against the spec's card_tables entry."""
    feats = arrays["features"]
    fails = []
    if feats.shape != [want["rows"], want["width"]] or feats.columns != want["columns"]:
        fails.append(f"featurize: the {kind} table's features are {feats.shape}, the spec says "
                     f"[{want['rows']}, {want['width']}] with its columns")
    if kind == "card" and arrays["back_face"].shape != [want["rows"]]:
        fails.append("featurize: the card table's back_face is not one entry per card row")
    if kind == "ability":
        keys = arrays["key_rows"]
        if keys.columns != spec["ab_keys"] or any(not 0 <= r < want["rows"] for r in keys.values):
            fails.append("featurize: the ability table's key_rows are not the spec's keys within its rows")
    if kind == "part":
        rows = spec["card_tables"]["ability"]["rows"]
        if any(not 0 <= r < rows for r in arrays["ability"].values):
            fails.append("featurize: a part row names no ability row")
    return fails


def _tensor_fails(arrays, spec):
    """A tensors file's per-decision shapes and indices against the spec."""
    widths = {"tok_num": spec["F"], "tok_cat": spec["C"], "tok_card": spec["K"], "tok_ability": spec["Q"], "rel": 3}
    fails = []
    for name, w in widths.items():
        if arrays[name].shape[1:] != [w]:
            fails.append(f"featurize: {name} has shape {arrays[name].shape}, width {w} in the spec")
    if fails:
        return fails
    types = spec["token_types"]
    menu = next(i for i, t in enumerate(types) if t["name"] == spec["menu"])
    cards, abilities = spec["card_tables"]["card"]["rows"], spec["card_tables"]["ability"]["rows"]
    tok_off, rel_off, act_off = (arrays[n + "_offsets"].values for n in ("tok_type", "rel", "act_tok"))
    for name in ("tok_num", "tok_cat", "tok_card", "tok_ability", "tok_mask"):
        if arrays[name + "_offsets"].values != tok_off:
            fails.append(f"featurize: {name}_offsets differ from tok_type_offsets")
    tt, rel, act = arrays["tok_type"].values, arrays["rel"].values, arrays["act_tok"].values
    if any(not 0 <= t < len(types) for t in tt):
        fails.append("featurize: a tok_type is no token type")
    if any(not -1 <= c < cards for c in arrays["tok_card"].values):
        fails.append("featurize: a tok_card is no card row")
    if any(not -1 <= a < abilities for a in arrays["tok_ability"].values):
        fails.append("featurize: a tok_ability is no ability row")
    if any(not 0 <= c < spec["categorical_size"] for c in arrays["tok_cat"].values):
        fails.append("featurize: a tok_cat is outside the categorical id space")
    for d in range(len(tok_off) - 1):
        n = tok_off[d + 1] - tok_off[d]
        for r in range(rel_off[d], rel_off[d + 1]):
            src, dst, ty = rel[3 * r:3 * r + 3]
            if not (0 <= src < n and 0 <= dst < n and 0 <= ty < len(spec["relation_types"])):
                fails.append(f"featurize: decision {d}: a relation outside its tokens")
                break
        for a in act[act_off[d]:act_off[d + 1]]:
            if not 0 <= a < n or tt[tok_off[d] + a] != menu:
                fails.append(f"featurize: decision {d}: an act_tok that is no menu token")
                break
    return fails[:10]


def check_featurize(expected_path):
    """The featurize part; returns failures."""
    if not expected_path:
        print("  OK   featurize: nothing to read (no --featurize-expected)", flush=True)
        return []
    with open(expected_path, encoding="utf-8") as f:
        expected = json.load(f)
    with open(expected["spec"], "rb") as f:
        spec_bytes = f.read().rstrip(b"\n")
    spec = json.loads(spec_bytes.decode("utf-8"))
    spec_hash = "%016x" % _fnv1a64(spec_bytes)
    fails = []
    for want in expected["files"]:
        name = os.path.basename(want["path"])
        try:
            meta, arrays = rmtb.read(want["path"])
        except (OSError, ValueError) as e:
            fails.append(f"featurize: {name}: {e}")
            continue
        if meta.get("kind") != want["kind"] or meta.get("spec_hash") != spec_hash or \
                meta.get("vocab_hash") != spec["vocab_hash"]:
            fails.append(f"featurize: {name}: kind, spec_hash or vocab_hash is not {want['kind']}, {spec_hash}, "
                         f"{spec['vocab_hash']}")
        for aname, a_want in want["arrays"].items():
            a = arrays.get(aname)
            if a is None:
                fails.append(f"featurize: {name}: no array {aname}")
                continue
            packed = struct.pack("<%d%s" % (len(a.values), _PACK[a.dtype]), *a.values)
            if a.shape != a_want["shape"] or "%016x" % _fnv1a64(packed) != a_want["fnv"]:
                fails.append(f"featurize: {name}: {aname} does not read as the values written")
        kind = want["kind"]
        if kind in ("card", "ability", "part"):
            fails += _table_fails(kind, arrays, spec["card_tables"][kind], spec)
        elif kind == "tensors":
            fails += _tensor_fails(arrays, spec)
        elif kind == "targets":
            depth, lists = spec["library_depth"], spec["metagame"].get("meta_list", 0)
            for lib in ("own_library", "opp_library"):
                off = arrays[lib + "_offsets"].values
                if any(off[d + 1] - off[d] > depth for d in range(len(off) - 1)):
                    fails.append(f"featurize: {name}: {lib} deeper than {depth}")
            if any(not -1 <= v < lists for v in arrays["opp_list"].values):
                fails.append(f"featurize: {name}: an opp_list is no meta_list row")
    if not fails:
        print(f"  OK   featurize: {len(expected['files'])} files read as written, spec {spec_hash}: shapes and "
              "indices agree", flush=True)
    return fails


def run(binary, out_dir=None, parts=PARTS, rmtb_samples=None, featurize_expected=None):
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
        if "selfkill" in parts:
            fails += check_selfkill(binary, work)
        if "driver" in parts:
            fails += check_driver(binary, work)
        if "rmtb" in parts:
            fails += check_rmtb(rmtb_samples)
        if "featurize" in parts:
            fails += check_featurize(featurize_expected)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pycheck", description="The kept Python's checks (robomage_check's "
                                 "smoke tier runs them).")
    ap.add_argument("--binary", default=rm_driver.DEFAULT_BINARY, help="the engine (default %(default)s)")
    ap.add_argument("--out-dir", default=None, help="the scratch directory's parent (default the system's)")
    ap.add_argument("--part", default=",".join(PARTS), help=f"comma list of {', '.join(PARTS)} (default all)")
    ap.add_argument("--rmtb-samples", default=None, help="the rmtb part's samples (robomage_check's smoke tier "
                    "writes them)")
    ap.add_argument("--featurize-expected", default=None, help="the featurize part's file list (robomage_check's "
                    "featurize tier writes it)")
    args = ap.parse_args(argv)
    parts = [p for p in args.part.split(",") if p]
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        ap.error(f"unknown part(s) {', '.join(unknown)}")
    fails = run(args.binary, args.out_dir, parts, args.rmtb_samples, args.featurize_expected)
    for f in fails:
        print(f"FAIL: {f}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
