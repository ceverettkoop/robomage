#!/usr/bin/env python3
"""RMSTATE decode regression: the C++ decoder, the Python decoder and the stdout stream agree.

Checks, over freshly played games (or the dumps given on the command line):

  1. recode — the engine's ``--rmstate-read DUMP`` decodes every record of a ``--rmstate-dump``
     file, re-encodes it and finds the bytes identical (exit 0, no ``ERROR:``);
  2. json — ``train/rmstate.py``'s decode of the dump renders exactly the JSON lines the engine's
     ``--state-dump json`` renders (from ``--rmstate-read``), byte for byte, and both equal the
     ``DUMP.jsonl`` the engine wrote live with ``--state-dump json``;
  3. stream — a ``--machine-format rmstate`` session driven over a pipe (seeded random replies,
     ``-1`` confirms, a game concession past a decision cap): stdout holds only RMSTATE records;
     each seat gets its SCHEMA / CARDS / DECKLISTS frames before its first STATE frame and a
     terminal frame (``header.terminal``) at match end; every STATE frame equals the dump's frame
     for that seat and decision apart from the event tables and the knowledge tables (``k_*``: each
     writer feeds a knowledge instance of its own, so the stream's tables come from the seat's
     stream frames and the dump's from its every-decision frames); each seat's events,
     concatenated, equal the dump's; and the stream's bytes are the same with and without
     ``--rmstate-dump``;
  4. refusals — a STATE frame with another version, another schema hash, an unknown kind or a
     known table at another row width is refused by both decoders; an unknown table block is
     skipped by both (listed under ``"skipped"``), and the engine will not re-encode it;
  5. name-a-card — a Disruptor Flute preset's opening name-a-card menu offers the opponent's
     registered cards with open decklists, and every vocab card (in name order, no object behind a
     candidate; CARDS lists the whole vocab, DECKLISTS only the seat's own) under
     ``--closed-decklist`` (``docs/state_format.md`` register 11a);
  6. knowledge isolation (``check_knowledge_isolation``; ``run()`` and ``--isolation`` only, not
     part of the ``rmstate`` tier) — ``rmknow --isolation`` on each seat's frames of a dump or a
     saved stream (``stream_<seed>.stream.rms``) recomputes every frame's ``k_*`` tables from the
     frames alone and finds them byte-identical to the ones the engine appended
     (``docs/knowledge_store.md`` §7, §9.6).

Games: bo1 and bo3 explore-fuzz games through ``runner.run_match`` with the dump flags passed by
``$ROBOMAGE_ENGINE_FLAGS``, and the stream sessions on league matchups (bo1 and bo3, and one under
``--closed-decklist``, which only the RMSTATE stream supports).

Entry points: ``run(binary=…, dumps=…, work_dir=…, quick=…)`` returns the list of failure
messages (empty = pass); ``check_dump``, ``check_refusals`` and ``stream_checks`` are its parts,
which the ``rmstate`` gate tier (``train/test_selfcheck_state.py``) calls on its own games;
``main()`` is the CLI (exit 1 on failure)::

    train/.venv/bin/python train/test_rmstate.py              # play games, check everything
    train/.venv/bin/python train/test_rmstate.py --quick      # fewer / shorter games
    train/.venv/bin/python train/test_rmstate.py --dumps F…   # check existing dumps (1, 2 only)
    train/.venv/bin/python train/test_rmstate.py --isolation F…   # check 6 on dumps / saved streams

Torch-free; needs the engine (debug tier by default, ``--binary`` to override).
"""
import argparse
import glob
import os
import random
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rmstate  # noqa: E402
from cli_spec import BIN_DIR, BINARY  # noqa: E402
from env import ENGINE_FLAGS_ENV, CONCEDE_GAME  # noqa: E402

# Fuzz matchups for the dump checks: (deck_a, deck_b, bo3, seed).
_FUZZ_GAMES = [
    ("league/ur_delver", "league/gw_maverick", False, 11),
    ("league/bug", "league/wubg_doomsday", False, 12),
    ("league/lands", "league/bw_dnt", True, 13),
    ("league/reanimator", "league/uw_control", True, 14),
]
# Stream sessions: (deck_a, deck_b, bo3, seed, extra engine flags). --narrative checks that the
# narrative stays off stdout.
_STREAM_GAMES = [
    ("league/ur_delver", "league/bug", False, 21, ["--narrative"]),
    ("league/wrb_energy", "league/burn", True, 22, []),
    ("league/wrb_energy", "league/bw_dnt", False, 23, ["--closed-decklist"]),
]
_STREAM_DECISION_CAP = 250  # a seat concedes the game past this decision number
_TIMEOUT_S = 600


def _py_json(path):
    with open(path, "rb") as fp:
        return b"".join(line.encode("utf-8", "surrogateescape") + b"\n" for line in rmstate.json_lines(fp))


def _first_diff_line(a, b):
    la, lb = a.split(b"\n"), b.split(b"\n")
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            return i + 1
    return min(len(la), len(lb)) + 1


def check_dump(path, binary=BINARY):
    """Checks 1 and 2 on one --rmstate-dump file; returns failure messages."""
    fails = []
    name = os.path.basename(path)
    r = subprocess.run([binary, "--rmstate-read", path, "--state-dump", "json"], cwd=BIN_DIR,
                       capture_output=True, timeout=_TIMEOUT_S)
    summary = [ln for ln in r.stderr.decode(errors="replace").splitlines() if ln.startswith("RMSTATE_READ:")]
    if r.returncode != 0 or b"ERROR:" in r.stderr:
        errs = [ln for ln in r.stderr.decode(errors="replace").splitlines() if "ERROR:" in ln]
        fails.append(f"{name}: --rmstate-read exit {r.returncode}: {errs[:3]} {summary}")
    cpp = r.stdout
    try:
        py = _py_json(path)
    except rmstate.RMStateError as exc:
        fails.append(f"{name}: Python decode refused the dump: {exc}")
        py = None
    if py is not None and py != cpp:
        fails.append(f"{name}: Python JSON differs from the engine's at line {_first_diff_line(py, cpp)}")
    live = path + ".jsonl"
    if os.path.exists(live):
        with open(live, "rb") as fp:
            if fp.read() != cpp:
                fails.append(f"{name}: the live --state-dump json file differs from --rmstate-read's JSON")
    if not fails:
        print(f"  OK   {name}: {summary[0] if summary else ''}", flush=True)
    return fails


def _fuzz_dumps(out_dir, quick):
    """Plays the fuzz games with dumps on; returns the dump paths."""
    import runner
    games = _FUZZ_GAMES[1:3] if quick else _FUZZ_GAMES
    saved = os.environ.get(ENGINE_FLAGS_ENV)
    paths = []
    try:
        for k, (deck_a, deck_b, bo3, seed) in enumerate(games):
            sub = os.path.join(out_dir, f"fuzz{k}")
            os.makedirs(sub, exist_ok=True)
            os.environ[ENGINE_FLAGS_ENV] = (
                f"{saved or ''} --rmstate-dump {sub}/d_%p.rms --state-dump json").strip()
            runner.run_match("explore", "explore", deck_a=deck_a, deck_b=deck_b, games=1, bo3=bo3,
                             seed=seed, transcript="quiet", max_decisions=600 if quick else 2500)
            paths += sorted(glob.glob(os.path.join(sub, "d_*.rms")))
    finally:
        if saved is None:
            os.environ.pop(ENGINE_FLAGS_ENV, None)
        else:
            os.environ[ENGINE_FLAGS_ENV] = saved
    return paths


# ── refusals: frames the decoders must refuse or skip ────────────────────────────────────────

def _rebuild(header, blocks):
    """A frame payload from (version, kind, viewer, hash) and [(id, width, rows, body)]."""
    import struct
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


def check_refusals(dump_path, binary=BINARY, work_dir=None):
    """Each mutation of a dump's first STATE frame is refused (or, for an unknown table, skipped)
    by both decoders, with the same JSON for the skipped case; returns failure messages."""
    fails = []
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
        return [f"{os.path.basename(dump_path)}: no STATE frame to mutate"]
    for label, mutant, needle, refused in _mutants(state):
        path = os.path.join(work_dir or os.path.dirname(dump_path), f"mutant_{label.replace(' ', '_')}.rms")
        with open(path, "wb") as fp:
            for p in head + [mutant]:
                fp.write(b"RMSTATE %d\n" % len(p) + p)
        r = subprocess.run([binary, "--rmstate-read", path, "--state-dump", "json"], cwd=BIN_DIR,
                           capture_output=True, timeout=_TIMEOUT_S)
        err = r.stderr.decode(errors="replace")
        if r.returncode != 1 or needle not in err:
            fails.append(f"mutant {label}: the engine should report {needle!r} (exit {r.returncode}): {err[-300:]}")
        try:
            py = _py_json(path)
            if refused:
                fails.append(f"mutant {label}: the Python decoder accepted it")
            elif py != r.stdout or b'"skipped":[65534]' not in py:
                fails.append(f"mutant {label}: Python and engine JSON differ, or the block is not listed as skipped")
        except rmstate.RMStateError:
            if not refused:
                fails.append(f"mutant {label}: the Python decoder refused it")
    if not fails:
        print(f"  OK   refusals ({len(_mutants(state))} mutations)", flush=True)
    return fails


# ── the --machine-format rmstate stream ───────────────────────────────────────────────────────

def run_stream(binary, deck_a, deck_b, bo3, seed, dump=None, cap=_STREAM_DECISION_CAP, err_path=os.devnull,
               flags=()):
    """Drives one --machine-format rmstate session; returns its payloads. Raises on a protocol
    error. Replies: a seeded random menu row, now and then -1 (confirm = the last row), and a game
    concession once a game passes `cap` decisions."""
    cmd = [binary, "--machine-format", "rmstate", "--seed", str(seed), "--deck-a", deck_a, "--deck-b", deck_b]
    if bo3:
        cmd.append("--bo3")
    if dump:
        cmd += ["--rmstate-dump", dump]
    cmd += list(flags)
    rng = random.Random(seed)
    dec = rmstate.StreamDecoder()
    payloads = []
    terminals = 0
    with open(err_path, "wb") as err:
        proc = subprocess.Popen(cmd, cwd=BIN_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err)
        try:
            for payload in rmstate.records(proc.stdout):
                payloads.append(payload)
                frame = dec.feed(payload)
                if frame.kind != "STATE":
                    continue
                header = frame.tables["header"][0]
                if header["terminal"]:
                    terminals += 1
                    continue
                n = len(frame.tables.get("menu", []))
                if n == 0:
                    raise rmstate.RMStateError(f"a decision frame with no menu (decision {header['decision']})")
                if header["decision"] > cap:
                    reply = CONCEDE_GAME
                elif rng.random() < 0.05:
                    reply = -1
                else:
                    reply = rng.randrange(n)
                proc.stdin.write(f"{reply}\n".encode())
                proc.stdin.flush()
            rc = proc.wait(timeout=_TIMEOUT_S)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    if rc != 0:
        raise rmstate.RMStateError(f"engine exit {rc}")
    if terminals != 2:
        raise rmstate.RMStateError(f"{terminals} terminal frames, expected 2")
    return payloads


def _events(frames):
    """Every event of `frames`, in order, as (event row, its cards, its players) with the row
    links dropped."""
    out = []
    for f in frames:
        cards = {}
        for row in f.tables.get("event_card", []):
            cards.setdefault(row["event"], []).append({k: v for k, v in row.items() if k != "event"})
        players = {}
        for row in f.tables.get("event_player", []):
            players.setdefault(row["event"], []).append({k: v for k, v in row.items() if k != "event"})
        for i, row in enumerate(f.tables.get("event", [])):
            out.append((row, cards.get(i, []), players.get(i, [])))
    return out


def _without_events(frame):
    """The frame's tables apart from the event and knowledge tables (each writer drains its own
    event queue and feeds its own knowledge instance)."""
    return {k: v for k, v in frame.tables.items() if not k.startswith(("event", "k_"))}


def rmknow_binary(binary=BINARY):
    """The rmknow tool built beside `binary` (bin/<config>/rmknow)."""
    return os.path.join(os.path.dirname(os.path.abspath(binary)), "rmknow")


def check_knowledge_isolation(path, binary=BINARY, levels=("expert", "expert")):
    """Check 6 on one RMSTATE stream file (a --rmstate-dump, or a saved --machine-format rmstate
    stream): for each seat, ``rmknow --isolation`` recomputes the k_* tables of every one of that
    seat's STATE frames from the frames themselves and compares them with the engine's, byte for
    byte. `levels` are the seats' --knowledge-a/-b levels. Returns failure messages."""
    fails = []
    name = os.path.basename(path)
    summaries = []
    for seat, level in zip(("A", "B"), levels):
        r = subprocess.run([rmknow_binary(binary), "--viewer", seat, "--level", level, "--isolation", "--in", path],
                           capture_output=True, timeout=_TIMEOUT_S)
        err = r.stderr.decode(errors="replace").splitlines()
        summary = [ln for ln in err if ln.startswith("RMKNOW:")]
        if r.returncode != 0 or not summary or "isolation_frames=0 " in summary[0]:
            bad = [ln for ln in err if "ERROR:" in ln]
            fails.append(f"{name} viewer {seat}: rmknow --isolation exit {r.returncode}: {bad[:3]} {summary}")
        summaries += summary
    if not fails:
        print(f"  OK   isolation {name}: " + "; ".join(summaries), flush=True)
    return fails


def write_stream(payloads, path):
    """Saves a session's payloads as RMSTATE records (the bytes the engine wrote on stdout)."""
    with open(path, "wb") as fp:
        for p in payloads:
            fp.write(b"RMSTATE %d\n" % len(p) + p)


def _frame_key(frame):
    """A STATE frame's decision: (viewer, game, decision, terminal, sideboarding). The sideboarding
    frames before a game carry that game's number and go on with the previous game's decision
    count, which the game itself restarts at 1, so the phase tells them apart."""
    h = frame.tables["header"][0]
    sideboarding = bool(frame.tables.get("match", [{}])[0].get("sideboard_phase"))
    return (frame.viewer, h["game"], h["decision"], h["terminal"], sideboarding)


def check_stream(payloads, dump_path, label):
    """Check 3 against the dump written alongside the stream; returns failure messages."""
    fails = []
    dec = rmstate.StreamDecoder()
    stream = [(p, dec.feed(p)) for p in payloads]
    with open(dump_path, "rb") as fp:
        ddec = rmstate.StreamDecoder()
        dump = [(p, ddec.feed(p)) for p in rmstate.records(fp)]

    dump_state = {}
    dump_match = {}
    dump_by_viewer = {}
    for p, f in dump:
        if f.kind == "STATE":
            dump_state[_frame_key(f)] = f
            dump_by_viewer.setdefault(f.viewer, []).append(f)
        else:
            dump_match.setdefault((f.viewer, f.kind), p)

    seen_match = set()
    by_seat = {}
    last_terminal = {}
    for i, (p, f) in enumerate(stream):
        if f.viewer not in ("A", "B"):
            fails.append(f"{label}: record {i + 1} is for viewer {f.viewer}")
            continue
        if f.kind != "STATE":
            seen_match.add((f.viewer, f.kind))
            if dump_match.get((f.viewer, f.kind)) != p:
                fails.append(f"{label}: {f.viewer}'s {f.kind} frame differs from the dump's")
            continue
        if {(f.viewer, k) for k in ("SCHEMA", "CARDS", "DECKLISTS")} - seen_match:
            fails.append(f"{label}: record {i + 1}: {f.viewer}'s STATE frame before its match frames")
        h = f.tables["header"][0]
        key = _frame_key(f)
        d = dump_state.get(key)
        if d is None:
            fails.append(f"{label}: no dump frame for {key}")
        elif _without_events(d) != _without_events(f):
            diff = sorted(k for k in set(d.tables) | set(f.tables)
                          if not k.startswith(("event", "k_")) and d.tables.get(k) != f.tables.get(k))
            fails.append(f"{label}: frame {key} differs from the dump's in {diff}")
        by_seat.setdefault(f.viewer, []).append(f)
        if h["terminal"]:
            last_terminal[f.viewer] = i
    for seat in ("A", "B"):
        if seat not in last_terminal:
            fails.append(f"{label}: no terminal frame for {seat}")
        if _events(by_seat.get(seat, [])) != _events(dump_by_viewer.get(seat, [])):
            fails.append(f"{label}: {seat}'s stream events differ from the dump's")
    if last_terminal and min(last_terminal.values()) < len(stream) - 2:
        fails.append(f"{label}: a terminal frame is not at the end of the stream")
    return fails


def stream_checks(binary, out_dir, games):
    """Check 3 (plus check 1 and 2 on the dump each session writes) on `games`, a list of
    (deck_a, deck_b, bo3, seed, extra engine flags); returns failure messages. A session's engine
    stderr is kept as ``stream_<seed>.err`` in `out_dir`."""
    fails = []
    for deck_a, deck_b, bo3, seed, flags in games:
        label = f"stream {os.path.basename(deck_a)} vs {os.path.basename(deck_b)} {'bo3' if bo3 else 'bo1'}"
        dump = os.path.join(out_dir, f"stream_{seed}.rms")
        try:
            with_dump = run_stream(binary, deck_a, deck_b, bo3, seed, dump=dump,
                                   err_path=os.path.join(out_dir, f"stream_{seed}.err"), flags=flags)
            alone = run_stream(binary, deck_a, deck_b, bo3, seed, flags=flags)
        except (rmstate.RMStateError, subprocess.TimeoutExpired) as exc:
            fails.append(f"{label}: {exc}")
            continue
        got = check_stream(with_dump, dump, label)
        with open(os.path.join(out_dir, f"stream_{seed}.err"), "rb") as fp:
            err = fp.read()
        if b"GAME_RESULT:" not in err or ("--narrative" in flags and b"-------- TURN" not in err):
            got.append(f"{label}: the results / narrative did not reach stderr")
        if with_dump != alone:
            got.append(f"{label}: the stream differs with and without --rmstate-dump")
        write_stream(with_dump, os.path.join(out_dir, f"stream_{seed}.stream.rms"))
        got += check_dump(dump, binary)
        if not got:
            print(f"  OK   {label}: {len(with_dump)} records", flush=True)
        fails += got
    return fails


# ── name-a-card candidates and --closed-decklist (docs/state_format.md register 11a) ────────────

# A Disruptor Flute preset on Player A's battlefield names a card as the game starts (its ETB
# choice), so the session's first name-a-card menu comes without any scripted play.
_NAME_CARD_FLAGS = ["--battlefield-a", "Disruptor Flute", "--selfcheck-state", "--selfcheck-state-sample", "1"]
_NAME_CARD_GAME = ("league/bw_dnt", "league/burn", 31)


def _first_name_card_menu(payloads):
    """(A's CARDS frame, A's DECKLISTS frame, the rows of A's first NAME_CARD menu, the schema)
    of a session."""
    dec = rmstate.StreamDecoder()
    cards = decklists = None
    for p in payloads:
        f = dec.feed(p)
        if f.viewer != "A":
            continue
        if f.kind == "CARDS":
            cards = f
        elif f.kind == "DECKLISTS":
            decklists = f
        elif f.kind == "STATE":
            cats = dec.schema.enums["action_category"]
            rows = [m for m in f.tables.get("menu", []) if cats[m["category"]] == "NAME_CARD"]
            if rows:
                return cards, decklists, rows, dec.schema
    return cards, decklists, [], dec.schema


def check_name_card_candidates(binary, out_dir):
    """Disruptor Flute's name-a-card menu with open decklists (the default: the opponent's
    registered cards, which DECKLISTS shows) and under --closed-decklist (every vocab card, by front
    face, in name order, with no object behind a candidate; CARDS then lists the whole vocab and
    DECKLISTS only the seat's own list); returns failure messages."""
    fails = []
    deck_a, deck_b, seed = _NAME_CARD_GAME
    for closed in (False, True):
        label = f"name-a-card {'closed' if closed else 'open'} decklist"
        flags = _NAME_CARD_FLAGS + (["--closed-decklist"] if closed else [])
        err = os.path.join(out_dir, f"name_card_{'closed' if closed else 'open'}.err")
        got = []
        try:
            payloads = run_stream(binary, deck_a, deck_b, False, seed, cap=20, err_path=err, flags=flags)
        except (rmstate.RMStateError, subprocess.TimeoutExpired) as exc:
            fails.append(f"{label}: {exc}")
            continue
        cards, decklists, rows, schema = _first_name_card_menu(payloads)
        if not rows:
            fails.append(f"{label}: Player A was never offered a name-a-card menu")
            continue
        seats = schema.enums["seat"]
        faces = schema.enums["face"]
        named = [m["source_card"] for m in rows]
        listed = {seats[r["player"]]: set() for r in decklists.tables.get("decklist_card", [])}
        for r in decklists.tables.get("decklist_card", []):
            listed[seats[r["player"]]].add(r["card"])
        fronts = sorted(r["card"] for r in cards.tables["card"] if faces[r["face"]] == "FRONT" and not r["token"])
        if closed:
            if sorted(named) != fronts:
                got.append(f"{label}: {len(named)} candidates, CARDS lists {len(fronts)} vocab cards")
            if any(m["source"] for m in rows):
                got.append(f"{label}: a candidate has an object handle")
            descs = [m["description"] for m in rows]
            if descs != sorted(descs):
                got.append(f"{label}: the candidates are not in name order")
            if set(listed) != {"PLAYER_A"}:
                got.append(f"{label}: DECKLISTS lists {sorted(listed)}, expected only A's own list")
        else:
            opp = listed.get("PLAYER_B", set())
            if not set(named) <= opp or len(set(named)) != len(named):
                got.append(f"{label}: candidates outside the opponent's registered list")
        with open(err, encoding="utf-8", errors="replace") as fh:
            bad = [ln for ln in fh if ln.startswith(("ERROR:", "FATAL:"))]
        if bad:
            got.append(f"{label}: {bad[0].strip()}")
        if not got:
            print(f"  OK   {label}: {len(rows)} candidates", flush=True)
        fails += got
    return fails


def run(binary=BINARY, dumps=None, work_dir=None, quick=False):
    """The whole check; returns failure messages (empty = pass). With `dumps`, checks 1 and 2 on
    those files only. `work_dir` keeps the games' dumps (default: a temporary directory)."""
    fails = []
    tmp = None
    if dumps:
        print("rmstate: checking given dumps", flush=True)
        for path in dumps:
            fails += check_dump(os.path.abspath(path), binary)
        return fails
    if work_dir is None:
        tmp = work_dir = tempfile.mkdtemp(prefix="rmstate_")
    os.makedirs(work_dir, exist_ok=True)
    try:
        print("rmstate: explore games with --rmstate-dump --state-dump json", flush=True)
        paths = _fuzz_dumps(work_dir, quick)
        if not paths:
            fails.append("no dump was written")
        for path in paths:
            fails += check_dump(path, binary)
        if paths:
            fails += check_refusals(paths[0], binary, work_dir)
        print("rmstate: --machine-format rmstate sessions", flush=True)
        games = _STREAM_GAMES[:1] if quick else _STREAM_GAMES
        fails += stream_checks(binary, work_dir, games)
        fails += check_name_card_candidates(binary, work_dir)
        print("rmstate: knowledge isolation (rmknow --isolation)", flush=True)
        for path in paths + sorted(glob.glob(os.path.join(work_dir, "stream_*.rms"))):
            fails += check_knowledge_isolation(path, binary)
    finally:
        if tmp is not None and not fails:
            shutil.rmtree(tmp, ignore_errors=True)
    if fails and tmp is not None:
        print(f"rmstate: dumps kept in {tmp}", flush=True)
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--binary", default=BINARY, help="engine binary (default: the debug tier)")
    ap.add_argument("--dumps", nargs="+", help="check these --rmstate-dump files instead of playing games")
    ap.add_argument("--work-dir", help="keep the games' dumps here")
    ap.add_argument("--quick", action="store_true", help="fewer, shorter games")
    ap.add_argument("--isolation", nargs="+", metavar="FILE",
                    help="only check 6 (rmknow --isolation) on these dumps / saved streams")
    ap.add_argument("--knowledge-a", default="expert", choices=["off", "expert"],
                    help="with --isolation: seat A's knowledge level in those files")
    ap.add_argument("--knowledge-b", default="expert", choices=["off", "expert"],
                    help="with --isolation: seat B's knowledge level in those files")
    args = ap.parse_args(argv)
    if args.isolation:
        fails = []
        for path in args.isolation:
            fails += check_knowledge_isolation(os.path.abspath(path), args.binary,
                                               (args.knowledge_a, args.knowledge_b))
    else:
        fails = run(args.binary, args.dumps, args.work_dir, args.quick)
    for f in fails:
        print(f"  FAIL {f}", flush=True)
    print(f"rmstate: {'FAIL' if fails else 'OK'} ({len(fails)} failure(s))", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
