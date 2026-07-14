#!/usr/bin/env python3
"""Snapshot / search-server regression: drive the `--search-server` machine
protocol over a raw subprocess and assert the search primitives are correct.

The `--search-server` flag turns the machine-mode decision read into a small
stdio command protocol for in-process MCTS: SNAPSHOT / RESTORE deep-copy and
roll back the whole game state, DETERMINIZE reshuffles the hidden zones from a
world-local RNG (never the real game's stream), and a game that ends while a
snapshot is live is intercepted as SIM_RESULT instead of exiting. This test
exercises each primitive and asserts the guarantees the search actor relies on:

  1. round-trip identity — SNAPSHOT then a divergent line then RESTORE returns
     byte-for-byte to the snapshot decision, and the resumed real line stays
     byte-identical to a control run played without any snapshotting (same
     outcome).
  2. RNG isolation — interleaving RESTORE+DETERMINIZE excursions does not
     perturb the real game: the resumed line still matches the control run.
  3. determinize efficacy + reproducibility — same seed reshuffles identically
     (equal BQUERY stream), different seeds diverge.
  4. terminal intercept — a simulated line that ends yields SIM_RESULT (not a
     process exit); RESTORE recovers the pre-terminal decision byte-identically;
     RELEASE + auto-play then finishes the REAL game with a clean exit + winner.
  5. determinize invariants — the priority player's own hand, the step, both
     players' life/hand/library counts, and the battlefield are unchanged by a
     DETERMINIZE (only hidden library order / opponent hidden cards move), and
     RESTORE undoes it byte-for-byte.
  6. perf — cost of a SNAPSHOT+RESTORE pair on the current build (reported;
     warns, does not fail, above a debug-build threshold).

Torch-free: spawns bin/robomage directly (like train/test_obs_invariants.py) and
imports every layout constant from ``env`` / ``_enums`` (zero magic numbers), so
a state-layout change reroutes the decode automatically.

Wired into ``train/ci_check.py`` as the ``snapshot`` tier (so ``make check`` runs
it); also runnable standalone::

    train/.venv/bin/python train/test_snapshot.py
"""
import hashlib
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _enums import (STATE_SIZE, MAX_ACTIONS,
                    CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT, CAT_SIDEBOARD_DONE)
from env import (
    N_CARD_TYPES, MAX_HAND_SLOTS,
    _HAND_START, _SELF_BLOCK_START, _OPP_BLOCK_START, _PB_LIFE, _PB_HAND_CT,
    _LIBRARY_CTX_START, _STEP_ONEHOT_START, _STEP_ONEHOT_SIZE,
    _SELF_PERM_START, _STACK_START)
from cli_spec import BINARY, BIN_DIR

# The binary payload framing after every BQUERY header, WITHOUT --narrative:
# STATE_SIZE float32 state, then six MAX_ACTIONS-wide metadata arrays
# (cats int32, ids f32, ctrl f32, pub f32, zone int32, refs int32). This is the
# exact size input_logger.cpp emits; a short/over read desyncs the whole stream.
PAYLOAD_BYTES = STATE_SIZE * 4 + 6 * MAX_ACTIONS * 4


class ProtocolError(AssertionError):
    """The engine's search protocol violated an invariant this test asserts."""


class R:
    """One outcome of reading the engine's stdout: a decision query, a simulated
    terminal (SIM_RESULT), or real process exit (EOF)."""

    __slots__ = ("kind", "nc", "safe", "payload", "notes", "result",
                 "returncode", "winner")

    def __init__(self, kind, nc=None, safe=None, payload=None, notes=None,
                 result=None, returncode=None, winner=None):
        self.kind = kind          # "q" | "sim" | "eof"
        self.nc = nc              # query: number of legal choices
        self.safe = safe          # query: SEARCHINFO safe flag (bool)
        self.payload = payload    # query: raw PAYLOAD_BYTES
        self.notes = notes or []  # intervening text lines (SNAPSHOT_OK, ...)
        self.result = result      # sim: "A" | "B" | "DRAW"
        self.returncode = returncode  # eof: process return code
        self.winner = winner      # last "Player X wins" narrative line seen

    def note_has(self, tok):
        return any(tok in n for n in self.notes)


class Engine:
    """A raw --search-server subprocess with the stdio command protocol."""

    def __init__(self, seed, extra=()):
        self.p = subprocess.Popen(
            [BINARY, "--machine", "--search-server", "--seed", str(seed),
             *extra],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=BIN_DIR, bufsize=-1)
        self.last_winner = None

    def _read_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.p.stdout.read(n - len(buf))
            if not chunk:
                raise EOFError("engine closed mid binary payload")
            buf.extend(chunk)
        return bytes(buf)

    def read(self):
        """Read stdout to the next decision query / SIM_RESULT / process exit.

        The "Player X wins" line is NOT a terminal marker here: state-based
        effects print it before search_intercept_game_end runs, so it appears in
        simulated ends too. The authoritative signals are SIM_RESULT (simulated)
        and EOF (real). We only stash the winner line for reporting.
        """
        notes = []
        safe = None
        while True:
            line = self.p.stdout.readline()
            if not line:
                return R("eof", notes=notes, returncode=self.p.wait(),
                         winner=self.last_winner)
            s = line.rstrip(b"\n")
            if s.startswith(b"SEARCHINFO"):
                safe = int(s.split(b"safe=")[1]) == 1
                notes.append(s)
            elif s.startswith(b"SIM_RESULT:"):
                return R("sim", notes=notes, winner=self.last_winner,
                         result=s.split(b":", 1)[1].strip().decode())
            elif s.startswith(b"BQUERY: "):
                f = s[8:].split()
                if len(f) != 3:
                    raise ProtocolError(f"malformed BQUERY header: {s!r}")
                nc, ss, ma = int(f[0]), int(f[1]), int(f[2])
                # Layout handshake: the trailing sizes must match the Python
                # constants, else the payload below would be misframed.
                if ss != STATE_SIZE or ma != MAX_ACTIONS:
                    raise ProtocolError(
                        f"BQUERY layout mismatch: engine STATE_SIZE={ss} "
                        f"MAX_ACTIONS={ma} vs python {STATE_SIZE}/{MAX_ACTIONS}")
                payload = self._read_exact(PAYLOAD_BYTES)
                return R("q", nc=nc, safe=safe, payload=payload, notes=notes,
                         winner=self.last_winner)
            else:
                if b"wins" in s:
                    self.last_winner = s.decode(errors="replace")
                notes.append(s)

    def _send(self, text):
        self.p.stdin.write(f"{text}\n".encode())
        self.p.stdin.flush()

    def play(self, action):
        self._send(action)
        return self.read()

    def snapshot(self, slot=0):
        self._send(f"SNAPSHOT {slot}")
        r = self.read()
        if not r.note_has(b"SNAPSHOT_OK"):
            raise ProtocolError(f"SNAPSHOT {slot} produced no SNAPSHOT_OK ack")
        return r

    def restore(self, slot=0):
        # No ack line: the engine unwinds and re-emits the restored decision.
        self._send(f"RESTORE {slot}")
        return self.read()

    def determinize(self, seed):
        self._send(f"DETERMINIZE {seed}")
        r = self.read()
        if not r.note_has(b"DETERMINIZE_OK"):
            raise ProtocolError(f"DETERMINIZE {seed} produced no DETERMINIZE_OK ack")
        return r

    def release(self):
        self._send("RELEASE")
        return self.read()

    def kill(self):
        try:
            self.p.kill()
        except Exception:
            pass


# ── decode / diff helpers ─────────────────────────────────────────────────────

def _state(payload):
    """First STATE_SIZE floats of a query payload."""
    return np.frombuffer(payload[:STATE_SIZE * 4], dtype=np.float32)


# The per-action category ints are the FIRST MAX_ACTIONS-wide metadata array after
# the STATE_SIZE state floats (cats int32, then ids/ctrl/pub f32, zone/refs int32).
_CATS_OFF = STATE_SIZE * 4
_SIDEBOARD_CATS = (CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT, CAT_SIDEBOARD_DONE)


def _query_cats(payload):
    return np.frombuffer(payload[_CATS_OFF:_CATS_OFF + MAX_ACTIONS * 4], dtype=np.int32)


def _is_sideboard_query(r):
    """True if a query's action menu is a bo3 sideboard prompt (IN/OUT/DONE)."""
    if r.kind != "q":
        return False
    cats = _query_cats(r.payload)
    return bool(np.isin(cats, _SIDEBOARD_CATS).any())


def _first_diff(a, b):
    """(byte_offset, human location) of the first differing byte, or None."""
    if a == b:
        return None
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            if i < STATE_SIZE * 4:
                loc = f"state float #{i // 4}"
            else:
                loc = f"action-metadata byte {i}"
            return (i, loc)
    return (n, f"length differs ({len(a)} vs {len(b)})")


def _assert_same_query(got, want_payload, want_nc, ctx):
    """Assert a query re-emitted after SNAPSHOT/RESTORE is byte-identical."""
    if got.kind != "q":
        raise ProtocolError(f"{ctx}: expected a re-emitted query, got {got.kind}")
    if got.nc != want_nc:
        raise ProtocolError(f"{ctx}: num_choices {got.nc} != snapshot {want_nc}")
    d = _first_diff(got.payload, want_payload)
    if d is not None:
        raise ProtocolError(f"{ctx}: payload differs at byte {d[0]} ({d[1]})")


# ── control-line recording ────────────────────────────────────────────────────

def record_line(seed, policy, cap=4000):
    """Play one full game with `policy(idx, nc) -> action` and no snapshotting;
    return (records, outcome). records[i] = (nc, payload, safe). outcome is a
    dict {kind, returncode, winner} at real EOF (games never draw here)."""
    eng = Engine(seed)
    records = []
    r = eng.read()
    idx = 0
    while r.kind == "q" and idx < cap:
        records.append((r.nc, r.payload, r.safe))
        r = eng.play(policy(idx, r.nc))
        idx += 1
    outcome = {"kind": r.kind, "returncode": r.returncode, "winner": r.winner}
    eng.kill()
    return records, outcome


def _mod_policy(idx, nc):
    return idx % nc


def _auto0(idx, nc):
    return 0


def _diverge(i, nc):
    return (i * 7 + 3) % nc


# ── tests ─────────────────────────────────────────────────────────────────────

def test_round_trip():
    """SNAPSHOT / divergent line / RESTORE returns byte-identically, and the
    resumed control line stays byte-identical to a no-snapshot control run with
    the same outcome. Snapshots are taken at decisions 10/30/60 (deferred to the
    next safe decision if one is unsafe). Also asserts the safe marker wiring:
    safe=0 during the pregame mulligans, safe=1 later."""
    seed = 5
    control, outcome = record_line(seed, _mod_policy)
    n = len(control)

    safes = [s for (_, _, s) in control]
    if not (False in safes[:4]):
        raise ProtocolError("expected an unsafe (safe=0) pregame decision in the "
                            f"first few decisions; got safe flags {safes[:4]}")
    if not (True in safes):
        raise ProtocolError("expected at least one safe=1 decision")

    eng = Engine(seed)
    cur = eng.read()
    pending = [10, 30, 60]
    excursions = 0
    for idx in range(n):
        # Continuous alignment: every real-line decision must match the control.
        _assert_same_query(cur, control[idx][1], control[idx][0],
                           f"round-trip alignment at decision {idx}")
        if pending and idx >= pending[0] and cur.safe:
            pending.pop(0)
            snap_nc, snap_pl = cur.nc, cur.payload
            q = eng.snapshot(0)
            _assert_same_query(q, snap_pl, snap_nc,
                              f"post-SNAPSHOT re-emit at decision {idx}")
            # An 8-decision divergent line; stop early if it ends the game.
            dq = q
            for i in range(8):
                dq = eng.play(_diverge(i, dq.nc))
                if dq.kind != "q":
                    break
            rq = eng.restore(0)
            _assert_same_query(rq, snap_pl, snap_nc,
                              f"post-RESTORE re-emit at decision {idx}")
            # Commit the real move: drop the snapshot so the game's true end
            # later exits cleanly (EOF) rather than being intercepted as
            # SIM_RESULT. RELEASE at a live decision re-emits the same query.
            rel = eng.release()
            _assert_same_query(rel, snap_pl, snap_nc,
                              f"post-RELEASE re-emit at decision {idx}")
            cur = rel
            excursions += 1
        cur = eng.play(_mod_policy(idx, cur.nc))
    if cur.kind != "eof":
        eng.kill()
        raise ProtocolError(f"resumed line did not end at EOF (got {cur.kind}) — "
                            "alignment drifted vs control")
    if cur.returncode != 0:
        eng.kill()
        raise ProtocolError(f"resumed game exited with code {cur.returncode}")
    if cur.winner != outcome["winner"]:
        eng.kill()
        raise ProtocolError(f"outcome mismatch: control {outcome['winner']!r} vs "
                            f"resumed {cur.winner!r}")
    eng.kill()
    if excursions != 3:
        raise ProtocolError(f"expected 3 snapshot excursions, ran {excursions}")
    return f"{n} decisions, 3 excursions, outcome={outcome['winner']!r}"


def test_rng_isolation():
    """RESTORE+DETERMINIZE excursions at a safe decision leave the real line
    untouched: after 6 (RESTORE, DETERMINIZE k, descend) cycles and a final
    RESTORE, the resumed auto-0 line is byte-identical to a clean control run."""
    seed = 5
    control, outcome = record_line(seed, _auto0)
    n = len(control)
    anchor = 15  # a safe mid-game decision (all decisions after mulligans are safe)

    eng = Engine(seed)
    cur = eng.read()
    for idx in range(anchor):
        _assert_same_query(cur, control[idx][1], control[idx][0],
                           f"pre-anchor alignment at decision {idx}")
        cur = eng.play(_auto0(idx, cur.nc))
    if not cur.safe:
        eng.kill()
        raise ProtocolError(f"anchor decision {anchor} is not safe")
    snap_nc, snap_pl = cur.nc, cur.payload
    q = eng.snapshot(0)
    _assert_same_query(q, snap_pl, snap_nc, "RNG-iso post-SNAPSHOT re-emit")
    for k in range(1, 7):
        rq = eng.restore(0)
        _assert_same_query(rq, snap_pl, snap_nc, f"RNG-iso RESTORE (cycle {k})")
        dq = eng.determinize(k)
        # Throwaway descent; stop at a simulated terminal.
        for _ in range(10):
            dq = eng.play(0)
            if dq.kind != "q":
                break
    cur = eng.restore(0)
    _assert_same_query(cur, snap_pl, snap_nc, "RNG-iso final RESTORE")
    # Commit: drop the snapshot so the resumed real line ends at a clean EOF.
    cur = eng.release()
    _assert_same_query(cur, snap_pl, snap_nc, "RNG-iso post-RELEASE re-emit")
    for idx in range(anchor, n):
        _assert_same_query(cur, control[idx][1], control[idx][0],
                           f"post-excursion alignment at decision {idx}")
        cur = eng.play(_auto0(idx, cur.nc))
    if cur.kind != "eof" or cur.returncode != 0:
        eng.kill()
        raise ProtocolError(f"resumed line ended abnormally: {cur.kind} "
                            f"rc={cur.returncode}")
    if cur.winner != outcome["winner"]:
        eng.kill()
        raise ProtocolError(f"outcome mismatch after excursions: {outcome['winner']!r} "
                            f"vs {cur.winner!r}")
    eng.kill()
    return f"{n} decisions, 6 determinize cycles, outcome preserved"


def _descend_hash(eng, first_q, count):
    """Auto-0 for up to `count` decisions from first_q, hashing each BQUERY
    payload; a simulated terminal contributes a terminal token and stops the
    descent early. Returns the hex digest."""
    h = hashlib.sha256()
    h.update(first_q.payload)
    for _ in range(count - 1):
        r = eng.play(0)
        if r.kind != "q":
            if r.kind == "sim":
                h.update(b"SIM:" + r.result.encode())
            break
        h.update(r.payload)
    return h.hexdigest()


def test_determinize_efficacy():
    """Same seed reshuffles identically (equal descent hash); a different seed
    diverges. SIM_RESULT during a descent is handled (terminal token + stop),
    then RESTORE returns for the next cycle."""
    seed = 5
    anchor = 15
    eng = Engine(seed)
    cur = eng.read()
    for idx in range(anchor):
        cur = eng.play(0)
    if not cur.safe:
        eng.kill()
        raise ProtocolError("determinize anchor not safe")
    snap_pl, snap_nc = cur.payload, cur.nc
    eng.snapshot(0)

    def cycle(dseed):
        r = eng.restore(0)
        _assert_same_query(r, snap_pl, snap_nc, f"efficacy RESTORE (seed {dseed})")
        dq = eng.determinize(dseed)
        return _descend_hash(eng, dq, 60)

    h1a = cycle(1)
    h1b = cycle(1)
    if h1a != h1b:
        eng.kill()
        raise ProtocolError("DETERMINIZE 1 not reproducible: two identical-seed "
                            f"descents hashed differently ({h1a[:12]} vs {h1b[:12]})")
    alt = None
    for dseed in (2, 3):
        h = cycle(dseed)
        if h != h1a:
            alt = (dseed, h)
            break
    if alt is None:
        eng.kill()
        raise ProtocolError("DETERMINIZE with seeds 2 and 3 both hashed identically "
                            "to seed 1 — reshuffle appears to have no effect")
    eng.kill()
    return f"seed1 reproducible ({h1a[:12]}), seed{alt[0]} diverged ({alt[1][:12]})"


def test_terminal_intercept():
    """A simulated line that ends yields SIM_RESULT (not process exit); RESTORE
    recovers the pre-terminal decision byte-identically; RELEASE at that live
    decision re-emits the query; auto-0 then finishes the REAL game with a clean
    exit and a real winner line."""
    seed = 5
    anchor = 15
    eng = Engine(seed)
    cur = eng.read()
    for idx in range(anchor):
        cur = eng.play(0)
    if not cur.safe:
        eng.kill()
        raise ProtocolError("terminal-intercept anchor not safe")
    snap_pl, snap_nc = cur.payload, cur.nc
    eng.snapshot(0)

    r = cur
    sim = None
    for _ in range(5000):
        r = eng.play(0)
        if r.kind == "sim":
            sim = r
            break
        if r.kind == "eof":
            break
    if sim is None:
        eng.kill()
        raise ProtocolError(f"no SIM_RESULT within 5000 decisions (got {r.kind}) — "
                            "a live snapshot should intercept game end")
    if sim.result not in ("A", "B", "DRAW"):
        eng.kill()
        raise ProtocolError(f"unexpected SIM_RESULT payload {sim.result!r}")
    rq = eng.restore(0)
    _assert_same_query(rq, snap_pl, snap_nc, "terminal-intercept post-RESTORE re-emit")
    rel = eng.release()
    if rel.kind != "q":
        eng.kill()
        raise ProtocolError(f"RELEASE at a live decision should re-emit the query, "
                            f"got {rel.kind}")
    _assert_same_query(rel, snap_pl, snap_nc, "terminal-intercept post-RELEASE re-emit")
    # Snapshots dropped: the game now really ends.
    fin = rel
    for _ in range(5000):
        fin = eng.play(0)
        if fin.kind != "q":
            break
    if fin.kind != "eof":
        eng.kill()
        raise ProtocolError(f"real game did not exit after RELEASE (got {fin.kind})")
    if fin.returncode != 0:
        eng.kill()
        raise ProtocolError(f"real game exited with code {fin.returncode}")
    if not fin.winner or b"wins" not in fin.winner.encode():
        eng.kill()
        raise ProtocolError(f"no real winner line printed (last: {fin.winner!r})")
    eng.kill()
    return f"SIM_RESULT={sim.result}, real winner={fin.winner!r}, clean exit"


def test_determinize_invariants():
    """DETERMINIZE preserves everything visible to the priority player except
    hidden library order: its own hand, the step, both players' life/hand/library
    counts, and the whole battlefield are byte/-value unchanged; RESTORE then
    reverts byte-for-byte."""
    seed = 7
    anchor = 15
    eng = Engine(seed)
    cur = eng.read()
    for idx in range(anchor):
        cur = eng.play(0)
    if not cur.safe:
        eng.kill()
        raise ProtocolError("invariants anchor not safe")
    snap_pl, snap_nc = cur.payload, cur.nc
    eng.snapshot(0)
    dq = eng.determinize(9)
    if dq.kind != "q":
        eng.kill()
        raise ProtocolError("DETERMINIZE did not re-emit a query")

    s0, s1 = _state(snap_pl), _state(dq.payload)

    def slice_(name, sl):
        if not np.array_equal(s0[sl], s1[sl]):
            raise ProtocolError(f"DETERMINIZE changed {name} (should be invariant "
                                "to the priority player)")

    try:
        slice_("own hand", slice(_HAND_START, _HAND_START + MAX_HAND_SLOTS))
        slice_("step one-hot",
               slice(_STEP_ONEHOT_START, _STEP_ONEHOT_START + _STEP_ONEHOT_SIZE))
        slice_("battlefield", slice(_SELF_PERM_START, _STACK_START))
        for name, off in (("self life", _SELF_BLOCK_START + _PB_LIFE),
                          ("opp life", _OPP_BLOCK_START + _PB_LIFE),
                          ("self hand count", _SELF_BLOCK_START + _PB_HAND_CT),
                          ("opp hand count", _OPP_BLOCK_START + _PB_HAND_CT),
                          ("self library count", _LIBRARY_CTX_START),
                          ("opp library count", _LIBRARY_CTX_START + 1)):
            if s0[off] != s1[off]:
                raise ProtocolError(f"DETERMINIZE changed {name} "
                                    f"({s0[off]} -> {s1[off]})")
    except ProtocolError:
        eng.kill()
        raise
    rq = eng.restore(0)
    _assert_same_query(rq, snap_pl, snap_nc, "invariants post-RESTORE re-emit")
    eng.kill()
    return "hand/step/battlefield/life/counts invariant under DETERMINIZE; RESTORE exact"


def _write_sb_decks():
    """Stacked bo3 decks for the sideboard-determinize test: identical 60-Swamp
    mains, and a MARKER basic in each sideboard that exists nowhere else (A:
    Plains, B: Mountain). With auto-0 sideboarding (action 0 = done, no swaps)
    the real decks never contain a marker, so a marker card being drawn in a
    simulated world proves the opponent's sideboard entered the exchange pool."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    paths = []
    for name, marker in (("sb_det_a", "Plains"), ("sb_det_b", "Mountain")):
        p = os.path.join(d, f"{name}.dk")
        with open(p, "w") as f:
            f.write(f"60 Swamp\nSIDEBOARD:\n15 {marker}\n")
        paths.append(p)
    return paths


_SB_MARKERS = (b"Plains", b"Mountain")


def _notes_marker_hits(r):
    return sum(1 for n in (r.notes or []) for m in _SB_MARKERS if m in n)


def _advance_to_safe(eng, cur, min_idx, ctx, cap=600):
    """Auto-0 until a safe (safe=1) IN-GAME query at decision index >= min_idx.

    Sideboard prompts (IN/OUT/DONE) are now safe=1 too (loop-safe search roots),
    but they are NOT the in-game decision this anchor wants — a bo3 g2 anchor must
    reach a genuine game-2 decision (match_game_number=1), not a between-game
    sideboard prompt (still match_game_number of the just-ended game). Skip them.
    """
    idx = 0
    while idx < cap:
        if cur.kind != "q":
            raise ProtocolError(f"{ctx}: expected queries while advancing, got "
                                f"{cur.kind} at index {idx}")
        if idx >= min_idx and cur.safe and not _is_sideboard_query(cur):
            return cur
        cur = eng.play(0)
        idx += 1
    raise ProtocolError(f"{ctx}: no safe decision within {cap} decisions")


def _determinize_descend_hits(eng, snap_pl, snap_nc, seeds, depth, ctx):
    """For each world seed: RESTORE (assert exact), DETERMINIZE, auto-0 descend
    `depth` decisions counting marker-card sightings in the narrative. Returns
    total hits. Leaves the engine mid-descent; caller must RESTORE."""
    hits = 0
    for ds in seeds:
        rq = eng.restore(0)
        _assert_same_query(rq, snap_pl, snap_nc, f"{ctx} RESTORE (seed {ds})")
        r = eng.determinize(ds)
        for _ in range(depth):
            r = eng.play(0)
            hits += _notes_marker_hits(r)
            if r.kind != "q":
                break
    return hits


def test_sideboard_determinize():
    """Post-board hidden-sideboard semantics of DETERMINIZE, over a real bo3:

    - game 1 (pre-board): the opponent's sideboard is NOT in the exchange pool —
      marker cards that exist only in sideboards never appear in any sampled
      world's narrative;
    - game 2 (post-board): the opponent's sideboard IS exchangeable with their
      unknown hand/library — sampled worlds draw marker cards even though the
      actual (no-swap) deck contains none;
    - the real line never shows a marker in either game (sim-only mixing), and
      DETERMINIZE leaves both players' hand/library counts unchanged post-board.
    """
    deck_paths = _write_sb_decks()
    eng = Engine(11, extra=["--bo3", "--deck-a", "temp/sb_det_a",
                            "--deck-b", "temp/sb_det_b", "--narrative"])
    try:
        cur = eng.read()
        # ── game 1: pre-board — sideboard must stay out of the pool ──────────
        cur = _advance_to_safe(eng, cur, 15, "g1 anchor")
        snap_pl, snap_nc = cur.payload, cur.nc
        eng.snapshot(0)
        g1_hits = _determinize_descend_hits(eng, snap_pl, snap_nc,
                                            seeds=(1, 2, 3, 4), depth=120,
                                            ctx="g1")
        if g1_hits:
            raise ProtocolError(
                f"pre-board DETERMINIZE leaked {g1_hits} sideboard marker "
                "sighting(s) into game-1 worlds — game 1 must model the known "
                "60, not the 75")
        rq = eng.restore(0)
        _assert_same_query(rq, snap_pl, snap_nc, "g1 final RESTORE")
        rel = eng.release()
        _assert_same_query(rel, snap_pl, snap_nc, "g1 post-RELEASE re-emit")
        cur = rel

        # ── play out game 1 for real; markers must never appear on the real
        # line (auto-0 sideboarding makes no swaps in either direction) ──────
        real_hits = 0
        saw_g1_result = False
        for _ in range(6000):
            cur = eng.play(0)
            real_hits += _notes_marker_hits(cur)
            if cur.kind != "q":
                raise ProtocolError(f"bo3 continuation ended unexpectedly "
                                    f"({cur.kind}) before game 2")
            if cur.note_has(b"GAME_RESULT:"):
                saw_g1_result = True
                break
        if not saw_g1_result:
            raise ProtocolError("game 1 did not finish within 6000 decisions")

        # ── game 2: post-board — opponent sideboard joins the pool ──────────
        cur = _advance_to_safe(eng, cur, 1, "g2 anchor")
        snap_pl, snap_nc = cur.payload, cur.nc
        eng.snapshot(0)
        # Count invariance on one world before the sighting sweep.
        dq = eng.determinize(1)
        s0, s1 = _state(snap_pl), _state(dq.payload)
        for name, off in (("self hand count", _SELF_BLOCK_START + _PB_HAND_CT),
                          ("opp hand count", _OPP_BLOCK_START + _PB_HAND_CT),
                          ("self library count", _LIBRARY_CTX_START),
                          ("opp library count", _LIBRARY_CTX_START + 1)):
            if s0[off] != s1[off]:
                raise ProtocolError(f"post-board DETERMINIZE changed {name} "
                                    f"({s0[off]} -> {s1[off]})")
        g2_hits = _determinize_descend_hits(eng, snap_pl, snap_nc,
                                            seeds=range(1, 9), depth=120,
                                            ctx="g2")
        if g2_hits == 0:
            raise ProtocolError(
                "post-board DETERMINIZE never surfaced a sideboard marker "
                "across 8 worlds x 120 decisions — the opponent's sideboard "
                "does not appear to join the exchange pool in game 2")
        rq = eng.restore(0)
        _assert_same_query(rq, snap_pl, snap_nc, "g2 final RESTORE")
        if real_hits:
            raise ProtocolError(f"{real_hits} marker sighting(s) on the REAL "
                                "line — sideboard mixing must be sim-only")
        return (f"g1 worlds clean (0 marker hits), g2 worlds mixed "
                f"({g2_hits} hits over 8 worlds), real line clean, counts "
                "invariant")
    finally:
        eng.kill()
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def _to_first_sideboard(eng, cap=8000):
    """Auto-0 from the opening read until the first sideboard prompt after game 1
    (the between-game phase). Returns that query (safe=1, IN menu of player A)."""
    cur = eng.read()
    for _ in range(cap):
        if _is_sideboard_query(cur):
            return cur
        cur = eng.play(0)
        if cur.kind != "q":
            raise ProtocolError(f"reached {cur.kind} before any sideboard prompt")
    raise ProtocolError("never reached a sideboard prompt")


def _match_result_from(r):
    for n in (r.notes or []):
        if n.startswith(b"MATCH_RESULT:"):
            return n.decode(errors="replace")
    return None


def _play_to_match_result(eng, cur, cap=40000):
    """Auto-0 to real EOF; return the MATCH_RESULT line seen along the way."""
    result = _match_result_from(cur)
    n = 0
    while cur.kind != "eof" and n < cap:
        cur = eng.play(0)
        m = _match_result_from(cur)
        if m:
            result = m
        n += 1
    if cur.kind != "eof":
        raise ProtocolError("match did not end at EOF")
    if cur.returncode != 0:
        raise ProtocolError(f"match exited with code {cur.returncode}")
    return result


def _control_match_result(seed, extra):
    eng = Engine(seed, extra=extra)
    try:
        return _play_to_match_result(eng, eng.read())
    finally:
        eng.kill()


def test_sideboard_search_roundtrip():
    """The go/no-go proof that a bo3 sideboard prompt is a valid MCTS search root:
    a MATCH-scoped snapshot rolls FORWARD across the init_ecs() game boundary into
    game 2 and RESTOREs the sideboard-node state byte-identically, with a
    deterministic post-restore re-descent, for BOTH an IN-menu and an OUT-menu
    root. Then RESTORE+RELEASE and auto-play to a clean MATCH_RESULT."""
    seed = 13
    extra = ["--bo3", "--deck-a", "temp/sb_det_a", "--deck-b", "temp/sb_det_b"]
    deck_paths = _write_sb_decks()
    try:
        control = _control_match_result(seed, extra)  # no-swap (auto-0) reference
        summaries = []
        for root_kind in ("in", "out"):
            eng = Engine(seed, extra=extra)
            try:
                cur = _to_first_sideboard(eng)  # player A IN menu
                if root_kind == "out":
                    # One SIDEBOARD_IN choice reaches the OUT menu (its pending
                    # decision names the chosen IN card).
                    cur = eng.play(1)
                    if not _is_sideboard_query(cur):
                        raise ProtocolError("expected an OUT menu after a "
                                            "SIDEBOARD_IN choice")
                if not cur.safe:
                    raise ProtocolError(f"{root_kind} sideboard root not safe=1")
                snap_pl, snap_nc = cur.payload, cur.nc
                q = eng.snapshot(0)
                _assert_same_query(q, snap_pl, snap_nc,
                                   f"{root_kind}: post-SNAPSHOT re-emit")

                # Descend across the game boundary: finish both sideboard phases
                # and land in game 2. Record the query stream.
                choices = [0] * 16
                desc1 = []
                reached_g2 = False
                for ch in choices:
                    r = eng.play(ch)
                    if r.kind != "q":
                        raise ProtocolError(f"{root_kind}: descent hit {r.kind} "
                                            "before reaching game 2")
                    desc1.append((r.nc, r.payload, r.safe))
                    if not _is_sideboard_query(r):
                        reached_g2 = True
                        break
                if not reached_g2:
                    raise ProtocolError(f"{root_kind}: never left the sideboard "
                                        "phase in 16 steps")

                # RESTORE: byte-identical return to the sideboard root.
                rq = eng.restore(0)
                _assert_same_query(rq, snap_pl, snap_nc,
                                   f"{root_kind}: post-RESTORE re-emit")
                # Re-descend with the identical choices: schedule determinism.
                for i in range(len(desc1)):
                    r = eng.play(choices[i])
                    _assert_same_query(r, desc1[i][1], desc1[i][0],
                                       f"{root_kind}: re-descend step {i}")

                # RESTORE + RELEASE, then auto-play the REAL match to completion.
                eng.restore(0)
                rel = eng.release()
                _assert_same_query(rel, snap_pl, snap_nc,
                                   f"{root_kind}: post-RELEASE re-emit")
                result = _play_to_match_result(eng, rel)
                if root_kind == "in":
                    # Auto-0 from the IN menu makes no swaps -> identical to control.
                    if result != control:
                        raise ProtocolError(
                            f"in-root released line diverged from control: "
                            f"{result!r} vs {control!r}")
                else:
                    # OUT root auto-0 performs a real swap -> line legitimately
                    # differs from the no-swap control; just require a clean finish.
                    if not result:
                        raise ProtocolError("out-root released line produced no "
                                            "MATCH_RESULT")
                summaries.append(f"{root_kind}:{len(desc1)} steps")
            finally:
                eng.kill()
        return f"IN+OUT roots byte-identical across boundary ({', '.join(summaries)}); " \
               f"control={control!r}"
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_sideboard_sim_result():
    """From a MATCH-scoped sideboard root with a snapshot live, a simulated line
    driven into game 2 that reaches game over yields SIM_RESULT (not GAME_RESULT /
    process exit); RESTORE then returns to the sideboard root byte-identically."""
    seed = 13
    extra = ["--bo3", "--deck-a", "temp/sb_det_a", "--deck-b", "temp/sb_det_b"]
    deck_paths = _write_sb_decks()
    try:
        eng = Engine(seed, extra=extra)
        try:
            cur = _to_first_sideboard(eng)
            if not cur.safe:
                raise ProtocolError("sideboard root not safe=1")
            snap_pl, snap_nc = cur.payload, cur.nc
            eng.snapshot(0)
            sim = None
            r = cur
            for _ in range(8000):
                r = eng.play(0)
                if r.kind == "sim":
                    sim = r
                    break
                if r.kind == "eof":
                    break
                if r.kind == "q" and r.note_has(b"GAME_RESULT:"):
                    raise ProtocolError("simulated line printed a real GAME_RESULT "
                                        "instead of being intercepted as SIM_RESULT")
            if sim is None:
                raise ProtocolError(f"no SIM_RESULT within the descent (got {r.kind}) "
                                    "— a live match snapshot must intercept game end")
            if sim.result not in ("A", "B", "DRAW"):
                raise ProtocolError(f"unexpected SIM_RESULT payload {sim.result!r}")
            rq = eng.restore(0)
            _assert_same_query(rq, snap_pl, snap_nc,
                               "sim-result post-RESTORE re-emit")
            return f"SIM_RESULT={sim.result} across the sideboard boundary, RESTORE exact"
        finally:
            eng.kill()
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def _write_var_decks():
    """Stacked bo3 decks with basic-land VARIETY (unlike the all-Swamp sb_det
    decks) so a different next-game shuffle produces a visibly different opening
    deal in the observation state vector. All-lands mirrors still deck out under
    auto-0, so game 1 ends with a real winner and the match reaches MATCH_RESULT.
    Both seats get the same 60 (15 each of Swamp/Island/Forest/Mountain) and a
    Plains sideboard (never sided in under auto-0)."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    paths = []
    body = ("15 Swamp\n15 Island\n15 Forest\n15 Mountain\n"
            "SIDEBOARD:\n15 Plains\n")
    for name in ("var_det_a", "var_det_b"):
        p = os.path.join(d, f"{name}.dk")
        with open(p, "w") as f:
            f.write(body)
        paths.append(p)
    return paths


def _record_full_stream(seed, extra, cap=40000):
    """Auto-0 a whole match from the opening read to real EOF, recording every
    query's (nc, payload, safe). Returns (records, match_result)."""
    eng = Engine(seed, extra=extra)
    try:
        records = []
        cur = eng.read()
        result = _match_result_from(cur)
        n = 0
        while cur.kind == "q" and n < cap:
            records.append((cur.nc, cur.payload, cur.safe))
            cur = eng.play(0)
            m = _match_result_from(cur)
            if m:
                result = m
            n += 1
        if cur.kind != "eof":
            raise ProtocolError(f"control match did not reach EOF (got {cur.kind})")
        return records, result
    finally:
        eng.kill()


def _descend_record(eng, depth):
    """Play [0]*depth, recording (kind, nc, payload, is_sb) per step; stop early
    at a non-query (terminal/eof)."""
    stream = []
    for _ in range(depth):
        r = eng.play(0)
        if r.kind != "q":
            stream.append((r.kind, None, None, None))
            break
        stream.append(("q", r.nc, r.payload, _is_sideboard_query(r)))
    return stream


def _payloads(stream):
    return [s[2] for s in stream]


def test_sideboard_world_variation():
    """A sideboard-root DETERMINIZE samples WHICH next-game deal this world gets
    (Stage 3). At a bo3 sideboard root with a live snapshot:

    - DETERMINIZE(s1), descend a fixed choice line across the game boundary into
      game 2 -> record the query stream (its game-2 portion reflects the deal);
    - RESTORE, DETERMINIZE(s2 != s1), same line -> the game-2 stream DIFFERS
      (a different shuffle/deal);
    - RESTORE, DETERMINIZE(s1) again, same line -> byte-identical to the first
      recording (same world => same deal => same menus);
    - RESTORE, RELEASE, auto-play the REAL match to completion -> byte-identical
      to a clean control run of the same seed with no snapshot/determinize
      commands (salt isolation: the real game 2 is unaffected).
    """
    seed = 13
    extra = ["--bo3", "--deck-a", "temp/var_det_a", "--deck-b", "temp/var_det_b"]
    deck_paths = _write_var_decks()
    try:
        control, control_result = _record_full_stream(seed, extra)

        eng = Engine(seed, extra=extra)
        try:
            # Auto-0 to the first sideboard root (player A IN menu), recording the
            # prefix so we can prove the real line matches the control end to end.
            prefix = []
            cur = eng.read()
            for _ in range(8000):
                if _is_sideboard_query(cur):
                    break
                if cur.kind != "q":
                    raise ProtocolError(f"reached {cur.kind} before a sideboard root")
                prefix.append((cur.nc, cur.payload, cur.safe))
                cur = eng.play(0)
            else:
                raise ProtocolError("never reached a sideboard root")
            if not cur.safe:
                raise ProtocolError("sideboard root not safe=1")
            snap_pl, snap_nc = cur.payload, cur.nc
            q = eng.snapshot(0)
            _assert_same_query(q, snap_pl, snap_nc, "world-var post-SNAPSHOT re-emit")

            # A fixed choice line: A done, B done, then into game 2. 24 steps lands
            # well inside game 2 where the deal is reflected in the observation.
            depth = 24
            # s1 vs s2 must produce different deals at this seed; s1==s3 identical.
            s1, s2 = 1, 2

            dq1 = eng.determinize(s1)
            _assert_same_query(dq1, snap_pl, snap_nc,
                               "world-var DETERMINIZE re-emits the (unchanged) root")
            stream1 = _descend_record(eng, depth)

            rq = eng.restore(0)
            _assert_same_query(rq, snap_pl, snap_nc, "world-var RESTORE #1")
            eng.determinize(s2)
            stream2 = _descend_record(eng, depth)

            rq = eng.restore(0)
            _assert_same_query(rq, snap_pl, snap_nc, "world-var RESTORE #2")
            eng.determinize(s1)
            stream3 = _descend_record(eng, depth)

            # Same world (s1) => identical deal => byte-identical descent.
            if _payloads(stream1) != _payloads(stream3):
                d = None
                for i, (a, b) in enumerate(zip(_payloads(stream1), _payloads(stream3))):
                    if a != b:
                        d = i
                        break
                raise ProtocolError(f"same-world (seed {s1}) descents diverged at "
                                    f"step {d} — determinization is not reproducible")

            # Different worlds (s1 vs s2) => different deal. The sideboard portion
            # of the descent is identical (salt does not touch the sideboard
            # prompt); the divergence must appear in the game-2 (non-sideboard)
            # portion. (Rarely two salts could share an opening prefix — s1/s2 are
            # chosen to differ at this seed; the deep 24-step window makes a full
            # collision astronomically unlikely.)
            p1, p2 = _payloads(stream1), _payloads(stream2)
            if p1 == p2:
                raise ProtocolError(
                    f"different world seeds ({s1} vs {s2}) produced an identical "
                    f"{depth}-step descent — sideboard-root DETERMINIZE did not "
                    "vary the next-game deal")
            first_diff = next(i for i in range(min(len(p1), len(p2)))
                              if p1[i] != p2[i])
            if stream1[first_diff][3]:  # is_sb flag on the first differing step
                raise ProtocolError(
                    f"world seeds diverged inside a SIDEBOARD prompt (step "
                    f"{first_diff}) — the salt must only affect the game-2 deal")

            # RESTORE + RELEASE, then auto-play the REAL match. Reassemble the real
            # query stream (prefix + released tail) and assert it is byte-identical
            # to the clean control run: salt back to 0, game 2 unaffected.
            eng.restore(0)
            rel = eng.release()
            _assert_same_query(rel, snap_pl, snap_nc, "world-var post-RELEASE re-emit")
            tail = [(rel.nc, rel.payload, rel.safe)]
            cur = rel
            result = _match_result_from(cur)
            for _ in range(40000):
                cur = eng.play(0)
                m = _match_result_from(cur)
                if m:
                    result = m
                if cur.kind != "q":
                    break
                tail.append((cur.nc, cur.payload, cur.safe))
            if cur.kind != "eof":
                raise ProtocolError(f"real line did not reach EOF (got {cur.kind})")

            real = prefix + tail
            if len(real) != len(control):
                raise ProtocolError(f"real line length {len(real)} != control "
                                    f"{len(control)} — salt leaked into the real game")
            for i, (a, b) in enumerate(zip(real, control)):
                if a[1] != b[1]:
                    raise ProtocolError(f"real line diverged from control at query "
                                        f"{i} (byte {_first_diff(a[1], b[1])}) — "
                                        "sideboard-root salt is not isolated")
            if result != control_result:
                raise ProtocolError(f"real MATCH_RESULT {result!r} != control "
                                    f"{control_result!r}")
            return (f"same-world identical ({depth} steps), worlds {s1}/{s2} differ "
                    f"at game-2 step {first_diff}, real line == control "
                    f"({len(real)} queries, {result!r})")
        finally:
            eng.kill()
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_perf():
    """Time SNAPSHOT+RESTORE pairs at one decision. Reported for CI logs; WARNs
    (does not fail) above a lenient debug-build threshold — the sub-ms bar is a
    release-build concern."""
    seed = 5
    anchor = 15
    pairs = 300
    warn_ms = 5.0
    eng = Engine(seed)
    cur = eng.read()
    for idx in range(anchor):
        cur = eng.play(0)
    if not cur.safe:
        eng.kill()
        raise ProtocolError("perf anchor not safe")
    snap_pl, snap_nc = cur.payload, cur.nc
    eng.snapshot(0)
    t0 = time.perf_counter()
    for _ in range(pairs):
        q = eng.snapshot(0)
        r = eng.restore(0)
    dt = time.perf_counter() - t0
    # Correctness is still asserted on the last pair.
    _assert_same_query(q, snap_pl, snap_nc, "perf SNAPSHOT re-emit")
    _assert_same_query(r, snap_pl, snap_nc, "perf RESTORE re-emit")
    eng.kill()
    per = dt / pairs * 1000.0
    warn = per > warn_ms
    detail = f"{per:.3f} ms/pair over {pairs} pairs (incl. stdio round-trip)"
    if warn:
        detail += f"  [WARN > {warn_ms} ms — expected on a debug build]"
    return detail


TESTS = [
    ("round_trip", test_round_trip),
    ("rng_isolation", test_rng_isolation),
    ("determinize_efficacy", test_determinize_efficacy),
    ("terminal_intercept", test_terminal_intercept),
    ("determinize_invariants", test_determinize_invariants),
    ("sideboard_determinize", test_sideboard_determinize),
    ("sideboard_search_roundtrip", test_sideboard_search_roundtrip),
    ("sideboard_sim_result", test_sideboard_sim_result),
    ("sideboard_world_variation", test_sideboard_world_variation),
    ("perf", test_perf),
]


def main():
    if not os.path.exists(BINARY):
        print(f"binary not found at {BINARY} — run `make` first", file=sys.stderr)
        return 2
    failures = 0
    for name, fn in TESTS:
        try:
            detail = fn()
            print(f"ok    {name}: {detail}", flush=True)
        except (ProtocolError, EOFError, AssertionError) as e:
            failures += 1
            print(f"FAIL  {name}: {e}", flush=True)
    print(f"\nsnapshot search-server: {len(TESTS) - failures}/{len(TESTS)} tests "
          f"passed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
