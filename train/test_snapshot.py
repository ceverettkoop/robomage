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
                    CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT, CAT_SIDEBOARD_DONE,
                    CAT_ATTACK_TARGET, CAT_BLOCK_TARGET, CAT_ASSIGN_DAMAGE,
                    CAT_OTHER_CHOICE, CAT_CAST_SPELL, CAT_TOP_LIBRARY,
                    CAT_BOTTOM_DECK_CARD, CAT_ORDER_TRIGGERS, CAT_SELECT_TARGET,
                    CAT_PLAY_LAND, CAT_KEEP_LEGEND, CAT_CHOOSE_TYPE,
                    CAT_NAME_CARD, CAT_PAY_UNLESS, CAT_CHOOSE_CARD,
                    CAT_CHOOSE_X, CAT_PAYING_COSTS, CAT_CHOOSE_MODE,
                    CAT_ACTIVATE_ABILITY, CAT_KEEP_HAND, CAT_MULLIGAN,
                    CAT_CHOOSE_REPLACEMENT)
from card_costs import _VOCAB_NAMES
from env import (
    N_CARD_TYPES, MAX_HAND_SLOTS,
    _HAND_START, _SELF_BLOCK_START, _OPP_BLOCK_START, _PB_LIFE, _PB_HAND_CT,
    _LIBRARY_CTX_START, _STEP_ONEHOT_START, _STEP_ONEHOT_SIZE,
    _SELF_PERM_START, _STACK_START)
from cli_spec import BINARY, BIN_DIR

# The binary payload framing after every BQUERY header, WITHOUT --narrative:
# STATE_SIZE float32 state, then seven MAX_ACTIONS-wide metadata arrays
# (cats int32, ids f32, ctrl f32, pub f32, zone int32, refs int32, ords int32).
# This is the exact size cli_output.cpp emits; a short/over read desyncs the
# whole stream.
PAYLOAD_BYTES = STATE_SIZE * 4 + 7 * MAX_ACTIONS * 4


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


# The per-action card-id floats (card_vocab_index / N_CARD_TYPES) are the SECOND
# metadata array, right after the category ints.
_IDS_OFF = _CATS_OFF + MAX_ACTIONS * 4


def _query_ids(payload):
    return np.frombuffer(payload[_IDS_OFF:_IDS_OFF + MAX_ACTIONS * 4], dtype=np.float32)


# The per-action option_ordinal ints are the SEVENTH (last) metadata array
# (cats, ids, ctrl, pub, zone, refs, ords). For a PAY_UNLESS menu the ordinal
# distinguishes pay (1) from decline (0); the interleaved tap-for-mana actions
# carry -1 (not applicable).
_ORDS_OFF = _CATS_OFF + 6 * MAX_ACTIONS * 4


def _query_ords(payload):
    return np.frombuffer(payload[_ORDS_OFF:_ORDS_OFF + MAX_ACTIONS * 4], dtype=np.int32)


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

def record_line(seed, policy, cap=4000, extra=()):
    """Play one full game with `policy(idx, nc) -> action` and no snapshotting;
    return (records, outcome). records[i] = (nc, payload, safe). outcome is a
    dict {kind, returncode, winner} at real EOF (games never draw here)."""
    eng = Engine(seed, extra=extra)
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
    since the Batch 13 pregame gate the mulligan decisions are loop-top pending
    decisions too, so the game opens safe=1 from decision 0."""
    seed = 5
    control, outcome = record_line(seed, _mod_policy)
    n = len(control)

    safes = [s for (_, _, s) in control]
    if not all(safes[:4]):
        raise ProtocolError("expected the pregame mulligan decisions to report "
                            "safe=1 (Batch 13 pregame gate); got safe flags "
                            f"{safes[:4]}")
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


def _write_combat_decks():
    """Lands-only stacked decks for the combat sub-prompt roots test. The combat
    board comes from --battlefield presets: A attacks with a Grizzly Bears into
    B's Jace, the Mind Sculptor + Grizzly Bears, so A's declare-attackers reaches
    a real attack-target choice (player + planeswalker, 2 options) and B's
    declare-blockers reaches the block-target prompt (always asked, here with 1
    legal attacker). Lands-only libraries deck the game out under auto-0."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    paths = []
    for name, land in (("combat_pq_a", "Mountain"), ("combat_pq_b", "Forest")):
        p = os.path.join(d, f"{name}.dk")
        with open(p, "w") as f:
            f.write(f"30 {land}\n")
        paths.append(p)
    return paths


_COMBAT_EXTRA = ["--deck-a", "temp/combat_pq_a", "--deck-b", "temp/combat_pq_b",
                 "--no-shuffle",
                 "--battlefield-a", "Grizzly Bears",
                 "--battlefield-b", "Jace the Mind Sculptor,Grizzly Bears"]


def _first_cat_index(control, cat):
    """Index of the first control decision whose menu contains category `cat`."""
    for i, (nc, pl, _safe) in enumerate(control):
        if bool((_query_cats(pl)[:nc] == cat).any()):
            return i
    return None


def test_combat_target_roundtrip():
    """Batch 1 (snapshot-safe combat): the attack-target and block-target
    sub-prompts are loop-top pending decisions — both report safe=1 and are
    valid SNAPSHOT/RESTORE roots. At each root: SNAPSHOT re-emits exactly, a
    divergent line (the other attack target / a scrambled continuation) then
    RESTORE returns byte-identically, and the resumed real line stays
    byte-identical to a no-snapshot control run with the same outcome. Also
    pins the pre-collapse conventions: the attack-target menu has 2 options
    (defending player + planeswalker; a single-target board never prompts) and
    the block-target menu is emitted even at size 1 (no pre-collapse exists)."""
    seed = 5
    deck_paths = _write_combat_decks()
    try:
        control, outcome = record_line(seed, _auto0, extra=_COMBAT_EXTRA)
        atk_idx = _first_cat_index(control, CAT_ATTACK_TARGET)
        blk_idx = _first_cat_index(control, CAT_BLOCK_TARGET)
        if atk_idx is None:
            raise ProtocolError("no attack-target decision in the control line — "
                                "the planeswalker board should force one")
        if blk_idx is None:
            raise ProtocolError("no block-target decision in the control line — "
                                "a declared blocker always prompts")
        if not (atk_idx < blk_idx):
            raise ProtocolError(f"attack-target ({atk_idx}) should precede "
                                f"block-target ({blk_idx})")
        if control[atk_idx][0] != 2:
            raise ProtocolError(f"attack-target menu has {control[atk_idx][0]} "
                                "options, expected 2 (player + planeswalker)")
        if control[blk_idx][0] != 1:
            raise ProtocolError(f"block-target menu has {control[blk_idx][0]} "
                                "options, expected the un-collapsed size-1 menu")
        for name, i in (("attack-target", atk_idx), ("block-target", blk_idx)):
            if not control[i][2]:
                raise ProtocolError(f"{name} decision reports safe=0 — it should "
                                    "be a loop-top pending decision now")

        eng = Engine(seed, extra=_COMBAT_EXTRA)
        cur = eng.read()
        excursions = 0
        for idx in range(len(control)):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"combat alignment at decision {idx}")
            if idx in (atk_idx, blk_idx):
                snap_nc, snap_pl = cur.nc, cur.payload
                q = eng.snapshot(0)
                _assert_same_query(q, snap_pl, snap_nc,
                                   f"combat post-SNAPSHOT re-emit at {idx}")
                # Divergent excursion: last menu option first (at the attack
                # root that is the planeswalker, not the control's player
                # choice), then a scrambled continuation.
                dq = q
                for i in range(8):
                    dq = eng.play(dq.nc - 1 if i == 0 else _diverge(i, dq.nc))
                    if dq.kind != "q":
                        break
                rq = eng.restore(0)
                _assert_same_query(rq, snap_pl, snap_nc,
                                   f"combat post-RESTORE re-emit at {idx}")
                rel = eng.release()
                _assert_same_query(rel, snap_pl, snap_nc,
                                   f"combat post-RELEASE re-emit at {idx}")
                cur = rel
                excursions += 1
            cur = eng.play(_auto0(idx, cur.nc))
        if cur.kind != "eof" or cur.returncode != 0:
            eng.kill()
            raise ProtocolError(f"resumed combat line ended abnormally: "
                                f"{cur.kind} rc={cur.returncode}")
        if cur.winner != outcome["winner"]:
            eng.kill()
            raise ProtocolError(f"outcome mismatch: control {outcome['winner']!r} "
                                f"vs resumed {cur.winner!r}")
        eng.kill()
        if excursions != 2:
            raise ProtocolError(f"expected 2 combat excursions, ran {excursions}")
        return (f"attack-target @ {atk_idx} (nc=2) and block-target @ {blk_idx} "
                f"(nc=1) both safe=1, round-trips exact, outcome="
                f"{outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def _write_damage_decks():
    """Lands-only stacked decks for the damage-assignment roots test. The combat
    board comes from --battlefield presets: A's Grizzly Bears (2/2) attacks and
    B's two Flying Men (1/1) both block it under auto-0, so the attacker's power
    (2) equals the total lethal needed (1+1) and the regular damage step reaches
    the prompted division — two consecutive DAMAGE_ASSIGN picks (menu sizes 3
    then 2: each pick re-arms with the shrunken pool). Lands-only libraries deck
    the game out under auto-0."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    paths = []
    for name, land in (("dmg_pq_a", "Mountain"), ("dmg_pq_b", "Forest")):
        p = os.path.join(d, f"{name}.dk")
        with open(p, "w") as f:
            f.write(f"30 {land}\n")
        paths.append(p)
    return paths


_DAMAGE_EXTRA = ["--deck-a", "temp/dmg_pq_a", "--deck-b", "temp/dmg_pq_b",
                 "--no-shuffle",
                 "--battlefield-a", "Grizzly Bears",
                 "--battlefield-b", "Flying Men,Flying Men"]


def test_damage_assign_roundtrip():
    """Batch 2 (snapshot-safe combat): the combat damage-assignment ordering
    prompt is a loop-top pending decision — every pick reports safe=1 and is a
    valid SNAPSHOT/RESTORE root. The bear-vs-two-Flying-Men board yields two
    consecutive picks (nc=3: both blockers + Done, then nc=2 after the pool
    shrinks — the resume→re-arm path). At each root: SNAPSHOT re-emits exactly,
    a divergent assignment order (Done first, dumping the leftover power) then
    RESTORE returns byte-identically, and the resumed real line stays
    byte-identical to a no-snapshot control run with the same outcome."""
    seed = 5
    deck_paths = _write_damage_decks()
    try:
        control, outcome = record_line(seed, _auto0, extra=_DAMAGE_EXTRA)
        da_idx = _first_cat_index(control, CAT_ASSIGN_DAMAGE)
        if da_idx is None:
            raise ProtocolError("no damage-assignment decision in the control "
                                "line — the two-blocker board should force one")
        if control[da_idx][0] != 3:
            raise ProtocolError(f"first assignment menu has {control[da_idx][0]} "
                                "options, expected 3 (two blockers + Done)")
        nxt = control[da_idx + 1]
        if not bool((_query_cats(nxt[1])[:nxt[0]] == CAT_ASSIGN_DAMAGE).any()):
            raise ProtocolError("decision after the first pick is not the second "
                                "assignment pick — the resume should re-arm")
        if nxt[0] != 2:
            raise ProtocolError(f"second assignment menu has {nxt[0]} options, "
                                "expected 2 (remaining blocker + Done)")
        roots = (da_idx, da_idx + 1)
        for i in roots:
            if not control[i][2]:
                raise ProtocolError(f"assignment decision {i} reports safe=0 — "
                                    "it should be a loop-top pending decision now")

        eng = Engine(seed, extra=_DAMAGE_EXTRA)
        cur = eng.read()
        excursions = 0
        for idx in range(len(control)):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"damage alignment at decision {idx}")
            if idx in roots:
                snap_nc, snap_pl = cur.nc, cur.payload
                q = eng.snapshot(0)
                _assert_same_query(q, snap_pl, snap_nc,
                                   f"damage post-SNAPSHOT re-emit at {idx}")
                # Divergent excursion: Done first (dumps the unassigned power
                # instead of the control's lethal pick), then a scrambled
                # continuation.
                dq = q
                for i in range(8):
                    dq = eng.play(dq.nc - 1 if i == 0 else _diverge(i, dq.nc))
                    if dq.kind != "q":
                        break
                rq = eng.restore(0)
                _assert_same_query(rq, snap_pl, snap_nc,
                                   f"damage post-RESTORE re-emit at {idx}")
                rel = eng.release()
                _assert_same_query(rel, snap_pl, snap_nc,
                                   f"damage post-RELEASE re-emit at {idx}")
                cur = rel
                excursions += 1
            cur = eng.play(_auto0(idx, cur.nc))
        if cur.kind != "eof" or cur.returncode != 0:
            eng.kill()
            raise ProtocolError(f"resumed damage line ended abnormally: "
                                f"{cur.kind} rc={cur.returncode}")
        if cur.winner != outcome["winner"]:
            eng.kill()
            raise ProtocolError(f"outcome mismatch: control {outcome['winner']!r} "
                                f"vs resumed {cur.winner!r}")
        eng.kill()
        if excursions != 2:
            raise ProtocolError(f"expected 2 damage excursions, ran {excursions}")
        return (f"assignment picks @ {da_idx} (nc=3) and {da_idx + 1} (nc=2) "
                f"both safe=1, round-trips exact, outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def _write_resolution_decks():
    """Stacked decks for the resolution-suspension roots test. A's Delver of
    Secrets (preset on the battlefield) fires its upkeep trigger every turn:
    the trigger resolves through effects::peek_and_reveal, whose reveal y/n
    prompt is now a suspended loop-top pending decision (Batch 3). A's deck is
    stacked so the 8th card — the library top at the first upkeep, before any
    draw — is a Lightning Bolt among Mountains, making the determinize pinning
    observable: only the pin keeps the Bolt on top of an otherwise
    indistinguishable all-Mountain library. Lands elsewhere deck the game out
    under auto-0 (Bolt is never cast: action 0 always passes)."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    pa = os.path.join(d, "res_pq_a.dk")
    with open(pa, "w") as f:
        f.write("7 Mountain\n1 Lightning Bolt\n22 Mountain\n")
    pb = os.path.join(d, "res_pq_b.dk")
    with open(pb, "w") as f:
        f.write("30 Forest\n")
    return [pa, pb]


_RESOLUTION_EXTRA = ["--deck-a", "temp/res_pq_a", "--deck-b", "temp/res_pq_b",
                     "--no-shuffle", "--narrative",
                     "--battlefield-a", "Delver of Secrets"]


def test_resolution_single_roundtrip():
    """Batch 3 (resolution suspension): a mid-resolution effect prompt — the
    Delver upkeep peek/reveal y/n from effects::peek_and_reveal — is a loop-top
    pending decision: it reports safe=1 and is a valid SNAPSHOT/RESTORE root.
    It is the first OTHER_CHOICE decision of the game (upkeep precedes the
    first main-phase menu; the only earlier queries are the pregame mulligans,
    themselves safe since Batch 13), a 2-option menu. At the root: SNAPSHOT
    re-emits exactly; a
    divergent line (Reveal instead of the control's Don't-reveal — transforming
    Delver) then RESTORE returns byte-identically; the resumed real line stays
    byte-identical to a no-snapshot control run with the same outcome. Also the
    first determinize-pinning smoke: a DETERMINIZE at the root succeeds, the
    re-emitted query is unchanged, and playing 'Reveal' inside that world still
    reveals the Lightning Bolt the parked menu refers to (the pin kept it on
    top through the world reshuffle); RESTORE then reverts byte-for-byte."""
    seed = 5
    deck_paths = _write_resolution_decks()
    try:
        control, outcome = record_line(seed, _auto0, extra=_RESOLUTION_EXTRA)
        root_idx = _first_cat_index(control, CAT_OTHER_CHOICE)
        if root_idx is None:
            raise ProtocolError("no OTHER_CHOICE decision in the control line")
        if root_idx == 0:
            raise ProtocolError("first decision is the peek/reveal — expected "
                                "the pregame mulligans first")
        nc, pl, safe = control[root_idx]
        if nc != 2:
            raise ProtocolError(f"peek/reveal decision has {nc} options, expected "
                                "the 2-option peek/reveal prompt")
        cats = _query_cats(pl)
        if not bool((cats[:nc] == CAT_OTHER_CHOICE).all()):
            raise ProtocolError(f"peek/reveal decision categories {cats[:nc]} are "
                                f"not the peek/reveal OTHER_CHOICE ({CAT_OTHER_CHOICE})")
        if not safe:
            raise ProtocolError("peek/reveal decision reports safe=0 — it should "
                                "be a loop-top pending decision")

        eng = Engine(seed, extra=_RESOLUTION_EXTRA)
        cur = eng.read()
        for idx in range(root_idx):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"resolution alignment at decision {idx}")
            cur = eng.play(_auto0(idx, cur.nc))
        _assert_same_query(cur, pl, nc, "resolution root alignment")
        if not cur.safe:
            eng.kill()
            raise ProtocolError("peek/reveal prompt reports safe=0 — it should be "
                                "a loop-top pending decision now")
        q = eng.snapshot(0)
        _assert_same_query(q, pl, nc, "resolution post-SNAPSHOT re-emit")

        # Divergent excursion: Reveal (choice 1, transforming Delver on the
        # instant on top), then a scrambled continuation.
        dq = q
        for i in range(8):
            dq = eng.play(1 if i == 0 else _diverge(i, dq.nc))
            if dq.kind != "q":
                break
        rq = eng.restore(0)
        _assert_same_query(rq, pl, nc, "resolution post-RESTORE re-emit")

        # Determinize pinning smoke: the parked menu references the library top
        # card (the stacked Lightning Bolt). DETERMINIZE must succeed, leave
        # the re-emitted query unchanged, and keep the Bolt pinned on top — so
        # a 'Reveal' inside the world still reveals Lightning Bolt.
        dq = eng.determinize(3)
        _assert_same_query(dq, pl, nc, "resolution post-DETERMINIZE re-emit")
        wq = eng.play(1)  # Reveal
        if not wq.note_has(b"Revealed: Lightning Bolt"):
            eng.kill()
            raise ProtocolError("determinized world did not reveal the pinned "
                                "Lightning Bolt — the parked menu's library top "
                                "card was shuffled away (pinning failed)")
        rq = eng.restore(0)
        _assert_same_query(rq, pl, nc, "resolution post-pin-world RESTORE re-emit")

        rel = eng.release()
        _assert_same_query(rel, pl, nc, "resolution post-RELEASE re-emit")
        cur = rel
        for idx in range(root_idx, len(control)):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"resolution resumed alignment at decision {idx}")
            cur = eng.play(_auto0(idx, cur.nc))
        if cur.kind != "eof" or cur.returncode != 0:
            eng.kill()
            raise ProtocolError(f"resumed resolution line ended abnormally: "
                                f"{cur.kind} rc={cur.returncode}")
        if cur.winner != outcome["winner"]:
            eng.kill()
            raise ProtocolError(f"outcome mismatch: control {outcome['winner']!r} "
                                f"vs resumed {cur.winner!r}")
        eng.kill()
        return (f"peek/reveal root @ {root_idx} (nc=2) safe=1, round-trip exact, "
                f"pinned Bolt revealed in world, outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def _write_dredge_decks():
    """Stacked decks for the dredge draw-replacement test (Batch 14). Life from
    the Loam (Dredge 3, the one dredge card in vocab) is preset in A's
    graveyard, so EVERY turn-based draw of A's offers the 2-option
    draw-or-dredge menu (CR 702.52a) — now a TURN_DRAW loop-top pending
    decision instead of a blocking prompt inside advance_step. All-land decks
    keep the auto-0 control line trivial (draw normally every time; the game
    ends by deck-out)."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    pa = os.path.join(d, "dredge_pq_a.dk")
    with open(pa, "w") as f:
        f.write("30 Mountain\n")
    pb = os.path.join(d, "dredge_pq_b.dk")
    with open(pb, "w") as f:
        f.write("30 Forest\n")
    return [pa, pb]


_DREDGE_EXTRA = ["--deck-a", "temp/dredge_pq_a", "--deck-b", "temp/dredge_pq_b",
                 "--no-shuffle", "--narrative",
                 "--graveyard-a", "Life from the Loam"]


def test_dredge_roundtrip():
    """Batch 14 (turn-based draw suspension): the dredge draw-replacement
    question at A's draw step — armed by resume_pending_draws inside
    advance_step's UPKEEP→DRAW case — is a loop-top pending decision (tag
    TURN_DRAW): it reports safe=1 and is a valid SNAPSHOT/RESTORE root. At the
    first such root: SNAPSHOT re-emits exactly; a divergent line (Dredge
    instead of the control's draw — milling 3 and returning Loam to hand) then
    RESTORE returns byte-identically; a DETERMINIZE at the root succeeds, the
    re-emitted query is unchanged, and playing 'Dredge' inside that world still
    dredges Life from the Loam (the graveyard is public and the parked menu's
    card pinned); the resumed real line stays byte-identical to a no-snapshot
    control run with the same outcome."""
    seed = 5
    deck_paths = _write_dredge_decks()
    try:
        control, outcome = record_line(seed, _auto0, extra=_DREDGE_EXTRA)
        root_idx = _first_cat_index(control, CAT_CHOOSE_REPLACEMENT)
        if root_idx is None:
            raise ProtocolError("no CHOOSE_REPLACEMENT decision in the control line")
        nc, pl, safe = control[root_idx]
        if nc != 2:
            raise ProtocolError(f"dredge decision has {nc} options, expected the "
                                "2-option draw-or-dredge menu")
        if not safe:
            raise ProtocolError("dredge decision reports safe=0 — it should be a "
                                "loop-top TURN_DRAW pending decision")

        eng = Engine(seed, extra=_DREDGE_EXTRA)
        cur = eng.read()
        for idx in range(root_idx):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"dredge alignment at decision {idx}")
            cur = eng.play(_auto0(idx, cur.nc))
        _assert_same_query(cur, pl, nc, "dredge root alignment")
        if not cur.safe:
            eng.kill()
            raise ProtocolError("dredge prompt reports safe=0 — it should be a "
                                "loop-top TURN_DRAW pending decision now")
        q = eng.snapshot(0)
        _assert_same_query(q, pl, nc, "dredge post-SNAPSHOT re-emit")

        # Divergent excursion: Dredge (choice 1 — mill 3, Loam to hand, no
        # draw), then a scrambled continuation.
        dq = q
        for i in range(8):
            dq = eng.play(1 if i == 0 else _diverge(i, dq.nc))
            if dq.kind != "q":
                break
        rq = eng.restore(0)
        _assert_same_query(rq, pl, nc, "dredge post-RESTORE re-emit")

        # Determinize smoke: the graveyard is public (never resampled) and the
        # parked menu's Loam is pinned, so a 'Dredge' inside the sampled world
        # must still resolve against it.
        dq = eng.determinize(3)
        _assert_same_query(dq, pl, nc, "dredge post-DETERMINIZE re-emit")
        wq = eng.play(1)  # Dredge
        if not wq.note_has(b"dredges Life from the Loam"):
            eng.kill()
            raise ProtocolError("determinized world did not dredge Life from the "
                                "Loam — the parked dredge menu no longer resolves")
        rq = eng.restore(0)
        _assert_same_query(rq, pl, nc, "dredge post-world RESTORE re-emit")

        rel = eng.release()
        _assert_same_query(rel, pl, nc, "dredge post-RELEASE re-emit")
        cur = rel
        for idx in range(root_idx, len(control)):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"dredge resumed alignment at decision {idx}")
            cur = eng.play(_auto0(idx, cur.nc))
        if cur.kind != "eof" or cur.returncode != 0:
            eng.kill()
            raise ProtocolError(f"resumed dredge line ended abnormally: "
                                f"{cur.kind} rc={cur.returncode}")
        if cur.winner != outcome["winner"]:
            eng.kill()
            raise ProtocolError(f"outcome mismatch: control {outcome['winner']!r} "
                                f"vs resumed {cur.winner!r}")
        eng.kill()
        return (f"dredge root @ {root_idx} (nc=2) safe=1, round-trip exact, "
                f"world dredge resolves, outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def _write_pool_decks():
    """Stacked decks for the frozen-pool (Batch 4) tests. With --no-shuffle A's
    opening hand is the first 7 deck cards (Preordain + 6 Islands); an Island
    battlefield preset pays the {U}, so a cast-first policy casts Preordain on
    A's first turn and reaches its scry-2 loop — two consecutive per-card
    keep/bottom picks over a FROZEN 2-card pool (Lightning Bolt then Grizzly
    Bears, the 8th/9th deck cards). The rest of A's library is a VARIED run of
    basics so a determinize resample of the unpinned cards becomes observable in
    later draws; B's all-Forest deck keeps B's resampled hand/library
    indistinguishable (any deal is 7 Forests), so sampled-world descents can
    only diverge through A's library order."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    pa = os.path.join(d, "pool_pq_a.dk")
    with open(pa, "w") as f:
        f.write("1 Preordain\n6 Island\n1 Lightning Bolt\n1 Grizzly Bears\n"
                "4 Mountain\n4 Forest\n4 Swamp\n4 Plains\n5 Island\n")
    pb = os.path.join(d, "pool_pq_b.dk")
    with open(pb, "w") as f:
        f.write("30 Forest\n")
    return [pa, pb]


_POOL_EXTRA = ["--deck-a", "temp/pool_pq_a", "--deck-b", "temp/pool_pq_b",
               "--no-shuffle", "--narrative", "--battlefield-a", "Island"]


def _record_pool_line(seed, scry_choices, cap=4000):
    """Play one full game with a payload-aware policy: cast the first castable
    spell seen (Preordain, exactly once), answer the scry keep/bottom picks with
    `scry_choices` in order, auto-0 everything else. Returns (records, choices,
    outcome) where choices[i] is the integer played at decision i — the caller
    replays the line by index, so the policy never has to be re-run."""
    eng = Engine(seed, extra=_POOL_EXTRA)
    records, choices = [], []
    cast_done = False
    scry_seen = 0
    r = eng.read()
    n = 0
    while r.kind == "q" and n < cap:
        cats = _query_cats(r.payload)[:r.nc]
        a = 0
        if not cast_done and bool((cats == CAT_CAST_SPELL).any()):
            a = int(np.argmax(cats == CAT_CAST_SPELL))
            cast_done = True
        elif bool((cats == CAT_BOTTOM_DECK_CARD).any()):
            if scry_seen < len(scry_choices):
                a = scry_choices[scry_seen]
            scry_seen += 1
        records.append((r.nc, r.payload, r.safe))
        choices.append(a)
        r = eng.play(a)
        n += 1
    outcome = {"kind": r.kind, "returncode": r.returncode, "winner": r.winner}
    eng.kill()
    return records, choices, outcome


def _scry_pick_indices(records):
    """Decision indices whose menu contains a BOTTOM_DECK_CARD action — in the
    pool scenario, exactly the per-card scry keep/bottom picks."""
    return [i for i, (nc, pl, _s) in enumerate(records)
            if bool((_query_cats(pl)[:nc] == CAT_BOTTOM_DECK_CARD).any())]


def _assert_scry_pick(records, i, ctx):
    nc, pl, safe = records[i]
    if nc != 2:
        raise ProtocolError(f"{ctx}: scry pick at {i} has {nc} options, expected "
                            "the 2-option keep/bottom menu")
    cats = _query_cats(pl)[:nc]
    if not (cats[0] == CAT_TOP_LIBRARY and cats[1] == CAT_BOTTOM_DECK_CARD):
        raise ProtocolError(f"{ctx}: scry pick at {i} categories {cats} are not "
                            "[TOP_LIBRARY, BOTTOM_DECK_CARD]")
    if not safe:
        raise ProtocolError(f"{ctx}: scry pick at {i} reports safe=0 — a frozen-"
                            "pool loop pick should be a loop-top pending decision")


def test_pool_loop_roundtrip():
    """Batch 4 (frozen-pool loops): the per-card picks of a multi-pick pool
    effect — Preordain's scry-2 loop through effects::scry — are loop-top
    pending decisions: BOTH picks (the second proving a MID-LOOP root, with a
    partially-consumed pool in the frame rt) report safe=1 and are valid
    SNAPSHOT/RESTORE roots. At each: SNAPSHOT re-emits exactly, a divergent
    line (bottom instead of the control's keep, then a scrambled continuation)
    then RESTORE returns byte-identically, and the resumed real line stays
    byte-identical to a no-snapshot control run with the same outcome."""
    seed = 5
    deck_paths = _write_pool_decks()
    try:
        control, choices, outcome = _record_pool_line(seed, scry_choices=(0, 0))
        picks = _scry_pick_indices(control)
        if len(picks) != 2:
            raise ProtocolError(f"expected exactly 2 scry picks in the control "
                                f"line, found {len(picks)}")
        p1, p2 = picks
        if p2 != p1 + 1:
            raise ProtocolError(f"scry picks at {p1}/{p2} are not consecutive — "
                                "the loop should re-arm immediately")
        _assert_scry_pick(control, p1, "pool control")
        _assert_scry_pick(control, p2, "pool control")

        eng = Engine(seed, extra=_POOL_EXTRA)
        cur = eng.read()
        excursions = 0
        for idx in range(len(control)):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"pool alignment at decision {idx}")
            if idx in (p1, p2):
                snap_nc, snap_pl = cur.nc, cur.payload
                q = eng.snapshot(0)
                _assert_same_query(q, snap_pl, snap_nc,
                                   f"pool post-SNAPSHOT re-emit at {idx}")
                # Divergent excursion: bottom the card (choice 1) instead of the
                # control's keep, then a scrambled continuation.
                dq = q
                for i in range(8):
                    dq = eng.play(1 if i == 0 else _diverge(i, dq.nc))
                    if dq.kind != "q":
                        break
                rq = eng.restore(0)
                _assert_same_query(rq, snap_pl, snap_nc,
                                   f"pool post-RESTORE re-emit at {idx}")
                rel = eng.release()
                _assert_same_query(rel, snap_pl, snap_nc,
                                   f"pool post-RELEASE re-emit at {idx}")
                cur = rel
                excursions += 1
            cur = eng.play(choices[idx])
        if cur.kind != "eof" or cur.returncode != 0:
            eng.kill()
            raise ProtocolError(f"resumed pool line ended abnormally: "
                                f"{cur.kind} rc={cur.returncode}")
        if cur.winner != outcome["winner"]:
            eng.kill()
            raise ProtocolError(f"outcome mismatch: control {outcome['winner']!r} "
                                f"vs resumed {cur.winner!r}")
        eng.kill()
        if excursions != 2:
            raise ProtocolError(f"expected 2 pool excursions, ran {excursions}")
        return (f"scry picks @ {p1} and @ {p2} (mid-loop) both safe=1, "
                f"round-trips exact, outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_pool_determinize_pin():
    """Batch 4 (determinize pinning for revealed pools): at a suspended MID-LOOP
    scry root (the SECOND Preordain pick — the already-kept Lightning Bolt is no
    longer in the parked menu, so keeping it in place relies purely on the
    ScryRt.lib pin fed into collect_pending_pins), DETERMINIZE with several
    seeds must:

    - re-emit the root byte-identically (the parked menu and everything visible
      to the chooser are world-invariant);
    - keep every pinned looked-at card exactly in place: replaying the control
      choices inside each world (bottom Grizzly Bears, then Preordain's draw)
      yields post-root queries BYTE-IDENTICAL to the control line — the draw
      still fetches the pinned Bolt off the top, which only holds if its
      zone row (library, distance 0) survived the resample;
    - still RESAMPLE the unpinned cards: deeper descents (later draws off the
      varied library body) must diverge across world seeds;
    - RESTORE back to the root byte-identically, with the released real line
      byte-identical to the control run."""
    seed = 5
    deck_paths = _write_pool_decks()
    try:
        # Control: keep Bolt (pick 1), bottom Bears (pick 2) — the draw then
        # takes Bolt, and A's later draws walk straight into the varied cards.
        control, choices, outcome = _record_pool_line(seed, scry_choices=(0, 1))
        picks = _scry_pick_indices(control)
        if len(picks) != 2:
            raise ProtocolError(f"expected exactly 2 scry picks in the control "
                                f"line, found {len(picks)}")
        p2 = picks[1]
        _assert_scry_pick(control, p2, "pin control")

        eng = Engine(seed, extra=_POOL_EXTRA)
        cur = eng.read()
        for idx in range(p2):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"pin alignment at decision {idx}")
            cur = eng.play(choices[idx])
        _assert_same_query(cur, control[p2][1], control[p2][0],
                           "pin root alignment")
        snap_nc, snap_pl = cur.nc, cur.payload
        eng.snapshot(0)

        head = 3    # post-root steps that must be world-invariant (pins held)
        depth = 120  # descent length hashed for cross-world divergence
        hashes = {}
        for ds in (2, 3, 4):
            rq = eng.restore(0)
            _assert_same_query(rq, snap_pl, snap_nc, f"pin RESTORE (seed {ds})")
            dq = eng.determinize(ds)
            _assert_same_query(dq, snap_pl, snap_nc,
                               f"pin post-DETERMINIZE re-emit (seed {ds})")
            h = hashlib.sha256()
            for j in range(depth):
                cidx = p2 + j
                a = choices[cidx] if cidx < len(choices) else 0
                r = eng.play(a)
                if r.kind != "q":
                    h.update(b"END:" + (r.result or "eof").encode())
                    break
                if j < head:
                    want_nc, want_pl, _ws = control[p2 + 1 + j]
                    _assert_same_query(
                        r, want_pl, want_nc,
                        f"pin world (seed {ds}) step {j} diverged from control "
                        "— a pinned looked-at card moved in the sampled world")
                h.update(r.payload)
            hashes[ds] = h.hexdigest()
        if len(set(hashes.values())) == 1:
            raise ProtocolError(
                "all determinize worlds descended byte-identically for "
                f"{depth} steps — the resample of UNPINNED library cards "
                "appears to have no effect")

        rq = eng.restore(0)
        _assert_same_query(rq, snap_pl, snap_nc, "pin final RESTORE")
        rel = eng.release()
        _assert_same_query(rel, snap_pl, snap_nc, "pin post-RELEASE re-emit")
        cur = rel
        for idx in range(p2, len(control)):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"pin resumed alignment at decision {idx}")
            cur = eng.play(choices[idx])
        if cur.kind != "eof" or cur.returncode != 0:
            eng.kill()
            raise ProtocolError(f"resumed pin line ended abnormally: "
                                f"{cur.kind} rc={cur.returncode}")
        if cur.winner != outcome["winner"]:
            eng.kill()
            raise ProtocolError(f"outcome mismatch: control {outcome['winner']!r} "
                                f"vs resumed {cur.winner!r}")
        eng.kill()
        n_worlds = len(set(hashes.values()))
        return (f"mid-loop root @ {p2}: {head} post-root steps world-invariant "
                f"(pins held), {n_worlds}/3 distinct world descents, RESTORE "
                f"exact, outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def _write_trigger_decks():
    """Lands-only stacked decks for the trigger-placement roots test (Batch 5).
    A's battlefield preset holds Delver of Secrets + Aether Vial: BOTH fire an
    upkeep trigger at A's first upkeep, so A's 603.3b ordering choice reaches a
    real 2-option ORDER_TRIGGERS decision (two DISTINCT triggers, so the two
    orders are observably different lines — the engine prompts for identical
    duplicates too, but those diverge invisibly). Lands-only libraries deck the
    game out under auto-0."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    paths = []
    for name, land in (("trig_pq_a", "Mountain"), ("trig_pq_b", "Forest")):
        p = os.path.join(d, f"{name}.dk")
        with open(p, "w") as f:
            f.write(f"30 {land}\n")
        paths.append(p)
    return paths


_TRIGGER_EXTRA = ["--deck-a", "temp/trig_pq_a", "--deck-b", "temp/trig_pq_b",
                  "--no-shuffle",
                  "--battlefield-a", "Delver of Secrets,Aether Vial"]


def test_trigger_order_roundtrip():
    """Batch 5 (trigger placement): the CR 603.3b trigger-ordering prompt is a
    loop-top pending decision (tag TRIGGER_PLACE) — it reports safe=1 and is a
    valid SNAPSHOT/RESTORE root. Two distinct simultaneous upkeep triggers
    (Delver of Secrets + Aether Vial, one controller) reach a 2-option
    ORDER_TRIGGERS decision at A's first upkeep. At the root: SNAPSHOT
    re-emits exactly; a divergent line picks the OTHER order and its very next
    query must differ from the control's (the stack order — hence resolution
    order — observably changed); RESTORE returns byte-identically; the resumed
    real line stays byte-identical to a no-snapshot control run with the same
    outcome."""
    seed = 5
    deck_paths = _write_trigger_decks()
    try:
        control, outcome = record_line(seed, _auto0, extra=_TRIGGER_EXTRA)
        root_idx = _first_cat_index(control, CAT_ORDER_TRIGGERS)
        if root_idx is None:
            raise ProtocolError("no ORDER_TRIGGERS decision in the control line "
                                "— the Delver+Vial board should force one at "
                                "A's first upkeep")
        nc, pl, safe = control[root_idx]
        if nc != 2:
            raise ProtocolError(f"ordering menu has {nc} options, expected 2 "
                                "(Delver trigger + Vial trigger)")
        if not safe:
            raise ProtocolError("ordering decision reports safe=0 — it should "
                                "be a loop-top pending decision now")
        if root_idx + 1 >= len(control):
            raise ProtocolError("control line ends at the ordering root")

        eng = Engine(seed, extra=_TRIGGER_EXTRA)
        cur = eng.read()
        for idx in range(root_idx):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"trigger alignment at decision {idx}")
            cur = eng.play(_auto0(idx, cur.nc))
        _assert_same_query(cur, pl, nc, "trigger ordering root alignment")
        q = eng.snapshot(0)
        _assert_same_query(q, pl, nc, "trigger post-SNAPSHOT re-emit")

        # Divergent excursion: the OTHER order (choice 1). The next query must
        # differ from the control's next — the other trigger is now on top of
        # the stack, so the resolution order visibly changed.
        dq = eng.play(1)
        if dq.kind != "q":
            eng.kill()
            raise ProtocolError("divergent order ended the game immediately")
        if _first_diff(dq.payload, control[root_idx + 1][1]) is None:
            eng.kill()
            raise ProtocolError("picking the other trigger order produced a "
                                "byte-identical next query — the ordering "
                                "choice had no observable effect")
        for i in range(1, 8):
            dq = eng.play(_diverge(i, dq.nc))
            if dq.kind != "q":
                break
        rq = eng.restore(0)
        _assert_same_query(rq, pl, nc, "trigger post-RESTORE re-emit")
        rel = eng.release()
        _assert_same_query(rel, pl, nc, "trigger post-RELEASE re-emit")
        cur = rel
        for idx in range(root_idx, len(control)):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"trigger resumed alignment at decision {idx}")
            cur = eng.play(_auto0(idx, cur.nc))
        if cur.kind != "eof" or cur.returncode != 0:
            eng.kill()
            raise ProtocolError(f"resumed trigger line ended abnormally: "
                                f"{cur.kind} rc={cur.returncode}")
        if cur.winner != outcome["winner"]:
            eng.kill()
            raise ProtocolError(f"outcome mismatch: control {outcome['winner']!r} "
                                f"vs resumed {cur.winner!r}")
        eng.kill()
        return (f"ORDER_TRIGGERS root @ {root_idx} (nc=2) safe=1, other order "
                f"diverges, round-trip exact, outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def _write_trigger_target_decks():
    """Stacked decks for the trigger target-selection root. With --no-shuffle
    A's opening hand is Flickerwisp + 6 Plains; a 3-Plains battlefield preset
    pays its {1}{W}{W}, so a cast-first policy casts Flickerwisp on A's first
    turn. Its ETB trigger ("exile another target permanent") chooses its target
    at PLACEMENT time (CR 603.3d window) — a 3-option SELECT_TARGET menu over
    the preset Plains, now a loop-top pending decision."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    pa = os.path.join(d, "trig_tgt_a.dk")
    with open(pa, "w") as f:
        f.write("1 Flickerwisp\n29 Plains\n")
    pb = os.path.join(d, "trig_tgt_b.dk")
    with open(pb, "w") as f:
        f.write("30 Forest\n")
    return [pa, pb]


_TRIGGER_TGT_EXTRA = ["--deck-a", "temp/trig_tgt_a", "--deck-b", "temp/trig_tgt_b",
                      "--no-shuffle",
                      "--battlefield-a", "Plains,Plains,Plains"]


def _record_cast_first_line(seed, extra, cap=4000):
    """Control recording for the trigger-target test: the FIRST decision whose
    menu offers a CAST_SPELL action casts it (Flickerwisp — the only spell in
    the deck), every other decision auto-0s. Returns (records, choices,
    outcome) with records[i] = (nc, payload, safe) and choices[i] the action
    played, so the round-trip replay can follow the identical line."""
    eng = Engine(seed, extra=extra)
    records, choices = [], []
    cast_done = False
    r = eng.read()
    idx = 0
    while r.kind == "q" and idx < cap:
        action = 0
        if not cast_done:
            cats = _query_cats(r.payload)[:r.nc]
            hits = np.nonzero(cats == CAT_CAST_SPELL)[0]
            if hits.size:
                action = int(hits[0])
                cast_done = True
        records.append((r.nc, r.payload, r.safe))
        choices.append(action)
        r = eng.play(action)
        idx += 1
    outcome = {"kind": r.kind, "returncode": r.returncode, "winner": r.winner}
    eng.kill()
    return records, choices, outcome


def test_trigger_target_roundtrip():
    """Batch 5 (trigger placement targets): a triggered ability's placement-time
    target selection (CR 603.3d) — Flickerwisp's ETB "exile another target
    permanent" — is a loop-top pending decision (tag TRIGGER_PLACE, shared
    TargetSelectRT machine): it reports safe=1 and is a valid SNAPSHOT/RESTORE
    root. At the root (a 3-option all-SELECT_TARGET menu over the preset
    Plains): SNAPSHOT re-emits exactly; a divergent line (another Plains, then
    a scrambled continuation) then RESTORE returns byte-identically; the
    resumed real line stays byte-identical to the control with the same
    outcome."""
    seed = 5
    deck_paths = _write_trigger_target_decks()
    try:
        control, choices, outcome = _record_cast_first_line(seed, _TRIGGER_TGT_EXTRA)
        root_idx = _first_cat_index(control, CAT_SELECT_TARGET)
        if root_idx is None:
            raise ProtocolError("no SELECT_TARGET decision in the control line "
                                "— Flickerwisp's ETB trigger should force one")
        nc, pl, safe = control[root_idx]
        if nc != 3:
            raise ProtocolError(f"trigger target menu has {nc} options, "
                                "expected the 3 preset Plains")
        cats = _query_cats(pl)
        if not bool((cats[:nc] == CAT_SELECT_TARGET).all()):
            raise ProtocolError(f"trigger target categories {cats[:nc]} are not "
                                f"all SELECT_TARGET ({CAT_SELECT_TARGET})")
        if not safe:
            raise ProtocolError("trigger target decision reports safe=0 — it "
                                "should be a loop-top pending decision now")

        eng = Engine(seed, extra=_TRIGGER_TGT_EXTRA)
        cur = eng.read()
        for idx in range(root_idx):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"trigger-target alignment at decision {idx}")
            cur = eng.play(choices[idx])
        _assert_same_query(cur, pl, nc, "trigger target root alignment")
        q = eng.snapshot(0)
        _assert_same_query(q, pl, nc, "trigger-target post-SNAPSHOT re-emit")
        # Divergent excursion: exile a different Plains, then scramble.
        dq = q
        for i in range(8):
            dq = eng.play(dq.nc - 1 if i == 0 else _diverge(i, dq.nc))
            if dq.kind != "q":
                break
        rq = eng.restore(0)
        _assert_same_query(rq, pl, nc, "trigger-target post-RESTORE re-emit")
        rel = eng.release()
        _assert_same_query(rel, pl, nc, "trigger-target post-RELEASE re-emit")
        cur = rel
        for idx in range(root_idx, len(control)):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"trigger-target resumed alignment at decision {idx}")
            cur = eng.play(choices[idx])
        if cur.kind != "eof" or cur.returncode != 0:
            eng.kill()
            raise ProtocolError(f"resumed trigger-target line ended abnormally: "
                                f"{cur.kind} rc={cur.returncode}")
        if cur.winner != outcome["winner"]:
            eng.kill()
            raise ProtocolError(f"outcome mismatch: control {outcome['winner']!r} "
                                f"vs resumed {cur.winner!r}")
        eng.kill()
        return (f"trigger target root @ {root_idx} (nc=3) safe=1, round-trip "
                f"exact, outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


# ── Batch 6: SBE prompts (legend rule, ETB choices) as latched roots ──────────

def _write_decks(specs):
    """Write temp .dk files from (name, content) pairs; return their paths."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    paths = []
    for name, content in specs:
        p = os.path.join(d, f"{name}.dk")
        with open(p, "w") as f:
            f.write(content)
        paths.append(p)
    return paths


def _record_play_land_first_line(seed, extra, land_name, cap=4000):
    """Control recording that plays ONE specific land: the FIRST decision whose
    menu offers PLAY_LAND of `land_name` plays it (identified by the action's
    vocab card id, since the hand offers several lands), every other decision
    auto-0s. Returns (records, choices, outcome) like _record_cast_first_line."""
    want = _VOCAB_NAMES.index(land_name)
    eng = Engine(seed, extra=extra)
    records, choices = [], []
    played = False
    r = eng.read()
    idx = 0
    while r.kind == "q" and idx < cap:
        action = 0
        if not played:
            cats = _query_cats(r.payload)[:r.nc]
            vocab = np.rint(_query_ids(r.payload)[:r.nc] * N_CARD_TYPES).astype(int)
            hits = np.nonzero((cats == CAT_PLAY_LAND) & (vocab == want))[0]
            if hits.size:
                action = int(hits[0])
                played = True
        records.append((r.nc, r.payload, r.safe))
        choices.append(action)
        r = eng.play(action)
        idx += 1
    outcome = {"kind": r.kind, "returncode": r.returncode, "winner": r.winner}
    eng.kill()
    return records, choices, outcome


def _find_sbe_root(control, cat, want_nc, ctx):
    """Locate the first control decision of category `cat` and assert it is a
    well-formed safe root (all options share the category, expected menu size,
    safe=1)."""
    root_idx = _first_cat_index(control, cat)
    if root_idx is None:
        raise ProtocolError(f"{ctx}: no decision with category {cat} in the "
                            "control line")
    nc, pl, safe = control[root_idx]
    cats = _query_cats(pl)
    if not bool((cats[:nc] == cat).all()):
        raise ProtocolError(f"{ctx}: root categories {cats[:nc]} are not all {cat}")
    if want_nc is not None and nc != want_nc:
        raise ProtocolError(f"{ctx}: root menu has {nc} options, expected {want_nc}")
    if not safe:
        raise ProtocolError(f"{ctx}: root reports safe=0 — it should be a "
                            "loop-top pending decision now")
    return root_idx


def _run_sbe_roundtrip(seed, extra, control, choices, outcome, root_idx, ctx,
                       expect_diverge):
    """Shared SNAPSHOT/diverge/RESTORE driver for the SBE-prompt roots: replay
    the control line to the root, SNAPSHOT (re-emit must be exact), play a
    divergent excursion starting with the option AFTER the control's pick
    (optionally asserting the very next query differs from the control's — the
    choice observably changed the state), RESTORE + RELEASE byte-identically,
    then resume the control line to EOF and compare the outcome."""
    nc, pl, _safe = control[root_idx]
    eng = Engine(seed, extra=extra)
    try:
        cur = eng.read()
        for idx in range(root_idx):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"{ctx} alignment at decision {idx}")
            cur = eng.play(choices[idx])
        _assert_same_query(cur, pl, nc, f"{ctx} root alignment")
        q = eng.snapshot(0)
        _assert_same_query(q, pl, nc, f"{ctx} post-SNAPSHOT re-emit")
        dq = eng.play((choices[root_idx] + 1) % nc)
        if expect_diverge:
            if dq.kind != "q":
                raise ProtocolError(f"{ctx}: divergent pick ended the game "
                                    "immediately")
            if root_idx + 1 < len(control) and \
                    _first_diff(dq.payload, control[root_idx + 1][1]) is None:
                raise ProtocolError(f"{ctx}: picking the other option produced "
                                    "a byte-identical next query — the choice "
                                    "had no observable effect")
        for i in range(1, 8):
            if dq.kind != "q":
                break
            dq = eng.play(_diverge(i, dq.nc))
        rq = eng.restore(0)
        _assert_same_query(rq, pl, nc, f"{ctx} post-RESTORE re-emit")
        rel = eng.release()
        _assert_same_query(rel, pl, nc, f"{ctx} post-RELEASE re-emit")
        cur = rel
        for idx in range(root_idx, len(control)):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"{ctx} resumed alignment at decision {idx}")
            cur = eng.play(choices[idx])
        if cur.kind != "eof" or cur.returncode != 0:
            raise ProtocolError(f"{ctx}: resumed line ended abnormally: "
                                f"{cur.kind} rc={cur.returncode}")
        if cur.winner != outcome["winner"]:
            raise ProtocolError(f"{ctx}: outcome mismatch: control "
                                f"{outcome['winner']!r} vs resumed {cur.winner!r}")
    finally:
        eng.kill()


def test_legend_rule_roundtrip():
    """Batch 6 (SBE prompts): the 704.5j legend-rule keep choice is a loop-top
    pending decision (tag SBE_LATCHED, latched-answer re-derivation): it
    reports safe=1 and is a valid SNAPSHOT/RESTORE root. A preset Thalia plus a
    second copy cast from hand (two preset Plains pay {1}{W}) collide when the
    spell resolves; at the 2-option KEEP_LEGEND root: SNAPSHOT re-emits
    exactly; keeping the OTHER copy must change the next query (the kept
    permanent's identity/summoning sickness differ); RESTORE returns
    byte-identically; the resumed line matches the control to EOF."""
    seed = 5
    deck_paths = _write_decks([
        ("legend_pq_a", "1 Thalia Guardian of Thraben\n29 Plains\n"),
        ("legend_pq_b", "30 Forest\n"),
    ])
    extra = ["--deck-a", "temp/legend_pq_a", "--deck-b", "temp/legend_pq_b",
             "--no-shuffle",
             "--battlefield-a", "Thalia Guardian of Thraben,Plains,Plains"]
    try:
        control, choices, outcome = _record_cast_first_line(seed, extra)
        root_idx = _find_sbe_root(control, CAT_KEEP_LEGEND, 2, "legend")
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, root_idx,
                           "legend", expect_diverge=True)
        return (f"KEEP_LEGEND root @ {root_idx} (nc=2) safe=1, keeping the "
                f"other copy diverges, round-trip exact, "
                f"outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_etb_choose_type_roundtrip():
    """Batch 6 (SBE prompts): the ETB choose-a-creature-type prompt (Cavern of
    Souls) is a loop-top pending decision (tag SBE_LATCHED). Playing the preset
    Cavern as A's first land reaches a 2-option CHOOSE_TYPE menu (Delver of
    Secrets in A's deck contributes Human + Wizard): safe=1, SNAPSHOT re-emits
    exactly, a divergent excursion (the other type) then RESTORE returns
    byte-identically, and the resumed line matches the control to EOF. (The
    chosen type itself is not serialized to the observation, so no
    next-query-differs assertion.)"""
    seed = 5
    deck_paths = _write_decks([
        ("ctype_pq_a", "1 Cavern of Souls\n1 Delver of Secrets\n28 Plains\n"),
        ("ctype_pq_b", "30 Forest\n"),
    ])
    extra = ["--deck-a", "temp/ctype_pq_a", "--deck-b", "temp/ctype_pq_b",
             "--no-shuffle"]
    try:
        control, choices, outcome = _record_play_land_first_line(
            seed, extra, "Cavern of Souls")
        root_idx = _find_sbe_root(control, CAT_CHOOSE_TYPE, 2, "etb-choose-type")
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, root_idx,
                           "etb-choose-type", expect_diverge=False)
        return (f"CHOOSE_TYPE root @ {root_idx} (nc=2) safe=1, round-trip "
                f"exact, outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_etb_name_card_roundtrip():
    """Batch 6 (SBE prompts): the ETB name-a-card prompt (Disruptor Flute) is a
    loop-top pending decision (tag SBE_LATCHED). Casting the Flute off two
    preset Plains reaches a 2-option NAME_CARD menu (B's deck holds Forest +
    Mountain): safe=1, SNAPSHOT re-emits exactly, naming the OTHER card must
    change the next query (the permanent's chosen-name id is serialized),
    RESTORE returns byte-identically, and the resumed line matches the control
    to EOF."""
    seed = 5
    deck_paths = _write_decks([
        ("flute_pq_a", "1 Disruptor Flute\n29 Plains\n"),
        ("flute_pq_b", "15 Forest\n15 Mountain\n"),
    ])
    extra = ["--deck-a", "temp/flute_pq_a", "--deck-b", "temp/flute_pq_b",
             "--no-shuffle",
             "--battlefield-a", "Plains,Plains"]
    try:
        control, choices, outcome = _record_cast_first_line(seed, extra)
        root_idx = _find_sbe_root(control, CAT_NAME_CARD, 2, "etb-name-card")
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, root_idx,
                           "etb-name-card", expect_diverge=True)
        return (f"NAME_CARD root @ {root_idx} (nc=2) safe=1, naming the other "
                f"card diverges, round-trip exact, "
                f"outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


# ── Batch 7: live-menu loops (change_zone picks, unless-mana) as roots ────────

def _record_cast_any_line(seed, extra, cap=4000):
    """Control recording for the live-menu-loop tests: EVERY decision whose menu
    offers a CAST_SPELL action casts the first one offered (here: A's creature,
    then B's Daze in response — the normal-cost entry precedes the alternate-cost
    duplicate), every other decision auto-0s. Auto-0 drives the unless-mana loop
    through its natural line (tap, tap, then Pay — the pay entry is index 0 once
    no untapped sources remain). Returns (records, choices, outcome)."""
    eng = Engine(seed, extra=extra)
    records, choices = [], []
    r = eng.read()
    idx = 0
    while r.kind == "q" and idx < cap:
        action = 0
        cats = _query_cats(r.payload)[:r.nc]
        hits = np.nonzero(cats == CAT_CAST_SPELL)[0]
        if hits.size:
            action = int(hits[0])
        records.append((r.nc, r.payload, r.safe))
        choices.append(action)
        r = eng.play(action)
        idx += 1
    outcome = {"kind": r.kind, "returncode": r.returncode, "winner": r.winner}
    eng.kill()
    return records, choices, outcome


def _unless_decline_index(nc, payload, ctx):
    """Index of the decline entry in a PAY_UNLESS menu (category PAY_UNLESS,
    option_ordinal 0 — tap actions carry other categories, pay carries 1)."""
    cats = _query_cats(payload)[:nc]
    ords = _query_ords(payload)[:nc]
    hits = np.nonzero((cats == CAT_PAY_UNLESS) & (ords == 0))[0]
    if hits.size != 1:
        raise ProtocolError(f"{ctx}: expected exactly one decline entry, found "
                            f"{hits.size}")
    return int(hits[0])


def test_unless_mana_roundtrip():
    """Batch 7 (live-menu loops): the pay-{N}-unless prompt with tap-for-mana
    sub-choices (run_unless_loop's MANA kind, under effects::counter) is a
    loop-top pending decision. B Dazes A's Grizzly Bears; A has two untapped
    Forests, so the unless loop runs THREE iterations under the control line —
    tap (nc=3: two taps + decline; Pay is not yet affordable from the empty
    pool), tap (nc=3: one tap + Pay + decline), Pay (nc=2: Pay + decline) — each
    menu rebuilt from live state as mana floats. All three report safe=1. The
    FIRST and THIRD iterations are exercised as SNAPSHOT/RESTORE roots (the
    third proving a mid-loop root with a DIFFERENT menu size round-trips): at
    each, SNAPSHOT re-emits exactly; declining instead (spell countered) must
    change the next query; RESTORE returns byte-identically; and the resumed
    real line stays byte-identical to the control run with the same outcome."""
    seed = 5
    deck_paths = _write_decks([
        ("unless_pq_a", "1 Grizzly Bears\n29 Mountain\n"),
        ("unless_pq_b", "1 Daze\n29 Island\n"),
    ])
    extra = ["--deck-a", "temp/unless_pq_a", "--deck-b", "temp/unless_pq_b",
             "--no-shuffle",
             "--battlefield-a", "Forest,Forest,Forest,Forest",
             "--battlefield-b", "Island,Island,Island"]
    try:
        control, choices, outcome = _record_cast_any_line(seed, extra)
        u1 = _first_cat_index(control, CAT_PAY_UNLESS)
        if u1 is None:
            raise ProtocolError("no PAY_UNLESS decision in the control line — "
                                "the Daze should force the unless prompt")
        want = ((u1, 3), (u1 + 1, 3), (u1 + 2, 2))
        for i, want_nc in want:
            nc, pl, safe = control[i]
            if not bool((_query_cats(pl)[:nc] == CAT_PAY_UNLESS).any()):
                raise ProtocolError(f"decision {i} is not an unless iteration — "
                                    "the loop should re-arm consecutively")
            if nc != want_nc:
                raise ProtocolError(f"unless iteration at {i} has {nc} options, "
                                    f"expected {want_nc}")
            if not safe:
                raise ProtocolError(f"unless iteration at {i} reports safe=0 — "
                                    "it should be a loop-top pending decision now")
        roots = (u1, u1 + 2)

        eng = Engine(seed, extra=extra)
        cur = eng.read()
        excursions = 0
        for idx in range(len(control)):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"unless alignment at decision {idx}")
            if idx in roots:
                snap_nc, snap_pl = cur.nc, cur.payload
                q = eng.snapshot(0)
                _assert_same_query(q, snap_pl, snap_nc,
                                   f"unless post-SNAPSHOT re-emit at {idx}")
                # Divergent excursion: decline (spell countered) instead of the
                # control's tap/Pay — the very next query must differ.
                dq = eng.play(_unless_decline_index(snap_nc, snap_pl,
                                                    f"unless root {idx}"))
                if dq.kind != "q":
                    eng.kill()
                    raise ProtocolError(f"unless root {idx}: declining ended "
                                        "the game immediately")
                if idx + 1 < len(control) and \
                        _first_diff(dq.payload, control[idx + 1][1]) is None:
                    eng.kill()
                    raise ProtocolError(f"unless root {idx}: declining produced "
                                        "a byte-identical next query — the "
                                        "choice had no observable effect")
                for i in range(1, 8):
                    if dq.kind != "q":
                        break
                    dq = eng.play(_diverge(i, dq.nc))
                rq = eng.restore(0)
                _assert_same_query(rq, snap_pl, snap_nc,
                                   f"unless post-RESTORE re-emit at {idx}")
                rel = eng.release()
                _assert_same_query(rel, snap_pl, snap_nc,
                                   f"unless post-RELEASE re-emit at {idx}")
                cur = rel
                excursions += 1
            cur = eng.play(choices[idx])
        if cur.kind != "eof" or cur.returncode != 0:
            eng.kill()
            raise ProtocolError(f"resumed unless line ended abnormally: "
                                f"{cur.kind} rc={cur.returncode}")
        if cur.winner != outcome["winner"]:
            eng.kill()
            raise ProtocolError(f"outcome mismatch: control {outcome['winner']!r} "
                                f"vs resumed {cur.winner!r}")
        eng.kill()
        if excursions != 2:
            raise ProtocolError(f"expected 2 unless excursions, ran {excursions}")
        return (f"unless iterations @ {u1}/{u1 + 1}/{u1 + 2} (nc=3/3/2) all "
                f"safe=1, roots @ {roots[0]} and @ {roots[1]} (mid-loop, "
                f"different menu size) round-trip exact, decline diverges, "
                f"outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_yorion_blink_roundtrip():
    """Batch 7 (live-menu loops): the non-targeted battlefield multi-select of a
    ChangeZone (Yorion, Sky Nomad's blink: "exile any number of other nonland
    permanents you own and control") is a loop-top pending decision. Casting the
    preset-affordable Yorion fires its ETB trigger, whose resolution reaches a
    2-option CHOOSE_CARD menu (Exile Grizzly Bears / Done): safe=1, SNAPSHOT
    re-emits exactly, picking Done instead of the control's exile must change
    the next query (the Bears stays on the battlefield), RESTORE returns
    byte-identically, and the resumed line — Bears exiled, then returned by the
    delayed trigger at end step — matches the control to EOF."""
    seed = 5
    deck_paths = _write_decks([
        ("yorion_pq_a", "1 Yorion Sky Nomad\n29 Mountain\n"),
        ("yorion_pq_b", "30 Forest\n"),
    ])
    extra = ["--deck-a", "temp/yorion_pq_a", "--deck-b", "temp/yorion_pq_b",
             "--no-shuffle",
             "--battlefield-a", "Plains,Plains,Plains,Plains,Island,Grizzly Bears"]
    try:
        control, choices, outcome = _record_cast_first_line(seed, extra)
        root_idx = _find_sbe_root(control, CAT_CHOOSE_CARD, 2, "yorion")
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, root_idx,
                           "yorion", expect_diverge=True)
        return (f"CHOOSE_CARD root @ {root_idx} (nc=2) safe=1, Done diverges, "
                f"round-trip exact, outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


# ── Batch 8: nested resolves (sub-abilities, charm modes) as roots ────────────

def test_subability_roundtrip():
    """Batch 8 (nested resolves): prompts INSIDE a chained sub-ability — a
    persisted SUB FrameLevel under the root resolve — are loop-top pending
    decisions. Cloak and Dagger, Entwined's ETB trigger chain (TrigRevealHand
    -> DBPump -> DBChangeZone) reaches two consecutive new roots while the
    trigger is mid-resolution: the DBPump sub's own target pick (2-option
    SELECT_TARGET: the opponent's Grizzly Bears / no creature) and the
    DBChangeZone sub's remembered-exile pick (the Batch 7 live-menu loop,
    reachable only as a sub-ability — 4 options: two revealed hand cards, the
    chosen creature, decline). At each root: SNAPSHOT re-emits exactly, a
    divergent pick (the next menu option — no creature / a different exile)
    then RESTORE returns byte-identically, and the resumed real line stays
    byte-identical to a no-snapshot control run with the same outcome."""
    seed = 5
    deck_paths = _write_decks([
        ("cloak_pq_a", "1 Cloak and Dagger Entwined\n29 Plains\n"),
        ("cloak_pq_b", "1 Grizzly Bears\n1 Lightning Bolt\n28 Forest\n"),
    ])
    extra = ["--deck-a", "temp/cloak_pq_a", "--deck-b", "temp/cloak_pq_b",
             "--no-shuffle",
             "--battlefield-a", "Plains,Swamp,Swamp",
             "--battlefield-b", "Grizzly Bears"]
    try:
        control, choices, outcome = _record_cast_first_line(seed, extra)
        # The DBPump pick is the first 2-option all-SELECT_TARGET menu (the
        # earlier trigger-placement opponent pick is a 1-option menu).
        pump_idx = None
        for i, (nc, pl, _s) in enumerate(control):
            if nc == 2 and bool((_query_cats(pl)[:nc] == CAT_SELECT_TARGET).all()):
                pump_idx = i
                break
        if pump_idx is None:
            raise ProtocolError("no 2-option SELECT_TARGET decision in the "
                                "control line — the DBPump sub's own target "
                                "pick should surface as one")
        exile_idx = _first_cat_index(control, CAT_CHOOSE_CARD)
        if exile_idx is None:
            raise ProtocolError("no CHOOSE_CARD decision in the control line — "
                                "the remembered-exile pick should surface as one")
        if exile_idx != pump_idx + 1:
            raise ProtocolError(f"exile pick at {exile_idx} does not immediately "
                                f"follow the pump pick at {pump_idx} — the sub "
                                "chain should re-arm consecutively")
        if control[exile_idx][0] != 4:
            raise ProtocolError(f"exile menu has {control[exile_idx][0]} options, "
                                "expected 4 (2 hand cards + creature + decline)")
        for name, i in (("pump-target", pump_idx), ("remembered-exile", exile_idx)):
            if not control[i][2]:
                raise ProtocolError(f"{name} decision reports safe=0 — it should "
                                    "be a loop-top pending decision now")

        eng = Engine(seed, extra=extra)
        cur = eng.read()
        excursions = 0
        for idx in range(len(control)):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"subability alignment at decision {idx}")
            if idx in (pump_idx, exile_idx):
                snap_nc, snap_pl = cur.nc, cur.payload
                q = eng.snapshot(0)
                _assert_same_query(q, snap_pl, snap_nc,
                                   f"subability post-SNAPSHOT re-emit at {idx}")
                # Divergent excursion: the next menu option (no creature / a
                # different exile), then a scrambled continuation.
                dq = q
                for i in range(8):
                    a = (choices[idx] + 1) % dq.nc if i == 0 else _diverge(i, dq.nc)
                    dq = eng.play(a)
                    if dq.kind != "q":
                        break
                rq = eng.restore(0)
                _assert_same_query(rq, snap_pl, snap_nc,
                                   f"subability post-RESTORE re-emit at {idx}")
                rel = eng.release()
                _assert_same_query(rel, snap_pl, snap_nc,
                                   f"subability post-RELEASE re-emit at {idx}")
                cur = rel
                excursions += 1
            cur = eng.play(choices[idx])
        if cur.kind != "eof" or cur.returncode != 0:
            eng.kill()
            raise ProtocolError(f"resumed subability line ended abnormally: "
                                f"{cur.kind} rc={cur.returncode}")
        if cur.winner != outcome["winner"]:
            eng.kill()
            raise ProtocolError(f"outcome mismatch: control {outcome['winner']!r} "
                                f"vs resumed {cur.winner!r}")
        eng.kill()
        if excursions != 2:
            raise ProtocolError(f"expected 2 subability excursions, ran {excursions}")
        return (f"pump-target @ {pump_idx} (nc=2) and remembered-exile @ "
                f"{exile_idx} (nc=4) both safe=1, round-trips exact, "
                f"outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def _record_witherbloom_line(seed, extra, cap=4000):
    """Control recording for the charm sub-chain test: cast Witherbloom Command
    at the first opportunity, then script its four announcement picks — mode 1
    = the mill mode (0), its target = Player A herself (1, so the mill fills
    A's OWN graveyard for the chained DBReturn), mode 2 = the lose-life mode
    (0), its target = the opponent (0) — and auto-0 everything else (including
    the DBReturn graveyard pick this test roots at). Returns (records,
    choices, outcome) like _record_cast_first_line."""
    eng = Engine(seed, extra=extra)
    records, choices = [], []
    cast_done = False
    post_cast = []
    r = eng.read()
    idx = 0
    while r.kind == "q" and idx < cap:
        action = 0
        if not cast_done:
            cats = _query_cats(r.payload)[:r.nc]
            hits = np.nonzero(cats == CAT_CAST_SPELL)[0]
            if hits.size:
                action = int(hits[0])
                cast_done = True
                post_cast = [0, 1, 0, 0]  # mill mode, self target, lose-life mode, opp target
        elif post_cast:
            action = post_cast.pop(0)
        records.append((r.nc, r.payload, r.safe))
        choices.append(action)
        r = eng.play(action)
        idx += 1
    outcome = {"kind": r.kind, "returncode": r.returncode, "winner": r.winner}
    eng.kill()
    return records, choices, outcome


def test_charm_subchain_roundtrip():
    """Batch 8 (charm containers): a prompt nested TWO levels under the root
    resolve — inside a sub-ability of an announced charm MODE (root CHARM_MODE
    level, then a SUB level) — is a loop-top pending decision. Witherbloom
    Command's mill mode (announced at cast, targeting A herself) chains
    DBReturn, whose mandatory graveyard pick over the three just-milled lands
    (Forest / Mountain / Plains — all distinct) surfaces mid-charm-resolution
    as a 3-option CHOOSE_CARD root: safe=1, SNAPSHOT re-emits exactly,
    returning a DIFFERENT land must change the next query, RESTORE returns
    byte-identically, and the resumed line — the second (lose-life) mode then
    resolving after the pick — matches the control to EOF. (The mode picks
    themselves happen at CAST time and stay blocking until the cast-family
    batches.)"""
    seed = 5
    deck_paths = _write_decks([
        ("charm_pq_a", "1 Witherbloom Command\n6 Swamp\n1 Forest\n1 Mountain\n"
                       "1 Plains\n20 Swamp\n"),
        ("charm_pq_b", "30 Forest\n"),
    ])
    extra = ["--deck-a", "temp/charm_pq_a", "--deck-b", "temp/charm_pq_b",
             "--no-shuffle",
             "--battlefield-a", "Swamp,Forest"]
    try:
        control, choices, outcome = _record_witherbloom_line(seed, extra)
        root_idx = _find_sbe_root(control, CAT_CHOOSE_CARD, 3, "charm-subchain")
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, root_idx,
                           "charm-subchain", expect_diverge=True)
        return (f"CHOOSE_CARD root @ {root_idx} (nc=3, depth-2 nested) safe=1, "
                f"other land diverges, round-trip exact, "
                f"outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_cast_x_roundtrip():
    """Batch 9 (cast flow state machine): the cast-time CHOOSE_X ladder is a
    loop-top pending decision (tag CAST, Game::PendingCast) — it reports
    safe=1 and is a valid SNAPSHOT/RESTORE root. With --no-shuffle A's opening
    hand is Chalice of the Void + Mountains and a 2-Mountain battlefield
    preset pays {X}{X}, so a cast-first policy casts Chalice on A's first turn
    and reaches the 3-option ladder (X = 0/1/2). At the root: SNAPSHOT
    re-emits exactly; a divergent X (1 instead of the control's 0) must change
    the very next query (different mana tapped, Chalice enters with different
    charge counters); RESTORE returns byte-identically; the resumed real line
    stays byte-identical to a no-snapshot control run with the same outcome."""
    seed = 5
    deck_paths = _write_decks([
        ("cast_x_a", "1 Chalice of the Void\n29 Mountain\n"),
        ("cast_x_b", "30 Forest\n"),
    ])
    extra = ["--deck-a", "temp/cast_x_a", "--deck-b", "temp/cast_x_b",
             "--no-shuffle",
             "--battlefield-a", "Mountain,Mountain"]
    try:
        control, choices, outcome = _record_cast_first_line(seed, extra)
        root_idx = _find_sbe_root(control, CAT_CHOOSE_X, 3, "cast-x")
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, root_idx,
                           "cast-x", expect_diverge=True)
        return (f"CHOOSE_X root @ {root_idx} (nc=3) safe=1, other X diverges, "
                f"round-trip exact, outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_cast_phyrexian_roundtrip():
    """Batch 9 (cast flow state machine): the per-pip phyrexian
    colored-mana-or-2-life prompts are loop-top pending decisions (tag CAST).
    Dismember ({1}{B/P}{B/P}) from A's opening hand over a 3-Swamp battlefield
    preset (B's preset Grizzly Bears is the target) yields TWO consecutive
    2-option PAYING_COSTS pips — the second proving the mid-loop resume→re-arm
    path, with the first pip's 2 life already paid at apply (its state floats
    show life 18, which only holds if the life payment lands before the next
    pip's arm). Both are valid SNAPSHOT/RESTORE roots: SNAPSHOT re-emits
    exactly, the divergent pick (pay {B} instead of the control's 2 life) must
    change the next query, RESTORE returns byte-identically, and the resumed
    real line stays byte-identical to the control with the same outcome. (The
    blocking target prompt that follows still reports safe=0 — Batch 10.)"""
    seed = 5
    deck_paths = _write_decks([
        ("cast_ph_a", "1 Dismember\n29 Swamp\n"),
        ("cast_ph_b", "30 Forest\n"),
    ])
    extra = ["--deck-a", "temp/cast_ph_a", "--deck-b", "temp/cast_ph_b",
             "--no-shuffle",
             "--battlefield-a", "Swamp,Swamp,Swamp",
             "--battlefield-b", "Grizzly Bears"]
    try:
        control, choices, outcome = _record_cast_first_line(seed, extra)
        p1 = _find_sbe_root(control, CAT_PAYING_COSTS, 2, "cast-phyrexian")
        nxt = control[p1 + 1]
        if not bool((_query_cats(nxt[1])[:nxt[0]] == CAT_PAYING_COSTS).all()) \
                or nxt[0] != 2:
            raise ProtocolError("decision after the first pip is not the second "
                                "2-option pip — the resume should re-arm")
        if not nxt[2]:
            raise ProtocolError("second pip reports safe=0 — it should be a "
                                "loop-top pending decision now")
        for root_idx in (p1, p1 + 1):
            _run_sbe_roundtrip(seed, extra, control, choices, outcome, root_idx,
                               f"cast-phyrexian pip @ {root_idx}",
                               expect_diverge=True)
        return (f"phyrexian pips @ {p1} and @ {p1 + 1} (nc=2, mid-loop) both "
                f"safe=1, pay-mana diverges, round-trips exact, "
                f"outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_cast_target_roundtrip():
    """Batch 10 (cast announce): the cast-time primary target selection —
    Lightning Bolt choosing its target as it is cast (CR 601.2c), before any
    cost is paid — is a loop-top pending decision (tag CAST, the shared
    TargetSelectRT machine embedded in Game::PendingCast). With --no-shuffle
    A's opening hand is Lightning Bolt over a 1-Mountain preset and B presets
    a Grizzly Bears, so a cast-first policy reaches a 3-option SELECT_TARGET
    menu (the Bears + both players). At the root: safe=1; SNAPSHOT re-emits
    exactly; a divergent target (the next option) must change the very next
    query (the spell on the stack carries a different target id); RESTORE
    returns byte-identically; the resumed real line stays byte-identical to a
    no-snapshot control run with the same outcome."""
    seed = 5
    deck_paths = _write_decks([
        ("cast_tgt_a", "1 Lightning Bolt\n29 Mountain\n"),
        ("cast_tgt_b", "30 Forest\n"),
    ])
    extra = ["--deck-a", "temp/cast_tgt_a", "--deck-b", "temp/cast_tgt_b",
             "--no-shuffle",
             "--battlefield-a", "Mountain",
             "--battlefield-b", "Grizzly Bears"]
    try:
        control, choices, outcome = _record_cast_first_line(seed, extra)
        root_idx = _find_sbe_root(control, CAT_SELECT_TARGET, 3, "cast-target")
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, root_idx,
                           "cast-target", expect_diverge=True)
        return (f"cast SELECT_TARGET root @ {root_idx} (nc=3) safe=1, other "
                f"target diverges, round-trip exact, "
                f"outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_charm_cast_roundtrip():
    """Batch 10 (cast announce): the cast-time modal announcement (CR 601.2b)
    — Witherbloom Command's mode picks AND the just-picked mode's target pick,
    interleaved mode -> targets -> mode — are loop-top pending decisions (tag
    CAST, run_cast_flow's CHARM_MODE / CHARM_TARGET steps). Same scenario as
    the Batch 8 charm-subchain test (B has no creatures/artifacts, so 2 of the
    4 modes are choosable): the first CHOOSE_MODE menu has 2 options and the
    decision immediately after it is the mill mode's 2-option player-target
    menu. Both are SNAPSHOT/RESTORE roots; both divergent excursions are
    exercised WITHOUT asserting the immediate next payload — the picked mode
    and its target live only in the pending cast's half-built ability (not
    serialized into the observation), and both modes here happen to present an
    identical 2-option player-target menu next, so the choices only become
    observable at resolution. Each round-trip resumes byte-identically to the
    control with the same outcome."""
    seed = 5
    deck_paths = _write_decks([
        ("charm_cast_a", "1 Witherbloom Command\n6 Swamp\n1 Forest\n1 Mountain\n"
                         "1 Plains\n20 Swamp\n"),
        ("charm_cast_b", "30 Forest\n"),
    ])
    extra = ["--deck-a", "temp/charm_cast_a", "--deck-b", "temp/charm_cast_b",
             "--no-shuffle",
             "--battlefield-a", "Swamp,Forest"]
    try:
        control, choices, outcome = _record_witherbloom_line(seed, extra)
        mode_idx = _find_sbe_root(control, CAT_CHOOSE_MODE, 2, "charm-cast mode")
        tgt_idx = mode_idx + 1
        nc, pl, safe = control[tgt_idx]
        cats = _query_cats(pl)
        if not bool((cats[:nc] == CAT_SELECT_TARGET).all()) or nc != 2:
            raise ProtocolError("decision after the mode pick is not the mill "
                                "mode's 2-option target menu — the mode -> "
                                "targets -> mode interleave broke")
        if not safe:
            raise ProtocolError("charm-cast target pick reports safe=0 — it "
                                "should be a loop-top pending decision now")
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, mode_idx,
                           "charm-cast mode", expect_diverge=False)
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, tgt_idx,
                           "charm-cast target", expect_diverge=False)
        return (f"CHOOSE_MODE root @ {mode_idx} (nc=2) and mode-target root @ "
                f"{tgt_idx} (nc=2) both safe=1, round-trips exact, "
                f"outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_storm_copy_target_roundtrip():
    """Batch 10 (spell copies): a storm copy's target selection — chosen by the
    copy's controller as the Storm trigger RESOLVES (CR 702.40a / 707.12), via
    the resumable CopySpellRT machine — is a loop-top pending decision (tag
    RESOLUTION). Counter war: A casts Lightning Bolt (2-option target menu:
    the players); B answers with Flusterstorm targeting it (1-option menu —
    Bolt is the only instant/sorcery on the stack); the storm trigger (count 1
    — the Bolt) resolves and its single copy picks its own target over BOTH
    spells now on the stack (Bolt + the original Flusterstorm) — the third
    SELECT_TARGET decision, a 2-option menu. At that root: safe=1; SNAPSHOT
    re-emits exactly; the divergent target (the other spell) must change the
    next query (the copy sits on the stack with a different target id);
    RESTORE returns byte-identically; the resumed real line stays
    byte-identical to the control with the same outcome. Also proves the
    partially built copy entity (CardData/Spell, no Zone) survives the
    suspension in the ECS."""
    seed = 5
    deck_paths = _write_decks([
        ("storm_pq_a", "1 Lightning Bolt\n29 Mountain\n"),
        ("storm_pq_b", "1 Flusterstorm\n29 Island\n"),
    ])
    extra = ["--deck-a", "temp/storm_pq_a", "--deck-b", "temp/storm_pq_b",
             "--no-shuffle",
             "--battlefield-a", "Mountain",
             "--battlefield-b", "Island"]
    try:
        control, choices, outcome = _record_cast_any_line(seed, extra)
        sel = [i for i, (nc, pl, _s) in enumerate(control)
               if bool((_query_cats(pl)[:nc] == CAT_SELECT_TARGET).all())
               and nc > 0]
        if len(sel) < 3:
            raise ProtocolError(f"expected 3 SELECT_TARGET decisions (Bolt, "
                                f"Flusterstorm, the storm copy), found {len(sel)}")
        bolt_idx, fluster_idx, copy_idx = sel[0], sel[1], sel[2]
        if control[fluster_idx][0] != 1:
            raise ProtocolError("Flusterstorm's own target menu should have 1 "
                                f"option, has {control[fluster_idx][0]}")
        if control[copy_idx][0] != 2:
            raise ProtocolError("the storm copy's target menu should have 2 "
                                f"options (both spells on the stack), has "
                                f"{control[copy_idx][0]}")
        if not control[copy_idx][2]:
            raise ProtocolError("storm copy target pick reports safe=0 — it "
                                "should be a loop-top pending decision now")
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, copy_idx,
                           "storm-copy target", expect_diverge=True)
        return (f"storm copy target root @ {copy_idx} (nc=2, after Bolt @ "
                f"{bolt_idx} / Flusterstorm @ {fluster_idx}) safe=1, other "
                f"spell diverges, round-trip exact, "
                f"outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_delve_pick_roundtrip():
    """Batch 11 (cast deferred payment): the delve count menu and the per-card
    delve exile picks (CR 702.66, run_cast_flow's DELVE_COUNT / DELVE_PICK
    steps) are loop-top pending decisions (tag CAST). Murktide Regent
    ({5}{U}{U}, Delve) from A's opening hand over a 4-Island battlefield preset
    and a 5-card preset graveyard: the count range is [3..5] (4 Islands cover
    {U}{U}+2 generic, so at least 3 generic pips must be delved), a 3-option
    CHOOSE_X menu carrying the spell as source; the control picks 3, reaching
    three CHOOSE_CARD pick menus of shrinking size (5, then 4, then 3
    candidates — all above the take-without-prompt threshold). Roots tested:
    the count menu (divergence NOT asserted on the next payload — the chosen
    count lives only in the un-serialized pending cast, and any count reaches
    the same first pick menu) and the first pick (a divergent card exiled must
    change the next query — different graveyard/exile contents). Both: safe=1,
    SNAPSHOT re-emits exactly, RESTORE byte-identical, resumed line == control
    with the same outcome."""
    seed = 5
    deck_paths = _write_decks([
        ("delve_a", "1 Murktide Regent\n29 Island\n"),
        ("delve_b", "30 Forest\n"),
    ])
    extra = ["--deck-a", "temp/delve_a", "--deck-b", "temp/delve_b",
             "--no-shuffle",
             "--battlefield-a", "Island,Island,Island,Island",
             "--graveyard-a",
             "Lightning Bolt,Ponder,Brainstorm,Unholy Heat,Mishras Bauble"]
    try:
        control, choices, outcome = _record_cast_first_line(seed, extra)
        count_idx = _find_sbe_root(control, CAT_CHOOSE_X, 3, "delve-count")
        pick_idx = count_idx + 1
        nc, pl, safe = control[pick_idx]
        cats = _query_cats(pl)
        if not bool((cats[:nc] == CAT_CHOOSE_CARD).all()) or nc != 5:
            raise ProtocolError("decision after the delve count is not the "
                                "5-candidate CHOOSE_CARD pick menu")
        if not safe:
            raise ProtocolError("delve pick reports safe=0 — it should be a "
                                "loop-top pending decision now")
        nxt = control[pick_idx + 1]
        if not bool((_query_cats(nxt[1])[:nxt[0]] == CAT_CHOOSE_CARD).all()) \
                or nxt[0] != 4:
            raise ProtocolError("second delve pick menu is not the 4-candidate "
                                "CHOOSE_CARD menu — the pick loop broke")
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, count_idx,
                           "delve-count", expect_diverge=False)
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, pick_idx,
                           "delve-pick", expect_diverge=True)
        return (f"delve CHOOSE_X count root @ {count_idx} (nc=3) and first "
                f"CHOOSE_CARD pick root @ {pick_idx} (nc=5, then 4) both "
                f"safe=1, other pick diverges, round-trips exact, "
                f"outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_alt_pitch_roundtrip():
    """Batch 11 (cast deferred payment): the alternate-cost pitch pick — Force
    of Will exiling a blue card from hand as it is cast (run_cast_flow's
    ALT_PITCH step) — is a loop-top pending decision (tag CAST). A casts a
    preset-affordable Grizzly Bears; B (landless, so only the alt-cost cast is
    offered) responds with Force of Will, whose pitch menu offers B's two other
    blue nonland cards (Brainstorm, Ponder — Islands are not blue cards) as a
    2-option PAYING_COSTS menu. At the root: safe=1; SNAPSHOT re-emits exactly;
    the divergent pitch (Ponder instead of the control's Brainstorm) must
    change the very next query (different card exiled from hand); RESTORE
    returns byte-identically; the resumed real line stays byte-identical to
    the control with the same outcome. (This branch pays before targets are
    chosen — the pre-existing alt-cost order, preserved by the conversion.)"""
    seed = 5
    deck_paths = _write_decks([
        ("alt_pitch_a", "1 Grizzly Bears\n29 Forest\n"),
        ("alt_pitch_b", "1 Force of Will\n1 Brainstorm\n1 Ponder\n27 Island\n"),
    ])
    extra = ["--deck-a", "temp/alt_pitch_a", "--deck-b", "temp/alt_pitch_b",
             "--no-shuffle",
             "--battlefield-a", "Forest,Forest"]
    try:
        control, choices, outcome = _record_cast_any_line(seed, extra)
        root_idx = _find_sbe_root(control, CAT_PAYING_COSTS, 2, "alt-pitch")
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, root_idx,
                           "alt-pitch", expect_diverge=True)
        return (f"alt-cost pitch root @ {root_idx} (nc=2) safe=1, other pitch "
                f"diverges, round-trip exact, outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def _record_activation_line(seed, extra, x_pick=None, cap=4000):
    """Control recording for the activation-family tests: the FIRST decision
    whose menu offers an ACTIVATE_ABILITY action activates it (the preset
    permanent's only offered ability), then — when x_pick is given — the first
    CHOOSE_X menu after it plays that index (the X-activation ladder); every
    other decision auto-0s. Returns (records, choices, outcome)."""
    eng = Engine(seed, extra=extra)
    records, choices = [], []
    activated = False
    x_done = x_pick is None
    r = eng.read()
    idx = 0
    while r.kind == "q" and idx < cap:
        action = 0
        cats = _query_cats(r.payload)[:r.nc]
        if not activated:
            hits = np.nonzero(cats == CAT_ACTIVATE_ABILITY)[0]
            if hits.size:
                action = int(hits[0])
                activated = True
        elif not x_done and bool((cats == CAT_CHOOSE_X).any()):
            action = int(x_pick)
            x_done = True
        records.append((r.nc, r.payload, r.safe))
        choices.append(action)
        r = eng.play(action)
        idx += 1
    outcome = {"kind": r.kind, "returncode": r.returncode, "winner": r.winner}
    eng.kill()
    return records, choices, outcome


def test_activation_target_roundtrip():
    """Batch 12 (activated abilities): the pre-cost target selection of an
    activated ability — Wasteland choosing its nonbasic-land target BEFORE the
    tap/sacrifice cost is paid (CR 602.2b / 601.2c) — is a loop-top pending
    decision (tag ACTIVATION, the shared TargetSelectRT machine embedded in
    Game::PendingActivation, armed through the ACTIVATION-tag asker with
    decision_source = the activating Wasteland). A presets a Wasteland; B
    presets Tundra and Volcanic Island, so activating the Destroy ability
    reaches a 3-option SELECT_TARGET menu (B's two duals opponent-first, then
    Wasteland itself — a nonbasic land is a legal target of its own ability).
    At the root: safe=1; SNAPSHOT re-emits exactly; a divergent target (the
    other dual — a different card id on the parked ability) must change the
    very next query; RESTORE returns byte-identically; the resumed real line
    stays byte-identical to a no-snapshot control run with the same outcome."""
    seed = 5
    deck_paths = _write_decks([
        ("act_tgt_a", "30 Mountain\n"),
        ("act_tgt_b", "30 Forest\n"),
    ])
    extra = ["--deck-a", "temp/act_tgt_a", "--deck-b", "temp/act_tgt_b",
             "--no-shuffle",
             "--battlefield-a", "Wasteland",
             "--battlefield-b", "Tundra,Volcanic Island"]
    try:
        control, choices, outcome = _record_activation_line(seed, extra)
        root_idx = _find_sbe_root(control, CAT_SELECT_TARGET, 3,
                                  "activation-target")
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, root_idx,
                           "activation-target", expect_diverge=True)
        return (f"activation SELECT_TARGET root @ {root_idx} (nc=3) safe=1, "
                f"other land diverges, round-trip exact, "
                f"outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_activation_x_roundtrip():
    """Batch 12 (activated abilities): the X-activation ladder and the
    exactly-X target picks that read it — Candelabra of Tawnos's {X}, {T}:
    Untap X target lands (Cost$ X T, TargetMin/Max$ X) — are loop-top pending
    decisions (tag ACTIVATION, run_activation_flow's X_LADDER then TARGET
    steps; the apply sets cur_game.x_paid BEFORE any later cost step runs). A
    presets Candelabra plus a Mountain and a Volcanic Island (two mana
    sources), so the ladder offers X=0/1/2 (nc=3) and the control picks X=1,
    reaching a 2-option exactly-one-land target menu (no Done — the minimum is
    X). Both are SNAPSHOT/RESTORE roots. The X excursion is exercised WITHOUT
    asserting the immediate next payload (a divergent X lives only in the
    pending activation / cur_game.x_paid, not serialized, and X=1 and X=2
    present an identical first target menu — the charm-mode caveat); the
    target excursion must diverge (a different land id reaches the stack).
    Each round-trip resumes byte-identically to the control with the same
    outcome."""
    seed = 5
    deck_paths = _write_decks([
        ("act_x_a", "30 Mountain\n"),
        ("act_x_b", "30 Forest\n"),
    ])
    extra = ["--deck-a", "temp/act_x_a", "--deck-b", "temp/act_x_b",
             "--no-shuffle",
             "--battlefield-a", "Candelabra of Tawnos,Mountain,Volcanic Island"]
    try:
        control, choices, outcome = _record_activation_line(seed, extra, x_pick=1)
        x_idx = _find_sbe_root(control, CAT_CHOOSE_X, 3, "activation-x ladder")
        tgt_idx = x_idx + 1
        nc, pl, safe = control[tgt_idx]
        cats = _query_cats(pl)
        if not bool((cats[:nc] == CAT_SELECT_TARGET).all()) or nc != 2:
            raise ProtocolError("decision after the X pick is not the 2-option "
                                "exactly-X land-target menu — the ladder -> "
                                "targets order broke")
        if not safe:
            raise ProtocolError("activation X target pick reports safe=0 — it "
                                "should be a loop-top pending decision now")
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, x_idx,
                           "activation-x ladder", expect_diverge=False)
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, tgt_idx,
                           "activation-x target", expect_diverge=True)
        return (f"CHOOSE_X root @ {x_idx} (nc=3, X=1) and exactly-X target "
                f"root @ {tgt_idx} (nc=2) both safe=1, round-trips exact, "
                f"outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def test_equip_target_roundtrip():
    """Batch 12 (activated abilities): the equip creature menu — chosen AFTER
    the equip cost is paid (machine mode auto-pays), from the candidate list
    frozen BEFORE payment (run_activation_flow's EQUIP_PAY -> EQUIP_TARGET
    steps) — is a loop-top pending decision (tag ACTIVATION). A presets
    Cori-Steel Cutter (Equip {1}{R}) with two creatures and two Mountains, so
    activating Equip auto-taps the Mountains and reaches a 2-option
    SELECT_TARGET menu (Grizzly Bears [2/2], Soul Warden [1/1]). At the root:
    safe=1; SNAPSHOT re-emits exactly; the divergent pick (equipping the other
    creature — a different permanent gains the +1/+1) must change the very
    next query; RESTORE returns byte-identically; the resumed real line stays
    byte-identical to a no-snapshot control run with the same outcome."""
    seed = 5
    deck_paths = _write_decks([
        ("equip_pq_a", "30 Mountain\n"),
        ("equip_pq_b", "30 Forest\n"),
    ])
    extra = ["--deck-a", "temp/equip_pq_a", "--deck-b", "temp/equip_pq_b",
             "--no-shuffle",
             "--battlefield-a",
             "Cori-Steel Cutter,Grizzly Bears,Soul Warden,Mountain,Mountain"]
    try:
        control, choices, outcome = _record_activation_line(seed, extra)
        root_idx = _find_sbe_root(control, CAT_SELECT_TARGET, 2, "equip-target")
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, root_idx,
                           "equip-target", expect_diverge=True)
        return (f"equip SELECT_TARGET root @ {root_idx} (nc=2) safe=1, other "
                f"creature diverges, round-trip exact, "
                f"outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


# ── Batch 13: pregame (mulligans + opening-hand) as loop-top gate decisions ──

def test_mulligan_pregame_roundtrip():
    """Batch 13 (pregame gate): the keep/mulligan decisions and the London
    bottoming picks are loop-top pregame decisions (the PregameState gate in
    the main loop) — they report safe=1 and are valid SNAPSHOT/RESTORE roots.
    Control line: A mulligans once then keeps (B keeps), so decision 0 is A's
    first keep/mull root and decision 3 is A's bottoming pick (7-card menu).
    At the keep/mull root: SNAPSHOT re-emits exactly; a divergent line (keep
    instead of the control's mulligan — no reshuffle + redraw happens) must
    observably diverge from the control within a few decisions; a roll-forward
    simulation then plays PAST the whole pregame into the main game and
    RESTORE still bounces the main loop back into the pregame gate,
    re-emitting the mulligan root byte-identically (the property that required
    a main-loop gate rather than a pre-loop wrapper). At the bottoming root:
    SNAPSHOT / divergent pick / RESTORE round-trips byte-identically. The
    resumed real line stays byte-identical to the control run with the same
    outcome."""
    seed = 5
    prefix = [1, 0, 0]  # A mulligans; B keeps; A keeps the redrawn hand

    def policy(idx, nc):
        return prefix[idx] if idx < len(prefix) else 0

    control, outcome = record_line(seed, policy)
    nc0, pl0, safe0 = control[0]
    if nc0 != 2:
        raise ProtocolError(f"decision 0 has {nc0} options, expected the "
                            "2-option keep/mulligan menu")
    cats0 = _query_cats(pl0)[:nc0]
    if not (cats0[0] == CAT_KEEP_HAND and cats0[1] == CAT_MULLIGAN):
        raise ProtocolError(f"decision 0 categories {cats0} are not "
                            "[KEEP_HAND, MULLIGAN]")
    if not safe0:
        raise ProtocolError("the first keep/mulligan reports safe=0 — it "
                            "should be a loop-top pregame decision now")
    btm_idx = 3  # A mull (0), B keep (1), A keep (2), then the bottoming pick
    nc3, pl3, safe3 = control[btm_idx]
    cats3 = _query_cats(pl3)[:nc3]
    if nc3 != 7 or not bool((cats3 == CAT_BOTTOM_DECK_CARD).all()):
        raise ProtocolError(f"decision {btm_idx} (nc={nc3}, cats {cats3}) is "
                            "not the 7-card London bottoming menu")
    if not safe3:
        raise ProtocolError("the bottoming pick reports safe=0 — it should be "
                            "a loop-top pregame decision now")

    eng = Engine(seed)
    try:
        cur = eng.read()
        # Root 1: the first keep/mulligan decision of the game.
        _assert_same_query(cur, pl0, nc0, "mulligan root alignment")
        q = eng.snapshot(0)
        _assert_same_query(q, pl0, nc0, "mulligan post-SNAPSHOT re-emit")
        # Divergent excursion: KEEP instead of the control's mulligan. The
        # divergent line must observably differ within a few decisions (the
        # control's A hand was shuffled back and redrawn; here it never is —
        # and the recorded action-history category already differs).
        dq = eng.play(0)
        diverged = False
        for i in range(4):
            if dq.kind != "q":
                diverged = True
                break
            want_nc, want_pl, _ws = control[1 + i]
            if dq.nc != want_nc or _first_diff(dq.payload, want_pl) is not None:
                diverged = True
                break
            dq = eng.play(policy(1 + i, dq.nc))
        if not diverged:
            raise ProtocolError("keeping instead of mulliganing stayed "
                                "byte-identical to the control for 4 decisions "
                                "— the mulligan had no observable effect")
        rq = eng.restore(0)
        _assert_same_query(rq, pl0, nc0, "mulligan post-RESTORE re-emit")
        # Roll-forward: keep/keep and auto-0 deep into the MAIN game (well past
        # the pregame gate), then RESTORE. The latched restore unwinds to the
        # loop top, which must bounce control back into the pregame gate and
        # re-derive the mulligan root byte-identically.
        dq = eng.play(0)
        for _ in range(40):
            if dq.kind != "q":
                break
            dq = eng.play(0)
        if dq.kind == "eof":
            raise ProtocolError("roll-forward simulation exited the process — "
                                "the game end was not intercepted")
        rq = eng.restore(0)
        _assert_same_query(rq, pl0, nc0, "mulligan roll-forward RESTORE re-emit")
        rel = eng.release()
        _assert_same_query(rel, pl0, nc0, "mulligan post-RELEASE re-emit")
        cur = rel
        # Continue the control line to the bottoming root.
        for idx in range(btm_idx):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"pregame alignment at decision {idx}")
            cur = eng.play(policy(idx, cur.nc))
        # Root 2: the London bottoming pick after the mulligan.
        _assert_same_query(cur, pl3, nc3, "bottoming root alignment")
        q = eng.snapshot(0)
        _assert_same_query(q, pl3, nc3, "bottoming post-SNAPSHOT re-emit")
        # Divergent excursion: bottom the LAST card instead of the control's
        # first, then a scrambled continuation. (Duplicates in the hand can
        # make the divergent successor coincide, so — like the charm-mode
        # roots — this asserts round-trip identity only.)
        dq = eng.play(nc3 - 1)
        for i in range(1, 6):
            if dq.kind != "q":
                break
            dq = eng.play(_diverge(i, dq.nc))
        rq = eng.restore(0)
        _assert_same_query(rq, pl3, nc3, "bottoming post-RESTORE re-emit")
        rel = eng.release()
        _assert_same_query(rel, pl3, nc3, "bottoming post-RELEASE re-emit")
        cur = rel
        for idx in range(btm_idx, len(control)):
            _assert_same_query(cur, control[idx][1], control[idx][0],
                               f"mulligan resumed alignment at decision {idx}")
            cur = eng.play(policy(idx, cur.nc))
        if cur.kind != "eof" or cur.returncode != 0:
            raise ProtocolError(f"resumed mulligan line ended abnormally: "
                                f"{cur.kind} rc={cur.returncode}")
        if cur.winner != outcome["winner"]:
            raise ProtocolError(f"outcome mismatch: control {outcome['winner']!r} "
                                f"vs resumed {cur.winner!r}")
    finally:
        eng.kill()
    return (f"keep/mull root @ 0 (nc=2) and bottoming root @ {btm_idx} (nc=7) "
            f"both safe=1, keep-diverges, roll-forward RESTORE re-enters the "
            f"gate, round-trips exact, outcome={outcome['winner']!r}")


def test_pregame_preplaced_etb_roundtrip():
    """Batch 13 (pregame gate × SBE_LATCHED): a preplaced permanent's ETB
    choice — a preset Cavern of Souls' choose-a-creature-type, fired by the
    FIAT_SETUP stage's SBE pass — now rides SBE_LATCHED through the pregame
    gate instead of the old pre-loop blocking fallback (the gate runs INSIDE
    the main loop, so in_main_loop() is true during fiat setup): it reports
    safe=1, sits between the mulligans and the first turn (decision 2, right
    after both keeps), and is a valid SNAPSHOT/RESTORE root. Round-trip:
    SNAPSHOT re-emits exactly, a divergent excursion (the other type) then
    RESTORE returns byte-identically, and the resumed line matches the control
    to EOF. (The chosen type is not serialized to the observation, so no
    next-query-differs assertion — same as the cast-Cavern test.)"""
    seed = 5
    deck_paths = _write_decks([
        ("pre_cavern_a", "1 Delver of Secrets\n29 Plains\n"),
        ("pre_cavern_b", "30 Forest\n"),
    ])
    extra = ["--deck-a", "temp/pre_cavern_a", "--deck-b", "temp/pre_cavern_b",
             "--no-shuffle", "--battlefield-a", "Cavern of Souls"]
    try:
        control, outcome = record_line(seed, _auto0, extra=extra)
        root_idx = _find_sbe_root(control, CAT_CHOOSE_TYPE, 2, "preplaced-etb")
        if root_idx != 2:
            raise ProtocolError(f"preplaced Cavern's choose-type root sits at "
                                f"decision {root_idx}, expected 2 (immediately "
                                "after the two keep decisions, i.e. during the "
                                "pregame fiat setup)")
        choices = [0] * len(control)
        _run_sbe_roundtrip(seed, extra, control, choices, outcome, root_idx,
                           "preplaced-etb", expect_diverge=False)
        return (f"preplaced CHOOSE_TYPE root @ {root_idx} (nc=2, mid-pregame) "
                f"safe=1, round-trip exact, outcome={outcome['winner']!r}")
    finally:
        for p in deck_paths:
            try:
                os.remove(p)
            except OSError:
                pass


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


PROMPT_SITE_WHITELIST = {
    # Every `get_input(` occurrence in src/**/*.cpp, classified (Batch 14 audit;
    # see the snapshot-safe plan). A count change means a prompt was added or
    # removed OUTSIDE the suspension framework: route new decisions through
    # FrameCtx::ask / a pending-query family (pending_query.h) so they stay
    # loop-top snapshot-safe, then update this table with the classification.
    #
    # (a) loop-top / loop-safe emitters:
    #   game_driver.cpp: pending-query emitter, priority decision, pregame_ask,
    #     the sideboard menu (4). The sideboard phase used to hold TWO separate
    #     prompt sites — pick a card to bring in, then pick one to cut — which
    #     were merged into a single per-move prompt; both halves of a swap now
    #     come through that one loop-safe site.
    #   action_processor.cpp: declare-attackers select, declare-blockers
    #     select, cleanup discard, miracle reveal, miracle cast (5 of its 8) —
    #     the miracle reveal and the miracle cast/do-not-cast decision (CR 702.94)
    #     both ride the mandatory-choice channel via proc_mandatory_choice,
    #     wrapped in search_set_loop_safe like cleanup discard, so they are
    #     loop-top snapshot-safe emitters
    # (b) interactive-only (machine mode auto-resolves; never a search root):
    #   action_processor.cpp: hybrid-pip interactive branch (1 of 6)
    #   mana_system.cpp: interactive mana payment (1)
    # (c) blocking fallbacks / blocking-shim residuals:
    #   resolution_frame.cpp: FrameCtx::ask blocking path — serves every
    #     non-suspendable resolve (opening-hand abilities, mana-ability
    #     SubAbility riders, pregame SBE, effect_choose_card mini-cast) (1)
    #   action_processor.cpp: BlockingTargetAsker + blocking
    #     announce_charm_modes — reachable only via effect_choose_card's
    #     cast-from-exile mini-cast (2 of 6)
    #   state_manager.cpp / state_manager_statics.cpp /
    #     state_manager_triggers.cpp: outside-main-loop fallbacks (legend keep,
    #     ETB choose-type, ETB name-card, trigger ordering) — defensive,
    #     believed unreachable since the Batch 13 pregame gate (1 + 2 + 1)
    # (d) documented residual blocking prompts:
    #   input_logger.cpp: get_input definition + request_optional_yesno, whose
    #     sole remaining caller is the outside-main-loop BLOCKING FALLBACK of
    #     the "enters tapped unless you pay N life" replacement (the in-loop
    #     path rides SBE_LATCHED — see replacement_effects.cpp apply_one) (2)
    #   systems/replacement_effects.cpp: choose_one 616.1 multi-replacement
    #     (counted by choose_one_prompt_count; residual note at the site) +
    #     dispatch_draw's blocking form (now reachable only from pregame
    #     mulligan/opening draws, where the graveyard is empty — the prompt
    #     cannot fire) + the Mox Diamond "discard a land as it enters or be
    #     put into the graveyard instead" replacement prompt (a MOVE_TO_ZONE→
    #     Battlefield self-replacement resolved inline like the other residual
    #     MOVE_TO_ZONE replacement prompts; blocks safe=0, same family) (3)
    "input_logger.cpp": 2,
    "game_driver.cpp": 4,
    "resolution_frame.cpp": 1,
    "action_processor.cpp": 8,
    "mana_system.cpp": 1,
    os.path.join("systems", "replacement_effects.cpp"): 3,
    os.path.join("systems", "state_manager.cpp"): 1,
    os.path.join("systems", "state_manager_statics.cpp"): 2,
    os.path.join("systems", "state_manager_triggers.cpp"): 1,
}


def test_prompt_site_guard():
    """Batch 14 close-out guard: every `get_input(` call site in src/ is one of
    the audited, classified prompt sites above. A future prompt added outside
    the suspension framework (a new blocking mid-flow get_input) changes a
    file's count and fails here with a pointer to the framework — keeping the
    every-decision-snapshot-safe invariant from silently eroding."""
    src = os.path.normpath(os.path.join(BIN_DIR, "..", "src"))
    found = {}
    for root, _dirs, files in os.walk(src):
        for fn in files:
            if not fn.endswith(".cpp"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, src)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            n = text.count("get_input(")
            if n:
                found[rel] = n
    if found != PROMPT_SITE_WHITELIST:
        added = {k: v for k, v in found.items()
                 if PROMPT_SITE_WHITELIST.get(k) != v}
        removed = {k: v for k, v in PROMPT_SITE_WHITELIST.items()
                   if found.get(k) != v}
        raise ProtocolError(
            "get_input call-site audit mismatch (see PROMPT_SITE_WHITELIST in "
            "test_snapshot.py and the snapshot-safe plan): a new prompt must go "
            "through the suspension framework (FrameCtx::ask or a pending-query "
            f"family), not a blocking get_input. found-vs-whitelist deltas: "
            f"found={added!r} whitelist={removed!r}")
    total = sum(found.values())
    return f"{total} get_input occurrences across {len(found)} files, all classified"


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
    ("combat_target_roundtrip", test_combat_target_roundtrip),
    ("damage_assign_roundtrip", test_damage_assign_roundtrip),
    ("resolution_single_roundtrip", test_resolution_single_roundtrip),
    ("dredge_roundtrip", test_dredge_roundtrip),
    ("pool_loop_roundtrip", test_pool_loop_roundtrip),
    ("pool_determinize_pin", test_pool_determinize_pin),
    ("trigger_order_roundtrip", test_trigger_order_roundtrip),
    ("trigger_target_roundtrip", test_trigger_target_roundtrip),
    ("legend_rule_roundtrip", test_legend_rule_roundtrip),
    ("etb_choose_type_roundtrip", test_etb_choose_type_roundtrip),
    ("etb_name_card_roundtrip", test_etb_name_card_roundtrip),
    ("unless_mana_roundtrip", test_unless_mana_roundtrip),
    ("yorion_blink_roundtrip", test_yorion_blink_roundtrip),
    ("subability_roundtrip", test_subability_roundtrip),
    ("charm_subchain_roundtrip", test_charm_subchain_roundtrip),
    ("cast_x_roundtrip", test_cast_x_roundtrip),
    ("cast_phyrexian_roundtrip", test_cast_phyrexian_roundtrip),
    ("cast_target_roundtrip", test_cast_target_roundtrip),
    ("charm_cast_roundtrip", test_charm_cast_roundtrip),
    ("storm_copy_target_roundtrip", test_storm_copy_target_roundtrip),
    ("delve_pick_roundtrip", test_delve_pick_roundtrip),
    ("alt_pitch_roundtrip", test_alt_pitch_roundtrip),
    ("activation_target_roundtrip", test_activation_target_roundtrip),
    ("activation_x_roundtrip", test_activation_x_roundtrip),
    ("equip_target_roundtrip", test_equip_target_roundtrip),
    ("mulligan_pregame_roundtrip", test_mulligan_pregame_roundtrip),
    ("pregame_preplaced_etb_roundtrip", test_pregame_preplaced_etb_roundtrip),
    ("rng_isolation", test_rng_isolation),
    ("determinize_efficacy", test_determinize_efficacy),
    ("terminal_intercept", test_terminal_intercept),
    ("determinize_invariants", test_determinize_invariants),
    ("sideboard_determinize", test_sideboard_determinize),
    ("sideboard_search_roundtrip", test_sideboard_search_roundtrip),
    ("sideboard_sim_result", test_sideboard_sim_result),
    ("sideboard_world_variation", test_sideboard_world_variation),
    ("prompt_site_guard", test_prompt_site_guard),
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
