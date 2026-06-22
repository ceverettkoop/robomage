#!/usr/bin/env python3
"""Automated tests for the opponent revealed-cards accumulator (inferred decklist).

Drives the engine in machine mode and inspects the revealed-cards multi-hot
block (state[_REVEALED_START : _REVEALED_START+128]) to verify:

  1. Empty start          — at game-1 turn-1 nothing is revealed (not god-mode).
  2. Single-game reveal    — a card the opponent casts/plays flips its bit and the
                             bit stays set after the card leaves the stack.
  3. Negative              — a card that only ever sits in the opponent's library
                             is never marked.
  4. Tutor reveal          — a card revealed by Personal Tutor but put to a hidden
                             zone (top of library) is still marked.
  5. Cross-game (bo3)      — a card revealed in game 1 is still set in game 2,
                             proving the accumulator survives the per-game ECS reset.

Run:  train/.venv/bin/python train/test_revealed_accumulator.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import STATE_SIZE, MAX_ACTIONS, _MANDATORY_CATS  # noqa: E402
from decode import decode_game_state  # noqa: E402
from test_harness import get_scripted_action, _make_deck_file, _BINARY, _BIN_DIR  # noqa: E402

# ── Layout constants (mirror src/machine_io.h) ───────────────────────────────
_REVEALED_START   = 33666           # opponent revealed-cards multi-hot start
_REVEALED_SIZE    = 128
_SELF_IS_A_IDX    = 32              # state[32] == 1.0 when viewer is Player A
_GAME_NUMBER_IDX  = 33018          # state[33018] = match game_number / 3.0

VOCAB = {
    "Mountain": 0, "Forest": 1, "Lightning Bolt": 2, "Grizzly Bears": 3,
    "Island": 19, "Ponder": 11, "Counterspell": 22, "Personal Tutor": 59,
    "Dragon's Rage Channeler": 20,
}

_STATE_BYTES = STATE_SIZE * 4
_META_BYTES  = MAX_ACTIONS * 4


def _bit(state, card):
    return state[_REVEALED_START + VOCAB[card]]


def _revealed_names(state):
    """Names whose revealed bit is set in this (opponent-of-viewer) state."""
    block = state[_REVEALED_START:_REVEALED_START + _REVEALED_SIZE]
    idxs = np.nonzero(block > 0.5)[0]
    inv = {v: k for k, v in VOCAB.items()}
    return [inv.get(int(i), f"idx{int(i)}") for i in idxs]


# ── Engine driver ────────────────────────────────────────────────────────────

class Driver:
    """Runs the engine with the scripted agent on both sides, recording every
    decision-point observation along with the running narrative."""

    def __init__(self, cmd):
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=str(_BIN_DIR))
        # Each record: dict(state, self_is_a, game_number, narrative_before)
        self.records = []
        self.narrative = []  # full narrative log (all lines)

    def _read_exactly(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                raise EOFError("engine ended mid binary read")
            buf.extend(chunk)
        return bytes(buf)

    def run(self, max_decisions=4000, action_fn=None):
        pending_narr = []
        decisions = 0
        while decisions < max_decisions:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.rstrip(b"\n")
            if line.startswith(b"BQUERY: "):
                n = int(line[8:])
                num = min(n, MAX_ACTIONS)
                state = np.frombuffer(self._read_exactly(_STATE_BYTES), dtype=np.float32).copy()
                cats  = np.frombuffer(self._read_exactly(_META_BYTES), dtype=np.int32).copy()
                ids   = np.frombuffer(self._read_exactly(_META_BYTES), dtype=np.float32).copy()
                ctrl  = np.frombuffer(self._read_exactly(_META_BYTES), dtype=np.float32).copy()
                _pub  = np.frombuffer(self._read_exactly(_META_BYTES), dtype=np.float32).copy()

                self.records.append({
                    "state": state,
                    "self_is_a": state[_SELF_IS_A_IDX] > 0.5,
                    "game_number": int(round(state[_GAME_NUMBER_IDX] * 3.0)),
                    "narrative_before": list(pending_narr),
                })
                pending_narr = []

                choice = None
                if action_fn is not None:
                    choice = action_fn(state, cats, ids, ctrl, num)
                if choice is None:
                    choice = get_scripted_action(state, cats, ids, ctrl, num)
                game_action = choice
                if any(int(cats[i]) in _MANDATORY_CATS for i in range(num)) and choice == num - 1:
                    game_action = -1
                self.proc.stdin.write(f"{game_action}\n".encode())
                self.proc.stdin.flush()
                decisions += 1
            else:
                text = line.decode("ascii", errors="replace")
                if text.strip():
                    self.narrative.append(text)
                    pending_narr.append(text)
        try:
            self.proc.stdin.close()
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        return self.records


def _drive(deck_a, deck_b, bo3=False, seed=1, max_decisions=4000):
    cmd = [str(_BINARY), "--machine", "--narrative", "--no-shuffle",
           "--seed", str(seed), "--deck-a", deck_a, "--deck-b", deck_b]
    if bo3:
        cmd.append("--bo3")
    return Driver(cmd).run(max_decisions=max_decisions)


# ── Assertions helper ────────────────────────────────────────────────────────

_failures = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        _failures.append(msg)


# ── Test 1+2+3: single-game reveal, empty start, negative ─────────────────────

def test_single_game():
    print("\n=== Test: single-game reveal / empty-start / negative ===")
    # Player B (the revealer) casts Dragon's Rage Channeler (R) and Lightning
    # Bolts — both castable off Mountains. 'Counterspell' sits at the bottom of
    # B's library, uncastable (no blue mana) and never drawn in a decisive game
    # -> must never be revealed.
    drc = "Dragon's Rage Channeler"
    b_hand = ["Mountain", "Mountain", "Lightning Bolt", drc,
              "Lightning Bolt", drc, "Mountain"]
    b_lib = (["Mountain"] * 3 + ["Lightning Bolt", drc] +
             ["Mountain"] * 22 + ["Counterspell"])
    a_hand = ["Forest"] * 7
    a_lib = ["Forest"] * 25
    deck_a, _ = _make_deck_file(a_hand, a_lib, "rev_a")
    deck_b, _ = _make_deck_file(b_hand, b_lib, "rev_b")

    records = _drive(deck_a, deck_b, max_decisions=4000)
    a_views = [r for r in records if r["self_is_a"]]
    check(len(a_views) > 0, f"collected Player-A-perspective observations ({len(a_views)})")

    # 1. Empty start: the very first A-perspective observation has nothing revealed.
    first = a_views[0]["state"]
    first_sum = float(np.sum(first[_REVEALED_START:_REVEALED_START + _REVEALED_SIZE]))
    check(first_sum == 0.0,
          f"accumulator empty at first observation (sum={first_sum}, names={_revealed_names(first)})")

    # 2. Positive: B's DRC (cast -> stack/battlefield, public) gets revealed.
    drc_ever = any(_bit(r["state"], drc) > 0.5 for r in a_views)
    check(drc_ever, f"{drc} revealed from Player A's perspective after B casts it")
    bolt_ever = any(_bit(r["state"], "Lightning Bolt") > 0.5 for r in a_views)
    check(bolt_ever, "Lightning Bolt revealed from Player A's perspective after B casts it")

    # 2b. Persistence within the game: once DRC is set it never clears.
    seen = False
    monotonic = True
    for r in a_views:
        b = _bit(r["state"], drc) > 0.5
        if b:
            seen = True
        elif seen:
            monotonic = False
            break
    check(seen and monotonic, f"{drc} bit is monotonic (never clears once set)")

    # 3. Negative: Counterspell stays in B's library -> never revealed.
    counter_ever = any(_bit(r["state"], "Counterspell") > 0.5 for r in a_views)
    check(not counter_ever, "Counterspell (only ever in B's library) is never revealed")

    # 3b. Perspective: from B's perspective, opponent A (lands only) reveals only Forest.
    b_views = [r for r in records if not r["self_is_a"]]
    a_revealed_nonland = set()
    for r in b_views:
        for name in _revealed_names(r["state"]):
            if name not in ("Forest", "Mountain", "Island"):
                a_revealed_nonland.add(name)
    check(len(a_revealed_nonland) == 0,
          f"Player A (lands only) reveals no spells from B's perspective ({a_revealed_nonland or 'none'})")


# ── Test 4: tutor reveal (card revealed but put to a hidden zone) ─────────────

def test_tutor_reveal():
    print("\n=== Test: tutor reveal (Personal Tutor -> top of library) ===")
    # B starts with an Island in play and Personal Tutor in hand; its only library
    # sorcery is Ponder, so the tutor deterministically reveals Ponder and puts it
    # on TOP of the library (a hidden zone). Only the tutor reveal hook can set
    # Ponder's bit there.
    from test_harness import _DECKS_DIR
    # Build decks manually so we control exact library order; B: Personal Tutor in
    # hand, Ponder as the single sorcery in library.
    # Ponder is buried mid-library so B does not simply draw it before tutoring;
    # when Personal Tutor resolves it is still in the library to be found.
    b_hand = ["Personal Tutor", "Island", "Island", "Forest", "Forest", "Forest", "Forest"]
    b_lib = ["Forest"] * 5 + ["Ponder"] + ["Forest"] * 20
    a_hand = ["Forest"] * 7
    a_lib = ["Forest"] * 25
    deck_a, _ = _make_deck_file(a_hand, a_lib, "tut_a")
    deck_b, _ = _make_deck_file(b_hand, b_lib, "tut_b")

    # The generic scripted agent picks "fail to find" (index 0) for a top-of-library
    # tutor search, so override: when a search offers Ponder, pick it.
    _SEARCH_CATS = (19, 20)  # SEARCH_LIBRARY, TOP_LIBRARY

    def tutor_picker(state, cats, ids, ctrl, num):
        if any(int(cats[i]) in _SEARCH_CATS for i in range(num)):
            for i in range(num):
                if int(cats[i]) in _SEARCH_CATS:
                    card_idx = int(round(ids[i] * _REVEALED_SIZE))
                    if card_idx == VOCAB["Ponder"]:
                        return i
        return None  # defer to the scripted agent

    cmd = [str(_BINARY), "--machine", "--narrative", "--no-shuffle",
           "--seed", "1", "--deck-a", deck_a, "--deck-b", deck_b,
           "--battlefield-b", "Island"]
    drv = Driver(cmd)
    records = drv.run(max_decisions=2000, action_fn=tutor_picker)

    # Confirm the tutor actually fired and revealed Ponder to the library.
    reveal_lines = [l for l in drv.narrative
                    if "reveals Ponder" in l and "top of library" in l]
    check(len(reveal_lines) > 0,
          f"Personal Tutor revealed Ponder to top of library (narrative: {reveal_lines[:1]})")

    # From A's perspective, Ponder's bit must be set AND Ponder must be hidden
    # (not on B's battlefield / graveyard) at that moment -> proves the tutor hook.
    proof = False
    for r in records:
        if not r["self_is_a"]:
            continue
        if _bit(r["state"], "Ponder") <= 0.5:
            continue
        gs = decode_game_state(r["state"])
        opp_bf = {p["name"] for p in gs["opp_battlefield"]}
        opp_gy = set(gs["opp_graveyard"])
        if "Ponder" not in opp_bf and "Ponder" not in opp_gy:
            proof = True
            break
    check(proof, "Ponder revealed from A's perspective while still hidden (in library, not bf/gy)")

    # Personal Tutor itself (cast -> stack, public) is also revealed.
    pt_ever = any(r["self_is_a"] and _bit(r["state"], "Personal Tutor") > 0.5 for r in records)
    check(pt_ever, "Personal Tutor (cast spell) revealed from A's perspective")


# ── Test 5: bo3 cross-game persistence ───────────────────────────────────────

def test_bo3_persistence():
    print("\n=== Test: bo3 cross-game persistence (survives ECS reset) ===")
    # B reliably reveals DRC + Lightning Bolt and beats A down; the match advances
    # past game 1. The revealed bits must persist into game 2.
    drc = "Dragon's Rage Channeler"
    b_hand = ["Mountain", "Mountain", drc, "Lightning Bolt",
              drc, "Lightning Bolt", "Mountain"]
    b_lib = (["Mountain"] * 4 + [drc, "Lightning Bolt"] * 2 +
             ["Mountain"] * 18)
    a_hand = ["Forest"] * 7
    a_lib = ["Forest"] * 30
    deck_a, _ = _make_deck_file(a_hand, a_lib, "bo3_a")
    deck_b, _ = _make_deck_file(b_hand, b_lib, "bo3_b")

    records = _drive(deck_a, deck_b, bo3=True, max_decisions=8000)

    game_numbers = sorted({r["game_number"] for r in records})
    print(f"  (observed game numbers: {game_numbers})")
    check(max(game_numbers) >= 1, "match advanced to game 2 or beyond")

    a_views = [r for r in records if r["self_is_a"]]

    # Bit becomes set somewhere in game 1.
    g1_set = any(r["game_number"] == 0 and _bit(r["state"], drc) > 0.5
                 for r in a_views)
    check(g1_set, f"{drc} revealed during game 1 (from A's perspective)")

    # Core property: in game 2+, the bit is STILL set from the first A observation
    # of that game — i.e. it survived init_ecs().
    later = [r for r in a_views if r["game_number"] >= 1]
    if later:
        first_g2 = later[0]
        check(_bit(first_g2["state"], drc) > 0.5,
              "DRC still revealed at the first Player-A observation of game 2")
        # Once set, never clears across the whole match.
        seen = False
        monotonic = True
        for r in a_views:
            b = _bit(r["state"], drc) > 0.5
            if b:
                seen = True
            elif seen:
                monotonic = False
                break
        check(monotonic, "DRC bit never clears across game boundary")
    else:
        check(False, "no game-2 Player-A observations collected")


def main():
    test_single_game()
    test_tutor_reveal()
    test_bo3_persistence()
    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)}):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
