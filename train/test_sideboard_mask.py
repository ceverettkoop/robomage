#!/usr/bin/env python3
"""Automated tests for the bo3 sideboard observation fixes.

At bo3 sideboard time the engine keeps the ended game's ECS alive, so a raw
sideboard-phase state vector describes the STALE terminal board of the previous
game. Three fixes are covered here:

  1. Env-side masking      — RoboMageEnv zeroes every block except the ones that
                             inform sideboarding (graveyards, action history,
                             match/library/turn ctx, opponent revealed multi-hot,
                             pending-decision ctx). Card-id slots go to the empty
                             sentinel, not vocab 0. (train/env.py)
  2. Pending-decision ctx  — during the OUT query, state[3183]/[3184] identify
                             the chosen IN card (which card the cut is FOR).
  3. Chooser repoint       — during sideboarding the history actor stamp and the
                             per-action controller_is_self flags name the
                             sideboarding seat, not whatever the ended game left.

The driver reads the engine's RAW (unmasked) machine-mode state so it can check
the engine-side fixes (2, 3); the masking check (1) applies env's shared mask
arrays to that raw state — the exact transform env performs in _read_until_query.

Run:  train/.venv/bin/python train/test_sideboard_mask.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import (STATE_SIZE, MAX_ACTIONS, _MATCH_CTX_START,  # noqa: E402
                 _SB_MASK_KEEP, _SB_MASK_FILL, _ACTION_CARD_ID_NULL,
                 _GLOBAL_SIZE, _SELF_PERM_START, _OPP_PERM_START, _PERM_SLOTS,
                 _PERM_SLOT_SIZE, _PERM_CARD_OFF, _HAND_START, _HAND_SLOTS_TOTAL,
                 _GY_START, _HIST_START, _HIST_END, _REVEALED_START, _REVEALED_END,
                 _PENDING_DECISION_START, _KNOWN_TOP_LIB_START,
                 _ACTION_HISTORY_ENTRY, ACTION_CATEGORY_MAX)
from card_costs import N_CARD_TYPES  # noqa: E402
from test_harness import (_card_to_deck_name, _TEMP_DECKS_DIR,  # noqa: E402
                          _DECKS_DIR, _BINARY, _BIN_DIR)

# ── Layout / enum constants ──────────────────────────────────────────────────
_SELF_IS_A_IDX = 32                 # state[32] == 1.0 when viewer is Player A
_IS_SB_IDX     = _MATCH_CTX_START + 3
_CAT_SB_IN, _CAT_SB_OUT, _CAT_SB_DONE = 24, 25, 26

_STATE_BYTES = STATE_SIZE * 4
_META_BYTES  = MAX_ACTIONS * 4

_failures = []


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


# ── Deck file writer (with a real sideboard section) ─────────────────────────

def _make_sb_deck(hand, library, sideboard, label):
    _TEMP_DECKS_DIR.mkdir(exist_ok=True)
    name = f"temp/_sbmask_{label}"
    path = _DECKS_DIR / f"{name}.dk"
    with open(path, "w") as f:
        for c in hand:
            f.write(f"1 {_card_to_deck_name(c)}\n")
        for c in library:
            f.write(f"1 {_card_to_deck_name(c)}\n")
        f.write("\nSIDEBOARD:\n")
        for qty, c in sideboard:
            f.write(f"{qty} {_card_to_deck_name(c)}\n")
    return name, str(path)


# ── Engine driver: records every decision's raw state + action metadata ──────

class Driver:
    def __init__(self, cmd):
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=str(_BIN_DIR))
        self.records = []

    def _read_exactly(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                raise EOFError("engine ended mid binary read")
            buf.extend(chunk)
        return bytes(buf)

    def run(self, action_fn, max_decisions=8000):
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
                _zone = np.frombuffer(self._read_exactly(_META_BYTES), dtype=np.int32).copy()

                catl = [int(c) for c in cats[:num]]
                rec = {
                    "state": state, "cats": catl, "ids": ids, "ctrl": ctrl,
                    "num": num, "self_is_a": bool(state[_SELF_IS_A_IDX] > 0.5),
                    "is_sb": bool(state[_IS_SB_IDX] > 0.5),
                }
                self.records.append(rec)

                choice = action_fn(state, catl, ids, ctrl, num)
                if choice is None:
                    choice = 0  # default: pass / take first choice
                rec["choice"] = choice
                self.proc.stdin.write(f"{choice}\n".encode())
                self.proc.stdin.flush()
                decisions += 1
        try:
            self.proc.stdin.close()
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        return self.records


def _make_sb_action(max_swaps=1):
    """Perform up to `max_swaps` IN->OUT swaps per seat's sideboard phase, then
    finish. Outside the sideboard phase, greedily play a land / cast if offered,
    else pass, so game 1 actually ends and the match reaches sideboarding."""
    st = {"seat": None, "swaps": 0}

    def fn(state, cats, ids, ctrl, num):
        if state[_IS_SB_IDX] > 0.5:
            seat = bool(state[_SELF_IS_A_IDX] > 0.5)
            if seat != st["seat"]:
                st["seat"], st["swaps"] = seat, 0
            if _CAT_SB_OUT in cats:                       # OUT menu
                return cats.index(_CAT_SB_OUT)
            if _CAT_SB_IN in cats and st["swaps"] < max_swaps:
                st["swaps"] += 1
                return cats.index(_CAT_SB_IN)             # IN menu -> pick a card
            return 0                                       # done sideboarding
        # In-game: prefer PLAY_LAND(11)/CAST_SPELL(10)/DECLARE_ATTACK — but a
        # simple "first non-pass, else pass" keeps game 1 progressing to an end.
        return 0

    return fn


def _drive_bo3(action_fn, seed=1, max_decisions=8000):
    a_hand = ["Mountain", "Mountain", "Lightning Bolt", "Dragon's Rage Channeler",
              "Lightning Bolt", "Dragon's Rage Channeler", "Mountain"]
    a_lib = ["Mountain"] * 6 + ["Lightning Bolt", "Dragon's Rage Channeler"] * 3 + ["Mountain"] * 10
    b_hand = ["Forest"] * 7
    b_lib = ["Forest"] * 30
    deck_a, _ = _make_sb_deck(a_hand, a_lib,
                              [(3, "Grizzly Bears"), (2, "Counterspell")], "a")
    deck_b, _ = _make_sb_deck(b_hand, b_lib,
                              [(3, "Grizzly Bears"), (2, "Lightning Bolt")], "b")
    cmd = [str(_BINARY), "--machine", "--narrative", "--no-shuffle", "--bo3",
           "--seed", str(seed), "--deck-a", deck_a, "--deck-b", deck_b]
    return Driver(cmd).run(action_fn, max_decisions=max_decisions)


# ── Test 1: env masking zeroes the stale board, keeps sideboard signal ───────

def _apply_mask(raw):
    masked = raw.copy()
    np.copyto(masked, _SB_MASK_FILL, where=~_SB_MASK_KEEP)
    return masked


def test_masking(records):
    print("\n=== Test: sideboard observation masking ===")
    sb = [r for r in records if r["is_sb"]]
    check(len(sb) > 0, f"reached the sideboard phase ({len(sb)} sideboard decisions)")
    if not sb:
        return

    raw = sb[0]["state"]
    # Non-vacuous: the raw sideboard state carries stale board info to be masked.
    stale = (float(np.sum(np.abs(raw[:_GLOBAL_SIZE]))) > 0.0)
    check(stale, "raw sideboard state has non-zero globals (stale board present to mask)")

    masked = _apply_mask(raw)

    # Kept blocks are byte-identical to raw.
    for lo, hi, label in (
        (_GY_START, _HAND_START, "graveyards"),
        (_HIST_START, _HIST_END, "action history"),
        (_MATCH_CTX_START, _KNOWN_TOP_LIB_START, "match/library/turn ctx"),
        (_REVEALED_START, _REVEALED_END, "revealed multi-hot"),
        (_PENDING_DECISION_START, STATE_SIZE, "pending-decision ctx"),
    ):
        check(np.array_equal(masked[lo:hi], raw[lo:hi]), f"kept unchanged: {label}")

    # Globals fully zeroed.
    check(np.all(masked[:_GLOBAL_SIZE] == 0.0), "globals [0:34] zeroed")

    # Permanent card-id slots -> empty sentinel; permanent scalar slots -> 0.
    sent_ok = scal_ok = True
    for start in (_SELF_PERM_START, _OPP_PERM_START):
        for s in range(_PERM_SLOTS):
            cid = start + s * _PERM_SLOT_SIZE + _PERM_CARD_OFF
            if not np.isclose(masked[cid], _ACTION_CARD_ID_NULL):
                sent_ok = False
            base = start + s * _PERM_SLOT_SIZE
            if np.any(masked[base:base + _PERM_CARD_OFF] != 0.0):
                scal_ok = False
    check(sent_ok, "permanent card-id slots -> empty sentinel (not vocab 0)")
    check(scal_ok, "permanent scalar slots -> 0")

    # Self hand card-id slots -> sentinel.
    hand = masked[_HAND_START:_HAND_START + _HAND_SLOTS_TOTAL]
    check(np.allclose(hand, _ACTION_CARD_ID_NULL), "self-hand card-id slots -> empty sentinel")


# ── Test 2: pending-decision context at the OUT query ────────────────────────

def test_pending_decision(records):
    print("\n=== Test: pending-decision context at OUT query ===")
    checked = 0
    for i, r in enumerate(records):
        if not (r["is_sb"] and _CAT_SB_OUT in r["cats"]):
            continue
        prev = records[i - 1]
        # The OUT menu is always preceded by the IN-menu pick that opened it.
        if not (prev["is_sb"] and _CAT_SB_IN in prev["cats"]
                and prev["cats"][prev["choice"]] == _CAT_SB_IN):
            continue
        chosen_in_id = float(prev["ids"][prev["choice"]])
        pend_id = float(r["state"][_PENDING_DECISION_START])
        pend_ctrl = float(r["state"][_PENDING_DECISION_START + 1])
        check(np.isclose(pend_id, chosen_in_id),
              f"OUT-query pending card id = chosen IN card "
              f"(state[3183]={pend_id:.5f}, in-card={chosen_in_id:.5f})")
        check(pend_ctrl == 1.0,
              f"OUT-query pending ctrl_is_self == 1.0 for the sideboarding seat "
              f"(state[3184]={pend_ctrl})")
        checked += 1
    check(checked > 0, f"exercised at least one OUT query ({checked})")


# ── Test 3: chooser repoint (history seat stamp + controller metadata) ───────

def _hist_entry(state, k):
    base = _HIST_START + k * _ACTION_HISTORY_ENTRY
    return (int(round(float(state[base]) * ACTION_CATEGORY_MAX)),  # category
            float(state[base + 2]))                                # is_self


def test_seat_flags(records):
    print("\n=== Test: sideboard chooser repoint (history + controller flags) ===")

    # (a) After a completed swap, the sideboarding seat's own IN/OUT actions must
    #     be stamped is_self=1.0 in its own action history (the repoint fixes the
    #     otherwise-stale actor stamp from the ended game).
    found = 0
    for i in range(1, len(records)):
        r, prev = records[i], records[i - 1]
        if not (r["is_sb"] and _CAT_SB_DONE in r["cats"]):
            continue                      # r must be an IN menu (offers "done")
        if not (prev["is_sb"] and _CAT_SB_OUT in prev["cats"]):
            continue                      # preceded by the OUT pick of a swap
        if prev["self_is_a"] != r["self_is_a"]:
            continue
        cat0, self0 = _hist_entry(r["state"], 0)   # newest: the OUT just chosen
        cat1, self1 = _hist_entry(r["state"], 1)   # next: the IN just chosen
        check(cat0 == _CAT_SB_OUT and self0 == 1.0,
              f"newest history = own SIDEBOARD_OUT, is_self=1.0 (cat={cat0}, self={self0})")
        check(cat1 == _CAT_SB_IN and self1 == 1.0,
              f"prior history = own SIDEBOARD_IN, is_self=1.0 (cat={cat1}, self={self1})")
        found += 1
    check(found > 0, f"found a post-swap sideboard observation ({found})")

    # (b) Per-action metadata: SB_IN/SB_OUT sources are zone-less template
    #     entities, so the controller flag is the null sentinel (not 1.0/0.0) —
    #     but the card id IS emitted, so the choice's card identity stays visible.
    id_ok, ctrl_null_ok, n_checked = True, True, 0
    for r in records:
        if not r["is_sb"]:
            continue
        for j, c in enumerate(r["cats"]):
            if c in (_CAT_SB_IN, _CAT_SB_OUT):
                n_checked += 1
                if float(r["ids"][j]) < 0.0:            # emitted, not id-null
                    id_ok = False
                if not np.isclose(float(r["ctrl"][j]), _ACTION_CARD_ID_NULL):
                    ctrl_null_ok = False
    check(n_checked > 0 and id_ok,
          f"every SB_IN/SB_OUT action emits its card id ({n_checked} actions)")
    check(n_checked > 0 and ctrl_null_ok,
          "SB_IN/SB_OUT controller flag is the null sentinel (zone-less source)")


def main():
    records = _drive_bo3(_make_sb_action())
    test_masking(records)
    test_pending_decision(records)
    test_seat_flags(records)
    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)}):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
