#!/usr/bin/env python3
"""Obs-invariant regression: drive a few seeded scripted games and assert
per-decision invariants on the RAW observation state vector.

This is a cheap structural gate on the machine-mode observation: whatever cards
are played, whatever the board looks like, the *encoding* must stay
self-consistent — card-id floats decode to a valid vocab index or the empty
sentinel, entity refs decode into range, recency-packed zones have no holes,
one-hots are one-hot, etc. A regression in serialize_state (a stale offset, a
truncation bug, a sentinel written as 0.0) corrupts training silently; this
catches it deterministically.

ZERO MAGIC NUMBERS. Every offset, width, and normalizer is imported from
``env``/``_enums`` (which mirror the C++ headers via codegen), so the file is
layout-change-proof: move a block in src/machine_io.h, regenerate, and the
checks follow automatically.

Wired into ``train/ci_check.py`` as the ``obsinv`` tier (so ``make check`` runs
it); also runnable standalone::

    train/.venv/bin/python train/test_obs_invariants.py
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import decode
import runner
from env import (
    RoboMageEnv, STATE_SIZE, N_CARD_TYPES, N_ENTITY_REF_SLOTS, BIN_DIR,
    ACTION_CATEGORY_MAX, MAX_ACTIONS, OPTION_ORDINAL_MAX,
    _SELF_PERM_START, _OPP_PERM_START, _PERM_SLOTS, _PERM_SLOT_SIZE,
    _PERM_CHOSEN_NAME_OFF, _PERM_RETURNABLE_OFF, _PERM_CARD_OFF,
    _OFF_ATTACHED_TO, _OFF_ATTACHED_BY, _OFF_ATTACK_TGT, _OFF_BLOCKING_TGT,
    _STACK_START, _STACK_SLOTS, _STACK_SLOT_SIZE, _STACK_TGT_START,
    _STACK_TGT_SLOTS, _STACK_TGT_FIELDS,
    _GY_START, _GY_SLOT_SIZE, _EXILE_START, _EXILE_SLOT_SIZE,
    _HAND_START, _HAND_SLOT_SIZE, MAX_GY_SLOTS, MAX_HAND_SLOTS,
    _KNOWN_TOP_LIB_START, _KNOWN_TOP_LIB_END,
    _OPP_KNOWN_HAND_START, _OPP_KNOWN_HAND_END,
    _PENDING_DECISION_START, _HIST_START, _ACTION_HISTORY_SIZE,
    _ACTION_HISTORY_ENTRY, _STEP_ONEHOT_START, _STEP_ONEHOT_SIZE,
    _EXTRAS_MC_ONEHOT_START, _SELF_BLOCK_START, _OPP_BLOCK_START,
    _PB_LIFE, _PB_HAND_CT, _LIBRARY_CTX_START, _REVEALED_START,
    _SELF_LIVE_LIB_START, _OPP_DECK_MAIN_START, _OPP_DECK_SIDE_START,
    _DECKLIST_SLOT_SIZE)
from _enums import (N_MANDATORY_CHOICES, DECKLIST_MAIN_SLOTS,
                    DECKLIST_SIDE_SLOTS, CAT_ACTIVATE_ABILITY)
from opponents import make_controller

# Card-id decode: sentinel (empty/unknown) -> -1; a real id -> [0, N_CARD_TYPES).
_CARD_ID_SENTINEL = -1
# Entity-slot ref decode range (unified viewer-relative space; -1 = none).
_REF_MIN, _REF_MAX = -1, N_ENTITY_REF_SLOTS - 1

# Derived per-side zone starts (mirrors decode.py's derivation).
_OPP_GY_START = _GY_START + MAX_GY_SLOTS * _GY_SLOT_SIZE
_OPP_EXILE_START = _EXILE_START + MAX_GY_SLOTS * _EXILE_SLOT_SIZE

_DECKS_DIR = os.path.join(BIN_DIR, "resources", "decks")


class InvariantError(AssertionError):
    """A per-decision observation invariant was violated."""


def _fail(decision_idx, seat, block, slot, val, msg):
    raise InvariantError(
        f"[decision {decision_idx}, seat {seat}] {block}[slot {slot}] "
        f"raw={val!r}: {msg}")


def _decode_card_id(val):
    """round(v*N_CARD_TYPES): -1 for the empty sentinel, else the vocab index."""
    return int(round(float(val) * N_CARD_TYPES))


def _decode_ref(val):
    """round(v*108)-1: -1 for 'none', else the entity-slot index."""
    return int(round(float(val) * N_ENTITY_REF_SLOTS)) - 1


# ── Slot enumerators (name, slot, absolute state offset) ──────────────────────
# Every id-family and ref-family float in the state vector, generated purely
# from the imported layout constants so a layout change reroutes them for free.

def _card_id_slots():
    """Yield (block_name, slot_key, offset) for every card-id-family float."""
    for side, start in (("self_perm", _SELF_PERM_START),
                        ("opp_perm", _OPP_PERM_START)):
        for s in range(_PERM_SLOTS):
            base = start + s * _PERM_SLOT_SIZE
            yield f"{side}.chosen_name", s, base + _PERM_CHOSEN_NAME_OFF
            yield f"{side}.returnable", s, base + _PERM_RETURNABLE_OFF
            yield f"{side}.card_id", s, base + _PERM_CARD_OFF
    for s in range(_STACK_SLOTS):
        base = _STACK_START + s * _STACK_SLOT_SIZE
        yield "stack.card_id", s, base + 1                 # [0]=ctrl, [1]=card id
        for t in range(_STACK_TGT_SLOTS):
            tbase = base + _STACK_TGT_START + t * _STACK_TGT_FIELDS
            yield f"stack.tgt{t}.card_id", s, tbase + (_STACK_TGT_FIELDS - 1)
    for name, start in (("self_gy", _GY_START), ("opp_gy", _OPP_GY_START)):
        for i in range(MAX_GY_SLOTS):
            yield name, i, start + i * _GY_SLOT_SIZE
    for name, start in (("self_exile", _EXILE_START),
                        ("opp_exile", _OPP_EXILE_START)):
        for i in range(MAX_GY_SLOTS):
            yield name, i, start + i * _EXILE_SLOT_SIZE
    for i in range(MAX_HAND_SLOTS):
        yield "self_hand", i, _HAND_START + i * _HAND_SLOT_SIZE
    for i, off in enumerate(range(_KNOWN_TOP_LIB_START, _KNOWN_TOP_LIB_END)):
        yield "known_top", i, off
    for i, off in enumerate(range(_OPP_KNOWN_HAND_START, _OPP_KNOWN_HAND_END)):
        yield "opp_known_hand", i, off
    yield "pending_decision", 0, _PENDING_DECISION_START
    for e in range(_ACTION_HISTORY_SIZE):
        yield "history.card_id", e, _HIST_START + e * _ACTION_HISTORY_ENTRY + 1
    # Deck-identity tail blocks: card id is the first float of each (card_id, count) slot.
    for name, start, n in _DECKLIST_BLOCKS:
        for s in range(n):
            yield name, s, start + s * _DECKLIST_SLOT_SIZE


# The three deck-identity blocks: (name, start offset, slot count).
_DECKLIST_BLOCKS = (
    ("self_live_lib", _SELF_LIVE_LIB_START, DECKLIST_MAIN_SLOTS),
    ("opp_deck_main", _OPP_DECK_MAIN_START, DECKLIST_MAIN_SLOTS),
    ("opp_deck_side", _OPP_DECK_SIDE_START, DECKLIST_SIDE_SLOTS),
)


def _decode_decklist_block(state, start, n_slots):
    """Return [(vocab_id, count_int)] for every slot (id=-1 => empty)."""
    out = []
    for s in range(n_slots):
        base = start + s * _DECKLIST_SLOT_SIZE
        cid = _decode_card_id(state[base])
        cnt = int(round(float(state[base + 1]) * 4.0))
        out.append((cid, cnt))
    return out


def _ref_slots():
    """Yield (block_name, slot_key, offset) for every norm_ref-family float."""
    for side, start in (("self_perm", _SELF_PERM_START),
                        ("opp_perm", _OPP_PERM_START)):
        for s in range(_PERM_SLOTS):
            base = start + s * _PERM_SLOT_SIZE
            yield f"{side}.attached_to", s, base + _OFF_ATTACHED_TO
            yield f"{side}.attached_by", s, base + _OFF_ATTACHED_BY
            yield f"{side}.attack_tgt", s, base + _OFF_ATTACK_TGT
            yield f"{side}.blocking_tgt", s, base + _OFF_BLOCKING_TGT
    for s in range(_STACK_SLOTS):
        base = _STACK_START + s * _STACK_SLOT_SIZE
        for t in range(_STACK_TGT_SLOTS):
            tbase = base + _STACK_TGT_START + t * _STACK_TGT_FIELDS
            yield f"stack.tgt{t}.slot_ref", s, tbase + 3   # [+3]=slot_ref


def _zone_block_offsets(start):
    """Absolute offsets of a MAX_GY_SLOTS-wide recency-packed zone block."""
    return [start + i * _GY_SLOT_SIZE for i in range(MAX_GY_SLOTS)]


# ── The invariant checks (all read `state` = obs[:STATE_SIZE]) ─────────────────

def check_decision(decision_idx, obs, priority_is_a, companion_by_seat, is_pregame,
                   deck_block_by_seat, num_choices=None):
    """Assert every observation invariant for one decision. Raises on violation.
    `num_choices` (when known) additionally enables the per-menu activation-
    ordinal uniqueness check (11)."""
    state = obs[:STATE_SIZE]
    seat = "A" if priority_is_a else "B"

    # (1) Every card-id-family float decodes to -1 or [0, N_CARD_TYPES).
    for block, slot, off in _card_id_slots():
        v = state[off]
        if not np.isfinite(v):
            _fail(decision_idx, seat, block, slot, v, "non-finite card-id float")
        cid = _decode_card_id(v)
        if cid != _CARD_ID_SENTINEL and not (0 <= cid < N_CARD_TYPES):
            _fail(decision_idx, seat, block, slot, v,
                  f"card id {cid} out of range (expected -1 or [0,{N_CARD_TYPES}))")

    # (2) Every norm_ref float round-trips into [-1, N_ENTITY_REF_SLOTS-1].
    for block, slot, off in _ref_slots():
        v = state[off]
        if not np.isfinite(v):
            _fail(decision_idx, seat, block, slot, v, "non-finite ref float")
        ref = _decode_ref(v)
        if not (_REF_MIN <= ref <= _REF_MAX):
            _fail(decision_idx, seat, block, slot, v,
                  f"entity ref {ref} out of range [{_REF_MIN},{_REF_MAX}]")

    # (3) GY/exile blocks are sentinel-suffixed (recency-packed, no holes).
    for block, start in (("self_gy", _GY_START), ("opp_gy", _OPP_GY_START),
                        ("self_exile", _EXILE_START), ("opp_exile", _OPP_EXILE_START)):
        seen_empty = False
        for i, off in enumerate(_zone_block_offsets(start)):
            empty = _decode_card_id(state[off]) == _CARD_ID_SENTINEL
            if seen_empty and not empty:
                _fail(decision_idx, seat, block, i, state[off],
                      "non-empty slot after an empty one (block must be recency-packed)")
            seen_empty = seen_empty or empty

    # (4) A non-sentinel returnable-exile id implies that id is in an exile block.
    exile_ids = set()
    for start in (_EXILE_START, _OPP_EXILE_START):
        for off in _zone_block_offsets(start):
            cid = _decode_card_id(state[off])
            if cid != _CARD_ID_SENTINEL:
                exile_ids.add(cid)
    for side, start in (("self_perm", _SELF_PERM_START),
                        ("opp_perm", _OPP_PERM_START)):
        for s in range(_PERM_SLOTS):
            base = start + s * _PERM_SLOT_SIZE
            rcid = _decode_card_id(state[base + _PERM_RETURNABLE_OFF])
            if rcid != _CARD_ID_SENTINEL and rcid not in exile_ids:
                _fail(decision_idx, seat, f"{side}.returnable", s,
                      state[base + _PERM_RETURNABLE_OFF],
                      f"returnable-exile id {rcid} not present in any exile block "
                      f"(exile ids: {sorted(exile_ids)})")

    # (5) chosen_name is only set on a slot whose card_id is non-sentinel.
    for side, start in (("self_perm", _SELF_PERM_START),
                        ("opp_perm", _OPP_PERM_START)):
        for s in range(_PERM_SLOTS):
            base = start + s * _PERM_SLOT_SIZE
            name_cid = _decode_card_id(state[base + _PERM_CHOSEN_NAME_OFF])
            card_cid = _decode_card_id(state[base + _PERM_CARD_OFF])
            if name_cid != _CARD_ID_SENTINEL and card_cid == _CARD_ID_SENTINEL:
                _fail(decision_idx, seat, f"{side}.chosen_name", s,
                      state[base + _PERM_CHOSEN_NAME_OFF],
                      "chosen_name set on an empty (sentinel card_id) slot")

    # (6) One-hots are one-hot: exactly one step bit; at most one mandatory bit.
    step_bits = int(np.sum(
        state[_STEP_ONEHOT_START:_STEP_ONEHOT_START + _STEP_ONEHOT_SIZE] > 0.5))
    if step_bits != 1:
        _fail(decision_idx, seat, "step_onehot", "-", step_bits,
              f"step one-hot has {step_bits} bits set (expected exactly 1)")
    mc_bits = int(np.sum(
        state[_EXTRAS_MC_ONEHOT_START:_EXTRAS_MC_ONEHOT_START + N_MANDATORY_CHOICES] > 0.5))
    if mc_bits > 1:
        _fail(decision_idx, seat, "mandatory_choice_onehot", "-", mc_bits,
              f"mandatory-choice one-hot has {mc_bits} bits set (expected <= 1)")

    # (7) Player-block sanity: life / hand / library counts finite & non-negative.
    for label, base in (("self", _SELF_BLOCK_START), ("opp", _OPP_BLOCK_START)):
        life = float(state[base + _PB_LIFE]) * 20.0
        hand = float(state[base + _PB_HAND_CT]) * 10.0
        for field, val in (("life", life), ("hand_count", hand)):
            if not np.isfinite(val) or val < -0.5:  # -0.5: tolerate float rounding at 0
                _fail(decision_idx, seat, f"{label}_player.{field}", "-", val,
                      f"{field} de-normalizes to {val} (expected finite & >= 0)")
    for label, off in (("self", _LIBRARY_CTX_START), ("opp", _LIBRARY_CTX_START + 1)):
        lib = float(state[off]) * 60.0
        if not np.isfinite(lib) or lib < -0.5:
            _fail(decision_idx, seat, f"{label}_player.library", "-", lib,
                  f"library count de-normalizes to {lib} (expected finite & >= 0)")

    # (8) Companion: a declared companion is revealed to the opponent for the
    # whole game proper. When the seat WITHOUT priority (the viewer's opponent)
    # declared a companion, its bit must be set in the revealed multi-hot.
    # Skipped during pregame (mulligan/bottom): this engine reveals the companion
    # in the post-mulligan game setup (src/main.cpp setup_companions), so the bit
    # is not yet present while mulligans/bottoming are still being decided.
    if not is_pregame:
        opp_seat = "B" if priority_is_a else "A"
        comp_idx = companion_by_seat.get(opp_seat)
        if comp_idx is not None:
            if not (state[_REVEALED_START + comp_idx] > 0.5):
                _fail(decision_idx, seat, "opp_revealed.companion", comp_idx,
                      state[_REVEALED_START + comp_idx],
                      f"opponent (seat {opp_seat}) declared a companion (vocab {comp_idx}) "
                      "but its revealed bit is not set")

    # (9) Deck-identity blocks: packed ascending by vocab id (no holes), counts in
    # range, self-live-library counts sum to self_library_ct, opp static blocks
    # constant per viewer seat.
    self_lib_sum = 0
    for name, start, n in _DECKLIST_BLOCKS:
        entries = _decode_decklist_block(state, start, n)
        seen_empty = False
        prev_id = -1
        for i, (cid, cnt) in enumerate(entries):
            empty = (cid == _CARD_ID_SENTINEL)
            if empty:
                # Empty slot: count must be ~0.
                if cnt != 0:
                    _fail(decision_idx, seat, name, i, cnt,
                          f"empty slot carries a non-zero count {cnt}")
                seen_empty = True
                continue
            if seen_empty:
                _fail(decision_idx, seat, name, i, cid,
                      "non-empty slot after an empty one (block must be packed)")
            if not (0 <= cid < N_CARD_TYPES):
                _fail(decision_idx, seat, name, i, cid,
                      f"card id {cid} out of range [0,{N_CARD_TYPES})")
            if cid <= prev_id:
                _fail(decision_idx, seat, name, i, cid,
                      f"card id {cid} not strictly greater than previous {prev_id} "
                      "(block must be ascending by vocab id)")
            prev_id = cid
            # A non-empty slot must carry >= 1 copy. The upper bound is a generous
            # deck-size cap (basics in a big-mana / 80-card Yorion list legitimately
            # far exceed the /4 normalizer's ~1.0 for a 4-of, so no tight bound holds).
            if not (1 <= cnt <= 100):
                _fail(decision_idx, seat, name, i, cnt,
                      f"count {cnt} out of range [1,100] on a non-empty slot")
            if name == "self_live_lib":
                self_lib_sum += cnt

    # (9c) self-live-library counts sum to the viewer's library card count.
    lib_ct = int(round(float(state[_LIBRARY_CTX_START]) * 60.0))
    if self_lib_sum != lib_ct:
        _fail(decision_idx, seat, "self_live_lib.sum", "-", self_lib_sum,
              f"library counts sum to {self_lib_sum} but self_library_ct is {lib_ct}")

    # (9d) opp static maindeck+sideboard blocks are constant across a game per seat.
    opp_block = (
        tuple(state[_OPP_DECK_MAIN_START:_OPP_DECK_MAIN_START
                    + DECKLIST_MAIN_SLOTS * _DECKLIST_SLOT_SIZE]),
        tuple(state[_OPP_DECK_SIDE_START:_OPP_DECK_SIDE_START
                    + DECKLIST_SIDE_SLOTS * _DECKLIST_SLOT_SIZE]),
    )
    prev = deck_block_by_seat.get(seat)
    if prev is not None and prev != opp_block:
        _fail(decision_idx, seat, "opp_deck.constancy", "-", "changed",
              "opponent static decklist block changed across decisions of the same seat")
    deck_block_by_seat[seat] = opp_block

    # (10) Every per-action option_ordinal float round-trips into
    # [-1, OPTION_ORDINAL_MAX]. The ords block is the 6th (last) action-metadata
    # block, normalized (ord + 1) / (OPTION_ORDINAL_MAX + 1) so -1 (n/a) -> 0.0.
    _ords_start = STATE_SIZE + 5 * MAX_ACTIONS
    for i in range(MAX_ACTIONS):
        v = obs[_ords_start + i]
        if not np.isfinite(v):
            _fail(decision_idx, seat, "action.option_ordinal", i, v,
                  "non-finite option_ordinal float")
        ordv = int(round(float(v) * (OPTION_ORDINAL_MAX + 1))) - 1
        if not (-1 <= ordv <= OPTION_ORDINAL_MAX):
            _fail(decision_idx, seat, "action.option_ordinal", i, v,
                  f"option_ordinal {ordv} out of range [-1,{OPTION_ORDINAL_MAX}]")

    # (11) Same-permanent activations are distinguishable. Every
    # ACTIVATE_ABILITY action carries option_ordinal >= 0 (the ability's index
    # in its source's ability list; synthesised equip/unattach use 32/33), and
    # two activations referencing the SAME entity slot never share an ordinal —
    # this is the encoding that lets the policy tell a planeswalker's loyalty
    # abilities apart (they are identical in every other action feature).
    if num_choices:
        cats = decode.action_categories(obs, num_choices)
        slots = decode.action_slot_refs(obs, num_choices)
        ords = decode.action_ordinals(obs, num_choices)
        seen_by_slot = {}
        for i in range(num_choices):
            if int(cats[i]) != CAT_ACTIVATE_ABILITY:
                continue
            o = int(ords[i])
            if o < 0:
                _fail(decision_idx, seat, "action.option_ordinal", i, o,
                      "ACTIVATE_ABILITY action without an ability ordinal")
            slot = int(slots[i])
            if slot < 0:
                continue          # no referenced entity slot (hand/gy source)
            if o in seen_by_slot.setdefault(slot, set()):
                _fail(decision_idx, seat, "action.option_ordinal", i, o,
                      f"duplicate ability ordinal {o} on entity slot {slot} — "
                      "same-permanent activations are indistinguishable")
            seen_by_slot[slot].add(o)


# ── Game driving ──────────────────────────────────────────────────────────────

def _make_scripted_pair(deck_a, deck_b):
    """Two scripted-hard controllers, primed like runner.run_games does."""
    ctrl_a = make_controller("scripted")
    ctrl_b = make_controller("scripted")
    for ctrl in (ctrl_a, ctrl_b):
        set_names = getattr(ctrl, "set_deck_names", None)
        if set_names is not None:
            set_names(deck_a, deck_b)
        new_game = getattr(ctrl, "new_game", None)
        if new_game is not None:
            new_game()
    return ctrl_a, ctrl_b


def run_matchup(deck_a, deck_b, seed, companion_by_seat=None, max_decisions=None):
    """Drive one bo1 scripted game, checking invariants at every decision.

    Uses runner.drive_game (the single project decision loop) via its per-decision
    on_query hook — no hand-rolled loop. Returns the number of decisions checked.
    """
    companion_by_seat = companion_by_seat or {}
    # bo1 (bo3=False) so the engine never enters the sideboard phase, whose
    # observation is a deliberately masked view of the stale prior board.
    env = RoboMageEnv(deck_a=deck_a, deck_b=deck_b, bo3=False)
    obs, _ = env.reset(seed=seed)
    # Match runner.run_games' RNG seeding so scripted tie-breaks are deterministic.
    random.seed(seed)
    ctrl_a, ctrl_b = _make_scripted_pair(deck_a, deck_b)

    checked = [0]
    deck_block_by_seat = {}

    def on_query(d):
        cats = np.round(d.obs[STATE_SIZE:STATE_SIZE + d.num_choices]
                        * ACTION_CATEGORY_MAX).astype(int)
        is_pregame = decode.is_mulligan(cats) or decode.is_bottom(cats)
        check_decision(d.index, d.obs, d.priority_is_a, companion_by_seat,
                       is_pregame, deck_block_by_seat, num_choices=d.num_choices)
        checked[0] += 1

    try:
        runner.drive_game(env, obs, ctrl_a, ctrl_b, on_query=on_query,
                          max_decisions=max_decisions)
    finally:
        env.close()
    return checked[0]


def _write_yorion80_deck():
    """Write an 80-card temp deck with Yorion in the sideboard so the engine
    declares it as a companion (DeckSizePlus20 => >= 80 main cards; that is the
    only gate the engine enforces — see src/companion.cpp). Basics keep it fully
    in-vocab and the game short. Returns the deck spec (relative to decks/)."""
    stem = "temp/obsinv_yorion80"
    path = os.path.join(_DECKS_DIR, "temp", "obsinv_yorion80.dk")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["80 Plains", "", "SIDEBOARD:", "1 Yorion, Sky Nomad", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return stem


# The matchups: cheap, deterministic, fixed seeds. delver vs mav is the vanilla
# case; bw_dnt vs ur_delver is exile-heavy (Skyclave Apparition, Phelia,
# Solitude, Swords to Plowshares, Wasteland) exercising the exile blocks and the
# per-permanent returnable-exile id; the Yorion probe drives the companion
# invariant with a real declared companion.
_MATCHUPS = [
    ("league/ur_delver", "league/gw_maverick", 1, {}, None),
    ("league/bw_dnt", "league/ur_delver", 7, {}, None),
]


def check_walker_activation_ordinals():
    """Guaranteed coverage for invariant (11): stage a planeswalker with
    multiple loyalty abilities (Jace, the Mind Sculptor) on the battlefield and
    assert its activation menu offers >= 2 same-entity ACTIVATE_ABILITY actions
    with DISTINCT ordinals >= 0 — the scripted matchups above only exercise the
    check when such a menu happens to occur. Auto-passes to the first main
    phase (deterministic); returns the number of loyalty actions verified."""
    env = RoboMageEnv(deck_a="delver", deck_b="delver",
                      battlefield_a="Jace the Mind Sculptor", bo3=False)
    try:
        env.reset(options={"engine_seed": 3})
        for _ in range(120):
            num = env._num_choices
            obs = env._obs
            cats = decode.action_categories(obs, num)
            slots = decode.action_slot_refs(obs, num)
            ords = decode.action_ordinals(obs, num)
            by_slot = {}
            for i in range(num):
                if int(cats[i]) == CAT_ACTIVATE_ABILITY and int(slots[i]) >= 0:
                    by_slot.setdefault(int(slots[i]), []).append(int(ords[i]))
            for slot, olist in by_slot.items():
                if len(olist) >= 2:
                    if any(o < 0 for o in olist):
                        raise InvariantError(
                            f"loyalty activation without ordinal: {olist}")
                    if len(set(olist)) != len(olist):
                        raise InvariantError(
                            f"duplicate loyalty-ability ordinals: {olist}")
                    return len(olist)
            env.step(0)
        raise InvariantError(
            "staged Jace never offered >= 2 loyalty activations in 120 decisions")
    finally:
        env.close()


def main():
    yorion_deck = _write_yorion80_deck()
    matchups = list(_MATCHUPS) + [
        # Seat A declares Yorion (vocab 289); the game is short (A only plays
        # lands), so cap it — a handful of decisions covers the companion check.
        (yorion_deck, "league/ur_delver", 3, {"A": 289}, 80),
    ]

    total = 0
    for deck_a, deck_b, seed, companions, cap in matchups:
        label = f"{deck_a} vs {deck_b} (seed {seed})"
        try:
            n = run_matchup(deck_a, deck_b, seed, companions, cap)
        except InvariantError as e:
            print(f"FAIL  {label}\n  {e}", flush=True)
            return 1
        comp_note = f" [companion: {companions}]" if companions else ""
        print(f"ok    {label}: {n} decisions checked{comp_note}", flush=True)
        total += n

    try:
        n_loyal = check_walker_activation_ordinals()
    except InvariantError as e:
        print(f"FAIL  walker activation ordinals\n  {e}", flush=True)
        return 1
    print(f"ok    walker activation ordinals: {n_loyal} distinct loyalty "
          "activations on one Jace", flush=True)

    print(f"\nobs invariants OK: {total} decisions checked across "
          f"{len(matchups)} games", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
