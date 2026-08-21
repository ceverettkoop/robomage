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
import math
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import decode
import runner
from env import (
    RoboMageEnv, NarrativeEnv, STATE_SIZE, N_CARD_TYPES, N_ENTITY_REF_SLOTS, BIN_DIR,
    ACTION_CATEGORY_MAX, MAX_ACTIONS, OPTION_ORDINAL_MAX,
    ACT_CATS_START, ACT_ORDS_START,
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
    _EXTRAS_MC_ONEHOT_START, _EXTRAS_PLAYS_FIRST, _EXTRAS_SB_SWAPS, _EXTRAS_SB_DELTA,
    _SELF_BLOCK_START, _OPP_BLOCK_START, _OFF_IS_LAND, _OFF_IS_PHASED_OUT,
    _MANA_DEV_START, _MANA_DEV_OPP_START,
    _MD_POTENTIAL_TOTAL, _MD_LANDS_IN_PLAY, _MD_SELF_LANDS_IN_HAND,
    _MD_SELF_LAND_DROPS, _MD_OPP_LAND_DROPS,
    _LOG_VITALS_START, _LOG_VITALS_OPP_START, _LV_LOG_LIFE, _LV_LOG_LIBRARY,
    _PB_LIFE, _PB_HAND_CT, _PB_MANA, _LIBRARY_CTX_START, _REVEALED_START,
    _SELF_LIVE_LIB_START, _SELF_DECK_MAIN_START, _SELF_DECK_SIDE_START,
    _OPP_DECK_MAIN_START, _OPP_DECK_SIDE_START,
    _DECKLIST_SLOT_SIZE, _SELF_IS_A_IDX, _IS_SIDEBOARD_IDX)
from _enums import (N_MANDATORY_CHOICES, DECKLIST_MAIN_SLOTS,
                    DECKLIST_SIDE_SLOTS, CAT_ACTIVATE_ABILITY,
                    CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT, CAT_SIDEBOARD_DONE,
                    SIDEBOARD_SWAP_CAP, MANA_DEV_COLORS, MANA_DEV_SELF_SIZE,
                    MANA_DEV_OPP_SIZE, MANA_COUNT_NORMALIZER,
                    LAND_DROPS_NORMALIZER, LOG_VITALS_PLAYER_SIZE,
                    LIFE_NORMALIZER, LIBRARY_NORMALIZER,
                    LOG_LIFE_DENOM, LOG_LIBRARY_DENOM)
from opponents import make_controller
from scripted_agent import scripted_action

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


# The deck-identity blocks: (name, start offset, slot count).
_DECKLIST_BLOCKS = (
    ("self_live_lib", _SELF_LIVE_LIB_START, DECKLIST_MAIN_SLOTS),
    ("self_deck_main", _SELF_DECK_MAIN_START, DECKLIST_MAIN_SLOTS),
    ("self_deck_side", _SELF_DECK_SIDE_START, DECKLIST_SIDE_SLOTS),
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

def _count_slot_lands(state, start):
    """Lands among one side's serialized permanent slots, phased-out excluded.

    The observation deliberately keeps phased-out permanents visible (with the
    is_phased_out flag set), while the engine's mana-development counts use the
    live-permanent guard — so this mirrors that guard rather than counting every
    is_land slot."""
    n = 0
    for s in range(_PERM_SLOTS):
        base = start + s * _PERM_SLOT_SIZE
        if state[base + _OFF_IS_LAND] > 0.5 and state[base + _OFF_IS_PHASED_OUT] < 0.5:
            n += 1
    return n


def _check_mana_dev(decision_idx, seat, state):
    """Invariant (12): the MANA DEVELOPMENT block (see machine_io.h)."""
    halves = (
        ("self_mana_dev", _MANA_DEV_START, MANA_DEV_SELF_SIZE,
         _MD_SELF_LAND_DROPS, _SELF_BLOCK_START, _SELF_PERM_START,
         _MD_SELF_LANDS_IN_HAND),
        ("opp_mana_dev", _MANA_DEV_OPP_START, MANA_DEV_OPP_SIZE,
         _MD_OPP_LAND_DROPS, _OPP_BLOCK_START, _OPP_PERM_START, None),
    )
    for name, start, width, drops_off, pb_start, perm_start, hand_off in halves:
        block = state[start:start + width]
        # Finite and non-negative (every field is a count; -0.5 tolerates float noise).
        for i, v in enumerate(block):
            if not np.isfinite(v) or v < -0.5 / MANA_COUNT_NORMALIZER:
                _fail(decision_idx, seat, name, i, v,
                      "mana-development float must be finite and non-negative")
        total = float(block[_MD_POTENTIAL_TOTAL]) * MANA_COUNT_NORMALIZER
        # Per-color potential <= the source total (a source counts once per color
        # it can make, and once toward the total).
        for c in range(MANA_DEV_COLORS):
            pot = float(block[c]) * MANA_COUNT_NORMALIZER
            if pot > total + 0.5:
                _fail(decision_idx, seat, f"{name}.potential", c, pot,
                      f"per-color potential {pot} exceeds potential_total {total}")
        # potential_total includes the floating pool, so it can never be smaller.
        pool = float(np.sum(state[pb_start + _PB_MANA:pb_start + _PB_MANA
                                  + MANA_DEV_COLORS])) * MANA_COUNT_NORMALIZER
        if total + 0.5 < pool:
            _fail(decision_idx, seat, f"{name}.potential_total", "-", total,
                  f"potential_total {total} < floating pool {pool} (it must include it)")
        # Land drops fit the normalizer's range: 0 .. LAND_DROPS_NORMALIZER. (If a
        # future card can grant more, the NORMALIZER is what must grow — this bound
        # follows it rather than restating a literal.)
        drops = float(block[drops_off]) * LAND_DROPS_NORMALIZER
        if not (-0.5 <= drops <= LAND_DROPS_NORMALIZER + 0.5):
            _fail(decision_idx, seat, f"{name}.land_drops_remaining", "-", drops,
                  f"land drops {drops} outside [0,{LAND_DROPS_NORMALIZER}]")
        # lands_in_play == the lands visible in this side's permanent slots.
        lands = int(round(float(block[_MD_LANDS_IN_PLAY]) * MANA_COUNT_NORMALIZER))
        slot_lands = _count_slot_lands(state, perm_start)
        if lands != slot_lands:
            _fail(decision_idx, seat, f"{name}.lands_in_play", "-", lands,
                  f"lands_in_play {lands} != {slot_lands} lands counted over this "
                  "side's permanent slots (is_land & !is_phased_out)")
        # lands_in_hand (self only) can never exceed the hand size.
        if hand_off is not None:
            in_hand = float(block[hand_off]) * MANA_COUNT_NORMALIZER
            hand_ct = float(state[pb_start + _PB_HAND_CT]) * 10.0
            if in_hand > hand_ct + 0.5:
                _fail(decision_idx, seat, f"{name}.lands_in_hand", "-", in_hand,
                      f"lands_in_hand {in_hand} exceeds hand count {hand_ct}")


# Tolerance for the log-vitals identity. The engine narrows a double log1p ratio to
# float32 (~1e-7 relative) and the count is recovered exactly from the linear float,
# so anything beyond this is a real disagreement, not arithmetic noise.
_LOG_VITALS_TOL = 1e-5


def _check_log_vitals(decision_idx, seat, state):
    """Invariant (13): the LOG VITALS block is exactly the log1p re-warping of the
    SAME observation's LINEAR life and library floats.

    The strong form of the check: the integer life total and library size are
    recovered from the linear encodings that already exist in this obs (life =
    round(f * LIFE_NORMALIZER), library = round(f * LIBRARY_NORMALIZER)), and the
    log floats must equal log1p(max(v, 0)) / the mirrored denominator. So the two
    encodings can never describe different numbers — a serializer that read the
    wrong player's life, ordered the halves backwards, or forgot to clamp a negative
    life into log1p's domain fails here rather than silently training on it.

    During the bo3 sideboard phase the block is MASKED (env._build_sideboard_mask /
    obs_builder.cpp's twin) because the player blocks it mirrors are masked, so the
    identity deliberately does not hold there; the masked all-zeros form is asserted
    instead — which is itself the check that the mask covers the new block."""
    in_sideboard = float(state[_IS_SIDEBOARD_IDX]) > 0.5
    halves = (
        ("self_log_vitals", _LOG_VITALS_START, _SELF_BLOCK_START, _LIBRARY_CTX_START),
        ("opp_log_vitals", _LOG_VITALS_OPP_START, _OPP_BLOCK_START, _LIBRARY_CTX_START + 1),
    )
    for name, start, pb_start, lib_off in halves:
        block = state[start:start + LOG_VITALS_PLAYER_SIZE]
        # Finite and non-negative: log1p over a clamped count can never be either.
        for i, v in enumerate(block):
            if not np.isfinite(v) or v < -_LOG_VITALS_TOL:
                _fail(decision_idx, seat, name, i, v,
                      "log-scaled vital must be finite and non-negative")
        if in_sideboard:
            for i, v in enumerate(block):
                if float(v) != 0.0:
                    _fail(decision_idx, seat, name, i, v,
                          "log-scaled vital must be masked to 0.0 during the "
                          "sideboard phase (it mirrors the masked player blocks)")
            continue
        # Recover the counts from THIS obs's linear floats, then re-derive the logs.
        life = int(round(float(state[pb_start + _PB_LIFE]) * LIFE_NORMALIZER))
        library = int(round(float(state[lib_off]) * LIBRARY_NORMALIZER))
        for off, field, count, denom in (
                (_LV_LOG_LIFE, "log_life", life, LOG_LIFE_DENOM),
                (_LV_LOG_LIBRARY, "log_library", library, LOG_LIBRARY_DENOM)):
            expected = math.log1p(max(count, 0)) / denom
            got = float(block[off])
            if abs(got - expected) > _LOG_VITALS_TOL:
                _fail(decision_idx, seat, f"{name}.{field}", off, got,
                      f"log float {got} != log1p(max({count},0))/{denom} = {expected} "
                      f"(count recovered from this obs's linear float)")


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
        life = float(state[base + _PB_LIFE]) * LIFE_NORMALIZER
        hand = float(state[base + _PB_HAND_CT]) * 10.0
        for field, val in (("life", life), ("hand_count", hand)):
            if not np.isfinite(val) or val < -0.5:  # -0.5: tolerate float rounding at 0
                _fail(decision_idx, seat, f"{label}_player.{field}", "-", val,
                      f"{field} de-normalizes to {val} (expected finite & >= 0)")
    for label, off in (("self", _LIBRARY_CTX_START), ("opp", _LIBRARY_CTX_START + 1)):
        lib = float(state[off]) * LIBRARY_NORMALIZER
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
    lib_ct = int(round(float(state[_LIBRARY_CTX_START]) * LIBRARY_NORMALIZER))
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

    # (12) Mana-development block: every float is a finite non-negative count, the
    # per-color potentials are bounded by the source total, the total covers the
    # floating pool it includes, land drops stay inside the normalizer's range, and
    # lands_in_play agrees with the SAME observation's permanent slots (the strong
    # cross-check: two independent derivations of "how many lands does this player
    # have" — the engine's live-permanent scan vs. the serialized per-slot is_land
    # flags — must match, with phased-out permanents excluded from BOTH per CR
    # 702.26e). lands_in_hand is self-only and bounded by the hand count.
    _check_mana_dev(decision_idx, seat, state)

    # (13) Log-scaled vitals: each log float is exactly log1p(max(count,0)) over the
    # mirrored denominator, where the count is recovered from the SAME obs's LINEAR
    # life/library float — the two encodings of one number, pinned to each other.
    # (Masked to zeros during the sideboard phase, which is asserted instead there.)
    _check_log_vitals(decision_idx, seat, state)

    # (10) Every per-action option_ordinal float round-trips into
    # [-1, OPTION_ORDINAL_MAX]. The ords block is the 6th (last) action-metadata
    # block, normalized (ord + 1) / (OPTION_ORDINAL_MAX + 1) so -1 (n/a) -> 0.0.
    _ords_start = ACT_ORDS_START
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
        cats = np.round(d.obs[ACT_CATS_START:ACT_CATS_START + d.num_choices]
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


# Bo3 sideboard-check tuning: swaps each seat makes per sideboard phase, how many
# post-board decisions are enough to prove the freeze held into game 2, and a hard
# decision cap so a pathological match can never hang the tier.
_SB_SWAPS_PER_PHASE = 2
_SB_POST_BOARD_MIN = 40
_SB_MAX_DECISIONS = 4000
# The bo3 matchup the sideboard checks drive: both decks carry a real sideboard,
# and the seed reaches a second game quickly.
_SB_DECK_A, _SB_DECK_B, _SB_SEED = "league/ur_delver", "league/gw_maverick", 3
# Seed for the stranding matchup the forced-out check drives (see
# _write_strand_decks): a beatdown deck vs a lands-only one, so game 1 ends
# quickly and the sideboard phase is reached.
_SB_STRAND_SEED = 7


def _decklist_diff(before, after):
    """Short human-readable diff of two decoded decklist blocks."""
    parts = []
    for slot, (b, a) in enumerate(zip(before, after)):
        if b != a:
            parts.append(f"slot {slot}: (id={b[0]}, ct={b[1]}) -> (id={a[0]}, ct={a[1]})")
        if len(parts) >= 4:
            parts.append("...")
            break
    return "; ".join(parts) or "(no slot differs — length mismatch)"


def _sideboard_action(cats, swaps_left):
    """Pick a move on the IN-FIRST sideboard menu. Returns
    (action, completed_a_swap), or None when this is not a sideboard menu.

    The scripted agent always answers Done (scripted_agent.py), so the bo3 sideboard
    checks drive the swaps themselves — otherwise nothing would ever mutate a deck
    and their assertions would pass vacuously.

    A Done action means the deck is balanced: open a swap with the first "+1" while
    budget remains, else finish. With a cut outstanding, Done is absent and only
    cuts are offered — take the first, which ALWAYS completes the pair (the engine
    credits even the forced cut of a stranded addition as a swap), so the swap
    counting in check_opponent_decklist_frozen stays exact.
    """
    done_idx = next((i for i, c in enumerate(cats) if c == CAT_SIDEBOARD_DONE), None)
    moves = [i for i, c in enumerate(cats)
             if c in (CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT)]
    if done_idx is None and not moves:
        return None

    if done_idx is not None:                       # balanced
        adds = [i for i in moves if cats[i] == CAT_SIDEBOARD_IN]
        if swaps_left > 0 and adds:
            return adds[0], False                  # opening half — always an add
        return done_idx, False

    return moves[0], True                          # the cut completes the pair


def _drive_bo3_sideboarding(env, seed=_SB_SEED, swaps_per_seat=_SB_SWAPS_PER_PHASE,
                            max_decisions=_SB_MAX_DECISIONS):
    """Yield (obs, num_choices, cats, seat) at every decision of a bo3 in which
    BOTH seats really sideboard.

    Shared by the bo3 sideboard checks below: each drives the same match and only
    differs in what it asserts per decision. The caller owns the env (and its
    close); breaking out of the loop early is fine.
    """
    obs, _ = env.reset(seed=seed)
    swaps_left = {"A": swaps_per_seat, "B": swaps_per_seat}
    for _ in range(max_decisions):
        num = env._num_choices
        seat = "A" if obs[_SELF_IS_A_IDX] > 0.5 else "B"
        cats = decode.action_categories(obs, num)
        yield obs, num, cats, seat

        picked = _sideboard_action(cats, swaps_left[seat])
        if picked is None:
            action = scripted_action(obs, num)
        else:
            action, completed = picked
            if completed:
                swaps_left[seat] -= 1
        obs, _r, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            return


def check_opponent_decklist_frozen():
    """The opponent decklist blocks must be the match's REGISTERED 75, FROZEN for
    the whole bo3 — a sideboard swap must never leak into the view the other seat
    sees (see src/classes/deck_state.h).

    Drives a real bo3 in which BOTH seats actually sideboard, and asserts that for
    each viewer seat the decoded opp_deck_main / opp_deck_side blocks are byte-
    identical at every decision of every game, including the between-games
    sideboard phases (where env's sideboard mask deliberately keeps them visible).
    Returns (swaps_made, post_board_decisions)."""
    env = RoboMageEnv(deck_a=_SB_DECK_A, deck_b=_SB_DECK_B, bo3=True,
                      auto_sideboard=False)
    first = {}                        # seat -> (main_block, side_block) at first sight
    swaps = 0
    saw_sideboard = False
    post_board = 0
    # Log-vitals coverage over this bo3: the main invariant loop is bo1 only, so
    # this is where the block's SIDEBOARD-phase form (masked to zeros) is exercised
    # alongside its normal form (the log identity). Counted so neither branch can
    # pass vacuously.
    lv_live, lv_masked = 0, 0
    try:
        for idx, (obs, _num, cats, seat) in enumerate(_drive_bo3_sideboarding(env)):
            _check_log_vitals(idx, seat, obs[:STATE_SIZE])
            if obs[_IS_SIDEBOARD_IDX] > 0.5:
                lv_masked += 1
            else:
                lv_live += 1
            blocks = (
                tuple(_decode_decklist_block(obs, _OPP_DECK_MAIN_START,
                                             DECKLIST_MAIN_SLOTS)),
                tuple(_decode_decklist_block(obs, _OPP_DECK_SIDE_START,
                                             DECKLIST_SIDE_SLOTS)),
            )
            if seat not in first:
                first[seat] = blocks
            elif blocks != first[seat]:
                which = "maindeck" if blocks[0] != first[seat][0] else "sideboard"
                i = 0 if which == "maindeck" else 1
                raise InvariantError(
                    f"opponent {which} block changed mid-match for viewer seat "
                    f"{seat} (it must stay the registered 75): "
                    f"{_decklist_diff(first[seat][i], blocks[i])}")

            # An unbalanced menu (sideboard moves offered but no Done) is answered
            # with the pairing move by _sideboard_action, so each one is exactly one
            # completed swap.
            if any(c in (CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT) for c in cats) and \
                    not any(c == CAT_SIDEBOARD_DONE for c in cats):
                swaps += 1
            if obs[_IS_SIDEBOARD_IDX] > 0.5:
                saw_sideboard = True
            elif saw_sideboard:
                post_board += 1
                if post_board >= _SB_POST_BOARD_MIN:
                    break
    finally:
        env.close()

    # Guard against a vacuous pass: without real swaps, and without decisions in a
    # post-board game, the assertion above proves nothing.
    if swaps == 0:
        raise InvariantError("no sideboard swap was made — the frozen-block "
                             "assertion would pass vacuously")
    if post_board == 0:
        raise InvariantError("the match never reached a post-board game — the "
                             "frozen-block assertion would pass vacuously")
    if lv_live == 0 or lv_masked == 0:
        raise InvariantError(
            f"log-vitals coverage was vacuous: {lv_live} live decisions, "
            f"{lv_masked} sideboard-masked ones (both branches must be seen)")
    return swaps, post_board, lv_live, lv_masked


# "Sideboard in: 4x Lightning Bolt" / "Sideboard out: 1x Island" — the count is the
# ground truth the per-action option_ordinal must reproduce.
_SB_DESC_COUNT_RE = re.compile(r"^Sideboard (?:in|out): (\d+)x ")


def check_sideboard_copy_ordinals():
    """Every sideboard IN/OUT action must carry its COPY COUNT as option_ordinal.

    Without it the policy sees "4x Lightning Bolt" and "1x Island" as the same
    action — category + card id are otherwise the only non-null per-action features
    at a sideboard root — so cutting one of four is indistinguishable from cutting a
    singleton. Cross-checks the decoded ordinal against the count parsed out of the
    engine's own action label (NarrativeEnv, which carries the descriptions), so
    this is an exact check rather than a range assertion. Returns (actions_checked,
    distinct_ordinals_seen)."""
    env = NarrativeEnv(deck_a=_SB_DECK_A, deck_b=_SB_DECK_B, bo3=True,
                       auto_sideboard=False)
    checked = 0
    seen_ordinals = set()
    try:
        for obs, num, cats, _seat in _drive_bo3_sideboarding(env):
            if not any(c in (CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT) for c in cats):
                continue
            ords = decode.action_ordinals(obs, num)
            descs = env._action_descriptions or []
            for i in range(num):
                if cats[i] not in (CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT):
                    continue
                desc = descs[i] if i < len(descs) else ""
                m = _SB_DESC_COUNT_RE.match(desc)
                if not m:
                    raise InvariantError(
                        f"sideboard action {i} has an unparseable label {desc!r} — "
                        "the ordinal cross-check cannot verify it")
                want = min(int(m.group(1)), OPTION_ORDINAL_MAX)
                got = int(ords[i])
                if got != want:
                    raise InvariantError(
                        f"sideboard action {i} ({desc!r}) has option_ordinal {got}, "
                        f"expected the copy count {want}")
                seen_ordinals.add(got)
                checked += 1
            # One phase's menus are plenty; stop as soon as we have both a
            # multi-copy and a single-copy entry to prove the ordinal really varies.
            if checked and len(seen_ordinals) >= 2:
                break
    finally:
        env.close()

    if checked == 0:
        raise InvariantError("no sideboard IN/OUT action was ever offered — the "
                             "ordinal assertion would pass vacuously")
    if len(seen_ordinals) < 2:
        raise InvariantError(
            f"every sideboard action carried the same ordinal {seen_ordinals} — a "
            "constant would satisfy the per-action check without encoding anything")
    return checked, len(seen_ordinals)


def _decode_sb_delta(obs):
    """Maindeck drift from its phase-start size, from the serialized (d + 1) / 2."""
    return int(round(float(obs[_EXTRAS_SB_DELTA]) * 2)) - 1


def _decode_sb_swaps(obs):
    """Swaps completed so far this phase, from the serialized count / cap."""
    return int(round(float(obs[_EXTRAS_SB_SWAPS]) * SIDEBOARD_SWAP_CAP))


def check_sideboard_delta_menu():
    """Deck balance must be an ACTION-MASK invariant on the IN-FIRST menu.

    A balanced sideboard decision offers Done plus every distinct sideboard card as
    a "+1"; each of those is answered by a menu of the distinct maindeck cards as
    the balancing "-1". A swap therefore always opens with its addition, and the
    engine must never let the model express an off-size deck:

      (i)   the serialized drift is always 0 or +1 (the -1 pole is unreachable —
            no decision ever offers a leading cut);
      (ii)  balanced (drift 0) => Done is offered AT INDEX 0 (the "first choice"
            convention every auto-driver relies on) and NO cut is offered;
      (iii) drift +1 => Done is WITHHELD, no further addition is offered, and at
            least one cut is (so the deck can never be stuck off-size);
      (iv)  a phase never exceeds the swap cap;
      (v)   at least one unbalanced decision actually occurred, so the balancing
            mask is exercised rather than assumed.

    Returns (decisions_checked, unbalanced_decisions)."""
    env = RoboMageEnv(deck_a=_SB_DECK_A, deck_b=_SB_DECK_B, bo3=True,
                      auto_sideboard=False)
    checked = 0
    unbalanced = 0
    max_swaps_seen = 0
    try:
        for obs, _num, cats, seat in _drive_bo3_sideboarding(env):
            if obs[_IS_SIDEBOARD_IDX] <= 0.5:
                continue
            checked += 1
            drift = _decode_sb_delta(obs)
            has_done = any(c == CAT_SIDEBOARD_DONE for c in cats)
            has_add = any(c == CAT_SIDEBOARD_IN for c in cats)
            has_cut = any(c == CAT_SIDEBOARD_OUT for c in cats)

            # (i)
            if drift not in (0, 1):
                raise InvariantError(
                    f"seat {seat}: sideboard drift {drift} outside {{0,+1}} — the "
                    "menu is in-first, so a swap can only ever oversize the deck")

            if drift == 0:
                # (ii) — a balanced menu is the only place Done may appear, it must
                # lead the menu, and it must never offer a leading cut.
                if not has_done:
                    raise InvariantError(
                        f"seat {seat}: balanced deck but Done is not offered — the "
                        "phase could never end")
                if cats[0] != CAT_SIDEBOARD_DONE:
                    raise InvariantError(
                        f"seat {seat}: balanced menu has category {cats[0]} at index "
                        "0, not Done — auto-drivers take index 0 and would make an "
                        "arbitrary swap instead of finishing")
                if has_cut:
                    raise InvariantError(
                        f"seat {seat}: balanced menu offers a cut; a swap must open "
                        "with its addition, so only additions and Done are legal")
            else:
                # (iii) — with an addition outstanding the mask must force it balanced.
                unbalanced += 1
                if has_done:
                    raise InvariantError(
                        f"seat {seat}: Done offered at drift {drift:+d} — the model "
                        "could submit an off-size deck")
                if has_add:
                    raise InvariantError(
                        f"seat {seat}: deck is +1 but an addition is still offered; "
                        "only cuts may balance it")
                if not has_cut:
                    raise InvariantError(
                        f"seat {seat}: deck is +1 and nothing is offered to balance "
                        "it — the stranded case must still offer its forced cut")

            # (iv) swaps completed so far, from the serialized counter.
            max_swaps_seen = max(max_swaps_seen, _decode_sb_swaps(obs))
    finally:
        env.close()

    if checked == 0:
        raise InvariantError("no sideboard decision occurred — these assertions "
                             "would pass vacuously")
    # (v)
    if unbalanced == 0:
        raise InvariantError("no unbalanced decision occurred — the balancing mask "
                             "(the whole point of the in-first menu) was never tested")
    if max_swaps_seen > SIDEBOARD_SWAP_CAP:
        raise InvariantError(
            f"a phase completed {max_swaps_seen} swaps, over the "
            f"{SIDEBOARD_SWAP_CAP} cap")
    return checked, unbalanced


def _drive_to_sideboard(env, seed, max_decisions=_SB_MAX_DECISIONS):
    """Play the match scripted from a fresh reset until the first sideboard
    prompt, and return that decision's observation."""
    obs, _ = env.reset(seed=seed)
    for _ in range(max_decisions):
        if obs[_IS_SIDEBOARD_IDX] > 0.5:
            return obs
        obs, _r, term, trunc, _i = env.step(scripted_action(obs, env._num_choices))
        if term or trunc:
            break
    raise InvariantError("the match ended before any sideboard prompt — the "
                         "sideboard assertions would pass vacuously")


def _write_strand_decks():
    """Write the two temp decks the forced-out check needs.

    Seat A has NO sideboard at all, so its balanced menu must be exactly [Done].
    Seat B's ONE sideboard card shares its name with its whole maindeck, so siding
    it in STRANDS the deck at +1: every maindeck name is then locked in and no
    genuine cut is left, which is the only case in which the engine offers the
    forced cut (see run_sideboard_phase in src/game_driver.cpp). Basics and bears
    keep both decks in-vocab and game 1 short. Returns the two deck specs
    (relative to decks/)."""
    specs = []
    for stem, lines in (("obsinv_strand_a", ["36 Grizzly Bears", "24 Forest"]),
                        ("obsinv_strand_b", ["60 Swamp", "SIDEBOARD:", "1 Swamp"])):
        path = os.path.join(_DECKS_DIR, "temp", stem + ".dk")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        specs.append("temp/" + stem)
    return specs


def check_sideboard_forced_out():
    """An outstanding addition must never leave the deck stuck off-size, and the
    action that guarantees that must not be an always-available no-op.

    The IN-FIRST menu answers each addition with the distinct maindeck cards, minus
    the ones already sided in. When that leaves NOTHING (the stranded case) the
    engine offers exactly ONE forced cut of an arbitrary maindeck card, ignoring the
    lock: it is credited as a completed swap and force-ends the phase, so the deck
    can neither stay off-size nor spin. Asserts:

      (i)   with a real sideboard, the balanced menu leads with Done and offers no
            cut; the follow-up to an addition is all cuts, withholds Done, and
            excludes the card just brought in (the one-shot lock), while the
            pending-decision source names it; completing it credits one swap and
            returns a Done-led balanced menu;
      (ii)  a deck with NO sideboard gets a balanced menu of exactly [Done];
      (iii) when siding in the only sideboard card strands the deck, the menu is
            exactly the lone forced cut (the lock notwithstanding);
      (iv)  taking it restores drift 0, CREDITS a swap, leaves the deck exactly as
            the phase started, and the next menu is exactly [Done] — which ends the
            phase.

    Returns the vocab id of the card the forced cut removed."""
    # ── (i) a genuine completion exists => in-first pairing, with the lock ────
    env = RoboMageEnv(deck_a=_SB_DECK_A, deck_b=_SB_DECK_B, bo3=True,
                      auto_sideboard=False)
    try:
        obs = _drive_to_sideboard(env, _SB_SEED)
        num = env._num_choices
        cats = decode.action_categories(obs, num)
        ids = decode.action_card_ids(obs)
        swaps_before = _decode_sb_swaps(obs)
        if cats[0] != CAT_SIDEBOARD_DONE:
            raise InvariantError(
                f"balanced sideboard menu leads with category {cats[0]}, not Done")
        if any(cats[i] == CAT_SIDEBOARD_OUT for i in range(num)):
            raise InvariantError(
                "balanced sideboard menu offers a cut — a swap must open with its "
                "addition")
        adds = [i for i in range(num) if cats[i] == CAT_SIDEBOARD_IN]
        if not adds:
            raise InvariantError("balanced sideboard menu offered no addition to "
                                 "open a swap with")
        add_id = _decode_card_id(ids[adds[0]])
        obs, _r, _term, _trunc, _i = env.step(adds[0])
        num = env._num_choices
        cats = decode.action_categories(obs, num)
        ids = decode.action_card_ids(obs)
        if _decode_sb_delta(obs) != 1:
            raise InvariantError(
                f"siding a card in left drift {_decode_sb_delta(obs):+d}, not +1")
        pending = _decode_card_id(obs[_PENDING_DECISION_START])
        if pending != add_id:
            raise InvariantError(
                f"pending-decision source is card {pending}, not the outstanding "
                f"card {add_id} — the observation does not say what to balance")
        if any(cats[i] != CAT_SIDEBOARD_OUT for i in range(num)):
            raise InvariantError(
                f"the follow-up menu is not all cuts (cats={cats.tolist()}); Done "
                "and further additions must both be withheld at +1")
        if any(_decode_card_id(ids[i]) == add_id for i in range(num)):
            raise InvariantError(
                f"card {add_id} was sided in and is offered straight back out — the "
                "one-shot lock must exclude it, or in/out could oscillate forever")
        obs, _r, _term, _trunc, _i = env.step(0)
        num = env._num_choices
        cats = decode.action_categories(obs, num)
        if _decode_sb_delta(obs) != 0 or cats[0] != CAT_SIDEBOARD_DONE:
            raise InvariantError(
                f"completing the swap left drift {_decode_sb_delta(obs):+d} with "
                f"category {cats[0]} at index 0; it must return a Done-led balanced "
                "menu")
        if _decode_sb_swaps(obs) != swaps_before + 1:
            raise InvariantError(
                f"completing the pair credited {_decode_sb_swaps(obs) - swaps_before} "
                "swap(s), expected exactly 1")
    finally:
        env.close()

    # ── (ii)-(iv) stranded => the lone forced cut, then a Done-only menu ──────
    deck_a, deck_b = _write_strand_decks()
    env = RoboMageEnv(deck_a=deck_a, deck_b=deck_b, bo3=True,
                      auto_sideboard=False)
    try:
        # (ii) seat A boards first and has no sideboard at all.
        obs = _drive_to_sideboard(env, _SB_STRAND_SEED)
        num = env._num_choices
        cats = decode.action_categories(obs, num)
        if not obs[_SELF_IS_A_IDX] > 0.5:
            raise InvariantError("the first sideboard prompt is not seat A's — the "
                                 "stranded fixture assumes A boards first")
        if num != 1 or cats[0] != CAT_SIDEBOARD_DONE:
            raise InvariantError(
                f"a deck with no sideboard must get a balanced menu of exactly "
                f"[Done]; got {num} choice(s) cats={cats.tolist()}")
        obs, _r, _term, _trunc, _i = env.step(0)

        # Seat B: one sideboard card, sharing its name with the whole maindeck.
        num = env._num_choices
        cats = decode.action_categories(obs, num)
        ids = decode.action_card_ids(obs)
        if obs[_IS_SIDEBOARD_IDX] <= 0.5 or obs[_SELF_IS_A_IDX] > 0.5:
            raise InvariantError("seat B never got its own sideboard prompt")
        deck_at_start = (
            tuple(_decode_decklist_block(obs, _SELF_DECK_MAIN_START,
                                         DECKLIST_MAIN_SLOTS)),
            tuple(_decode_decklist_block(obs, _SELF_DECK_SIDE_START,
                                         DECKLIST_SIDE_SLOTS)),
        )
        swaps_before = _decode_sb_swaps(obs)
        adds = [i for i in range(num) if cats[i] == CAT_SIDEBOARD_IN]
        if not adds:
            raise InvariantError("seat B's balanced menu offered no addition")
        add_id = _decode_card_id(ids[adds[0]])
        obs, _r, _term, _trunc, _i = env.step(adds[0])
        num = env._num_choices
        cats = decode.action_categories(obs, num)
        ids = decode.action_card_ids(obs)
        # (iii) every maindeck name is now locked in, so only the forced cut is left
        # — and it is the locked name itself, which the lock deliberately ignores.
        if num != 1 or cats[0] != CAT_SIDEBOARD_OUT or \
                _decode_card_id(ids[0]) != add_id:
            raise InvariantError(
                f"a stranded addition of card {add_id} must offer exactly one forced "
                f"cut of it; got {num} choice(s) cats={cats.tolist()} "
                f"ids={[_decode_card_id(ids[i]) for i in range(num)]}")
        forced_id = _decode_card_id(ids[0])
        obs, _r, _term, _trunc, _i = env.step(0)
        num = env._num_choices
        cats = decode.action_categories(obs, num)
        # (iv)
        drift = _decode_sb_delta(obs)
        if drift != 0:
            raise InvariantError(
                f"drift is {drift:+d} after the forced cut; it must restore balance")
        swaps = _decode_sb_swaps(obs)
        if swaps != swaps_before + 1:
            raise InvariantError(
                f"the forced cut credited {swaps - swaps_before} swap(s) "
                f"({swaps_before} -> {swaps}); it completes the pair, so it must "
                "count as exactly one")
        deck_now = (
            tuple(_decode_decklist_block(obs, _SELF_DECK_MAIN_START,
                                         DECKLIST_MAIN_SLOTS)),
            tuple(_decode_decklist_block(obs, _SELF_DECK_SIDE_START,
                                         DECKLIST_SIDE_SLOTS)),
        )
        if deck_now != deck_at_start:
            raise InvariantError(
                "cutting the very card just sided in must leave the 75 exactly as "
                f"the phase started: {_decklist_diff(deck_at_start[0], deck_now[0])}")
        if num != 1 or cats[0] != CAT_SIDEBOARD_DONE:
            raise InvariantError(
                f"after a forced cut the next menu must be exactly [Done] (the phase "
                f"force-ends); got {num} choice(s) cats={cats.tolist()}")
        obs, _r, _term, _trunc, _i = env.step(0)
        if obs[_IS_SIDEBOARD_IDX] > 0.5:
            raise InvariantError("taking the forced Done did not end the sideboard "
                                 "phase")
        return forced_id
    finally:
        env.close()


def _block_total(obs, start, n_slots):
    """Total card count across a decoded decklist block (empty slots contribute 0)."""
    return sum(ct for cid, ct in _decode_decklist_block(obs, start, n_slots) if cid >= 0)


def check_sideboard_self_context():
    """The sideboarding player must be able to see its OWN deck and the game it is
    boarding for.

    Before these blocks existed the phase observation showed neither: the self
    live-library block is stale between games and the sideboard mask zeroes it, so
    the player was configuring a 75 it could not observe. Asserts, across a real
    bo3 with both seats sideboarding:

      (i)   self_deck_main / self_deck_side are populated at every decision;
      (ii)  the maindeck TOTAL is invariant across the whole match — swaps are 1:1,
            so the deck can never change size;
      (iii) the self-deck blocks actually MOVE during a sideboard phase (they are
            the live view; the complement of the frozen opponent blocks);
      (iv)  at a sideboard root the match context describes the UPCOMING game
            (game_number advanced, is_post_board set), not the one that just ended;
      (v)   the two seats disagree about self_plays_first at the same boundary —
            exactly one of them is on the play next.

    Returns (sideboard_decisions, self_deck_changes)."""
    env = RoboMageEnv(deck_a=_SB_DECK_A, deck_b=_SB_DECK_B, bo3=True,
                      auto_sideboard=False)
    main_total = None
    last_self_deck = {}                  # seat -> (main_block, side_block)
    changes = 0
    sb_decisions = 0
    plays_first_at_boundary = {}         # seat -> bool, first sideboard phase only
    try:
        for obs, _num, _cats, seat in _drive_bo3_sideboarding(env):
            main_block = tuple(_decode_decklist_block(obs, _SELF_DECK_MAIN_START,
                                                      DECKLIST_MAIN_SLOTS))
            side_block = tuple(_decode_decklist_block(obs, _SELF_DECK_SIDE_START,
                                                      DECKLIST_SIDE_SLOTS))
            # (i) populated — a card id >= 0 in the first slot means real content.
            if main_block[0][0] < 0:
                raise InvariantError(
                    f"seat {seat}: self_deck_main is empty — the sideboarding "
                    "player cannot see its own maindeck")
            if side_block[0][0] < 0:
                raise InvariantError(
                    f"seat {seat}: self_deck_side is empty — the sideboarding "
                    "player cannot see its own sideboard")

            # (ii) The maindeck size is its balanced baseline plus the SERIALIZED
            # drift — so this cross-validates sb_delta against the actual block
            # contents. Swaps are 1:1, so the only legal departures from the
            # baseline are the +/-1 of a move awaiting its pair.
            drift = _decode_sb_delta(obs)
            total = _block_total(obs, _SELF_DECK_MAIN_START, DECKLIST_MAIN_SLOTS)
            if main_total is None:
                if drift != 0:
                    raise InvariantError(
                        f"seat {seat}: first observation already reports drift "
                        f"{drift:+d}; a phase must begin balanced")
                main_total = total
            elif total != main_total + drift:
                raise InvariantError(
                    f"seat {seat}: maindeck total {total} but expected "
                    f"{main_total} {drift:+d} = {main_total + drift}; the "
                    "serialized sideboard delta must match the deck contents")

            # (iii) the live view moves as swaps land.
            if seat in last_self_deck and last_self_deck[seat] != (main_block, side_block):
                changes += 1
            last_self_deck[seat] = (main_block, side_block)

            if obs[_IS_SIDEBOARD_IDX] > 0.5:
                sb_decisions += 1
                match = decode._decode_match_context(obs[:STATE_SIZE])
                # (iv) boarding for game 2 means game_number 1 (0-based) and a
                # post-board game ahead. Reading the ended game would give 0/False.
                if match["game_number"] <= 0 or not match["is_post_board"]:
                    raise InvariantError(
                        f"seat {seat}: sideboard root reports game_number "
                        f"{match['game_number']} / is_post_board "
                        f"{match['is_post_board']} — it must describe the UPCOMING "
                        "game, not the one that just ended")
                plays_first_at_boundary.setdefault(
                    seat, bool(obs[_EXTRAS_PLAYS_FIRST] > 0.5))
    finally:
        env.close()

    if sb_decisions == 0:
        raise InvariantError("no sideboard decision occurred — these assertions "
                             "would pass vacuously")
    if changes == 0:
        raise InvariantError("the self-deck blocks never changed — they are meant "
                             "to be the LIVE view and must track each swap")
    # (v) exactly one seat is on the play in the upcoming game.
    if len(plays_first_at_boundary) == 2 and \
            plays_first_at_boundary["A"] == plays_first_at_boundary["B"]:
        raise InvariantError(
            f"both seats report self_plays_first="
            f"{plays_first_at_boundary['A']} for the upcoming game; exactly one "
            "of them starts it")
    return sb_decisions, changes


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

    try:
        n_swaps, n_post, lv_live, lv_masked = check_opponent_decklist_frozen()
    except InvariantError as e:
        print(f"FAIL  opponent decklist frozen across bo3\n  {e}", flush=True)
        return 1
    print(f"ok    opponent decklist frozen across bo3: {n_swaps} swaps made, "
          f"{n_post} post-board decisions checked", flush=True)
    print(f"ok    log vitals across bo3: {lv_live} live decisions match the log "
          f"identity, {lv_masked} sideboard decisions masked to zeros", flush=True)

    try:
        n_acts, n_ords = check_sideboard_copy_ordinals()
    except InvariantError as e:
        print(f"FAIL  sideboard copy-count ordinals\n  {e}", flush=True)
        return 1
    print(f"ok    sideboard copy-count ordinals: {n_acts} actions checked, "
          f"{n_ords} distinct counts", flush=True)

    try:
        n_sb, n_chg = check_sideboard_self_context()
    except InvariantError as e:
        print(f"FAIL  sideboard self-deck context\n  {e}", flush=True)
        return 1
    print(f"ok    sideboard self-deck context: {n_sb} sideboard decisions, "
          f"{n_chg} live self-deck updates", flush=True)

    try:
        n_dec, n_unbal = check_sideboard_delta_menu()
    except InvariantError as e:
        print(f"FAIL  sideboard delta-menu balance\n  {e}", flush=True)
        return 1
    print(f"ok    sideboard delta-menu balance: {n_dec} decisions, {n_unbal} "
          f"unbalanced (mask-forced)", flush=True)

    try:
        fo_id = check_sideboard_forced_out()
    except InvariantError as e:
        print(f"FAIL  sideboard forced out\n  {e}", flush=True)
        return 1
    print(f"ok    sideboard forced out: in-first pairing with the lock held; card "
          f"{fo_id} stranded, force-cut as a swap, then Done-only", flush=True)

    print(f"\nobs invariants OK: {total} decisions checked across "
          f"{len(matchups)} games", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
