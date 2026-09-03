"""Rule-based "scripted" opponent agent.

This module owns the logic that drives the scripted agent the RL models train
against.  It was extracted verbatim from ``env.py`` (Phase 1 of the scripted-agent
refactor) so the behaviour is byte-identical; later phases add skill levels and
heuristic ("minimax-flavored") play on top of the GREEDY baseline kept here.

IMPORTANT — no multi-ply lookahead: Python cannot fork/snapshot/restore the C++
game state, so the agent reasons strictly single-ply.  Combat decisions are the
one place exact local lookahead is cheap (attacker/blocker P/T arithmetic) and
that is the only "simulation" performed.  Do not add rollout/minimax search that
assumes hypothetical engine queries — the engine does not expose them.

The decision functions are STATELESS pure functions of the (perspective-relative)
observation vector, because they run inside parallel vec-env subprocesses and
diag/watch determinism depends on it. The only exceptions are small per-game,
per-agent sets that are themselves deterministic functions of the trajectory
(EXPLORE's novelty set and the GREEDY/HEURISTIC fruitless-activation stall
guard — see ``ScriptedAgent``).

Layout/vocab constants are owned by ``env.py`` (they are shared engine layout used
by env's reward shaping and observation mirroring) and imported here.
"""

import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from env import (
    # sizes / normalisers
    STATE_SIZE, MAX_ACTIONS, ACTION_CATEGORY_MAX, N_CARD_TYPES, MAX_HAND_SLOTS,
    # per-action metadata block starts (never re-count blocks off STATE_SIZE)
    ACT_CATS_START, ACT_IDS_START, ACT_CTRL_START,
    # action category constants
    _CAT_PASS, _CAT_SEL_ATK, _CAT_CONF_ATK, _CAT_SEL_BLK, _CAT_CONF_BLK,
    _CAT_ACTIVATE, _CAT_CAST, _CAT_TARGET, _CAT_LAND, _CAT_MULLIGAN, _CAT_KEEP_HAND,
    _CAT_SEARCH,
    _CAT_OTHER, _CAT_PAYING, _CAT_DIG, _CAT_TOP_LIBRARY,
    _CAT_SB_IN, _CAT_SB_OUT, _CAT_SB_DONE,
    _CAT_COMPANION, _CAT_CHOOSE_X, _CAT_CHOOSE_CARD, _CAT_YESNO,
    _CAT_DISCARD, _CAT_KEEP_LEGEND, _CAT_CHOOSE_REPLACEMENT, _CAT_SACRIFICE,
    # battlefield / stack / graveyard layout
    _BF_START, _BF_SLOT_SIZE, _PERM_A_SLOTS, _BF_CARD_OFF, _STACK_START,
    _STACK_SLOT_SIZE, _HAND_START, _HAND_SLOT_SIZE, _CUR_TURN_IDX,
    _GY_START, _GY_SLOT_SIZE, MAX_GY_SLOTS,
    _OFF_IS_TAPPED, _OFF_IS_ATTACKING, _OFF_HAS_SICKNESS, _OFF_IS_CREATURE, _OFF_POWER,
    _OFF_IS_LAND, _OFF_IS_PHASED_OUT, _OFF_OTHER_COUNTERS,
    # per-action entity-reference slots (KEEP_LEGEND duplicate disambiguation)
    ACT_REFS_START, N_ENTITY_REF_SLOTS,
    # per-slot card-id decoder
    _slot_card_idx,
    # vocab / targeting constants
    _ACTION_CARD_ID_NULL, _ACTION_CTRL_NULL, _BLUE_POOL_IDX, _WASTELAND_VOCAB_IDX,
    _AETHER_VIAL_VOCAB_IDX,
    _BASIC_LAND_IDS, _COUNTER_SPELL_VOCAB_IDS, _COUNTERSPELL_VOCAB_IDX,
    _COUNTER_EXEMPT_CANTRIP_IDS, _TARGETED_REMOVAL_IDS, _WRATH_OF_SKIES_VOCAB_IDX,
    _TOKEN_VOCAB_BASE,
    _DOOMSDAY_VOCAB_IDX, _DARK_RITUAL_VOCAB_IDX, _THASSAS_ORACLE_VOCAB_IDX,
    _LOTUS_PETAL_VOCAB_IDX,
    _STREET_WRAITH_VOCAB_IDX, _EDGE_OF_AUTUMN_VOCAB_IDX, _DOOMSDAY_DECK_IDS,
    _PONDER_VOCAB_IDX, _BRAINSTORM_VOCAB_IDX, _CONSIDER_VOCAB_IDX,
    _KEEP_ONE_LANDER_IDS,
    _GREEN_SUNS_ZENITH_VOCAB_IDX, _KEEN_EYED_CURATOR_VOCAB_IDX, _MANA_DORK_IDS,
    _UNRELIABLE_LAND_IDS,
    _URZAS_MINE_VOCAB_IDX, _URZAS_POWER_PLANT_VOCAB_IDX, _URZAS_TOWER_VOCAB_IDX,
    _PLANAR_NEXUS_VOCAB_IDX, _TRON_LAND_IDS,
    _KARN_GREAT_CREATOR_VOCAB_IDX, _CITYSCAPE_LEVELER_VOCAB_IDX,
    _MYCOSYNTH_LATTICE_VOCAB_IDX,
    _CANDELABRA_VOCAB_IDX, _MULTI_MANA_LAND_IDS,
    _SOLITUDE_VOCAB_IDX, _THE_ONE_RING_VOCAB_IDX,
    # reanimation spells + the fatties they cheat out
    _REANIMATION_SPELL_IDS, _REANIMATION_FATTY_IDS,
    # lands-deck combo/engine pieces
    _LIFE_FROM_LOAM_VOCAB_IDX, _URZAS_SAGA_VOCAB_IDX,
    _DARK_DEPTHS_VOCAB_IDX, _THESPIANS_STAGE_VOCAB_IDX,
    _EXPLORATION_VOCAB_IDX, _TABERNACLE_VOCAB_IDX, _SAC_LAST_LAND_IDS,
    # Lion's Eye Diamond crack: blue-mana category + LED vocab id + draw-on-stack test
    _CAT_MANA_U, _LED_VOCAB_IDX, _self_has_draw_on_stack,
    # pending-decision source index (obs[_PENDING_DECISION_START] == source card id)
    _PENDING_DECISION_START,
    # library-count context index (obs[_LIBRARY_CTX_START] == self_library_ct / 60)
    _LIBRARY_CTX_START,
    # action-history ring start (newest entry first: cat, card id, is_self, turn)
    _HIST_START,
    # known top-of-library slot 0 (obs[_KNOWN_TOP_LIB_START] == top card id, sentinel=unknown)
    _KNOWN_TOP_LIB_START,
    # header flag / step one-hot indices + self player-block offsets
    _SELF_IS_A_IDX, _STEP_FIRST_MAIN_IDX, _STEP_SECOND_MAIN_IDX, _IS_ACTIVE_IDX,
    _SELF_BLOCK_START, _PB_LIFE,
    # shared helpers (also used by env's reward shaping, so they stay in env.py)
    _hand_has_card, _stack_is_empty,
    # unified deck-name identification (same rule as env's shaping opt-outs)
    _deck_named,
)
from decode import decode_game_state
from card_costs import _LAND_VOCAB_IDS, _CARD_COST_MATRIX
from _enums import CAT_BOTTOM_DECK_CARD as _CAT_BOTTOM_CARD


class Skill(Enum):
    """Scripted-agent skill tiers.

    RANDOM    — picks uniformly among the legal actions (a weak floor opponent).
    GREEDY    — the original rule-based behaviour; the baseline opponent.
    HEURISTIC — GREEDY's scan with selected decision points overridden by
                single-ply heuristics (board eval, combat sim, smarter targeting).
    EXPLORE   — coverage-oriented fuzzer: a deterministic, seed-mixed weighted
                choice biased toward the actions that drive the most engine code
                (casting, activating, combat, varied targets/modes/X), plus a
                per-game novelty boost for not-yet-chosen cast/activate cards.
                Not meant to play well — meant so that sweeping a range of seeds
                exercises as many distinct engine paths as possible
                (see _explore_action).
    """
    RANDOM = "random"
    GREEDY = "greedy"
    HEURISTIC = "heuristic"
    EXPLORE = "explore"


@dataclass(frozen=True)
class AgentConfig:
    """Behaviour toggles for a ScriptedAgent.

    HEURISTIC reuses GREEDY everywhere a toggle is off, so anything not explicitly
    improved falls back to today's proven behaviour (notably the Doomsday combo
    lines, which must never regress — see ``enable_combo_lines``).
    """
    skill: Skill = Skill.GREEDY
    use_combat_sim: bool = False        # profitable attacks/blocks via local P/T sim
    use_eval_targeting: bool = False    # removal -> biggest threat, burn -> lethal-only
    use_smart_mulligan: bool = False    # keep based on opening-hand land count
    use_tron_synergy: bool = False      # prioritize completing the Urza's Tron land set
    enable_combo_lines: bool = True     # keep deck-specific combo logic (Doomsday etc.)
    rng_seed: int | None = None         # seed for RANDOM's generator (reproducible)
    explore_profile: str = "default"    # EXPLORE weight profile: "default" (coverage)
                                        # or "patient" (big-mana; _EXPLORE_PATIENT_WEIGHTS)


# The "hard" configuration: heuristic play with combat simulation, evaluation-based
# targeting, and smart mulligans. This is ALSO the default a bare "scripted" resolves
# to — every default scripted opponent (the league/self-play scripted anchor, the
# mixed opponent pools, observe/baseline/analysis, the test harness) plays at this
# tier unless an explicit weaker suffix ("scripted:easy"/"greedy"/"random") is given.
_HARD_CONFIG = AgentConfig(skill=Skill.HEURISTIC, use_combat_sim=True,
                           use_eval_targeting=True, use_smart_mulligan=True,
                           use_tron_synergy=True)

# Canonical presets resolved from a spec-string suffix (see make_agent). The config
# object is read-only (ScriptedAgent only reads it), so the three hard aliases may
# safely share one instance.
_PRESETS: dict[str, AgentConfig] = {
    "random":    AgentConfig(skill=Skill.RANDOM, enable_combo_lines=False),
    "easy":      AgentConfig(skill=Skill.GREEDY),
    "greedy":    AgentConfig(skill=Skill.GREEDY),
    "scripted":  _HARD_CONFIG,     # bare "scripted"/default now == hard (was greedy)
    "hard":      _HARD_CONFIG,
    "heuristic": _HARD_CONFIG,
    # Coverage fuzzer (see Skill.EXPLORE). Combo lines are intentionally OFF so it
    # never collapses into a fixed deck-specific line — it fans out across cards.
    "explore":   AgentConfig(skill=Skill.EXPLORE, enable_combo_lines=False),
    "fuzz":      AgentConfig(skill=Skill.EXPLORE, enable_combo_lines=False),
    # Big-mana explore variant (see _EXPLORE_PATIENT_WEIGHTS): develops mana and
    # holds expensive cards until they are castable. Plain explore never even sees
    # a 7-drop as a LEGAL option — this profile exists to create that availability.
    "explore:patient": AgentConfig(skill=Skill.EXPLORE, enable_combo_lines=False,
                                   explore_profile="patient"),
    "patient":         AgentConfig(skill=Skill.EXPLORE, enable_combo_lines=False,
                                   explore_profile="patient"),
}


# Per-action-category weights for the EXPLORE tier (_explore_action). Higher weight =
# more likely to be chosen. The bias is toward the categories that exercise the most
# engine code (casting, activating, combat), with PASS kept low-but-nonzero so the game
# still advances, and the "finish this phase" confirms (attackers/blockers/sideboard)
# kept below their per-item counterparts so a few items get declared before confirming.
# Any category not listed (targets, X, modes, mana, search, dig, discard, sacrifice,
# name-a-card, ...) uses _EXPLORE_DEFAULT_WEIGHT — a plain uniform pick among that
# decision's options, so which target / X / mode is taken varies freely across seeds
# (one exception: the SEARCH_LIBRARY fail-to-find slot, see
# _EXPLORE_SEARCH_FAIL_WEIGHT).
_EXPLORE_DEFAULT_WEIGHT = 2.0
_EXPLORE_WEIGHTS: dict[int, float] = {
    _CAT_CAST:     8.0,   # cast a spell — stack, resolution, effects, triggers, SBAs
    _CAT_ACTIVATE: 6.0,   # activated abilities (mana, sacrifice, ETB-independent effects)
    _CAT_LAND:     4.0,   # play a land (fuels casts; land ETBs / fetch / man-lands)
    _CAT_SEL_ATK:  3.0,   # declare an attacker — combat triggers, damage, first strike
    _CAT_SEL_BLK:  3.0,   # declare a blocker — blocking, combat damage assignment
    _CAT_PASS:     1.0,   # advance priority / the game (low, but never zero)
    _CAT_CONF_ATK: 0.6,   # finish declaring attackers (declare a few first)
    _CAT_CONF_BLK: 0.6,   # finish declaring blockers
    _CAT_SB_DONE:  0.6,   # finish sideboarding (swap a few cards first, bo3)
    _CAT_COMPANION: 8.0,  # once-per-game and rarely offered — take it when legal so
                          # the companion machinery (and the card itself) gets fuzzed
}

# Novelty bias for the EXPLORE tier: a CAST_SPELL / ACTIVATE_ABILITY option whose
# card id this agent has NOT yet chosen this game gets its category weight
# multiplied by this factor. Without it, expensive cards sharing a deck with
# cheaper lines were NEVER cast across a whole seed sweep (e.g. Meteor Sword,
# hard-cast Lorien Revealed) — the weighted draw kept landing on the cheap
# alternatives. First-time options winning ~3x more often fixes that while the
# already-seen cards keep their normal weight. Tracked per game in
# ``ScriptedAgent._explore_seen`` (see ``new_game``).
_EXPLORE_NOVELTY_FACTOR = 3.0

# SEARCH_LIBRARY fail-to-find de-weight (relative to 1.0 per real pick; applies to
# BOTH explore profiles — a category-level tweak, not a patient one). A library
# search menu offers "Fail to find" alongside the real picks at the same uniform
# default weight, so a uniform draw failed ~1,200 tutors across a 441-game fuzz
# campaign — wasting coverage of the ChangeZone search/shuffle/ETB paths behind
# every fetch land and tutor. 0.3 keeps the fail branch reachable (the
# shuffle-without-find path still wants occasional fuzzing) while a real pick wins
# the large majority of searches. The fail slot is identified by its null card-id
# sentinel, not by index: the engine builds it from Entity(0) — no card — so its
# emitted card id decodes negative (src/components/ability.cpp "Fail to find" /
# machine_io.cpp action_card_vocab_idx), while every real pick is a library card
# with a vocab id >= 0. Mandatory searches simply have no such slot, and a
# fail-only menu is a 1-choice decision handled before weighting.
_EXPLORE_SEARCH_FAIL_WEIGHT = 0.3

# ── "Patient" big-mana EXPLORE profile (explore:patient) ────────────────────
# The plain-explore weight draw only picks among LEGAL actions, and empirically
# (130+ explore games) expensive cards never became legal: explore never develops
# enough mana for a 7-drop, and cheap alternate lines with the same card id (e.g.
# islandcycling Lorien Revealed) remove the card from hand long before it could be
# hard-cast. No weighting among legal options fixes availability — this profile
# does, by developing mana and holding expensive cards. Shares all the EXPLORE
# machinery (novelty bias on casts, seed-mixed global-random draw); only the
# weights below differ. Plain "explore" reads only _EXPLORE_WEIGHTS and is
# untouched by this table.
_EXPLORE_PATIENT_WEIGHTS: dict[int, float] = {
    _CAT_CAST:     8.0,   # base; scaled up by mana value (_PATIENT_CAST_MV_FACTOR)
                          # so a legal expensive cast dominates the draw
    _CAT_ACTIVATE: 1.0,   # demoted hard: cheap activations (cycling, sac lines)
                          # are what strip expensive cards from hand early
    _CAT_LAND:     100.0, # near-always take the land drop — mana development is
                          # the whole point of this profile
    _CAT_SEL_ATK:  3.0,   # keep combat as in plain explore so games stay decisive
    _CAT_SEL_BLK:  3.0,
    _CAT_PASS:     6.0,   # bank mana-development turns: an early cheap ACTIVATE
                          # now competes with a strong PASS instead of winning
    _CAT_CONF_ATK: 0.6,   # confirms below their per-item counterparts, as in
    _CAT_CONF_BLK: 0.6,   # _EXPLORE_WEIGHTS, so a few get declared first
    _CAT_SB_DONE:  0.6,
    _CAT_COMPANION: 8.0,  # as in _EXPLORE_WEIGHTS: once-per-game, take it when legal
}

# Patient-only CAST scaling: weight *= (1 + mana_value * factor). A legal MV-7
# cast gets 8x the pull of an MV-0 one, so once the mana finally exists the big
# card is cast almost immediately instead of losing the draw to cheap spells.
_PATIENT_CAST_MV_FACTOR = 1.0

# Patient mode gives NO novelty boost to ACTIVATE options. The novelty factor is
# what makes plain explore eagerly try an unseen islandcycle/sac activation the
# first turn it appears — exactly the card-stripping behaviour this profile must
# suppress. Casts keep the boost (a novel expensive cast should win the draw).
_PATIENT_ACTIVATE_NOVELTY = False

# Patient-only "hold" scaling for card-consuming activations: an ACTIVATE whose
# card id is still in the acting player's HAND (cycling, Talon Gates-style
# from-hand lines) gets its weight DIVIDED by (1 + mana_value * factor) — the
# mirror image of the CAST scaling. A flat ACTIVATE demotion is not enough: the
# option reappears at every priority window, so even a small per-window pick
# probability compounds into "always cycled before turn 5" over a game. Scaling
# the demotion by MV holds exactly the cards this profile exists to hard-cast
# while leaving cheap from-hand activations barely affected. Battlefield
# activations (Wasteland, equip, ...) are untouched — their card is not in hand.
_PATIENT_HOLD_MV_FACTOR = 1.0


def _card_mana_value(cid: int) -> float:
    """Printed mana value of a vocab card via the cast-cost matrix.

    ``_CARD_COST_MATRIX`` rows are the 7 pip counts (W/U/B/R/G/C/generic)
    divided by 10, so the row sum times 10 is the card's total mana value
    (X contributes 0). Unknown/out-of-range ids count as 0.
    """
    if 0 <= cid < len(_CARD_COST_MATRIX):
        return float(_CARD_COST_MATRIX[cid].sum()) * 10.0
    return 0.0


def _action_card_id(card_ids: np.ndarray, i: int) -> int:
    """Decode the card vocab index from the action's card_id float."""
    return int(round(float(card_ids[i]) * N_CARD_TYPES))


def _is_doomsday_deck(obs: np.ndarray) -> bool:
    """Heuristic: a doomsday-deck-only card is in hand, on our battlefield
    (Cavern of Souls, Lotus Petal, Lion's Eye Diamond), or in a graveyard —
    post-Doomsday the hand is often cantrips and counters only, while the
    resolved Doomsday and the Dark Ritual that paid for it sit in the
    graveyard, and that is exactly when the post-pile rules must fire."""
    for slot in range(MAX_HAND_SLOTS):
        idx = _slot_card_idx(obs, _HAND_START + slot * _HAND_SLOT_SIZE)
        if idx in _DOOMSDAY_DECK_IDS:
            return True
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if _slot_card_idx(obs, base + _BF_CARD_OFF) in _DD_EXCLUSIVE_IDS:
            return True
    return _gy_has_any(obs, _DD_EXCLUSIVE_IDS)


# Cards only the Doomsday deck runs — the battlefield / graveyard half of
# _is_doomsday_deck scans for these alone. The wider _DOOMSDAY_DECK_IDS hand
# set includes Dark Ritual and Lotus Petal, which reanimator also runs: a
# Ritual in reanimator's graveyard must not flip it onto the Doomsday rules
# for the rest of the game.
_DD_EXCLUSIVE_IDS = frozenset({
    53,   # Doomsday
    57,   # Lion's Eye Diamond
    58,   # Thassa's Oracle
    59,   # Personal Tutor
    60,   # Street Wraith
    61,   # Edge of Autumn
})


def _phased(obs: np.ndarray, base: int) -> bool:
    """True if the permanent slot at `base` is phased out.

    Phased-out permanents ARE serialized (new obs layout) but the rules treat
    them as nonexistent (CR 702.26e) — and the pre-enrichment engine never
    emitted them — so EVERY battlefield scan here must skip them. That keeps
    scripted decisions identical to the pre-change engine (replay-corpus
    stability), besides being rules-correct."""
    return obs[base + _OFF_IS_PHASED_OUT] > 0.5


def _all_eligible_creatures_attacking(obs: np.ndarray) -> bool:
    """Return True if every untapped, non-sick creature in self's slots (0-47) is attacking."""
    any_eligible = False
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if _phased(obs, base):
            continue  # phased out — treated as not on the battlefield
        if obs[base + _OFF_IS_CREATURE] < 0.5:
            continue  # not a creature (empty slot, land, or other permanent)
        if obs[base + _OFF_IS_TAPPED] > 0.5 or obs[base + _OFF_HAS_SICKNESS] > 0.5:
            continue  # can't attack
        any_eligible = True
        if obs[base + _OFF_IS_ATTACKING] <= 0.5:
            return False  # eligible but not yet attacking
    return any_eligible


def _opponent_has_nonbasic_land(obs: np.ndarray) -> bool:
    """Return True if opponent has at least one nonbasic land (opp perm slots 48-95)."""
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + (slot + _PERM_A_SLOTS) * _BF_SLOT_SIZE
        if _phased(obs, base):
            continue
        if obs[base + _OFF_IS_LAND] < 0.5:
            continue  # not a land
        idx = _slot_card_idx(obs, base + _BF_CARD_OFF)
        if idx >= 0 and idx not in _BASIC_LAND_IDS:
            return True
    return False


def _opponent_has_creature(obs: np.ndarray) -> bool:
    """Return True if opponent controls at least one creature (opp perm slots 48-95)."""
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + (slot + _PERM_A_SLOTS) * _BF_SLOT_SIZE
        if obs[base + _OFF_IS_CREATURE] > 0.5 and not _phased(obs, base):
            return True
    return False


def _controls_card(obs: np.ndarray, cid: int) -> bool:
    """True if self controls an unphased permanent with this vocab id (slots 0-47)."""
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if _phased(obs, base):
            continue
        if _slot_card_idx(obs, base + _BF_CARD_OFF) == cid:
            return True
    return False


def _gy_has_any(obs: np.ndarray, ids: frozenset) -> bool:
    """True if EITHER graveyard holds a card whose vocab id is in ``ids``.

    The self and opponent graveyard blocks are contiguous card-id slots
    (self first), so one scan over 2*MAX_GY_SLOTS covers both. Both sides
    matter to the caller (Reanimate can take a creature from any graveyard).
    """
    for slot in range(2 * MAX_GY_SLOTS):
        if _slot_card_idx(obs, _GY_START + slot * _GY_SLOT_SIZE) in ids:
            return True
    return False


def _action_slot_ref(obs: np.ndarray, i: int) -> int:
    """Decode action i's entity-reference slot (-1 = none; 0-47 self perms,
    48-95 opp perms, 96-107 stack — the unified reference space)."""
    return int(round(float(obs[ACT_REFS_START + i]) * N_ENTITY_REF_SLOTS)) - 1


def _tapped_multi_mana_lands(obs: np.ndarray) -> int:
    """Count self-controlled TAPPED lands that make more than one mana (self slots 0-47).

    These are the lands worth untapping with Candelabra of Tawnos — untapping a tapped
    Ancient Tomb / metalcraft Urza's Workshop / assembled Urza's tron land frees 2-3 mana
    apiece, whereas untapping a one-mana land (or any untapped land) gains nothing.
    """
    count = 0
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if _phased(obs, base):
            continue
        if obs[base + _OFF_IS_TAPPED] < 0.5:
            continue  # untapped — nothing to untap
        if _slot_card_idx(obs, base + _BF_CARD_OFF) in _MULTI_MANA_LAND_IDS:
            count += 1
    return count


def _controlled_tron_pieces(obs: np.ndarray) -> set[int]:
    """Vocab ids of Urza's Mine/Power-Plant/Tower self currently controls
    (self battlefield slots 0-47). Planar Nexus is deliberately excluded — it's
    the substitute offered once a real piece is missing, not a piece itself."""
    found: set[int] = set()
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if _phased(obs, base):
            continue
        idx = _slot_card_idx(obs, base + _BF_CARD_OFF)
        if idx in _TRON_LAND_IDS:
            found.add(idx)
    return found


def _tron_choice(cats, card_ids, cat: int, controlled: set[int]) -> int | None:
    """Pick whichever offered ``cat``-category action completes the Tron set:
    a real Urza's Mine/Power-Plant/Tower self doesn't yet control, else Planar
    Nexus (every nonbasic land type via its self-CDA, so it substitutes for any
    missing piece — see state_manager_statics.cpp). Returns None if nothing
    offered helps, so the caller can fall back to its normal pick.
    """
    for target in _TRON_LAND_IDS - controlled:
        for i, c in enumerate(cats):
            if c == cat and _action_card_id(card_ids, i) == target:
                return i
    for i, c in enumerate(cats):
        if c == cat and _action_card_id(card_ids, i) == _PLANAR_NEXUS_VOCAB_IDX:
            return i
    return None


def _opponent_has_spell_on_stack(obs: np.ndarray) -> bool:
    """Return True if at least one spell/ability on the stack is not controlled by self."""
    for i in range(12):
        base = _STACK_START + i * _STACK_SLOT_SIZE
        ctrl_is_self = obs[base]
        if _slot_card_idx(obs, base + 1) >= 0 and ctrl_is_self < 0.5:
            return True
    return False


def _self_stack_has_card(obs: np.ndarray, cid: int) -> bool:
    """Return True if self controls a stack object with the given card id."""
    for i in range(12):
        base = _STACK_START + i * _STACK_SLOT_SIZE
        if obs[base] > 0.5 and _slot_card_idx(obs, base + 1) == cid:
            return True
    return False


def _opponent_threat_on_stack(obs: np.ndarray) -> bool:
    """Threat-type counter triage: True if a non-self stack object is worth a
    counterspell — anything that is NOT a known pure cantrip
    (_COUNTER_EXEMPT_CANTRIP_IDS). Creatures/planeswalkers/enchantments at any
    MV and every other MV>=2 spell fall through the exemption and count."""
    for i in range(12):
        base = _STACK_START + i * _STACK_SLOT_SIZE
        if obs[base] >= 0.5:
            continue  # self-controlled
        cid = _slot_card_idx(obs, base + 1)
        if cid >= 0 and cid not in _COUNTER_EXEMPT_CANTRIP_IDS:
            return True
    return False


def _hand_land_count(obs: np.ndarray) -> int:
    """Lands in the priority player's hand."""
    count = 0
    for slot in range(MAX_HAND_SLOTS):
        if _slot_card_idx(obs, _HAND_START + slot * _HAND_SLOT_SIZE) in _LAND_VOCAB_IDS:
            count += 1
    return count


def _self_land_count(obs: np.ndarray) -> int:
    """Lands self controls (unphased, self battlefield slots 0-47)."""
    count = 0
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if obs[base + _OFF_IS_LAND] > 0.5 and not _phased(obs, base):
            count += 1
    return count


# ── Doomsday combo constants/helpers ────────────────────────────────────────
# Doomsday keeps exactly 5 cards on top of the library (doomsday.txt ChangeNum$ 5).
# Thassa's Oracle wins the game when devotion-to-blue (>= 2 from its own UU) is
# >= the library size, so casting Oracle with <= 2 cards left is a guaranteed win.
_DD_PILE_SIZE = 5
_ORACLE_WIN_LIBRARY = 2

# Green Sun's Zenith costs {X}{G}. Hold it until at least this many untapped mana
# sources are available so it is never cast for X=0 (which can only fetch the
# 0-mana Dryad Arbor); 2 sources buy X=1, more buy a bigger creature.
_GSZ_MIN_MANA_SOURCES = 2

# Street Wraith cycling costs 2 life; never pay it below this so the agent can't
# cycle itself to death digging through the pile.
_STREET_WRAITH_MIN_LIFE = 3
_SELF_LIFE_IDX = _SELF_BLOCK_START + _PB_LIFE   # priority player's life / 20

# Permanents that can produce blue mana for Thassa's Oracle (UU). Cavern of Souls
# counts because Oracle is a creature spell; Lotus Petal makes any colour when
# sacrificed. We gate Doomsday on controlling at least two so the UU for Oracle
# is actually available after the combo resolves.
_BLUE_SOURCES_NEEDED = 2
_BLUE_SOURCE_IDS = frozenset({
    4,    # Volcanic Island (U/R)
    15,   # Tundra (W/U)
    19,   # Island (U)
    63,   # Undercity Sewers (U/B)
    64,   # Underground Sea (U/B)
    67,   # Cavern of Souls (any colour for creature spells, e.g. Oracle)
    56,   # Lotus Petal (sacrifice: one mana of any colour)
    300,  # Tropical Island (U/G) — wubg_doomsday runs 2; omitting it
          # undercounted blue sources and held Doomsday longer than needed
})

# ── Doomsday all-in mulligan (vs decks with no Force of Will) ───────────────
# Hand-based mana classification for the fast-kill keep test. A fetch counts
# for every colour the deck's duals let it find (Flooded Strand and Polluted
# Delta fetch Underground Sea, Verdant Catacombs fetches Sea or Tropical);
# Lotus Petal counts for both sides.
_DD_HAND_B_SOURCE_IDS = frozenset({
    62,   # Swamp
    63,   # Undercity Sewers (U/B)
    64,   # Underground Sea (U/B)
    153,  # Scrubland (W/B)
    65,   # Bloodstained Mire (fetches a Swamp-typed dual)
    66,   # Verdant Catacombs (fetches Underground Sea)
    6,    # Flooded Strand (fetches Underground Sea / Scrubland)
    7,    # Polluted Delta (fetches Sea / Sewers)
    56,   # Lotus Petal
})
_DD_HAND_U_SOURCE_IDS = frozenset({
    19,   # Island
    15,   # Tundra (W/U)
    4,    # Volcanic Island (U/R)
    63,   # Undercity Sewers (U/B)
    64,   # Underground Sea (U/B)
    67,   # Cavern of Souls (creature spells — pays for Oracle)
    300,  # Tropical Island (U/G)
    5,    # Scalding Tarn (fetches an Island-typed dual)
    9,    # Misty Rainforest (fetches an Island-typed dual)
    6,    # Flooded Strand
    7,    # Polluted Delta
    66,   # Verdant Catacombs (fetches Tropical Island)
    56,   # Lotus Petal
})

# Instant-speed removal that can kill Thassa's Oracle mid-combo (in response
# to the cast or the win trigger) — what a Doomsday deck's Thoughtseize-style
# discard strips right after countermagic. The instant subset of
# _TARGETED_REMOVAL_IDS: Prismatic Ending is sorcery-speed, so it can't
# disrupt the kill and stays on the highest-MV fallback.
_DD_DISRUPTOR_REMOVAL_IDS = frozenset({
    48,   # Swords to Plowshares
    84,   # Fatal Push
})

_DECKS_DIR = Path(__file__).resolve().parent.parent / "bin" / "resources" / "decks"
_deck_fow_cache: dict = {}


def _deck_runs_force_of_will(deck_name: str):
    """Whether the named deck's .dk file (main or sideboard) contains Force of
    Will. Returns True/False, or None when the list can't be read (unknown
    deck name, sculpted temp deck already cleaned up, ...) — callers treat
    None as "assume it does" and stay conservative."""
    if deck_name in _deck_fow_cache:
        return _deck_fow_cache[deck_name]
    try:
        text = (_DECKS_DIR / f"{deck_name}.dk").read_text()
    except OSError:
        result = None
    else:
        result = "force of will" in text.lower()
    _deck_fow_cache[deck_name] = result
    return result


def _dd_fast_hand(obs: np.ndarray, hand_ids) -> bool:
    """Can this opening hand cast Doomsday by turn 1-3 and convert right
    after? Requirements: Doomsday itself; BBB reachable on at most three land
    drops plus Petals plus a Dark Ritual's net +2; a black-capable source to
    kick things off; and two blue-capable sources for Thassa's Oracle's UU —
    or Lion's Eye Diamond, which pays for Oracle by itself in the crack
    window. Sources are counted per-card across turns (the ritual land untaps
    before the Oracle turn), so one dual serving both colours is fine.
    Battlefield sources count too — normally none exist at the mulligan, but
    sculpted harness scenarios preset them."""
    if _DOOMSDAY_VOCAB_IDX not in hand_ids:
        return False
    rituals = sum(1 for c in hand_ids if c == _DARK_RITUAL_VOCAB_IDX)
    petals = sum(1 for c in hand_ids if c == _LOTUS_PETAL_VOCAB_IDX)
    b_lands = sum(1 for c in hand_ids
                  if c in _DD_HAND_B_SOURCE_IDS and c != _LOTUS_PETAL_VOCAB_IDX)
    bf_b = _count_untapped_sources(obs, _DD_HAND_B_SOURCE_IDS)
    if min(b_lands, 3) + bf_b + petals == 0:
        return False
    if min(b_lands, 3) + bf_b + petals + 2 * min(rituals, 1) < 3:
        return False
    blue = (sum(1 for c in hand_ids if c in _DD_HAND_U_SOURCE_IDS)
            + _count_untapped_sources(obs, _DD_HAND_U_SOURCE_IDS))
    if _LED_VOCAB_IDX in hand_ids:
        blue += 2
    return blue >= 2


def _led_crack_choice(obs: np.ndarray, cats, card_ids) -> int | None:
    """Lion's Eye Diamond: crack it for blue only in the Doomsday kill window. All of:

      1. a self-controlled draw/cycling spell or ability is on the stack, AND
      2. Thassa's Oracle is the known top card of the library (Doomsday has resolved
         and laid the pile — RearrangeTopOfLibrary sets the known-top slots), AND
      3. the library is down to <= 2 cards (the pile is drawn down to the win).

    In that window the resolving draw puts Oracle into hand and the library hits the
    Oracle-win size; cracking LED floats UUU that outlives the draw and pays Oracle's
    UU. Gated this tightly, LED's "discard your hand" cost never sheds a card that
    still matters, so no empty-hand check is needed. Returns the MANA_U action index
    for LED, or None when the window / the offer isn't present.
    """
    if not _self_has_draw_on_stack(obs):
        return None
    if _slot_card_idx(obs, _KNOWN_TOP_LIB_START) != _THASSAS_ORACLE_VOCAB_IDX:
        return None
    # The pending draw takes Oracle off the top, so the library is one card
    # above the win size while it sits on the stack (standard pile with LED
    # already in play: the Street Wraith draw fires at three cards).
    if _self_library_count(obs) > _ORACLE_WIN_LIBRARY + 1:
        return None
    for i, c in enumerate(cats):
        if c == _CAT_MANA_U and _action_card_id(card_ids, i) == _LED_VOCAB_IDX:
            return i
    return None


def _self_library_count(obs: np.ndarray) -> int:
    """Cards in the priority player's library (obs stores it as count / 60)."""
    return int(round(float(obs[_LIBRARY_CTX_START]) * 60))


def _self_life(obs: np.ndarray) -> int:
    """Priority player's life total (obs stores it as life / 20)."""
    return int(round(float(obs[_SELF_LIFE_IDX]) * 20))


def _count_untapped_sources(obs: np.ndarray, id_set) -> int:
    """Untapped permanents self controls whose card id is in ``id_set``
    (self battlefield slots 0..47)."""
    count = 0
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if obs[base + _OFF_IS_TAPPED] > 0.5 or _phased(obs, base):
            continue
        idx = _slot_card_idx(obs, base + _BF_CARD_OFF)
        if idx >= 0 and idx in id_set:
            count += 1
    return count


def _count_blue_sources(obs: np.ndarray) -> int:
    """Blue-mana sources self controls, tapped or not. The Doomsday gate asks
    whether Oracle's UU will exist by KILL time (next untap, or the LED crack
    window), so a source tapped mid-combo-turn — e.g. the Underground Sea
    that just paid for Dark Ritual — still counts. Counting only untapped
    sources made the ritual itself flip the Doomsday gate false."""
    count = 0
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if _phased(obs, base):
            continue
        idx = _slot_card_idx(obs, base + _BF_CARD_OFF)
        if idx >= 0 and idx in _BLUE_SOURCE_IDS:
            count += 1
    return count


def _count_untapped_mana_sources(obs: np.ndarray) -> int:
    """Untapped mana-producers self controls (self battlefield slots 0..47).

    Counts every untapped land plus the untapped mana-dork creatures (Birds /
    Noble & Ignoble Hierarch). Used to decide whether Green Sun's Zenith can
    afford a meaningful X — its cost is {X}{G}, so N sources buy X = N-1.
    """
    count = 0
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if obs[base + _OFF_IS_TAPPED] > 0.5 or _phased(obs, base):
            continue
        idx = _slot_card_idx(obs, base + _BF_CARD_OFF)
        is_land = obs[base + _OFF_IS_LAND] > 0.5 and idx not in _UNRELIABLE_LAND_IDS
        is_dork = idx in _MANA_DORK_IDS
        if not (is_land or is_dork):
            continue
        # A creature mana-source (dork, or a creature-land like Dryad Arbor) can't
        # tap for mana while summoning-sick, so it can't pay toward X yet.
        if obs[base + _OFF_IS_CREATURE] > 0.5 and obs[base + _OFF_HAS_SICKNESS] > 0.5:
            continue
        count += 1
    return count


# Free-draw enablers for the Doomsday pile: cycling that costs no mana (Street
# Wraith pays life, Edge of Autumn sacs a land), so a drawn one is immediately
# re-cycled to keep drawing down the pile the same turn.
_DD_FREE_DRAW_IDS = frozenset({_STREET_WRAITH_VOCAB_IDX, _EDGE_OF_AUTUMN_VOCAB_IDX})
# Cheap cantrips: preferred pile filler (a drawn one is castable for more digging)
# but NOT counted toward Oracle's position — they cost mana, so the pile plan
# only relies on the free draws plus natural draw steps.
_DD_CANTRIP_IDS = frozenset({_PONDER_VOCAB_IDX, _BRAINSTORM_VOCAB_IDX,
                             _CONSIDER_VOCAB_IDX})


def _hand_free_draw_count(obs: np.ndarray) -> int:
    """Free-draw cyclers (Street Wraith / Edge of Autumn) in the priority
    player's hand. The Brainstorm same-turn pile plan needs at least one:
    Brainstorm's two put-backs cost two draws, so the pile's own cyclers
    always come up exactly one draw short without a cycler already in hand."""
    count = 0
    for slot in range(MAX_HAND_SLOTS):
        if _slot_card_idx(obs, _HAND_START + slot * _HAND_SLOT_SIZE) in _DD_FREE_DRAW_IDS:
            count += 1
    return count


# Blue sources a hand can put onto the battlefield WITHOUT a library search:
# the direct-play half of _DD_HAND_U_SOURCE_IDS. Fetch lands are excluded on
# purpose — after Doomsday lays the pile a fetch can't be cracked (the search
# shuffles the pile away), so a fetch in hand covers nothing for Oracle.
_DD_HAND_U_DIRECT_IDS = frozenset({19, 15, 4, 63, 64, 67, 300, 56})


def _hand_direct_blue_sources(obs: np.ndarray) -> int:
    """Blue sources in the priority player's hand that a land drop / a free
    Lotus Petal cast puts into play before the kill turn."""
    count = 0
    for slot in range(MAX_HAND_SLOTS):
        if _slot_card_idx(obs, _HAND_START + slot * _HAND_SLOT_SIZE) in _DD_HAND_U_DIRECT_IDS:
            count += 1
    return count


def _petals_spent_on_black(obs: np.ndarray) -> int:
    """How many Lotus Petals on the battlefield Doomsday's BBB (or the Dark
    Ritual starting it) will consume. A Petal is in both the blue and the
    black source sets, but it is one mana once: the agent once cracked its
    only Petal for Dark Ritual, then held Doomsday because that same Petal
    had been counted as Oracle's second blue source."""
    petals = _count_controlled(obs, _LOTUS_PETAL_VOCAB_IDX)
    if petals == 0:
        return 0
    black_lands = _count_black_sources(obs) - petals
    black_needed = 1 if _hand_has_card(obs, _DARK_RITUAL_VOCAB_IDX) else 3
    return max(0, min(petals, black_needed - black_lands))


def _dd_blue_for_oracle(obs: np.ndarray) -> bool:
    """Will Thassa's Oracle's UU exist by kill time? Two blue sources on the
    battlefield (Petals the BBB will eat excluded), or an LED that pays for
    Oracle by itself, or one blue source in play plus a directly playable one
    in hand: the kill lands on the next turn's draw step at the earliest, and
    the post-pile Edge / Wraith chain waits for that land drop
    (_dd_edge_cycle_safe) before drawing Oracle. Waiting for the second blue
    land BEFORE casting Doomsday was the main reason the pilot went off on
    turn 7-13 instead of turn 3-5."""
    on_board = _count_blue_sources(obs) - _petals_spent_on_black(obs)
    if on_board >= _BLUE_SOURCES_NEEDED or _led_covers_oracle(obs):
        return True
    if on_board < 1:
        return False
    # One real source is enough when the second can come off the top of the
    # pile: the basic Island is still in the library, and the pile plan
    # (_dd_pile_wants_land) puts it there, drawn before the chain reaches
    # Oracle. Waiting for a second blue land drop instead cost a turn. Not
    # from a one-land board, though: a Wasteland on that land leaves no mana
    # at all for the kill.
    return (on_board + _hand_direct_blue_sources(obs) >= _BLUE_SOURCES_NEEDED
            or (_dd_pile_land_available(obs) and _count_lands(obs) >= 2))


def _dd_pile_land_available(obs: np.ndarray) -> bool:
    """Is the basic Island (the pile's blue land) still in the library? It is
    a single copy, so: not on our battlefield, not in hand, not in a
    graveyard. (Exile is not checked — pre-Doomsday nothing of ours is
    exiled, and post-Doomsday the question no longer arises.)"""
    island = _DD_PILE_LAND_IDS[0]
    return (not _controls_card(obs, island)
            and not _hand_has_card(obs, island)
            and not _gy_has_any(obs, frozenset({island})))


def _led_covers_oracle(obs: np.ndarray) -> bool:
    """Lion's Eye Diamond pays Thassa's Oracle's UU on its own: cracked in the
    kill window (_led_crack_choice) it floats three blue that outlive the final
    cycle draw. So an LED in hand (castable for 0) or already on the
    battlefield satisfies the blue coverage the Doomsday gate demands even
    when fewer than two untapped blue sources are up."""
    return (_hand_has_card(obs, _LED_VOCAB_IDX)
            or _controls_card(obs, _LED_VOCAB_IDX))


# Black-mana producers a Doomsday deck can control (Doomsday costs BBB; Dark
# Ritual needs one B to start). Lotus Petal makes any colour.
_DD_BLACK_SOURCE_IDS = frozenset({
    62,   # Swamp
    63,   # Undercity Sewers (U/B)
    64,   # Underground Sea (U/B)
    153,  # Scrubland (W/B)
    56,   # Lotus Petal
})
# Cards a pre-pile surveil must keep on top rather than mill: the kill itself,
# the mana that casts it, and the pile's free draws.
_DD_COMBO_PIECE_IDS = frozenset({
    _DOOMSDAY_VOCAB_IDX, _DARK_RITUAL_VOCAB_IDX, _THASSAS_ORACLE_VOCAB_IDX,
    _LED_VOCAB_IDX, _LOTUS_PETAL_VOCAB_IDX, _STREET_WRAITH_VOCAB_IDX,
    _EDGE_OF_AUTUMN_VOCAB_IDX,
    59,   # Personal Tutor
})
# Fetch-land / land-search preference while piloting Doomsday: a dual serving
# BOTH combo colours first (Underground Sea untapped ahead of the enters-tapped
# Sewers), then whichever single colour the board is missing.
_DD_FETCH_BOTH_COLOURS = (64, 63)
_DD_FETCH_BLACK = (153, 62)
_DD_FETCH_BLUE = (19, 15, 300)
# Smallest library each cantrip may be cast into: Brainstorm draws three,
# Ponder draws one, Consider may surveil its one card away and then draw.
# Drawing from an empty library loses the game.
_DD_CANTRIP_MIN_LIBRARY = {_BRAINSTORM_VOCAB_IDX: 3, _PONDER_VOCAB_IDX: 1,
                           _CONSIDER_VOCAB_IDX: 2}


def _count_lands(obs: np.ndarray) -> int:
    """Unphased lands self controls, tapped or not."""
    count = 0
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if not _phased(obs, base) and obs[base + _OFF_IS_LAND] > 0.5:
            count += 1
    return count


def _count_controlled(obs: np.ndarray, cid: int) -> int:
    """How many unphased permanents with this vocab id self controls."""
    count = 0
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if not _phased(obs, base) and _slot_card_idx(obs, base + _BF_CARD_OFF) == cid:
            count += 1
    return count


# Our own spells on the stack that countermagic in hand must protect against
# an opponent's response: the kill, Doomsday, and the Ritual paying for it.
_DD_PROTECT_ON_STACK_IDS = (_THASSAS_ORACLE_VOCAB_IDX, _DOOMSDAY_VOCAB_IDX,
                            _DARK_RITUAL_VOCAB_IDX)
# Spells a Doomsday deck may still cast off Petal mana while Doomsday waits in
# hand: the combo and its protection. Anything else (Tamiyo, Teferi, Quantum
# Riddler) must be paid by lands alone — three Petals once went to Ponder,
# Tamiyo and Teferi with Doomsday in hand and the black lands eight turns away.
_DD_PETAL_WORTHY_IDS = frozenset({
    _DOOMSDAY_VOCAB_IDX, _DARK_RITUAL_VOCAB_IDX, _THASSAS_ORACLE_VOCAB_IDX,
    _LOTUS_PETAL_VOCAB_IDX, _LED_VOCAB_IDX,
    _PONDER_VOCAB_IDX, _BRAINSTORM_VOCAB_IDX, _CONSIDER_VOCAB_IDX,
    283,  # Veil of Summer (protection for the kill turn)
}) | _COUNTER_SPELL_VOCAB_IDS
# Payment / pitch preference: mana sources and pitch fodder ranked worst-first.
# Cracking Petal / LED to pay for a cantrip wastes the combo's mana; pitching
# Oracle to Force of Will ends the game.
_DD_PAY_LAST_IDS = frozenset({_LOTUS_PETAL_VOCAB_IDX, _LED_VOCAB_IDX})


def _count_black_sources(obs: np.ndarray) -> int:
    """Black-mana producers self controls, tapped or not (the black-side twin
    of _count_blue_sources)."""
    count = 0
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if _phased(obs, base):
            continue
        idx = _slot_card_idx(obs, base + _BF_CARD_OFF)
        if idx >= 0 and idx in _DD_BLACK_SOURCE_IDS:
            count += 1
    return count


def _dd_bbb_reachable(obs: np.ndarray) -> bool:
    """Can Doomsday's BBB be paid this turn or next? Black sources on the
    board plus the Lotus Petals in hand (free to cast), and a Dark Ritual in
    hand turns one of those into three."""
    black = _count_black_sources(obs)
    for slot in range(MAX_HAND_SLOTS):
        if _slot_card_idx(obs, _HAND_START + slot * _HAND_SLOT_SIZE) == _LOTUS_PETAL_VOCAB_IDX:
            black += 1
    if black >= 1 and _hand_has_card(obs, _DARK_RITUAL_VOCAB_IDX):
        black += 2
    return black >= 3


def _dd_combo_ready(obs: np.ndarray) -> bool:
    """Doomsday is castable this turn or next: BBB reachable and Oracle's UU
    covered. Gates the pre-Doomsday Brainstorm hold — while the mana isn't
    there the agent must dig with Brainstorm for the Ritual / Petal / black
    land, not sit on it for a kill it can't start (it once held Brainstorm
    for eight turns with three Doomsdays and Oracle in hand on two black
    sources and no Ritual)."""
    return _dd_bbb_reachable(obs) and _dd_blue_for_oracle(obs)


def _opponent_board_power(obs: np.ndarray) -> int:
    """Total power of the opponent's unphased creatures (opp perm slots 48-95),
    summoning-sick ones included — all of them attack on the opponent's next
    turn."""
    total = 0
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + (slot + _PERM_A_SLOTS) * _BF_SLOT_SIZE
        if obs[base + _OFF_IS_CREATURE] > 0.5 and not _phased(obs, base):
            total += int(round(float(obs[base + _OFF_POWER]) * 10))
    return total


def _dd_clock_allows_doomsday(obs: np.ndarray) -> bool:
    """Doomsday costs half our life (rounded up), and the kill lands on the NEXT
    draw step at the earliest unless a free-draw cycler plus Brainstorm are
    already in hand (the same-turn pile line). Hold it when the opponent's
    board would kill us in between — we survive their next attack at full life
    but not at half — since firing it then is a guaranteed loss. If the board
    kills us either way, go off: the combo is the only out."""
    power = _opponent_board_power(obs)
    if power == 0:
        return True
    if (_hand_free_draw_count(obs) >= 1
            and _hand_has_card(obs, _BRAINSTORM_VOCAB_IDX)):
        return True
    life = _self_life(obs)
    if life // 2 > power:
        return True
    return life <= power


def _dd_fetch_pick(obs: np.ndarray, cats, card_ids) -> int | None:
    """Land-search pick for a Doomsday deck: Underground Sea / Undercity Sewers
    (both colours) first, then the colour the board lacks — black when nothing
    makes B yet (Dark Ritual and Doomsday both stall without it), else blue
    until Oracle's UU is covered. Returns None when no preferred land is
    offered (a non-land search, or a menu of lands outside the table)."""
    offered: dict[int, int] = {}
    for i, c in enumerate(cats):
        if c == _CAT_SEARCH:
            cid = _action_card_id(card_ids, i)
            if cid >= 0 and cid not in offered:
                offered[cid] = i
    if not offered:
        return None
    # Doomsday costs BBB: three black sources from lands alone, one plus a
    # Dark Ritual in hand. Blue is covered at Oracle's UU.
    black_needed = 1 if _hand_has_card(obs, _DARK_RITUAL_VOCAB_IDX) else 3
    order = list(_DD_FETCH_BOTH_COLOURS)
    if _count_black_sources(obs) < black_needed:
        order += _DD_FETCH_BLACK + _DD_FETCH_BLUE
    else:
        order += _DD_FETCH_BLUE + _DD_FETCH_BLACK
    for cid in order:
        if cid in offered:
            return offered[cid]
    return None


def _dd_land_sac_key(obs: np.ndarray, cid: int, tapped: bool):
    """Sort key for sacrificing one of our lands while piloting Doomsday —
    lower is better. In order: would it drop the UNTAPPED blue sources below
    Oracle's UU (the kill may be this very turn); would it drop the total blue
    sources below UU (next turn's kill); would it remove our only black source
    (pre-Doomsday, Dark Ritual and Doomsday still need it); then plain
    blue / black membership as tie-breaks so a colourless or off-colour land
    goes first."""
    is_blue = cid in _BLUE_SOURCE_IDS
    is_black = cid in _DD_BLACK_SOURCE_IDS
    untapped_blue = _count_untapped_sources(obs, _BLUE_SOURCE_IDS)
    blue = _count_blue_sources(obs)
    black = _count_black_sources(obs)
    return (is_blue and not tapped and untapped_blue - 1 < _BLUE_SOURCES_NEEDED,
            is_blue and blue - 1 < _BLUE_SOURCES_NEEDED,
            is_black and black - 1 < 1,
            is_blue, is_black)


def _dd_edge_cycle_safe(obs: np.ndarray) -> bool:
    """Can Edge of Autumn's cycle eat one of our lands and still leave Thassa's
    Oracle's UU available — two untapped blue sources behind the sacrifice (or
    an LED covering the cast)? Post-Doomsday the alternative — waiting for the
    natural draw step to reach the next pile card — costs a turn, while
    drawing the pile down onto a board that can't cast Oracle strands it in
    hand until the deck-out. With no UU on the board at all the cycle is
    never worth it: the pile holds no lands, so only a drawn Petal or the
    turn's land drop can fix the mana, and every pile card should wait."""
    if _led_covers_oracle(obs):
        return True
    untapped_blue = _count_untapped_sources(obs, _BLUE_SOURCE_IDS)
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if _phased(obs, base) or obs[base + _OFF_IS_LAND] < 0.5:
            continue
        cid = _slot_card_idx(obs, base + _BF_CARD_OFF)
        if cid < 0 or cid in _SAC_LAST_LAND_IDS:
            continue
        untapped_blue_land = (cid in _BLUE_SOURCE_IDS
                              and obs[base + _OFF_IS_TAPPED] < 0.5)
        if untapped_blue - (1 if untapped_blue_land else 0) >= _BLUE_SOURCES_NEEDED:
            return True
    return False


def _dd_sacrifice_pick(obs: np.ndarray, cats, card_ids) -> int | None:
    """Sacrifice-cost pick for a Doomsday deck (Edge of Autumn's cycle): the
    land with the lowest _dd_land_sac_key — keep Thassa's Oracle's UU on the
    board above all (sacrificing the Island that was one of two blue sources
    leaves Oracle uncastable with the pile already drawn), then the only
    black source, then prefer an off-colour land. The menu doesn't say which
    offered land is tapped, so a land is treated as tapped only when EVERY
    copy we control is (the untapped-UU term is then moot for it). Protected
    utility lands are never offered up."""
    untapped_ids: set[int] = set()
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if _phased(obs, base) or obs[base + _OFF_IS_TAPPED] > 0.5:
            continue
        untapped_ids.add(_slot_card_idx(obs, base + _BF_CARD_OFF))
    best, best_key = None, None
    for i, c in enumerate(cats):
        if c != _CAT_SACRIFICE:
            continue
        cid = _action_card_id(card_ids, i)
        if cid in _SAC_LAST_LAND_IDS:
            continue
        key = _dd_land_sac_key(obs, cid, cid not in untapped_ids)
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


def _dd_oracle_mana_choice(obs: np.ndarray, cats, card_ids, own_main: bool) -> int | None:
    """Thassa's Oracle kill window with the UU not yet available: Lotus Petal
    is a sacrifice mana ability the engine offers only as an explicit MANA_U
    action at priority (lands auto-pay). When Oracle is in hand, the library
    is at the win size, it's our main phase on an empty stack and the engine
    is NOT offering the cast, crack a Petal for blue so the next query offers
    it. Lion's Eye Diamond's MANA_U is excluded — its cost discards the hand,
    Oracle included (its own draw-on-stack window is _led_crack_choice)."""
    if not own_main or not _stack_is_empty(obs):
        return None
    if not _hand_has_card(obs, _THASSAS_ORACLE_VOCAB_IDX):
        return None
    if _self_library_count(obs) > _ORACLE_WIN_LIBRARY:
        return None
    for i, c in enumerate(cats):
        if c == _CAT_CAST and _action_card_id(card_ids, i) == _THASSAS_ORACLE_VOCAB_IDX:
            return None
    for i, c in enumerate(cats):
        if c == _CAT_MANA_U and _action_card_id(card_ids, i) == _LOTUS_PETAL_VOCAB_IDX:
            return i
    return None


# The blue land the Doomsday pile carries on top when the board is short of
# blue sources (_dd_pile_wants_land): the basic Island ONLY — Wasteland-proof
# and a single copy, so the selection phase can never pick a second land. A
# Tundra fallback once got picked alongside it, and the extra land pushed the
# cyclers below Oracle.
_DD_PILE_LAND_IDS = (19,)


def _dd_pile_wants_land(obs: np.ndarray) -> bool:
    """Should the pile carry a blue land on top? With only two blue sources
    Edge of Autumn can never be cycled safely (its sacrifice breaks Oracle's
    UU), so the chain stalls on natural draws — a turn slower, and one
    Wasteland on a dual ends the game. A third source drawn off the top of
    the pile makes Edge cyclable and survives the Wasteland. Not needed when
    a blue land is already in hand (the land drop is coming anyway) or LED
    pays for Oracle by itself."""
    return (not _led_covers_oracle(obs)
            and _count_blue_sources(obs) < 3
            and _hand_direct_blue_sources(obs) == 0)


def _classify_top_library(cats, card_ids, want_land: bool = False):
    """Bucket the offered TOP_LIBRARY choices into
    (oracle_idx, led_idx, land_idx, free_draw_idxs, cantrip_idxs, filler_idxs).
    ``land_idx`` is the pile's blue land (see _DD_PILE_LAND_IDS) when
    ``want_land``; otherwise lands are filler."""
    oracle_i, led_i, draw_is, cantrip_is, filler_is = None, None, [], [], []
    land_by_id: dict[int, int] = {}
    for i, c in enumerate(cats):
        if c != _CAT_TOP_LIBRARY:
            continue
        cid = _action_card_id(card_ids, i)
        if cid == _THASSAS_ORACLE_VOCAB_IDX:
            if oracle_i is None:
                oracle_i = i
        elif cid == _LED_VOCAB_IDX:
            if led_i is None:
                led_i = i
        elif want_land and cid in _DD_PILE_LAND_IDS and cid not in land_by_id:
            land_by_id[cid] = i
        elif cid in _DD_FREE_DRAW_IDS:
            draw_is.append(i)
        elif cid in _DD_CANTRIP_IDS:
            cantrip_is.append(i)
        else:
            filler_is.append(i)
    land_i = None
    for cid in _DD_PILE_LAND_IDS:
        if cid in land_by_id:
            land_i = land_by_id[cid]
            break
    return oracle_i, led_i, land_i, draw_is, cantrip_is, filler_is


def _dd_return_pick(obs: np.ndarray, cats, card_ids, cat) -> int:
    """Pick a card to give back from hand (Brainstorm's put-back to the top,
    or London-mulligan bottoming — ``cat`` selects which menu), protecting
    the Doomsday combo pieces and threats. Return, in preference order: an
    excess land (while the hand still holds >= 3, the same threshold as the
    discard rule), then a cheap cantrip, then any other non-combo card, then
    free-draw cyclers (the cycle chain re-draws whatever sits on top, so they
    are the least-bad combo piece for the put-back case), then Lion's Eye
    Diamond and Doomsday, with Thassa's Oracle only as the forced last
    resort."""
    land_heavy = _hand_land_count(obs) >= 3
    tiers: list[list[int]] = [[], [], [], [], [], []]
    for i, c in enumerate(cats):
        if c != cat:
            continue
        cid = _action_card_id(card_ids, i)
        if cid == _THASSAS_ORACLE_VOCAB_IDX:
            tiers[5].append(i)
        elif cid in (_LED_VOCAB_IDX, _DOOMSDAY_VOCAB_IDX):
            tiers[4].append(i)
        elif cid in _DD_FREE_DRAW_IDS:
            tiers[3].append(i)
        elif cid in _DD_CANTRIP_IDS:
            tiers[1].append(i)
        elif cid in _LAND_VOCAB_IDS:
            tiers[0 if land_heavy else 2].append(i)
        else:
            tiers[2].append(i)
    for tier in tiers:
        if tier:
            return tier[0]
    return 0


def _doomsday_pile_pick(obs: np.ndarray, cats, card_ids) -> int:
    """Choose a card during Doomsday pile building (TOP_LIBRARY queries).

    A TOP_LIBRARY menu whose pending source is Brainstorm is its put-back
    (cards from HAND going on top), not pile building — that goes to
    _brainstorm_putback_pick so combo pieces are kept over junk.

    Doomsday resolves in two TOP_LIBRARY phases:

      (a) SELECTION — choose WHICH 5 cards to keep, from the whole library and
          graveyard (many options). Composition only; the order is fixed by the
          rearrange that follows: keep Thassa's Oracle, then every free-draw
          enabler (Street Wraith / Edge of Autumn), then cheap cantrips, then
          whatever fills the rest.

      (b) REARRANGE — order the (<= 5) kept cards. The engine fills the pile
          bottom-first (effect_rearrange_top_of_library.cpp), so the
          position-from-top being placed equals the number of remaining choices.

    Two pile plans:

    STANDARD, top -> bottom: [free draws..., Thassa's Oracle, rest...] with
    Oracle's position ADAPTIVE — directly under however many free-draw enablers
    the pile actually has (the old plan hardcoded Oracle 3rd behind two Street
    Wraiths, which stalls a list running only one Wraith, e.g. wubg_doomsday).
    The kill: each drawn enabler is immediately cycled (the post-Doomsday
    cycling rules), chaining down to Oracle; natural draw steps and any cantrip
    filler drain the remainder until the library hits the Oracle-win size.
    When the board is short of blue sources (_dd_pile_wants_land) a blue land
    sits on the very top: [land, free draws..., Thassa's Oracle, rest...] —
    the first draw delivers it, the next land drop makes Edge of Autumn
    cyclable and the kill Wasteland-proof.

    BRAINSTORM SAME-TURN (bs_line: Brainstorm in hand, LED offered, and a
    free-draw cycler already in hand), top -> bottom:
    [free draws..., Lion's Eye Diamond, Thassa's Oracle, rest...].
    Brainstorm lifts the top three pile cards (cyclers + LED), puts back two
    junk cards, LED is cast for 0, and the in-hand cycler chain eats the
    put-backs down to Oracle: the final cycle's draw trigger is the LED crack
    window (_led_crack_choice) — sac LED for UUU with Oracle still in the
    library, draw Oracle, cast it off the floating mana the same turn. The
    extra cycler in HAND is load-bearing: Brainstorm's two put-backs cost two
    draws, so a pile-only chain always comes up exactly one draw short.
    """
    if _slot_card_idx(obs, _PENDING_DECISION_START) == _BRAINSTORM_VOCAB_IDX:
        return _dd_return_pick(obs, cats, card_ids, _CAT_TOP_LIBRARY)

    # The land plan applies only while Doomsday itself is resolving (its
    # selection and rearrange menus); a Ponder rearrange keeps lands as filler.
    want_land = (_slot_card_idx(obs, _PENDING_DECISION_START) == _DOOMSDAY_VOCAB_IDX
                 and _dd_pile_wants_land(obs))
    oracle_i, led_i, land_i, draw_is, cantrip_is, filler_is = \
        _classify_top_library(cats, card_ids, want_land)
    bs_line = (led_i is not None
               and _hand_has_card(obs, _BRAINSTORM_VOCAB_IDX)
               and _hand_free_draw_count(obs) >= 1)
    if not bs_line and led_i is not None:
        # Without the Brainstorm line LED is ordinary filler, as before.
        filler_is = sorted(filler_is + [led_i])
        led_i = None
    n_choices = sum(1 for c in cats if c == _CAT_TOP_LIBRARY)

    # (a) SELECTION: Oracle, LED (bs_line only), the blue land, free draws,
    # cantrips, filler.
    if n_choices > _DD_PILE_SIZE:
        if oracle_i is not None:
            return oracle_i
        if led_i is not None:
            return led_i
        if land_i is not None:
            return land_i
        if draw_is:
            return draw_is[0]
        if cantrip_is:
            return cantrip_is[0]
        return 0

    # Small rearrange with no combo pieces (Ponder / Brainstorm): legacy behaviour.
    if oracle_i is None and not draw_is and led_i is None and land_i is None:
        return 0

    # (b) REARRANGE, bottom-first: position-from-top == remaining choice count.
    # The blue land (if any) is the top card; Oracle goes right under the
    # (remaining) free-draw enablers — with LED slotted between them on the
    # bs_line — positions above get the enablers, positions below get
    # filler-then-cantrips (cantrips shallower than dead filler, since a drawn
    # one digs further).
    pos = n_choices
    land_off = 1 if land_i is not None else 0
    led_pos = (land_off + len(draw_is) + 1) if led_i is not None else 0
    oracle_slot = land_off + len(draw_is) + (2 if led_i is not None else 1)
    oracle_pos = oracle_slot if oracle_i is not None else 0
    # Deepest combo-piece position: below it the pile is bury-order. With
    # neither Oracle nor LED among the remaining picks it is 0, i.e. every
    # position buries (the legacy oracle-absent behaviour).
    if oracle_i is not None:
        combo_floor = oracle_slot
    elif led_i is not None:
        combo_floor = led_pos
    else:
        combo_floor = 0
    if pos == 1 and land_i is not None:
        return land_i
    if pos == led_pos and led_i is not None:
        return led_i
    if pos == oracle_pos and oracle_i is not None:
        return oracle_i
    if pos > combo_floor:                    # below Oracle: bury the worst first
        if filler_is:
            return filler_is[0]
        if cantrip_is:
            return cantrip_is[0]
        if draw_is:
            return draw_is[0]
        if oracle_i is not None:
            return oracle_i
        return land_i if land_i is not None else 0
    # Above Oracle/LED: the free-draw chain.
    if draw_is:
        return draw_is[0]
    if cantrip_is:
        return cantrip_is[0]
    if oracle_i is not None:
        return oracle_i
    return land_i if land_i is not None else 0


# ── Reanimation / lands-combo helpers (card-id gated, deck-agnostic) ────────
# Don't dredge (skip the draw, mill 3) once the library is this small — the
# lands deck grinds long games and a draw/deck-out loss is never acceptable.
_LOAM_DREDGE_MIN_LIBRARY = 10


def _reanimation_target_choice(cats, card_ids, ctrl_arr, num_choices: int) -> int | None:
    """Target pick for Reanimate / Animate Dead: the highest-mana-value creature
    card offered, tie-broken toward self-owned (our own fatty beats stealing an
    equal one). Fixes the generic target rule's prefer-opponent polarity, which
    is meant for removal and is exactly backwards for reanimation. Returns None
    when no TARGET action is offered."""
    best_i, best_key = None, None
    for i in range(num_choices):
        if cats[i] != _CAT_TARGET:
            continue
        cid = _action_card_id(card_ids, i)
        if cid < 0:
            continue  # "no target" slot — never preferred
        key = (_card_mana_value(cid), 1 if ctrl_arr[i] > 0.5 else 0)
        if best_key is None or key > best_key:
            best_i, best_key = i, key
    return best_i


def _keep_legend_choice(obs: np.ndarray, cats, card_ids, num_choices: int) -> int | None:
    """Legend-rule keep pick (704.5j) for duplicate Dark Depths: keep the copy
    with the FEWEST 'other' counters — the freshly-cloned Thespian's Stage at
    zero ice counters — so its "no ice counters" trigger fires and makes Marit
    Lage. Keeping the 10-counter original fizzles the combo. The duplicates
    share one card id, so they are told apart through the action's slot_ref
    into the battlefield slots' counter float. Other legend conflicts return
    None (generic behaviour)."""
    best_i, best_ct = None, None
    for i in range(num_choices):
        if (cats[i] != _CAT_KEEP_LEGEND
                or _action_card_id(card_ids, i) != _DARK_DEPTHS_VOCAB_IDX):
            continue
        ref = _action_slot_ref(obs, i)
        if 0 <= ref < 2 * _PERM_A_SLOTS:
            ct = float(obs[_BF_START + ref * _BF_SLOT_SIZE + _OFF_OTHER_COUNTERS])
        else:
            ct = float("inf")  # unresolvable ref — never preferred
        if best_ct is None or ct < best_ct:
            best_i, best_ct = i, ct
    return best_i


# ── GREEDY/HEURISTIC fruitless-activation stall guard ───────────────────────
# Action categories a resolution-time zone search emits (src/components/
# ability.cpp search_zone): SEARCH_LIBRARY for library searches, TOP_LIBRARY for
# search-to-top-of-library, CHOOSE_CARD for non-library picks (e.g. Aether
# Vial's put-a-creature-from-hand). When such a menu offers ONLY the
# "Fail to find" slot (a single choice carrying the null card-id sentinel — the
# engine builds it from Entity(0), so it decodes negative, see
# _EXPLORE_SEARCH_FAIL_WEIGHT), the search cannot succeed. If the resolving
# item on top of the stack is a self-controlled ability, re-activating its
# source card right now would just fail again, so the hard tier records the
# fail against the card id and skips its ACTIVATE for the rest of the TURN —
# not the game, because eligibility can change in ways the obs can't see
# (Aether Vial's search keys on its charge-counter count, which isn't in the
# state vector: a fail at 0 counters says nothing about next turn's 1). After
# _FRUITLESS_ACTIVATION_LIMIT failed tries the card is dropped for the game:
# hard-scripted bw_dnt once activated Vial into a guaranteed fail-to-find
# every turn (~60 activations) to the decision cap — the one non-decisive
# game of a 441-game fuzz campaign — so retries must be bounded.
_SEARCH_MENU_CATS = frozenset((_CAT_SEARCH, _CAT_TOP_LIBRARY, _CAT_CHOOSE_CARD))
_FRUITLESS_ACTIVATION_LIMIT = 3


def _greedy_action(obs: np.ndarray, num_choices: int,
                   fruitless: set[int] | None = None,
                   hold_casts: frozenset | set | None = None,
                   hold_activations: frozenset | set | None = None,
                   doomsday: bool | None = None) -> int:
    """
    Rule-based agent for test_minimal.dk (blue/red fetch-land deck).
    Works correctly for either Player A or Player B because the observation
    is always emitted from the priority player's perspective.

      - Never blocks (confirms immediately)
      - Attacks with every eligible creature each combat
      - Selects target 0 (opponent player or first offered spell/permanent)
      - Searches library: always finds the first offered card (never fails to find)
      - Casts every spell the moment it becomes affordable
      - Plays the first available land
      - Activates non-mana abilities (fetch lands, Wasteland destroy) during main phase
      - Taps mana during main phases, preferring the color the hand needs most
        (blue for Flying Men, Delver, Ponder, Daze, Counterspell, Air Elemental;
         red for Dragon's Rage Channeler, Lightning Strike)
      - Passes priority otherwise

    Action categories are stored in obs[ACT_CATS_START:] normalised by ACTION_CATEGORY_MAX.
    """
    cats     = np.round(obs[ACT_CATS_START:ACT_CATS_START + num_choices] * ACTION_CATEGORY_MAX).astype(int)
    card_ids = obs[ACT_IDS_START  : ACT_IDS_START + MAX_ACTIONS]
    ctrl_arr = obs[ACT_CTRL_START : ACT_CTRL_START + MAX_ACTIONS]

    in_main_phase = obs[_STEP_FIRST_MAIN_IDX] > 0.5 or obs[_STEP_SECOND_MAIN_IDX] > 0.5
    own_main = in_main_phase and obs[_IS_ACTIVE_IDX] > 0.5
    # Are we piloting Doomsday? The caller's deck-name identification when it
    # has one (``doomsday``); the hand-content heuristic only for unnamed
    # decks — that heuristic also fires on Thoughtseize / Dark Ritual / Lotus
    # Petal hands (reanimator, bug, wrb_energy), which must NOT inherit the
    # Doomsday fetch and pile rules.
    dd_deck = _is_doomsday_deck(obs) if doomsday is None else doomsday
    # Doomsday state shared by the rules below. Once the five-card pile is laid
    # every library touch is part of the kill: the whole chain (cycles, the
    # LED crack, post-pile cantrips) runs in OUR MAIN PHASE only — Oracle is a
    # creature, so mana floated at the draw step is gone before it can be
    # cast — and the generic library-touching rules (surveil, activations,
    # fetches) stand down. With Oracle already in hand at the win size every
    # further draw is pointless and one closer to decking, so digging stops
    # and the main-phase cast rule wins.
    dd_pile_laid = _self_library_count(obs) <= _DD_PILE_SIZE and dd_deck
    oracle_kill_ready = (_hand_has_card(obs, _THASSAS_ORACLE_VOCAB_IDX)
                         and _self_library_count(obs) <= _ORACLE_WIN_LIBRARY)

    # 0a. Sideboarding: the scripted agent never sideboards, so it takes Done as
    # soon as the deck is balanced (Done is index 0 whenever it is offered). On the
    # IN-FIRST menu that is every decision it can reach, since it never opens a
    # swap. The fallback matters only if it inherits a phase mid-swap (Done is
    # withheld while a cut is outstanding): pick the first offered move, which is
    # necessarily the balancing cut, so the phase still terminates instead of
    # falling through to the combat/casting logic below with a sideboard menu in
    # hand.
    if any(c == _CAT_SB_DONE for c in cats):
        return 0
    if any(c in (_CAT_SB_IN, _CAT_SB_OUT) for c in cats):
        return next(i for i, c in enumerate(cats) if c in (_CAT_SB_IN, _CAT_SB_OUT))

    # 0. Mulligan: always keep — return the first non-mulligan action (the keep action)
    if any(c == _CAT_MULLIGAN for c in cats):
        for i, c in enumerate(cats):
            if c != _CAT_MULLIGAN:
                return i

    # 1. Confirm blockers immediately — never block
    for i, c in enumerate(cats):
        if c == _CAT_CONF_BLK:
            return i

    # 2. Attacker selection: select until all eligible creatures are attacking, then confirm.
    #    The game re-offers already-attacking creatures as SEL_ATK (for deselection), so we
    #    must check the battlefield state rather than blindly picking SEL_ATK every time.
    if any(c == _CAT_SEL_ATK for c in cats):
        if _all_eligible_creatures_attacking(obs):
            for i, c in enumerate(cats):
                if c == _CAT_CONF_ATK:
                    return i
        else:
            for i, c in enumerate(cats):
                if c == _CAT_SEL_ATK:
                    return i

    # 3. Confirm attack declaration (fallback, e.g. no eligible attackers)
    for i, c in enumerate(cats):
        if c == _CAT_CONF_ATK:
            return i

    # 4. Select target — prefer non-self-controlled targets.
    #    ctrl_arr[i] == 1.0 means self-controlled; 0.0 = opponent permanent/spell;
    #    _ACTION_CTRL_NULL (-0.03125) = player target (also non-self).
    #    The C++ game sorts targets opponent-first so action 0 is usually correct,
    #    but guard against accidentally targeting own spells/permanents.
    for i, c in enumerate(cats):
        if c == _CAT_TARGET and ctrl_arr[i] < 0.5:
            return i
    # Fallback: all targets are self-controlled — return first (shouldn't happen in practice).
    for i, c in enumerate(cats):
        if c == _CAT_TARGET:
            return i

    # 5. Search library — action 0 = fail to find; action 1+ = actual cards.
    #    For Doomsday pile building: prefer Thassa's Oracle, then draw spells.
    #    For Personal Tutor: prefer Doomsday.
    #    For fetch lands: pick action 1 (first land found).
    if any(c == _CAT_SEARCH for c in cats):
        # Prefer Thassa's Oracle when searching (Doomsday pile building)
        for i, c in enumerate(cats):
            if c == _CAT_SEARCH and _action_card_id(card_ids, i) == _THASSAS_ORACLE_VOCAB_IDX:
                return i
        # Prefer Doomsday when searching (Personal Tutor)
        for i, c in enumerate(cats):
            if c == _CAT_SEARCH and _action_card_id(card_ids, i) == _DOOMSDAY_VOCAB_IDX:
                return i
        # Doomsday deck land search: fetch the colours the combo needs
        # (see _dd_fetch_pick) instead of the first land offered.
        if dd_deck:
            pick = _dd_fetch_pick(obs, cats, card_ids)
            if pick is not None:
                return pick
        return 1 if num_choices > 1 else 0

    # 5b. Paying costs (tapping lands for mana during spell/ability payment,
    #     or the pitch pick of an alternative cost such as Force of Will's
    #     "exile a blue card"). First available option, except that a
    #     Doomsday deck pays with lands before Petal / LED and pitches its
    #     combo pieces last (Thassa's Oracle never while anything else is
    #     offered).
    if any(c == _CAT_PAYING for c in cats):
        if dd_deck:
            best, best_key = 0, None
            for i, c in enumerate(cats):
                if c != _CAT_PAYING:
                    continue
                cid = _action_card_id(card_ids, i)
                key = (cid == _THASSAS_ORACLE_VOCAB_IDX,
                       cid in _DD_COMBO_PIECE_IDS,
                       cid in _DD_PAY_LAST_IDS,
                       cid in _DD_CANTRIP_IDS)
                if best_key is None or key < best_key:
                    best, best_key = i, key
            return best
        return 0

    # 5c. Delve exile count (a CHOOSE_X menu whose actions carry the delve spell's card
    #     id; a plain X-cost ladder emits the null sentinel). Exile the maximum — the
    #     cheapest cast, matching the old auto-payer (and for Murktide Regent also the
    #     biggest body). Every offered count is payable (the engine constrains the menu).
    if (all(c == _CAT_CHOOSE_X for c in cats)
            and card_ids[0] > _ACTION_CARD_ID_NULL + 0.01):
        return num_choices - 1

    # 5d. Zone-change card pick (CHOOSE_CARD). An optional search-style pick (Aether
    #     Vial's put-a-creature-from-hand) leads with a null-id "Fail to find" slot —
    #     take the first REAL card instead, so Vial actually deploys a creature when
    #     one is eligible (declining was strictly worse: the guard below never
    #     blacklists a menu that has a real pick, so the agent just failed forever).
    #     Delve per-card exile picks are all-real menus, so they still take slot 0
    #     (the first pick may eat a card a smarter agent would keep, or skip an
    #     instant/sorcery Murktide would count — kept simple at this tier).
    if any(c == _CAT_CHOOSE_CARD for c in cats):
        # A surveil menu (keep on top / put in graveyard) while piloting
        # Doomsday keeps a combo piece — Consider and Undercity Sewers
        # surveils binned two Doomsdays and a Dark Ritual in one 30-game
        # sample — and keeps EVERY card once the pile is laid (a Sewers
        # surveil once binned Thassa's Oracle off the top of the pile).
        if dd_deck and num_choices == 2 and any(c == _CAT_TOP_LIBRARY for c in cats):
            seen = {_action_card_id(card_ids, i) for i in range(num_choices)}
            if dd_pile_laid or seen & _DD_COMBO_PIECE_IDS:
                for i, c in enumerate(cats):
                    if c == _CAT_TOP_LIBRARY:
                        return i
        for i, c in enumerate(cats):
            if c == _CAT_CHOOSE_CARD and _action_card_id(card_ids, i) >= 0:
                return i
        return 0

    # 5e. Optional "may" trigger (OPTIONAL_YESNO, 0=Decline / 1=Accept). The tier's
    #     default is to decline (the final fallback returns 0), but Aether Vial's
    #     upkeep charge-counter trigger must be ACCEPTED or the Vial sits at 0
    #     counters forever and never deploys a creature. The resolving trigger is on
    #     top of the stack while the yes/no is asked, so the source is identifiable.
    if (num_choices == 2 and all(c == _CAT_YESNO for c in cats)
            and obs[_STACK_START] > 0.5 and obs[_STACK_START + 2] < 0.5
            and _slot_card_idx(obs, _STACK_START + 1) == _AETHER_VIAL_VOCAB_IDX):
        return 1

    # 6. Cast spells.
    #    Counter spells (Counterspell, Daze, Force of Will) require an opponent's spell
    #    on the stack; skip them when the stack holds only own spells or is empty.
    opponent_spell_on_stack = _opponent_has_spell_on_stack(obs)

    # Priority: if opponent has a spell on stack and we have UU, cast Counterspell first
    # (unless the caller's triage put it on hold — a cantrip not worth countering).
    if (opponent_spell_on_stack and int(round(obs[_BLUE_POOL_IDX] * 10)) >= 2
            and not (hold_casts and _COUNTERSPELL_VOCAB_IDX in hold_casts)):
        for i, c in enumerate(cats):
            if c == _CAT_CAST and _action_card_id(card_ids, i) == _COUNTERSPELL_VOCAB_IDX:
                return i

    # Crack Lion's Eye Diamond in the Doomsday kill window — a self draw is resolving,
    # Oracle is the known top card, and library <= 2 — to float UUU for the Thassa's
    # Oracle win. LED is offered as instant-speed MANA_* actions at priority (unlike
    # ordinary mana sources, which stay hidden and auto-pay), so it is picked here.
    if own_main:
        led = _led_crack_choice(obs, cats, card_ids)
        if led is not None:
            return led

    # --- Doomsday deck rules ---
    has_doomsday_in_hand = _hand_has_card(obs, _DOOMSDAY_VOCAB_IDX)

    # Win the game: cast Thassa's Oracle as soon as the library is small enough
    # that its devotion-to-blue (>= 2 from its own UU) meets/exceeds the library
    # size. "Mana is there" == the engine is offering Oracle as a legal cast;
    # when it isn't, a Lotus Petal cracked for blue can make it so.
    if _self_library_count(obs) <= _ORACLE_WIN_LIBRARY:
        for i, c in enumerate(cats):
            if c == _CAT_CAST and _action_card_id(card_ids, i) == _THASSAS_ORACLE_VOCAB_IDX:
                return i
        petal = _dd_oracle_mana_choice(obs, cats, card_ids, own_main)
        if petal is not None:
            return petal

    # Cast Doomsday once Thassa's Oracle's UU will exist by kill time
    # (_dd_blue_for_oracle) and the opponent's board can't kill us at half
    # life before the kill lands (_dd_clock_allows_doomsday). Firing it
    # earlier just empties our library into a deck-out or halves us into
    # lethal, so we hold it until the mana to win is reachable and the clock
    # allows.
    # Never a SECOND Doomsday once the pile is laid: it re-piles the same
    # five cards for another half of our life (one game went 6 -> 2 -> 1
    # with Oracle in hand and the kill one cycle away).
    dd_ready = (_dd_blue_for_oracle(obs) and _dd_clock_allows_doomsday(obs)
                and not dd_pile_laid)
    for i, c in enumerate(cats):
        if c == _CAT_CAST and _action_card_id(card_ids, i) == _DOOMSDAY_VOCAB_IDX:
            if dd_ready:
                return i
            # else hold Doomsday until the blue mana to cast Oracle is in play

    # Only cast Dark Ritual if Doomsday is also in hand and we're ready to combo
    # (don't waste ritual mana without the payoff online). Main phase of our
    # own turn only: the mana pool empties at step boundaries, so a ritual cast
    # at upkeep (its first legal window — it's an instant) strands the BBB
    # before the sorcery-speed Doomsday can ever be cast with it.
    for i, c in enumerate(cats):
        if c == _CAT_CAST and _action_card_id(card_ids, i) == _DARK_RITUAL_VOCAB_IDX:
            if has_doomsday_in_hand and dd_ready and own_main:
                return i
            # else skip it

    for i, c in enumerate(cats):
        if c == _CAT_CAST:
            cid = _action_card_id(card_ids, i)
            if hold_casts and cid in hold_casts:
                continue  # caller-computed hold (e.g. Reanimate with no fatty
                          # in a graveyard yet — see _heuristic_action)
            if cid in _COUNTER_SPELL_VOCAB_IDS and not opponent_spell_on_stack:
                continue
            if cid == _DARK_RITUAL_VOCAB_IDX:
                continue  # handled above
            if cid == _DOOMSDAY_VOCAB_IDX:
                continue  # handled above (held until blue mana for Oracle is online)
            if cid == _THASSAS_ORACLE_VOCAB_IDX:
                continue  # only cast Oracle as the wincon (handled above)
            if cid in _DD_FREE_DRAW_IDS and dd_pile_laid:
                continue  # post-pile the cyclers are draws, never casts: a
                          # hard-cast Edge of Autumn searches the library and
                          # shuffles the pile away (the cycle rules below)
            if (dd_deck and has_doomsday_in_hand
                    and cid not in _DD_PETAL_WORTHY_IDS
                    and _count_controlled(obs, _LOTUS_PETAL_VOCAB_IDX) > 0
                    and _count_untapped_mana_sources(obs) < _card_mana_value(cid)):
                continue  # keep the Petals for Doomsday: a spell the lands
                          # alone can't pay would crack one
            if cid == _BRAINSTORM_VOCAB_IDX and (
                    _self_stack_has_card(obs, _DOOMSDAY_VOCAB_IDX)
                    or (has_doomsday_in_hand
                        and _self_library_count(obs) > _DD_PILE_SIZE
                        and _dd_combo_ready(obs))):
                continue  # hold Brainstorm: post-Doomsday it lifts the top of
                          # the pile (the same-turn kill's draw engine). The
                          # stack check matters — casting it in response to our
                          # own Doomsday resolves it FIRST, drawing from the
                          # un-piled library. The hold applies only while the
                          # combo is actually startable (black + UU on board,
                          # pile not yet laid): a mana-screwed hand digs with
                          # Brainstorm instead of sitting on it, and a spare
                          # Doomsday in hand after the pile is laid must not
                          # freeze the draw engine.
            if cid in _DD_CANTRIP_IDS and (
                    _self_library_count(obs) < (_DD_CANTRIP_MIN_LIBRARY[cid]
                                                if dd_deck else 3)
                    or _controls_card(obs, _THASSAS_ORACLE_VOCAB_IDX)):
                continue  # never deck ourselves with a cantrip: drawing from
                          # an empty library is a loss, and with Oracle already
                          # on our battlefield its win trigger is pending — a
                          # greedy Brainstorm here resolves FIRST and kills us
                          # in response to our own winning trigger
            if (cid in _DD_CANTRIP_IDS and dd_pile_laid
                    and (not own_main or oracle_kill_ready
                         or (_count_untapped_sources(obs, _BLUE_SOURCE_IDS) - 1
                             < _BLUE_SOURCES_NEEDED))):
                continue  # post-Doomsday a cantrip is a kill-chain step: our
                          # main phase only, never once Oracle is in hand at
                          # the win size (a draw-step Consider once milled the
                          # last card with Oracle already in hand), and never
                          # tapping the UU that casts Oracle this turn (two
                          # upkeep Brainstorms off both Underground Seas once
                          # left Oracle in hand with no blue mana)
            if (cid == _GREEN_SUNS_ZENITH_VOCAB_IDX
                    and _count_untapped_mana_sources(obs) < _GSZ_MIN_MANA_SOURCES):
                continue  # hold GSZ until it can pay X>=1 (avoids the X=0 Dryad Arbor cast)
            if cid == _SOLITUDE_VOCAB_IDX and not _opponent_has_creature(obs):
                continue  # Solitude's value is its ETB exile — don't hard-cast it
                          # as a vanilla 3/2 when there's no opposing creature to hit
            return i

    # 7. Play land (may unlock mana for a spell next query)
    for i, c in enumerate(cats):
        if c == _CAT_LAND:
            return i

    # 8. Activate non-mana abilities:
    #    - Cycling (Street Wraith, Edge of Autumn): cycle as part of the combo
    #    - Fetch lands: tap + pay 1 life + sacrifice → search for land
    #    - Wasteland: tap + sacrifice → destroy target nonbasic land

    # Cycle Street Wraith (pay 2 life, draw) only once we're going off: a small
    # library means we are drawing through the Doomsday pile toward Thassa's
    # Oracle. Cycling earlier would dump the Wraiths we need for the free-draw
    # chain and pull Oracle out of the pile, breaking the combo. Never pay the
    # 2 life below 3 life, so the dig can't kill us before Oracle resolves.
    # Only cycle on an empty stack: stacked cycles resolve LIFO, scrambling the
    # pile's draw order, and an unresolved LED spell must hit the battlefield
    # before the final cycle's draw opens its crack window.
    if (1 <= _self_library_count(obs) <= _DD_PILE_SIZE
            and _self_life(obs) >= _STREET_WRAITH_MIN_LIFE
            and _stack_is_empty(obs)
            and own_main and not oracle_kill_ready):
        for i, c in enumerate(cats):
            if c == _CAT_ACTIVATE and _action_card_id(card_ids, i) == _STREET_WRAITH_VOCAB_IDX:
                return i

    # Cycle Edge of Autumn (sac a land, draw a card) only once we're going off,
    # like Street Wraith: pre-Doomsday every land is needed for BBB + UU. Only
    # on an empty stack (its draw is the LED crack window on the same-turn
    # line, so LED must have resolved) and never off an empty library (that
    # draw would be the game loss). Only when some land can go without
    # breaking Oracle's UU (_dd_edge_cycle_safe; _dd_sacrifice_pick then
    # chooses it) — otherwise the natural draw step reaches the next pile
    # card a turn later with the kill mana intact.
    if (1 <= _self_library_count(obs) <= _DD_PILE_SIZE
            and own_main and not oracle_kill_ready
            and _dd_edge_cycle_safe(obs)):
        for i, c in enumerate(cats):
            if (c == _CAT_ACTIVATE and _action_card_id(card_ids, i) == _EDGE_OF_AUTUMN_VOCAB_IDX
                    and _stack_is_empty(obs)):
                return i

    # Always activate The One Ring's {T} draw ability whenever able — each use draws
    # a card per burden counter (pure card advantage), and it only taps once per turn.
    # Ungated by phase so it still fires on turns the agent has nothing else to do.
    # Never off an empty library: that draw is the game loss (with Thassa's
    # Oracle in hand the win at library 0 is handled above — just don't draw).
    library_empty = _self_library_count(obs) == 0
    if not library_empty:
        for i, c in enumerate(cats):
            if c == _CAT_ACTIVATE and _action_card_id(card_ids, i) == _THE_ONE_RING_VOCAB_IDX:
                return i

    if in_main_phase:
        for i, c in enumerate(cats):
            if c == _CAT_ACTIVATE:
                cid = _action_card_id(card_ids, i)
                if (library_empty or dd_pile_laid) and cid != _WASTELAND_VOCAB_IDX:
                    continue  # a draw-shaped activation (Clue, planeswalker
                              # card advantage) off an empty library loses on
                              # the spot; after Doomsday has laid the pile a
                              # fetch would shuffle it away and a Clue crack
                              # once tapped both blue lands the turn Oracle
                              # was due — the kill chain above is the only
                              # thing that touches the library or the mana
                if cid == _WASTELAND_VOCAB_IDX and not _opponent_has_nonbasic_land(obs):
                    continue  # no opponent nonbasic land to target — skip
                if cid == _STREET_WRAITH_VOCAB_IDX:
                    continue  # only ever cycled via the gated combo rule above
                if cid == _EDGE_OF_AUTUMN_VOCAB_IDX:
                    continue  # likewise: only via its stack-empty rule above,
                              # so its draw can serve as the LED crack window
                if cid == _KEEN_EYED_CURATOR_VOCAB_IDX and not _stack_is_empty(obs):
                    continue  # don't pile copies onto the stack (each extra one
                              # fizzles when its graveyard target is already exiled)
                if cid == _CANDELABRA_VOCAB_IDX and _tapped_multi_mana_lands(obs) == 0:
                    continue  # nothing worth untapping — don't tap Candelabra for X=0
                if hold_activations and cid in hold_activations:
                    continue  # caller-computed hold (e.g. Thespian's Stage with
                              # no Dark Depths to copy — see _heuristic_action)
                if fruitless and cid in fruitless:
                    continue  # stall guard: this card's search found NOTHING
                              # earlier this game (see _SEARCH_MENU_CATS) —
                              # re-activating it would just fail to find again
                return i

    # 9. (Removed - mana abilities no longer offered during normal priority in machine mode;
    #     mana is tapped automatically during cost payment via PAYING_COSTS.)

    # 10. Doomsday pile building (TOP_LIBRARY): select the 5 pile cards, then
    #     order them [free draws..., (LED,) Thassa's Oracle, rest...] for the
    #     cycle-chain kill; a Brainstorm put-back menu instead protects combo
    #     pieces. See _doomsday_pile_pick for both pile plans.
    if any(c == _CAT_TOP_LIBRARY for c in cats):
        return _doomsday_pile_pick(obs, cats, card_ids)

    # 10b. Dig choice (Once Upon a Time): pick action 1 (first matching card) if available.
    if any(c == _CAT_DIG for c in cats):
        return 1 if num_choices > 1 else 0

    # 10c. X-value ladder (e.g. Green Sun's Zenith): the engine offers the whole
    #      query as a contiguous run of OTHER_CHOICE actions "X = 0".."X = max".
    #      Pay the maximum affordable X so GSZ fetches the biggest creature it can,
    #      never settling for X=0. The only other all-OTHER query in these decks is
    #      Sylvan Library's pay/return prompt, whose last option ("put on top of
    #      library") is a safe, no-life-loss pick — so always taking the last
    #      option here is correct for both.
    if num_choices >= 2 and all(c == _CAT_OTHER for c in cats):
        return num_choices - 1

    # 11. Other choice (Sylvan Library pay/return, unless costs): pick randomly.
    other_idxs = [i for i, c in enumerate(cats) if c == _CAT_OTHER]
    if other_idxs:
        return random.choice(other_idxs)

    # 12. Surveil / Delver reveal: both use two PASS_PRIORITY (0) choices whose
    #     source entity is the same library card (non-null, equal card IDs).
    #     For surveil: action 1 = put in graveyard (good for delirium and Murktide).
    #     For Delver's PeekAndReveal: action 1 = reveal (safe; triggers transform
    #     only if the top card is an instant or sorcery, which is desirable).
    if num_choices == 2 and all(c == _CAT_PASS for c in cats[:2]):
        id0, id1 = card_ids[0], card_ids[1]
        if id0 > _ACTION_CARD_ID_NULL + 0.01 and abs(id1 - id0) < 0.001:
            return 1

    # Default: pass priority
    return 0


# ── HEURISTIC helpers (single-ply; combat sim is the only exact lookahead) ──
#
# These operate on the readable dict from decode.decode_game_state(obs[:STATE_SIZE]).
# Creatures are perm dicts that carry a "power" key; lands/other permanents do not.

# Board-evaluation weights. Anchored to the same intuitions as env's reward
# shaping (card advantage compounds, so it is weighted highest; life is a large
# pool so each point is cheap).
_W_LIFE  = 1.0
_W_POWER = 2.0
_W_TOUGH = 1.0
_W_CARDS = 3.0
_W_BOARD = 2.0
_W_MANA  = 0.5

# Opponent life at or below which a burn spell should go to the face.
_BURN_FACE_LIFE = 3


def _creatures(perms: list) -> list:
    """Creature permanents (those carrying P/T) from a decoded battlefield list."""
    return [p for p in perms if "power" in p]


def evaluate(g: dict) -> float:
    """Static board score from the priority player's perspective (higher = better)."""
    me, opp = g["self"], g["opponent"]
    my_cr, op_cr = _creatures(g["self_battlefield"]), _creatures(g["opp_battlefield"])
    my_lands = sum(1 for p in g["self_battlefield"] if p.get("is_land"))
    op_lands = sum(1 for p in g["opp_battlefield"] if p.get("is_land"))
    return (
        _W_LIFE  * (me["life"] - opp["life"])
        + _W_POWER * (sum(p["power"] for p in my_cr) - sum(p["power"] for p in op_cr))
        + _W_TOUGH * (sum(p["toughness"] for p in my_cr) - sum(p["toughness"] for p in op_cr))
        + _W_CARDS * (me["hand_count"] - opp["hand_count"])
        + _W_BOARD * (len(my_cr) - len(op_cr))
        + _W_MANA  * (my_lands - op_lands)
    )


def _should_attack(ap: int, at: int, opp_blockers: list, alpha: bool) -> bool:
    """Single-ply attack decision for a vanilla attacker (P/T only).

    Attack unless an untapped opposing blocker would kill us while surviving
    (a pure loss).  Even trades and free hits are allowed.  ``alpha`` forces the
    attack when the team is lethal this turn.
    """
    if alpha:
        return True
    if not opp_blockers:
        return True
    for b in opp_blockers:
        if b["power"] >= at and b["toughness"] > ap:
            return False  # we die, the blocker lives — don't attack into it
    return True


def _lookup_pt(perms: list, card_idx: int, *, exclude_attacking=False,
               exclude_blocking=False, require_untapped=False, require_unsick=False):
    """First creature matching card_idx under the given filters → (power, toughness)."""
    for p in perms:
        if p.get("card_idx") != card_idx or "power" not in p:
            continue
        if require_untapped and p.get("tapped"):
            continue
        if require_unsick and p.get("summoning_sick"):
            continue
        if exclude_attacking and p.get("attacking"):
            continue
        if exclude_blocking and p.get("blocking"):
            continue
        return p["power"], p["toughness"]
    return None


def _best_opp_creature_target(g: dict, cats, card_ids, ctrl_arr, num_choices: int) -> int | None:
    """Among SELECT_TARGET actions, the opponent creature with the highest power.

    Robust to the action controller encoding (self ~= 1.0, opponent permanent ~= 0.0,
    a player / "no target" slot is the negative null sentinel): opp-controlled entity
    targets are the ones with ctrl in (~0, 0.5). Returns None if no opp creature is
    offered, so the caller can fall back to its usual pick.
    """
    best_i, best_pow = None, None
    for i in range(num_choices):
        if cats[i] != _CAT_TARGET:
            continue
        ctrl = ctrl_arr[i]
        if ctrl > 0.5 or ctrl < -1e-4:
            continue  # self-controlled, or a player / "no target" slot (negative null)
        pt = _lookup_pt(g["opp_battlefield"], _action_card_id(card_ids, i))
        power = pt[0] if pt else 0
        if best_pow is None or power > best_pow:
            best_i, best_pow = i, power
    return best_i


def _wrath_energy_plan(g: dict) -> tuple[int, int, int]:
    """Best energy amount Y for Wrath of the Skies, with its board impact.

    Wrath destroys EACH artifact, creature, and enchantment with MV <= Y on
    BOTH sides (lands and planeswalkers are immune). Evaluate every candidate
    Y — the distinct MVs of the opponent's destroyable permanents, plus 0 —
    and pick the one maximizing (opponent permanents destroyed - own
    permanents destroyed), tie-broken by destroyed-power differential and
    then by the SMALLEST Y (cheaper X, and no collateral own losses beyond
    what the score already accepted). Token card ids cost-decode to MV 0,
    matching their real mana value.

    Returns (best_y, count_net, power_net) so the caster can also gate on
    whether the sweep is worth it at all.
    """
    def destroyables(perms):
        out = []
        for p in perms:
            if p.get("is_land") or p.get("loyalty", 0) > 0:
                continue
            out.append((int(_card_mana_value(p.get("card_idx", -1))),
                        p.get("power", 0)))
        return out
    opp = destroyables(g["opp_battlefield"])
    own = destroyables(g["self_battlefield"])
    best_key, best = None, (0, 0, 0)
    for y in sorted({mv for mv, _ in opp} | {0}):
        count_net = (sum(1 for mv, _ in opp if mv <= y)
                     - sum(1 for mv, _ in own if mv <= y))
        power_net = (sum(pw for mv, pw in opp if mv <= y)
                     - sum(pw for mv, pw in own if mv <= y))
        key = (count_net, power_net, -y)
        if best_key is None or key > best_key:
            best_key, best = key, (y, count_net, power_net)
    return best


def _desired_block_count(g: dict, attackers: list) -> int:
    """How many blockers to commit this combat (survival need vs. free-kill value)."""
    powers = sorted((a["power"] for a in attackers), reverse=True)
    my_life = g["self"]["life"]
    # Survival: fewest blocks (removing the biggest attackers) to drop incoming
    # damage below lethal.  We can't see which attackers are already blocked, so we
    # assume committed blockers absorbed the biggest ones (consistent each query).
    survival = 0
    while survival < len(powers) and sum(powers[survival:]) >= my_life:
        survival += 1
    # Value: free-kill blocks (a blocker that kills the attacker and survives).
    my_blockers = [p for p in _creatures(g["self_battlefield"]) if not p.get("tapped")]
    used, value = set(), 0
    for a in sorted(attackers, key=lambda x: -x["power"]):
        for j, b in enumerate(my_blockers):
            if j in used:
                continue
            if b["power"] >= a["toughness"] and b["toughness"] > a["power"]:
                used.add(j)
                value += 1
                break
    return min(len(attackers), max(survival, value))


class ScriptedAgent:
    """A stateless rule-based agent parameterised by an AgentConfig.

    ``act`` is a pure function of the (perspective-relative) observation, except
    for the RANDOM tier which draws from a per-agent seeded generator so parallel
    vec-env subprocesses stay reproducible and independent of the global RNG, the
    EXPLORE tier which carries a per-game set of already-chosen card ids for
    its novelty bias, and the GREEDY/HEURISTIC path which carries a per-game set
    of "fruitless" activation card ids (the stall guard, see ``_hard_act``).
    Both per-game sets are cleared by ``new_game`` and are deterministic
    functions of the trajectory, so same-seed replays stay identical.
    """

    def __init__(self, config: AgentConfig = AgentConfig()):
        self.config = config
        # Per-seat deck names (keyed by seat-is-A), pushed by callers that know
        # them via set_deck_names(); None = unknown, fall back to content-based
        # identification. Per-seat because one shared instance may drive both
        # players, each piloting a different deck.
        self._deck_names: dict[bool, str | None] = {True: None, False: None}
        # Per-agent generator: only consulted by the RANDOM tier.
        self._rng = np.random.default_rng(config.rng_seed)
        # Card ids already cast/activated this game: only touched by the EXPLORE
        # tier (novelty bias). Reset per game via new_game().
        self._explore_seen: set[int] = set()
        # Per-seat count of mulligans taken this game (London: the redrawn
        # hand is always 7 at the keep/mull decision, so the Doomsday all-in
        # mulligan's "never below 6" floor must count mulligans, not cards).
        # Zeroed on every keep, so a bo3's later games start fresh.
        self._mull_counts: dict[bool, int] = {True: 0, False: 0}
        # Stall guard (GREEDY/HEURISTIC only): per-seat map of card id ->
        # [failed_search_count, turn_of_last_fail] for activated abilities whose
        # search found NOTHING. Keyed by seat (obs[_SELF_IS_A_IDX] > 0.5) so one shared
        # instance driving both players (e.g. _DEFAULT_AGENT via scripted_action)
        # can't leak one seat's dead activation to the other. Reset per game via
        # new_game(). See _SEARCH_MENU_CATS / _hard_act.
        self._fruitless_activations: dict[bool, dict[int, list[int]]] = {}

    def new_game(self) -> None:
        """Reset per-game state at the start of a game.

        Called by ``runner.run_games`` (via ``ScriptedController.new_game``) before
        each game so a reused agent instance can't leak state across games. The
        per-game state is the EXPLORE novelty-bias set and the GREEDY/HEURISTIC
        fruitless-activation set; as a belt-and-braces for reuse paths that never
        call this (e.g. a scripted opponent living across vec-env episodes), both
        ``_explore_action`` and ``_hard_act`` also clear their own set whenever
        they see a pregame mulligan decision.
        """
        self._explore_seen.clear()
        self._fruitless_activations.clear()
        self._mull_counts = {True: 0, False: 0}

    def set_deck_names(self, deck_a: str | None, deck_b: str | None) -> None:
        """Tell the agent which deck each seat pilots.

        Callers that know the decks (runner.run_games, ModelVsScriptedEnv, the
        self-play scripted fallback) push the names here so deck identification
        uses the same name rule as the env's shaping opt-outs (_deck_named:
        'doomsday'/'tron' anywhere in the name — league/car_doomsday and
        league/tron count). A matching name is authoritative; a missing or
        non-matching name falls back to content-based identification, so
        sculpted temp decks in the test harness keep working."""
        self._deck_names = {True: deck_a, False: deck_b}

    def _seat_deck(self, obs: np.ndarray) -> str | None:
        """Deck name piloted by the seat this obs belongs to (None if unknown)."""
        return self._deck_names[bool(obs[_SELF_IS_A_IDX] > 0.5)]

    def _opp_deck(self, obs: np.ndarray) -> str | None:
        """Deck name piloted by this obs's OPPONENT (None if unknown)."""
        return self._deck_names[not bool(obs[_SELF_IS_A_IDX] > 0.5)]

    def _deck_is_doomsday(self, obs: np.ndarray) -> bool:
        """Unified doomsday identification: a known deck name is authoritative
        either way; only an unnamed deck (sculpted harness decks) falls back to
        the content heuristic, which also fires on the Thoughtseize / Dark
        Ritual / Lotus Petal hands of reanimator, bug and wrb_energy."""
        deck = self._seat_deck(obs)
        if deck is not None:
            return _deck_named(deck, "doomsday")
        return _is_doomsday_deck(obs)

    def _deck_is_tron(self, obs: np.ndarray) -> bool:
        """Unified tron identification: deck name first, content fallback.

        With no name known the synergy stays available (the historical
        behaviour — each tron block self-gates on the obs anyway); with a
        non-matching name, a Tron land controlled or in hand still counts."""
        deck = self._seat_deck(obs)
        if deck is None or _deck_named(deck, "tron"):
            return True
        return (bool(_controlled_tron_pieces(obs))
                or any(_hand_has_card(obs, cid) for cid in _TRON_LAND_IDS))

    def act(self, obs: np.ndarray, num_choices: int) -> int:
        if self.config.skill is Skill.RANDOM:
            return self._random_action(num_choices)
        if self.config.skill is Skill.EXPLORE:
            return self._explore_action(obs, num_choices)
        return self._hard_act(obs, num_choices)

    def _hard_act(self, obs: np.ndarray, num_choices: int) -> int:
        """GREEDY/HEURISTIC entry: fruitless-activation stall guard + dispatch.

        The guard piggybacks on the fact that a resolution-time search query is
        answered while the resolving ability is still on top of the stack (the
        engine's search_zone runs inside the ability's resolution), so the obs
        itself names the culprit — no cross-decision correlation state needed.
        When a search-category menu offers ONLY the fail-to-find slot, the
        search provably cannot succeed; if the top of the stack is a
        self-controlled ability, the fail is recorded against its source card
        id and the greedy ACTIVATE scan skips that card for the rest of the
        TURN — retrying next turn, because eligibility can change invisibly
        (Aether Vial's search keys on its charge counters, which are not in
        the obs: a fail at 0 counters says nothing about 1). A card that fails
        _FRUITLESS_ACTIVATION_LIMIT times is dropped for the game, so the
        original stall (Vial re-activated into a guaranteed fail every turn to
        the decision cap) stays bounded at a few wasted activations. Everything
        is O(1) per decision except a small set build when fails exist (rare);
        the stack slot is read only on a fail-only menu.
        """
        # cats[0] is enough to classify the menus we care about: a mulligan
        # query opens with the KEEP_HAND action (index 0), and a fail-only search
        # menu has exactly one action. Avoids decoding the whole category array.
        c0 = int(round(float(obs[STATE_SIZE]) * ACTION_CATEGORY_MAX))
        if c0 in (_CAT_MULLIGAN, _CAT_KEEP_HAND):
            # A game always opens with a mulligan query, so seeing one means a
            # new game started: clear the guard even when nobody called
            # new_game() (same safety net as _explore_seen in _explore_action).
            self._fruitless_activations.clear()
        elif (num_choices == 1 and c0 in _SEARCH_MENU_CATS
              and float(obs[ACT_IDS_START]) < 0.0):
            # Fail-only search menu (single choice, null card-id sentinel):
            # blame the resolving self-controlled ability on top of the stack.
            cid = _slot_card_idx(obs, _STACK_START + 1)
            if (cid >= 0 and obs[_STACK_START] > 0.5          # self-controlled
                    and obs[_STACK_START + 2] < 0.5):         # ability, not spell
                seat = bool(obs[_SELF_IS_A_IDX] > 0.5)
                turn = int(round(float(obs[_CUR_TURN_IDX]) * 50))
                entry = self._fruitless_activations.setdefault(
                    seat, {}).setdefault(cid, [0, -1])
                entry[0] += 1
                entry[1] = turn
        fruitless = None
        seat_fails = self._fruitless_activations.get(bool(obs[_SELF_IS_A_IDX] > 0.5))
        if seat_fails:
            turn = int(round(float(obs[_CUR_TURN_IDX]) * 50))
            fruitless = {cid for cid, (n, t) in seat_fails.items()
                         if n >= _FRUITLESS_ACTIVATION_LIMIT or t == turn}
        if self.config.skill is Skill.HEURISTIC:
            return self._heuristic_action(obs, num_choices, fruitless)
        return _greedy_action(obs, num_choices, fruitless,
                              doomsday=self._deck_is_doomsday(obs))

    # ------------------------------------------------------------------
    # HEURISTIC decision points (everything else falls back to GREEDY).
    # ------------------------------------------------------------------

    def _heuristic_action(self, obs: np.ndarray, num_choices: int,
                          fruitless: set[int] | None = None) -> int:
        cfg = self.config
        # Tron-synergy gating uses the unified deck identification (name rule,
        # content fallback) — computed once per decision for all blocks below.
        tron_synergy = cfg.use_tron_synergy and self._deck_is_tron(obs)
        cats = np.round(obs[ACT_CATS_START:ACT_CATS_START + num_choices]
                        * ACTION_CATEGORY_MAX).astype(int)
        card_ids = obs[ACT_IDS_START:ACT_IDS_START + MAX_ACTIONS]
        ctrl_arr = obs[ACT_CTRL_START:ACT_CTRL_START + MAX_ACTIONS]

        # Decode the readable game state lazily — only when a heuristic branch needs it.
        g_cache = {}
        def g():
            if "g" not in g_cache:
                gs = decode_game_state(obs[:STATE_SIZE])
                # Drop phased-out permanents before any heuristic reads the board:
                # the rules treat them as nonexistent (CR 702.26e) and the
                # pre-enrichment engine never serialized them, so filtering here
                # keeps every dict-based decision replay-stable (see _phased).
                for key in ("self_battlefield", "opp_battlefield"):
                    gs[key] = [p for p in gs[key] if not p.get("phased_out")]
                g_cache["g"] = gs
            return g_cache["g"]

        # Smart mulligan
        if cfg.use_smart_mulligan and any(c == _CAT_MULLIGAN for c in cats):
            return self._mulligan_choice(obs, g(), cats)

        if cfg.use_combat_sim:
            # Declare blockers (SEL_BLK + CONF_BLK offered together)
            if any(c == _CAT_CONF_BLK for c in cats):
                return self._block_choice(g(), cats, card_ids)
            # "Which attacker to block" sub-query (OTHER_CHOICE during Declare Blk)
            if g()["step"] == "Declare Blk" and any(c == _CAT_OTHER for c in cats):
                return self._block_target_choice(g(), cats, card_ids)
            # Declare attackers
            if any(c == _CAT_SEL_ATK for c in cats):
                return self._attack_choice(g(), cats, card_ids)

        # ── Reanimation / lands-combo interceptions ─────────────────────────
        # All card-id / pending-source gated (no deck-name gate needed): each
        # rule can only fire on a menu that names its specific card, so no
        # other deck's decisions change. Kept out of GREEDY so the easy tier
        # stays byte-identical.

        # Legend rule: duplicate Dark Depths → keep the 0-counter clone so
        # Marit Lage happens (see _keep_legend_choice).
        if any(c == _CAT_KEEP_LEGEND for c in cats):
            pick = _keep_legend_choice(obs, cats, card_ids, num_choices)
            if pick is not None:
                return pick

        # Life from the Loam: take the dredge draw-replacement whenever the
        # library can afford the mill — Loam back to hand every turn is the
        # lands deck's engine (recast it to return Wasteland / Urza's Saga).
        if (any(c == _CAT_CHOOSE_REPLACEMENT for c in cats)
                and _self_library_count(obs) > _LOAM_DREDGE_MIN_LIBRARY):
            for i, c in enumerate(cats):
                if (c == _CAT_CHOOSE_REPLACEMENT
                        and _action_card_id(card_ids, i) == _LIFE_FROM_LOAM_VOCAB_IDX):
                    return i

        # Beneficial-target polarity: the generic target rule prefers
        # opponent-controlled targets (right for removal, backwards for
        # effects that help their target). Keyed on the pending-decision
        # source so only the named cards change behaviour.
        if any(c == _CAT_TARGET for c in cats):
            pending = _slot_card_idx(obs, _PENDING_DECISION_START)
            # Reanimate / Animate Dead: the biggest creature card offered,
            # preferring our own graveyard.
            if pending in _REANIMATION_SPELL_IDS:
                pick = _reanimation_target_choice(cats, card_ids, ctrl_arr, num_choices)
                if pick is not None:
                    return pick
            # Thespian's Stage: copy our own Dark Depths (the Marit Lage line).
            if pending == _THESPIANS_STAGE_VOCAB_IDX:
                for i, c in enumerate(cats):
                    if (c == _CAT_TARGET and ctrl_arr[i] > 0.5
                            and _action_card_id(card_ids, i) == _DARK_DEPTHS_VOCAB_IDX):
                        return i
            # Life from the Loam: return the engine lands first (Urza's Saga
            # to redeploy, then Wasteland to keep stripping their mana).
            if pending == _LIFE_FROM_LOAM_VOCAB_IDX:
                for want in (_URZAS_SAGA_VOCAB_IDX, _WASTELAND_VOCAB_IDX):
                    for i, c in enumerate(cats):
                        if c == _CAT_TARGET and _action_card_id(card_ids, i) == want:
                            return i

        # London-mulligan bottoming: the Doomsday all-in mulligan keeps hands
        # for their specific combo pieces, so don't bottom blind (slot 0 could
        # be Doomsday itself) — give back junk by the same tiers as the
        # Brainstorm put-back. Other decks keep the legacy first-slot pick.
        if any(c == _CAT_BOTTOM_CARD for c in cats) and self._deck_is_doomsday(obs):
            return _dd_return_pick(obs, cats, card_ids, _CAT_BOTTOM_CARD)

        # Discard choice (lands-first conservative; replaces "first listed
        # card"). Own-hand discards, in priority order:
        #   1. a reanimation fatty — its home is the graveyard (the scripted
        #      agent never sideboards, so no post-board Show and Tell plan to
        #      protect);
        #   2. an excess land, while the hand still holds >= 3;
        #   3. the highest-MV card not castable any time soon
        #      (MV > lands in play + 1);
        #   4. the highest-MV card (lands are MV 0, so a spell goes first).
        # A menu of OPPONENT-owned cards (our Thoughtseize-style pick) takes
        # their highest-MV card instead — strip the biggest threat. A
        # Doomsday deck instead strips whatever can disrupt the combo:
        # countermagic first, then instant-speed removal (Thassa's Oracle
        # dies to it mid-kill), then the highest-MV fallback.
        if any(c == _CAT_DISCARD for c in cats):
            own_dc, opp_dc = [], []
            for i, c in enumerate(cats):
                if c != _CAT_DISCARD:
                    continue
                cid = _action_card_id(card_ids, i)
                (own_dc if ctrl_arr[i] > 0.5 else opp_dc).append((i, cid))
            for i, cid in own_dc:
                if cid in _REANIMATION_FATTY_IDS:
                    return i
            if own_dc:
                if _hand_land_count(obs) >= 3:
                    for i, cid in own_dc:
                        if cid in _LAND_VOCAB_IDS:
                            return i
                castable_mv = _self_land_count(obs) + 1
                uncastable = [(_card_mana_value(cid), i) for i, cid in own_dc
                              if _card_mana_value(cid) > castable_mv]
                if uncastable:
                    return max(uncastable)[1]
                return max((_card_mana_value(cid), i) for i, cid in own_dc)[1]
            if opp_dc:
                if self._deck_is_doomsday(obs):
                    for i, cid in opp_dc:
                        if cid in _COUNTER_SPELL_VOCAB_IDS:
                            return i
                    for i, cid in opp_dc:
                        if cid in _DD_DISRUPTOR_REMOVAL_IDS:
                            return i
                return max((_card_mana_value(cid), i) for i, cid in opp_dc)[1]

        # Urza's Saga chapter-II ability: always make the Construct when
        # offered. The Saga's mana ability is auto-pay (never offered at
        # priority), so the only Saga ACTIVATE here is the {2},{T} token
        # ability — and the Saga sacrifices itself after chapter III, so the
        # window is short. Ungated by phase, like The One Ring.
        for i, c in enumerate(cats):
            if (c == _CAT_ACTIVATE
                    and _action_card_id(card_ids, i) == _URZAS_SAGA_VOCAB_IDX):
                return i

        # Thespian's Stage: with our own Dark Depths in play, take the Clone
        # activation immediately (copy Depths → legend rule keeps the
        # 0-counter copy → Marit Lage). Without Depths, hold it — copying a
        # random land wastes the {2} and the tap.
        hold_activations = None
        if any(c == _CAT_ACTIVATE
               and _action_card_id(card_ids, i) == _THESPIANS_STAGE_VOCAB_IDX
               for i, c in enumerate(cats)):
            if _controls_card(obs, _DARK_DEPTHS_VOCAB_IDX):
                for i, c in enumerate(cats):
                    if (c == _CAT_ACTIVATE
                            and _action_card_id(card_ids, i) == _THESPIANS_STAGE_VOCAB_IDX):
                        return i
            else:
                hold_activations = {_THESPIANS_STAGE_VOCAB_IDX}

        # Sacrifice-cost pick: never eat a combo piece or irreplaceable
        # utility land while any other option exists (Crop Rotation sacking
        # the Dark Depths it plays around defeats itself). A Doomsday deck
        # keeps Thassa's Oracle's UU and its black source on the board
        # (_dd_sacrifice_pick). First non-protected option, else the generic
        # first pick.
        if any(c == _CAT_SACRIFICE for c in cats):
            if self._deck_is_doomsday(obs):
                pick = _dd_sacrifice_pick(obs, cats, card_ids)
                if pick is not None:
                    return pick
            for i, c in enumerate(cats):
                if (c == _CAT_SACRIFICE
                        and _action_card_id(card_ids, i) not in _SAC_LAST_LAND_IDS):
                    return i

        # Land-search preference (Crop Rotation, Expedition Map, any land
        # tutor — self-gating, the options only exist when the search can
        # fetch them): complete the missing Depths/Stage half, else grab
        # Tabernacle against a wide opposing board our own board shrugs at,
        # else Urza's Saga. Tron decks run none of these cards, so their own
        # search priorities (the tron blocks below) are unaffected.
        if any(c == _CAT_SEARCH for c in cats):
            search_prefs = []
            has_depths = _controls_card(obs, _DARK_DEPTHS_VOCAB_IDX)
            has_stage = _controls_card(obs, _THESPIANS_STAGE_VOCAB_IDX)
            if has_depths and not has_stage:
                search_prefs.append(_THESPIANS_STAGE_VOCAB_IDX)
            if has_stage and not has_depths:
                search_prefs.append(_DARK_DEPTHS_VOCAB_IDX)
            gs_creatures = None
            if any(_action_card_id(card_ids, i) == _TABERNACLE_VOCAB_IDX
                   for i, c in enumerate(cats) if c == _CAT_SEARCH):
                gs_creatures = (len(_creatures(g()["opp_battlefield"])),
                                len(_creatures(g()["self_battlefield"])))
                if gs_creatures[0] >= 3 and gs_creatures[1] <= 1:
                    search_prefs.append(_TABERNACLE_VOCAB_IDX)
            search_prefs.append(_URZAS_SAGA_VOCAB_IDX)
            for want in search_prefs:
                for i, c in enumerate(cats):
                    if c == _CAT_SEARCH and _action_card_id(card_ids, i) == want:
                        return i

        # Exploration first among casts: the extra land drop compounds every
        # later turn, so it beats any other one-shot play this turn.
        for i, c in enumerate(cats):
            if (c == _CAT_CAST
                    and _action_card_id(card_ids, i) == _EXPLORATION_VOCAB_IDX):
                return i

        # Land drop: complete the Depths/Stage pair when half is already down,
        # else lead with Urza's Saga (its chapters start ticking — mana, then
        # Constructs, then the tutor). Card-id gated: decks without these
        # lands in hand fall through unchanged.
        if any(c == _CAT_LAND for c in cats):
            land_prefs = []
            if _controls_card(obs, _DARK_DEPTHS_VOCAB_IDX):
                land_prefs.append(_THESPIANS_STAGE_VOCAB_IDX)
            if _controls_card(obs, _THESPIANS_STAGE_VOCAB_IDX):
                land_prefs.append(_DARK_DEPTHS_VOCAB_IDX)
            land_prefs.append(_URZAS_SAGA_VOCAB_IDX)
            for want in land_prefs:
                for i, c in enumerate(cats):
                    if c == _CAT_LAND and _action_card_id(card_ids, i) == want:
                        return i

        # Cast holds, accumulated for the greedy fallback's cast scan. Each is
        # card-id gated: it only fires when that cast is actually on the menu.
        hold_casts: set[int] = set()
        cast_ids = {_action_card_id(card_ids, i)
                    for i, c in enumerate(cats) if c == _CAT_CAST}
        # Reanimate/Animate Dead stay in hand until a creature worth cheating
        # out is in a graveyard (either side's — the engine only offers the
        # cast when SOME creature card is there, but a random 2-drop isn't
        # worth the card).
        if (cast_ids & _REANIMATION_SPELL_IDS
                and not _gy_has_any(obs, _REANIMATION_FATTY_IDS)):
            hold_casts |= _REANIMATION_SPELL_IDS
        # Counter triage. A Doomsday deck's counters are combo PROTECTION:
        # hold them until our own Thassa's Oracle — the spell or its win
        # trigger (both stack slots carry Oracle's card id) — is on the
        # stack, then counter ANY opponent spell that goes on top of it,
        # cantrip or not (a "cantrip" there is a response to the kill).
        # Other decks: an opponent spell is on the stack but nothing
        # threat-shaped (only exempt cantrips) — let it resolve and keep the
        # counter for their threat.
        if cast_ids & _COUNTER_SPELL_VOCAB_IDS:
            if self._deck_is_doomsday(obs):
                # ...and likewise fight for Doomsday itself and the Dark
                # Ritual paying for it — four Doomsdays were countered in one
                # 30-game sample while Force of Wills sat in hand — but only
                # while a SECOND counter stays back for Oracle: a countered
                # Doomsday costs a card, a countered Oracle with the pile
                # drawn costs the game (two wins turned into deck-outs when
                # the last counter went to Doomsday).
                n_counters = sum(
                    1 for slot in range(MAX_HAND_SLOTS)
                    if _slot_card_idx(obs, _HAND_START + slot * _HAND_SLOT_SIZE)
                    in _COUNTER_SPELL_VOCAB_IDS)
                fight = _opponent_has_spell_on_stack(obs) and (
                    _self_stack_has_card(obs, _THASSAS_ORACLE_VOCAB_IDX)
                    or (n_counters >= 2
                        and any(_self_stack_has_card(obs, cid)
                                for cid in _DD_PROTECT_ON_STACK_IDS)))
                if not fight:
                    hold_casts |= _COUNTER_SPELL_VOCAB_IDS
            elif not _opponent_threat_on_stack(obs):
                hold_casts |= _COUNTER_SPELL_VOCAB_IDS
        # Targeted removal (Swords/Push/Prismatic) waits for a real threat:
        # power >= 2, or a NONTOKEN 1-power creature (an unflipped Delver /
        # Dragon's Rage Channeler is worth killing; a 1/1 token is not).
        if cast_ids & _TARGETED_REMOVAL_IDS:
            threat = any(p["power"] >= 2
                         or (p["power"] >= 1
                             and p.get("card_idx", _TOKEN_VOCAB_BASE) < _TOKEN_VOCAB_BASE)
                         for p in _creatures(g()["opp_battlefield"]))
            if not threat:
                hold_casts |= _TARGETED_REMOVAL_IDS
        # Wrath of the Skies (symmetric X sweeper) only into a board where the
        # best energy amount nets a real sweep: 2+ more of their permanents
        # destroyed than ours, or a 4+ destroyed-power differential.
        if _WRATH_OF_SKIES_VOCAB_IDX in cast_ids:
            _, count_net, power_net = _wrath_energy_plan(g())
            if not (count_net >= 2 or power_net >= 4):
                hold_casts.add(_WRATH_OF_SKIES_VOCAB_IDX)
        hold_casts = hold_casts or None

        # Wrath of the Skies X / energy-pay amounts: both arrive as bare
        # number ladders (index == amount), and both should be the planned
        # energy Y — X buys exactly the energy the sweep needs, and the pay
        # step spends exactly that much (the generic pickers defaulted these
        # to 0, sweeping nothing). The resolution-time pay menu names Wrath
        # as the pending source; the cast-time X ladder fires BEFORE the
        # spell is pending (601.2b announcement), so that one is identified
        # from the newest action-history entry — our own just-recorded CAST
        # of Wrath (a plain null-id ladder, so no delve-menu overlap).
        wrath_x_menu = (
            _slot_card_idx(obs, _PENDING_DECISION_START) == _WRATH_OF_SKIES_VOCAB_IDX
            or (all(c == _CAT_CHOOSE_X for c in cats)
                and card_ids[0] <= _ACTION_CARD_ID_NULL + 0.01
                and int(round(float(obs[_HIST_START]) * ACTION_CATEGORY_MAX)) == _CAT_CAST
                and obs[_HIST_START + 2] > 0.5
                and _slot_card_idx(obs, _HIST_START + 1) == _WRATH_OF_SKIES_VOCAB_IDX))
        if (num_choices >= 2 and wrath_x_menu
                and all(c in (_CAT_CHOOSE_X, _CAT_OTHER) for c in cats)):
            y, _, _ = _wrath_energy_plan(g())
            return min(y, num_choices - 1)

        # Tron synergy: Candelabra of Tawnos untaps X target lands. It's only a mana
        # engine when it untaps lands that make MORE THAN ONE mana (Ancient Tomb, a
        # metalcraft Urza's Workshop, the assembled Urza's tron lands), so pay X for
        # exactly the tapped ones and target those, instead of letting the generic
        # picker choose X=0 / the first land. The pending-decision source names the
        # ability making each choice even before it hits the stack, so it can't be
        # confused with any other X ability or land-targeting effect.
        if (tron_synergy
                and _slot_card_idx(obs, _PENDING_DECISION_START) == _CANDELABRA_VOCAB_IDX):
            # X value: untap as many of our tapped >1-mana lands as X allows.
            if all(c == _CAT_CHOOSE_X for c in cats):
                return min(num_choices - 1, _tapped_multi_mana_lands(obs))
            # Targets: pick the >1-mana lands.
            if any(c == _CAT_TARGET for c in cats):
                for i, c in enumerate(cats):
                    if c == _CAT_TARGET and _action_card_id(card_ids, i) in _MULTI_MANA_LAND_IDS:
                        return i

        # Solitude's ETB exiles "up to one" creature, so its target menu leads with a
        # "No target" slot the generic picker can fall into. We only cast Solitude with
        # an opponent creature to hit (see _greedy_action), so here take that creature —
        # the biggest one — rather than whiffing on the optional "No target".
        if (any(c == _CAT_TARGET for c in cats)
                and _slot_card_idx(obs, _PENDING_DECISION_START) == _SOLITUDE_VOCAB_IDX):
            pick = _best_opp_creature_target(g(), cats, card_ids, ctrl_arr, num_choices)
            if pick is not None:
                return pick

        # Eval-based targeting — skipped for combo decks (R1: never disturb Doomsday).
        if (cfg.use_eval_targeting and any(c == _CAT_TARGET for c in cats)
                and not self._deck_is_doomsday(obs)):
            return self._target_choice(g(), cats, card_ids, ctrl_arr)

        # Tron synergy: Karn, the Great Creator's -2 wishes an artifact from the
        # sideboard/exile into hand. Grab Mycosynth Lattice first — with Karn in play
        # it locks the opponent out (every permanent becomes an artifact Karn's static
        # silences) — and fall back to Cityscape Leveler as the beater once Lattice is
        # already in hand/play (so no longer offered from the wishboard). Its fetch is a
        # hidden reveal, so the choice can arrive as a search or a choose-card menu;
        # match by pending source + card id across either.
        if (tron_synergy
                and _slot_card_idx(obs, _PENDING_DECISION_START) == _KARN_GREAT_CREATOR_VOCAB_IDX):
            for wish_id in (_MYCOSYNTH_LATTICE_VOCAB_IDX, _CITYSCAPE_LEVELER_VOCAB_IDX):
                for i in range(num_choices):
                    if _action_card_id(card_ids, i) == wish_id:
                        return i

        # Tron synergy: prioritize casting the Karn payoffs ahead of any other affordable
        # spell — Mycosynth Lattice first (completes the Karn lock), then Cityscape Leveler
        # (the premier threat / artifact-and-land destruction beater).
        if tron_synergy and any(c == _CAT_CAST for c in cats):
            for cast_id in (_MYCOSYNTH_LATTICE_VOCAB_IDX, _CITYSCAPE_LEVELER_VOCAB_IDX):
                for i, c in enumerate(cats):
                    if c == _CAT_CAST and _action_card_id(card_ids, i) == cast_id:
                        return i

        # Tron synergy: when activating Karn, the Great Creator, prefer his -2 (wish an
        # artifact — Mycosynth Lattice, then Cityscape Leveler — from the sideboard into hand) over the +1
        # (Animate). The two loyalty abilities are indistinguishable in the observation
        # (same ACTIVATE category + Karn card id), differing only in menu order: the
        # engine emits them in script order, +1 Animate then -2 ChangeZone, so the LAST
        # Karn activation offered is the wish. Only force it when both are on the menu
        # (Karn has the 2 loyalty to spend); with just +1 left, greedy takes it.
        if tron_synergy:
            karn_acts = [i for i, c in enumerate(cats)
                         if c == _CAT_ACTIVATE
                         and _action_card_id(card_ids, i) == _KARN_GREAT_CREATOR_VOCAB_IDX]
            if len(karn_acts) >= 2:
                return karn_acts[-1]

        # Tron synergy: once we control at least one of Urza's Mine/Power-Plant/
        # Tower, prefer playing a hand land that completes the set (a missing
        # real piece, else Planar Nexus) over any other offered land drop.
        if tron_synergy and any(c == _CAT_LAND for c in cats):
            controlled = _controlled_tron_pieces(obs)
            if controlled:
                pick = _tron_choice(cats, card_ids, _CAT_LAND, controlled)
                if pick is not None:
                    return pick

        # Tron synergy: a library search (Expedition Map, fetch effects) finds
        # the Tron piece we're missing, or Planar Nexus as a substitute, ahead
        # of any other offered card.
        if tron_synergy and any(c == _CAT_SEARCH for c in cats):
            pick = _tron_choice(cats, card_ids, _CAT_SEARCH, _controlled_tron_pieces(obs))
            if pick is not None:
                return pick

        # Anything not explicitly improved: proven GREEDY behaviour (with the
        # reanimation/Stage holds computed above threaded through its scans).
        return _greedy_action(obs, num_choices, fruitless,
                              hold_casts=hold_casts,
                              hold_activations=hold_activations,
                              doomsday=self._deck_is_doomsday(obs))

    @staticmethod
    def _confirm(cats, confirm_cat: int) -> int:
        for i, c in enumerate(cats):
            if c == confirm_cat:
                return i
        return 0

    def _mulligan_choice(self, obs: np.ndarray, g: dict, cats) -> int:
        """Keep a hand with a sane land count; otherwise mulligan.

        A one-land hand is keepable if it contains a cheap card-selection spell
        (Brainstorm, Ponder, Once Upon a Time) that can dig toward more lands.
        Hands with 5 or more lands are too land-heavy to keep.

        Doomsday all-in: piloting a Doomsday deck against an opponent whose
        decklist runs no Force of Will (nothing can stop the resolved combo),
        mulligan aggressively for a turn-1-3 kill hand (_dd_fast_hand),
        keeping ANY hand rather than go below 6 cards. London mulligan means
        the hand at this decision is ALWAYS 7 fresh cards — the "never below
        6" floor counts mulligans already taken (1 mulligan -> keeping now
        bottoms to 6), via the per-seat _mull_counts. An unknown opponent
        list counts as running Force of Will (stay on the standard rule).
        """
        hand_ids = [c["card_idx"] for c in g["self_hand"]]
        if self._deck_is_doomsday(obs):
            opp_deck = self._opp_deck(obs)
            has_fow = _deck_runs_force_of_will(opp_deck) if opp_deck else None
            if has_fow is False:
                seat = bool(obs[_SELF_IS_A_IDX] > 0.5)
                mulls = self._mull_counts.get(seat, 0)
                keep = mulls >= 1 or _dd_fast_hand(obs, hand_ids)
                self._mull_counts[seat] = 0 if keep else mulls + 1
                for i, c in enumerate(cats):
                    if (c != _CAT_MULLIGAN) == keep:
                        return i
                return 0
        lands = sum(1 for c in g["self_hand"] if c["card_idx"] in _LAND_VOCAB_IDS)
        has_dig = any(c["card_idx"] in _KEEP_ONE_LANDER_IDS for c in g["self_hand"])
        keepable = 2 <= lands <= 4 or (lands == 1 and has_dig)
        # Mulligan query: index 0 = keep, any other index = mulligan.
        if keepable:
            for i, c in enumerate(cats):
                if c != _CAT_MULLIGAN:
                    return i
            return 0  # only keep offered
        for i, c in enumerate(cats):
            if c == _CAT_MULLIGAN:
                return i
        return 0

    def _attack_choice(self, g: dict, cats, card_ids) -> int:
        opp_blockers = [p for p in _creatures(g["opp_battlefield"]) if not p.get("tapped")]
        my_attackers = [p for p in _creatures(g["self_battlefield"])
                        if not p.get("tapped") and not p.get("summoning_sick")]
        total_power = sum(p["power"] for p in my_attackers)
        alpha = total_power > 0 and total_power >= g["opponent"]["life"]
        for i, c in enumerate(cats):
            if c != _CAT_SEL_ATK:
                continue
            pt = _lookup_pt(g["self_battlefield"], _action_card_id(card_ids, i),
                            exclude_attacking=True, require_untapped=True, require_unsick=True)
            if pt is None or _should_attack(pt[0], pt[1], opp_blockers, alpha):
                return i  # attack (or attack by default if P/T unknown)
        return self._confirm(cats, _CAT_CONF_ATK)

    def _block_choice(self, g: dict, cats, card_ids) -> int:
        attackers = [p for p in _creatures(g["opp_battlefield"]) if p.get("attacking")]
        if not attackers:
            return self._confirm(cats, _CAT_CONF_BLK)
        committed = sum(1 for p in _creatures(g["self_battlefield"]) if p.get("blocking"))
        desired = _desired_block_count(g, attackers)
        blk = [(i, _action_card_id(card_ids, i)) for i, c in enumerate(cats)
               if c == _CAT_SEL_BLK]
        if committed < desired and blk:
            # Commit the sturdiest available blocker (highest toughness, then power).
            def quality(ia):
                pt = _lookup_pt(g["self_battlefield"], ia[1], exclude_blocking=True,
                                require_untapped=True)
                return (pt[1] * 10 + pt[0]) if pt else 0
            return max(blk, key=quality)[0]
        return self._confirm(cats, _CAT_CONF_BLK)

    def _block_target_choice(self, g: dict, cats, card_ids) -> int:
        """Assign the just-selected blocker to the biggest offered attacker."""
        best_i, best_pow = None, -1
        for i, c in enumerate(cats):
            if c != _CAT_OTHER:
                continue
            pt = _lookup_pt(g["opp_battlefield"], _action_card_id(card_ids, i))
            pw = pt[0] if pt else 0
            if pw > best_pow:
                best_pow, best_i = pw, i
        return best_i if best_i is not None else 0

    def _target_choice(self, g: dict, cats, card_ids, ctrl_arr) -> int:
        """Removal → biggest threat; burn → face only when lethal-ish."""
        opp_creatures, player_targets = [], []
        for i, c in enumerate(cats):
            if c != _CAT_TARGET:
                continue
            ctrl = ctrl_arr[i]
            if ctrl >= 0.5:
                continue  # self-controlled — avoid
            if ctrl < _ACTION_CTRL_NULL + 0.005:
                player_targets.append(i)  # player / non-entity target (burn to face)
            else:
                pt = _lookup_pt(g["opp_battlefield"], _action_card_id(card_ids, i))
                opp_creatures.append((i, pt[0] if pt else 0))
        if player_targets and g["opponent"]["life"] <= _BURN_FACE_LIFE:
            return player_targets[0]
        if opp_creatures:
            return max(opp_creatures, key=lambda x: x[1])[0]
        if player_targets:
            return player_targets[0]
        # Fallback: first non-self target, mirroring GREEDY.
        for i, c in enumerate(cats):
            if c == _CAT_TARGET and ctrl_arr[i] < 0.5:
                return i
        for i, c in enumerate(cats):
            if c == _CAT_TARGET:
                return i
        return 0

    def _random_action(self, num_choices: int) -> int:
        """Uniform choice over the legal actions.

        Every offered action is legal, and a confirm action is always among the
        choices during mandatory attacker/blocker declaration, so uniform picking
        terminates with probability 1 (the engine also truncates as a backstop).
        """
        if num_choices <= 1:
            return 0
        return int(self._rng.integers(0, num_choices))

    def _explore_action(self, obs: np.ndarray, num_choices: int) -> int:
        """Coverage fuzzer: a game-seed-deterministic, coverage-biased weighted choice.

        Goal: sweeping a range of seeds should touch as many distinct engine paths as
        possible — the opposite of GREEDY/HEURISTIC, which converge on one strong line
        per state. Three ingredients:

        * **Coverage bias.** Each legal action is weighted by its category
          (``_EXPLORE_WEIGHTS``) so the pick leans toward the actions that drive the
          most engine code — casting spells, activating abilities, attacking/blocking —
          while still passing often enough to advance the game. Same-category menus
          (targets, X values, modes, search/dig picks) get uniform weight, so *which*
          target/mode/X is taken varies freely across seeds — except a library
          search's "Fail to find" slot, which is de-weighted to
          ``_EXPLORE_SEARCH_FAIL_WEIGHT`` so tutors/fetches usually complete.

        * **Novelty bias.** A cast/activate option whose card id this agent has not
          yet chosen this game gets its weight multiplied by
          ``_EXPLORE_NOVELTY_FACTOR``, so expensive cards with cheaper alternatives
          in the same deck still get cast somewhere in a seed sweep instead of losing
          the draw every time. Options with the null card sentinel are non-novel (no
          boost). Chosen cast/activate card ids are recorded in ``_explore_seen``,
          which is per game: ``new_game()`` clears it, and the pregame mulligan
          decision clears it too as a safety net for agent instances reused across
          games without a ``new_game()`` call.

        * **Determinism, tied to the GAME seed.** The weighted draw comes from Python's
          global ``random`` — the same source GREEDY breaks ties with — which
          ``runner.run_games`` (and the test harness) seed to the engine ``--seed``
          before each game (``random.seed(seed)``). So the whole explore trajectory is
          reproducible for a given seed, and a different seed reshuffles both the deck
          (a different observation at every step) and this draw, branching into a
          different line. No separate per-agent seed; the only cross-call state is the
          per-game novelty set, itself a deterministic function of the trajectory.

        Not a strong opponent — combo lines are off (see the "explore" preset), so it
        fans out across cards instead of collapsing into a deck's known line.

        The ``explore:patient`` preset (``AgentConfig.explore_profile ==
        "patient"``) runs this same machinery with a big-mana weight profile —
        land drops near-mandatory, PASS boosted over cheap activations, CAST
        weight scaled by mana value — so expensive cards that plain explore never
        even sees as legal options get developed toward and hard-cast (see
        ``_EXPLORE_PATIENT_WEIGHTS``).
        """
        if num_choices <= 1:
            return 0
        cats = np.round(obs[ACT_CATS_START:ACT_CATS_START + num_choices]
                        * ACTION_CATEGORY_MAX).astype(int)
        # A game always opens with a mulligan query, so seeing one means a new game
        # started: clear the novelty set even when nobody called new_game() (e.g. a
        # scripted opponent reused across vec-env episodes). Repeated clears during
        # London mulligans are harmless — nothing is cast pregame.
        if any(int(c) == _CAT_MULLIGAN for c in cats):
            self._explore_seen.clear()
        card_ids = obs[ACT_IDS_START:ACT_IDS_START + MAX_ACTIONS]
        # The "patient" profile swaps in the big-mana weight table, scales CAST
        # weight by mana value, and withholds the novelty boost from ACTIVATE
        # (see the _EXPLORE_PATIENT_* constants). The default profile takes the
        # exact same code path with the original table, so plain explore
        # behaviour (and its RNG consumption) is unchanged.
        patient = self.config.explore_profile == "patient"
        table = _EXPLORE_PATIENT_WEIGHTS if patient else _EXPLORE_WEIGHTS
        weights = []
        for i, c in enumerate(cats):
            w = table.get(int(c), _EXPLORE_DEFAULT_WEIGHT)
            # De-weight a search's "Fail to find" (the null-card-id slot) so the
            # tutor/fetch usually finds — see _EXPLORE_SEARCH_FAIL_WEIGHT.
            if int(c) == _CAT_SEARCH and _action_card_id(card_ids, i) < 0:
                w *= _EXPLORE_SEARCH_FAIL_WEIGHT
            if int(c) in (_CAT_CAST, _CAT_ACTIVATE):
                cid = _action_card_id(card_ids, i)
                if cid >= 0 and cid not in self._explore_seen:
                    if int(c) == _CAT_CAST or not patient or _PATIENT_ACTIVATE_NOVELTY:
                        w *= _EXPLORE_NOVELTY_FACTOR
                if patient and cid >= 0:
                    if int(c) == _CAT_CAST:
                        w *= 1.0 + _card_mana_value(cid) * _PATIENT_CAST_MV_FACTOR
                    elif _hand_has_card(obs, cid):
                        # Card-consuming from-hand activation (cycling etc.):
                        # hold the card — see _PATIENT_HOLD_MV_FACTOR.
                        w /= 1.0 + _card_mana_value(cid) * _PATIENT_HOLD_MV_FACTOR
            weights.append(w)
        # Mulligan is a 2-way [keep, mulligan] choice — bias toward keeping (index 0)
        # so most games actually start, but still mulligan ~1/4 of the time to exercise
        # the mulligan + London-bottoming paths.
        if num_choices == 2 and all(int(c) == _CAT_MULLIGAN for c in cats):
            weights = [3.0, 1.0]
        if sum(weights) <= 0:
            weights = [1.0] * num_choices
        choice = int(random.choices(range(num_choices), weights=weights, k=1)[0])
        # Record the chosen card so this game's later draws treat it as non-novel.
        if int(cats[choice]) in (_CAT_CAST, _CAT_ACTIVATE):
            cid = _action_card_id(card_ids, choice)
            if cid >= 0:
                self._explore_seen.add(cid)
        return choice


def make_agent(spec: str = "scripted") -> ScriptedAgent:
    """Resolve a spec string into a ScriptedAgent.

    Accepts ``"scripted"``, ``"scripted:hard"``, ``"scripted:easy"``,
    ``"scripted:random"``, ``"scripted:explore"``, or the bare suffixes
    ``"greedy"``/``"hard"``/``"heuristic"``/``"easy"``/``"random"``/
    ``"explore"``/``"fuzz"``/``"explore:patient"``/``"patient"``
    (case-insensitive). ``explore``/``fuzz`` select the coverage fuzzer
    (Skill.EXPLORE) — vary the game ``--seed`` to fan it across engine paths.
    ``explore:patient`` (or bare ``patient``; ``scripted:explore:patient`` also
    resolves) is the fuzzer's big-mana profile: it develops mana and holds
    expensive cards until they are castable (see ``_EXPLORE_PATIENT_WEIGHTS``).
    """
    s = (spec or "scripted").strip().lower()
    suffix = s.split(":", 1)[1] if s.startswith("scripted:") else s
    if suffix in ("", "scripted"):
        suffix = "scripted"
    if suffix not in _PRESETS:
        raise ValueError(
            f"unknown scripted-agent spec {spec!r}; "
            f"valid suffixes: {sorted(_PRESETS)}"
        )
    return ScriptedAgent(_PRESETS[suffix])


# ── Module-level public entry point ─────────────────────────────────────────
# ``scripted_action`` is the bare function form used by callers that don't build a
# ScriptedAgent (analysis/baseline/observe scripted opponent, tui_game, bench). It
# now defaults to the "hard" heuristic agent, matching make_agent("scripted"), so
# every default scripted call plays at the hard tier. The heuristic is a pure
# function of (obs, num_choices) except for the per-game fruitless-activation
# stall guard, which is seat-keyed and self-clearing on the pregame mulligan
# (see _hard_act), so one shared instance stays safe across games/processes and
# across both seats. Callers wanting the old greedy behaviour build
# make_agent("easy"/"greedy") explicitly.
_DEFAULT_AGENT = ScriptedAgent(_PRESETS["scripted"])


def scripted_action(obs: np.ndarray, num_choices: int) -> int:
    return _DEFAULT_AGENT.act(obs, num_choices)
