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

import numpy as np

from env import (
    # sizes / normalisers
    STATE_SIZE, MAX_ACTIONS, ACTION_CATEGORY_MAX, N_CARD_TYPES, MAX_HAND_SLOTS,
    # action category constants
    _CAT_PASS, _CAT_SEL_ATK, _CAT_CONF_ATK, _CAT_SEL_BLK, _CAT_CONF_BLK,
    _CAT_ACTIVATE, _CAT_CAST, _CAT_TARGET, _CAT_LAND, _CAT_MULLIGAN, _CAT_SEARCH,
    _CAT_OTHER, _CAT_PAYING, _CAT_DIG, _CAT_TOP_LIBRARY, _CAT_SB_DONE,
    _CAT_COMPANION, _CAT_CHOOSE_X, _CAT_CHOOSE_CARD, _CAT_YESNO,
    # battlefield / stack layout
    _BF_START, _BF_SLOT_SIZE, _PERM_A_SLOTS, _BF_CARD_OFF, _STACK_START,
    _STACK_SLOT_SIZE, _HAND_START, _HAND_SLOT_SIZE, _CUR_TURN_IDX,
    _OFF_IS_TAPPED, _OFF_IS_ATTACKING, _OFF_HAS_SICKNESS, _OFF_IS_CREATURE,
    _OFF_IS_LAND,
    # per-slot card-id decoder
    _slot_card_idx,
    # vocab / targeting constants
    _ACTION_CARD_ID_NULL, _ACTION_CTRL_NULL, _BLUE_POOL_IDX, _WASTELAND_VOCAB_IDX,
    _AETHER_VIAL_VOCAB_IDX,
    _BASIC_LAND_IDS, _COUNTER_SPELL_VOCAB_IDS, _COUNTERSPELL_VOCAB_IDX,
    _DOOMSDAY_VOCAB_IDX, _DARK_RITUAL_VOCAB_IDX, _THASSAS_ORACLE_VOCAB_IDX,
    _STREET_WRAITH_VOCAB_IDX, _EDGE_OF_AUTUMN_VOCAB_IDX, _DOOMSDAY_DECK_IDS,
    _KEEP_ONE_LANDER_IDS,
    _GREEN_SUNS_ZENITH_VOCAB_IDX, _KEEN_EYED_CURATOR_VOCAB_IDX, _MANA_DORK_IDS,
    _UNRELIABLE_LAND_IDS,
    _URZAS_MINE_VOCAB_IDX, _URZAS_POWER_PLANT_VOCAB_IDX, _URZAS_TOWER_VOCAB_IDX,
    _PLANAR_NEXUS_VOCAB_IDX, _TRON_LAND_IDS,
    _KARN_GREAT_CREATOR_VOCAB_IDX, _CITYSCAPE_LEVELER_VOCAB_IDX,
    _CANDELABRA_VOCAB_IDX, _MULTI_MANA_LAND_IDS,
    _SOLITUDE_VOCAB_IDX, _THE_ONE_RING_VOCAB_IDX,
    # Lion's Eye Diamond crack: blue-mana category + LED vocab id + draw-on-stack test
    _CAT_MANA_U, _LED_VOCAB_IDX, _self_has_draw_on_stack,
    # pending-decision source index (obs[_PENDING_DECISION_START] == source card id)
    _PENDING_DECISION_START,
    # library-count context index (obs[_LIBRARY_CTX_START] == self_library_ct / 60)
    _LIBRARY_CTX_START,
    # known top-of-library slot 0 (obs[_KNOWN_TOP_LIB_START] == top card id, sentinel=unknown)
    _KNOWN_TOP_LIB_START,
    # shared helpers (also used by env's reward shaping, so they stay in env.py)
    _hand_has_card, _stack_is_empty,
)
from decode import decode_game_state
from card_costs import _LAND_VOCAB_IDS, _CARD_COST_MATRIX


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
    """Heuristic: check if any hand card is a doomsday-deck-only card."""
    for slot in range(MAX_HAND_SLOTS):
        idx = _slot_card_idx(obs, _HAND_START + slot * _HAND_SLOT_SIZE)
        if idx in _DOOMSDAY_DECK_IDS:
            return True
    return False


def _all_eligible_creatures_attacking(obs: np.ndarray) -> bool:
    """Return True if every untapped, non-sick creature in self's slots (0-47) is attacking."""
    any_eligible = False
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
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
        if obs[base + _OFF_IS_CREATURE] > 0.5:
            return True
    return False


def _tapped_multi_mana_lands(obs: np.ndarray) -> int:
    """Count self-controlled TAPPED lands that make more than one mana (self slots 0-47).

    These are the lands worth untapping with Candelabra of Tawnos — untapping a tapped
    Ancient Tomb / metalcraft Urza's Workshop / assembled Urza's tron land frees 2-3 mana
    apiece, whereas untapping a one-mana land (or any untapped land) gains nothing.
    """
    count = 0
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
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
_SELF_LIFE_IDX = 0   # obs[0] == priority player's life / 20

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
})


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
    if _self_library_count(obs) > _ORACLE_WIN_LIBRARY:
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


def _count_blue_sources(obs: np.ndarray) -> int:
    """Untapped blue-mana sources self controls (self battlefield slots 0..47)."""
    count = 0
    for slot in range(_PERM_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if obs[base + _OFF_IS_TAPPED] > 0.5:
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
        if obs[base + _OFF_IS_TAPPED] > 0.5:
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


def _classify_top_library(cats, card_ids):
    """Bucket the offered TOP_LIBRARY choices into (oracle_idx, wraith_idxs, filler_idxs)."""
    oracle_i, wraith_is, filler_is = None, [], []
    for i, c in enumerate(cats):
        if c != _CAT_TOP_LIBRARY:
            continue
        cid = _action_card_id(card_ids, i)
        if cid == _THASSAS_ORACLE_VOCAB_IDX:
            if oracle_i is None:
                oracle_i = i
        elif cid == _STREET_WRAITH_VOCAB_IDX:
            wraith_is.append(i)
        else:
            filler_is.append(i)
    return oracle_i, wraith_is, filler_is


def _doomsday_pile_pick(cats, card_ids) -> int:
    """Choose a card during Doomsday pile building (TOP_LIBRARY queries).

    Doomsday resolves in two TOP_LIBRARY phases:

      (a) SELECTION — choose WHICH 5 cards to keep, from the whole library and
          graveyard (many options). Composition only; the order is fixed by the
          rearrange that follows, so just keep Thassa's Oracle and the Street
          Wraiths and let anything fill the rest.

      (b) REARRANGE — order the (<= 5) kept cards. The engine fills the pile
          bottom-first (effect_rearrange_top_of_library.cpp), so the
          position-from-top being placed equals the number of remaining choices.

    Target pile, top -> bottom: [Wraith, Wraith, Thassa's Oracle, filler, filler].
    The kill is then: cycle a Street Wraith to draw Wraith(pos1), cycle to draw
    Wraith(pos2), cycle to draw Oracle(pos3) -> library is now 2 -> cast Oracle
    and win. Oracle sits 3rd from the top (== 3rd from the bottom of the pile).
    """
    oracle_i, wraith_is, filler_is = _classify_top_library(cats, card_ids)
    n_choices = sum(1 for c in cats if c == _CAT_TOP_LIBRARY)

    # (a) SELECTION: keep Oracle, then the Street Wraiths, then any filler.
    if n_choices > _DD_PILE_SIZE:
        if oracle_i is not None:
            return oracle_i
        if wraith_is:
            return wraith_is[0]
        return 0

    # Small rearrange with no combo pieces (Ponder / Brainstorm): legacy behaviour.
    if oracle_i is None and not wraith_is:
        return 0

    # (b) REARRANGE: position-from-top being placed == remaining choice count.
    pos = n_choices
    if pos == 3 and oracle_i is not None:
        return oracle_i                      # Thassa's Oracle 3rd from top
    if pos <= 2:                             # top slots: free-draw enablers first
        if wraith_is:
            return wraith_is[0]
        if oracle_i is not None:
            return oracle_i
    # Bottom slots (pos 4, 5): bury filler, keeping Oracle/Wraiths shallower.
    if filler_is:
        return filler_is[0]
    if wraith_is:
        return wraith_is[0]
    if oracle_i is not None:
        return oracle_i
    return 0


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
                   fruitless: set[int] | None = None) -> int:
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

    Action categories are stored in obs[STATE_SIZE:] normalised by ACTION_CATEGORY_MAX.
    """
    cats     = np.round(obs[STATE_SIZE:STATE_SIZE + num_choices] * ACTION_CATEGORY_MAX).astype(int)
    card_ids = obs[STATE_SIZE + MAX_ACTIONS     : STATE_SIZE + 2 * MAX_ACTIONS]
    ctrl_arr = obs[STATE_SIZE + 2 * MAX_ACTIONS : STATE_SIZE + 3 * MAX_ACTIONS]

    _STEP_FIRST_MAIN  = 21   # obs[18 + 3]
    _STEP_SECOND_MAIN = 28   # obs[18 + 10] (shifted +1 by FIRST_STRIKE_DAMAGE step)
    in_main_phase = obs[_STEP_FIRST_MAIN] > 0.5 or obs[_STEP_SECOND_MAIN] > 0.5

    # 0a. Sideboarding: scripted agent never sideboards — always pick done (index 0)
    if any(c == _CAT_SB_DONE for c in cats):
        return 0

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
        return 1 if num_choices > 1 else 0

    # 5b. Paying costs (tapping lands for mana during spell/ability payment).
    #     Pick the first available option — this taps a source to pay the cost.
    if any(c == _CAT_PAYING for c in cats):
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

    # Priority: if opponent has a spell on stack and we have UU, cast Counterspell first.
    if opponent_spell_on_stack and int(round(obs[_BLUE_POOL_IDX] * 10)) >= 2:
        for i, c in enumerate(cats):
            if c == _CAT_CAST and _action_card_id(card_ids, i) == _COUNTERSPELL_VOCAB_IDX:
                return i

    # Crack Lion's Eye Diamond in the Doomsday kill window — a self draw is resolving,
    # Oracle is the known top card, and library <= 2 — to float UUU for the Thassa's
    # Oracle win. LED is offered as instant-speed MANA_* actions at priority (unlike
    # ordinary mana sources, which stay hidden and auto-pay), so it is picked here.
    led = _led_crack_choice(obs, cats, card_ids)
    if led is not None:
        return led

    # --- Doomsday deck rules ---
    has_doomsday_in_hand = _hand_has_card(obs, _DOOMSDAY_VOCAB_IDX)

    # Win the game: cast Thassa's Oracle as soon as the library is small enough
    # that its devotion-to-blue (>= 2 from its own UU) meets/exceeds the library
    # size. "Mana is there" == the engine is offering Oracle as a legal cast.
    if _self_library_count(obs) <= _ORACLE_WIN_LIBRARY:
        for i, c in enumerate(cats):
            if c == _CAT_CAST and _action_card_id(card_ids, i) == _THASSAS_ORACLE_VOCAB_IDX:
                return i

    # Cast Doomsday once we control enough blue sources to follow it up with
    # Thassa's Oracle (UU). Firing it earlier just empties our library into a
    # deck-out, so we hold it until the mana to win is on the board.
    has_blue_for_oracle = _count_blue_sources(obs) >= _BLUE_SOURCES_NEEDED
    for i, c in enumerate(cats):
        if c == _CAT_CAST and _action_card_id(card_ids, i) == _DOOMSDAY_VOCAB_IDX:
            if has_blue_for_oracle:
                return i
            # else hold Doomsday until the blue mana to cast Oracle is in play

    # Only cast Dark Ritual if Doomsday is also in hand and we're ready to combo
    # (don't waste ritual mana without the payoff online).
    for i, c in enumerate(cats):
        if c == _CAT_CAST and _action_card_id(card_ids, i) == _DARK_RITUAL_VOCAB_IDX:
            if has_doomsday_in_hand and has_blue_for_oracle:
                return i
            # else skip it

    for i, c in enumerate(cats):
        if c == _CAT_CAST:
            cid = _action_card_id(card_ids, i)
            if cid in _COUNTER_SPELL_VOCAB_IDS and not opponent_spell_on_stack:
                continue
            if cid == _DARK_RITUAL_VOCAB_IDX:
                continue  # handled above
            if cid == _DOOMSDAY_VOCAB_IDX:
                continue  # handled above (held until blue mana for Oracle is online)
            if cid == _THASSAS_ORACLE_VOCAB_IDX:
                continue  # only cast Oracle as the wincon (handled above)
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
    if (_self_library_count(obs) <= _DD_PILE_SIZE
            and _self_life(obs) >= _STREET_WRAITH_MIN_LIFE):
        for i, c in enumerate(cats):
            if c == _CAT_ACTIVATE and _action_card_id(card_ids, i) == _STREET_WRAITH_VOCAB_IDX:
                return i

    # Always cycle Edge of Autumn (sac a land, draw a card) — only offered when legal
    for i, c in enumerate(cats):
        if c == _CAT_ACTIVATE and _action_card_id(card_ids, i) == _EDGE_OF_AUTUMN_VOCAB_IDX:
            return i

    # Always activate The One Ring's {T} draw ability whenever able — each use draws
    # a card per burden counter (pure card advantage), and it only taps once per turn.
    # Ungated by phase so it still fires on turns the agent has nothing else to do.
    for i, c in enumerate(cats):
        if c == _CAT_ACTIVATE and _action_card_id(card_ids, i) == _THE_ONE_RING_VOCAB_IDX:
            return i

    if in_main_phase:
        for i, c in enumerate(cats):
            if c == _CAT_ACTIVATE:
                cid = _action_card_id(card_ids, i)
                if cid == _WASTELAND_VOCAB_IDX and not _opponent_has_nonbasic_land(obs):
                    continue  # no opponent nonbasic land to target — skip
                if cid == _STREET_WRAITH_VOCAB_IDX:
                    continue  # only ever cycled via the gated combo rule above
                if cid == _KEEN_EYED_CURATOR_VOCAB_IDX and not _stack_is_empty(obs):
                    continue  # don't pile copies onto the stack (each extra one
                              # fizzles when its graveyard target is already exiled)
                if cid == _CANDELABRA_VOCAB_IDX and _tapped_multi_mana_lands(obs) == 0:
                    continue  # nothing worth untapping — don't tap Candelabra for X=0
                if fruitless and cid in fruitless:
                    continue  # stall guard: this card's search found NOTHING
                              # earlier this game (see _SEARCH_MENU_CATS) —
                              # re-activating it would just fail to find again
                return i

    # 9. (Removed - mana abilities no longer offered during normal priority in machine mode;
    #     mana is tapped automatically during cost payment via PAYING_COSTS.)

    # 10. Doomsday pile building (TOP_LIBRARY): select the 5 pile cards, then
    #     order them so Thassa's Oracle sits 3rd from the top (== 3rd from the
    #     bottom of the five-card pile) behind two Street Wraiths, ready for the
    #     cycle-chain kill. See _doomsday_pile_pick for the full line.
    if any(c == _CAT_TOP_LIBRARY for c in cats):
        return _doomsday_pile_pick(cats, card_ids)

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
        # Per-agent generator: only consulted by the RANDOM tier.
        self._rng = np.random.default_rng(config.rng_seed)
        # Card ids already cast/activated this game: only touched by the EXPLORE
        # tier (novelty bias). Reset per game via new_game().
        self._explore_seen: set[int] = set()
        # Stall guard (GREEDY/HEURISTIC only): per-seat map of card id ->
        # [failed_search_count, turn_of_last_fail] for activated abilities whose
        # search found NOTHING. Keyed by seat (obs[32] > 0.5) so one shared
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
        # query is all-MULLIGAN, and a fail-only search menu has exactly one
        # action. Avoids decoding the whole category array a second time.
        c0 = int(round(float(obs[STATE_SIZE]) * ACTION_CATEGORY_MAX))
        if c0 == _CAT_MULLIGAN:
            # A game always opens with a mulligan query, so seeing one means a
            # new game started: clear the guard even when nobody called
            # new_game() (same safety net as _explore_seen in _explore_action).
            self._fruitless_activations.clear()
        elif (num_choices == 1 and c0 in _SEARCH_MENU_CATS
              and float(obs[STATE_SIZE + MAX_ACTIONS]) < 0.0):
            # Fail-only search menu (single choice, null card-id sentinel):
            # blame the resolving self-controlled ability on top of the stack.
            cid = _slot_card_idx(obs, _STACK_START + 1)
            if (cid >= 0 and obs[_STACK_START] > 0.5          # self-controlled
                    and obs[_STACK_START + 2] < 0.5):         # ability, not spell
                seat = bool(obs[32] > 0.5)
                turn = int(round(float(obs[_CUR_TURN_IDX]) * 50))
                entry = self._fruitless_activations.setdefault(
                    seat, {}).setdefault(cid, [0, -1])
                entry[0] += 1
                entry[1] = turn
        fruitless = None
        seat_fails = self._fruitless_activations.get(bool(obs[32] > 0.5))
        if seat_fails:
            turn = int(round(float(obs[_CUR_TURN_IDX]) * 50))
            fruitless = {cid for cid, (n, t) in seat_fails.items()
                         if n >= _FRUITLESS_ACTIVATION_LIMIT or t == turn}
        if self.config.skill is Skill.HEURISTIC:
            return self._heuristic_action(obs, num_choices, fruitless)
        return _greedy_action(obs, num_choices, fruitless)

    # ------------------------------------------------------------------
    # HEURISTIC decision points (everything else falls back to GREEDY).
    # ------------------------------------------------------------------

    def _heuristic_action(self, obs: np.ndarray, num_choices: int,
                          fruitless: set[int] | None = None) -> int:
        cfg = self.config
        cats = np.round(obs[STATE_SIZE:STATE_SIZE + num_choices]
                        * ACTION_CATEGORY_MAX).astype(int)
        card_ids = obs[STATE_SIZE + MAX_ACTIONS:STATE_SIZE + 2 * MAX_ACTIONS]
        ctrl_arr = obs[STATE_SIZE + 2 * MAX_ACTIONS:STATE_SIZE + 3 * MAX_ACTIONS]

        # Decode the readable game state lazily — only when a heuristic branch needs it.
        g_cache = {}
        def g():
            if "g" not in g_cache:
                g_cache["g"] = decode_game_state(obs[:STATE_SIZE])
            return g_cache["g"]

        # Smart mulligan
        if cfg.use_smart_mulligan and any(c == _CAT_MULLIGAN for c in cats):
            return self._mulligan_choice(g(), cats)

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

        # Tron synergy: Candelabra of Tawnos untaps X target lands. It's only a mana
        # engine when it untaps lands that make MORE THAN ONE mana (Ancient Tomb, a
        # metalcraft Urza's Workshop, the assembled Urza's tron lands), so pay X for
        # exactly the tapped ones and target those, instead of letting the generic
        # picker choose X=0 / the first land. The pending-decision source names the
        # ability making each choice even before it hits the stack, so it can't be
        # confused with any other X ability or land-targeting effect.
        if (cfg.use_tron_synergy
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
                and not _is_doomsday_deck(obs)):
            return self._target_choice(g(), cats, card_ids, ctrl_arr)

        # Tron synergy: Karn, the Great Creator's -2 wishes an artifact from the
        # sideboard/exile into hand — grab Cityscape Leveler, Tron's marquee wish
        # target. Its fetch is a hidden reveal, so the choice can arrive as a search
        # or a choose-card menu; match by pending source + card id across either.
        if (cfg.use_tron_synergy
                and _slot_card_idx(obs, _PENDING_DECISION_START) == _KARN_GREAT_CREATOR_VOCAB_IDX):
            for i in range(num_choices):
                if _action_card_id(card_ids, i) == _CITYSCAPE_LEVELER_VOCAB_IDX:
                    return i

        # Tron synergy: once Cityscape Leveler is in hand (typically off a Karn wish),
        # prioritize casting it — it's the deck's premier threat/artifact-and-land
        # destruction payoff, so cast it ahead of any other affordable spell.
        if cfg.use_tron_synergy and any(c == _CAT_CAST for c in cats):
            for i, c in enumerate(cats):
                if c == _CAT_CAST and _action_card_id(card_ids, i) == _CITYSCAPE_LEVELER_VOCAB_IDX:
                    return i

        # Tron synergy: when activating Karn, the Great Creator, prefer his -2 (wish an
        # artifact — Cityscape Leveler — from the sideboard into hand) over the +1
        # (Animate). The two loyalty abilities are indistinguishable in the observation
        # (same ACTIVATE category + Karn card id), differing only in menu order: the
        # engine emits them in script order, +1 Animate then -2 ChangeZone, so the LAST
        # Karn activation offered is the wish. Only force it when both are on the menu
        # (Karn has the 2 loyalty to spend); with just +1 left, greedy takes it.
        if cfg.use_tron_synergy:
            karn_acts = [i for i, c in enumerate(cats)
                         if c == _CAT_ACTIVATE
                         and _action_card_id(card_ids, i) == _KARN_GREAT_CREATOR_VOCAB_IDX]
            if len(karn_acts) >= 2:
                return karn_acts[-1]

        # Tron synergy: once we control at least one of Urza's Mine/Power-Plant/
        # Tower, prefer playing a hand land that completes the set (a missing
        # real piece, else Planar Nexus) over any other offered land drop.
        if cfg.use_tron_synergy and any(c == _CAT_LAND for c in cats):
            controlled = _controlled_tron_pieces(obs)
            if controlled:
                pick = _tron_choice(cats, card_ids, _CAT_LAND, controlled)
                if pick is not None:
                    return pick

        # Tron synergy: a library search (Expedition Map, fetch effects) finds
        # the Tron piece we're missing, or Planar Nexus as a substitute, ahead
        # of any other offered card.
        if cfg.use_tron_synergy and any(c == _CAT_SEARCH for c in cats):
            pick = _tron_choice(cats, card_ids, _CAT_SEARCH, _controlled_tron_pieces(obs))
            if pick is not None:
                return pick

        # Anything not explicitly improved: proven GREEDY behaviour.
        return _greedy_action(obs, num_choices, fruitless)

    @staticmethod
    def _confirm(cats, confirm_cat: int) -> int:
        for i, c in enumerate(cats):
            if c == confirm_cat:
                return i
        return 0

    def _mulligan_choice(self, g: dict, cats) -> int:
        """Keep a hand with a sane land count; otherwise mulligan.

        A one-land hand is keepable if it contains a cheap card-selection spell
        (Brainstorm, Ponder, Once Upon a Time) that can dig toward more lands.
        Hands with 5 or more lands are too land-heavy to keep.
        """
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
        cats = np.round(obs[STATE_SIZE:STATE_SIZE + num_choices]
                        * ACTION_CATEGORY_MAX).astype(int)
        # A game always opens with a mulligan query, so seeing one means a new game
        # started: clear the novelty set even when nobody called new_game() (e.g. a
        # scripted opponent reused across vec-env episodes). Repeated clears during
        # London mulligans are harmless — nothing is cast pregame.
        if any(int(c) == _CAT_MULLIGAN for c in cats):
            self._explore_seen.clear()
        card_ids = obs[STATE_SIZE + MAX_ACTIONS:STATE_SIZE + 2 * MAX_ACTIONS]
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
