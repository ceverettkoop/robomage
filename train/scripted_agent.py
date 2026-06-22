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
diag/watch determinism depends on it.

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
    # battlefield / stack layout
    _BF_START, _BF_SLOT_SIZE, _PERM_A_SLOTS, _BF_CARD_OFF, _STACK_START,
    _STACK_SLOT_SIZE, _HAND_START,
    _OFF_IS_TAPPED, _OFF_IS_ATTACKING, _OFF_HAS_SICKNESS, _OFF_IS_CREATURE,
    _OFF_IS_LAND,
    # vocab / targeting constants
    _ACTION_CARD_ID_NULL, _ACTION_CTRL_NULL, _BLUE_POOL_IDX, _WASTELAND_VOCAB_IDX,
    _BASIC_LAND_IDS, _COUNTER_SPELL_VOCAB_IDS, _COUNTERSPELL_VOCAB_IDX,
    _DOOMSDAY_VOCAB_IDX, _DARK_RITUAL_VOCAB_IDX, _THASSAS_ORACLE_VOCAB_IDX,
    _STREET_WRAITH_VOCAB_IDX, _EDGE_OF_AUTUMN_VOCAB_IDX, _DOOMSDAY_DECK_IDS,
    # shared helper (also used by env's reward shaping, so it stays in env.py)
    _hand_has_card,
)
from decode import decode_game_state
from card_costs import _LAND_VOCAB_IDS


class Skill(Enum):
    """Scripted-agent skill tiers.

    RANDOM    — picks uniformly among the legal actions (a weak floor opponent).
    GREEDY    — the original rule-based behaviour; the baseline opponent.
    HEURISTIC — GREEDY's scan with selected decision points overridden by
                single-ply heuristics (board eval, combat sim, smarter targeting).
    """
    RANDOM = "random"
    GREEDY = "greedy"
    HEURISTIC = "heuristic"


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
    enable_combo_lines: bool = True     # keep deck-specific combo logic (Doomsday etc.)
    rng_seed: int | None = None         # seed for RANDOM's generator (reproducible)


# Canonical presets resolved from a spec-string suffix (see make_agent).
_PRESETS: dict[str, AgentConfig] = {
    "random":    AgentConfig(skill=Skill.RANDOM, enable_combo_lines=False),
    "easy":      AgentConfig(skill=Skill.GREEDY),
    "greedy":    AgentConfig(skill=Skill.GREEDY),
    "scripted":  AgentConfig(skill=Skill.GREEDY),     # default alias == today
    "hard":      AgentConfig(skill=Skill.HEURISTIC, use_combat_sim=True,
                             use_eval_targeting=True, use_smart_mulligan=True),
    "heuristic": AgentConfig(skill=Skill.HEURISTIC, use_combat_sim=True,
                             use_eval_targeting=True, use_smart_mulligan=True),
}


def _action_card_id(card_ids: np.ndarray, i: int) -> int:
    """Decode the card vocab index from the action's card_id float."""
    return int(round(float(card_ids[i]) * N_CARD_TYPES))


def _is_doomsday_deck(obs: np.ndarray) -> bool:
    """Heuristic: check if any hand card is a doomsday-deck-only card."""
    for slot in range(MAX_HAND_SLOTS):
        base = _HAND_START + slot * N_CARD_TYPES
        card_vec = obs[base:base + N_CARD_TYPES]
        idx = int(np.argmax(card_vec))
        if card_vec[idx] > 0.5 and idx in _DOOMSDAY_DECK_IDS:
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
        card_vec = obs[base + _BF_CARD_OFF : base + _BF_CARD_OFF + N_CARD_TYPES]
        idx = int(np.argmax(card_vec))
        if card_vec[idx] > 0.5 and idx not in _BASIC_LAND_IDS:
            return True
    return False


def _opponent_has_spell_on_stack(obs: np.ndarray) -> bool:
    """Return True if at least one spell/ability on the stack is not controlled by self."""
    for i in range(12):
        base = _STACK_START + i * _STACK_SLOT_SIZE
        ctrl_is_self = obs[base]
        card_vec = obs[base + 1 : base + 1 + N_CARD_TYPES]
        if np.max(card_vec) > 0.5 and ctrl_is_self < 0.5:
            return True
    return False


def _greedy_action(obs: np.ndarray, num_choices: int) -> int:
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

    # 5b. Paying costs (tapping lands for mana during spell/ability payment, delve exile).
    #     Pick the first available option — this taps a source to pay the cost.
    if any(c == _CAT_PAYING for c in cats):
        return 0

    # 6. Cast spells.
    #    Counter spells (Counterspell, Daze, Force of Will) require an opponent's spell
    #    on the stack; skip them when the stack holds only own spells or is empty.
    opponent_spell_on_stack = _opponent_has_spell_on_stack(obs)

    # Priority: if opponent has a spell on stack and we have UU, cast Counterspell first.
    if opponent_spell_on_stack and int(round(obs[_BLUE_POOL_IDX] * 10)) >= 2:
        for i, c in enumerate(cats):
            if c == _CAT_CAST and _action_card_id(card_ids, i) == _COUNTERSPELL_VOCAB_IDX:
                return i

    # --- Doomsday deck rules ---
    has_doomsday_in_hand = _hand_has_card(obs, _DOOMSDAY_VOCAB_IDX)

    # Always cast Doomsday when offered
    for i, c in enumerate(cats):
        if c == _CAT_CAST and _action_card_id(card_ids, i) == _DOOMSDAY_VOCAB_IDX:
            return i

    # Only cast Dark Ritual if Doomsday is also in hand
    # (don't waste ritual mana without the combo piece)
    for i, c in enumerate(cats):
        if c == _CAT_CAST and _action_card_id(card_ids, i) == _DARK_RITUAL_VOCAB_IDX:
            if has_doomsday_in_hand:
                return i
            # else skip it

    for i, c in enumerate(cats):
        if c == _CAT_CAST:
            cid = _action_card_id(card_ids, i)
            if cid in _COUNTER_SPELL_VOCAB_IDS and not opponent_spell_on_stack:
                continue
            if cid == _DARK_RITUAL_VOCAB_IDX:
                continue  # handled above
            return i

    # 7. Play land (may unlock mana for a spell next query)
    for i, c in enumerate(cats):
        if c == _CAT_LAND:
            return i

    # 8. Activate non-mana abilities:
    #    - Cycling (Street Wraith, Edge of Autumn): always cycle immediately (any time)
    #    - Fetch lands: tap + pay 1 life + sacrifice → search for land
    #    - Wasteland: tap + sacrifice → destroy target nonbasic land
    #    - LED: activate when Doomsday is on the stack (discard hand for 3 mana)

    # Always cycle Street Wraith (pay 2 life, draw a card)
    for i, c in enumerate(cats):
        if c == _CAT_ACTIVATE and _action_card_id(card_ids, i) == _STREET_WRAITH_VOCAB_IDX:
            return i

    # Always cycle Edge of Autumn (sac a land, draw a card) — only offered when legal
    for i, c in enumerate(cats):
        if c == _CAT_ACTIVATE and _action_card_id(card_ids, i) == _EDGE_OF_AUTUMN_VOCAB_IDX:
            return i

    if in_main_phase:
        for i, c in enumerate(cats):
            if c == _CAT_ACTIVATE:
                cid = _action_card_id(card_ids, i)
                if cid == _WASTELAND_VOCAB_IDX and not _opponent_has_nonbasic_land(obs):
                    continue  # no opponent nonbasic land to target — skip
                return i

    # 9. (Removed - mana abilities no longer offered during normal priority in machine mode;
    #     mana is tapped automatically during cost payment via PAYING_COSTS.)

    # 10. Doomsday pile ordering (TOP_LIBRARY): put Thassa's Oracle on bottom of pile
    #     (first pick = deepest position). For all other cards, pick action 0.
    if any(c == _CAT_TOP_LIBRARY for c in cats):
        # Prefer Thassa's Oracle first (goes to bottom of pile = wins with empty library)
        for i, c in enumerate(cats):
            if c == _CAT_TOP_LIBRARY and _action_card_id(card_ids, i) == _THASSAS_ORACLE_VOCAB_IDX:
                return i
        return 0  # pick first available for remaining cards

    # 10b. Dig choice (Once Upon a Time): pick action 1 (first matching card) if available.
    if any(c == _CAT_DIG for c in cats):
        return 1 if num_choices > 1 else 0

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
    vec-env subprocesses stay reproducible and independent of the global RNG.
    """

    def __init__(self, config: AgentConfig = AgentConfig()):
        self.config = config
        # Per-agent generator: only consulted by the RANDOM tier.
        self._rng = np.random.default_rng(config.rng_seed)

    def act(self, obs: np.ndarray, num_choices: int) -> int:
        if self.config.skill is Skill.RANDOM:
            return self._random_action(num_choices)
        if self.config.skill is Skill.HEURISTIC:
            return self._heuristic_action(obs, num_choices)
        return _greedy_action(obs, num_choices)

    # ------------------------------------------------------------------
    # HEURISTIC decision points (everything else falls back to GREEDY).
    # ------------------------------------------------------------------

    def _heuristic_action(self, obs: np.ndarray, num_choices: int) -> int:
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

        # Eval-based targeting — skipped for combo decks (R1: never disturb Doomsday).
        if (cfg.use_eval_targeting and any(c == _CAT_TARGET for c in cats)
                and not _is_doomsday_deck(obs)):
            return self._target_choice(g(), cats, card_ids, ctrl_arr)

        # Anything not explicitly improved: proven GREEDY behaviour.
        return _greedy_action(obs, num_choices)

    @staticmethod
    def _confirm(cats, confirm_cat: int) -> int:
        for i, c in enumerate(cats):
            if c == confirm_cat:
                return i
        return 0

    def _mulligan_choice(self, g: dict, cats) -> int:
        """Keep a hand with a sane land count; otherwise mulligan."""
        lands = sum(1 for c in g["self_hand"] if c["card_idx"] in _LAND_VOCAB_IDS)
        keepable = 2 <= lands <= 5
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


def make_agent(spec: str = "scripted") -> ScriptedAgent:
    """Resolve a spec string into a ScriptedAgent.

    Accepts ``"scripted"``, ``"scripted:hard"``, ``"scripted:easy"``,
    ``"scripted:random"``, or the bare suffixes ``"greedy"``/``"hard"``/
    ``"heuristic"``/``"easy"``/``"random"`` (case-insensitive).
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


# ── Back-compat module-level shim ───────────────────────────────────────────
# Historically the public entry point is ``scripted_action`` and it is GREEDY.
# Keep it an alias of the greedy implementation so behaviour stays byte-identical
# and the function identity is preserved for importers.
scripted_action = _greedy_action
