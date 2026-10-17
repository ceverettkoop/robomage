"""The scripted agent on the BQUERY observation vector.

The agent's decisions live in ``scripted_core`` (``ScriptedCore``), over an abstract view of a
decision. This module is the view that reads the engine's BQUERY float vector — the observation
the RL stack trains on — and the entry points its consumers use: ``ScriptedAgent.act(obs,
num_choices)``, ``make_agent``, ``scripted_action`` (PPO's scripted opponents, the AlphaZero
actor's scripted oracle, the gate tiers that play scripted games over BQUERY). The RMSTATE view
is ``rm_agent.FrameView``.

``VectorView`` answers every view read exactly as the agent read the vector before the core was
split out (its caps, slot orders, sentinels and float tests included), so the agent plays the
same games on BQUERY: the scans of the permanent slots skip empty ones and the phased-out ones
where they always did, an action is joined to a permanent by its card id (``exact_refs`` False),
a keyword-built ability is recognised by its ability index (``_keyword_ability_ordinal``), and
cards are vocab ids turned into vocab names (an id with no vocab name reads as ``#<id>``).

Card knowledge from outside the engine — the generated vocab tables (``card_costs``,
``card_props``) and the card scripts on disk, or the pass-1 RMSTATE adapter's
``rmstate_adapter.FrameCardKnowledge`` — is keyed by vocab id and installed module-wide with
``use_card_knowledge``; the view reads it by name through ``_NameCardKnowledge``.
"""

from pathlib import Path

import numpy as np

import _enums
from env import (
    STATE_SIZE, MAX_ACTIONS, ACTION_CATEGORY_MAX, N_CARD_TYPES, MAX_HAND_SLOTS,
    ACT_CATS_START, ACT_IDS_START, ACT_CTRL_START, ACT_REFS_START, N_ENTITY_REF_SLOTS,
    ACT_ZONE_START, ACT_ORDS_START, OPTION_ORDINAL_MAX, REF_ZONE_MAX,
    _BF_START, _BF_SLOT_SIZE, _PERM_A_SLOTS, _BF_CARD_OFF, _STACK_START, _STACK_SLOT_SIZE,
    _STACK_SLOTS, _HAND_START, _HAND_SLOT_SIZE, _CUR_TURN_IDX, _GY_START, _GY_SLOT_SIZE,
    MAX_GY_SLOTS, _EXILE_START, _EXILE_SLOT_SIZE, EXILE_COUNTERS_OFF, ZONE_COUNTER_NORMALIZER,
    _OFF_IS_TAPPED, _OFF_IS_ATTACKING, _OFF_IS_BLOCKING, _OFF_IS_BLOCKED, _OFF_HAS_SICKNESS,
    _OFF_IS_CREATURE, _OFF_POWER, _OFF_TOUGHNESS, _OFF_IS_LAND, _OFF_LOYALTY,
    _OFF_OTHER_COUNTERS, _OFF_ATTACHED_TO, _OFF_ATTACHED_BY, _OFF_IS_PHASED_OUT,
    _OFF_KEYWORDS_START, _PB_MANA, _PB_LIFE, _SELF_BLOCK_START, _OPP_BLOCK_START,
    _slot_card_idx, _ACTION_CARD_ID_NULL, _TOKEN_VOCAB_BASE,
    _PENDING_DECISION_START, _LIBRARY_CTX_START, _KNOWN_TOP_LIB_START, _KNOWN_TOP_LIB_SLOT_SIZE,
    _SELF_IS_A_IDX, _STEP_FIRST_MAIN_IDX, _STEP_SECOND_MAIN_IDX, _IS_ACTIVE_IDX,
    _STEP_ONEHOT_START, _STEP_ONEHOT_SIZE, _EXTRAS_SELF_MULLIGANS, MULLIGAN_NORMALIZER,
)
from decode import (EXILE_SLOT_SIZE, _OPP_EXILE_START, _OPP_GY_START, card_types,
                    decode_opp_decklist, _card_script_lines, _decode_graveyard,
                    _decode_opp_known_hand)
from card_costs import _LAND_VOCAB_IDS, _CARD_COST_MATRIX, _VOCAB_NAMES
from card_props import _CARD_PROP_MATRIX, _PROP_NAMES
from _enums import _OBS_KEYWORDS
from scripted_core import (  # noqa: F401  (Skill, AgentConfig: the public agent API)
    AgentConfig, ScriptedCore, Skill, preset_config,
)


class VocabCardKnowledge:
    """The agent's card knowledge from outside the engine: the generated vocab
    tables (``card_costs``, ``card_props``) and the card scripts on disk, by vocab
    id. Every card read of the vector view goes through the installed knowledge
    object (:func:`use_card_knowledge`), so the pass-1 RMSTATE driver can swap in
    the same answers built from the match's ``CARDS`` frame
    (``rmstate_adapter.FrameCardKnowledge``)."""

    def __init__(self):
        self._deck_cache: dict = {}

    def cost_row(self, cid: int):
        """The card's cast-cost row (W/U/B/R/G/C/generic pip counts / 10,
        float32), or None for an id outside the table."""
        return _CARD_COST_MATRIX[cid] if 0 <= cid < len(_CARD_COST_MATRIX) else None

    def prop(self, cid: int, col: int) -> float:
        """Column ``col`` of the card's property row (0 outside the table)."""
        if 0 <= cid < len(_CARD_PROP_MATRIX):
            return float(_CARD_PROP_MATRIX[cid, col])
        return 0.0

    def is_land(self, cid: int) -> bool:
        return cid in _LAND_VOCAB_IDS

    def script_lines(self, cid: int):
        """The script lines of the card's face, or None."""
        return _card_script_lines(cid)

    def types_line(self, cid: int) -> str:
        """The face's ``Types:`` value ('' when unknown; '' for a token)."""
        return card_types(cid)

    def deck_runs_card(self, deck_name: str, card_name: str):
        """Whether the named deck's list (main or sideboard) holds ``card_name``,
        from its ``.dk`` file; None when the file can't be read."""
        key = (deck_name, card_name)
        if key not in self._deck_cache:
            try:
                text = (_DECKS_DIR / f"{deck_name}.dk").read_text()
            except OSError:
                self._deck_cache[key] = None
            else:
                self._deck_cache[key] = card_name.lower() in text.lower()
        return self._deck_cache[key]


_DECKS_DIR = Path(__file__).resolve().parent.parent / "bin" / "resources" / "decks"


_VOCAB_CARD_KNOWLEDGE = VocabCardKnowledge()
_card_knowledge = _VOCAB_CARD_KNOWLEDGE


def use_card_knowledge(knowledge=None) -> None:
    """Install the card knowledge every vector-view card read consults (None: the
    vocab tables). Both seats of one match share it: a match's card table is the
    same for every viewer."""
    global _card_knowledge
    _card_knowledge = knowledge if knowledge is not None else _VOCAB_CARD_KNOWLEDGE


# Card name -> vocab id (card_vocab.h, via the generated cost table).
_VOCAB_ID = {name: i for i, name in enumerate(_VOCAB_NAMES) if name}


def card_name_mismatches(knowledge) -> list:
    """The cards a match's card table (``knowledge.names``: id -> name, the CARDS frame) names
    with another id than the vocab names: (name, frame id, vocab id). Empty when every name the
    frame and the vocab share has one id."""
    return [(name, cid, _VOCAB_ID[name]) for cid, name in sorted(knowledge.names.items())
            if name in _VOCAB_ID and _VOCAB_ID[name] != cid]


def _id_name(cid: int):
    """A vocab id as the view's card name: its vocab name, ``#<id>`` for an id with none, None
    for no card (a negative id)."""
    if cid < 0:
        return None
    name = _VOCAB_NAMES[cid] if cid < len(_VOCAB_NAMES) else ""
    return name or f"#{cid}"


def _name_id(name) -> int:
    """The vocab id a view card name stands for (-1 for None or an unknown name)."""
    if name is None:
        return -1
    cid = _VOCAB_ID.get(name)
    if cid is not None:
        return cid
    if name.startswith("#") and name[1:].isdigit():
        return int(name[1:])
    return -1


_PROP_COL = {name: i for i, name in enumerate(_PROP_NAMES)}
# Colour index (W/U/B/R/G, the cost matrix's pip order) of each basic land type.
_LAND_TYPE_COLOR = (("land_plains", 0), ("land_island", 1), ("land_swamp", 2),
                    ("land_mountain", 3), ("land_forest", 4))
# How many activated abilities each ability-building keyword adds after the
# A: lines (src/parse.cpp): Reconfigure is an attach plus an unattach.
_KEYWORD_ABILITY_COUNT = {"Equip": 1, "Reconfigure": 2, "Ninjutsu": 1, "Cycling": 1}


def _keyword_ability_ordinal(knowledge, cid: int, keyword: str) -> int:
    """Ability index the engine gives a keyword-built activated ability
    (Equip, Reconfigure's attach, Ninjutsu): the card's A: abilities come
    first, then the keyword abilities in script order (src/parse.cpp).
    -1 when the card lacks the keyword."""
    lines = (knowledge.script_lines(cid) if cid >= 0 else None) or []
    ordinal = sum(1 for line in lines if line.startswith("A:"))
    for line in lines:
        if not line.startswith("K:"):
            continue
        name = line[2:].split(":", 1)[0]
        if name == keyword:
            return ordinal
        ordinal += _KEYWORD_ABILITY_COUNT.get(name, 0)
    return -1


class _NameCardKnowledge:
    """The core's card-knowledge interface (by card name) over an id-keyed knowledge object
    (``VocabCardKnowledge`` / the pass-1 ``FrameCardKnowledge``)."""

    __slots__ = ("k",)

    def __init__(self, knowledge):
        self.k = knowledge

    def _prop(self, name, prop: str) -> float:
        return self.k.prop(_name_id(name), _PROP_COL[prop])

    def mana_value(self, name) -> float:
        """Printed mana value via the cast-cost row: the 7 pip counts (W/U/B/R/G/C/generic)
        divided by 10, so the row sum times 10 is the card's total mana value (X contributes 0),
        rounded back to a whole number (the float32 row sum of a mana value 4 card is
        4.00000006, which compares greater than 4). Unknown ids count as 0."""
        row = self.k.cost_row(_name_id(name))
        if row is not None:
            return float(round(float(row.sum()) * 10.0))
        return 0.0

    def cost_colors(self, name):
        """The colours (W/U/B/R/G as 0-4) of the card's coloured pips, None when unknown."""
        row = self.k.cost_row(_name_id(name))
        if row is None:
            return None
        return {col for col in range(5) if row[col] > 0}

    def is_land(self, name) -> bool:
        return self.k.is_land(_name_id(name))

    def power(self, name) -> int:
        """Printed power (``card_props.power`` is power / 10)."""
        return int(round(self._prop(name, "power") * 10))

    def is_delve(self, name) -> bool:
        return self._prop(name, "kw_delve") > 0.5

    def is_equipment(self, name) -> bool:
        return self._prop(name, "sub_equipment") > 0.5 or self._prop(name, "kw_equip") > 0.5

    def has_miracle(self, name) -> bool:
        """True if the card's script carries ``K:Miracle``."""
        cid = _name_id(name)
        if cid < 0:
            return False
        return any(line.startswith("K:Miracle") for line in (self.k.script_lines(cid) or []))

    def is_legendary(self, name) -> bool:
        return self._prop(name, "legendary") > 0.5

    def land_colors(self, name) -> set:
        """Colours a land taps for through its basic land types (Scrubland: W, B).
        Lands without a basic type count as colourless here."""
        return {col for prop, col in _LAND_TYPE_COLOR if self._prop(name, prop) > 0.5}

    def deals_damage(self, name) -> bool:
        return self._prop(name, "cat_dealdamage") > 0.5

    def has_nonmana_activation(self, name) -> bool:
        """An ``A:`` line other than ``AB$ Mana``."""
        return any(line.startswith("A:") and "AB$ Mana" not in line
                   for line in (self.k.script_lines(_name_id(name)) or []))

    def is_artifact_or_enchantment(self, name) -> bool:
        types = self.k.types_line(_name_id(name)).split()
        return "Artifact" in types or "Enchantment" in types


# Engine enum names of the BQUERY ints (the RMSTATE schema spells the same names).
_CAT_NAME = [name for _v, name in sorted((v, k[len("CAT_"):]) for k, v in vars(_enums).items()
                                         if k.startswith("CAT_") and isinstance(v, int))]
assert len(_CAT_NAME) == ACTION_CATEGORY_MAX + 1
_REF_NAME = [name for _v, name in sorted((v, k[len("REF_"):]) for k, v in vars(_enums).items()
                                         if k.startswith("REF_") and k != "REF_ZONE_MAX"
                                         and isinstance(v, int))]
assert len(_REF_NAME) == REF_ZONE_MAX + 1
_STEP_NAME = ["UNTAP", "UPKEEP", "DRAW", "FIRST_MAIN", "BEGIN_COMBAT", "DECLARE_ATTACKERS",
              "DECLARE_BLOCKERS", "FIRST_STRIKE_DAMAGE", "COMBAT_DAMAGE", "END_OF_COMBAT",
              "SECOND_MAIN", "END_STEP", "CLEANUP"]
assert len(_STEP_NAME) == _STEP_ONEHOT_SIZE

_NULL_CARD_TEST = _ACTION_CARD_ID_NULL + 0.01
_SELF_LIFE_IDX = _SELF_BLOCK_START + _PB_LIFE
_OPP_LIFE_IDX = _OPP_BLOCK_START + _PB_LIFE


class _Perm:
    """A permanent slot as the view reads it. Its card's characteristics (``mana_value``,
    ``art_ench``, ``legendary``) are read from the card knowledge on first use."""

    __slots__ = ("card", "token", "tapped", "sick", "creature", "land", "power", "toughness",
                 "attacking", "blocking", "blocked", "loyalty", "other_counters", "attached",
                 "has_attachment", "keywords", "phased", "mana_value", "art_ench", "legendary",
                 "slot", "cards")

    def __getattr__(self, name):
        if name == "mana_value":
            value = self.cards.mana_value(self.card)
        elif name == "art_ench":
            value = self.cards.is_artifact_or_enchantment(self.card)
        elif name == "legendary":
            value = self.cards.is_legendary(self.card)
        else:
            raise AttributeError(name)
        object.__setattr__(self, name, value)
        return value


class _StackItem:
    __slots__ = ("card", "mine", "spell")

    def __init__(self, card, mine, spell):
        self.card = card
        self.mine = mine
        self.spell = spell


class VectorView:
    """One BQUERY decision (the observation vector and its menu) as the core's view
    (``scripted_core`` module docstring). Lazy: each part is read on first use."""

    exact_refs = False
    pending_perm = None

    def __init__(self, obs, num_choices):
        self.obs = obs
        self.n = num_choices
        cats = np.round(obs[ACT_CATS_START:ACT_CATS_START + num_choices] * ACTION_CATEGORY_MAX).astype(int)
        self.cats = [_CAT_NAME[c] for c in cats.tolist()]
        self.cards = _NameCardKnowledge(_card_knowledge)
        self._perms = None
        self._stack = None
        self._hand = None

    # ── menu ────────────────────────────────────────────────────────────────────────────────
    def card_id(self, i) -> int:
        return int(round(float(self.obs[ACT_IDS_START + i]) * N_CARD_TYPES))

    def card(self, i):
        return _id_name(self.card_id(i))

    def nonnull_card(self, i) -> bool:
        """The menu-shape rules' null-card test: the normalised id above the null sentinel plus
        0.01 (ids 0-9 read as null)."""
        return float(self.obs[ACT_IDS_START + i]) > _NULL_CARD_TEST

    def same_card(self, i, j) -> bool:
        """The normalised ids within 0.001 of each other (ids one apart count as the same)."""
        return abs(self.obs[ACT_IDS_START + j] - self.obs[ACT_IDS_START + i]) < 0.001

    def ctrl(self, i) -> int:
        f = float(self.obs[ACT_CTRL_START + i])
        return 1 if f > 0.5 else (-1 if f < -1e-4 else 0)

    def zone(self, i) -> str:
        return _REF_NAME[int(round(float(self.obs[ACT_ZONE_START + i]) * REF_ZONE_MAX))]

    def ordinal(self, i) -> int:
        return int(round(float(self.obs[ACT_ORDS_START + i]) * (OPTION_ORDINAL_MAX + 1))) - 1

    def ref(self, i) -> int:
        """The action's entity-reference slot (-1 none; 0-47 own permanents, 48-95 the
        opponent's, 96-107 the stack)."""
        return int(round(float(self.obs[ACT_REFS_START + i]) * N_ENTITY_REF_SLOTS)) - 1

    def ref_perm(self, i):
        ref = self.ref(i)
        if not 0 <= ref < 2 * _PERM_A_SLOTS:
            return None
        return self._perm_slots().get(ref)

    def is_keyword_ability(self, i, keyword) -> bool:
        return self.ordinal(i) == _keyword_ability_ordinal(_card_knowledge, self.card_id(i), keyword)

    # ── players and turn ────────────────────────────────────────────────────────────────────
    @property
    def seat_is_a(self) -> bool:
        return self.obs[_SELF_IS_A_IDX] > 0.5

    @property
    def is_active(self) -> bool:
        return self.obs[_IS_ACTIVE_IDX] > 0.5

    @property
    def in_main(self) -> bool:
        return self.obs[_STEP_FIRST_MAIN_IDX] > 0.5 or self.obs[_STEP_SECOND_MAIN_IDX] > 0.5

    @property
    def step(self):
        vec = self.obs[_STEP_ONEHOT_START:_STEP_ONEHOT_START + _STEP_ONEHOT_SIZE]
        idx = int(np.argmax(vec))
        return _STEP_NAME[idx] if vec[idx] >= 0.5 else None

    @property
    def life(self) -> int:
        return int(round(float(self.obs[_SELF_LIFE_IDX]) * 20))

    @property
    def opp_life(self) -> int:
        return int(round(self.obs[_OPP_LIFE_IDX] * 20))

    @property
    def library(self) -> int:
        return int(round(float(self.obs[_LIBRARY_CTX_START]) * 60))

    @property
    def turn(self) -> int:
        return int(round(float(self.obs[_CUR_TURN_IDX]) * 50))

    def pool(self, color) -> int:
        return int(round(float(self.obs[_SELF_BLOCK_START + _PB_MANA + color]) * 10))

    @property
    def pool_total(self) -> int:
        return sum(self.pool(k) for k in range(6))

    @property
    def mulligans(self) -> int:
        return int(round(float(self.obs[_EXTRAS_SELF_MULLIGANS]) * MULLIGAN_NORMALIZER))

    @property
    def pending(self):
        return _id_name(_slot_card_idx(self.obs, _PENDING_DECISION_START))

    # ── objects ─────────────────────────────────────────────────────────────────────────────
    @property
    def hand(self) -> list:
        if self._hand is None:
            ids = [_slot_card_idx(self.obs, _HAND_START + slot * _HAND_SLOT_SIZE)
                   for slot in range(MAX_HAND_SLOTS)]
            self._hand = [_id_name(c) for c in ids if c >= 0]
        return self._hand

    def _perm_slots(self) -> dict:
        """slot -> _Perm for every non-empty permanent slot (phased-out ones included)."""
        if self._perms is None:
            obs = self.obs
            cards = self.cards
            out = {}
            for slot in range(2 * _PERM_A_SLOTS):
                base = _BF_START + slot * _BF_SLOT_SIZE
                cid = _slot_card_idx(obs, base + _BF_CARD_OFF)
                if cid < 0:
                    continue
                p = _Perm()
                p.slot = slot
                p.card = _id_name(cid)
                p.token = cid >= _TOKEN_VOCAB_BASE
                p.tapped = obs[base + _OFF_IS_TAPPED] > 0.5
                p.sick = obs[base + _OFF_HAS_SICKNESS] > 0.5
                p.creature = obs[base + _OFF_IS_CREATURE] > 0.5
                p.land = obs[base + _OFF_IS_LAND] > 0.5
                if p.creature:
                    p.power = int(round(float(obs[base + _OFF_POWER]) * 10))
                    p.toughness = int(round(float(obs[base + _OFF_TOUGHNESS]) * 10))
                else:
                    p.power = p.toughness = 0
                p.attacking = obs[base + _OFF_IS_ATTACKING] > 0.5
                p.blocking = obs[base + _OFF_IS_BLOCKING] > 0.5
                p.blocked = obs[base + _OFF_IS_BLOCKED] > 0.5
                p.loyalty = int(round(float(obs[base + _OFF_LOYALTY]) * 10))
                p.other_counters = int(round(float(obs[base + _OFF_OTHER_COUNTERS]) * 10))
                p.attached = obs[base + _OFF_ATTACHED_TO] > 1e-6
                p.has_attachment = int(round(float(obs[base + _OFF_ATTACHED_BY]) * N_ENTITY_REF_SLOTS)) - 1 >= 0
                p.keywords = tuple(kw for k, kw in enumerate(_OBS_KEYWORDS)
                                   if obs[base + _OFF_KEYWORDS_START + k] > 0.5)
                p.phased = obs[base + _OFF_IS_PHASED_OUT] > 0.5
                p.cards = cards
                out[slot] = p
            self._perms = out
            self.my_perms_all = [p for s, p in out.items() if s < _PERM_A_SLOTS]
            self.my_perms = [p for p in self.my_perms_all if not p.phased]
            self.opp_perms = [p for s, p in out.items() if s >= _PERM_A_SLOTS and not p.phased]
        return self._perms

    def __getattr__(self, name):
        # The permanent lists are built together on first use (_perm_slots).
        if name in ("my_perms", "my_perms_all", "opp_perms"):
            self._perm_slots()
            return self.__dict__[name]
        raise AttributeError(name)

    def _stack_slots(self):
        if self._stack is None:
            items = []
            for k in range(_STACK_SLOTS):
                base = _STACK_START + k * _STACK_SLOT_SIZE
                items.append(_StackItem(_id_name(_slot_card_idx(self.obs, base + 1)),
                                        self.obs[base] > 0.5, self.obs[base + 2] > 0.5))
            self._stack = items
        return self._stack

    @property
    def stack(self) -> list:
        """The stack slots holding a card, top first."""
        return [s for s in self._stack_slots() if s.card is not None]

    @property
    def stack_top(self):
        """Stack slot 0 as it stands (a slot with no card reads card None)."""
        return self._stack_slots()[0]

    @property
    def stack_empty(self) -> bool:
        return all(s.card is None for s in self._stack_slots())

    @property
    def graveyard_cards(self) -> set:
        """Both graveyards' cards (64 slots each, self first)."""
        out = set()
        for slot in range(2 * MAX_GY_SLOTS):
            cid = _slot_card_idx(self.obs, _GY_START + slot * _GY_SLOT_SIZE)
            if cid >= 0:
                out.add(_id_name(cid))
        return out

    @property
    def opp_graveyard(self) -> list:
        return [n for n in _decode_graveyard(self.obs[:STATE_SIZE], _OPP_GY_START) if n in _VOCAB_ID]

    @property
    def opp_exile(self) -> list:
        return [n for n in _decode_graveyard(self.obs[:STATE_SIZE], _OPP_EXILE_START, EXILE_SLOT_SIZE)
                if n in _VOCAB_ID]

    @property
    def opp_exile_counters(self) -> list:
        out = []
        for slot in range(MAX_GY_SLOTS):
            base = _EXILE_START + (MAX_GY_SLOTS + slot) * _EXILE_SLOT_SIZE
            cid = _slot_card_idx(self.obs, base)
            if cid < 0:
                continue
            out.append((_id_name(cid),
                        int(round(float(self.obs[base + EXILE_COUNTERS_OFF] * ZONE_COUNTER_NORMALIZER)))))
        return out

    # ── memory ──────────────────────────────────────────────────────────────────────────────
    def known_top(self, n) -> list:
        return [_id_name(_slot_card_idx(self.obs, _KNOWN_TOP_LIB_START + k * _KNOWN_TOP_LIB_SLOT_SIZE))
                for k in range(n)]

    def top_candidates(self, n) -> list:
        return [None if c is None else {c} for c in self.known_top(n)]

    @property
    def opp_known_hand(self) -> list:
        return [n for n in _decode_opp_known_hand(self.obs[:STATE_SIZE]) if n in _VOCAB_ID]

    def opp_hand_possible(self, _name):
        return None

    def own_library_has(self, _name):
        return None

    @property
    def opp_deck(self) -> dict:
        deck: dict = {}
        for cid, ct, _rev in decode_opp_decklist(self.obs[:STATE_SIZE]):
            name = _id_name(cid)
            deck[name] = deck.get(name, 0) + ct
        return deck

    def opp_runs_card(self, card, deck_name):
        return _card_knowledge.deck_runs_card(deck_name, card) if deck_name else None


class ScriptedAgent(ScriptedCore):
    """The scripted agent on the BQUERY vector: ``act(obs, num_choices)``.

    ``act`` is a pure function of the (perspective-relative) observation, except
    for the RANDOM tier which draws from a per-agent seeded numpy generator so parallel
    vec-env subprocesses stay reproducible and independent of the global RNG, and the
    per-game sets of the EXPLORE and GREEDY/HEURISTIC tiers (``ScriptedCore``).
    """

    def __init__(self, config: AgentConfig = AgentConfig()):
        super().__init__(config)
        self._rng = np.random.default_rng(config.rng_seed)

    def _random_index(self, n: int) -> int:
        return int(self._rng.integers(0, n))

    def act(self, obs: np.ndarray, num_choices: int) -> int:
        return self.decide(VectorView(obs, num_choices))


def make_agent(spec: str = "scripted") -> ScriptedAgent:
    """Resolve a spec string into a ScriptedAgent (``scripted_core.preset_config``)."""
    return ScriptedAgent(preset_config(spec))


# ── Module-level public entry point ─────────────────────────────────────────
# ``scripted_action`` is the bare function form used by callers that don't build a
# ScriptedAgent (analysis/baseline/observe scripted opponent, tui_game, bench). It
# defaults to the "hard" heuristic agent, matching make_agent("scripted"), so
# every default scripted call plays at the hard tier. The heuristic is a pure
# function of (obs, num_choices) except for the per-game fruitless-activation
# stall guard, which is seat-keyed and self-clearing on the pregame mulligan
# (see ScriptedCore._hard_act), so one shared instance stays safe across games/processes and
# across both seats. Callers wanting the old greedy behaviour build
# make_agent("easy"/"greedy") explicitly.
_DEFAULT_AGENT = make_agent("scripted")


def scripted_action(obs: np.ndarray, num_choices: int) -> int:
    return _DEFAULT_AGENT.act(obs, num_choices)
