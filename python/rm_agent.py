"""The scripted agent on RMSTATE: one seat's decoded frames as the decision core's view.

Plan step 2, pass 2 (``docs/engine_fork_plan.md``). The agent's decisions are
``scripted_core.ScriptedCore``'s; this module is the view they read on the RMSTATE protocol, built
from the seat's decoded frames (``python/rmstate.py``) and the knowledge sidecar's ``k_*`` tables
(``docs/knowledge_store.md`` §6), with nothing copied from the engine's layout: tables, columns and
enum names come from the stream's SCHEMA frame, cards from its CARDS frame. No numpy.

- :class:`FrameCards` is the core's card knowledge, from the match's ``CARDS`` frame's structured
  columns (printed characteristics, keywords, subtypes, ability lists and their costs; no raw
  scripts), with :class:`LifeEffects`, what a card's oracle text and costs do to life.
- :class:`SeatFrames` follows one seat's frames in stream order (SCHEMA, CARDS, DECKLISTS,
  METAGAME, then STATE) and keeps what the view needs across them: the schema's enums, the card
  knowledge, the registered lists and the reference metagame.
- :class:`FrameView` is one decision frame as the core's view (``scripted_core`` docstring).
- :func:`decide` asks an agent (a ``ScriptedCore``) for a decision frame's action.

What the view reads (``docs/state_format.md`` Appendix A): ``header``, ``turn``, ``player``,
``object`` with ``object_keyword`` / ``object_counter``, ``stack``, ``pending`` with
``pending_cost`` / ``pending_cost_pip`` (the life a payment row pays), ``menu``,
``k_position`` / ``k_pool`` / ``k_pool_card`` / ``k_known_hand`` (memory), ``k_template`` (the
opponent's possible lists under ``--closed-decklist``). It has no caps (every permanent, hand
card, graveyard and exile card and stack object is there), reads summoning sickness as the rule
(``object.sick``: a creature, haste considered), all keywords, a token's real characteristics,
each permanent's own mana value and types, joins an action to its own object by handle
(``exact_refs``), recognises a keyword-built ability by the menu's ability key, and knows what the
knowledge sidecar knows (candidate sets of library positions, the cards that can be in the
opponent's hand, the own library's contents, the lists the opponent can be on).
"""
from __future__ import annotations

import re
import unicodedata
from functools import cached_property

_NONE8, _NONE16, _NONE32 = 0xFF, 0xFFFF, 0xFFFFFFFF
_SELF, _OPP = 1, 2                    # player-rel values
_BASIC_LAND_COLORS = {"Plains": 0, "Island": 1, "Swamp": 2, "Mountain": 3, "Forest": 4}
_PIP_COLORS = ("WHITE", "BLUE", "BLACK", "RED", "GREEN")    # the cost colours, as 0-4
# Oracle text of an effect that deals damage to a chosen target (a charm's or a sub-ability's,
# which the ability list's top-level categories do not show).
_TARGETED_DAMAGE = re.compile(r"\bdeals?\b[^.]*\bdamage\b[^.]*\b(any|targets?)\b", re.IGNORECASE)
# A life payment in a cost (the ``card_ability.cost`` notation): ``PayLife<N>`` / ``PayLife<X>``.
_PAY_LIFE = re.compile(r"PayLife<(\d+|X)>")
# Oracle sentences of what a card does to life (``LifeEffects``). The resolution ones are read
# from an instant's or sorcery's text only (a permanent's "you lose" sits in a trigger).
_LOSE_LIFE = re.compile(r"\byou lose (\d+) life\b", re.IGNORECASE)
_LOSE_HALF = re.compile(r"\byou lose half your life, rounded (up|down)\b", re.IGNORECASE)
_LOSE_TARGET_MV = re.compile(r"\byou lose life equal to its mana value\b", re.IGNORECASE)
_EACH_DAMAGE = re.compile(r"\bdeals (\d+) damage to each (?:creature and each )?player\b", re.IGNORECASE)
_EACH_LOSE = re.compile(r"\beach player loses (\d+) life\b", re.IGNORECASE)
_EACH_NONBASIC_DAMAGE = re.compile(r"\bdeals damage to each player equal to (twice )?the number of nonbasic "
                                   r"lands that player controls\b", re.IGNORECASE)
_YOU_DAMAGE = re.compile(r"\bdeals (\d+) damage to you\b", re.IGNORECASE)
_TARGET_DAMAGE_AMOUNT = re.compile(r"\bdeals (\d+) damage to (?:any target|target player\b|target creature or "
                                   r"player|target player or planeswalker)", re.IGNORECASE)
_ALT_COST = re.compile(r"\byou may ([^.]*?) rather than pay this spell's mana cost", re.IGNORECASE)
_KEYWORD_COST = re.compile(r"\b(Evoke|Flashback)\u2014([^(\n]*)")
_PAY_N_LIFE = re.compile(r"\bpay (\d+) life\b", re.IGNORECASE)
_ENTERS_PAY_LIFE = re.compile(r"\byou may pay (\d+) life\. If you don't", re.IGNORECASE)
_CAST_DAMAGE = re.compile(r"\bWhenever (a player|an opponent|you) casts? an? spell(?: with mana value (\d+) or "
                          r"less)?(, if no mana was spent to cast (?:that spell|it))?, [^.]*?\bdeals (\d+) damage "
                          r"to (?:that player|them|you)\b")

# Every table the view reads, of every frame kind (a driver may decode only these).
FRAME_TABLES = frozenset({
    "schema_table", "schema_column", "schema_enum",
    "card", "card_subtype", "card_keyword", "card_ability", "card_pip",
    "decklist", "decklist_card", "meta_list", "meta_list_card",
    "header", "turn", "player", "object", "object_keyword", "object_counter", "stack", "pending",
    "pending_cost", "pending_cost_pip", "menu", "match", "k_position", "k_pool", "k_pool_card",
    "k_known_hand", "k_template",
})


def fold_name(name: str) -> str:
    """A card name as the agent's constants spell it: accents folded to ASCII (Lórien Revealed
    is Lorien Revealed in the card vocabulary)."""
    if name.isascii():
        return name
    return "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))


def _card(v):
    """A ``card`` column value as an id, -1 for none."""
    return -1 if v == _NONE16 else v


def _bit(names, name):
    return 1 << names.index(name) if name in names else 0


# ── Card knowledge from the CARDS frame ─────────────────────────────────────────────────────

class _Face:
    __slots__ = ("row", "keywords", "subtypes", "abilities", "pip_colors", "phyrexian", "spell_cost", "life")

    def __init__(self, row):
        self.row = row
        self.keywords = set()
        self.subtypes = set()
        self.abilities = []          # (ability_kind name, category name)
        self.pip_colors = set()      # colours (0-4) of the hybrid and Phyrexian pips
        self.phyrexian = []          # each Phyrexian pip's colours (0-4)
        self.spell_cost = ""         # the spell ability's cost (its additional costs)
        self.life = None             # LifeEffects, built on first use


def _life_amount(cost):
    """(fixed life, whether an X life payment) a cost text pays."""
    fixed, has_x = 0, False
    for m in _PAY_LIFE.finditer(cost or ""):
        if m.group(1) == "X":
            has_x = True
        else:
            fixed += int(m.group(1))
    return fixed, has_x


def _first_int(rx, text, default=0):
    m = rx.search(text)
    return int(m.group(1)) if m else default


class LifeEffects:
    """What a card's printed text does to life, from its oracle text and its spell ability's cost
    (the scripted core's self-lethal guard, ``scripted_core`` docstring).

    - Costs: ``spell_life`` / ``spell_life_x`` (an additional cost's fixed life / an X life
      payment, Toxic Deluge), ``alt_life`` and ``alt_no_mana`` (the "rather than pay this spell's
      mana cost" alternative: the life it pays, whether it spends no mana), ``evoke_no_mana``,
      ``flashback_life``, ``enters_life`` (an "as it enters, you may pay N life" land).
    - An instant's or sorcery's resolution: ``lose`` (you lose N life), ``lose_half`` (``"up"`` /
      ``"down"``: you lose half your life), ``lose_target_mv`` (you lose life equal to its target's
      mana value, Reanimate), ``each_loss`` (damage or life loss to each player),
      ``each_nonbasic_damage`` (damage to each player per nonbasic land they control: the
      multiplier, Price of Progress).
    - ``you_damage`` (an ability's "deals N damage to you", Ancient Tomb), ``target_damage`` (the
      damage it deals to a chosen target), ``cast_damage`` (a cast trigger that damages the
      caster, Eidolon of the Great Revel / Roiling Vortex: (whose casts — ``"player"``,
      ``"opponent"`` or ``"you"`` —, damage, the most mana value it triggers on or None, whether
      only on a spell no mana was spent on), or None)."""

    __slots__ = ("spell_life", "spell_life_x", "alt_life", "alt_no_mana", "evoke_no_mana", "flashback_life",
                 "enters_life", "lose", "lose_half", "lose_target_mv", "each_loss", "each_nonbasic_damage",
                 "you_damage", "target_damage", "cast_damage")

    def __init__(self, oracle, spell_cost, resolves_once):
        self.spell_life, self.spell_life_x = _life_amount(spell_cost)
        alt = _ALT_COST.search(oracle)
        self.alt_life = _first_int(_PAY_N_LIFE, alt.group(1)) if alt else 0
        self.alt_no_mana = alt is not None and "{" not in alt.group(1)
        keyword_costs = {m.group(1): m.group(2) for m in _KEYWORD_COST.finditer(oracle)}
        self.evoke_no_mana = "Evoke" in keyword_costs and "{" not in keyword_costs["Evoke"]
        self.flashback_life = _first_int(_PAY_N_LIFE, keyword_costs.get("Flashback", ""))
        self.enters_life = _first_int(_ENTERS_PAY_LIFE, oracle)
        text = oracle if resolves_once else ""
        self.lose = _first_int(_LOSE_LIFE, text)
        half = _LOSE_HALF.search(text)
        self.lose_half = half.group(1).lower() if half else None
        self.lose_target_mv = bool(_LOSE_TARGET_MV.search(text))
        self.each_loss = _first_int(_EACH_DAMAGE, text) + _first_int(_EACH_LOSE, text)
        nonbasic = _EACH_NONBASIC_DAMAGE.search(text)
        self.each_nonbasic_damage = (2 if nonbasic.group(1) else 1) if nonbasic else 0
        self.you_damage = _first_int(_YOU_DAMAGE, oracle)
        self.target_damage = _first_int(_TARGET_DAMAGE_AMOUNT, oracle)
        cast = _CAST_DAMAGE.search(oracle)
        self.cast_damage = None
        if cast:
            who = {"a player": "player", "an opponent": "opponent", "you": "you"}[cast.group(1)]
            self.cast_damage = (who, int(cast.group(4)), int(cast.group(2)) if cast.group(2) else None,
                                bool(cast.group(3)))


_NO_LIFE_EFFECTS = LifeEffects("", "", False)


class FrameCards:
    """The decision core's card knowledge (by name) from one match's ``CARDS`` frame.

    A card id is a physical card (a double-faced card's front face); each face is a ``card`` row
    whose name names it. A name reads the first face row with that name: its printed mana value,
    plain pips (``cost_*``) and hybrid / Phyrexian pips (``card_pip``), types, supertypes, power,
    oracle text, its ``card_keyword`` / ``card_subtype`` rows and its ``card_ability`` rows (kind
    and category)."""

    def __init__(self, schema, cards_frame):
        en = schema.enums
        types, supers = en["card_type"], en["supertype"]
        self._land_bit = _bit(types, "Land")
        self._creature_bit = _bit(types, "Creature")
        self._one_shot_bits = _bit(types, "Instant") | _bit(types, "Sorcery")
        self._art_ench_bits = _bit(types, "Artifact") | _bit(types, "Enchantment")
        self._legendary_bit = _bit(supers, "Legendary")
        kw_names, sub_names = en["keyword"], en["subtype"]
        ab_kinds, effect_kinds = en["ability_kind"], en["effect_kind"]
        t = cards_frame.tables
        self.front = {}              # card id -> front face name
        self.back = {}               # card id -> back face name
        self.token = {}              # card id -> is a token
        faces = {}                   # (card id, face) -> _Face
        self._by_name = {}           # name -> _Face (the first face row with the name)
        self._ability_cost = {}      # (card id, ability index) -> (category name, cost text)
        for r in t.get("card", []):
            name = fold_name(r["name"])
            face = _Face(r)
            faces[(r["card"], r["face"])] = face
            (self.back if r["face"] else self.front).setdefault(r["card"], name)
            self.token[r["card"]] = bool(r["token"])
            self._by_name.setdefault(name, face)
        for r in t.get("card_keyword", []):
            f = faces.get((r["card"], r["face"]))
            if f is not None and r["kw"] != _NONE8 and r["kw"] < len(kw_names):
                f.keywords.add(kw_names[r["kw"]])
        for r in t.get("card_subtype", []):
            f = faces.get((r["card"], r["face"]))
            if f is not None and r["subtype"] < len(sub_names):
                f.subtypes.add(sub_names[r["subtype"]])
        for r in t.get("card_ability", []):
            f = faces.get((r["card"], r["face"]))
            if f is not None:
                cat = effect_kinds[r["category"]] if r["category"] < len(effect_kinds) else None
                kind = ab_kinds[r["ability_kind"]]
                f.abilities.append((kind, cat))
                if kind == "SPELL" and not f.spell_cost:
                    f.spell_cost = r["cost"]
            self._ability_cost[(r["card"], r["index"])] = (cat, r["cost"])
        colors = en["color"]
        pip_bits = [(k, _bit(colors, c)) for k, c in enumerate(_PIP_COLORS)]
        phyrexian = en["pip_kind"].index("PHYREXIAN")
        for r in t.get("card_pip", []):
            f = faces.get((r["card"], r["face"]))
            if f is not None:
                pip = {k for k, b in pip_bits if r["colors"] & b}
                f.pip_colors |= pip
                if r["kind"] == phyrexian:
                    f.phyrexian.append(pip)

    def name(self, cid, back=False):
        """The name of card ``cid``'s front (or back) face; None for no card."""
        if cid < 0:
            return None
        if back:
            name = self.back.get(cid)
            if name is not None:
                return name
        return self.front.get(cid, f"#{cid}")

    def _face(self, name):
        return self._by_name.get(name)

    # The core's interface (scripted_core module docstring).
    def mana_value(self, name) -> float:
        f = self._face(name)
        return float(f.row["mana_value"]) if f is not None else 0.0

    def cost_colors(self, name):
        f = self._face(name)
        if f is None:
            return None
        r = f.row
        return ({col for col, key in enumerate(("cost_w", "cost_u", "cost_b", "cost_r", "cost_g")) if r[key] > 0}
                | f.pip_colors)

    def is_land(self, name) -> bool:
        f = self._face(name)
        return f is not None and bool(f.row["types"] & self._land_bit)

    def power(self, name) -> int:
        f = self._face(name)
        return f.row["power"] if f is not None else 0

    def is_delve(self, name) -> bool:
        f = self._face(name)
        return f is not None and "Delve" in f.keywords

    def is_equipment(self, name) -> bool:
        f = self._face(name)
        return f is not None and ("Equipment" in f.subtypes or "Equip" in f.keywords)

    def has_miracle(self, name) -> bool:
        f = self._face(name)
        return f is not None and "Miracle" in f.keywords

    def is_legendary(self, name) -> bool:
        f = self._face(name)
        return f is not None and bool(f.row["supertypes"] & self._legendary_bit)

    def land_colors(self, name) -> set:
        f = self._face(name)
        return {_BASIC_LAND_COLORS[s] for s in f.subtypes if s in _BASIC_LAND_COLORS} if f is not None else set()

    def deals_damage(self, name) -> bool:
        """A DealDamage ability, or oracle text dealing damage to a chosen target."""
        f = self._face(name)
        return f is not None and (any(cat == "DealDamage" for _kind, cat in f.abilities)
                                  or bool(_TARGETED_DAMAGE.search(f.row["oracle"])))

    def has_nonmana_activation(self, name) -> bool:
        f = self._face(name)
        return f is not None and any(kind == "ACTIVATED" and cat != "AddMana" for kind, cat in f.abilities)

    def is_artifact_or_enchantment(self, name) -> bool:
        f = self._face(name)
        return f is not None and bool(f.row["types"] & self._art_ench_bits)

    def is_creature(self, name) -> bool:
        f = self._face(name)
        return f is not None and bool(f.row["types"] & self._creature_bit)

    def phyrexian_pips(self, name) -> list:
        """Each Phyrexian pip of the card's cost, as its set of colours (0-4)."""
        f = self._face(name)
        return f.phyrexian if f is not None else []

    def life_effects(self, name) -> LifeEffects:
        """The card's ``LifeEffects`` (all zero for an unknown card)."""
        f = self._face(name)
        if f is None:
            return _NO_LIFE_EFFECTS
        if f.life is None:
            f.life = LifeEffects(f.row["oracle"], f.spell_cost, bool(f.row["types"] & self._one_shot_bits))
        return f.life

    def ability(self, cid, index):
        """(category name, cost text) of card ``cid``'s ability ``index``; None for none."""
        return self._ability_cost.get((cid, index))


# ── One seat's frames ───────────────────────────────────────────────────────────────────────

class SeatFrames:
    """Follows one seat's RMSTATE frames (``feed``, in stream order) and builds each decision
    frame's view (``view``)."""

    def __init__(self, seat: str):
        self.seat = seat                    # "A" / "B"
        self.schema = None
        self.en = {}
        self.cards_frame = None
        self.cards = None                   # FrameCards
        self.deck_names = {}                # seat value (1 A, 2 B) -> registered deck name
        self.lists = {}                     # player-rel -> {card name: copies} (main + side)
        self.meta_lists = []                # per reference list: {card name: copies}
        self._metagame = None

    def feed(self, frame, schema):
        if frame.kind == "SCHEMA":
            self.schema = schema
            self.en = schema.enums
        elif frame.kind == "CARDS":
            self.cards_frame = frame
            self.cards = FrameCards(schema, frame)
            if self._metagame is not None:
                self._read_metagame(self._metagame)
        elif frame.kind == "DECKLISTS":
            t = frame.tables
            self.deck_names = {r["player"]: r["name"] for r in t.get("decklist", [])}
            self.lists = {}
            for r in t.get("decklist_card", []):
                name = self.cards.name(_card(r["card"]))
                d = self.lists.setdefault(r["player_rel"], {})
                d[name] = d.get(name, 0) + r["main"] + r["side"]
        elif frame.kind == "METAGAME":
            self._metagame = frame
            if self.cards is not None:
                self._read_metagame(frame)

    def _read_metagame(self, frame):
        t = frame.tables
        self.meta_lists = [{} for _ in t.get("meta_list", [])]
        for r in t.get("meta_list_card", []):
            if r["list"] < len(self.meta_lists):
                d = self.meta_lists[r["list"]]
                name = self.cards.name(_card(r["card"]))
                d[name] = d.get(name, 0) + r["main"] + r["side"]

    def deck_name_pair(self):
        """(deck A, deck B) names from DECKLISTS (None where withheld)."""
        return self.deck_names.get(1), self.deck_names.get(2)

    def view(self, frame) -> "FrameView":
        return FrameView(self, frame)


# ── The view ────────────────────────────────────────────────────────────────────────────────

class _Perm:
    __slots__ = ("card", "token", "tapped", "sick", "creature", "land", "power", "toughness",
                 "attacking", "blocking", "blocked", "loyalty", "other_counters", "attached",
                 "has_attachment", "keywords", "phased", "mana_value", "art_ench", "legendary",
                 "basic", "handle", "mine")


class _StackItem:
    __slots__ = ("card", "mine", "spell")

    def __init__(self, card, mine, spell):
        self.card = card
        self.mine = mine
        self.spell = spell


# ActionRefZone names of an object's zone (viewer-relative).
_ZONE_REF = {"BATTLEFIELD": ("SELF_BATTLEFIELD", "OPP_BATTLEFIELD"), "HAND": ("SELF_HAND", "OPP_HAND"),
             "STACK": ("STACK", "STACK"), "GRAVEYARD": ("SELF_GY", "OPP_GY"),
             "EXILE": ("SELF_EXILE", "OPP_EXILE")}


class FrameView:
    """One seat's decision frame as the decision core's view (``scripted_core`` docstring).
    Lazy: each part is built on first use."""

    exact_refs = True

    def __init__(self, seat: SeatFrames, frame):
        self._s = seat
        self.t = frame.tables
        self.cards = seat.cards
        en = seat.en
        self._zones = en["zone"]
        self.menu_rows = self.t.get("menu", [])
        self.n = len(self.menu_rows)
        cat_names = en["action_category"]
        self.cats = [cat_names[r["category"]] for r in self.menu_rows]
        self.seat_is_a = seat.seat == "A"
        header = self.t["header"][0]
        self._sideboarding = bool(header["sideboarding"])
        turn = self.t.get("turn")
        self._turn = turn[0] if turn and not self._sideboarding else None
        self._players = {p["player_rel"]: p for p in self.t.get("player", [])}

    # ── objects ─────────────────────────────────────────────────────────────────────────────
    @cached_property
    def _objects(self) -> dict:
        return {o["handle"]: o for o in self.t.get("object", [])}

    def _zone(self, o) -> str:
        return self._zones[o["zone"]]

    def _name(self, o):
        """An object's card name as it is now: a copy's copied card, a transformed card's back
        face (a spell cast as its back face too), else its card; None when face down."""
        cid = _card(o["card"])
        if cid < 0:
            return None
        if o["token"]:
            return self.cards.name(cid)
        if o["copy_of"] != _NONE16:
            cid = o["copy_of"]
        return self.cards.name(cid, back=bool(o["transformed"]))

    @cached_property
    def _perms(self) -> dict:
        """handle -> _Perm for every battlefield permanent (phased-out ones included; a card whose
        permanent is not built yet, its entry waiting on a choice, is none)."""
        en = self._s.en
        types, supers = en["card_type"], en["supertype"]
        creature_bit, land_bit = _bit(types, "Creature"), _bit(types, "Land")
        art_ench_bits = _bit(types, "Artifact") | _bit(types, "Enchantment")
        legendary_bit, basic_bit = _bit(supers, "Legendary"), _bit(supers, "Basic")
        kw_names = en["keyword"]
        bf = self._zones.index("BATTLEFIELD")
        perms = {}
        for o in self.t.get("object", []):
            if o["zone"] != bf or o["timestamp"] == _NONE32:
                continue
            p = _Perm()
            p.handle = o["handle"]
            p.mine = o["controller_rel"] == _SELF
            p.card = self._name(o)
            p.token = bool(o["token"])
            p.tapped = bool(o["tapped"])
            p.sick = bool(o["sick"])
            p.creature = bool(o["types"] & creature_bit)
            p.land = bool(o["types"] & land_bit)
            p.power = o["power"] if p.creature else 0
            p.toughness = o["toughness"] if p.creature else 0
            p.attacking, p.blocking, p.blocked = bool(o["attacking"]), bool(o["blocking"]), bool(o["blocked"])
            p.loyalty = p.other_counters = 0
            p.attached = o["attached_to"] != 0
            p.has_attachment = False
            p.keywords = set()
            p.phased = bool(o["phased_out"])
            p.mana_value = o["mana_value"]
            p.art_ench = bool(o["types"] & art_ench_bits)
            p.legendary = bool(o["supertypes"] & legendary_bit)
            p.basic = bool(o["supertypes"] & basic_bit)
            perms[p.handle] = p
        for o in self.t.get("object", []):
            host = perms.get(o["attached_to"]) if o["attached_to"] else None
            if host is not None and o["handle"] in perms:
                host.has_attachment = True
        for r in self.t.get("object_counter", []):
            p = perms.get(r["obj"])
            if p is None:
                continue
            if r["kind"] == "LOYALTY":
                p.loyalty = r["count"]
            elif r["kind"] not in ("P1P1", "M1M1"):
                p.other_counters += r["count"]
        for r in self.t.get("object_keyword", []):
            p = perms.get(r["obj"])
            if p is not None:
                p.keywords.add(kw_names[r["kw"]] if r["kw"] < len(kw_names) else r["text"])
        return perms

    @cached_property
    def my_perms_all(self) -> list:
        return [p for p in self._perms.values() if p.mine]

    @cached_property
    def my_perms(self) -> list:
        return [p for p in self.my_perms_all if not p.phased]

    @cached_property
    def opp_perms(self) -> list:
        return [p for p in self._perms.values() if not p.mine and not p.phased]

    def _zone_cards(self, zone, owner_rel):
        z = self._zones.index(zone)
        return [o for o in self.t.get("object", []) if o["zone"] == z and o["owner_rel"] == owner_rel]

    @cached_property
    def hand(self) -> list:
        return [n for n in (self._name(o) for o in self._zone_cards("HAND", _SELF)) if n is not None]

    @cached_property
    def graveyard_cards(self) -> set:
        gy = self._zones.index("GRAVEYARD")
        return {self._name(o) for o in self.t.get("object", []) if o["zone"] == gy and _card(o["card"]) >= 0}

    @cached_property
    def opp_graveyard(self) -> list:
        return [n for n in (self._name(o) for o in self._zone_cards("GRAVEYARD", _OPP)) if n is not None]

    @cached_property
    def opp_exile(self) -> list:
        return [n for n, _c in self.opp_exile_counters]

    @cached_property
    def opp_exile_counters(self) -> list:
        counters = {}
        for r in self.t.get("object_counter", []):
            if r["count"] > 0:
                counters[r["obj"]] = counters.get(r["obj"], 0) + r["count"]
        return [(self._name(o), counters.get(o["handle"], 0)) for o in self._zone_cards("EXILE", _OPP)
                if _card(o["card"]) >= 0]

    # ── the stack ───────────────────────────────────────────────────────────────────────────
    def _stack_card(self, r):
        """A stack object's card name: a spell's card (its back face when cast as it), an
        ability's source as it is now; none for a trigger with no source of its own (a floating
        or delayed trigger an effect created)."""
        en = self._s.en
        if (en["stack_kind"][r["kind"]] == "TRIGGERED" and not r["source"]
                and r["ab_card"] == _NONE16 and not r["ab_key"]):
            return None
        o = self._objects.get(r["handle"])
        if o is not None and _card(o["card"]) >= 0:
            return self._name(o)
        src = self._objects.get(r["source"]) if r["source"] else None
        if src is not None and self._zone(src) == "BATTLEFIELD":
            return self._name(src)
        return self.cards.name(_card(r["source_card"]))

    @cached_property
    def _stack_rows(self) -> list:
        return sorted(self.t.get("stack", []), key=lambda r: r["pos"])

    @cached_property
    def _stack_items(self) -> list:
        kinds = self._s.en["stack_kind"]
        return [_StackItem(self._stack_card(r), r["controller_rel"] == _SELF,
                           kinds[r["kind"]] in ("SPELL", "COPY")) for r in self._stack_rows]

    @cached_property
    def stack(self) -> list:
        return [s for s in self._stack_items if s.card is not None]

    @property
    def stack_top(self):
        return self._stack_items[0] if self._stack_items else None

    @property
    def stack_empty(self) -> bool:
        return not self._stack_rows

    # ── players and turn ────────────────────────────────────────────────────────────────────
    def _player(self, rel):
        return self._players.get(rel) or {}

    @property
    def life(self) -> int:
        return self._player(_SELF).get("life", 0)

    @property
    def opp_life(self) -> int:
        return self._player(_OPP).get("life", 0)

    @property
    def library(self) -> int:
        return self._player(_SELF).get("library", 0)

    @property
    def mulligans(self) -> int:
        return 0 if self._sideboarding else self._player(_SELF).get("mulligans", 0)

    def pool(self, color) -> int:
        return self._player(_SELF).get("mana_" + "wubrgc"[color], 0)

    @property
    def pool_total(self) -> int:
        return sum(self.pool(k) for k in range(6))

    @property
    def turn(self) -> int:
        return self._turn["turn"] if self._turn else 0

    @property
    def step(self):
        return self._s.en["step"][self._turn["step"]] if self._turn else None

    @property
    def in_main(self) -> bool:
        return self.step in ("FIRST_MAIN", "SECOND_MAIN")

    @property
    def is_active(self) -> bool:
        return bool(self._turn) and self._turn["active_rel"] == _SELF

    @cached_property
    def _pending_row(self):
        rows = self.t.get("pending")
        return rows[0] if rows else None

    @cached_property
    def pending(self):
        """The pending decision's source card: a battlefield permanent as it is now, else its
        card."""
        p = self._pending_row
        if p is None:
            return None
        o = self._objects.get(p["source"]) if p["source"] else None
        if o is not None and self._zone(o) == "BATTLEFIELD":
            return self._name(o)
        return self.cards.name(_card(p["source_card"]))

    @property
    def pending_is_cast(self) -> bool:
        """The pending decision is a cast-time prompt (a target, X, a payment of the spell being
        cast, CR 601.2)."""
        p = self._pending_row
        return p is not None and self._s.en["pending_tag"][p["tag"]] == "CAST"

    @property
    def pending_perm(self):
        p = self._pending_row
        return self._perms.get(p["source"]) if p is not None and p["source"] else None

    # ── menu ────────────────────────────────────────────────────────────────────────────────
    def _source(self, i):
        r = self.menu_rows[i]
        return self._objects.get(r["source"]) if r["source"] else None

    def card(self, i):
        """The action's card: its object's name as it is now (a stack object's card), a back face
        played or cast as it, a named card; none for a player or a card-less choice."""
        r = self.menu_rows[i]
        o = self._source(i)
        if o is not None:
            zone = self._zone(o)
            if zone == "STACK":
                for row in self._stack_rows:
                    if row["handle"] == o["handle"]:
                        return self._stack_card(row)
            if zone == "BATTLEFIELD" or _card(o["card"]) < 0:
                return self._name(o)
        elif r["source_player_rel"]:
            return None
        cid = _card(r["source_card"])
        back = bool(r["variant"] & _bit(self._s.en["cast_variant"], "BACK_FACE"))
        return self.cards.name(cid, back=back)

    def nonnull_card(self, i) -> bool:
        return self.card(i) is not None

    def same_card(self, i, j) -> bool:
        return self.card(i) == self.card(j)

    def zone(self, i) -> str:
        r = self.menu_rows[i]
        o = self._source(i)
        if o is not None:
            pair = _ZONE_REF.get(self._zone(o))
            if pair is None:
                return "NONE"
            rel = o["controller_rel"] if self._zone(o) == "BATTLEFIELD" else o["owner_rel"]
            return pair[0] if rel == _SELF else pair[1]
        if r["source_player_rel"]:
            return "PLAYER_SELF" if r["source_player_rel"] == _SELF else "PLAYER_OPP"
        return "NONE"

    def ctrl(self, i) -> int:
        """1 for a choice of our own object (or ourselves), 0 for the opponent's, -1 for none
        (no object, or an object in no zone the menu refers to, such as a library card)."""
        r = self.menu_rows[i]
        o = self._source(i)
        if o is not None:
            if self.zone(i) == "NONE":
                return -1
            return 1 if o["controller_rel"] == _SELF else 0
        if r["source_player_rel"]:
            return 1 if r["source_player_rel"] == _SELF else 0
        return -1

    def ordinal(self, i) -> int:
        return self.menu_rows[i]["ordinal"]

    def ref(self, i):
        return self.menu_rows[i]["source"]

    def ref_perm(self, i):
        h = self.menu_rows[i]["source"]
        return self._perms.get(h) if h else None

    def is_keyword_ability(self, i, keyword) -> bool:
        return self.menu_rows[i]["ab_key"] == keyword

    def cast_variants(self, i) -> frozenset:
        """The cast variants (``cast_variant`` names: ``ALT_COST``, ``FLASHBACK``, ``EVOKE``, ...)
        the action casts with."""
        mask = self.menu_rows[i]["variant"]
        return frozenset(n for k, n in enumerate(self._s.en["cast_variant"]) if mask >> k & 1)

    def ability_category(self, i):
        """The effect category of the ability the action activates; None for none."""
        r = self.menu_rows[i]
        ab = self.cards.ability(r["ab_card"], r["ab_index"]) if r["ab_card"] != _NONE16 else None
        return ab[0] if ab else None

    @cached_property
    def _pending_cost(self):
        rows = self.t.get("pending_cost")
        return rows[0] if rows else None

    def life_cost(self, i) -> int:
        """The life the action pays as a cost (CR 119.4): its ability's ``PayLife`` cost, a spell's
        additional, alternative (``ALT_COST``) or flashback life cost, a Phyrexian pip's 2 life (the
        mana-less ordinal-0 payment row while the cost has a Phyrexian pip), an "unless" life
        payment's Pay, an X life payment's X (the CHOOSE_X row's ordinal), and the accept of an
        "as it enters, you may pay N life" choice. 0 when none (an X still to choose counts 0)."""
        r = self.menu_rows[i]
        cat = self.cats[i]
        if r["ab_card"] != _NONE16:
            ab = self.cards.ability(r["ab_card"], r["ab_index"])
            return _life_amount(ab[1])[0] if ab else 0
        if cat == "CAST_SPELL":
            fx = self.cards.life_effects(self.card(i))
            variants = self.cast_variants(i)
            return (fx.spell_life + (fx.alt_life if "ALT_COST" in variants else 0)
                    + (fx.flashback_life if "FLASHBACK" in variants else 0))
        pc = self._pending_cost
        if cat == "PAYING_COSTS":
            phyrexian = self._s.en["pip_kind"].index("PHYREXIAN")
            if (pc is not None and r["ordinal"] == 0 and not r["source"] and r["source_card"] == _NONE16
                    and any(p["kind"] == phyrexian for p in self.t.get("pending_cost_pip", []))):
                return 2
            return 0
        if cat == "PAY_UNLESS":
            unless = self._s.en["cost_part"].index("UNLESS")
            return pc["rem_life"] if pc is not None and pc["part"] == unless and r["ordinal"] == 1 else 0
        if cat == "CHOOSE_X":
            return max(0, r["ordinal"]) if self.cards.life_effects(self.pending).spell_life_x else 0
        if cat == "OPTIONAL_YESNO" and self.n == 2 and i == 1:
            return self.cards.life_effects(self.pending).enters_life
        return 0

    # ── memory: the knowledge sidecar's tables ──────────────────────────────────────────────
    @cached_property
    def _pool_cards(self) -> dict:
        """k_pool row -> {card name: (min, max)}."""
        out = {}
        for r in self.t.get("k_pool_card", []):
            out.setdefault(r["pool"], {})[self.cards.name(_card(r["card"]))] = (r["min"], r["max"])
        return out

    @cached_property
    def _own_top(self) -> dict:
        """position from the top -> (exact name or None, candidate pool row or None)."""
        top = self._s.en["lib_anchor"].index("TOP")
        out = {}
        for r in self.t.get("k_position", []):
            if r["owner_rel"] != _SELF or r["at_anchor"] != top:
                continue
            exact = self.cards.name(_card(r["card"])) if r["card"] != _NONE16 else None
            out[r["at_pos"]] = (exact, r["pool"] if r["pool"] != _NONE16 else None)
        return out

    def known_top(self, n) -> list:
        return [self._own_top.get(k, (None, None))[0] for k in range(n)]

    def top_candidates(self, n) -> list:
        out = []
        for k in range(n):
            exact, pool = self._own_top.get(k, (None, None))
            if exact is not None:
                out.append({exact})
            elif pool is not None:
                out.append({c for c, (_lo, hi) in self._pool_cards.get(pool, {}).items() if hi > 0} or None)
            else:
                out.append(None)
        return out

    @cached_property
    def opp_known_hand(self) -> list:
        out = []
        for r in self.t.get("k_known_hand", []):
            out += [self.cards.name(_card(r["card"]))] * r["copies"]
        return out

    @cached_property
    def _opp_hand_pool(self):
        kinds = self._s.en["pool_kind"]
        for i, r in enumerate(self.t.get("k_pool", [])):
            if r["owner_rel"] == _OPP and kinds[r["kind"]] == "HAND":
                return self._pool_cards.get(i, {})
        return None

    def opp_hand_possible(self, name):
        """Whether the card can be in the opponent's hand (its hand pool's most copies > 0);
        None without a hand pool (level off, or between games)."""
        pool = self._opp_hand_pool
        if pool is None:
            return None
        lo_hi = pool.get(name)
        return lo_hi is not None and lo_hi[1] > 0

    def own_library_has(self, name):
        """Whether the own library can hold the card (an exact position, or a library pool with
        a copy possible); None without library knowledge."""
        kinds = self._s.en["pool_kind"]
        pools = [i for i, r in enumerate(self.t.get("k_pool", []))
                 if r["owner_rel"] == _SELF and kinds[r["kind"]] == "LIBRARY_RANGE"]
        exact = [e for e, _p in self._own_top.values() if e is not None]
        if not pools and not exact:
            return None
        return name in exact or any(self._pool_cards.get(i, {}).get(name, (0, 0))[1] > 0 for i in pools)

    @cached_property
    def _opp_possible_lists(self):
        """The reference lists the opponent can be on (k_template consistent rows), when its
        registered list is withheld; None without them."""
        rows = [r for r in self.t.get("k_template", []) if r["player_rel"] == _OPP and r["consistent"]]
        if not rows or not self._s.meta_lists:
            return None
        return [self._s.meta_lists[r["list"]] for r in rows if r["list"] < len(self._s.meta_lists)]

    @cached_property
    def opp_deck(self) -> dict:
        """The opponent's registered list (name -> copies); under a closed decklist each card's
        most copies over the reference lists it can be on."""
        deck = self._s.lists.get(_OPP)
        if deck is not None:
            return deck
        lists = self._opp_possible_lists or []
        out = {}
        for lst in lists:
            for name, n in lst.items():
                out[name] = max(out.get(name, 0), n)
        return out

    def opp_runs_card(self, card, _deck_name=None):
        """Whether the opponent's list holds ``card``: its registered list when visible, else
        whether any reference list it can still be on does (None without either)."""
        deck = self._s.lists.get(_OPP)
        if deck is not None:
            return card in deck
        lists = self._opp_possible_lists
        if lists is None:
            return None
        return any(card in lst for lst in lists)


def decide(agent, seat: SeatFrames, frame) -> int:
    """The action ``agent`` (a ``scripted_core.ScriptedCore``) takes at decision ``frame`` of
    ``seat`` (deck names from the seat's DECKLISTS frame)."""
    agent.set_deck_names(*seat.deck_name_pair())
    return agent.decide(seat.view(frame))
