"""Pass-1 parity adapter: the old BQUERY observation rebuilt from one seat's RMSTATE frames.

Plan step 2, pass 1 (``docs/engine_fork_plan.md``): the scripted agent keeps its decision logic
unchanged and reads, in place of the engine's float state vector, a vector of the same layout
(``env.OBS_SIZE``) that this module writes from the seat's decoded frames
(``train/rmstate.py``) and the knowledge sidecar's ``k_*`` tables. Every position the agent
reads holds the value the old vector gave; the rest are left empty (zeros, with the empty
card-id sentinel where a card id goes). The emulation list is ``docs/state_format_agent_map.md``
§5, pinned down in its "Pass-1 adapter as built" section.

- :class:`FrameCardKnowledge` answers the agent's card-knowledge reads (``card_costs``,
  ``card_props``, script lines, the decklist of a deck name) from the match's ``CARDS`` and
  ``DECKLISTS`` frames: the generators' own parsing (``gen_card_costs`` / ``gen_card_props`` /
  ``gen_util``) applied to each card's raw script, which the engine puts in ``CARDS`` under
  ``--rmstate-card-scripts``, so a card reads exactly as the generated tables have it.
- :class:`SeatAdapter` follows one seat's frames (``feed``) and builds the observation of each
  decision frame (``observation``).

Sources (``docs/state_format.md`` Appendix A): ``player`` / ``player_counter`` (player blocks,
extras, per-turn counters), ``turn`` (step, active seat, priority window, mandatory choice),
``object`` and its child tables (battlefield, hand, graveyard and exile slots), ``combat``,
``stack`` / ``stack_target`` / ``stack_mode`` / ``lki`` (stack slots), ``play_permission``,
``pending``, ``match``, ``decklist_card`` / ``own_deck`` (decklist blocks), ``menu`` (the
per-action arrays and descriptions), ``k_position`` (known library tops) and ``k_known_hand``
(the opponent's known hand cards), and the events (entry turns, for the summoning-sickness
flag).
"""
from __future__ import annotations

import numpy as np

import env as E
import _enums as R
from _enums import _OBS_KEYWORDS, LIBRARY_NORMALIZER, SIDEBOARD_SWAP_CAP
from gen_card_costs import N_FEATS, parse_mana_cost
from gen_card_props import COL as _PROP_COL, N_PROPS, parse_face
from gen_util import card_face_from_text, card_row_lines, mana_value_lines, script_field, split_faces

_N = E.N_CARD_TYPES
_NONE8, _NONE16, _NONE32 = 0xFF, 0xFFFF, 0xFFFFFFFF
_SELF, _OPP = 1, 2                              # player-rel values
_SEAT_A, _SEAT_B = 1, 2                         # enum:seat values


def _f32(num, den=1.0):
    """num / den as the C++ serializer's float32 division gives it, once stored in a float32
    array: the double quotient of two small integers rounds to the same float32 (it is never within
    a double's precision of a float32 rounding midpoint)."""
    return num / den


def _norm_id(cid):
    """norm_card_id: idx / N_CARD_TYPES, -1 / N_CARD_TYPES for none."""
    return _f32(cid if cid >= 0 else -1, _N)


def _norm_ref(ref):
    """norm_ref: (ref + 1) / N_ENTITY_REF_SLOTS (0.0 = none)."""
    return _f32(ref + 1, E.N_ENTITY_REF_SLOTS)


def _card(v):
    """A `card` column value as a vocab id, -1 for none."""
    return -1 if v == _NONE16 else v


def _empty_state():
    """The state vector of nothing: zeros, with the empty card-id sentinel at every card-id
    position (machine_io.cpp writes an empty slot that way)."""
    s = np.zeros(E.STATE_SIZE, dtype=np.float32)
    null = _norm_id(-1)
    idx = []
    for side in (E._SELF_PERM_START, E._OPP_PERM_START):
        for k in range(E._PERM_SLOTS):
            base = side + k * E._PERM_SLOT_SIZE
            idx += [base + E._PERM_CHOSEN_NAME_OFF, base + E._PERM_RETURNABLE_OFF, base + E._PERM_CARD_OFF]
    for k in range(E._STACK_SLOTS):
        base = E._STACK_START + k * E._STACK_SLOT_SIZE
        idx.append(base + 1)
        for t in range(E._STACK_TGT_SLOTS):
            idx.append(base + E._STACK_TGT_START + t * E._STACK_TGT_FIELDS + E._STACK_TGT_FIELDS - 1)
    for k in range(2 * E.MAX_GY_SLOTS):
        idx.append(E._GY_START + k * E._GY_SLOT_SIZE + E.ZONE_CARD_ID_OFF)
        idx.append(E._EXILE_START + k * E._EXILE_SLOT_SIZE + E.ZONE_CARD_ID_OFF)
    idx += range(E._HAND_START, E._HAND_START + E.MAX_HAND_SLOTS)
    idx += range(E._KNOWN_TOP_LIB_START, E._OPP_KNOWN_HAND_END)
    idx.append(E._PENDING_DECISION_START)
    for start, n, width in ((E._SELF_LIVE_LIB_START, E.DECKLIST_MAIN_SLOTS, E._DECKLIST_SLOT_SIZE),
                            (E._SELF_DECK_MAIN_START, E.DECKLIST_MAIN_SLOTS, E._DECKLIST_SLOT_SIZE),
                            (E._SELF_DECK_SIDE_START, E.DECKLIST_SIDE_SLOTS, E._DECKLIST_SLOT_SIZE),
                            (E._OPP_DECK_MAIN_START, E.DECKLIST_MAIN_SLOTS, E._OPP_DECKLIST_SLOT_SIZE),
                            (E._OPP_DECK_SIDE_START, E.DECKLIST_SIDE_SLOTS, E._OPP_DECKLIST_SLOT_SIZE)):
        idx += range(start, start + n * width, width)
    for k in range(E._DELAYED_SLOTS):
        base = E._DELAYED_START + k * E._DELAYED_SLOT_SIZE
        idx += [base + E._DT_CREATOR_ID, base + E._DT_SUBJECT_ID]
    for base in (E._PLAYER_EFFECTS_START, E._PLAYER_EFFECTS_OPP_START):
        idx += range(base + E._PE_EMBLEM_OFF, base + E._PE_FLOATING_OFF + 1)
    s[np.array(idx)] = null
    s.flags.writeable = False
    return s


_EMPTY_STATE = _empty_state()
_N_STATUS = E._PERM_SLOT_SIZE - 3               # floats before the three id-family floats


# ── Card knowledge from the CARDS frame ─────────────────────────────────────────────────────

class FrameCardKnowledge:
    """The scripted agent's card knowledge (``scripted_agent.VocabCardKnowledge``'s interface)
    from one match's ``CARDS`` and ``DECKLISTS`` frames.

    A vocab id names a face: a ``card`` row's ``face_card`` (a double-faced card's back face has
    its own). The raw script of the card (the front row's ``script`` column) is resolved to that
    face with ``gen_util.card_face_from_text`` and read with the generators' functions, so every
    row equals the generated table's: the cast-cost row (``card_costs._CARD_COST_MATRIX``), the
    land flag (``_LAND_VOCAB_IDS``), the property row (``card_props._CARD_PROP_MATRIX``), the
    script lines and ``Types:`` of the face (``decode._card_script_lines`` / ``card_types``).
    Tokens read as the vocab tables have them: a zero cost row, not a land, the property row of
    their token script, and no card script (``decode._card_script_lines`` resolves a token's
    name to no card script, so ``card_types`` is ''). A card the frame does not list reads as an
    id outside the tables (no cost row, zero properties)."""

    def __init__(self, cards_frame, decklists_frame=None):
        self._faces = {}       # vocab id -> (script text, is_back, is_token, name)
        self.names = {}        # vocab id -> face name
        self.ids = {}          # face name -> vocab id
        self._back = {}        # card id -> its back face's vocab id (-1: none in the vocab)
        self.types = {}        # vocab id -> its face's printed card-type mask
        scripts = {}
        for r in cards_frame.tables.get("card", []):
            if r["face"] == 0:
                scripts[r["card"]] = r["script"]
            else:
                self._back[r["card"]] = _card(r["face_card"])
        # Front rows first: a split card's second half names the card's own id, which is its
        # front row's.
        for r in sorted(cards_frame.tables.get("card", []), key=lambda r: r["face"]):
            fid = _card(r["face_card"])
            if fid < 0 or fid in self._faces:
                continue
            self._faces[fid] = (scripts.get(r["card"], ""), r["face"] == 1, bool(r["token"]), r["name"])
            self.names[fid] = r["name"]
            self.types[fid] = r["types"]
            self.ids.setdefault(r["name"], fid)
        self._cost, self._land, self._props, self._lines = {}, {}, {}, {}
        # deck name -> set of card names on its registered list (main or sideboard)
        self.deck_cards = {}
        if decklists_frame is not None:
            name_of = {}
            for r in decklists_frame.tables.get("decklist", []):
                name_of[r["player"]] = r["name"]
                self.deck_cards.setdefault(r["name"], set())
            for r in decklists_frame.tables.get("decklist_card", []):
                deck = name_of.get(r["player"])
                cid = _card(r["card"])
                if deck is not None and cid in self.names and (r["main"] or r["side"]):
                    self.deck_cards[deck].add(self.names[cid])

    def back_face(self, cid):
        """The vocab id of a double-faced card's back face (-1 when it has none in the vocab;
        the card itself when it has no back face)."""
        return self._back.get(cid, cid)

    def has_scripts(self) -> bool:
        """Whether the CARDS frame carried the raw scripts (--rmstate-card-scripts)."""
        return any(text for text, _b, _t, _n in self._faces.values())

    def _face(self, cid):
        """The CardFace a vocab id names, or None (a token, an unlisted id, no script)."""
        entry = self._faces.get(cid)
        if entry is None or entry[2] or not entry[0]:
            return None
        return card_face_from_text(entry[0], entry[1], self.names[cid])

    def _derive(self, cid):
        cost = None
        land = False
        props = None
        lines = None
        entry = self._faces.get(cid)
        if entry is not None:
            text, _is_back, is_token, _name = entry
            props = [0.0] * N_PROPS
            if is_token:
                cost = [0] * N_FEATS
                if text:
                    parse_face(split_faces(text)[0], props)
            else:
                face = self._face(cid)
                if face is not None:
                    row_lines = card_row_lines(face)
                    cost_str = script_field(row_lines, "ManaCost")
                    cost = parse_mana_cost(cost_str) if cost_str is not None else [0] * N_FEATS
                    types_line = script_field(row_lines, "Types")
                    land = types_line is not None and "Land" in types_line.split()
                    parse_face(row_lines, props)
                    mv_lines = mana_value_lines(face)
                    if mv_lines is not face.lines:
                        pips = parse_mana_cost(script_field(mv_lines, "ManaCost") or "no cost")
                        props[_PROP_COL["cmc"]] = (sum(pips[:6]) + pips[6]) / 10.0
                    lines = face.lines
        self._cost[cid] = (np.array([float(f"{v / 10.0:.1f}") for v in cost], dtype=np.float32)
                           if cost is not None else None)
        self._land[cid] = land
        self._props[cid] = (np.array([float(f"{v:.1f}") for v in props], dtype=np.float32)
                            if props is not None else None)
        self._lines[cid] = lines

    def _get(self, table, cid):
        if cid not in table:
            self._derive(cid)
        return table[cid]

    # The VocabCardKnowledge interface (scripted_agent.py).
    def cost_row(self, cid):
        return self._get(self._cost, cid)

    def prop(self, cid, col):
        row = self._get(self._props, cid)
        return float(row[col]) if row is not None else 0.0

    def is_land(self, cid):
        return self._get(self._land, cid)

    def script_lines(self, cid):
        return self._get(self._lines, cid)

    def types_line(self, cid):
        lines = self.script_lines(cid)
        return (script_field(lines, "Types") or "") if lines is not None else ""

    def deck_runs_card(self, deck_name, card_name):
        cards = self.deck_cards.get(deck_name)
        return None if cards is None else card_name in cards


# ── One seat's frames → the old observation ─────────────────────────────────────────────────

# Every table the adapter reads, of every frame kind (rmstate_env decodes only these).
ADAPTER_TABLES = frozenset({
    "schema_table", "schema_column", "schema_enum",
    "card", "decklist", "decklist_card",
    "header", "turn", "player", "player_counter", "object", "object_keyword", "object_counter",
    "object_link", "combat", "stack", "stack_mode", "stack_target", "lki", "play_permission", "pending",
    "menu", "match", "own_deck", "event", "event_card", "k_position", "k_known_hand",
})


class SeatAdapter:
    """Follows one seat's RMSTATE frames and rebuilds, at each decision frame, the old
    observation vector and the per-action side-channels.

    ``feed(frame)`` takes every frame of the seat in stream order (SCHEMA, CARDS, DECKLISTS,
    METAGAME, STATE); ``observation(frame)`` builds the decision frame's vector. The adapter
    keeps only public history the frames carried: each permanent's entry turn (from the
    ``zone_move`` events) and which seat was active in each turn, which the old vector's raw
    summoning-sickness flag needs (see ``_sick``)."""

    def __init__(self, seat: str):
        self.seat = seat                    # "A" / "B"
        self.schema = None
        self._cards_frame = None            # the CARDS frame, until DECKLISTS completes it
        self.cards = None                   # FrameCardKnowledge
        self.decklists = None               # the DECKLISTS frame
        self.deck_names = {}                # seat value -> registered deck name
        self._entered = {}                  # battlefield handle -> turn it entered
        self._preset = set()                # battlefield handles placed by a harness preset
        self._active_by_turn = {}           # game turn -> active seat value
        self._game = None
        self._deck_patches = {}             # own-deck configuration -> decklist-block patch

    # ── frames ─────────────────────────────────────────────────────────────────────────────
    def feed(self, frame, schema):
        if frame.kind == "SCHEMA":
            self._bind_schema(schema)
        elif frame.kind == "CARDS":
            self._cards_frame = frame
        elif frame.kind == "DECKLISTS":
            self.decklists = frame
            self.deck_names = {r["player"]: r["name"] for r in frame.tables.get("decklist", [])}
            self.cards = FrameCardKnowledge(self._cards_frame, frame)
        elif frame.kind == "STATE":
            self._note_history(frame)

    def _bind_schema(self, schema):
        self.schema = schema
        en = schema.enums
        self._zone = {n: i for i, n in enumerate(en["zone"])}
        self._kind = {n: i for i, n in enumerate(en["event_kind"])}
        types = en["card_type"]
        self._creature_bit = 1 << types.index("Creature")
        self._land_bit = 1 << types.index("Land")
        cv = en["cast_variant"]
        self._variant_bit = {n: 1 << i for i, n in enumerate(cv)}
        self._stack_kind = {n: i for i, n in enumerate(en["stack_kind"])}
        self._target_player = en["target_kind"].index("PLAYER")
        self._returnable = en["link_kind"].index("RETURNABLE")
        self._top = en["lib_anchor"].index("TOP")
        self._planeswalker_target = en["attack_target_kind"].index("PLANESWALKER")
        self._dn = {n: i for i, n in enumerate(en["day_night"])}

    def _note_history(self, frame):
        t = frame.tables
        header = t["header"][0]
        if header["game"] != self._game:
            self._game = header["game"]
            self._entered.clear()
            self._preset.clear()
            self._active_by_turn.clear()
        turn = t.get("turn")
        if turn and not header["sideboarding"]:
            self._active_by_turn[turn[0]["turn"]] = turn[0]["active"]
        events = t.get("event", [])
        if not events:
            return
        bf, none = self._zone["BATTLEFIELD"], _NONE8
        move = self._kind["ZONE_MOVE"]
        for c in t.get("event_card", []):
            ev = events[c["event"]]
            if ev["kind"] != move or ev["to_zone"] != bf or c["handle_after"] == 0:
                continue
            self._entered[c["handle_after"]] = ev["turn"]
            if ev["zone"] == none and not c["token"]:
                self._preset.add(c["handle_after"])

    # ── the observation ────────────────────────────────────────────────────────────────────
    def observation(self, frame):
        """(obs, side) for a decision frame: the OBS_SIZE float32 vector and a dict of the
        BQUERY side-channels (``num_choices``, ``cats``, ``ords``, ``public``, ``descriptions``,
        ``pending_confirm``)."""
        t = frame.tables
        b = _Build(self, frame)
        state = b.state
        if t["header"][0]["sideboarding"]:
            np.copyto(state, E._SB_MASK_FILL, where=~E._SB_MASK_KEEP)
        obs = np.zeros(E.OBS_SIZE, dtype=np.float32)
        obs[:E.STATE_SIZE] = state
        side = b.menu(obs)
        self_deck = self.deck_names.get(_SEAT_A if self.seat == "A" else _SEAT_B)
        opp_deck = self.deck_names.get(_SEAT_B if self.seat == "A" else _SEAT_A)
        E.write_matchup_tail(obs, self_deck, opp_deck)
        return obs, side

    def seat_value(self):
        return _SEAT_A if self.seat == "A" else _SEAT_B

    def active_seat_at(self, turn, current_turn, current_active):
        """The seat active in game turn ``turn``: recorded from a frame of that turn, else by
        alternation from the current turn."""
        seen = self._active_by_turn.get(turn)
        if seen is not None:
            return seen
        if (current_turn - turn) % 2 == 0:
            return current_active
        return _SEAT_B if current_active == _SEAT_A else _SEAT_A


class _Build:
    """The state vector of one decision frame (a helper of SeatAdapter.observation)."""

    def __init__(self, ad: SeatAdapter, frame):
        self.ad = ad
        self.t = t = frame.tables
        self.cards = ad.cards
        self.state = _EMPTY_STATE.copy()
        self.objects = {o["handle"]: o for o in t.get("object", [])}
        self.counters = {}
        for r in t.get("object_counter", []):
            self.counters.setdefault(r["obj"], []).append((r["kind"], r["count"]))
        self.keywords = {}
        for r in t.get("object_keyword", []):
            self.keywords.setdefault(r["obj"], set()).add(r["text"])
        self.links = {}
        for r in t.get("object_link", []):
            if r["kind"] == ad._returnable:
                self.links[r["obj"]] = r["linked"]
        self.turn = t["turn"][0] if t.get("turn") else None
        self.players = {p["player_rel"]: p for p in t.get("player", [])}
        self.slot = {}                     # handle -> unified reference slot
        self._header()
        self._battlefield()
        self._stack()
        self._zones()
        self._hand()
        self._context()
        self._knowledge()
        self._pending()
        self._extras()
        self._decklists()
        self._per_turn()

    # ── card ids as the old serializer gives them ──────────────────────────────────────────
    def _face_id(self, o):
        """The id machine_io.cpp's get_card_vocab_idx gives an object: a token's band id; a
        card's by the name it has now (a copy's copied card, a transformed card's back face, -1
        when that face has no vocab entry)."""
        cid = _card(o["card"])
        if cid < 0 or o["token"]:
            return cid
        if o["copy_of"] != _NONE16:
            cid = o["copy_of"]
        if o["transformed"]:
            return self._back_face(cid)
        return cid

    def _back_face(self, cid):
        return self.cards.back_face(cid) if self.cards else cid

    def _name(self, cid):
        return self.cards.names.get(cid, "") if self.cards else ""

    # ── header: player blocks, step, flags ─────────────────────────────────────────────────
    def _player_block(self, base, p):
        s = self.state
        s[base + E._PB_LIFE] = _f32(p["life"], E.LIFE_NORMALIZER)
        s[base + E._PB_HAND_CT] = _f32(p["hand"], 10)
        ctr = {r["kind"]: r["count"] for r in self.t.get("player_counter", []) if r["player"] == p["player"]}
        s[base + E._PB_POISON] = _f32(ctr.get("POISON", 0), 10)
        for i, c in enumerate("wubrgc"):
            s[base + E._PB_MANA + i] = _f32(p["mana_" + c], 10)
        s[base + E._PB_ENERGY] = _f32(ctr.get("ENERGY", 0), 10)

    def _header(self):
        s = self.state
        me, opp = self.players.get(_SELF), self.players.get(_OPP)
        if me:
            self._player_block(E._SELF_BLOCK_START, me)
        if opp:
            self._player_block(E._OPP_BLOCK_START, opp)
        if self.turn:
            s[E._STEP_ONEHOT_START + self.turn["step"]] = 1.0
            s[E._IS_ACTIVE_IDX] = 1.0 if self.turn["active_rel"] == _SELF else 0.0
        s[E._SELF_IS_A_IDX] = 1.0 if self.ad.seat == "A" else 0.0
        s[E._STACK_SIZE_IDX] = _f32(len(self.t.get("stack", [])), 10)

    # ── battlefield slots ──────────────────────────────────────────────────────────────────
    def _order_key(self, o):
        """Slot order: the old serializer packs by entity id, which the format does not carry
        (it would tell copies apart by deck position). Entity ids run through the deck lists
        (Player A's, then B's), then the objects made later (tokens), and every league list is
        in name order, so this sorts by (token, owner, name) and then by handle."""
        return (o["token"], o["owner"], self._name(_card(o["card"])), o["handle"])

    def _battlefield(self):
        bf = self.ad._zone["BATTLEFIELD"]
        # A card entering the battlefield whose permanent is not built yet (its entry waits on a
        # choice: a shock land's life payment) has no timestamp; the serializer skips it.
        perms = [o for o in self.t.get("object", []) if o["zone"] == bf and o["timestamp"] != _NONE32]
        mine = sorted((o for o in perms if o["controller_rel"] == _SELF), key=self._order_key)
        theirs = sorted((o for o in perms if o["controller_rel"] != _SELF), key=self._order_key)
        self.bf = (mine[:E._PERM_SLOTS], theirs[:E._PERM_SLOTS])
        for side, objs in enumerate(self.bf):
            for k, o in enumerate(objs):
                self.slot[o["handle"]] = side * E._PERM_SLOTS + k
        stack = sorted(self.t.get("stack", []), key=lambda r: r["pos"])
        self.stack_by_handle = {r["handle"]: r for r in stack}
        self.stack_rows = stack[:E._STACK_SLOTS]
        for k, r in enumerate(self.stack_rows):
            self.slot[r["handle"]] = 2 * E._PERM_SLOTS + k
        attack_tgt, block_tgt = {}, {}
        for c in self.t.get("combat", []):
            if c["target_kind"] == self.ad._planeswalker_target and c["target"]:
                attack_tgt[c["attacker"]] = c["target"]
            if c["blocker"]:
                block_tgt.setdefault(c["blocker"], c["attacker"])
        attached_by = {}
        for side, objs in enumerate(self.bf):
            for k, o in enumerate(objs):
                host = self.slot.get(o["attached_to"], -1) if o["attached_to"] else -1
                if 0 <= host < 2 * E._PERM_SLOTS and host not in attached_by:
                    attached_by[host] = side * E._PERM_SLOTS + k
        s = self.state
        for side, objs in enumerate(self.bf):
            start = E._SELF_PERM_START if side == 0 else E._OPP_PERM_START
            for k, o in enumerate(objs):
                cid = self._face_id(o)
                if cid < 0:
                    continue                            # the serializer's empty slot
                base = start + k * E._PERM_SLOT_SIZE
                ref = side * E._PERM_SLOTS + k
                creature = bool(o["types"] & self.ad._creature_bit)
                vals = np.zeros(_N_STATUS, dtype=np.float32)
                if creature:
                    vals[E._OFF_POWER] = _f32(o["power"], 10)
                    vals[E._OFF_TOUGHNESS] = _f32(o["toughness"], 10)
                    vals[E._OFF_IS_ATTACKING] = o["attacking"]
                    vals[E._OFF_IS_BLOCKING] = o["blocking"]
                    vals[E._OFF_IS_BLOCKED] = o["blocked"]
                    vals[E._OFF_CANT_BE_BLOCKED] = o["cant_be_blocked"]
                vals[E._OFF_IS_TAPPED] = o["tapped"]
                vals[E._OFF_HAS_SICKNESS] = 1.0 if self._sick(o, creature) else 0.0
                vals[E._OFF_DAMAGE] = _f32(o["damage"], 10)
                vals[E._OFF_IS_CREATURE] = 1.0 if creature else 0.0
                vals[E._OFF_IS_LAND] = 1.0 if o["types"] & self.ad._land_bit else 0.0
                ctrs = dict(self.counters.get(o["handle"], ()))
                vals[E._OFF_LOYALTY] = _f32(ctrs.get("LOYALTY", 0), 10)
                vals[E._OFF_P1P1_NET] = _f32(ctrs.get("P1P1", 0) - ctrs.get("M1M1", 0), 10)
                vals[E._OFF_OTHER_COUNTERS] = _f32(sum(n for kd, n in ctrs.items()
                                                       if kd not in ("P1P1", "M1M1", "LOYALTY")), 10)
                vals[E._OFF_ATTACHED_TO] = _norm_ref(self.slot.get(o["attached_to"], -1)
                                                     if o["attached_to"] else -1)
                vals[E._OFF_ATTACHED_BY] = _norm_ref(attached_by.get(ref, -1))
                at = attack_tgt.get(o["handle"]) if creature else None
                vals[E._OFF_ATTACK_TGT] = _norm_ref(self.slot.get(at, -1) if at else -1)
                bt = block_tgt.get(o["handle"]) if creature else None
                vals[E._OFF_BLOCKING_TGT] = _norm_ref(self.slot.get(bt, -1) if bt else -1)
                vals[E._OFF_IS_PHASED_OUT] = o["phased_out"]
                vals[E._OFF_ENTERED_THIS_TURN] = o["entered_turn"]
                vals[E._OFF_RESOLUTIONS_THIS_TURN] = _f32(o["resolutions_turn"], E.PER_TURN_COUNT_NORMALIZER)
                vals[E._OFF_ACTIVATIONS_THIS_TURN] = _f32(o["activations_turn"], E.PER_TURN_COUNT_NORMALIZER)
                vals[E._OFF_COMBAT_DMG_PREVENTED] = o["damage_shielded"]
                vals[E._OFF_PENDING_DELAYED_SUBJECT] = o["delayed_subject"]
                kws = self.keywords.get(o["handle"], ())
                for kk, kw in enumerate(_OBS_KEYWORDS):
                    if kw in kws:
                        vals[E._OFF_KEYWORDS_START + kk] = 1.0
                s[base:base + _N_STATUS] = vals
                s[base + E._PERM_CHOSEN_NAME_OFF] = _norm_id(_card(o["chosen_name"]))
                ret = self.links.get(o["handle"])
                ret_id = _card(self.objects[ret]["card"]) if ret in self.objects else -1
                s[base + E._PERM_RETURNABLE_OFF] = _norm_id(ret_id)
                s[base + E._PERM_CARD_OFF] = _norm_id(cid)

    def _sick(self, o, creature):
        """The old vector's summoning-sickness flag: Permanent::has_summoning_sickness, the raw
        flag (set as a creature or any token enters, cleared at its controller's untap step),
        where the frame's ``sick`` is the effective rule (a creature, haste considered). They
        differ for a creature with haste, a token, and a creature card that is not a creature
        now (impending): those read the raw flag from the permanent's entry turn (its zone_move
        event) against the start of its controller's most recent turn. A card a harness preset
        put into play starts without it."""
        if creature and "Haste" not in self.keywords.get(o["handle"], ()):
            return bool(o["sick"])
        if not (creature or o["token"] or self.cards.types.get(self._face_id(o), 0) & self.ad._creature_bit):
            return False
        h = o["handle"]
        if h in self.ad._preset:
            return False
        entered = self.ad._entered.get(h)
        if entered is None or self.turn is None:
            return bool(o["entered_turn"])
        cur, active = self.turn["turn"], self.turn["active"]
        ctrl = o["controller"]
        last = cur
        while last > 0 and self.ad.active_seat_at(last, cur, active) != ctrl:
            last -= 1
        return entered >= last

    # ── stack slots ────────────────────────────────────────────────────────────────────────
    def _stack(self):
        s = self.state
        targets = {}
        for r in self.t.get("stack_target", []):
            targets.setdefault(r["stack"], []).append(r)
        modes = {}
        for r in self.t.get("stack_mode", []):
            modes.setdefault(r["stack"], []).append(r["mode"])
        lki = self.t.get("lki", [])
        vb = self.ad._variant_bit
        spell_kinds = (self.ad._stack_kind["SPELL"], self.ad._stack_kind["COPY"])
        for k, r in enumerate(self.stack_rows):
            base = E._STACK_START + k * E._STACK_SLOT_SIZE
            s[base] = 1.0 if r["controller_rel"] == _SELF else 0.0
            s[base + 1] = _norm_id(self._stack_card(r))
            is_spell = r["kind"] in spell_kinds
            s[base + 2] = 1.0 if is_spell else 0.0
            amount = (r["x"] if r["x"] >= 0 else 0) if is_spell else r["amount"]
            s[base + E._STACK_XAMT_OFF] = _f32(amount, 10)
            if is_spell:
                quals = (r["copy"], r["kicked"] > 0, r["variant"] & vb["FLASHBACK"],
                         r["variant"] & vb["EVOKE"], r["variant"] & vb["ESCAPE"],
                         r["variant"] & vb["OFFSPRING"], r["variant"] & vb["IMPENDING"])
                for q, on in enumerate(quals):
                    s[base + E._STACK_QUAL_START + q] = 1.0 if on else 0.0
            for m in modes.get(r["handle"], ()):
                if m < E._STACK_MODE_SLOTS:
                    s[base + E._STACK_MODE_START + m] = 1.0
            for n, tr in enumerate(sorted(targets.get(r["handle"], ()), key=lambda x: x["order"])[:E._STACK_TGT_SLOTS]):
                tb = base + E._STACK_TGT_START + n * E._STACK_TGT_FIELDS
                s[tb] = 1.0
                if tr["kind"] == self.ad._target_player:
                    s[tb + 1] = 1.0
                    s[tb + 2] = 1.0 if tr["target_player_rel"] == _SELF else 0.0
                    s[tb + 3] = _norm_ref(-1)
                    s[tb + 4] = _norm_id(-1)
                    continue
                o = self.objects.get(tr["target"]) if tr["target"] else None
                if o is not None:
                    s[tb + 2] = 1.0 if o["controller_rel"] == _SELF else 0.0
                    s[tb + 3] = _norm_ref(self.slot.get(tr["target"], -1))
                    s[tb + 4] = _norm_id(self._target_card(o))
                elif tr["lki"] != _NONE16 and tr["lki"] < len(lki):
                    lr = lki[tr["lki"]]
                    s[tb + 2] = 1.0 if lr["controller_rel"] == _SELF else 0.0
                    s[tb + 3] = _norm_ref(-1)
                    s[tb + 4] = _norm_id(_card(lr["card"]))
                else:
                    s[tb + 3] = _norm_ref(-1)
                    s[tb + 4] = _norm_id(-1)

    def _stack_card(self, r):
        """machine_io.cpp's action_card_vocab_idx of a stack object: a spell's card (its back
        face when cast as it), an ability's source as it is now (its face, a copy, a token). A
        trigger with no source of its own (a floating or delayed trigger an effect created: no
        source, no ability reference into a card's list) has none, where the frame shows its
        creator's card."""
        cid = _card(r["source_card"])
        if (r["kind"] == self.ad._stack_kind["TRIGGERED"] and not r["source"]
                and r["ab_card"] == _NONE16 and not r["ab_key"]):
            return -1
        o = self.objects.get(r["source"]) if r["source"] else None
        if o is not None and o["zone"] == self.ad._zone["BATTLEFIELD"]:
            return self._face_id(o)
        if r["variant"] & self.ad._variant_bit["BACK_FACE"] and cid >= 0:
            return self._back_face(cid)
        return cid

    def _target_card(self, o):
        if o["zone"] == self.ad._zone["BATTLEFIELD"]:
            return self._face_id(o)
        if o["zone"] == self.ad._zone["STACK"] and o["handle"] in self.stack_by_handle:
            return self._stack_card(self.stack_by_handle[o["handle"]])
        return _card(o["card"])

    # ── graveyard / exile / hand ───────────────────────────────────────────────────────────
    def _zones(self):
        s = self.state
        perm = {}
        for r in self.t.get("play_permission", []):
            d = perm.setdefault(r["obj"], {})
            d[r["player_rel"]] = d.get(r["player_rel"], True) and bool(r["expires_turn"])
        gy, ex = self.ad._zone["GRAVEYARD"], self.ad._zone["EXILE"]
        for zone, start, width, counters in ((gy, E._GY_START, E._GY_SLOT_SIZE, False),
                                             (ex, E._EXILE_START, E._EXILE_SLOT_SIZE, True)):
            for side, rel in enumerate((_SELF, _OPP)):
                objs = sorted((o for o in self.t.get("object", []) if o["zone"] == zone and o["owner_rel"] == rel),
                              key=lambda o: (o["pos"], o["handle"]))
                for k, o in enumerate(objs[:E.MAX_GY_SLOTS]):
                    base = start + (side * E.MAX_GY_SLOTS + k) * width
                    cid = _card(o["card"])
                    s[base + E.ZONE_CARD_ID_OFF] = _norm_id(cid)
                    if cid < 0:
                        continue                         # a face-down card the viewer cannot see
                    pp = perm.get(o["handle"], {})
                    by_self, by_opp = _SELF in pp, _OPP in pp
                    s[base + E.ZONE_PLAYABLE_SELF_OFF] = 1.0 if by_self else 0.0
                    s[base + E.ZONE_PLAYABLE_OPP_OFF] = 1.0 if by_opp else 0.0
                    expires = (by_self or by_opp) and (not by_self or pp[_SELF]) and (not by_opp or pp[_OPP])
                    s[base + E.ZONE_EXPIRES_OFF] = 1.0 if expires else 0.0
                    if counters:
                        n = sum(c for _k, c in self.counters.get(o["handle"], ()) if c > 0)
                        s[base + E.EXILE_COUNTERS_OFF] = _f32(n, E.ZONE_COUNTER_NORMALIZER)

    def _hand(self):
        hand = self.ad._zone["HAND"]
        objs = sorted((o for o in self.t.get("object", []) if o["zone"] == hand and o["owner_rel"] == _SELF),
                      key=self._order_key)
        for k, o in enumerate(objs[:E.MAX_HAND_SLOTS]):
            self.state[E._HAND_START + k] = _norm_id(_card(o["card"]))

    # ── match, library, turn context ───────────────────────────────────────────────────────
    def _context(self):
        s = self.state
        header = self.t["header"][0]
        match = self.t["match"][0] if self.t.get("match") else None
        me, opp = self.players.get(_SELF), self.players.get(_OPP)
        s[E._MATCH_CTX_START] = _f32(header["game"], E.MATCH_GAME_NORMALIZER)
        if me and opp:
            s[E._MATCH_CTX_START + 1] = _f32(me["match_wins"], 2)
            s[E._MATCH_CTX_START + 2] = _f32(opp["match_wins"], 2)
            s[E._LIBRARY_CTX_START] = _f32(me["library"], LIBRARY_NORMALIZER)
            s[E._LIBRARY_CTX_START + 1] = _f32(opp["library"], LIBRARY_NORMALIZER)
        s[E._IS_SIDEBOARD_IDX] = 1.0 if match and match["sideboard_phase"] else 0.0
        if self.turn:
            s[E._CUR_TURN_IDX] = _f32(self.turn["turn"], 50)

    # ── memory: the knowledge sidecar's tables ─────────────────────────────────────────────
    def _knowledge(self):
        """Known library tops from ``k_position`` (the viewer's library and the opponent's,
        positions 0-4 from the top whose card is exact) and the opponent's known hand cards from
        ``k_known_hand`` (each card's copies; ten slots, in the slot order of the hand)."""
        s = self.state
        top = self.ad._top
        for r in self.t.get("k_position", []):
            if r["at_anchor"] != top or r["at_pos"] >= E._KNOWN_TOP_LIB_SLOTS or r["card"] == _NONE16:
                continue
            start = E._KNOWN_TOP_LIB_START if r["owner_rel"] == _SELF else E._OPP_KNOWN_TOP_LIB_START
            s[start + r["at_pos"]] = _norm_id(r["card"])
        known = []
        for r in self.t.get("k_known_hand", []):
            known += [_card(r["card"])] * r["copies"]
        known.sort(key=lambda c: (self._name(c), c))
        for k, cid in enumerate(known[:E._OPP_KNOWN_HAND_SLOTS]):
            s[E._OPP_KNOWN_HAND_START + k] = _norm_id(cid)

    def _pending(self):
        rows = self.t.get("pending")
        if not rows:
            return
        p = rows[0]
        cid = _card(p["source_card"])
        o = self.objects.get(p["source"]) if p["source"] else None
        if o is not None and o["zone"] == self.ad._zone["BATTLEFIELD"]:
            cid = self._face_id(o)
        self.state[E._PENDING_DECISION_START] = _norm_id(cid)
        if p["source_controller_rel"]:
            ctrl = p["source_controller_rel"] == _SELF
        else:
            match = self.t["match"][0] if self.t.get("match") else None
            ctrl = bool(match and match["sideboard_phase"])
        self.state[E._PENDING_DECISION_START + 1] = 1.0 if ctrl else 0.0

    def _extras(self):
        s = self.state
        me, opp = self.players.get(_SELF), self.players.get(_OPP)
        if not (me and opp):
            return
        header = self.t["header"][0]
        s[E._EXTRAS_LANDS_SELF] = _f32(me["lands_played"], 10)
        s[E._EXTRAS_LANDS_OPP] = _f32(opp["lands_played"], 10)
        s[E._EXTRAS_MONARCH_SELF] = me["monarch"]
        s[E._EXTRAS_MONARCH_OPP] = opp["monarch"]
        s[E._EXTRAS_BLESSING_SELF] = me["city_blessing"]
        s[E._EXTRAS_BLESSING_OPP] = opp["city_blessing"]
        s[E._EXTRAS_REVOLT_SELF] = me["revolt"]
        s[E._EXTRAS_REVOLT_OPP] = opp["revolt"]
        s[E._EXTRAS_EXTRA_TURNS_SELF] = _f32(me["extra_turns"], 3)
        s[E._EXTRAS_EXTRA_TURNS_OPP] = _f32(opp["extra_turns"], 3)
        if self.turn:
            s[E._EXTRAS_IS_DAY] = 1.0 if self.turn["day_night"] == self.ad._dn["DN_DAY"] else 0.0
            s[E._EXTRAS_IS_NIGHT] = 1.0 if self.turn["day_night"] == self.ad._dn["DN_NIGHT"] else 0.0
            if self.turn["priority_window"]:
                s[E._EXTRAS_IS_PRIORITY_WINDOW] = 1.0
                s[E._EXTRAS_SELF_PASSED] = me["passed"]
                s[E._EXTRAS_OPP_PASSED] = opp["passed"]
            s[E._EXTRAS_MC_ONEHOT_START + self.turn["mandatory"]] = 1.0
        if not header["sideboarding"]:
            s[E._EXTRAS_SELF_MULLIGANS] = _f32(me["mulligans"], E.MULLIGAN_NORMALIZER)
            s[E._EXTRAS_OPP_MULLIGANS] = _f32(opp["mulligans"], E.MULLIGAN_NORMALIZER)
            s[E._EXTRAS_SELF_BOTTOM_REMAINING] = _f32(me["to_bottom"], E.MULLIGAN_NORMALIZER)
        match = self.t["match"][0] if self.t.get("match") else None
        if match:
            s[E._EXTRAS_PLAYS_FIRST] = 1.0 if match["plays_first_rel"] == _SELF else 0.0
            s[E._EXTRAS_SB_SWAPS] = _f32(match["sb_swaps"], SIDEBOARD_SWAP_CAP)
            s[E._EXTRAS_SB_DELTA] = np.float32(match["sb_drift"])

    def _decklists(self):
        """The viewer's current 75 (``own_deck``) and the opponent's registered list
        (``decklist_card``), packed ascending by card id, counts / 4. The opponent slots' revealed
        bit is not rebuilt (the agent never reads it): 0. Cached per own-deck configuration."""
        own = tuple(sorted((r["card"], r["main"], r["side"]) for r in self.t.get("own_deck", [])
                           if r["player_rel"] == _SELF))
        patch = self.ad._deck_patches.get(own)
        if patch is None:
            idx, vals = [], []
            self._pack(idx, vals, E._SELF_DECK_MAIN_START, E.DECKLIST_MAIN_SLOTS, E._DECKLIST_SLOT_SIZE,
                       [(c, m) for c, m, _s in own if m])
            self._pack(idx, vals, E._SELF_DECK_SIDE_START, E.DECKLIST_SIDE_SLOTS, E._DECKLIST_SLOT_SIZE,
                       [(c, sd) for c, _m, sd in own if sd])
            if self.ad.decklists is not None:
                opp = sorted((r for r in self.ad.decklists.tables.get("decklist_card", [])
                              if r["player_rel"] == _OPP), key=lambda r: r["card"])
                self._pack(idx, vals, E._OPP_DECK_MAIN_START, E.DECKLIST_MAIN_SLOTS, E._OPP_DECKLIST_SLOT_SIZE,
                           [(r["card"], r["main"]) for r in opp if r["main"]])
                self._pack(idx, vals, E._OPP_DECK_SIDE_START, E.DECKLIST_SIDE_SLOTS, E._OPP_DECKLIST_SLOT_SIZE,
                           [(r["card"], r["side"]) for r in opp if r["side"]])
            patch = (np.array(idx, dtype=np.int64), np.array(vals, dtype=np.float32))
            self.ad._deck_patches[own] = patch
        self.state[patch[0]] = patch[1]

    @staticmethod
    def _pack(idx, vals, start, n_slots, width, entries):
        for k, (cid, count) in enumerate(entries[:n_slots]):
            idx += [start + k * width, start + k * width + 1]
            vals += [_norm_id(cid), _f32(count, 4)]

    def _per_turn(self):
        s = self.state
        for rel, start in ((_SELF, E._PER_TURN_START), (_OPP, E._PER_TURN_OPP_START)):
            p = self.players.get(rel)
            if p is None:
                continue
            for off, key in ((E._PT_SPELLS, "spells_turn"), (E._PT_NONCREATURE, "noncreature_turn"),
                             (E._PT_INSTANT_SORCERY, "instant_sorcery_turn"), (E._PT_CARDS_DRAWN, "drawn_turn")):
                s[start + off] = _f32(p[key], E.PER_TURN_COUNT_NORMALIZER)
            s[start + E._PT_LIFE_GAINED] = _f32(p["life_gained_turn"], E.LIFE_NORMALIZER)
            s[start + E._PT_LIFE_LOST] = _f32(p["life_lost_turn"], E.LIFE_NORMALIZER)
            for c in range(E.PER_TURN_COLOR_FIELDS):
                s[start + E._PT_COLORS_START + c] = 1.0 if p["colors_cast_turn"] & (1 << c) else 0.0

    # ── the menu: per-action arrays ────────────────────────────────────────────────────────
    def menu(self, obs):
        """Fills the per-action blocks of ``obs`` from the ``menu`` table and returns the
        side-channels. Per action, as machine_io.cpp's populate_query gives them for its
        source: the category, the card id (action_card_vocab_idx: a permanent's face, a back
        face played or cast as it), controller-is-self (the null sentinel when it has no zone
        reference), the zone reference, the entity slot, the option ordinal; the card-is-public
        flag and the description (cut to MAX_CHOICE_DESC - 1 bytes) ride beside them."""
        rows = self.t.get("menu", [])
        n = min(len(rows), E.MAX_ACTIONS)
        cats = np.zeros(E.MAX_ACTIONS, dtype=np.int32)
        ids = np.full(E.MAX_ACTIONS, _norm_id(-1), dtype=np.float32)
        ctrl = np.full(E.MAX_ACTIONS, _norm_id(-1), dtype=np.float32)
        pub = np.zeros(E.MAX_ACTIONS, dtype=np.float32)
        zone = np.zeros(E.MAX_ACTIONS, dtype=np.int32)
        refs = np.full(E.MAX_ACTIONS, -1, dtype=np.int32)
        ords = np.full(E.MAX_ACTIONS, -1, dtype=np.int32)
        descs = []
        z = self.ad._zone
        back = self.ad._variant_bit["BACK_FACE"]
        for i, r in enumerate(rows[:n]):
            cats[i] = r["category"]
            o = self.objects.get(r["source"]) if r["source"] else None
            cid = _card(r["source_card"])
            zr = R.REF_NONE
            is_self = False
            if o is not None:
                if o["zone"] == z["BATTLEFIELD"]:
                    cid = self._face_id(o)
                elif o["zone"] == z["STACK"]:
                    cid = self._target_card(o)
                is_self = o["controller_rel"] == _SELF
                zr = {z["BATTLEFIELD"]: R.REF_SELF_BATTLEFIELD if is_self else R.REF_OPP_BATTLEFIELD,
                      z["HAND"]: R.REF_SELF_HAND if o["owner_rel"] == _SELF else R.REF_OPP_HAND,
                      z["STACK"]: R.REF_STACK,
                      z["GRAVEYARD"]: R.REF_SELF_GY if o["owner_rel"] == _SELF else R.REF_OPP_GY,
                      z["EXILE"]: R.REF_SELF_EXILE if o["owner_rel"] == _SELF else R.REF_OPP_EXILE,
                      }.get(o["zone"], R.REF_NONE)
                refs[i] = self.slot.get(r["source"], -1)
            elif r["source_player_rel"]:
                is_self = r["source_player_rel"] == _SELF
                zr = R.REF_PLAYER_SELF if is_self else R.REF_PLAYER_OPP
                cid = -1
            if r["variant"] & back and cid >= 0 and (o is None or o["zone"] != z["BATTLEFIELD"]):
                cid = self._back_face(cid)
            ids[i] = _norm_id(cid)
            zone[i] = zr
            if zr != R.REF_NONE:
                ctrl[i] = 1.0 if is_self else 0.0
            pub[i] = r["card_public"]
            ords[i] = r["ordinal"]
            descs.append(r["description"].encode("utf-8", "surrogateescape")[:E.MAX_CHOICE_DESC - 1]
                         .decode("utf-8", errors="replace"))
        descs += [""] * (E.MAX_ACTIONS - len(descs))
        obs[E.ACT_CATS_START:E.ACT_CATS_START + E.MAX_ACTIONS] = cats / E.ACTION_CATEGORY_MAX
        obs[E.ACT_IDS_START:E.ACT_IDS_START + E.MAX_ACTIONS] = ids
        obs[E.ACT_CTRL_START:E.ACT_CTRL_START + E.MAX_ACTIONS] = ctrl
        obs[E.ACT_ZONE_START:E.ACT_ZONE_START + E.MAX_ACTIONS] = zone / E.REF_ZONE_MAX
        obs[E.ACT_REFS_START:E.ACT_REFS_START + E.MAX_ACTIONS] = (refs + 1) / E.N_ENTITY_REF_SLOTS
        obs[E.ACT_ORDS_START:E.ACT_ORDS_START + E.MAX_ACTIONS] = (ords + 1) / (E.OPTION_ORDINAL_MAX + 1)
        return {"num_choices": n, "cats": cats, "ords": ords, "public": pub, "descriptions": descs,
                "pending_confirm": any(int(c) in E.MANDATORY_CATS for c in cats[:n])}
