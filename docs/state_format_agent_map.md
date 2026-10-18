# Agent read map: scripted agent → RMSTATE

Step-1 companion to [`state_format.md`](state_format.md). It maps every value the scripted agent
(`train/scripted_agent.py`) reads today onto what will serve it in RMSTATE. It is the
completeness checklist for the step-2 port and the specification of the pass-1 parity adapter.

**Status:** step 2 pass 1 is built (section 7): the parity adapter (`train/rmstate_adapter.py`,
`train/rmstate_env.py`) serves every read below, and the `agentparity` gate tier
(`train/test_agent_parity.py`, `docs/ci.md`) holds the agent's RMSTATE games byte-identical with its
BQUERY games, except listed and classified old-vector leaks and quirks. Pass 2 is built (section 8):
the agent's decisions are one view-independent core read through a decoded-frame view
(`train/rm_agent.py`) on RMSTATE and through the vector view on BQUERY; the replay corpus holds its
RMSTATE games. Step 2 is complete (section 9): the stdlib-only RMSTATE driver
(`train/rm_driver.py`) plays the scripted tiers and the explore fuzzer, and the `replay`, `smoke`
and `fuzz` gate tiers run on it.

**Legend for "served by":**

| Prefix | Source |
|---|---|
| `S.<table>.<column>` | snapshot table (state_format §4) |
| `M.<column>` | the menu table |
| `C.` | the once-per-match `CARDS` table |
| `D.` | the once-per-match `DECKLISTS` table |
| `K.` | the knowledge sidecar's `k_*` tables at level `expert` ([`knowledge_store.md`](knowledge_store.md)) |
| `—` | not needed (dead code or unused) |

Line numbers are `scripted_agent.py` unless a file is named. The agent reaches the state four
ways: raw `obs[...]` reads, `decode.decode_game_state` (`g`, built once per decision at
`:3004`), the per-action metadata arrays, and card knowledge from outside the engine.

## Findings

1. **Every read has a home** in the spec as drafted. The exceptions are three small additions
   listed at the end (deck name, `is_token`, keyword key on ability references).
2. **Engine-side memory the agent actually uses is small:**
   - the viewer's own known library top (`_led_crack_choice` :732 tests for Thassa's Oracle on
     top; `_shuffle_pick` :2332 reads the top 3);
   - `opp_known_hand` (`_name_card_pick` :2361).

   It never reads the opponent's known library top, and it binds but ignores the decklist
   "revealed" bit (`_rev`, :2355). The agent therefore needs only those two outputs from the
   knowledge sidecar. The full shadow comparison with today's engine memory
   (`knowledge_store.md` §7, §9.7) is the broader event-completeness check; its classified
   divergences (`tests/knowledge_divergences.json`) are the parity caveats of the
   port (`knowledge_store.md` §8).
3. **Player identity is never needed beyond "self/opp".** The seat flag is read only to pick
   which deck *name* belongs to the agent (`_seat_deck` :2895).
4. **Out-of-band inputs exist today:**
   - deck names are pushed by `set_deck_names` (:2883) from the driver;
   - the opponent's `.dk` file is read from disk (`_deck_runs_force_of_will` :671);
   - card scripts are read from disk (`decode._card_script_lines`).

   All of them move into `D.`/`C.`, so the agent needs no filesystem access.
5. **Menu order is part of the contract.** About a dozen decisions are positional: index 0 /
   last index / "index = X" (table 2b). The engine's menus are unchanged until step 4a, so
   parity holds. Step 4a must keep these orders or update the agent with them.
6. **Dead code:** `evaluate(g)` (:2071) is never called; its reads are dropped.

## 1. Snapshot reads

### 1a. Players, turn, header

| Quantity | Read today | Served by | Parity note |
|---|---|---|---|
| own life | `obs[0]`×20 (`_self_life` :752); `g["self"]["life"]` | `S.player.life` | integer, exact |
| opponent life | `g["opponent"]["life"]` (:2673, 3586, 3648) | `S.player.life` (opp row) | exact |
| own floating mana W,U,B,R,G,C | `obs[_PB_MANA+k]`×10 (`_floating_mana` :1458; blue ≥ 2 at :1799) | `S.player` mana pool, by colour | colour order W,U,B,R,G,C baked in (adapter) |
| step is main 1 / main 2 | `obs[23]`, `obs[30]` > 0.5 (:1601, 3421) | `S.turn.step` | enum string table |
| step name | `g["step"] == "Declare Blk"` (`_ninjutsu_pick` :2448) | `S.turn.step` | name from schema enum |
| I am the active player | `obs[33]` (:1602, 2448, 3420, 3456) | `S.turn.active_player` == viewer | — |
| I am seat A | `obs[34]` (only to pick own deck name, :2895-2901, 2968, 3552) | `S.header.viewer` | used only with deck names (D.) |
| turn number | `obs[5776]`×50 (`_hard_act` :2969) | `S.turn.turn` | exact |
| own library count | `obs[5774]`×60; `g["self_library"]` (`_self_library_count` :747, many) | `S.player.library_count` | exact |
| own mulligans taken | `g["extras"]["self_mulligans"]` (:3565) | `S.player.mulligans` | — |
| hand counts | `g[...]["hand_count"]` (dead `evaluate` only) | — | — |

### 1b. Permanents

The agent addresses permanents through 48 self + 48 opp slots, via raw slot scans
(`_BF_START + slot*43`), the decoded lists `g["self_battlefield"]` / `g["opp_battlefield"]`,
and the menu's `ref` (§2).

| Field | Read today | Served by | Parity note |
|---|---|---|---|
| card id | slot offset 42 (`_controls_card` :463, many); `p["card_idx"]` | `S.object.card` | tokens: today a token's id is the token band (≥ 900), and `_is_token_id` :2216 relies on it. Needs `S.object.is_token` (addition 2) |
| phased out | offset 17 (`_phased` :414, almost every scan); `g()` filter :3010 drops them | `S.object.phased_out` | phased-out permanents stay in `S.object`, flagged, as today |
| tapped | offset 2; `p["tapped"]` | `S.object.tapped` | — |
| attacking / blocking / blocked | offsets 3, 4, 16; `p[...]` (:429, 2114-2116, 2403, 2451, 3600-3603) | `S.combat` (+ an `S.object` attacking/blocking flag) | — |
| summoning sick | offset 5; `p["summoning_sick"]` | `S.object.sick` | engine's own flag, as today |
| is creature | offset 7; `"power" in p` (`_creatures` :2068) | `S.object` post-layer types | adapter: power/toughness keys only for creatures |
| is land | offset 8; `p["is_land"]` | `S.object` post-layer types | — |
| power / toughness | offsets 0, 1 ×10; `p["power"]` / `p["toughness"]` (many) | `S.object.power` / `.toughness` (post-layer) | exact |
| loyalty | offset 9 ×10 (`_wrath_energy_plan` :2162, `_kaito_ability_pick` :2724) | `S.object_counter` (loyalty) | old key present only when nonzero |
| other counters | offset 11 (`_keep_legend_choice` :1427 by ref, raw /10; `p["other_counters"]` :2648) | `S.object_counter` (sum of kinds other than +1/+1, −1/−1, loyalty) | adapter returns the old sum (and /10 in the raw read) |
| attached-to | offset 12 > 1e-6 (`_equip_holds` :2510 by ref) | `S.object.attached_to` | bool |
| something is attached to it | `"attached_by_slot" in p` (:2551) | any `S.object` row whose `attached_to` is this handle | — |
| keywords Flying, Hexproof, Shroud | `p["keywords"]` (:2552, 2669) | `S.object_keyword` | full list now; adapter filters to the old 16 |

### 1c. Stack, hand, graveyard, exile, pending decision

| Quantity | Read today | Served by | Parity note |
|---|---|---|---|
| any stack object: controller is self, card id | 12 slots × (offset 0, offset 1) (`_opponent_has_spell_on_stack` :542, `_self_stack_has_card` :552, `_opponent_threat_on_stack` :564, `_opponent_stack_has_any` :2819, env `_self_has_draw_on_stack`) | `S.stack.controller`, `S.stack.source` → card | adapter caps at 12 |
| top of stack: controller, card id, spell vs ability | slot 0, offsets 0-2 (:1776-1777, 2965-2967, 3452-3453) | first `S.stack` row (kind) | row 0 = top (keep that order) |
| stack empty | env `_stack_is_empty`: all 12 ids < 0 | `S.stack` row count | — |
| own hand card ids | `obs[5760+slot]`, 10 slots (`_hand_land_count` :576, `_cast_holds` :3490, …; env `_hand_has_card`); `g["self_hand"]` | `S.object` rows in own hand | adapter caps at 10 |
| card in either graveyard | 128 slots, id at offset 0 (`_gy_has_any` :476) | `S.object` rows in graveyards | order-insensitive; adapter caps 64 per side |
| opponent graveyard / exile names | `g["opp_graveyard"]`, `g["opp_exile"]` (`_name_card_pick` :2358) | `S.object` rows in opp graveyard / exile | face-down exiled cards excluded (no identity), as today |
| opp exile card id + counters | opp exile slots, offsets 0 and 4 (`_best_void_card` :2698-2702) | `S.object` + `S.object_counter` (exile) | — |
| pending decision source card | `obs[5797]` (≈15 call sites: :1304, 1480, 1509, 1547, 2364, 2401, 2537, 3034…, 3627) | `S.pending.source` → card | the second float (controller) is unused |

### 1d. Memory (knowledge sidecar outputs)

| Quantity | Read today | Served by | Parity note |
|---|---|---|---|
| own library top card is X | `obs[5777]` (`_led_crack_choice` :732) | `K.k_position` (owner self, `at` top 0, `card` set) | engine source today: `Game::KnownLibraryTop` (by owner). Shadow comparison (`knowledge_store.md` §9.7): expert never knows less in a game; it knows more during a look (`pending_look`) and by deduction (`deduction`); between games the old vector still shows the finished game (`between_games`) |
| own known top 3 | `_known_top_ids(obs, 3)` (:2323 → `_shuffle_pick` :2332) | `K.k_position` (owner self, `at` top 0-2, `card` set) | same |
| opponent known hand cards | `g["opp_known_hand"]` (:2361) | `K.k_known_hand` | engine source today: `Zone::identity_known` (`opp_known_hand_cards`, `machine_io.cpp`; the vector shows the first 10). Shadow comparison: expert knows less after a hidden put-back (`put_back_leak`) and after a copy of a known card leaves the hand (`copy_leak`), both old-vector leaks; more after a known top card is drawn (`known_card_drawn`), a companion is put into hand (`public_entry_untracked`) and by deduction |
| opponent decklist (cid, count) | `decode_opp_decklist` (:2355) | `D.` opponent registered list | the `revealed` bit is unused |

## 2. Menu reads

### 2a. Per-action fields

| Field today | Uses (summary) | Served by | Parity note |
|---|---|---|---|
| category (`cats`, ~40 categories referenced; never CHOOSE_MANA_COLOR, CHOOSE_TYPE, ORDER_TRIGGERS, ASSIGN_DAMAGE, EXILE_FROM_YARD) | menu-shape predicates (`_is_yesno_menu` :1537, `_is_look_split_menu` :2302, all-CHOOSE_X :1723, …) and per-category handlers | `M.category` (enum by name from the schema) | category values identical (same engine) |
| card id (`ids`) | find the action for card X; classify options; null sentinel = non-card option (fail-to-find, player target, X ladder) | `M.source` → `S.object.card` / `C.` | null ⇔ no source handle |
| controller is self (`ctrl`) | prefer opponent targets; own-benefit targets; split discard menus | `M.target` → `S.object.controller` (or player) | the old "negative null" for player targets ⇔ target is a player |
| ref slot (`refs`) | join an action to its permanent (`_perm_by_slot` :2205; `_keep_legend_choice` :1427, `_equip_holds` :2510 index the vector by ref) | `M.source` / `M.target` handle → `S.object` row | adapter gives each handle a synthetic slot: self 0-47, opp 48-95, stack 96-107. The agent only joins on refs, never relies on their order |
| zone (`zone`; uses REF_NONE, REF_SELF_BATTLEFIELD, REF_SELF_HAND, REF_PLAYER_SELF, REF_PLAYER_OPP) | attack the player vs a planeswalker; ninjutsu = activation from hand; Kaito's battlefield abilities; burn at face | zone of `M.source`/`M.target` | adapter recomputes the old enum |
| ordinal (`ords`) | pay/decline; Sylvan; X value; activated-ability index (Ninjutsu, Equip, Reconfigure, Blast Zone, Kaito); mode index (Knight of Autumn) | `M.ordinal` (kept verbatim) + `M.ability` reference | pass 2 matches abilities by `M.ability` keyword key instead of `_keyword_ability_ordinal`'s script count (addition 3) |
| `num_choices` | everywhere | menu row count | — |
| action masks | never read | — | — |

### 2b. Positional (menu-order) assumptions to preserve

These are valid while engine menus are unchanged (through step 3). Step 4a must keep them or
change the agent alongside:

- sideboard Done = index 0 (:1630);
- the first non-MULLIGAN option = keep (:1636, 3572, 3760);
- search option 0 = fail-to-find, so the agent returns 1 (:1695);
- dig returns 1 (:2026);
- delve takes `num_choices-1` (:1725);
- X ladder index = X (:1483, 3060, 3287, 3300);
- yes/no: 0 = decline, 1 = accept, and option 1 carries the card id (:1545, 1778, 1786, 3444);
- surveil / Delver 2-PASS picks 1 (:2041);
- Karn −2 = the last of two activations (:3365);
- Doomsday rearrange position = count of remaining TOP_LIBRARY choices (:1347).

## 3. Card knowledge from outside the engine

| Knowledge | Read today | Served by |
|---|---|---|
| mana value | `_CARD_COST_MATRIX` row sum (`_card_mana_value` :299; ~14 callers) | `C.` mana value |
| coloured pips of a card | `_CARD_COST_MATRIX` W..G > 0 (`_land_search_pick` :2772) | `C.` mana cost |
| is a land | `_LAND_VOCAB_IDS` (~10 sites) | `C.` types |
| card types Artifact / Enchantment | `decode.card_types` → script `Types:` (`_is_artifact_or_enchantment` :2594) | `C.` types. Quirk: tokens return '' today (adapter reproduces; pass 2 uses real token types) |
| printed power | `card_props.power` (`_ninjutsu_pick` :2458, `_tutor_pick` :2742) | `C.` printed P/T |
| Delve, Equip, Miracle keywords | `card_props.kw_delve`, `.kw_equip`; script `K:Miracle` (`_card_has_keyword` :344) | `C.` keyword list |
| Equipment subtype, legendary, basic land types | `card_props.sub_equipment`, `.legendary`, `.land_*` (via `_LAND_TYPE_COLOR` :2749) | `C.` subtypes / supertypes |
| has a damage ability | `card_props.cat_dealdamage` (burn detection :3628) | `C.` ability list (category `DealDamage`) |
| land has a non-mana activated ability | script `A:` lines without `AB$ Mana` (`_has_nonmana_activation` :2384) | `C.` ability list (activated, category ≠ `AddMana`) |
| index of a keyword-built ability (Ninjutsu, Equip, Reconfigure) | count `A:` lines + keyword order (`_keyword_ability_ordinal` :352). It counts `A:SP$` lines too; correct only while the parser's index matches | `M.ability` keyword key (addition 3); parity pass keeps the ordinal comparison |
| name ↔ id | `_VOCAB_NAMES` / `_vid` (:324-327) | `C.` id ↔ name |
| ~40 named-card constants and ~25 card sets: Doomsday combo, mana-source tables, counterspells, removal, burn amounts, unless costs, ETB-value creatures, activation gates, Tron, Reanimator, Lands | env.py `_*_VOCAB_IDX` / `_*_IDS`, local `_vid(...)` and numeric tables | resolved by **name** through `C.`, built once per match. Numeric tables (e.g. `_BLUE_SOURCE_IDS` :613) are rewritten as name sets in pass 1; ids stay identical because the vocab is unchanged |

## 4. Out-of-band inputs

| Input | Today | Served by |
|---|---|---|
| deck names (own, opponent) | `set_deck_names` from the driver (:2883); `_deck_named` substring tests (doomsday, tron) | `D.` deck name per list (addition 1). The opponent's name is withheld under `--closed-decklist` |
| opponent's deck file contents ("runs Force of Will") | reads `bin/resources/decks/<name>.dk` (`_deck_runs_force_of_will` :671) | `D.` opponent list |
| card scripts | `decode._card_script_lines` | `C.` |

## 5. Pass-1 parity adapter: what it must emulate

1. **Caps:** 48 permanents per side, 12 stack objects, 10 hand cards, 64 graveyard and 64
   exile slots per side. Overflow drops the same entries the old packer dropped (its pack
   order: `machine_io.cpp`).
2. **Empty slots read as zero flags.** Several scans test only `is_land` / `is_creature` and
   rely on empty slots being 0. `_dd_sacrifice_pick` :1135 even adds the −1 id of empty untapped
   slots to `untapped_ids`. The adapter's raw-slot views must present empty slots the same way.
3. **Normalisers:** values decode to the same integers (life /20, mana /10, P/T /10, library
   /60, turn /50, counters /10, card id ×1024). The raw `other_counters` compare uses count/10,
   and `attached_to` > 1e-6 means "attached".
4. **Order:** stack row 0 = top; mana W,U,B,R,G,C; known-top index 0 = top. Graveyard,
   exile, hand and battlefield order are not relied on.
5. **Phased out:** phased-out permanents are present and flagged; the agent filters them.
6. **Token quirk:** `card_types` returns '' for tokens, and token ids sit in the token band.
7. **Keywords:** `p["keywords"]` holds only the old 16 (`_OBS_KEYWORDS`).

Pass 2 removes items 1–3 and 6–7 and reads decoded objects directly.

## 6. Additions to the spec from this map

1. **Deck name** per registered list in `D.` (opponent's withheld under `--closed-decklist`).
   The registered list is explicit information, and its name is part of it.
2. **`S.object.is_token`** (and token entries in `C.` with their real types).
3. **Keyword key on ability references** in `M.ability` / `S.object_ability` (e.g. `Ninjutsu`,
   `Equip`, `Reconfigure:attach`), so agents identify keyword-built abilities without counting
   script lines.

## 7. Pass-1 adapter as built (plan step 2)

**Design.** The agent's decision logic is unchanged. `rmstate_env.RMStateNarrativeEnv` runs the
engine with `--machine-format rmstate --rmstate-card-scripts`; `runner.run_games(machine_format=
"rmstate")` (`--machine-format rmstate` on `train.py observe` and the harness) selects it. Each seat's
frames feed that seat's `rmstate_adapter.SeatAdapter`; at each decision the deciding seat's STATE frame
becomes a vector in the old layout (`env.OBS_SIZE`) holding, at every position the agent reads, the
value the old vector gave, plus the per-action arrays, descriptions and card-is-public flags of the
menu. The BQUERY vector is never written in this mode. The agent's card knowledge goes through
`scripted_agent.use_card_knowledge`: `VocabCardKnowledge` (the generated tables, the scripts on disk,
the `.dk` files) on BQUERY, `rmstate_adapter.FrameCardKnowledge` (the match's `CARDS` and `DECKLISTS`
frames) on RMSTATE, installed by `opponents.ScriptedController` before each decision with the deck
names `DECKLISTS` gives. The vocab constants (`env.py`'s `_*_VOCAB_IDX` / `_*_IDS`, the agent's id
tables) are name lookups (`_card_id`, `_vid`); on RMSTATE the controller checks once per match that
the frame names every card the vocab knows with the vocab's id (`card_name_mismatches`).

**Seats and agents.** Both seats share one engine; the stream carries only the deciding seat's frames,
so each seat's adapter sees its own frames (its knowledge tables, its events) and the self
perspective is the frame's viewer. Every controller that declares `rmstate_capable` can play: the
scripted tiers (hard, easy/greedy, explore, explore:patient; random too, though it is not
deterministic), `auto`, `actions:` and `play:` (a script whose players are capable). Model, search
and human controllers read the whole vector and are refused (`runner.require_rmstate_capable`).

**Emulation, pinned down** (section 5's list, as built):

1. *Caps and order.* Battlefield: 48 per side by controller; graveyard and exile: 64 per side by
   owner in `pos` order (newest first); hand: 10; stack: 12 from the top. The old serializer packs the
   battlefield and the hand by entity id, which the format does not carry (it would tell copies of a
   card apart by deck position): entity ids run through Player A's list, then B's, then later objects
   (tokens), and every league list is in name order, so the adapter packs by (token, owner, card name,
   handle). This is entity order for league decks up to copies of one card; the agent was measured
   order-insensitive (one choice changed in 18,609 decisions with every hand and battlefield
   order permuted, 55 bo3 matches).
2. *Empty slots* are the serializer's empty slot (zeros, the empty card-id sentinel in every id
   float), also for an object the serializer writes empty: a permanent whose current face has no
   vocab entry.
3. *Normalisers* are computed in float32 as the serializer divides (life / 20, P/T and counters
   / 10, library / 60, turn / 50, …).
4. *Card ids as the serializer gives them:* a token's band id; a permanent by the name it has now (a
   transformed card's back face, a copy's copied card: `object.transformed`, `object.copy_of`); a
   menu action that plays or casts a back face (`variant` BACK_FACE) by that face; a stack ability by
   its source's face; a trigger with no source of its own (a floating or delayed trigger an effect
   created: no source, no ability reference into a card's list) as none, where the frame shows its
   creator; a card entering the battlefield whose permanent is not built yet (a shock land's life
   payment: no `timestamp`) is no permanent.
5. *Summoning sickness:* the old flag is `Permanent::has_summoning_sickness`, the raw flag (set as a
   creature or any token enters, cleared at its controller's untap step); the frame's `sick` is the
   rule (a creature, haste considered). They differ for a creature with haste, a token and a creature
   card that is not a creature now (impending); the adapter reads those from the permanent's entry turn
   (its `zone_move` event, kept per seat) against the start of its controller's most recent turn; a
   harness preset starts without it.
6. *Keywords:* the old 16 (`_OBS_KEYWORDS`), by the exact keyword text.
7. *Knowledge reads:* own and opponent library tops 0–4 from `k_position` rows with an exact card;
   the opponent's known hand from `k_known_hand` (each card's copies, ten slots, in name order).
8. *Card knowledge:* `FrameCardKnowledge` reads each card's raw script from `CARDS` with the
   generators' own parsing, so every cost row, land flag, property column, script line and type line
   equals the generated tables' (checked over the whole vocab by the tier's `cards` part); the token
   quirk is kept (a token has no card script, so its type line is '').
9. *Deck names and the opponent's list* from `DECKLISTS` (the "runs Force of Will" test reads the
   opponent's registered list); the opponent decklist block from `decklist_card`, the own deck block
   from `own_deck`.
10. *Between games* the vector is masked as the BQUERY env masks a sideboarding observation.
11. *Not rebuilt* (the agent reads none of it): the opponent decklist slots' revealed bit (0), the own
    live-library block, mana development, log vitals, delayed triggers, player effects, the
    per-permanent counter and token-name side channels.

**Differences that remain** (field level, from `test_agent_parity.py --lockstep` over the corpus and
smoke games; none of them reached a decision except the knowledge reads):

| Field | Old vector | Adapter | Why |
|---|---|---|---|
| own library top, opponent's known hand | the old memory | `k_position`, `k_known_hand` | the classified divergences of `knowledge_divergences.json` (`knowledge_store.md` §8, §9.7) |
| zone and controller of a name-a-card choice | the zone of the card the engine picked to stand for the name (it can be in the opponent's hand or a library) | none | an old-vector leak; the frame has no handle for a hidden representative |
| stack slot "is a spell" of a resolving permanent spell waiting on its entry's choice (Mox Diamond's discard) | 0 (its Spell component is gone) | 1 (`stack.kind` SPELL) | an old-vector quirk: it is still a spell (CR 608.3) |
| card of a stack target that has left the stack (a countered spell's target spell) | its card | none | a format gap: last-known information was kept for permanents only (`lki` rows); not read by the agent. Fixed in pass 2 (register 40: an `lki` row for a departed stack target too) |

Format fixes made for the port: `object.types` has Creature exactly while the permanent is a
creature (an impending permanent and an attached reconfigure permanent were reported as creatures);
a phased-out creature's `power` / `toughness` are its own; `CARDS` carries the raw scripts on request
(`--rmstate-card-scripts`, register 39); a decision's narrative reaches stderr before its frame.

**Parity results.** All 64 replay-corpus games, the 40 smoke-tier games and a bo3 match per league
mirror are byte-identical through the projection; a sweep of 300 bo3 matches (every ordered league
pairing, three seeds) found six divergent games, each a Cabal Therapy name pick reading the
opponent's known hand: five `put_back_leak` (the old vector still listed a card the opponent had put
back with Brainstorm) and one `public_entry_untracked` (a companion put into hand, known to both
players). They are listed in `train/regression/agent_parity_divergences.json` and re-recorded. The
explore, explore:patient and easy tiers are identical on both protocols too.

**For pass 2:** read decoded objects instead of the vector; the caps, the slot order, the empty-slot
and normaliser emulation and the token type-line quirk go. The structured `CARDS` columns can replace
the raw scripts (the tier's `cards` part shows what each column must reproduce); `M.ability`'s keyword
key replaces `_keyword_ability_ordinal`; the summoning-sickness emulation goes once the agent reads
`object.sick` (the rule); the richer knowledge tables (pools, positions as candidate sets) can serve
the Ponder / Brainstorm / Doomsday decisions directly.

## 8. Pass 2 as built (plan step 2)

**Structure.** The decisions are written once, in `train/scripted_core.py` (`ScriptedCore`: the
tiers, presets and every rule), over an abstract view of one decision (its module docstring lists
the view's reads): cards are names (the constants of the rules are card names, checked against a
closed-decklist `CARDS` frame by the `rmagent` tier), categories, steps and zone references are the
engine's enum names, card characteristics come from the view's card knowledge by name. It imports no
numpy and nothing of the BQUERY layout. Two views implement it:

- `scripted_agent.VectorView` (BQUERY; `ScriptedAgent.act(obs, num_choices)`, `make_agent`,
  `scripted_action` keep their API): every read exactly as the agent read the vector before the
  split, quirks included (the caps and slot order, the null-card float test that reads ids 0-9 as
  null, ids one apart as one card in the two-pass reveal rule, joins by card id, keyword abilities
  by script-counted ordinal, the vocab tables' card knowledge). Checked in lockstep with the
  previous agent: 654 bo3 matches (every ordered league pairing and the curated decks; hard at
  three seeds, easy, explore and explore:patient at one), 278,125 decisions, no choice and no RNG
  state differs. The BQUERY consumers (PPO's scripted opponents, `scripted_oracle.py`, the
  BQUERY gate tiers) play the same games, about 1.6x faster (release, hard vs hard: 650 to about
  1,050 decisions/s).
- `rm_agent.FrameView` (RMSTATE; `rm_agent.decide`): built from one seat's decoded frame
  (`rm_agent.SeatFrames` follows the seat's SCHEMA, CARDS, DECKLISTS and METAGAME frames).

On `--machine-format rmstate` a plain scripted spec reads the frame; `scripted?view=vector` keeps
the pass-1 path (section 7), which the `agentparity` tier still holds to the BQUERY games. The env
builds the adapter's whole observation (and asks for `--rmstate-card-scripts`) only for that path;
otherwise only the fields the driver and the menu-reading controllers (`auto`, `actions:`,
`play:`) read, and the menu through the driver is unbounded.

**What the frame view reads, against section 5's emulation list:**

| Read | Pass 2 |
|---|---|
| caps (48 / 12 / 10 / 64) | none: every permanent, stack object, hand, graveyard and exile card |
| slot order, empty slots, normalisers | none: `object` rows by handle, values as decoded |
| summoning sickness | `object.sick`, the rule (a creature, haste considered) |
| keywords | every `object_keyword` (by keyword name) |
| tokens | real characteristics from `object` / `CARDS` (types, mana value) |
| a permanent's mana value, types, legendary | the `object` row's (post-layer, rules-correct), not its card's |
| an action's object | `menu.source` by handle (`exact_refs`): attackers, blockers, the pending blocker, targets |
| keyword-built abilities | the menu's ability key (`ab_key`: `Ninjutsu`, `Equip`, `Reconfigure`) |
| card knowledge | `CARDS`' structured columns (`rm_agent.FrameCards`): mana value, plain and hybrid / Phyrexian pips, types, supertypes, subtypes, keywords, printed power, ability kinds and categories, oracle text; no raw scripts |
| own known top, opponent's known hand | `k_position` (exact rows), `k_known_hand` |

**Knowledge-table use** beyond pass 1's two reads:

- Cabal Therapy names a card the opponent's hand can hold first (`k_pool` HAND, most copies > 0):
  in game 1 a sideboard card cannot be in hand (it named Leyline of the Void before);
- the Doomsday pile's Island: whether the own library can hold it, from the own library pools and
  positions (it was "not on our battlefield, in hand or in a graveyard");
- Ponder's shuffle reads each top position's candidate set (an exact card is a set of one);
- under `--closed-decklist` the opponent's list is the reference lists `k_template` still holds
  consistent (METAGAME): whether it can run Force of Will (the Doomsday all-in mulligan) and each
  card's most copies (name-a-card's unseen counts).

**Differences from the pass-1 agent** are those intended above. Over 100 bo1 games played by the
pass-1 agent with both views deciding at every frame, 7 of 15,760 choices differed: Cabal Therapy
names (2: sideboard cards), reanimation targets (3: Raph & Mikey, Troublemakers' mana value is 7,
not the cost table's 9), attacks (2: a hasty creature is not summoning sick, so its P/T is weighed).
The re-recorded replay corpus changed in 4 of its 64 games, for the same reasons.

**Head-to-head** (release, both agents over RMSTATE, scripted hard, bo3; every ordered league
pairing with each agent in each seat at the same seed, 13 seeds): 2,600 matches, no draw, none
capped; pass 2 won 1,306, 0.502 (95% Wilson interval 0.483-0.522). The two agents agree on almost
every decision (over 300 bo3 matches played by pass 1 with both views deciding, 69 of 115,167
choices differ), so the paired count is the sharper one: of the 1,300 seed pairs, 20 had the same
agent win both seatings, 13 pass 2 and 7 pass 1 (sign test p = 0.26: no significant difference,
none in pass 1's favour).

**Performance** (release, one process, scripted hard vs hard, 20 league matchups): BQUERY about
1,050 decisions/s; RMSTATE pass 1 about 390; pass 2 about 450. On pass 2 the engine's frames and
knowledge sidecar are about 45% of the time, the Python decode about 25%, the agent about 13%.

**numpy:** `scripted_core.py` and `rm_agent.py` import none; nor does the RMSTATE driver
(section 9). The BQUERY driver (`env.py`, `runner.py`, `opponents.py`) and the pass-1 adapter do.

## 9. The driver and the explore tier (plan step 2, end)

**Driver.** `train/rm_driver.py` replaces `env.py` / `runner.py` / `opponents.py` for agent games
on RMSTATE. It imports the standard library, `rmstate.py`, `rm_agent.py` and the agent modules
only (the `rmagent` tier's `driver` part runs a game in a subprocess and fails on any module of the
RL stack in `sys.modules`). Its parts:

- `Engine`: the subprocess (`--machine-format rmstate`, the engine's cwd `bin/`), records read from
  stdout and lines from stderr under one `select` loop (the engine flushes a decision's narrative
  before its frame, so a non-blocking drain after the frame has all of it), every frame fed to its
  seat's `rm_agent.SeatFrames`, decision frames returned, replies written; diagnostics to the
  driver's stderr; a nonzero exit raises.
- agents (`make_agent`): the scripted tiers and explore (`ScriptedCore` through `rm_agent.decide`),
  `first` / `auto`, `actions:`; `play:` stays on `runner.py` until step 3a's C++ resolver.
- `play_game` (one game or match: `random.seed`, `new_game`, the decision loop, the confirm-row
  `-1` remap of attacker/blocker declarations, concessions, `GAME_RESULT` scoring) and `run_games`
  (transcripts, results, draw logs, the bo3 per-game split).
- its CLI: a flag table of its own (`CLI_FLAGS`, Arg fields as dicts) built with `argparse`;
  `cli_spec.DRIVER_TOOL` registers the same table, so the parser needs no `cli_spec`.

What it reads from the frames besides the agent's view: `header` (terminal, sideboarding),
`menu` (category, description, ordinal: the transcript and the reply), `turn` (turn, step, active
seat), `match.plays_first`, own `HAND` `object` rows and the `CARDS` names (the hand lines).

**Corpus format: unchanged.** The verbose transcript reproduces `runner.run_games`' RMSTATE
transcript byte for byte, so the replay corpus is identical (all 64 games) and was not re-recorded.
What that took: the engine seed of game seed *s* is the one the gymnasium env draws
(`np.random.Generator(PCG64(SeedSequence(s))).integers(0, 2**31 - 1)`, `rm_driver.engine_seed`,
pure Python: numpy's seed-sequence hash, PCG64 XSL-RR and its 32-bit Lemire draw); the hand lines
keep the adapter's order (token, owner, front-face name, handle), cap (10) and names (the card
vocabulary's: accents folded, a split card's halves joined by "/"), and while sideboarding the
masked observation's readings (no hand, "Player" the other seat). Over 80 bo3 matches and bo1
games (scripted hard, explore, explore:patient, every league deck) the two drivers wrote identical
transcripts; three of those stay in the `rmagent` tier. The compact transcript names categories
and steps by the schema's enum names (not pinned by any tier).

**Explore.** The tier already decided on the shared core's view (pass 2); its rule now lives in
`train/explore_agent.py` (`explore_choice`, called by `ScriptedCore`), with the rules listed in its
docstring for the step-3a C++ port (weights by category, the fail-to-find de-weight, the novelty
and patient factors, the draw by Python's `random.choices`). It reads `n`, `cats`, `card(i)`,
`hand` and `cards.mana_value` only, so its games are the same on both protocols: the `fuzz` tier,
moved to the driver, plays the same 80 games as on BQUERY (transcripts equal but for the board
block). Moving the code changed no BQUERY explore game either.

**Gate tiers on the driver** (debug build, 4 cores): `replay` 21.6 s → 20.5 s; `smoke` 13.5 s →
12.7 s (transcripts byte-identical); `fuzz` 21 s on BQUERY → 37 s on RMSTATE (the debug engine's
frames and knowledge sidecar; the sidecar alone is about a third of it); the `rmagent` tier's
`closed` part runs on the driver, and with its new `driver` part the tier takes about 13 s (10 s
before). `agentparity` stays on `runner.py`: every part needs the pass-1 adapter or BQUERY.

**Performance** (release, one process, 20 league matchups, bo1, no narrative, quiet): scripted hard
vs hard: BQUERY 1,116 decisions/s, RMSTATE through `runner.py` 491, through the driver 533; explore
vs explore: 1,194, 447, 477. Under the driver about half the time is the engine (frames, knowledge
sidecar), a quarter the Python decode (`rmstate.Schema.decode`), a sixth the agent.

**What still needs the old stack for agent games:** `play:` specs and the test harness (`--play`,
zone presets; step 3a ports the resolver and adds engine presets), `human` seats and the model
specs (deleted in 3b), the `agentparity` tier (the pass-1 adapter), and `train.py observe` /
`runner.run_match` callers that are not tiers.
