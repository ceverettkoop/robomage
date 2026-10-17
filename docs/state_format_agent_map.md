# Agent read map: scripted agent → RMSTATE

Step-1 companion to [`state_format.md`](state_format.md). It maps every value the scripted agent
(`train/scripted_agent.py`) reads today onto what will serve it in RMSTATE. It is the
completeness checklist for the step-2 port and the specification of the pass-1 parity adapter.

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
   divergences (`train/regression/knowledge_divergences.json`) are the parity caveats of the
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
