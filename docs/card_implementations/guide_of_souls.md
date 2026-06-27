# Guide of Souls  (vocab index 171)

## Oracle text
Whenever another creature you control enters, you gain 1 life and get {E} (an energy counter).

Whenever you attack, you may pay {E}{E}{E}. When you do, put two +1/+1 counters and a flying
counter on target attacking creature. It becomes an Angel in addition to its other types.

(1/2 Human Cleric, mana cost {W}.)

## Forge script
- Source: pre-existing local script — `bin/resources/cardsfolder/g/guide_of_souls.txt` (not edited).
- Key tags:
  - `T:Mode$ ChangesZone | ... | ValidCard$ Creature.Other+YouCtrl | Execute$ TrigGainLife`
    → `DB$ GainLife | Defined$ You` → `SubAbility$ DBEnergy = DB$ PutCounter | Defined$ You |
    CounterType$ ENERGY | CounterNum$ 1`.
  - `T:Mode$ AttackersDeclared | AttackingPlayer$ You | Execute$ TrigImmediateTrig`.
  - `TrigImmediateTrig:AB$ ImmediateTrigger | Cost$ PayEnergy<3> | Execute$ TrigPutCounter`
    ("you MAY pay {E}{E}{E}. When you do, ...").
  - `TrigPutCounter:DB$ Pump | ValidTgts$ Creature.attacking | SubAbility$ DBCounter`.
  - `DBCounter:DB$ PutCounterAll | ValidCards$ Creature.targetedBy | CounterType$ P1P1 |
    CounterNum$ 2 | CounterType2$ Flying | CounterNum2$ 1 | SubAbility$ DBAnimate`.
  - `DBAnimate:DB$ Animate | Defined$ Targeted | Types$ Angel | Duration$ Permanent`.
- Tags parsed as written; no category was retagged. `TriggerDescription$` (cosmetic prose on
  the ImmediateTrigger) is now ignored like `SpellDescription$`/`StackDescription$`.

## Engine work (all general, keyed on each tag's intended meaning)

### Energy foundation (reused later by Wrath of the Skies and Amped Raptor)
- **Generic player counter map + poison migration** — `src/components/player.h`. Replaced the
  dedicated `uint8_t poison_counters` field with `std::map<std::string,int> counters` plus
  `counter_count(type)` / `add_counters(type, delta)` helpers. Poison now lives under
  `"POISON"`, energy under `"ENERGY"` (CR 122.1c). Every former reader migrated:
  `src/machine_io.cpp` (`fill_player_stats` → `counter_count("POISON")` — the state vector stays
  byte-identical, no energy slot added), `src/error.cpp` (debug dump), `src/classes/game.cpp`
  (`gen_player` no longer zero-inits, the map defaults empty). The display struct
  `PlayerState.poison_counters` (`gamestate.h`) is unchanged.
- **`pay_energy` helper** — `src/game_queries.h`: `int player_energy(const Player&)` and
  `bool pay_energy(Player&, int n)` (returns false and leaves the pool untouched if insufficient;
  deducts on success; `n<=0` is a no-op true). Single read/spend path for {E}.
- **`PayEnergy<N>` cost token** — parsed in BOTH cost parsers in `src/parse.cpp`
  (`parse_alt_cost_tokens` and `parse_activation_cost`), mirroring `PayLife<N>`. Stored as
  `AltCost::energy_cost` (`src/components/carddata.h`) and `Ability::energy_cost`
  (`src/components/ability.h`).
- **PutCounter on a player (`Defined$ You`)** — `src/effects/effect_put_counter.cpp`: when
  `ab.defined_you`, the counters go on the controlling Player's counter map instead of a
  permanent (CR 122.1c). Energy counter type `"ENERGY"`.

### Triggers / reflexive cost
- **`Mode$ AttackersDeclared` ("whenever you attack")** — new `Events::ATTACKERS_DECLARED`
  (id 14, PLAYER = attacking player) in `src/ecs/events.h`, fired once when one or more
  attackers are declared in `src/action_processor.cpp`. Parsed in `src/parse.cpp`
  (`AttackingPlayer$ You` → `trigger_valid_player_is_controller`). The generic trigger scan
  already filters on the PLAYER param.
- **ImmediateTrigger with an optional cost** — `src/effects/effect_immediate_trigger.cpp`: when
  the ImmediateTrigger carries `energy_cost > 0`, it offers an OPTIONAL_YESNO "Pay N energy"
  decision (only when the controller can actually pay); accept → pay + run the Execute chain,
  decline / insufficient → skip the reflexive effect (CR 603.2c reflexive trigger). The yes/no
  prompt is the new reusable `request_optional_yesno(chooser, prompt)` in
  `src/input_logger.{h,cpp}`.
- **AB$ in an Execute$/SubAbility$ SVar** — `parse_svar_ability` (`src/parse.cpp`) previously
  only recognized a leading `DB$`; it now also accepts `AB$`/`SP$` (same 4-char offset), so the
  `AB$ ImmediateTrigger | Cost$ PayEnergy<3>` SVar parses with its category and cost (it had been
  silently parsed as an empty ability).

### Counters / Animate
- **PutCounterAll second counter + keyword counters** —
  `src/effects/effect_put_counter_all.cpp` / `src/components/ability_params.h`: `CounterParams`
  gained `type2`/`count2`, parsed from `CounterType2$`/`CounterNum2$`
  (`src/effects/effect_put_counter.cpp`). `ValidCards$ Creature.targetedBy` resolves to the
  inherited parent target (`ab.target`). A counter whose type names a keyword grants that keyword:
  `is_keyword_counter_type` in `src/game_queries.h` + the keyword-counter rebuild in
  `gather_active_statics` (`src/systems/state_manager_statics.cpp`) merges such keywords onto the
  creature each layer pass (CR 122.1d). So the flying counter grants Flying.
- **`DB$ Animate` (new general effect)** — `EffectKind::Animate` registered in
  `effect_kind.{h,cpp}`, `effect_table.cpp`, `effects.h`; handler `src/effects/effect_animate.cpp`.
  Parsed `Types$` (classified TYPE/SUBTYPE/SUPERTYPE via `parse_types`) and `Duration$`
  (`src/parse.cpp`) onto `Ability::animate_types` / `animate_duration_permanent`. The handler
  bakes the granted types onto the target's `Permanent::animate_added_types`; the layer-4 pass
  (`apply_type_changing_effects`) reapplies them every SBE so they survive the per-pass rebuild
  for the rest of the game. **Extension points** (`Permanent::animate_added_keywords`,
  `animate_set_pt`/`animate_power`/`animate_toughness`, `animate_make_creature`) are wired through
  the same fields for a later Earthbend land-animation card (set base P/T, grant Haste, add a
  Creature component to a noncreature) — currently only the type-add path is implemented/tested.

## Behavioral decisions (made in lieu of asking a human)
- **Energy is internal player state, not exposed in the observation/state vector** (per the run
  constraint): `STATE_SIZE`/`OBS_SIZE`/`N_CARD_TYPES` and the poison index are unchanged.
- **"you may pay {E}{E}{E}" is offered only when payable** (CR: an optional cost a player can't
  afford simply isn't taken): the pay prompt does not appear with <3 energy, and declining spends
  nothing and applies no effect.
- **The flying counter grants Flying** (CR 122.1d keyword counters): implemented generally for the
  standard keyword-counter names, not special-cased to this card.
- **"becomes an Angel in addition to its other types"** adds the subtype without removing existing
  types (Birds of Paradise keeps Bird), and persists for the rest of the game (Duration$
  Permanent).

## Tests (test_harness, seed 1, Guide on battlefield)
- **ETB life + energy:** another creature (Containment Priest) entering → "Player A gains 1 life
  (now at 21)" and "Player A gets 1 ENERGY counter(s) (now 1)"; accumulates 1 per creature.
- **Attack, pay 3 energy:** after declaring an attacker with 3 energy, the "Pay 3 energy"
  OPTIONAL_YESNO appears; on Accept, 3 energy paid, target attacker gets two +1/+1 counters and a
  flying counter (Birds of Paradise 0/1 → 2/3), and becomes an Angel (persists across turns).
- **Flying functional:** the animated flyer attacks and an opposing ground creature is "not
  eligible to block" — the flying counter granted Flying.
- **Decline:** choosing "Decline: Pay 3 energy" spends no energy and applies no counters.
- **Unavailable with <3 energy:** with only 1 energy the ImmediateTrigger resolves and offers no
  pay decision; no effect.
- **Regression (--scripted full games, seeds 1-3):** Guide mirror decks (4× Guide + creatures +
  Plains) — all three decisive (no draws), no non-fatal errors/warnings; Guide triggers/energy
  exercised heavily.

## Result
implemented
