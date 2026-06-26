# Chalice of the Void  (vocab index 127)

## Oracle text
Chalice of the Void enters with X charge counters on it.

Whenever a player casts a spell with mana value equal to the number of charge counters on
Chalice of the Void, counter that spell.

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/c/chalice_of_the_void.txt`
- Type: `Artifact`, mana cost `X X` (so X charge counters cost {2X}).
- Key tags:
  - `K:etbCounter:CHARGE:X` with `SVar:X:Count$xPaid` — "enters with X charge counters", the
    count equal to the X value paid at cast time (CR 601.2b X-spells / 614.1c enters-with).
  - `T:Mode$ SpellCast | ValidCard$ Card.cmcEQY | ValidActivatingPlayer$ Player |
    TriggerZones$ Battlefield | Execute$ TrigCounter` — "whenever a player (any player) casts a
    spell whose mana value equals Y, do TrigCounter" (CR 603.2 cast trigger).
  - `SVar:Y:Count$CardCounters.CHARGE` — Y is this permanent's own charge-counter count.
  - `SVar:TrigCounter:DB$ Counter | Defined$ TriggeredSpellAbility` — counter the spell that
    fired the trigger (CR 701.5 counter).

No tags were retagged or repurposed; every mechanic below is keyed on the tag's intended meaning.

## Engine work
Reuses the existing etbCounter parsing, the `Count$CardCounters.<Type>` SVar (added for Aether
Vial), the X-cost casting path, the SPELL_CAST trigger event, and the `Counter` effect. The
gaps filled, each a general handler keyed on the tag's meaning:

1. **"Enters with X counters" where X = `Count$xPaid`** (CR 601.2b / 614.1c). The existing
   etbCounter replacement only handled `P1P1` counters whose count came from delve.
   - `src/parse.cpp`: `etbCounter:<TYPE>:<svar>` now also recognizes a count SVar resolving to
     `Count$xPaid` and records `counter_count_from_xpaid` (new `StaticAbility` field), keeping
     the declared counter type (`CHARGE`).
   - `src/components/spell.h` / `src/action_processor.cpp`: the X value chosen at cast is stored
     per-spell on the `Spell` component (`cur_game.x_paid` is global and can be overwritten by a
     later cast before this spell resolves).
   - `src/systems/stack_manager.cpp`: when an X-cost permanent resolves off the stack, its
     per-spell X is carried into `cur_game.pending_etb_xpaid[entity]` before the `Spell`
     component is removed.
   - `src/systems/replacement_effects.{h,cpp}`: the ETB-counters replacement now fires for any
     counter type (not just `P1P1`), with the count read from delve **or** from
     `pending_etb_xpaid`, and carries the counter type out on `ReplacementEvent::etb_counter_type`.
   - `src/systems/state_manager_statics.cpp`: a non-`P1P1` "enters with" counter is now applied
     to **any** permanent (artifacts included, CR 122.1) right after its `Permanent` component is
     created; `P1P1` counters still flow through the creature block so P/T can be logged.

2. **Cast trigger with a dynamic mana-value filter** (`ValidCard$ Card.cmcEQY`,
   Y = `Count$CardCounters.CHARGE`):
   - `src/parse.cpp`: a `cmcEQ<svar>` (or the other comparators) qualifier inside a `SpellCast`
     trigger's `ValidCard$` is detected, its SVar reference resolved to the runtime `Count$…`
     expression, and stored as `trigger_cmc_expr` + a two-letter comparator `trigger_cmc_op`
     (new `Ability` fields), mirroring the existing Aether-Vial `change_type_cmc_*` handling but
     on the trigger side. Such a `SpellCast` trigger is mapped to `Events::SPELL_CAST` (any
     spell, any player — `ValidActivatingPlayer$ Player` leaves `trigger_valid_player_is_controller`
     false). The fields are also carried through the `Execute$` SVar copy.
   - `src/action_processor.cpp` / `src/ecs/events.h`: the `SPELL_CAST` event now also carries
     `Params::ENTITY` = the cast spell, so the trigger can read the spell's mana value and target
     it.
   - `src/systems/state_manager_triggers.cpp`: at trigger-fire time, a `SPELL_CAST` trigger with
     `trigger_cmc_expr` compares the cast spell's mana value (`CardData::mana_cost.size()`)
     against `evaluate_sa_svar(expr, controller, source)` using `trigger_cmc_op`; non-matching
     spells are skipped.

3. **`Defined$ TriggeredSpellAbility`** (the Counter effect's target):
   - `src/parse.cpp` / `src/components/ability.h`: `Defined$ TriggeredSpellAbility` sets a new
     `defined_triggered_spell` flag.
   - `src/systems/state_manager_triggers.cpp`: when a trigger with that flag is queued, its
     `target` is set to the cast spell from the event. The `Counter` effect
     (`src/effects/effect_counter.cpp`) then counters that spell — it already verifies the target
     is still on the stack and honors "can't be countered" (CR 701.5).

## Behavioral decisions (made in lieu of asking a human)
- **MV equality is exact** (`cmcEQ`): a Chalice with N charge counters counters a spell only
  when its mana value equals N exactly; MV ≠ N spells resolve normally. Verified N=1 counters
  MV1 but not MV0/MV2, and N=0 counters MV0.
- **"a player" = any player** (`ValidActivatingPlayer$ Player`): the trigger fires on spells cast
  by either player, including Chalice's controller. Verified against an opponent's spell.
- **The CHARGE counter is a generic marker on the artifact** (CR 122.1) — it does not use the
  +1/+1 P/T resync path; it just accrues in the permanent's counter map.
- **X = 0 is legal and meaningful**: Chalice cast for X=0 enters with 0 counters and counters
  mana-value-0 spells (Lotus Petal, Mishra's Bauble) — a real competitive use ("Chalice for 0").
- **Mana value used is the spell's printed/paid mana value** (`CardData::mana_cost.size()`),
  consistent with how the rest of the engine reads MV.

## Tests
Isolation (`train/test_harness.py`), Chalice cast live and resolved before the opponent's spell:
- **X=1 enters with 1 charge counter.** `cast:Chalice of the Void` then `X = 1` →
  `Chalice of the Void enters with 1 CHARGE counter(s).` PASS.
- **MV-1 spell countered.** Opponent casts Lightning Bolt (MV1) → `Chalice of the Void triggered`,
  `Resolving ability (category: Counter, amount: 0)`, `Lightning Bolt is countered`; Player A
  takes no damage. PASS.
- **MV-2 spell NOT countered.** Opponent casts Grizzly Bears (MV2) under an X=1 Chalice →
  `Grizzly Bears enters the battlefield` (no trigger). PASS — the negative case.
- **MV-0 spell NOT countered at X=1.** Opponent casts Lotus Petal (MV0) under an X=1 Chalice →
  `Lotus Petal enters the battlefield` (no trigger). PASS — proves equality (not ≤).
- **X=0 counters MV-0 spells.** Chalice cast for X=0 (no counters); opponent casts Lotus Petal
  (MV0) → `Chalice of the Void triggered`, `Lotus Petal is countered`. PASS.

Regression (`train/test_harness.py --scripted`, seeds 1–6): deck `temp/chalice_delver`
(stock delver with 4 Daze swapped for 4 Chalice of the Void) vs stock `delver`. All 6 games
finished decisively (A wins 2, B wins 4), no draws, no asserts/tracebacks. Chalice was drawn,
cast, and its cast trigger countered opponents' spells in real games with the engine stable. The
only error lines were the pre-existing, Chalice-unrelated `Could not open token script:
w_1_1_monk_prowess.txt` from Cori-Steel Cutter, which also appears in a stock delver-vs-delver
run. Temp decks cleaned up.

## Result
implemented
