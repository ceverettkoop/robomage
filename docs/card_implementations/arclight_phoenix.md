# Arclight Phoenix  (vocab index 124)

## Oracle text
Flying, haste

At the beginning of combat on your turn, if you've cast three or more instant and sorcery
spells this turn, return Arclight Phoenix from your graveyard to the battlefield.

(Creature — Phoenix, {3}{R}, 3/2.)

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/a/arclight_phoenix.txt`
- Key tags:
  - `K:Flying` / `K:Haste` — already supported keyword abilities.
  - `T:Mode$ Phase | Phase$ BeginCombat | ValidPlayer$ You | TriggerZones$ Graveyard |
    CheckSVar$ X | SVarCompare$ GE3 | Execute$ TrigReturn` — the recursion trigger. It is a
    **Phase trigger** firing at the beginning of combat **on your turn** (`ValidPlayer$ You`),
    that **functions from the graveyard** (`TriggerZones$ Graveyard`, CR 113.6 / 603.6), gated by
    an **intervening-if** (`CheckSVar$ X | SVarCompare$ GE3`, CR 603.4).
  - `SVar:TrigReturn:DB$ ChangeZone | Defined$ Self | Origin$ Graveyard | Destination$ Battlefield`
    — the effect: return this card (`Defined$ Self`) from the graveyard to the battlefield.
  - `SVar:X:Count$ThisTurnCast_Instant.YouCtrl,Sorcery.YouCtrl` — the intervening-if count: the
    number of instant and sorcery spells **you** have cast this turn. Compared `GE3`.
  - Ignored: `DeckNeeds:Type$Instant|Sorcery` (a Forge deckbuilding hint; no gameplay effect).

## Engine work
All changes are general (keyed on the tag's intended meaning), not card-specific.

- **`Phase$ BeginCombat` trigger (`src/ecs/events.h`, `src/classes/game.cpp`, `src/parse.cpp`).**
  Added a new `Events::BEGIN_COMBAT_BEGAN` event, fired in the step machine when the active
  player's turn advances from `FIRST_MAIN` into `BEGIN_COMBAT` (mirroring how `UPKEEP_BEGAN` /
  `DRAW_STEP_BEGAN` / `END_STEP_BEGAN` are emitted on step entry), carrying `PLAYER` = the active
  player. The trigger parser maps `Mode$ Phase | Phase$ BeginCombat` to this event with
  `ValidPlayer$ You` honored as the controller filter — exactly as the other phase triggers.

- **`TriggerZones$ Graveyard` — graveyard-functioning triggers (`src/components/ability.h`,
  `src/parse.cpp`, `src/systems/state_manager_triggers.cpp`).** Added `Ability::trigger_from_graveyard`,
  set when a trigger line carries `TriggerZones$ Graveyard`. The main trigger scan only iterates
  battlefield permanents (`is_battlefield_permanent`), so it can never see a card in the graveyard.
  `check_triggered_abilities` now runs a second, focused scan over graveyard cards: for each
  card in a graveyard whose `CardData` has a triggered ability with `trigger_from_graveyard`, the
  matching event, the `ValidPlayer$ You` controller filter (against the source's owner), and the
  603.4 intervening-if are all evaluated, then the trigger is queued into the same APNAP placement
  path as battlefield triggers. This is the engine realization of CR 113.6 / 603.6 — an ability
  that functions while its source is in the graveyard.

- **Typed instant/sorcery cast counter (`src/components/player.h`, `src/action_processor.cpp`,
  `src/classes/game.cpp`).** Added `Player::instant_sorcery_spells_cast_this_turn`, incremented at
  cast time whenever the cast spell has type Instant or Sorcery (in the same block that already
  increments `spells_cast_this_turn` / `noncreature_spells_cast_this_turn`), and reset each turn
  alongside the other per-turn cast counters. The pre-existing untyped `spells_cast_this_turn`
  (used by Mindbreak Trap) could not distinguish instants/sorceries from other spells, so a typed
  counter was required for the "instant and sorcery spells" filter.

- **`Count$ThisTurnCast_Instant.YouCtrl,Sorcery.YouCtrl` condition reader
  (`src/systems/state_manager_actions.cpp::evaluate_present_condition`).** Added a branch that
  recognizes this Count expression and compares the controller's
  `instant_sorcery_spells_cast_this_turn` against the `SVarCompare$` operator (`GE3`). This sits
  next to the existing `Count$LifeYouGainedThisTurn` (Ocelot Pride) branch — both are non-board
  intervening-if counts read from the controller's `Player` component.

- **`ChangeZone Defined$ Self` Graveyard→Battlefield** needed no change: the existing
  `effects::change_zone` `defined_self` path already moves the ability's own source card to the
  destination and (for a battlefield destination) sets its controller to the owner. Returning from
  the graveyard is just that path with `Origin$ Graveyard`.

- **Test infrastructure: `--graveyard-a` / `--graveyard-b`** (`src/main.cpp`,
  `src/systems/orderer.{h,cpp}`, `train/env.py`, `train/runner.py`, `train/test_harness.py`).
  Added a `place_in_graveyard` helper (mirroring `place_on_battlefield`) and engine/harness flags
  to seed cards into a player's graveyard, so graveyard-functioning cards can be exercised in
  isolation. General-purpose test seeding, not card-specific.

## Behavioral decisions (made in lieu of asking a human)
- **The script is mandatory in tag form (no `Optional$`/`OptionalDecider$`), so the return is
  resolved mandatorily.** The Oracle wording says "you may return", but the fetched Forge script
  carries no optional tag, and per project policy the script governs and is never modified. In
  practice the difference is inconsequential here — returning a 3/2 flier with haste from the
  graveyard is essentially always taken — and tags are honored as written rather than retagged.
- **The intervening-if is checked when the trigger would go on the stack (603.4).** With fewer than
  three instant/sorcery casts this turn, the trigger does not fire at all (verified). The condition
  is re-checked at resolution by the shared `intervening_if` path.
- **Only instant and sorcery spells count.** A creature (or other non-instant/sorcery) spell does
  not increment the counter — verified that 2 bolts + 1 Grizzly Bears does not satisfy `GE3`.

## Tests
- Isolation (test_harness, Arclight Phoenix seeded into Player A's graveyard via `--graveyard-a`):
  - **Positive (3 instants → return):** A (3 Mountains in play, 3 Lightning Bolts in hand) casts
    all three Bolts in its first main, then at the beginning of combat "Arclight Phoenix triggered"
    / "Arclight Phoenix is moved to the battlefield"; it enters as a 3/2 and attacks the same turn
    (haste), dealing 3 damage. No fizzle / non-fatal error.
  - **Negative (2 instants → no trigger, intervening-if):** identical setup but A casts only two
    Bolts; Arclight Phoenix remains in the graveyard through begin-of-combat — the trigger never
    fires.
  - **Type filter (2 instants + 1 creature → no trigger):** A casts 2 Lightning Bolts and a Grizzly
    Bears (creature). The creature does not count toward the three, so Arclight stays in the
    graveyard — confirming the Instant/Sorcery type filter.
- Regression (test_harness --scripted, full games): UR deck with 4× Arclight Phoenix + Lightning
  Bolt / Ponder / Brainstorm + lands, mirror match, seeds 1-6 — all six decisive (4 A wins, 2 B
  wins), no draws, no max-decisions caps, no non-fatal errors or assertions. The only warning is
  the pre-existing cosmetic `Unrecognized ability param: Reorder$ True (card: Brainstorm)`. (The
  scripted agent rarely casts three instants/sorceries in a single turn, so the recursion path is
  not naturally hit in scripted full games; it is fully covered by the isolation tests above.)

## Result
implemented
