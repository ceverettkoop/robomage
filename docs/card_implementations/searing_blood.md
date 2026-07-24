# Searing Blood (vocab index 327)

## Oracle text
Searing Blood deals 2 damage to target creature. When that creature dies this turn, Searing
Blood deals 3 damage to the creature's controller.

## Forge script (Source: pre-existing local; key tags)
`bin/resources/cardsfolder/s/searing_blood.txt`
- `Types:Instant`, `ManaCost:R R`
- `A:SP$ DealDamage | ValidTgts$ Creature | NumDmg$ 2 | SubAbility$ DBDelayedTrigger` — the
  2-damage bolt, with a chained delayed-trigger sub-ability.
- `SVar:DBDelayedTrigger:DB$ DelayedTrigger | Mode$ ChangesZone | RememberObjects$ Targeted |
  ValidCard$ Card.IsTriggerRemembered | Origin$ Battlefield | Destination$ Graveyard |
  ThisTurn$ True | Execute$ TrigDealDamage` — arm a "when that creature dies THIS turn" watcher.
- `SVar:TrigDealDamage:DB$ DealDamage | Defined$ TriggeredCardController | NumDmg$ 3` — the
  3 damage the death delivers, to the dead creature's controller.

## Engine work
Mechanics added (general): **changeszone-delayed-trigger** — a `Mode$ ChangesZone` delayed
trigger that watches a specific remembered object leaving one zone for another (CR 603.7b),
bounded to `ThisTurn$`, plus the `Defined$ TriggeredCardController` player reference.

- `src/effects/effect_delayed_trigger.cpp`
  - `parse_delayed_trigger`: now scoped to `DelayedTrigger` abilities; detects `Mode$ ChangesZone`
    (→ `DelayedTriggerParams::mode_changes_zone`), `ThisTurn$ True` (→ `this_turn`), and consumes
    the informational `ValidCard$ Card.IsTriggerRemembered`.
  - `delayed_trigger` handler: new ChangesZone branch. The watched object is the parent spell's
    target — `RememberObjects$ Targeted` put it in `cur_game.remembered_entities` (via
    `Ability::remember_targeted`, filled at resolution before the handler runs). It registers a
    `fire_on_leave_battlefield` delayed trigger (reusing the earthbend infrastructure:
    `DelayedTrigger::watch_entity` / `fire_on_leave_battlefield` / `fire_dest_zones`) filtered to
    the `Destination$` zone (Graveyard), with `expires_end_of_turn = this_turn`.
- `src/classes/game.h` — `DelayedTrigger::expires_end_of_turn`.
- `src/classes/game.cpp` — cleanup drops every `expires_end_of_turn` delayed trigger (CR 603.7b:
  the "this turn" watch lapses if the object never left play; reached after the end step, so any
  death this turn already fired and removed it).
- `src/systems/state_manager_triggers.cpp` — at fire time, a leave-battlefield delayed trigger
  whose fire ability is `Defined$ TriggeredCardController` binds `triggered_player` from
  `last_known_controller(watch_entity)` (the card is in the graveyard by then, CR 608.2g).
- `src/components/ability.h` — `Ability::defined_triggered_card_controller` (shares
  `triggered_player` storage). `src/parse.cpp` parses `Defined$ TriggeredCardController`.
- `src/game_queries.cpp` — `resolve_defined_player` returns `triggered_player` for the new flag.
- `src/effects/effect_deal_damage.cpp` — routes `defined_triggered_card_controller` through the
  existing defined-player damage path.

The script's real tags are honoured (no retag): the `SP$ DealDamage` bolt and its `DB$
DelayedTrigger | Mode$ ChangesZone` sub-ability keep their categories; `Origin$`/`Destination$`
are parsed by the shared `parse_change_zone` and read back as the zone filter.

## Behavioral decisions (CR cites)
- CR 603.7b — a delayed triggered ability that sets up on resolution ("when that creature dies
  this turn…") triggers only during the specified duration; the watch expires at end of turn.
- CR 704.x — the creature's death (2 damage ≥ toughness) is a state-based action moving it to the
  graveyard AFTER Searing Blood finishes resolving, so the watch is armed first, then fires.
- CR 608.2g — "that creature's controller" is read from last-known information once it has left
  the battlefield.
- Two-player scope: `TriggeredCardController` resolves to the single controller of the dead card.

## Tests
Built `make` clean (only the pre-existing cosmetic parse.cpp `tolower`/`stream.get` warnings and
DFC-back zero-cost warnings). Verified with `train/test_harness.py` (inline hands/battlefield,
semantic `--play` with seat keys):
- (a) A casts Searing Blood at a 2/2 Grizzly Bears: 2 damage kills it, the delayed trigger fires,
  and 3 damage is dealt to its controller (Player B 20 → 17).
- (b) A casts Searing Blood at a 4/4 Air Elemental (survives with 2 damage marked): the trigger is
  registered but never fires, and no damage is dealt to the controller (B stays 20).
- (c) Expiry: A casts Searing Blood at a 3/3 Murktide Regent on turn 1 (survives); the creature is
  killed by Lightning Bolt on turn 3. The turn-1 delayed trigger has expired at end of turn 1 —
  Murktide's death fires **no** delayed trigger and deals **0** damage to its controller (B stays
  20, 0 "Dealt 3 damage to player" events).
- CI gate below.

## Result
Done. General changeszone-delayed-trigger infrastructure (watch a remembered object dying this
turn, `Defined$ TriggeredCardController`, and end-of-turn expiry) implemented and reused from the
existing leave-battlefield delayed-trigger machinery; Searing Blood deals 2, and on the creature's
same-turn death deals 3 to its controller, doing nothing if it survives the turn.
