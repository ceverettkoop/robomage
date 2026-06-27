# Solitude  (vocab index 141)

## Oracle text
Flash
Lifelink
When Solitude enters, exile up to one other target creature. That creature's
controller gains life equal to its power.
Evoke—Exile a white card from your hand.

## Forge script  (Source: pre-existing local; Key tags)
- `K:Flash`, `K:Lifelink` — keyword statics (existing handlers).
- `K:Evoke:ExileFromHand<1/Card.White+Other/white card>` — evoke alternate cost
  (identical template to Endurance's `Card.Green+Other`).
- `T:Mode$ ChangesZone | ... Destination$ Battlefield | Execute$ TrigExile` — ETB trigger.
- `SVar:TrigExile:DB$ ChangeZone | Origin$ Battlefield | Destination$ Exile |
  ValidTgts$ Creature.Other | TargetMin$ 0 | TargetMax$ 1 | SubAbility$ DBGainLife`.
- `SVar:DBGainLife:DB$ GainLife | Defined$ TargetedController | LifeAmount$ X`,
  `SVar:X:Targeted$CardPower`.

## Engine work
One small, general correctness fix (not a new mechanic — all of Solitude's mechanics
already existed):

- **`.Other` target self-exclusion** in `src/components/ability.cpp` `Ability::is_legal_target`.
  Previously `ValidTgts$ ...Other` was not enforced at the target-legality level, so the
  source could illegally be chosen as its own target. Added a uniform guard: when
  `valid_tgts` contains `Other`, the ability's own `source` is not a legal target
  (CR 115.1 — "other" is a targeting restriction). This is general and also corrects
  Flickerwisp/Phelia-style "another/other target" cards.

Everything else is covered by existing handlers:
- Flash/Lifelink statics; Evoke alternate cost (`effect`/cost path proven by Endurance).
- ETB `ChangesZone` → `Battlefield` trigger registration (`src/parse.cpp`,
  `src/systems/state_manager_triggers.cpp`).
- `ChangeZone` Battlefield→Exile (`src/effects/effect_change_zone.cpp`).
- `GainLife` to `Defined$ TargetedController` with dynamic `LifeAmount$ X` =
  `Targeted$CardPower` (`src/effects/effect_gain_life.cpp`, `src/parse.cpp:886`,
  `src/components/ability.cpp`).

## Behavioral decisions
- "up to one other target creature": optional target (`TargetMin$ 0`) and the source
  excluded via the new `.Other` guard (CR 115.1). "No target" is offered.
- Lifegain goes to the *exiled* creature's controller (`TargetedController`), equal to
  its power at the time it was last on the battlefield (dynamic `Targeted$CardPower`).

## Tests
- Hard cast (3WW), exile opponent's Dragon's Rage Channeler (1/1): creature exiled,
  "Player B gains 1 life (now at 21)". Pass.
- Target menu after fix offers only "No target" and the opposing creature — Solitude
  itself is no longer a legal self-target. Pass.
- Evoke: cast via alternate cost, exiled Swords to Plowshares as the white card, both
  the ETB exile trigger and the evoke-sacrifice trigger went on the stack; ETB exiled
  the DRC + lifegain, then "Solitude is moved to graveyard" (evoke sacrifice). Pass.
- Scripted full game with Solitude in the deck: decisive result, no non-fatal errors,
  no draw. Pass.

## Result
Done. Covered card; one general `.Other` target-legality fix applied (benefits all
"other target" cards).
