# Super Shredder (vocab index 148)

## Oracle text
Menace
Whenever another permanent leaves the battlefield, put a +1/+1 counter on Super Shredder.

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/s/super_shredder.txt`).
Key tags:
- `K:Menace`
- `T:Mode$ ChangesZone | ValidCard$ Permanent.Other | Origin$ Battlefield | Destination$ Any | Execute$ TrigPutCounter | TriggerZones$ Battlefield`
- `SVar:TrigPutCounter:DB$ PutCounter | Defined$ Self | CounterType$ P1P1 | CounterNum$ 1`

## Engine work
None new — covered.
- `K:Menace` — parsed/applied as an evasion keyword.
- The leaves-the-battlefield trigger `ChangesZone | ValidCard$ Permanent.Other |
  Origin$ Battlefield | Destination$ Any` is handled by the battlefield-permanent trigger
  scan in `src/systems/state_manager_triggers.cpp`: `.Other` sets `trigger_self_excluded`
  (skips the event when the changing card is the source itself, line ~135),
  `Origin$ Battlefield` sets `trigger_zone_origin`, `Destination$ Any` leaves
  `trigger_zone_destination` unset (matches any destination), and `Permanent` restricts to
  permanent card types (line ~216). This is the same mechanism Kappa Cannoneer uses for
  "another artifact entering."
- `DB$ PutCounter | Defined$ Self | CounterType$ P1P1` → `EffectKind::PutCounter`
  (`src/effects/effect_put_counter.cpp`) puts the +1/+1 counter on the source.

## Behavioral decisions
None new. Follows CR 603.6b / 603.10 (zone-change triggers); the +1/+1 counter is added
when another permanent leaves the battlefield for any zone.

## Tests
- Scenario: Player A controls Super Shredder [1/1], Goblin Bombardment, and Birds of
  Paradise. A activates Goblin Bombardment, sacrificing Birds of Paradise. Observed: "Super
  Shredder triggered" → "Resolving ability (category: PutCounter)" → "Put 1 +1/+1 counter(s)
  on creature (now 2/2)." and board shows "Super Shredder [2/2]". Result: pass — another
  permanent leaving the battlefield correctly puts a +1/+1 counter on Super Shredder.

## Result
Implemented (registration only; mechanic pre-existing via the battlefield-permanent
leaves-the-battlefield trigger scan + PutCounter).
