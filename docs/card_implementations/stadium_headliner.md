# Stadium Headliner  (vocab index 142)

## Oracle text
Mobilize 1 (Whenever this creature attacks, create a tapped and attacking 1/1 red
Warrior creature token. Sacrifice it at the beginning of the next end step.)
{1}{R}, Sacrifice this creature: It deals damage equal to the number of creatures
you control to target creature.

## Forge script  (Source: pre-existing local; Key tags)
- `K:Mobilize:1` — Mobilize keyword (proven by Voice of Victory / Cori-Steel Cutter).
- `A:AB$ DealDamage | Cost$ 1 R Sac<1/CARDNAME/this creature> | ValidTgts$ Creature |
  NumDmg$ X` with `SVar:X:Count$Valid Creature.YouCtrl`.

## Engine work  (none — covered by existing handlers)
- **Mobilize**: `src/effects/effect_mobilize.cpp` — creates the tapped+attacking 1/1 red
  Warrior token and registers the end-step sacrifice delayed trigger.
- **Sacrifice-self activation cost**: `Sac<1/CARDNAME/...>` parses to `ability.sac_self`
  (`src/parse.cpp:188-208`).
- **DealDamage** with dynamic amount: `src/effects/effect_deal_damage.cpp`; the
  `Count$Valid Creature.YouCtrl` amount is evaluated at resolution by
  `src/components/ability.cpp` (line 779) — number of creatures the controller has.
- `ValidTgts$ Creature` target legality via the standard creature-target path.

## Behavioral decisions  (none / CR timing)
- Damage amount = creatures you control, counted **at resolution** (after the
  sacrifice-self cost is paid), so the sacrificed Headliner is no longer counted
  (CR 608.2 — characteristic-defining values are determined as the ability resolves).
  This matches the test result (one of two Headliners sacrificed → 1 remaining → 1 damage).
- The activated ability is "target creature" (no `.Other`), so it may legally target
  any creature including another copy of Stadium Headliner.

## Tests
- Activated ability: two Headliners on battlefield, activated one (paid {1}{R} + sacrificed
  itself), dealt 1 damage to opponent's Murktide Regent (1 creature remained after sac). Pass.
- Mobilize: attacked with Headliner → "Mobilize 1: Player A creates 1 tapped and attacking
  1/1 Warrior(s)", token dealt 1 combat damage, then "Mobilize: Warrior Token is sacrificed"
  at the next end step. Pass.

## Result
Done. Covered card; both Mobilize and the sacrifice-self damage ability work with no engine changes.
