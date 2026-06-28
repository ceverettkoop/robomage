# Planar Nexus (vocab index 266)

## Oracle text
Planar Nexus is every nonbasic land type.
{T}: Add {C}.
{1}, {T}: Add one mana of any color.

(Land)

## Forge script
Source: pre-existing local script at `bin/resources/cardsfolder/p/planar_nexus.txt`.

```
S:Mode$ Continuous | Affected$ Card.Self | CharacteristicDefining$ True | AddType$ AllNonBasicLandType | Description$ CARDNAME is every nonbasic land type.
A:AB$ Mana | Cost$ T | Produced$ C
A:AB$ Mana | Cost$ 1 T | Produced$ Any
```

## Engine work
The two mana abilities use existing handlers (`Produced$ C`; `Produced$ Any` with a `{1}{T}`
activation cost). The type static needed one general addition in
`src/systems/state_manager_statics.cpp` (`apply_type_changing_effects`, layer 4):

- A self-CDA `AddType$ AllNonBasicLandType` (`CharacteristicDefining$ True`, `Affected$
  Card.Self`) now inserts the full set of nonbasic land subtypes (Desert, Gate, Lair, Locus,
  Mine, Power-Plant, Sphere, Tower, Urza's, Cave) onto the source permanent. Reasserted every
  SBA pass (idempotent set). `AllNonBasicLandType` is also excluded from the `Land.nonBasic`
  type-changer collection (it is a self-CDA, not an affector of other lands). General for any
  AllNonBasicLandType source.

The pre-existing `Land.nonBasic` type-setter (Blood Moon / Magus of the Moon) runs after this
self-CDA pass, so it still wins: under Magus, Planar Nexus is reset to a Mountain. No retagging;
no card-script edits.

## Behavioral decisions (CR cites)
- "Is every nonbasic land type" (CR 305.6 / characteristic-defining ability): Planar Nexus has all
  nonbasic land subtypes in every zone. Functionally this means it *is* an Urza's land, a Locus,
  etc., so type-matters effects (e.g. Urza's Workshop's "for each Urza's land you control")
  count it.
- Blood Moon-type effects (CR 305.7) override: a static that sets a nonbasic land's subtype to a
  basic type strips Planar Nexus's abilities and makes it that basic land (the Land.nonBasic pass
  runs last and resets subtypes).

## Tests (`train/test_harness.py`)
- **Mana**: with two Planar Nexus in play, Lightning Bolt `{R}` is cast — one Nexus taps for
  `{C}` to pay the `{1}` of the other's "{1},{T}: Add any color", which produces `R`
  ("activated Planar Nexus for 1(R)"); the bear is destroyed.
- **AllNonBasicLandType is functional**: Urza's Workshop + Planar Nexus + 3 artifacts → casting a
  {3} spell forces Urza's Workshop's Metalcraft ability, which produces **2** ("activated Urza's
  Workshop for 2(C)") = the count of Urza's lands (Urza's Workshop **plus Planar Nexus**),
  proving Planar Nexus gained the "Urza's" land type.
- **Blood Moon override**: with Magus of the Moon out, Planar Nexus taps for a single `R`
  (a Mountain), confirming the type-setter still wins over the CDA.

Regression: scripted full games, `temp/nexus_a` (Planar Nexus + Lightning Bolt + Grizzly Bears +
Forest/Mountain) vs `temp/toxi_a`, seeds 1 and 5 — decisive (A / B win), Planar Nexus produces
{C} and any color, no draws, no non-fatal errors.

## Result
Implemented. Both mana abilities work; `AllNonBasicLandType` self-CDA adds the nonbasic land
subtypes (verified functionally via Urza's Workshop's count) and is correctly overridden by
Blood Moon-type effects. Build clean; regression decisive with no errors.
