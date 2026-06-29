# Meteor Sword  (vocab index 226)
## Oracle text
When this Equipment enters, destroy target permanent.
Equipped creature gets +3/+3.
Equip {3} ({3}: Attach to target creature you control. Equip only as a sorcery.)
## Forge script
- Source: pre-existing local
- Key tags:
  - `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigDestroy` (self-ETB trigger)
  - `SVar:TrigDestroy:DB$ Destroy | ValidTgts$ Permanent`
  - `S:Mode$ Continuous | Affected$ Creature.EquippedBy | AddPower$ 3 | AddToughness$ 3`
  - `K:Equip:3`
## Engine work
- none — fully covered by existing handlers
- Mechanics:
  - Self-ETB ChangesZone trigger → DB$ Destroy: parser ChangesZone trigger + Destroy effect (`effects/effect_destroy.cpp`)
  - EquippedBy +P/+T static: continuous AddPower/AddToughness static handling
  - Equip keyword: `K:Equip` parsing + attach logic (`effects/effect_attach.cpp`)
## Behavioral decisions (made in lieu of asking a human)
- none — behavior unambiguous (covered card)
## Tests
- Isolation: skipped — mechanics already proven by White Orchid Phantom (self-ETB ChangesZone→Destroy) and Cori-Steel Cutter (EquippedBy +P/+T static + Equip)
- Regression: skipped (verify_skip)
## Result
implemented (verification skipped — proven by White Orchid Phantom + Cori-Steel Cutter)
