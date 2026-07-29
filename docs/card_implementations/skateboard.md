# Skateboard  (vocab index 314)
## Oracle text
When this Equipment enters, tap target permanent.
Equipped creature gets +1/+0 and has haste.
Equip {1} ({1}: Attach to target creature you control. Equip only as a sorcery.)
## Forge script
- Source: fetched Forge@pin
- Key tags: `K:Equip:1`; `T:Mode$ ChangesZone | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigTap` with `SVar:TrigTap:DB$ Tap | ValidTgts$ Permanent`; `S:Mode$ Continuous | Affected$ Creature.EquippedBy | AddKeyword$ Haste | AddPower$ 1`
## Engine work
- none — fully covered by existing handlers (Equipment `Equip` activated ability + attach, ETB `ChangesZone` trigger firing a `DB$ Tap`, continuous +1/+0 & keyword-grant static)
- Mechanics added: none
## Behavioral decisions
- none — behavior unambiguous
## Tests
- Isolation (test_harness): cast Skateboard, ETB tap targeted opponent's Island → "Island is tapped." Equipped to Grizzly Bears → became [3/2] with [Haste]; attacked and dealt 3 damage the same turn (haste from the equipment).
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
