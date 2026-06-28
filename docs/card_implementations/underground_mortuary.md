# Underground Mortuary  (vocab index 239)
## Oracle text
({T}: Add {B} or {G}.)
Underground Mortuary enters tapped.
When Underground Mortuary enters, surveil 1. (Look at the top card of your library. You may put it into your graveyard.)
## Forge script
- Source: pre-existing local
- Key tags:
  - `Types:Land Swamp Forest` (dual-subtype land; B/G mana injected from subtypes)
  - `R:Event$ Moved | ValidCard$ Card.Self | Destination$ Battlefield | ReplaceWith$ ETBTapped` + `SVar:ETBTapped:DB$ Tap | Defined$ Self | ETB$ True`
  - `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigSurveil` + `SVar:TrigSurveil:DB$ Surveil | Amount$ 1`
## Engine work
- none — fully covered by existing handlers
- Mechanics:
  - ETB-tapped replacement effect (`R:Event$ Moved ... ReplaceWith$ ETBTapped`): `src/systems/replacement_effects.cpp`
  - ETB Surveil 1 trigger: ChangesZone trigger → `effects/effect_surveil.cpp`
  - Dual-subtype land mana (Swamp + Forest): `StateManager::apply_land_abilities`
## Behavioral decisions (made in lieu of asking a human)
- none — behavior unambiguous (covered card)
## Tests
- Isolation: skipped — mechanics already proven by Undercity Sewers (ETB-tapped replacement + ETB Surveil 1 + dual-subtype land)
- Regression: skipped (verify_skip)
## Result
implemented (verification skipped — proven by Undercity Sewers)
