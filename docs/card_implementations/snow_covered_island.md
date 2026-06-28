# Snow-Covered Island  (vocab index 235)
## Oracle text
({T}: Add {U}.)
## Forge script
- Source: pre-existing local
- Key tags:
  - `Types:Basic Snow Land Island` (mana ability injected from the Island subtype; `Snow` supertype recognized)
## Engine work
- none — fully covered by existing handlers
- Mechanics:
  - Basic land mana ability auto-injected from the Island subtype by `StateManager::apply_land_abilities` (`src/systems/state_manager_statics.cpp`)
  - `Snow` supertype recognized in `all_supertypes` (`src/constants.cpp`), so the type line parses with no non-fatal error
## Behavioral decisions (made in lieu of asking a human)
- none — behavior unambiguous (covered card)
## Tests
- Isolation: skipped — mechanics already proven by Island (basic land, mana injected from Island subtype; Snow supertype recognized)
- Regression: skipped (verify_skip)
## Result
implemented (verification skipped — proven by Island)
