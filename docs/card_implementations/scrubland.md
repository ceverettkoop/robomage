# Scrubland (vocab index 153)

## Oracle text
({T}: Add {W} or {B}.)

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/s/scrubland.txt`)

Key tags:
- `Types:Land Plains Swamp` — dual land with basic land subtypes; no script abilities.

## Engine work
None — covered. `StateManager::apply_land_abilities` injects the {T}: Add {W}/{B}
mana abilities from the Plains/Swamp subtypes (no script needed), proven by the other
original dual lands (Underground Sea, Bayou).

## Behavioral decisions
None.

## Tests
Skipped (VERIFY-SKIP): subtype-driven dual-land mana proven by Underground Sea / Bayou.
Clean `make HEADLESS=TRUE` build.

## Result
Done — registered in vocab, clean build.
