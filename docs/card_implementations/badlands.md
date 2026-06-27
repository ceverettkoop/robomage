# Badlands (vocab index 155)

## Oracle text
({T}: Add {B} or {R}.)

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/b/badlands.txt`)

Key tags:
- `Types:Land Swamp Mountain` — dual land with basic land subtypes; no script abilities.

## Engine work
None — covered. `StateManager::apply_land_abilities` injects the {T}: Add {B}/{R}
mana abilities from the Swamp/Mountain subtypes, proven by Underground Sea / Bayou.

## Behavioral decisions
None.

## Tests
Skipped (VERIFY-SKIP): subtype-driven dual-land mana proven by Underground Sea / Bayou.
Clean `make HEADLESS=TRUE` build.

## Result
Done — registered in vocab, clean build.
