# Bayou (vocab index 150)

## Oracle text
({T}: Add {B} or {G}.)

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/b/bayou.txt`).
Key tags:
- `Types:Land Swamp Forest` — a dual land with the basic land subtypes Swamp and Forest.
- No `A:`/`T:`/`S:` ability lines: the mana ability is intrinsic to the land subtypes.

## Engine work
None — covered. `StateManager::apply_land_abilities` injects the intrinsic mana ability for
each basic land subtype a land has (Swamp → {B}, Forest → {G}), so Bayou taps for {B} or
{G} without any scripted ability. This is the same mechanism used by every dual land already
in the vocab (Underground Sea = Island Swamp, Savannah = Plains Forest, Plateau = Mountain
Plains, etc.).

## Behavioral decisions
None.

## Tests
Skipped: dual land proven by Underground Sea (and Savannah/Plateau/Tundra/Volcanic Island)
— mana injected from land subtypes by `apply_land_abilities`, no script abilities. Clean
build is sufficient.

## Result
Implemented (registration only; mechanic pre-existing via subtype-derived land mana
abilities).
