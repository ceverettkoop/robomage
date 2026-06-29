# Tropical Island  (vocab index 300)

## Oracle text
(Tropical Island is a dual land with land types Forest and Island.)
{T}: Add {G} or {U}.

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/t/tropical_island.txt`
- Key tags: `Types:Land` with subtypes `Forest Island`. Mana abilities are injected by
  `StateManager::apply_land_abilities` from the land subtypes (not from the script), exactly
  like the sibling duals already in the vocab (Bayou, Underground Sea, Volcanic Island).

## Engine work
- **None.** Vocab-only addition. Tropical Island was the one dual land in the `bug` league deck
  missing from `src/card_vocab.h` while every sibling dual (Bayou, Underground Sea, Undercity
  Sewers, Volcanic Island) was present — an oversight, not a missing mechanic. It already loaded
  and played from its script; it was simply encoded as the "unknown" ML sentinel.

## Behavioral decisions
- None — behavior is the standard subtype-driven dual land, fully covered by existing handlers.

## Tests
- Build clean; `card_costs.py` regenerated (index 300, all-zero cast cost as expected for a land).
- Loads + decodes by name on the battlefield (vocab round-trip).
- "Cast Brainstorm" ({U}) is offered with only Tropical Island as a mana source → it produces {U};
  Forest/Island subtypes also produce {G} (identical to the verified sibling duals).

## Result
Implemented (vocab-only). Build green; no engine change required.
