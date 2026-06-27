# Wight of the Reliquary (vocab index 158)

## Oracle text
Vigilance
Wight of the Reliquary gets +1/+1 for each creature card in your graveyard.
{T}, Sacrifice another creature: Search your library for a land card, put it onto
the battlefield tapped, then shuffle.

## Forge script
Source: pre-existing local script (`bin/resources/cardsfolder/w/wight_of_the_reliquary.txt`),
not retagged. Key tags:
- `K:Vigilance`
- `S:Mode$ Continuous | Affected$ Card.Self | AddPower$ X | AddToughness$ X` with
  `SVar:X:Count$ValidGraveyard Creature.YouOwn` — a characteristic-defining continuous
  self-buff (layer 7b, CR 613.4 / 604.3), power and toughness raised by the number of
  creature cards in your graveyard.
- `A:AB$ ChangeZone | Cost$ T Sac<1/Creature.Other/another creature> | Origin$ Library |
  Destination$ Battlefield | Tapped$ True | ChangeType$ Land | ChangeNum$ 1` — the
  activated tutor; the cost is a tap plus the sacrifice of **another creature**.

## Engine work
The continuous +1/+1-per-graveyard-creature buff and the library→battlefield-tapped land
search were already handled (Knight of the Reliquary precedent: the CDA static layer and
`ChangeZone` with `Tapped$ True`), and both verified below.

The single gap was the sacrifice-cost permanent filter. `Cost$ Sac<.../...>` is parsed by
`parse_activation_cost` (`src/parse.cpp`) into `Ability::sac_cost_spec` (here
`Creature.Other`). That spec is matched by `permanent_matches_subtype_spec` /
`controlled_permanents_matching` in `src/game_queries.h`, which previously only compared each
`';'`-delimited alternative against a permanent's type-name list. A top-level card type
("Creature") already matched (because `Permanent::types` stores both card types and subtypes
by name), but the `.Other` qualifier was treated as part of a literal subtype name
(`"Creature.Other"`) and never matched anything.

General, reusable change (not card-specific):
- `permanent_matches_subtype_spec(perm, spec, perm_entity = 0, exclude_entity = 0)` now splits
  each alternative on `'.'` into a type-name and an optional qualifier. The `.Other` qualifier
  means self-exclusion: the matching permanent must not be the cost's source. This works for
  any main type or subtype head (`Creature.Other`, `Land.Other`, etc.).
- `controlled_permanents_matching(player, spec, entities, exclude_entity = 0)` threads the
  source entity through so the `.Other` qualifier can exclude it.
- Call sites pass the ability's source: `src/action_processor.cpp` (payment) and
  `src/systems/state_manager_actions.cpp` (legality, both the battlefield-ability and
  hand-ability loops). Default args keep every other Sac/Return cost site (Goblin Bombardment,
  Cycling "Sac a land", etc.) unchanged.

## Behavioral decisions
- "Sacrifice another creature" = a creature you control other than the source. "Another"
  means a different object than the ability's source (CR 109.3 — controller/identity is not a
  characteristic, so a same-named creature still counts as "another"); implemented as
  entity-identity self-exclusion, not a name/type exclusion.
- The fetched land enters tapped (`Tapped$ True`), then the library is shuffled — standard
  `ChangeZone` library-search behavior.
- The buff counts only creature **cards** in **your** graveyard (`Count$ValidGraveyard
  Creature.YouOwn`); non-creature cards in the graveyard do not contribute (CR 613.4 CDA).

## Tests (`train/test_harness.py`, seed 1)
- **Self-exclusion + tutor:** Wight + Birds of Paradise on the battlefield; activating Wight's
  ability offered exactly one sacrifice — "Sacrifice Birds of Paradise" (Wight itself excluded).
  After sacrificing, the library search fetched a Mountain that "enters tapped", then the
  library was shuffled.
- **Continuous buff:** with 3 creature cards (Birds of Paradise) in the graveyard, Wight read
  `[5/5]` (2/2 base + 3). With 2 non-creature cards (Lightning Bolt) in the graveyard it read
  `[2/2]` — confirming only creature cards count.
- **Regression:** scripted full games on seeds 1, 2, 3 with a Wight deck (4 Wight, Birds,
  bolts, basics) vs a creature deck. All three ended decisively (A wins, B wins on deck-out, A
  wins) — no draws, no non-fatal errors, no assertions. Wight's P/T was observed scaling 2/2
  up to 6/6 as creatures died into the graveyard during the game.

## Result
Implemented. General main-type + `.Other` self-exclusion support added to the sacrifice-cost
permanent filter; Wight of the Reliquary registered at vocab index 158. Build clean
(`make HEADLESS=TRUE`); all tests pass.
