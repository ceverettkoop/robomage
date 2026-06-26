# Buried Alive  (vocab index 126)

## Oracle text
Search your library for up to three creature cards, put them into your graveyard, then shuffle.

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/b/buried_alive.txt`
- Key tags:
  - `A:SP$ ChangeZone | Origin$ Library | Destination$ Graveyard | ChangeType$ Creature | ChangeNum$ 3`
  - `ManaCost:2 B`, `Types:Sorcery`

## Engine work
- None — fully covered by the existing search-based `ChangeZone` handler in
  `src/effects/effect_change_zone.cpp`.
  - `ChangeNum$ 3` is parsed by `apply_param_to_ability` (`src/parse.cpp`) into `ab.amount = 3`.
  - The non-targeted, search branch of `change_zone()` loops `num_to_move = ab.amount` times,
    calling `search_zone()` against `Origin$ Library` each iteration, then moving the chosen
    card to `Destination$ Graveyard`.
  - "Up to three": `search_zone()` offers a **Fail to find** choice (a non-mandatory search),
    and `change_zone()` `break`s out of the loop the moment the player fails to find. So the
    player may take 0, 1, 2, or 3 creatures.
  - The library is shuffled (`orderer->shuffle_library`) after each pick because the origin is
    the library; the resolved end-state (chosen creatures in graveyard, library shuffled) is
    correct.
  - Destination `Graveyard` is a public zone, so the moved cards are logged publicly
    ("puts X to graveyard") — no reveal bookkeeping needed.
- Mechanics added (general, not card-specific): none.

## Behavioral decisions (made in lieu of asking a human)
- "Up to three" is an optional/variable count, not a forced three. CR 701.18b: a search is
  performed by its controller, and "up to N" permits finding fewer (including zero). The engine
  implements this via the per-pick Fail-to-find option; verified in isolation (1-creature case
  and explicit fail). Unambiguous otherwise.
- Search of one's own library shuffles afterward (CR 701.18g / the card's "then shuffle");
  the existing handler shuffles per library origin. No decision needed.

## Tests
- Isolation (test_harness, `--no-shuffle`, battlefield-preset 3 Swamps):
  - Cast Buried Alive, search three Grizzly Bears → all three moved to graveyard, library
    shuffled, Buried Alive (sorcery) resolved to graveyard. Final Self GY contained
    `Buried Alive` + 3× `Grizzly Bears`. PASS.
  - Library with only one creature: searched 1 Grizzly Bears, then **Fail to find** offered and
    taken → graveyard = Buried Alive + 1 Grizzly Bears (fewer-than-three / fail-to-find path).
    PASS.
- Regression (scripted full games, 6 seeds, Buried Alive ×4 in a mono-black deck vs a
  Bolt/Bears deck): all 6 games produced a decisive winner (no draws); Buried Alive was cast
  in real games and milled creatures with no non-fatal errors. Only pre-existing cosmetic
  `WARNING: Unrecognized ability param: Mode$ RevealYouChoose` lines (from Duress/Thoughtseize,
  unrelated) appeared. PASS.

## Result
implemented
