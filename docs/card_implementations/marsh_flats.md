# Marsh Flats  (vocab index 111)

## Oracle text
{T}, Pay 1 life, Sacrifice Marsh Flats: Search your library for a Plains or Swamp card, put it onto the battlefield, then shuffle.

## Forge script
- Source: fetched (Forge@master) → `bin/resources/cardsfolder/m/marsh_flats.txt`
- Key tags:
  - `A:AB$ ChangeZone | Cost$ T PayLife<1> Sac<1/CARDNAME> | Origin$ Library | Destination$ Battlefield | ChangeType$ Plains,Swamp`
- This is the standard Zendikar "fetchland" template, identical to the already-implemented
  Arid Mesa (index 102) except `ChangeType$ Plains,Swamp` instead of `Mountain,Plains`.

## Engine work
- None — fully covered by existing handlers. The activated `ChangeZone` path already
  implements: tap cost (`T`), `PayLife<1>`, `Sac<1/CARDNAME>` self-sacrifice as activation
  costs; a library search filtered by `ChangeType$` (offering only matching land cards plus a
  "Fail to find" option); putting the chosen card onto the battlefield; and shuffling the
  library afterward.
- Mechanics added (general, not card-specific): none.

## Behavioral decisions (made in lieu of asking a human)
- Search may legally fail to find even when a matching card is in the library (CR 701.18c — a
  player searching is not required to find a card unless the effect says "if you do"/uses a
  cost). The engine exposes a "Fail to find" choice, which is correct; verified in the
  fail-to-find isolation test (1 life still paid, card still sacrificed, library still
  shuffled, no land fetched).
- The fetched land enters untapped (no `| Tapped$ True` on the script), matching the Oracle
  text ("put it onto the battlefield").
- The search is by subtype `Plains`/`Swamp` (CR 205.3i land types), so nonbasic lands with
  those subtypes (e.g. Savannah, a Plains-typed dual) are valid targets in addition to the
  basics — behavior unambiguous and inherited from the shared `ChangeType$` matcher.

## Tests
- Isolation (test_harness, stacked temp deck, `--no-shuffle`):
  - Play Marsh Flats → activate → `search:Swamp`: search menu offered exactly
    {Fail to find, Plains, Swamp} (Islands excluded → `ChangeType$` filter correct); life
    20→19; Marsh Flats sacrificed to graveyard; "Player A shuffles their library"; "Player A
    puts Swamp to the battlefield"; Self battlefield shows Swamp. PASS.
  - Fail-to-find path (`search:fail`): 1 life paid (20→19), Marsh Flats sacrificed to
    graveyard, library shuffled, no land fetched. PASS.
- Regression (test_harness `--scripted`, 6 seeds, mav-derived deck with 4 Marsh Flats vs mav):
  all 6 games decisive (3 A-wins / 3 B-wins), no draws, no non-fatal/fatal errors (only the
  pre-existing cosmetic `WARNING: Unrecognized ability param` lines). Marsh Flats observed
  being sacrificed in-game (seed 2). PASS.

## Result
implemented
