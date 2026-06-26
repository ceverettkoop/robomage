# Arid Mesa  (vocab index 102)

## Oracle text
{T}, Pay 1 life, Sacrifice Arid Mesa: Search your library for a Mountain or Plains card, put it onto the battlefield, then shuffle.

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/a/arid_mesa.txt`
- Key tags:
  - `A:AB$ ChangeZone | Cost$ T PayLife<1> Sac<1/CARDNAME> | Origin$ Library | Destination$ Battlefield | ChangeType$ Mountain,Plains`
  - It is a Zendikar-style fetchland: tap + pay 1 life + sacrifice itself, then a `ChangeZone`
    library search for a `Mountain` or `Plains` card to the battlefield, followed by a shuffle.

## Engine work
- None — fully covered by existing handlers. Arid Mesa is mechanically identical to the
  already-implemented Mountain/Plains-color-pair fetchlands and shares the exact `ChangeZone`
  activated-ability shape used by the other allied/enemy fetchlands already in the vocab
  (Windswept Heath, Wooded Foothills, Scalding Tarn, Flooded Strand, Polluted Delta,
  Misty Rainforest, Bloodstained Mire, Verdant Catacombs).
  - The activation cost (`T` tap + `PayLife<1>` + `Sac<1/CARDNAME>`) is parsed by `src/parse.cpp`'s
    `Cost$` handling.
  - The `ChangeZone` category (Origin Library → Destination Battlefield, with a `ChangeType$`
    subtype filter) is resolved by the existing `ChangeZone` effect handler, which prompts the
    controller to search the library, moves the chosen card, then shuffles.
- Mechanics added (general, not card-specific): none.

## Behavioral decisions (made in lieu of asking a human)
- The search is non-mandatory (a player may **fail to find**): consistent with CR 701.18 (search)
  and the fact that "Search your library for a Mountain or Plains card" does not say "if you do".
  The engine offers a "Fail to find" choice (verified — see Tests). No new decision was made here;
  it matches existing fetchland behavior.
- The fetched land enters the battlefield (it is "put onto the battlefield", not "into play
  tapped"), so it enters untapped — handled by the shared `ChangeZone` → Battlefield path.
- Cost is paid on activation (1 life, sacrifice) before the ability resolves (CR 601/602 activated
  abilities); the search/shuffle happens on resolution. Verified in the isolation test (life paid
  and Arid Mesa sacrificed at activation, land placed + shuffle on resolution).
- None of this is card-specific ambiguity — behavior is unambiguous and identical to existing
  fetchlands.

## Tests
- Isolation (test_harness, stacked deck `temp/arid_a` with library top = Mountain then Plains):
  - `activate:Arid Mesa, search:Mountain` → "Player A pays 1 life" (20→19), "Player A sacrifices
    Arid Mesa", ability resolves, search menu offers exactly {Mountain, Plains}, Mountain put onto
    Player A's battlefield, "Player A shuffles their library". Arid Mesa in graveyard. PASS.
  - `activate:Arid Mesa, search:Plains` → same flow, Plains put onto battlefield, shuffle. PASS.
  - Fail-to-find path is offered when no valid land remains in library (search menu shows
    "Fail to find" as choice 0). PASS.
- Regression (test_harness `--scripted`, 6 games, seeds 1–6): deck `temp/mav_arid` (the `mav`
  Maverick deck with 4 Wooded Foothills swapped to 4 Arid Mesa, retaining its Plains targets)
  vs `delver`. All 6 games finished with a decisive winner (A wins seeds 1,2,6; B wins seeds
  3,4,5), no draws. Arid Mesa was cast/activated by the scripted agent and seen in play and
  graveyards. No non-fatal errors / asserts / tracebacks. Only pre-existing cosmetic
  `WARNING: Unrecognized ability param` lines on unrelated cards (Green Sun's Zenith, Once Upon a
  Time, Delver of Secrets, Mishra's Bauble, Cori-Steel Cutter, Brainstorm).

## Result
implemented
