# Prismatic Vista  (vocab index 258)

## Oracle text
{T}, Pay 1 life, Sacrifice Prismatic Vista: Search your library for a basic land card, put it onto
the battlefield, then shuffle.

(Land.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/p/prismatic_vista.txt`
- Key tags:
  - `A:AB$ ChangeZone | Cost$ T PayLife<1> Sac<1/CARDNAME> | Origin$ Library |
    Destination$ Battlefield | ChangeType$ Land.Basic` — fetch a basic onto the battlefield.

## Engine work
- none — fully covered by existing handlers:
  - Search-based `AB$ ChangeZone` Library→Battlefield with `ChangeType$ Land.Basic`:
    `src/effects/effect_change_zone.cpp` (shuffles after the library search).
  - `PayLife<1>`, `Sac<1/CARDNAME>`, and `{T}` activation costs: `src/parse.cpp` /
    `src/action_processor.cpp`.
- `ChangeTypeDesc$` is a cosmetic prose param, added to the parser's ignored set (`src/parse.cpp`).

## Behavioral decisions
- None novel — standard fetchland (the basic enters **untapped**: no `Tapped$` param).

## Tests (test_harness)
- A controls Prismatic Vista, library has basics. Activated → "Player A pays 1 life" (20→19),
  "Player A sacrifices Prismatic Vista", searched and put a Mountain onto the battlefield, "Player A
  shuffles their library". PASS.
- Regression (`--scripted`, seeds 1-3): all decisive, no draws, no non-fatal errors.

## Result
implemented
