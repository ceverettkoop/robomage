# Crop Rotation  (vocab index 312)
## Oracle text
As an additional cost to cast this spell, sacrifice a land.
Search your library for a land card, put that card onto the battlefield, then shuffle.
## Forge script
- Source: pre-existing local
- Key tags: `A:SP$ ChangeZone | Cost$ G Sac<1/Land> | Origin$ Library | Destination$ Battlefield | ChangeType$ Land | ChangeNum$ 1`
## Engine work
- none — fully covered by existing handlers (`ChangeZone` library-search effect + `Sac<>` additional-cost handling, as used by the fetch lands)
- Mechanics added: none
## Behavioral decisions
- none — behavior unambiguous
## Tests
- Isolation (test_harness): battlefield 2 Forests, hand Crop Rotation, Rishadan Port deep in library. Cast Crop Rotation → prompted to sacrifice a land (sacrificed Forest), mana paid, searched library, put Rishadan Port onto the battlefield. Confirmed one land sacrificed + one land fetched to battlefield.
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
