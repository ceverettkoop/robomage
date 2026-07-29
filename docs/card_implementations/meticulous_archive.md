# Meticulous Archive  (vocab index 306)
## Oracle text
({T}: Add {W} or {U}.)
Meticulous Archive enters tapped.
When Meticulous Archive enters, surveil 1. (Look at the top card of your library. You may put it into your graveyard.)
## Forge script
- Source: pre-existing local
- Key tags: `Types:Land Plains Island`, `R:Event$ Moved ... ReplaceWith$ ETBTapped` (`SVar:ETBTapped:DB$ Tap | Defined$ Self | ETB$ True`), `T:Mode$ ChangesZone | Destination$ Battlefield | Execute$ TrigSurveil` (`SVar:TrigSurveil:DB$ Surveil | Amount$ 1`)
## Engine work
- none — fully covered by existing handlers (proven pattern: Undercity Sewers idx63, a Plains Island dual). Mana abilities for W/U are injected by `apply_land_abilities` from the Plains/Island subtypes; the enters-tapped replacement and the ETB surveil-1 trigger are already handled.
- Mechanics added: none
## Behavioral decisions
- none — behavior unambiguous.
## Tests
- Isolation (test_harness): play Meticulous Archive → "enters tapped" + ETB "surveils 1" with top/graveyard choice, shown `Meticulous Archive (T)`. Casting a {U} spell (Tune the Narrative) paid by the Archive logs "activated Meticulous Archive for 1(U)", confirming its U mana ability (W symmetric via the Plains subtype).
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
