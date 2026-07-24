# Tune the Narrative  (vocab index 308)
## Oracle text
Draw a card. You get {E}{E} (two energy counters).
## Forge script
- Source: pre-existing local
- Key tags: `A:SP$ Draw | Defined$ You | NumCards$ 1 | SubAbility$ DBEnergy`, `SVar:DBEnergy:DB$ PutCounter | Defined$ You | CounterType$ ENERGY | CounterNum$ 2`
## Engine work
- none — fully covered by existing handlers (proven pattern: Guide of Souls idx171 for energy). The `SP$ Draw` category and the chained `DB$ PutCounter` of `ENERGY` counters on a player are already handled.
- Mechanics added: none
## Behavioral decisions
- none — behavior unambiguous.
## Tests
- Isolation (test_harness): cast Tune the Narrative → "draws" 1 card, then "Player A gets 2 ENERGY counter(s) (now 2)"; board shows `energy 2`. Net hand unchanged (cast −1, draw +1), library 8→7.
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
