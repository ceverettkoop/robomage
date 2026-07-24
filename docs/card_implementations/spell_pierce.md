# Spell Pierce  (vocab index 307)
## Oracle text
Counter target noncreature spell unless its controller pays {2}.
## Forge script
- Source: pre-existing local
- Key tags: `A:SP$ Counter | TargetType$ Spell | ValidTgts$ Card.nonCreature | UnlessCost$ 2`
## Engine work
- none — fully covered by existing handlers. The `SP$ Counter` category with a `Card.nonCreature` spell target and the `UnlessCost$` "pay N or be countered" branch are already handled.
- Mechanics added: none
## Behavioral decisions
- none — behavior unambiguous.
## Tests
- Isolation (test_harness): opponent casts a noncreature instant (Tune the Narrative); Player A responds with Spell Pierce targeting it → on resolution "Tune the Narrative's controller may pay {2} to save it" with a "Don't pay (spell is countered)" option → opponent (no mana) declines → "Tune the Narrative is countered".
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
