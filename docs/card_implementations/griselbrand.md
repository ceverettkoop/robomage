# Griselbrand  (vocab index 304)
## Oracle text
Flying, lifelink
Pay 7 life: Draw seven cards.
## Forge script
- Source: pre-existing local
- Key tags: `K:Flying`, `K:Lifelink`, `A:AB$ Draw | Cost$ PayLife<7> | NumCards$ 7`
## Engine work
- none — fully covered by existing handlers. Flying/Lifelink keywords, the `PayLife<N>` activation cost, and the `AB$ Draw` category (draw N cards) are all already handled.
- Mechanics added: none
## Behavioral decisions
- none — behavior unambiguous.
## Tests
- Isolation (test_harness): preset Griselbrand + Island on battlefield → activate "Pay 7 life: Draw seven cards" → life 20→13, ability on stack, resolved to draw 7 (hand 7→14, library 8→1). Displayed as `Griselbrand [7/7] [Flying, Lifelink]`.
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
