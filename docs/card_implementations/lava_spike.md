# Lava Spike  (vocab index 310)
## Oracle text
Lava Spike deals 3 damage to target player or planeswalker.
## Forge script
- Source: pre-existing local
- Key tags: `A:SP$ DealDamage | ValidTgts$ Player,Planeswalker | NumDmg$ 3`
## Engine work
- none — fully covered by existing handlers. The `SP$ DealDamage` category with a `Player,Planeswalker` target and a fixed `NumDmg$` is already handled.
- Mechanics added: none
## Behavioral decisions
- none — behavior unambiguous.
## Tests
- Isolation (test_harness): cast Lava Spike targeting Player B → "Dealt 3 damage to player (now at 17 life)"; opponent 20→17.
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
