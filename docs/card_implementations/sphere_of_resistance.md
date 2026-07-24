# Sphere of Resistance  (vocab index 315)
## Oracle text
Spells cost {1} more to cast.
## Forge script
- Source: pre-existing local
- Key tags: `S:Mode$ RaiseCost | ValidCard$ Card | Type$ Spell | Amount$ 1`
## Engine work
- none — fully covered by existing handlers (`RaiseCost` static cost increase, as proven by Trinisphere idx270 / Damping Sphere idx277)
- Mechanics added: none
## Behavioral decisions
- none — behavior unambiguous
## Tests
- Isolation (test_harness): Sphere of Resistance + Mountains preset for A, hand Lightning Bolt (normally {R} = 1 mana). Casting Lightning Bolt tapped TWO Mountains ({1}{R}), confirming the +{1} cost increase applies.
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
