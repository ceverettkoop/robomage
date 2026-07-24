# Exploration  (vocab index 311)
## Oracle text
You may play an additional land on each of your turns.
## Forge script
- Source: pre-existing local
- Key tags: `S:Mode$ Continuous | Affected$ You | AdjustLandPlays$ 1`
## Engine work
- none — fully covered by existing handlers (the `rules_modifying` land_play_bonus / `AdjustLandPlays$` static land-play modifier)
- Mechanics added: none
## Behavioral decisions
- none — behavior unambiguous
## Tests
- Isolation (test_harness): Exploration preset in play, hand Forest+Island. Played Forest, then "Play Island" was still offered and played the same turn; final battlefield showed both Forest and Island in play.
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
