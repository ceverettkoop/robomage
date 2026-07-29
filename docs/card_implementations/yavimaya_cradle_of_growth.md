# Yavimaya, Cradle of Growth  (vocab index 318)
## Oracle text
Each land is a Forest in addition to its other land types.
## Forge script
- Source: pre-existing local
- Key tags: `S:Mode$ Continuous | Affected$ Land | AddType$ Forest`
## Engine work
- none — fully covered by existing handlers (continuous `AddType$` type-adding static in the layer system; the injected basic-land mana ability follows from the Forest subtype on the next SBA pass)
- Mechanics added: none
## Behavioral decisions
- none — behavior unambiguous
## Tests
- Isolation (test_harness): Yavimaya loaded via temp stacked deck (comma name). Preset two Islands (which normally tap only for {U}), played Yavimaya, cast Endurance ({1}{G}{G}). The engine tapped Yavimaya → {G} AND an Island → {G} ("Player A activated Island for 1(G)") to pay the second green pip, with the other Island paying the generic. This confirms every land (including basic Islands and opponent lands) gains a Forest's {G} mana ability on the next SBA pass.
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
