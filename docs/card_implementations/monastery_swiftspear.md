# Monastery Swiftspear  (vocab index 351)

## Oracle text
Haste
Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)

## Forge script
- Source: pre-existing local (`bin/resources/cardsfolder/m/monastery_swiftspear.txt`)
- Key tags: `PT:1/2`; `K:Haste`; `K:Prowess`

## Engine work
- none — fully covered by existing handlers. Haste is a standard keyword; Prowess is parsed at
  `src/parse.cpp:804-806` and applied via `apply_keyword_abilities` → the `ProwessBonus` effect
  (`src/effects/effect_prowess_bonus.cpp`, dispatched through `effect_kind.cpp`/`effect_table.cpp`),
  the same path used by the engine's other prowess sources (e.g. the Cori-Steel Cutter Monk token).
- Mechanics added: none

## Behavioral decisions
- none — behavior unambiguous.

## Tests
- Isolation (test_harness): Monastery Swiftspear + Mountain preset on A; A casts Lightning Bolt
  (a noncreature spell) → engine logs "Resolving ability (category: ProwessBonus) / Prowess:
  creature gets +1/+1 until end of turn." and the board updates Monastery Swiftspear [1/2] → [2/3].
  Haste shown on the permanent from the start.
- CI gate (`ci_check.py --tier pygen,vocab,smoke`): 0 errors, no draws.

## Result
implemented
