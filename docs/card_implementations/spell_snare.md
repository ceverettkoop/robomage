# Spell Snare  (vocab index 321)
## Oracle text
Counter target spell with mana value 2.
## Forge script
- Source: pre-existing local
- Key tags: `A:SP$ Counter | TargetType$ Spell | ValidTgts$ Card.cmcEQ2`
## Engine work
- **Fix — counter-target-cmc-filter** (`src/components/ability.cpp`, `is_legal_target` stack-object branch ~line 845): the counterspell target branch built its `MatchCtx` with the default (unset) cmc bound and never called `extract_static_cmc_bound`. The filter evaluator (`game_queries.cpp` `eval_qualifier`) returns `true` for a bare `cmcEQ2` token, deferring the actual comparison to `ctx.cmc_bound` (`eval_alternative`, `game_queries.cpp:389`). With the bound left at its `-1` sentinel the comparator is skipped, so Spell Snare could counter ANY spell. Fixed by constructing a local `MatchCtx` and calling `extract_static_cmc_bound(vt, spell_ctx)` before `card_matches_any`, seeding `cmc_op`/`cmc_bound` from the `ValidTgts$` spec.
- CR: 115.1 (target restrictions checked when the target is chosen and re-checked at resolution), 202.3 (mana value)
- Mechanics added (general): static numeric cmc target restriction on a counterspell (any `Card.cmcEQ/LE/GE/...N` counter target)
## Behavioral decisions
- none — behavior unambiguous. A spell whose only candidate targets fail the cmc restriction cannot be cast at all (CR 601.2c), so "Cast Spell Snare" is not offered when no MV2 spell is on the stack.
## Tests
- Isolation (test_harness), A holds Spell Snare:
  - B casts Grizzly Bears (MV2) → Spell Snare targets and counters it ("Grizzly Bears is countered").
  - B casts Cabal Ritual (MV2, `1 B`) → also legal target, countered.
  - B casts Lightning Bolt (MV1) → "Cast Spell Snare" not offered (no legal target); Bolt resolves.
  - B casts Endurance (MV3, `1 G G`) → "Cast Spell Snare" not offered; Endurance resolves uncountered.
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
