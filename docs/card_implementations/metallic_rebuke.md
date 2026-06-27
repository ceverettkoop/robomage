# Metallic Rebuke (vocab index 151)

## Oracle text
Improvise (Your artifacts can help cast this spell. Each artifact you tap after you're done activating mana abilities pays for {1}.)
Counter target spell unless its controller pays {3}.

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/m/metallic_rebuke.txt`)

Key tags:
- `ManaCost:2 U`
- `K:Improvise`
- `A:SP$ Counter | TargetType$ Spell | ValidTgts$ Card | UnlessCost$ 3` — counter target spell unless its controller pays {3}.

## Engine work
None — covered. Existing handlers:
- `SP$ Counter` with `UnlessCost$` → counter-unless-pay (effect_counter / PAY_UNLESS), proven by Daze.
- `K:Improvise` parsed in `src/parse.cpp` (`has_improvise`) and applied in `src/mana_system.cpp` (tap untapped artifacts to pay {1} each, CR 702.126), proven by Kappa Cannoneer.

## Behavioral decisions
None — both mechanics fully covered by pre-existing cards.

## Tests
Skipped (VERIFY-SKIP): counter-unless-pay-3 proven by Daze; Improvise proven by Kappa Cannoneer. Clean `make HEADLESS=TRUE` build.

## Result
Done — registered in vocab, clean build.
