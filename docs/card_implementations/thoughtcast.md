# Thoughtcast (vocab index 145)

## Oracle text
Affinity for artifacts (This spell costs {1} less to cast for each artifact you control.)
Draw two cards.

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/t/thoughtcast.txt`).
Key tags:
- `A:SP$ Draw | NumCards$ 2` — spell ability, draw two cards.
- `K:Affinity:Artifact` — affinity for artifacts cost reduction.

## Engine work
None — covered. Handlers:
- `SP$ Draw` → `EffectKind::Draw` (`src/effects/effect_draw.cpp`).
- `K:Affinity:Artifact` parsed in `src/parse.cpp` (sets `CardData::affinity_artifact`)
  and applied by `effective_base_cost` in `src/systems/state_manager_statics.cpp`
  (reduces the generic portion of the cost by {1} per artifact the caster controls,
  CR 702.41; never reduces colored pips and never below 0).

## Behavioral decisions
None new. Cost reduction follows CR 702.41 / 601.2f (reductions applied after additions,
only generic mana removed).

## Tests
- Scenario: Player A with 3 artifact lands (Seat of the Synod) on the battlefield casts
  Thoughtcast (base {4}{U} = 5 mana) tapping only 2 Islands for {U}{U}. The spell cast
  successfully and resolved drawing 2 cards → affinity reduced {4} to {1}, total cost
  {1}{U}, paid with 2 mana. Result: pass.

## Result
Implemented (registration only; mechanic pre-existing).
