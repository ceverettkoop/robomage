# Liquimetal Coating  (vocab index 253)

## Oracle text
{T}: Target permanent becomes an artifact in addition to its other types until end of turn.

(Artifact, mana cost {2}.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/l/liquimetal_coating.txt`
- Key tags:
  - `A:AB$ Animate | Cost$ T | ValidTgts$ Permanent | Types$ Artifact` — target permanent gains the
    Artifact type in addition to its others (default Animate duration = until end of turn).

## Engine work
- none — fully covered by existing handlers:
  - `AB$ Animate` with `Types$`: `src/effects/effect_animate.cpp` adds each parsed type to
    `Permanent::animate_added_types` and merges into the live type line (layer 4 reapply), proven by
    Guide of Souls. `Types$ Artifact` is parsed into `ability.animate_types` (`src/parse.cpp`).
  - `{T}` activation cost and `ValidTgts$ Permanent` targeting: standard.

## Behavioral decisions
- No `Duration$` param → the added type lasts until end of turn (the Animate default), matching the
  Forge script. (The paper card reads "until your next turn"; we follow the checked-in script.)

## Tests (test_harness)
- A controls Liquimetal Coating + a Forest. Activated Liquimetal targeting the Forest →
  "Forest becomes an Artifact in addition to its other types." PASS.
- Regression (`--scripted`, seeds 1-3): all decisive, no draws, no non-fatal errors.

## Result
implemented
