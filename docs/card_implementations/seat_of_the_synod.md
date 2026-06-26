# Seat of the Synod  (vocab index 140)

## Oracle text
(Seat of the Synod isn't a spell.)
{T}: Add {U}.

## Forge script  (Source: pre-existing local; Key tags)
- `Types:Artifact Land`
- `A:AB$ Mana | Cost$ T | Produced$ U` — single tap-for-blue mana ability.

## Engine work  (none — covered by existing handlers)
- Mana ability parsed by the `AB$ Mana` → `AddMana` path in `src/parse.cpp` and produced by
  `src/effects/effect_add_mana.cpp`. The `Cost$ T` tap cost and `Produced$ U` color are the
  standard mana-land template (same as any tapland that produces a fixed color). The
  Artifact + Land type line requires no special handling for the mana ability.

## Behavioral decisions  (none)
Plain fixed-color mana land. No targeting, no conditions.

## Tests
- Played Seat of the Synod from hand → it entered the battlefield as a land (no land-drop
  anomaly). Result: pass.
- With Seat on the battlefield, cast Ponder ({U}). Engine logged
  "Player A activated Seat of the Synod for 1(U)" and Ponder cast and resolved
  (RearrangeTopOfLibrary), confirming it taps for blue. Result: pass.

## Result
Done. Covered card — taps for {U} correctly; no engine changes needed.
