# Silent Clearing  (vocab index 234)
## Oracle text
{T}, Pay 1 life: Add {W} or {B}.
{1}, {T}, Sacrifice Silent Clearing: Draw a card.
## Forge script
- Source: pre-existing local
- Key tags:
  - `A:AB$ Mana | Cost$ T PayLife<1> | Produced$ Combo W B`
  - `A:AB$ Draw | Cost$ 1 T Sac<1/CARDNAME> | NumCards$ 1`
## Engine work
- none — fully covered by existing handlers
- Mechanics:
  - PayLife mana ability producing a choice of two colors (`Produced$ Combo W B`): `effects/effect_add_mana.cpp` (Combo color choice) + PayLife cost grammar
  - `{1}, {T}, Sacrifice: Draw`: Sac self-cost + tap + generic cost grammar + `effects/effect_draw.cpp`
## Behavioral decisions (made in lieu of asking a human)
- none — behavior unambiguous (covered card)
## Tests
- Isolation: skipped — mechanics already proven by Noble Hierarch (Produced$ Combo W B + PayLife mana ability) and the shared Sac/Draw cost grammar
- Regression: skipped (verify_skip)
## Result
implemented (verification skipped — proven by Noble Hierarch + shared Sac/Draw cost grammar)
