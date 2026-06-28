# Wastes  (vocab index 240)
## Oracle text
{T}: Add {C}.
## Forge script
- Source: pre-existing local
- Key tags:
  - `Types:Basic Land`
  - `A:AB$ Mana | Cost$ T | Produced$ C` (colorless mana ability)
## Engine work
- none — fully covered by existing handlers
- Mechanics:
  - Colorless mana ability `AB$ Mana Produced$ C`: `effects/effect_add_mana.cpp`
## Behavioral decisions (made in lieu of asking a human)
- none — behavior unambiguous (covered card)
## Tests
- Isolation: skipped — mechanics already proven by the basic lands (colorless AB$ Mana Produced$ C)
- Regression: skipped (verify_skip)
## Result
implemented (verification skipped — proven by basic lands)
