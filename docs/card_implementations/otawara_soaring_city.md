# Otawara, Soaring City  (vocab index 228)
## Oracle text
{T}: Add {U}.
Channel — {3}{U}, Discard Otawara, Soaring City: Return target artifact, creature, enchantment, or planeswalker to its owner's hand. This ability costs {1} less to activate for each legendary creature you control.
## Forge script
- Source: pre-existing local
- Key tags:
  - `A:AB$ Mana | Cost$ T | Produced$ U` (tap for blue)
  - `A:AB$ ChangeZone | PrecostDesc$ Channel — | Cost$ 3 U Discard<1/CARDNAME> | ActivationZone$ Hand | ValidTgts$ Artifact,Creature,Enchantment,Planeswalker | Origin$ Battlefield | Destination$ Hand | ReduceCost$ X`
  - `SVar:X:Count$Valid Creature.Legendary+YouCtrl`
## Engine work
- none — fully covered by existing handlers
- Mechanics:
  - Mana ability: `AB$ Mana Produced$ U` (basic mana ability handling)
  - Channel: `ActivationZone$ Hand` + `Discard<1/CARDNAME>` self-discard cost + `ReduceCost$ X` (`Count$Valid` cost reduction) — all parsed in `src/parse.cpp`
  - Targeted ChangeZone Battlefield→Hand bounce: `effects/effect_change_zone.cpp`
## Behavioral decisions (made in lieu of asking a human)
- none — behavior unambiguous (covered card)
## Tests
- Isolation: skipped — mechanics already proven by Eiganjo, Seat of the Empire (Channel: ActivationZone$ Hand, Discard self-cost, ReduceCost$ X) and generic targeted ChangeZone Battlefield→Hand bounce
- Regression: skipped (verify_skip)
## Result
implemented (verification skipped — proven by Eiganjo, Seat of the Empire)
