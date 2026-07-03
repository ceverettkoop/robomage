# Prismari Charm  (vocab index 231)
## Oracle text
Choose one —
• Surveil 2, then draw a card.
• Prismari Charm deals 1 damage to each of one or two targets.
• Return target nonland permanent to its owner's hand.
## Forge script
- Source: pre-existing local
- Key tags:
  - `A:SP$ Charm | Choices$ DBSurveil,DBDealDamage,DBChangeZone`
  - `SVar:DBSurveil:DB$ Surveil | Amount$ 2 | SubAbility$ DBDraw`
  - `SVar:DBDraw:DB$ Draw`
  - `SVar:DBDealDamage:DB$ DealDamage | ValidTgts$ Any | TargetMin$ 1 | TargetMax$ 2 | NumDmg$ 1`
  - `SVar:DBChangeZone:DB$ ChangeZone | ValidTgts$ Permanent.nonLand | Origin$ Battlefield | Destination$ Hand`
## Engine work
- none — fully covered by existing handlers
- Mechanics:
  - `SP$ Charm` modal dispatch with comma-separated SVar choices: `effects/effect_charm.cpp`
  - Surveil + chained Draw via SubAbility: `effects/effect_surveil.cpp` + `effects/effect_draw.cpp`
  - DealDamage with TargetMin/TargetMax (1–2 targets): `effects/effect_deal_damage.cpp`
  - ChangeZone Battlefield→Hand bounce: `effects/effect_change_zone.cpp`
## Behavioral decisions (made in lieu of asking a human)
- none — behavior unambiguous (covered card)
## Tests
- Isolation: skipped — mechanics already proven by Lorehold Charm (SP$ Charm modal dispatch; Surveil+Draw / DealDamage 1-2 targets / ChangeZone bounce)
- Regression: skipped (verify_skip)
- 2026-07-02 (cast-time announcement fix): mode + 1–2 damage targets are chosen at CAST
  (CR 601.2b/c) and are public on the stack; verified the opponent can respond to the announced
  target (Bolt kills the targeted Bears in response → the damage mode fizzles at resolution,
  CR 608.2b). Each mode re-exercised via test_harness `--play`.
## Result
implemented (verification skipped — proven by Lorehold Charm)
