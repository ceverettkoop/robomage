# Summon: Bahamut

**Vocab index:** 297

## Oracle
```
Summon: Bahamut    {9}
Enchantment Creature — Saga Dragon    9/9
(As this Saga enters and after your draw step, add a lore counter. Sacrifice after IV.)
I, II — Destroy up to one target nonland permanent.
III   — Draw two cards.
IV    — Mega Flare — This creature deals damage equal to the total mana value of other
        permanents you control to each opponent.
Flying
```

## Forge source
Pre-existing local script `bin/resources/cardsfolder/s/summon_bahamut.txt`. Key tags:
- `K:Chapter:4:DBDestroy,DBDestroy,DBDraw,DBDamage` — four chapter slots; chapters I and II both
  name `DBDestroy` (two independent destroy triggers at lore 1 and lore 2).
- `SVar:DBDestroy: DB$ Destroy | TargetMin$ 0 | TargetMax$ 1 | ValidTgts$ Permanent.nonLand` —
  up-to-one targeting.
- `SVar:DBDraw: DB$ Draw | Defined$ You | NumCards$ 2`.
- `SVar:DBDamage: DB$ DealDamage | Defined$ Player.Opponent | NumDmg$ X`,
  `SVar:X: Count$Valid Permanent.YouCtrl+Other$CardManaCost` (Mega Flare).
- `PT:9/9` + `K:Flying` — a 9/9 flyer independent of its chapter abilities (CR 714.1a).

## Engine work
Uses the **shared CR 714 Saga engine** (`src/saga.{h,cpp}`) introduced with Urza's Saga: chapter
parse, lore counters (enters with one + one per precombat main), `SAGA_CHAPTER` chapter-trigger
dispatch, and the 714.4 sacrifice SBA gated on `Permanent::saga_chapters_in_flight`. Bahamut is a
creature Saga, and per CR 714.4 (no creature exception) and its printed "Sacrifice after IV" it is
sacrificed by that same SBA — no creature special-casing.

Card-specific addition for **Mega Flare** (chapter IV):
- New dynamic-amount form `Count$Valid <filter>$CardManaCost` in `evaluate_dynamic_amount`
  (`src/components/ability.cpp`): the **sum** of mana values (not the count) of battlefield
  permanents matching the filter. The `+Other` qualifier excludes the ability's own source via the
  match context, so it sums "**other** permanents you control" (Bahamut's own MV 9 is excluded).
  Reuses `card_mana_value` (folds hybrid MV).
- `effect_deal_damage` (`src/effects/effect_deal_damage.cpp`) now threads `ab.source` into the
  dynamic-amount evaluation so the `+Other` self-exclusion resolves against Bahamut.

Chapters I–III use existing effects: `DB$ Destroy` with `TargetMin$ 0 / TargetMax$ 1` (up-to-one
targeting, already supported), `DB$ Draw` (`NumCards$ 2`), and `Defined$ Player.Opponent` damage
(each opponent — the single opponent in the two-player engine).

## Behavioral decisions
- "Up to one target" chapters offer a "No target" choice (choosing zero is legal) alongside each
  legal nonland permanent (including Bahamut itself).
- Mega Flare's source-relative sum excludes Bahamut; if Bahamut is the only permanent, it deals 0.

## Tests → results
- **Full arc** (pre-placed Bahamut, auto-advance): lore 1→2→3→4 with chapter I/II destroy, chapter
  III **draws two**, chapter IV Mega Flare, then **"Summon: Bahamut is sacrificed (final chapter
  completed)"**. ✓
- **Up-to-one targeting**: chapter I menu offered `No target` / opponent's artifact / Bahamut
  itself; selecting the artifact destroyed it; selecting "No target" did nothing. ✓
- **Mega Flare value**: with three MV-1 artifacts also out, Mega Flare dealt exactly **3** (sum of
  the three, **excluding** Bahamut's own MV 9); with no other permanents it dealt **0**. ✓
- **9/9 flyer**: Bahamut attacked and dealt 9 combat damage while its Saga lifecycle progressed. ✓
- **Regression**: scripted Bahamut-deck games (vs Urza deck), multiple seeds — decisive results, no
  draws, no non-fatal errors; Bahamut's chapter IV / Mega Flare observed in real games.

## Result
Implemented and verified — all four chapters (two destroys, draw two, Mega Flare), the
sum-of-other-permanents'-mana-value damage, the 9/9 flyer body, and the post-IV sacrifice of the
creature Saga behave per the Comprehensive Rules, on the shared CR 714 Saga engine.
