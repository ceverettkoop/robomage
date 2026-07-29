# Reanimate

## Oracle text
Put target creature card from a graveyard onto the battlefield under your control. You lose life equal to its mana value.

(`{B}` sorcery)

## Forge script
- **Source:** pre-existing local (`bin/resources/cardsfolder/r/reanimate.txt`).
- **Key tags:**
  - `A:SP$ ChangeZone | Origin$ Graveyard | Destination$ Battlefield | GainControl$ True | ValidTgts$ Creature | ChangeNum$ 1 | RememberLKI$ True | SubAbility$ DBLoseLifeYou`
  - `SVar:DBLoseLifeYou:DB$ LoseLife | Defined$ You | LifeAmount$ X | SubAbility$ DBCleanup`
  - `SVar:DBCleanup:DB$ Cleanup | ClearRemembered$ True`
  - `SVar:X:RememberedLKI$CardManaCost`

## Engine work
Fix key (general): **remembered-lki-cardmanacost**.

The dynamic-amount plumbing already understood `Remembered$CardManaCost` (Birthing Ritual,
Skyclave Apparition), and a `RememberLKI$ True` ChangeZone already pushes the moved card's
last-known-info snapshot into `cur_game.remembered_entities`
(`effect_change_zone.cpp:244`). But the `RememberedLKI$` token form was recognized nowhere:

1. `src/parse.cpp` — the sub-ability `amount_svar` → `dynamic_amount_expr` promotion only
   matched `sv.find("Remembered$")`, so `RememberedLKI$CardManaCost` was never promoted and
   `LoseLife` fell back to `ab.amount` (0). Added a `RememberedLKI$` branch alongside it.
2. `src/components/ability.cpp` — `evaluate_dynamic_amount`'s `Remembered$CardManaCost` branch
   now also accepts `RememberedLKI$CardManaCost`. Both read the mana value of the first
   remembered entity, so they resolve through the identical code path.

CR 608.2h (last-known information): the reanimated card has already left the graveyard as it
enters play, so its mana value is read from the LKI snapshot captured at the move.

Mechanics added (general): **RememberedLKI$CardManaCost** dynamic-amount token — any
`RememberLKI$` ChangeZone can now feed the moved card's last-known mana value into a chained
numeric effect (life loss, counters, damage, …), not just Reanimate.

## Behavioral decisions
- **Control:** `GainControl$ True` is not separately parsed (it logs a cosmetic
  "Unrecognized ability param" warning), but the ChangeZone target-move path already sets the
  reanimated permanent's controller to the ability's controller (the caster). Verified: a
  creature reanimated from the OPPONENT's graveyard enters under Player A's control and attacks
  the opponent. No additional work needed.
- **Life loss = mana value:** read via LKI, so X/hybrid/Phyrexian pips resolve to the printed
  mana value of the card that just entered.

## Tests (isolation)
- `--graveyard-a "Grizzly Bears"` (MV2), cast Reanimate targeting it → Grizzly Bears enters
  Player A's battlefield, **Player A loses 2 life** (20 → 18). PASS.
- `--graveyard-a "Griselbrand"` (MV8) → Griselbrand enters, **Player A loses 8 life**
  (20 → 12). PASS.
- `--graveyard-b "Grizzly Bears"` (opponent's GY) → creature enters under **Player A's**
  control, attacks Player B for 2, A loses 2 life. PASS (control confirmed).
- CI gate: `ci_check.py --tier pygen,vocab,smoke` — see batch report.

## Result
Implemented.
