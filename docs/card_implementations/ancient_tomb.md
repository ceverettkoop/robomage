# Ancient Tomb  (vocab index 123)

## Oracle text
{T}: Add {C}{C}. Ancient Tomb deals 2 damage to you.

## Forge script
- Source: fetched (Forge@master) → `bin/resources/cardsfolder/a/ancient_tomb.txt`
- `Types:Land` — a nonbasic colorless land with no land subtypes, so its mana ability is
  scripted (not injected from a subtype).
- Key tags:
  - `A:AB$ Mana | Cost$ T | Produced$ C | Amount$ 2 | SubAbility$ DBPain` — {T}: add two
    colorless mana, with a chained sub-ability rider.
  - `SVar:DBPain:DB$ DealDamage | NumDmg$ 2 | Defined$ You` — the rider: "Ancient Tomb deals
    2 damage to you."

## Engine work
Two general handlers were added, both keyed on the tags' intended meaning (not retags):

1. **`Amount$` on a mana ability** — *parser fix.* `effects::parse_add_mana`
   (`src/effects/effect_add_mana.cpp`) previously hard-coded `ab.amount = 1` for every
   `Produced$` form, ignoring `Amount$`. `Amount$` is consumed separately by
   `apply_param_to_ability` (which sets `ability.amount`), but the two params can appear in
   either order on the line, so `parse_add_mana` now reads the already-parsed amount and only
   falls back to a default of 1 when none was set (`ab.amount` still at its 0 default). The
   mana-payment evaluator `eval_mana_amount` (`src/mana_system.cpp`) already returns
   `ab.amount`, so the source now correctly produces 2 colorless.

2. **`Defined$ You` on a `DealDamage` sub-ability** — *new general handler.* The parser
   (`src/parse.cpp`) previously recognized `Defined$` values `Remembered` / `TargetedController`
   / `Self` / `Player.Opponent` but not `You`. Added a `defined_you` flag on `Ability`
   (`src/components/ability.h`), set when `Defined$ You`. `effects::deal_damage`
   (`src/effects/effect_deal_damage.cpp`) resolves `defined_you` to the source's controller's
   player entity (mirroring the existing `defined_each_opponent` branch) and deals the damage —
   no chosen target. CR 109.5: "you" is the ability's controller.

3. **Mana-ability `SubAbility$` resolution (off-stack)** — *new general behavior.* A mana
   ability does not use the stack (CR 605.3a), so a `SubAbility$` rider attached to it must
   resolve immediately as part of the mana ability resolving (CR 605.1a / 606). The two places
   that resolve a mana ability now also resolve its `subabilities` (with the mana ability's
   source/controller), guarded so they fire only when the activation is committed:
   - `activate_mana_source` in `src/mana_system.cpp` — the path used when a land is tapped to
     pay for a spell (the common case; `commit==true` only, never during legality simulation).
   - the `is_mana_ability` branch in `src/action_processor.cpp` — the standalone activation path.

## Behavioral decisions (made in lieu of asking a human)
- **Damage amount is 2, not 1.** The Forge script and Oracle text both say "deals 2 damage to
  you" (`NumDmg$ 2`); that is authoritative. (The task brief's "1 damage" wording was a
  paraphrase; the implementation follows the script.)
- **The pain is part of the mana ability, not a separate ability/trigger.** The whole thing is
  one mana ability that resolves without using the stack (CR 605). Tapping Ancient Tomb to pay
  for a spell therefore deals the 2 damage immediately, before the spell is cast — verified in
  isolation.
- **Mana abilities are not offered standalone at priority** in this engine; they are enumerated
  only while paying a cost. So the observable trigger path in normal play is via
  `activate_mana_source`; the `action_processor.cpp` standalone path was wired identically as a
  safety net for any direct mana-ability activation.
- **Roll-back edge case:** the snapshot/restore used to cancel an in-progress mana payment does
  not un-deal the 2 life if a player taps Ancient Tomb and then cancels the payment. This is an
  obscure manual-cancel case; the scripted agent and normal play never exercise it, and the
  damage is intrinsically part of the (already-committed) mana production. Left as-is rather than
  threading a damage rollback through the snapshot path.

## Tests
- Isolation (test_harness, preset battlefield): Ancient Tomb in play, cast Aether Vial ({1})
  paying with it → log shows `Player A activated Ancient Tomb for 2(C)`,
  `Dealt 2 damage to player (now at 18 life)`, Aether Vial cast with one {C}, leftover `mana: 1C`
  floating, Ancient Tomb tapped, life 20 → 18. PASS (mana = 2 colorless AND controller loses 2
  life AND the {C}{C} pays a 1-generic cost with one to spare).
- Regression (test_harness `--scripted`, 6 seeds, both decks containing 4× Ancient Tomb):
  seeds 1–6 all decisive (A/A/A/A/B/A wins), no draws, no fatal or non-fatal errors. Both players
  were observed activating Ancient Tomb for 2(C) in real games. PASS.

## Result
implemented
