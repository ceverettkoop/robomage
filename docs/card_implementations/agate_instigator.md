# Agate Instigator  (vocab index 122)

## Oracle text
Offspring {1}{R} (You may pay an additional {1}{R} as you cast this spell. If you do, when
this creature enters, create a 1/1 token copy of it.)

Whenever another creature you control enters, this creature deals 1 damage to each opponent.

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/a/agate_instigator.txt`
- Type: `Creature Lizard Rogue`, mana cost `1 R`, P/T `1/3`.
- Key tags:
  - `K:Offspring:1 R` — the Offspring keyword (CR 702.171), an **optional additional cost**.
  - `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield |
    ValidCard$ Creature.Other+YouCtrl | TriggerZones$ Battlefield | Execute$ TrigDamage` —
    "Whenever another creature you control enters, ...".
  - `SVar:TrigDamage:DB$ DealDamage | Defined$ Player.Opponent | NumDmg$ 1` — the damage
    body (each opponent, 1 damage).
  - `SVar:BuffedBy:Creature` — purely informational (AI hint); ignored (cosmetic).

## Engine work
Two mechanics needed engine support: the `Defined$ Player.Opponent` damage target, and the
`Offspring` keyword. Both are implemented as **general** handlers keyed on the tag's intended
meaning (no retagging).

1. **`Defined$ Player.Opponent` → "deal to each opponent"** (CR 109.5 / 102.1).
   - `src/parse.cpp`: the `Defined`/`DefinedPlayer` handler now recognizes `Player.Opponent`
     (and bare `Opponent`) and sets a new `Ability::defined_each_opponent` flag
     (`src/components/ability.h`). This survives the `Execute$` SVar copy in
     `parse_one_trigger` (the resolved effect ability replaces the trigger ability wholesale).
   - `src/effects/effect_deal_damage.cpp`: when `defined_each_opponent` is set, the handler
     resolves the source controller's opponent at resolution time and deals `dmg` to that
     player (no chosen target). In a two-player game "each opponent" is the single opponent
     (the standard `(ctrl == PLAYER_A) ? PLAYER_B : PLAYER_A` pattern used elsewhere).
     The trigger carries `ValidTgts$ N_A` (no target selection), so this is a defined,
     untargeted effect — matching the Oracle wording "each opponent" (not "target").

2. **Offspring keyword** (CR 702.171) — an **optional additional cost** with an ETB rider.
   Modeled closely on the existing Evoke implementation (an optional cost option whose payment
   gates a synthetic ETB self-trigger), but Offspring is *additional* (paid on top of the
   spell's mana cost) rather than *alternative*:
   - `src/components/carddata.h`: `CardData::has_offspring` + `CardData::offspring_cost`
     (`ManaValue`), parsed from `K:Offspring:<cost>` in `src/parse.cpp` via `parse_mana_cost`.
   - The parser also synthesizes a self-ETB trigger
     (`category = "CopyPermanent"`, `trigger_only_self`, `is_offspring_token = true`,
     destination Battlefield) on the card — analogous to Evoke's synthetic self-sacrifice ETB.
   - **Cast option** (`src/systems/state_manager_actions.cpp`): when `has_offspring`, a second
     "Cast X (offspring)" action is offered whenever the player can pay base cost + offspring
     cost together; it sets a new `LegalAction::use_offspring` flag (`src/classes/action.h`).
   - **Payment** (`src/action_processor.cpp`): the regular-cost cast branch adds the offspring
     cost to the mana to pay when `use_offspring`; the resulting `Spell` carries
     `cast_with_offspring` (`src/components/spell.h`).
   - **ETB gating** (mirroring `pending_evoked` → `Permanent::evoked`):
     `src/systems/stack_manager.cpp` moves `cast_with_offspring` into a one-shot
     `Game::pending_offspring` set as the permanent resolves;
     `src/systems/state_manager_statics.cpp` consumes it to set
     `Permanent::entered_with_offspring`. The synthetic trigger is suppressed in
     `src/systems/state_manager_triggers.cpp` unless that flag is set (so casting normally
     makes no token).
   - **Token copy** (`src/effects/effect_copy_permanent.cpp`): an `is_offspring_token` branch
     reuses the existing `copyable_token_of(source)` snapshot (CR 707.2) and overrides the
     copy's P/T to 1/1 (CR 702.171c, "a 1/1 token that's a copy of it"), then creates it on
     the battlefield under the caster's control via `bootstrap_token_components`.

## Behavioral decisions (made in lieu of asking a human)
- **`SVar:BuffedBy:Creature` ignored (cosmetic).** It is a Forge AI evaluation hint only; it
  has no rules effect. No behavior change; no new warning is emitted for it.
- **"each opponent" in a two-player game.** Per CR 102.1 every player other than the source's
  controller is an opponent; with exactly two players the effect deals 1 to the single
  opponent. The engine is two-player, so `defined_each_opponent` resolves to that one player.
- **Offspring token is a true copy.** The 1/1 token copies Agate Instigator's name, types,
  keywords, and abilities (it is itself an Agate Instigator with the damage trigger), with P/T
  set to 1/1 (702.171c). Its ETB is "another creature you control entering" for the original,
  so it correctly triggers the original's 1-damage ability once. Token color is not a modeled
  field in the engine, matching the engine's copy fidelity elsewhere.
- **Offspring is additional, not alternative.** Implemented as a distinct cast option that pays
  base + offspring cost, not via the `AltCost` (alternative-cost) path used by Evoke/Flashback,
  so the base mana cost is still paid (CR 601.2f, additional cost).

## Tests
Isolation (`train/test_harness.py`, pre-set battlefield, seed 1):
- **Base trigger fires only on *another* creature.** A has `Agate Instigator` + `Grizzly Bears`
  in hand and 5 lands in play. Casting Agate Instigator alone deals **no** damage (its own ETB
  is excluded by `Creature.Other`). Then casting Grizzly Bears: "Grizzly Bears enters the
  battlefield" → "Dealt 1 damage to player (now at 19 life)"; opponent 20 → **19**. PASS.
- **Offspring → 1/1 token copy + trigger.** Casting "Agate Instigator (offspring)" (paying
  base {1}{R} + offspring {1}{R}): "Offspring token copy created: 1/1 Agate Instigator", board
  shows the original `Agate Instigator [1/3]` plus a `Token [1/1]`, and the token's ETB
  triggers the original's ability: "Dealt 1 damage to player (now at 19 life)"; opponent
  20 → **19**. The token is itself an Agate Instigator (a real copy). PASS.

Regression (`train/test_harness.py --scripted`, seeds 1–6): a mono-ish RG deck `temp/agate_test`
(4 Agate Instigator / 4 Grizzly Bears / 4 Lightning Bolt / 12 Mountain / 12 Forest) vs
`temp/agate_opp`. All 6 games finished **decisively** (A wins 5, B wins 1), **no draws**, and no
non-fatal errors / asserts / tracebacks. The damage trigger fired repeatedly across games
("Agate Instigator deals 1 damage to Player B"). Only pre-existing cosmetic `Unrecognized
ability param` warnings on unrelated cards appear. (`train.py observe` could not be used —
`torch` is not installed — so the scripted regression ran directly through the harness.)

## Result
implemented
