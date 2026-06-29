# The Fantasticar (vocab index 267)

## Oracle text
Flying

Whenever you cast a noncreature spell, you may have The Fantasticar become an artifact
creature until end of turn.

Whenever you cast your fourth noncreature spell each turn, you may sacrifice The Fantasticar.
If you do, create four 4/4 colorless Construct artifact creature tokens with flying and haste.

(Legendary Artifact — Vehicle, 4/4, `{3}`)

## Forge script
Source: provided combined script at `bin/resources/cardsfolder/t/the_fantasticar.txt`.

```
K:Flying
T:Mode$ SpellCast | ValidCard$ Card.nonCreature | ValidActivatingPlayer$ You | TriggerZones$ Battlefield | Execute$ TrigAnimate | OptionalDecider$ You
SVar:TrigAnimate:DB$ Animate | Defined$ Self | Types$ Creature,Artifact
T:Mode$ SpellCast | ValidCard$ Card.nonCreature | ValidActivatingPlayer$ You | ActivatorThisTurnCast$ EQ4 | TriggerZones$ Battlefield | Execute$ TrigToken
SVar:TrigToken:AB$ Token | Cost$ Sac<1/CARDNAME> | TokenOwner$ You | TokenScript$ c_4_4_a_construct_flying_haste | TokenAmount$ 4
```

Key tags: trigger 1 is an optional (`OptionalDecider$ You`) `DB$ Animate | Defined$ Self |
Types$ Creature,Artifact` with the Forge-default "until end of turn" duration (no
`Duration$ Permanent`). Trigger 2 fires on the controller's fourth NONCREATURE spell
(`ActivatorThisTurnCast$ EQ4` combined with `ValidCard$ Card.nonCreature`) and its execute is an
`AB$ Token` whose `Cost$ Sac<1/CARDNAME>` is the "you may sacrifice... if you do" gate.

## Engine work
All four changes are general (keyed on the tag's meaning), not Fantasticar-specific.

1. **Comma-separated `Types$` in Animate** (`src/parse.cpp`). Forge separates an Animate
   `Types$` list with commas (`Creature,Artifact`), unlike the space-separated printed `Types:`
   line. The Animate `Types$` branch now normalizes commas to spaces before `parse_types`, so the
   list splits into distinct types instead of one bogus subtype token.

2. **Until-end-of-turn Animate duration that reverts at cleanup** (CR 613 layer 4 / CR 514.2).
   New EOT bucket fields on `Permanent` (`src/components/permanent.h`):
   `animate_added_types_eot` and `animate_make_creature_eot`, mirroring the rest-of-game
   `animate_added_types` / `animate_make_creature`. `effects::animate`
   (`src/effects/effect_animate.cpp`) routes by `ab.animate_duration_permanent`:
   `Duration$ Permanent` → persistent fields; anything else (the default) → EOT fields. The EOT
   path records only types the permanent does NOT already have, so the cleanup revert erases
   exactly the grant (The Fantasticar keeps its printed Artifact type). The layer pass reapplies
   the EOT bucket every SBE (`src/systems/state_manager_statics.cpp`:
   `apply_type_changing_effects` + the creature bootstrap), and the CLEANUP step
   (`src/classes/game.cpp`) erases the EOT-added types and strips the bootstrapped
   `Creature`/`Damage` components (unless the permanent is a creature by a permanent means), so
   the permanent stops being a creature at end of turn. A future card opts into the EOT duration
   simply by giving its Animate no `Duration$ Permanent`.

3. **`Defined$ Self` Animate target + Vehicle base P/T** (`src/effects/effect_animate.cpp`).
   The Animate handler now reads `ab.defined_self ? ab.source : ab.target`, matching the existing
   `defined_self` pattern (e.g. `effect_tap.cpp`), so "have CARDNAME become…" animates its own
   source. `apply_animate_creature_bootstrap` falls back to the permanent's PRINTED P/T from
   `CardData` when no Animate `PT$` was set, so a Vehicle animates as its printed 4/4 (a land
   still animates as its printed 0/0).

4. **"Your Nth NONCREATURE spell" count trigger** (`src/parse.cpp`,
   `src/components/ability.{h,cpp}`, `src/systems/state_manager_triggers.cpp`). Previously the
   `ActivatorThisTurnCast$ EQN` block and the `Card.nonCreature` block conflicted, and the count
   compared against ALL spells cast. Now when both are present the trigger binds to `SPELL_CAST`
   (fired AFTER the per-cast counters bump, unlike `NONCREATURE_SPELL_CAST` which fires before),
   filters the triggering card to noncreature (`trigger_valid_card_non_creature`), and compares
   the count against `Player::noncreature_spells_cast_this_turn`
   (`trigger_spell_count_noncreature`). Both new trigger fields are carried through the Execute
   SVar copy in `parse_svar_ability`.

5. **Reflexive "you may sacrifice CARDNAME" cost on a triggered ability**
   (`src/components/ability.cpp`, `Ability::resolve`). An activated ability pays its
   `Sac<…/CARDNAME>` cost up front at activation; a triggered ability pays it as it resolves
   (CR 603.2), and the sac cost makes the whole effect optional. So a TRIGGERED ability with
   `sac_self` now prompts the controller at resolution, sacrifices the source on accept, and does
   nothing (skips the effect and its subabilities) on decline. Activated abilities never reach
   this branch with `ability_type == TRIGGERED`, so their already-paid sac is never double-charged.

Token script `c_4_4_a_construct_flying_haste` fetched via `tools/forge_fetch/fetch_script.py
--token`. No retagging; no card-script edits.

## Behavioral decisions (CR cites)
- The animate grant is "until end of turn" (Forge default duration) and ends during the cleanup
  step (CR 514.2). The Creature type is added in layer 4 (CR 613.1d); the Vehicle becomes a 4/4
  artifact creature that can attack, then reverts to a noncreature Vehicle at end of turn.
- Trigger 2 counts only noncreature spells (Oracle "your fourth noncreature spell each turn").
- The sacrifice is optional and is the gate for the tokens ("you may sacrifice… If you do,
  create…"): decline → no sac, no tokens. The four Constructs are 4/4 with flying and haste.

## Tests (`train/test_harness.py`)
- **Animate + EOT revert (key test)**: The Fantasticar in play, cast Lightning Bolt → accept the
  optional trigger → it logs "becomes an Artifact" + "becomes a Creature", shows `[4/4]`, is
  declared as an attacker, and deals 4 combat damage. On the controller's NEXT turn it shows with
  NO `[4/4]` and "No creatures eligible to attack" — the Creature type reverted at cleanup.
- **Fourth noncreature spell → sac → tokens**: cast four Lightning Bolts; on the fourth the second
  trigger fires, prompts "Sacrifice The Fantasticar", sacrifices it, and creates four 4/4 Construct
  tokens that attack the same turn (haste) — flying tokens. Declining the prompt creates no tokens
  and keeps The Fantasticar (verified both branches).

Regression: scripted full games, `temp/fant_a` (The Fantasticar + Lightning Bolt + Brainstorm +
Ponder + Preordain + Island/Mountain) vs `temp/fant_b`, seeds 1/2/3 — all decisive (no draws), no
non-fatal errors (only the pre-existing cosmetic Brainstorm `Reorder$` warning).

## Result
Implemented. Animate-until-end-of-turn now has a real reverting duration (general EOT bucket on
`Permanent`); `Defined$ Self` animate animates its source as its printed P/T; the fourth-
noncreature-spell trigger counts noncreature spells; and a triggered ability's reflexive
`Sac<…/CARDNAME>` cost is an optional sacrifice gate. Build clean; tests and regression pass.
