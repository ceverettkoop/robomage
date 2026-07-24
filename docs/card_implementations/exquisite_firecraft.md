# Exquisite Firecraft

## Oracle text
Exquisite Firecraft deals 4 damage to any target.
Spell mastery — If there are two or more instant and/or sorcery cards in your graveyard, this spell can't be countered.

(`{1}{R}{R}` sorcery)

## Forge script
- **Source:** pre-existing local (`bin/resources/cardsfolder/e/exquisite_firecraft.txt`).
- **Key tags:**
  - `A:SP$ DealDamage | ValidTgts$ Any | NumDmg$ 4`
  - `R:Event$ Counter | ValidCard$ Card.Self | ValidSA$ Spell | Layer$ CantHappen | IsPresent$ Instant.YouOwn,Sorcery.YouOwn | PresentZone$ Graveyard | PresentCompare$ GE2`

## Engine work
Fix key (general): **conditional-cant-be-countered**.

The self-only `CANT_BE_COUNTERED` replacement was already built from `Event$ Counter` +
`ValidCard$ Card.Self` + `Layer$ CantHappen`, but the spell-mastery gate
(`IsPresent$`/`PresentZone$`/`PresentCompare$`) was never parsed into it — so the card was
UNCONDITIONALLY uncounterable. Made the gate a real, general condition:

1. `src/components/effect.h` — added `cant_counter_present` / `cant_counter_zone` /
   `cant_counter_compare` to `Effect::Replacement`.
2. `src/parse.cpp` — the self-form `CANT_BE_COUNTERED` block now captures `IsPresent$` (a
   comma-OR filter), `PresentZone$`, and `PresentCompare$` onto those fields.
3. `src/action_processor.cpp` — the cast-time stamp of `Spell::cant_be_countered` now fires
   ONLY for the unconditional self form (`cant_counter_present` empty, e.g. Long Goodbye); a
   conditional (spell-mastery) form is left unstamped.
4. `src/effects/effect_counter.cpp` — a new `spell_uncounterable_by_own_condition` helper
   re-evaluates the gate **as the counter would resolve** (CR 614.13 — the replacement fires
   now): it counts the caster-owned cards matching the comma-OR filter in the named zone
   (`card_matches_any` + `compare_svar`) and, if the compare holds, the counter is prevented.
   Wired into the existing can't-be-countered check next to `spell_uncounterable_by_static`.

CR 614.13 (can't-be-countered as a prevention / CantHappen). Evaluating at counter time (not
cast time) is correct: the graveyard is read at the moment the countering effect resolves.

Mechanics added (general): **conditional self can't-be-countered** — any card whose self
`CANT_BE_COUNTERED` replacement carries an `IsPresent$/PresentZone$/PresentCompare$` gate (a
zone count of matching cards) now honors that condition, re-checked at counter time.

## Behavioral decisions
- **Timing:** re-evaluated at counter resolution, not stamped at cast, so a graveyard that
  changes between cast and the counter attempt is read correctly (and matches how the
  battlefield-form `spell_uncounterable_by_static` is consulted at counter time).
- **Filter:** `Instant.YouOwn,Sorcery.YouOwn` is a comma-OR spec routed through the shared
  `card_matches_any`; `YouOwn` is honored via `MatchCtx.controller` = the caster, and the scan
  is additionally restricted to the caster-owned cards in the `PresentZone$` (Graveyard).
- **Zone mapping** covers Graveyard/Exile/Hand/Library (printed-characteristic zones), defaulting
  to Graveyard — the only zone any current spell-mastery card counts.

## Tests (isolation)
- (a) Graveyard empty — A casts Firecraft, B casts Counterspell targeting it → **"Exquisite
  Firecraft is countered"**. PASS.
- (b) Graveyard = Lightning Bolt + Ponder (2 inst/sorc) — B casts Counterspell targeting it →
  **"Exquisite Firecraft can't be countered"**, then the spell resolves and **deals 4** (Player
  B 20 → 16). PASS.
- Boundary: graveyard = Lightning Bolt only (1) → GE2 unmet → **still countered**. PASS.
- CI gate: `ci_check.py --tier pygen,vocab,smoke` — see batch report.

## Result
Implemented.
