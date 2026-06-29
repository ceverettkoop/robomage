# Candelabra of Tawnos  (vocab index 242)

## Oracle text
{X}, {T}: Untap X target lands.

(Artifact, mana cost {1}.)

## Forge script
- Source: pre-existing local script — `bin/resources/cardsfolder/c/candelabra_of_tawnos.txt` (not edited).
- Key tags:
  - `A:AB$ Untap | Cost$ X T | TargetMin$ X | TargetMax$ X | ValidTgts$ Land`
  - `SVar:X:Count$xPaid`
- Tags parsed as written; `TgtPrompt$`/`SpellDescription$`/`AI:` are cosmetic.

## Engine work (builds on the shared "variable-X" unit; see toxic_deluge.md for `Count$xPaid`/`-X`)

This card needs two pieces both new to this unit: an **X cost on an activated ability** and
**exactly-X targeting**.

### X activation cost `Cost$ X T`
- `src/parse.cpp` `parse_activation_cost`: a bare `X` token in the cost (which `parse_mana_cost`
  otherwise drops, since {X} has no fixed value) now sets the new flag `Ability::activation_has_x`
  (`src/components/ability.h`).
- `src/action_processor.cpp` `process_activate_ability`: for a non-mana ability with
  `activation_has_x`, X is prompted (ActionCategory `CHOOSE_X`, bounded by
  `max_available_mana` beyond the rest of the cost) **before** targets are selected (CR 602.2b →
  601.2b), `cur_game.x_paid` is set, and X generic mana is added to the activation mana cost paid
  below. (No persistence of `x_paid` past resolution is needed here — Untap's targets are chosen at
  activation, so resolution reads nothing X-dependent.)

### Exactly-X targeting `TargetMin$ X | TargetMax$ X`
- `src/parse.cpp` `apply_param_to_ability`: `TargetMin$`/`TargetMax$` given as a non-numeric SVar
  key are stashed (`Ability::target_min_svar` / `target_max_svar`) instead of `stoi`-ing (the old
  `TargetMin$ X` crash); `TargetMax$ X` still falls back to `MAX_ENTITIES` until resolved.
- Shared helper `resolve_xpaid_target_counts()` (`src/parse.cpp`, called from both
  `parse_abilities` and `parse_svar_ability`) resolves each stashed token through the SVar map; when
  it is `Count$xPaid` it sets `target_min_from_xpaid` / `target_max_from_xpaid`
  (`src/components/ability.h`).
- `src/action_processor.cpp` `select_target` (multi-target loop): the cap is `x_paid` when
  `target_max_from_xpaid`, and the minimum is `x_paid` when `target_min_from_xpaid`. With BOTH set
  (TargetMin$ X = TargetMax$ X) the loop neither offers "Done" nor stops before X targets are
  chosen and clamps at X → **exactly X** targets. (`target_max_from_xpaid` alone is the pre-existing
  "up to X", e.g. Kozilek's Command.) `x_paid` was chosen before targets, so it is known here.
- `src/effects/effect_untap.cpp`: generalized to untap **every** chosen target (`ab.targets` for a
  multi-target Untap; falls back to `ab.target` for single-target), instead of only `ab.target`.

### How a future card opts in
Any ability whose script gives `TargetMin$`/`TargetMax$` as an SVar that resolves to `Count$xPaid`
automatically gets up-to-X (max only) or exactly-X (min = max) targeting — no per-card code. Any
activated ability with `{X}` in its `Cost$` gets the X prompt + generic-mana injection.

## Behavioral decisions (made in lieu of asking a human)
- **X is chosen at activation, before targets** (CR 601.2b), bounded by the mana the activator can
  actually produce for the rest of the cost.
- **Exactly X** targets: with TargetMin$ = TargetMax$ = X the player cannot choose fewer or more
  than X lands; X=0 untaps nothing (legal no-op).
- `ValidTgts$ Land` has no controller qualifier, so any land (either player's) is a legal target.

## Tests (test_harness, seed 1; board A: Candelabra + 3 Islands)
- **Activate, choose X=2:** "Choose X value (0-3)" offered (3 available mana sources); targeting
  required exactly two of the three Islands — the menu offered **no** "Done"/"No target" at the
  first pick and **stopped at 2** without offering a third (so the player can pick neither 1 nor 3);
  paying {2} tapped two Islands, and on resolution both targeted Islands untapped
  (tapped→untapped flip confirmed). PASS.
- **Regression (`--scripted` full games, seeds 1-6):** Candelabra activated 6-8 times per game in
  seeds 3-6, all games decisive (no draws), no non-fatal errors.

## Result
implemented
