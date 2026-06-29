# Cabal Ritual  (vocab index 269)

## Oracle text
Add {B}{B}{B}.
Threshold — Add {B}{B}{B}{B}{B} instead if there are seven or more cards in your graveyard.

## Forge script
- Source: present locally → `bin/resources/cardsfolder/c/cabal_ritual.txt`
- `Types:Instant`, `ManaCost:1 B`.
- Key tags:
  - `A:SP$ Mana | Produced$ B | Amount$ X` — a mana-adding instant (the Dark Ritual family),
    resolving via `effects::add_mana`, not an activated mana ability.
  - `SVar:X:Count$Threshold.5.3` — dynamic amount: 5 black mana with Threshold, else 3.
  - `AILogic$/AINoRecursiveCheck$/AI:` — cosmetic AI hints (ignored).

## Engine work
- **`Count$Threshold.<hi>.<lo>` dynamic amount — new general handler.** Threshold (CR 702.27,
  historical keyword action; modern cards spell the condition out) is "you have seven or more
  cards in your graveyard." Added to the shared runtime-amount evaluator
  `evaluate_dynamic_amount` (`src/components/ability.cpp`, immediately after the analogous
  `Count$Revolt.high.low` branch): parses the `.<hi>.<lo>` pair and returns `<hi>` if
  `orderer->get_graveyard(ctrl).size() >= 7`, else `<lo>`. Keyed on the tag's meaning, so any
  card scaling a dynamic amount (mana / damage / draw / tokens) by Threshold reuses it.
- **Parser preservation.** `src/parse.cpp` (top-level Amount$ SVar resolution) now also preserves
  a `Count$Threshold…` expression into `ability.dynamic_amount_expr` (alongside the existing
  `Count$Valid` / `Count$Revolt` / `xPaid` forms) so it survives to resolution.
- **Mana-adding spell scales by the dynamic amount.** `effects::add_mana`
  (`src/effects/effect_add_mana.cpp`) previously used `ab.amount` verbatim (Dark Ritual makes a
  fixed 3). It now evaluates `ab.dynamic_amount_expr` via `evaluate_dynamic_amount` when present,
  so a mana-adding spell (SP$ Mana) scales by the same Count$/Targeted$ grammar as dynamic
  damage/draw/token counts.
- **Activated mana abilities too (generalization).** `eval_mana_amount`
  (`src/mana_system.cpp`) now falls back to `evaluate_dynamic_amount` for any non-empty
  `dynamic_amount_expr` not matched by its source-aware `Count$Valid` branch, so an AB$ Mana
  source with a dynamic amount (e.g. a future Threshold mana rock) is handled identically. The
  existing `Count$Valid` path still short-circuits first — no behavioral change to Gaea's Cradle
  et al.

## Behavioral decisions (made in lieu of asking a human)
- **"Your graveyard" = the controller's graveyard** (CR 109.5 / 400.3). The spell's controller is
  the "you"; `ctrl` flows from the activating/casting player. Graveyard membership is by owner, and
  the caster owns their own graveyard, so `get_graveyard(ctrl)` is the right set.
- **Threshold is checked at resolution.** Cabal Ritual is still on the stack (not in the
  graveyard) while it resolves, so its own card does not count toward the seven — matching CR
  608 (the spell moves to the graveyard as the last step of resolution).
- **The cosmetic `amount: 1` resolve log** prints `ab.amount`'s default; the actual mana added is
  the evaluated dynamic amount (`Player A adds 3{B}` / `5{B}`), which is what matters.

## Tests
- Isolation (test_harness, inline hand + preset battlefield, 2 Swamps to pay {1}{B}):
  - **Below threshold (3 cards in graveyard):** `--graveyard-a "Swamp,Swamp,Swamp"`, cast Cabal
    Ritual → `Player A adds 3{B}`. PASS.
  - **At/above threshold (7 cards in graveyard):** `--graveyard-a "Swamp×7"`, cast Cabal Ritual →
    `Player A adds 5{B}`. PASS.
  - No draws, no fatal or non-fatal errors in either run.

## Result
implemented
