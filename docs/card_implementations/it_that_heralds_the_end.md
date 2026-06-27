# It That Heralds the End

## Oracle text
Colorless spells you cast with mana value 7 or greater cost {1} less to cast.
Other colorless creatures you control get +1/+1.

(2/2 Creature — Eldrazi Drone, mana cost `{1}{C}`.)

## Script source + key tags
`bin/resources/cardsfolder/i/it_that_heralds_the_end.txt`

```
S:Mode$ ReduceCost | ValidCard$ Card.Colorless+cmcGE7 | Type$ Spell | Activator$ You | Amount$ 1 | ...
S:Mode$ Continuous | Affected$ Creature.Colorless+Other+YouCtrl | AddPower$ 1 | AddToughness$ 1 | ...
```

Key tags:
- **`Mode$ ReduceCost` / `ValidCard$ Card.Colorless+cmcGE7` / `Activator$ You` / `Amount$ 1`** —
  reduce the generic portion of *your* colorless spells with mana value ≥ 7 by `{1}` (CR 601.2f / 118.7).
- **`Mode$ Continuous` / `Affected$ Creature.Colorless+Other+YouCtrl` / `AddPower$ 1` / `AddToughness$ 1`** —
  an anthem (continuous static, layer 7c) granting +1/+1 to every *other* colorless creature *you* control
  (CR 611 / 613.4c). `+Other` excludes the source; `+YouCtrl` scopes to the static's controller.

## Engine work + CR rules

The two statics were already parsed (`parse_static_abilities` in `src/parse.cpp`) into a
`StaticAbility` (`reduce_cost` / `reduce_cost_filter` / `reduce_cost_you_only`, and `add_power` /
`add_toughness` / `affected`). Two gaps remained:

1. **General `Affected$` resolver for continuous P/T statics (the anthem).**
   The layer-7c additive applier (`StateManager::apply_layer7_pt_effects`,
   `src/systems/state_manager_statics.cpp`) only resolved the `EquippedBy` / `Self` forms and
   otherwise fell back to buffing the *source* permanent. A `Affected$` value that is a general
   permanent *filter* (`Creature.Colorless+Other+YouCtrl`) was never resolved to the matching
   creature set.
   - Added a fully general, reusable resolver
     **`affected_permanents_for_static(const ActiveStatic&, const std::set<Entity>&)`**
     (declared in `src/systems/state_manager.h`, defined in `src/systems/state_manager_statics.cpp`).
     It returns the battlefield permanents a continuous static's `Affected$` filter designates,
     evaluated through the shared `permanent_matches_filter` (`src/game_queries.h`) — so the full
     qualifier grammar (colors/`Colorless`, subtypes, P/T, …) is honoured. `YouCtrl`/`OppCtrl` are
     scoped to the static's controller via `MatchCtx::controller`, and `+Other` self-exclusion via
     `MatchCtx::source = as.entity`. Returns empty for the `EquippedBy` / `Self` / no-filter forms
     (those resolve to a single creature in the applier). **Petrified Hamlet's `AddAbility$` static
     is intended to reuse this same resolver.**
   - The 7c applier now, for a general `Affected$` filter, emits one `ContinuousEffect` per matched
     permanent (with the source's timestamp, CR 613.7a); the `EquippedBy`/`Self`/no-filter forms keep
     the original single-target behaviour.
   - `apply_layer6_ability_effects` now skips a pure-P/T anthem (general filter, no `AddKeyword$`) so
     its single-target (source) logging path doesn't mis-log a +N/+N "grant" on the anthem source
     — P/T is applied entirely in layer 7.

2. **`cmcGE7` static mana-value bound on the `ReduceCost` filter.**
   The filter evaluator defers a static `cmc` comparator (`cmcGE7`, `cmcLE3`, …) to
   `MatchCtx::cmc_bound`; with the default `MatchCtx` the bare `cmcGE7` token returns `true`, so the
   reduction would have applied to **every** colorless spell regardless of mana value.
   - Added reusable **`extract_static_cmc_bound(spec, MatchCtx&)`** in `src/game_queries.h` (mirrors
     the per-token extraction in `svar_eval.cpp`), and `active_reduce_cost_for`
     (`src/systems/state_manager_statics.cpp`) now seeds the bound before matching the spell. The
     same helper seeds the bound inside `affected_permanents_for_static`.

Cost reduction continues to flow through `effective_base_cost` → `active_reduce_cost_for`: only the
generic portion is removed, clamped at zero, and never a colored pip (CR 118.7), gated to the
source's controller by `reduce_cost_you_only` (`Activator$ You`).

## Decisions
- The `Affected$` resolution was built as a single reusable function rather than inline so future
  anthems (`AddAbility$` / `AddKeyword$` statics, e.g. Petrified Hamlet) reuse the exact same
  controller-scoped, `+Other`-aware filter step.
- Static numeric `cmc` extraction was factored into a shared header helper so any static-cmc filter
  site cannot drift on the parsing.
- No card script was edited; no tag was retagged.

## Tests → results
Test harness (`train/test_harness.py`, scripted/`--play`, seed 1), and a 3-seed scripted regression
with a curated colorless deck vs a Bolt/green deck.

- **Anthem (positive):** battlefield `It That Heralds the End` + `Thought-Knot Seer` (colorless 4/4)
  + `Scythecat Cub` (green 2/2) → `It That … [2/2]` (NOT buffed — `Other`), `Thought-Knot Seer [5/5]`
  (buffed +1/+1), `Scythecat Cub [2/2]` (NOT buffed — not colorless). ✓
- **Anthem (dynamic removal):** Player B bolts `It That Heralds the End`; once it dies,
  `Thought-Knot Seer` drops `5/5 → 4/4` the same SBA pass. ✓
- **Cost reduction (positive):** with `It That …` in play, `Sire of Seven Deaths` (colorless, MV 7)
  was castable with only **6** Wastes (reduced `{7} → {6}`); it also entered as `8/8` (anthem). ✓
- **Cost reduction (negative, MV bound):** `Thought-Knot Seer` (colorless, **MV 4**) was NOT castable
  with 3 Wastes (no reduction) but cast normally with 4 — `cmcGE7` correctly excludes MV < 7. ✓
- **Regression:** 3 scripted seeds all finished with a decisive winner; no draws, no assertions,
  no non-fatal errors (only pre-existing cosmetic "Unrecognized ability param" warnings).

## Result
**IMPLEMENTED** — anthem (+1/+1 to other colorless creatures you control) via a general,
reusable `Affected$`-filter resolver for continuous P/T statics; colorless-MV≥7 cost reduction with
the `cmcGE7` bound honoured. Builds clean headless; vocab index 199.
