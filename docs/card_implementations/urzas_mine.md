# Urza's Mine / Urza's Power Plant / Urza's Tower  (vocab indices 274 / 275 / 276)

The three "Tron" lands. Implemented as one batch sharing the new
`Count$UrzaLands.<hi>.<lo>` dynamic mana-amount handler.

## Oracle text
- **Urza's Mine** — {T}: Add {C}. If you control an Urza's Power-Plant and an Urza's Tower, add {C}{C} instead.
- **Urza's Power Plant** — {T}: Add {C}. If you control an Urza's Mine and an Urza's Tower, add {C}{C} instead.
- **Urza's Tower** — {T}: Add {C}. If you control an Urza's Mine and an Urza's Power-Plant, add {C}{C}{C} instead.

## Forge scripts
- Source: present locally → `bin/resources/cardsfolder/u/urzas_mine.txt`,
  `urzas_power_plant.txt`, `urzas_tower.txt`.
- `Types:Land <subtype>` (the subtype line uses Forge's hyphenated `Urza's Power-Plant`; the
  card *names* are unhyphenated `Urza's Power Plant`). `ManaCost:no cost`.
- Key tags (all three share the shape):
  - `A:AB$ Mana | Cost$ T | Produced$ C | Amount$ UrzaAmount` — a tap-for-colorless-mana
    activated ability (resolves through `eval_mana_amount` / `activate_mana_source`, never goes
    on the stack — CR 605.1a/605.3a).
  - `SVar:UrzaAmount:Count$UrzaLands.<hi>.<lo>` — the dynamic amount:
    - Urza's Mine: `Count$UrzaLands.2.1`  → **hi = 2, lo = 1**
    - Urza's Power Plant: `Count$UrzaLands.2.1`  → **hi = 2, lo = 1**
    - Urza's Tower: `Count$UrzaLands.3.1`  → **hi = 3, lo = 1**
  - `AI:`/`DeckNeeds:`/`DeckHints:` — cosmetic AI/deckbuilding hints (ignored).

## Engine work (CR 305 lands, CR 605 mana abilities)
- **`Count$UrzaLands.<hi>.<lo>` dynamic amount — new general handler.** Added to the shared
  runtime-amount evaluator `evaluate_dynamic_amount` (`src/components/ability.cpp`, immediately
  after the analogous `Count$Threshold.<hi>.<lo>` branch). Parses the `.<hi>.<lo>` pair and
  returns `<hi>` when the ability's controller controls a *complete* Tron set — at least one
  Urza's Mine **and** one Urza's Power Plant **and** one Urza's Tower — else `<lo>`. The
  battlefield scan goes through the shared `battlefield_permanents(orderer->mEntities, ctrl)`
  accessor (`src/game_queries.h`), matching pieces by card **name** (the type-line subtype is
  hyphenated, the name is not; matching the name avoids that mismatch). Keyed on the tag's
  meaning, so any future card scaling a dynamic amount by Tron assembly reuses it.
- **Parser preservation.** `src/parse.cpp` (top-level `Amount$` SVar resolution) now also
  preserves a `Count$UrzaLands…` expression into `ability.dynamic_amount_expr` (alongside the
  existing `Count$Valid` / `Count$Revolt` / `Count$Threshold` / `xPaid` forms) so it survives to
  activation.
- **Mana ability scales via the existing fallback.** No new mana-system code was needed:
  `eval_mana_amount` (`src/mana_system.cpp`) already falls back to `evaluate_dynamic_amount` for
  any non-empty `dynamic_amount_expr` not matched by its source-aware `Count$Valid` branch (added
  for Cabal Ritual, commit 2170b6d). The Produced$ C makes the output colorless.

## Behavioral decisions (made in lieu of asking a human)
- **"Control" = the activating land's controller** (CR 109.5 / 605.3). `ctrl` flows from the
  permanent activating its mana ability; the Tron-set check counts only permanents that player
  controls, via `battlefield_permanents(..., ctrl)`.
- **Match by name, not subtype.** Forge's type-line subtype is `Urza's Power-Plant` (hyphenated)
  while the card name is `Urza's Power Plant`. The handler matches the card *name* so the three
  pieces are identified unambiguously regardless of the hyphenation quirk.
- **Phasing already handled.** Using the shared battlefield accessor means a phased-out Tron piece
  correctly does not count toward the set (CR 702.26e) without a call-site `is_phased_out` check.

## Tests (test_harness, inline hand + preset battlefield, seed 1)
- **Assembled (hi):** `--battlefield-a "Urza's Mine,Urza's Power Plant,Urza's Tower"`, cast
  Paradox Engine ({5}). Narrative: `activated Urza's Mine for 2(C)`, `Urza's Power Plant for
  2(C)`, `Urza's Tower for 3(C)` → 7 total; Paradox Engine resolves and `2C` remain in pool
  (7 − 5). Confirms hi = 2/2/3 and colorless. PASS.
- **Single piece (lo):** `--battlefield-a "Urza's Tower"`, cast Pithing Needle ({1}). Narrative:
  `activated Urza's Tower for 1(C)`; Pithing Needle resolves with an empty pool afterward.
  Confirms lo = 1 and colorless. PASS.
- No draws, no fatal or non-fatal errors in either run.

## Result
implemented
