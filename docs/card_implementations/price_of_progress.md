# Price of Progress  (vocab index 115)

## Oracle text
Price of Progress deals damage to each player equal to twice the number of nonbasic lands
that player controls.

(Instant — mana cost {1}{R}.)

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/p/price_of_progress.txt`
- Key tags:
  - `A:SP$ RepeatEach | RepeatPlayers$ Player | RepeatSubAbility$ DBDamage | DamageMap$ True`
    — a spell ability that loops once per player, resolving the named sub-ability for each.
    `RepeatPlayers$ Player` = every player; `DamageMap$ True` is Forge's flag that the
    per-iteration damage is collected into one simultaneous damage event.
  - `SVar:DBDamage:DB$ DealDamage | Defined$ Remembered | NumDmg$ X` — the per-player body:
    deal `X` damage to the **remembered** player (the player the loop is currently on).
  - `SVar:X:Count$Valid Land.nonBasic+RememberedPlayerCtrl/Times.2` — the dynamic amount:
    count the nonbasic lands controlled by the remembered player, multiplied by 2.
  - `AI:RemoveDeck:Random` — Forge AI hint; ignored.

## Engine work
All changes are general (keyed on each tag's intended meaning), not card-specific.

- **`RepeatEach` over players — new effect (`src/effects/effect_repeat_each.cpp`,
  `EffectKind::RepeatEach`).** Added a general handler for `SP$/AB$ RepeatEach` with
  `RepeatPlayers$`. The handler iterates over the players in **APNAP order** (active player
  first, then non-active — CR 101.4 / 608.2g), and for each player: sets
  `cur_game.remembered_entities` to that player's entity, then resolves the parsed
  `RepeatSubAbility` (a copy per player) with `source`/`controller` propagated and, when the
  sub-ability is `Defined$ Remembered`, its target pinned to that player. It restores the
  prior remembered list afterward and returns `false` so the default single subability-chain
  is suppressed (the loop already resolved the body once per player).

- **`RepeatPlayers$` / `RepeatSubAbility$` parsing (`src/parse.cpp`).** `RepeatPlayers$` is
  stored in a new `Ability::repeat_players` field. `RepeatSubAbility$` is resolved exactly
  like `SubAbility$` — its value names an SVar holding a `DB$` ability, parsed into
  `subabilities` — so no separate sub-ability storage path is introduced.

- **`Count$Valid Land.nonBasic+RememberedPlayerCtrl[/Times.N]` evaluation
  (`src/components/ability.cpp::evaluate_dynamic_amount`).** Added a branch that counts live
  battlefield nonbasic lands controlled by the **remembered player**
  (`cur_game.remembered_entities[0]`, set by the RepeatEach loop), using the shared
  `is_battlefield_permanent(e, ctrl)` accessor and `has_basic_supertype()` to exclude basics,
  then multiplies by the `/Times.N` factor. The existing `Count$Valid …` parser branch
  already routes the SVar to `dynamic_amount_expr`, and the `DealDamage` handler already
  evaluates `dynamic_amount_expr` at resolution — so the damage amount is recomputed per
  player automatically with no DealDamage-specific change.

- **`DamageMap$` ignored (documented).** Added to the parser's `ignored_keys` set: for a
  one-shot instant the per-player damage is dealt in the loop and no player can respond
  between the two amounts, so collecting them into one simultaneous map is cosmetic.

## Behavioral decisions (made in lieu of asking a human)
- **Per-player sequential damage is observationally identical to the simultaneous event.**
  CR 608.2 has the spell deal damage to both players as one event. Resolving the two amounts
  back-to-back inside the RepeatEach loop produces the same result for a one-shot instant:
  there is no priority window between them, and neither player's damage depends on the other's
  outcome. The amount for each player is computed independently from that player's own nonbasic
  land count.
- **Basic lands deal 0.** The `nonBasic` filter excludes any land with the Basic supertype
  (the six basics, dual-typed basics, etc.), matched via `has_basic_supertype()` — consistent
  with the engine's other nonbasic-land filters.
- **Both players are affected, including the caster.** "each player" is `RepeatPlayers$ Player`
  (all players), so the controller takes damage for their own nonbasics too.

## Tests
- Isolation (test_harness):
  - **Asymmetric counts (A casts):** A controls 2 Plateau + 1 Mountain, B controls 1 Plateau +
    2 Forest. A takes 4 (2×2), B takes 2 (1×2) — A→16, B→18. Basics dealt 0. No fizzle/error.
  - **All basics:** A controls 2 Mountain, B controls 3 Forest → 0 damage to each, both stay
    at 20.
  - **B casts, APNAP order (3 vs 2):** A (active player) controls 2 Plateau, B controls 3
    Plateau + 1 Mountain. Active player A is dealt first (4 damage → 16), then B (6 damage →
    14); the basic Mountain on B's side is excluded. Confirms the loop targets each player's
    own nonbasic count regardless of who cast it.
- Regression (test_harness --scripted, full games): deck = delver shell with 3× Price of
  Progress (in place of 3× Unholy Heat main) vs mav, seeds 1-6 — all six decisive (4 A wins,
  2 B wins), no draws, no max-decisions caps, no non-fatal errors. Price of Progress was cast
  in the games. The only warnings are the pre-existing cosmetic `Unrecognized ability param`
  lines for other cards.

## Result
implemented
