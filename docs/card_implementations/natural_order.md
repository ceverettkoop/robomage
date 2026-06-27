# Natural Order (vocab index 162)

## Oracle text
As an additional cost to cast this spell, sacrifice a green creature.
Search your library for a green creature card, put it onto the battlefield, then shuffle.

(Sorcery — {2}{G}{G})

## Forge script (Source: pre-existing local — `bin/resources/cardsfolder/n/natural_order.txt`)
Key tags:
- `A:SP$ ChangeZone | Cost$ 2 G G Sac<1/Creature.Green/green creature> | Origin$ Library | Destination$ Battlefield | ChangeType$ Creature.Green | ChangeNum$ 1`
- The `Cost$` carries both the mana cost (`2 G G`) and the additional sacrifice cost token
  `Sac<1/Creature.Green/...>` (qty `1`, filter `Creature.Green`, with a display label that the
  parser strips).
- The `AILogic$`/`AISearchGoal$`/`AIPreference` fields are AI hints only and are ignored (the
  parser already emits a harmless "Unrecognized ability param: AISearchGoal$" note — no behavior
  depends on it).

The search-creature-to-battlefield half of the effect (`SP$ ChangeZone Origin$ Library
Destination$ Battlefield ChangeType$ Creature.Green`) was already proven by Green Sun's Zenith;
the ChangeZone library-search handler shuffles the library after the search, satisfying "then
shuffle". No new ChangeZone work was required.

## Engine work — general additional-Sac-cost on spell cast
The gap was that the `CAST_SPELL` path paid only mana / flashback / alt / X / phyrexian costs; it
never paid the SP$ line's *additional* `Sac<...>` cost. Implemented generally (any spell whose
SPELL ability `Cost$` contains a `Sac<qty/filter>` token), reusing the activated-ability sacrifice
machinery (SACRIFICE_PERMANENT) rather than adding a Natural-Order special case.

Files / functions:
- `src/parse.cpp` — unchanged for this card: `parse_activation_cost` (already shared by `Cost$`)
  parses the `Sac<...>` token into the SPELL ability's `Ability::sac_cost_spec` (= `Creature.Green`).
  The card's real `Cost$` tag is honored as-is (no retag).
- `src/game_queries.h`
  - `spell_additional_sac_spec(const CardData&)` — new helper returning the SPELL ability's
    `sac_cost_spec` (empty when none). Single source consumed by both legality and payment.
  - `permanent_matches_subtype_spec(...)` — extended to honor a `.<Color>` / `.non<Color>`
    qualifier in a `Sac<.../Type.Color>` spec by testing the permanent's `effective_colors`
    via the existing `color_set_passes` / `color_set_passes_noncolor` matchers (previously only
    the `.Other` self-exclusion qualifier was handled; a color qualifier was silently ignored,
    which would have let a non-green creature pay the cost). General to any color-qualified
    sac/return cost; reuses the same color filter used by targeting.
- `src/systems/state_manager_actions.cpp` — cast legality: a regular cast is illegal unless a
  permanent matching `spell_additional_sac_spec` is available to sacrifice (mirrors the existing
  flashback `sac_cost_spec` legality gate).
- `src/action_processor.cpp` — `CAST_SPELL` regular-cost branch: after mana is paid, pay the
  additional sacrifice cost using `controlled_permanents_matching` + `prompt_permanent_choice`
  (SACRIFICE_PERMANENT), exactly as the flashback-sac and activated-ability-sac paths do. Paid as
  part of casting, before the spell goes on the stack. Flashback / alternate casts pay their own
  sac cost in their own branches, so there is no double charge.

## Behavioral decisions (Comprehensive Rules)
- CR 601.2f / 601.2g — additional costs (the sacrifice) are determined and then paid while the
  spell is being cast, together with the mana cost, before the spell is put on the stack.
- CR 601.2f — if the total cost can't be paid (no green creature to sacrifice), the spell can't
  be cast: it is excluded from the legal-action list.
- CR 118.x — sacrificing a permanent as a cost. A tapped creature is still a legal sacrifice (no
  untapped requirement); the test confirms Birds of Paradise can be tapped for mana and still
  sacrificed.

## Tests (`train/test_harness.py`, semantic `--play`)
- (a) Green creature (Birds of Paradise) on battlefield + {2}{G}{G} available, Scythecat Cub in
  library: casting Natural Order prompts "Sacrifice Birds of Paradise" (the additional cost),
  sacrifices it, then searches the library for a green creature (Scythecat Cub), puts it onto the
  battlefield, and shuffles. PASS.
- (b) No green creature to sacrifice (only Forests on battlefield): Natural Order is **not** in
  the legal-action list — the cost cannot be paid. PASS.
- (c) Non-green creature only (White Orchid Phantom): Natural Order is still **not** castable,
  proving the `.Green` color qualifier is enforced. PASS.
- Regression: scripted vs scripted full games (green deck with Natural Order + Birds / Scythecat
  Cub / Endurance vs Maverick), seeds 1/2/3 — all complete with a decisive result, no draws, no
  non-fatal errors. Natural Order is cast and resolves in-game (seed 3).

## Result
Implemented. General additional-sacrifice-cost-on-cast support added; Natural Order casts,
pays its sacrifice, and fetches a green creature correctly. Built clean (`make HEADLESS=TRUE`).
