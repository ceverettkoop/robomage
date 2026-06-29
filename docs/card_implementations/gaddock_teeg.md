# Gaddock Teeg  (vocab index 247)

## Oracle text
Noncreature spells with mana value 4 or greater can't be cast.
Noncreature spells with {X} in their mana costs can't be cast.

(Legendary Creature — Kithkin Advisor, mana cost {G}{W}, 2/2.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/g/gaddock_teeg.txt`
- Key tags (two static prohibitions):
  - `S:Mode$ CantBeCast | ValidCard$ Card.nonCreature+cmcGE4`
  - `S:Mode$ CantBeCast | ValidCard$ Card.nonCreature+hasXCost`

## Engine work
The `CantBeCast` static was parsed and gathered, but `rules_mod::cast_prohibited` only enforced
the special-cased forms (origin-restriction, opponent-lock, per-turn limit). A flat,
characteristic-keyed prohibition (Gaddock Teeg) was silently not enforced, and the `cmcGE4` /
`hasXCost` qualifiers weren't evaluated against a spell. Added a real, general handler:

- **`rules_mod::cast_prohibited` now takes the spell's `const CardData&`**
  (`src/systems/rules_modifying.{h,cpp}`) instead of just a `card_is_creature` bool, so it can
  read the spell's printed characteristics. After the existing special branches (which each now
  `continue` once handled, so they never fall through), a final **characteristic-based branch**
  matches the static's full `ValidCard$` filter against the spell via the shared
  `card_matches_filter` + `extract_static_cmc_bound` machinery in `game_queries.h`. This makes the
  `CantBeCast` static honor ANY characteristic filter (`cmcGE4`, `nonCreature`, color, type, …),
  not just Gaddock Teeg's two. Applied at all five cast sites (hand, flashback, escape,
  cast-from-graveyard, impulse) in `src/systems/state_manager_actions.cpp`.
- **`hasXCost` qualifier** added to the general filter evaluator
  (`src/game_queries.cpp`): `CharView` gained a `has_x_cost` field (populated in `card_view` /
  `permanent_view` from `CardData::has_x_cost`), and `eval_qualifier` returns it for the
  `hasXCost` token. Reusable by any future "{X} in mana cost" filter.
- Mechanics added (general): characteristic-based `CantBeCast` enforcement; `hasXCost` filter
  qualifier.

## Behavioral decisions
- **Symmetric lock (CR 611/614 continuous prohibition).** "Can't be cast" applies to every player
  including Gaddock Teeg's controller, so the new branch returns true regardless of caster. CR
  601.2/601.3e: a spell that can't be cast is never a legal action.
- **Mana value treats {X} as 0** (CR 202.3b): the engine's cmc (`mana_cost.size()`) does not
  count X, so the cmcGE4 clause correctly ignores X spells (those are caught by clause 2 only if
  X is in the *mana* cost). A card whose only "X" is a life/additional cost (Toxic Deluge, `{2}{B}`
  + PayLife<X>) has no {X} in its mana cost and is correctly NOT prohibited.

## Tests
- Isolation (test_harness), Gaddock Teeg on the battlefield, all spells otherwise affordable:
  - Cast menu offered **only** Lightning Bolt (mv1 noncreature) and Grizzly Bears (creature).
  - **Leyline of the Void** (mv4 noncreature, no target) — NOT offered (clause 1). PASS.
  - **Green Sun's Zenith** (`{X}{G}`, noncreature) — NOT offered (clause 2). PASS.
  - Control (same board, no Gaddock Teeg): all four — Lightning Bolt, Grizzly Bears, Leyline of the
    Void, Green Sun's Zenith — were offered, confirming the two are blocked specifically by Teeg.
- Regression (test_harness --scripted, full games): GW Gaddock Teeg deck (Teeg, Grizzly Bears,
  Lightning Bolt, Leyline of the Void) vs a green/red deck (Green Sun's Zenith, Grizzly Bears,
  Lightning Bolt), seeds 1-3 — all decisive (2 A wins, 1 B win), no draws, no non-fatal errors, no
  unrecognized-filter-qualifier warnings.

## Result
implemented
