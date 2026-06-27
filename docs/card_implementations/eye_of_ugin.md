# Eye of Ugin (vocab index 160)

## Oracle text
Colorless Eldrazi spells you cast cost {2} less to cast.
{7}, {T}: Search your library for a colorless creature card, reveal it, put it into
your hand, then shuffle.

## Forge script (Source: pre-existing local `bin/resources/cardsfolder/e/eye_of_ugin.txt`)
Key tags:
- `S:Mode$ ReduceCost | ValidCard$ Eldrazi.Colorless | Type$ Spell | Activator$ You | Amount$ 2`
  — the continuous cost-reduction static (the gap implemented here).
- `A:AB$ ChangeZone | Cost$ 7 T | Origin$ Library | Destination$ Hand |
  ChangeType$ Creature.Colorless | ChangeNum$ 1` — the `{7}, {T}` library tutor.
- `Types:Legendary Land`, `ManaCost:no cost`.
- `ChangeTypeDesc$`, `AI:`, `DeckNeeds$`/`DeckHints$` — cosmetic/AI metadata (ignored; the
  parser emits a benign "Unrecognized ability param: ChangeTypeDesc$" warning for the label).

## Engine work
The activated `ChangeZone` tutor was already covered by the generic `ChangeZone` handler (it
honors `ChangeType$ Creature.Colorless` via `matches_filter_spec`); verified by test below.

The gap was that the engine had **no `ReduceCost` continuous cost-reduction static** — only
`RaiseCost`. Added a general, reusable `ReduceCost` static as the mirror of `RaiseCost`:

- `src/components/static_ability.h`: new fields `int reduce_cost`, `std::string
  reduce_cost_filter` (the full `ValidCard$` spec), and `bool reduce_cost_you_only`
  (`Activator$ You` gate).
- `src/parse.cpp` (`parse_static_abilities`): the `Amount$` key now sets `reduce_cost` when the
  static's category is `ReduceCost` (else `raise_cost`, unchanged); a new `Activator$ You`
  branch sets `reduce_cost_you_only`; the `ValidCard$` key stores the full filter spec for
  `ReduceCost`. No tag is retagged — the script's real `Mode$ ReduceCost` is honored.
- `src/systems/state_manager_statics.cpp`:
  - `card_matches_reduce_filter(CardData, filter)` — a general `Head.Qualifier` matcher
    (head = type/subtype name, with `Card`/`Permanent`/`Spell` wildcards; qualifier =
    `Colorless`/`Creature`/`nonCreature`/a color/a subtype). Handles `Eldrazi.Colorless`
    and any similar filter.
  - `active_reduce_cost_for(CardData, caster)` — the mirror of `active_raise_cost_for`: sums
    the `Amount$` of every active `ReduceCost` static whose filter matches the spell, gating
    `Activator$ You` statics to spells cast by the static's controller.
  - `effective_base_cost(...)` — after folding in the `RaiseCost` additions, removes
    `active_reduce_cost_for` generic ({1}) pips, reusing the exact clamp pattern already used
    for Affinity (only generic pips removed, never below 0). This is the single
    effective-base-cost builder shared by legality (`determine_legal_actions`) and payment
    (`action_processor`), so affordability and payment agree.
- `src/components/ability.cpp` (`matches_filter_spec`): added a general `Colorless` qualifier
  case (`is_colorless_card`, CR 105.2c) so dotted filters like `Eldrazi.Colorless` resolve
  colorlessness rather than mis-matching it as a subtype name.
- `src/systems/state_manager.h`: declared `active_reduce_cost_for` next to
  `active_raise_cost_for`.

## Behavioral decisions (CR cites)
- Cost reductions reduce only the **generic** portion of a cost and never a colored/colorless
  pip, and are applied **after** cost increases (CR 118.7, 601.2f). Implemented by erasing
  `GENERIC` entries only and clamping at zero — Glaring Fleshraker's `{2}{C}` becomes `{C}`
  (the `{C}` pip is untouched; it still needs colorless mana).
- `Activator$ You`: the reduction applies only to spells the source's controller casts
  (`reduce_cost_you_only` + `as.controller == caster`).
- The reduction is caster-relative, so it is only folded in when `effective_base_cost` is
  called with a known `caster` (both real call sites pass the casting player).

## Tests (`train/test_harness.py`, scripted + semantic `--play`)
- **Positive reduction:** Eye of Ugin + one Eldrazi Temple ({C} only) on the battlefield;
  Glaring Fleshraker ({2}{C}) in hand. Cast succeeds tapping just the single `{C}` source —
  the {2} generic was reduced to 0; the `{C}` pip was still paid (not reduced).
- **Control:** same board *without* Eye of Ugin → "Cast Glaring Fleshraker" is not offered off
  one Eldrazi Temple (needs the full {2}{C}). Confirms the reduction is what enables the cast.
- **Negative filter:** Eye of Ugin + one Swamp; Barrowgoyf ({2}{B}, non-Eldrazi) in hand →
  not castable off one Swamp (its cost is unchanged). The filter correctly excludes
  non-Eldrazi spells.
- **Activated tutor:** Eye of Ugin + 7 Forests; `{7}, {T}` activated → Eye and all 7 Forests
  tapped (cost paid), Glaring Fleshraker revealed and put to hand, library shuffled.
- **Regression:** scripted full games, deck with Eye of Ugin + Eldrazi Temple + colorless
  Eldrazi creatures vs a burn/lands deck, seeds 1/2/3 — all decisive (no draws), no non-fatal
  errors.

## Result
General `ReduceCost` cost-reduction static implemented as the mirror of `RaiseCost`; Eye of
Ugin's {2}-less reduction and {7},{T} tutor both verified. Build clean (`make HEADLESS=TRUE`).
