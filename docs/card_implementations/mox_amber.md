# Mox Amber  (vocab index 203)

## Oracle text
{T}: Add one mana of any color among colors of legendary creatures and planeswalkers you control.

(Legendary Artifact, mana cost {0}.)

## Forge script
- Source: pre-existing local script — `bin/resources/cardsfolder/m/mox_amber.txt` (not edited).
- The ability line:
  `A:AB$ ManaReflected | Cost$ T | ColorOrType$ Color | Valid$ Creature.Legendary+YouCtrl,Planeswalker.Legendary+YouCtrl | ReflectProperty$ Is | SpellDescription$ ...`
- Key tags:
  - `AB$ ManaReflected` → the NEW category: a mana ability (CR 605) whose producible colors are
    *reflected* from a set of permanents rather than fixed. Resolved off-stack at activation like
    any other mana ability — it never uses the stack.
  - `Cost$ T` → tap cost.
  - `Valid$ Creature.Legendary+YouCtrl,Planeswalker.Legendary+YouCtrl` → the (controller-scoped)
    permanent filter whose colors are reflected. The comma is Forge's OR separator for a `Valid$`
    list.
  - `ColorOrType$ Color` + `ReflectProperty$ Is` → "reflect the colors the matching permanents
    actually *are*" (the implemented and only mode for this card). Validated/consumed by the parse
    hook, not warned as unrecognized.
- Tags parsed as written; **no category was retagged** (ManaReflected stays ManaReflected — it is
  not rewritten to `AddMana Combo`).

## Engine work (general, keyed on each tag's intended meaning)

### General `AB$ ManaReflected` mana ability (reusable)
A mana ability whose producible color set is the **union of the colors of the `Valid$`-matching
permanents** the controller controls, computed live from the board each time mana sources are
enumerated. The player then chooses one of those colors (the existing per-color expansion path,
which surfaces as a `CHOOSE_MANA_COLOR`-style choice for any-color producers). An empty color set
(no legendary creature/planeswalker, or only colorless ones) means the ability can produce
nothing, so it is not offered as a usable mana source.

This is implemented as a first-class mana-ability category, sharing every existing mana code path
(off-stack resolution, affordability gate, auto-payer, legal-action enumeration) with ordinary
`AddMana` producers — so it can never diverge from them.

- **`Ability::reflected_mana_filter`** (`src/components/ability.h`) — the stored `Valid$` spec;
  non-empty marks the ability as a reflected-mana ability. Also compared in
  `Ability::operator==` (`src/components/ability.cpp`).
- **Parse** (`src/effects/effect_add_mana.cpp`, `parse_add_mana`) — for `category == "ManaReflected"`
  consumes `Valid$` into `reflected_mana_filter`, defaults `amount` to 1 ("Add **one** mana"), and
  accepts `ColorOrType$`/`ReflectProperty$` (the Color/Is mode). The category string is left as
  `ManaReflected`.
- **`ability_is_mana(const Ability&)`** (`src/mana_system.h` / `src/mana_system.cpp`) — single
  predicate for "is this an off-stack mana ability", returning true for `AddMana` **and**
  `ManaReflected`. Replaces the open-coded `category == "AddMana"` checks at the three
  classification sites: `collect_available_mana_sources` (mana-source enumeration),
  `process_activate_ability`'s `is_mana_ability` (action processor — never pushes it on the stack),
  and the legal-action enumeration in `state_manager_actions.cpp` (mana abilities are collected
  separately and never go on the stack).
- **`reflected_color_set(...)`** (`src/mana_system.cpp`) — the dynamic color set: scans
  `battlefield_permanents(..., player)`, matches each against the `Valid$` filter via the shared
  `permanent_matches_filter` (with the `MatchCtx` controller = the activator, so `YouCtrl` resolves
  correctly), and unions each match's `effective_colors` (CR 105.2c: colorless contributes no
  color). The filter's comma-OR alternatives are normalized to the matcher's `;`-OR form locally.
  Returned in WUBRG order for a stable choice menu.
- **Per-color expansion** (`collect_available_mana_sources`) — a reflected-mana ability is expanded
  into one selectable source per producible color (each a copy with `color` set), mirroring the
  pre-existing `mana_choices` (Birds of Paradise / Mox Opal "any color") expansion. Empty set ⇒ no
  sources offered.

## Rules references
- **CR 605** — mana abilities don't use the stack and resolve immediately.
- **CR 105.2 / 202.2** — an object's color is determined by its mana cost (or color indicator); a
  colorless object has no color.
- **CR 704.5j** — the legend rule (Mox Amber is Legendary; unchanged behavior, confirmed in test).

## Tests (test harness, `--play` isolation; comma-named legendaries placed via stacked deck files)
1. **Single color (black).** Mox Amber + Mai, Scornful Striker (mono-black legendary creature) in
   play, Swamps tapped out by casting Mai → casting Thoughtseize ({B}) auto-taps **Mox Amber for
   1(B)**; Thoughtseize resolves (Player A loses 2 life, Player B reveals hand). Proves Mox reflects
   black from the legendary.
2. **Empty set.** Only Mox Amber (a Legendary *Artifact*, not a creature/planeswalker) + Mountains
   in play. Consider ({U}) is **not** castable (Mox's reflected set is empty), the action menu omits
   it, and there is no crash — the game continues normally.
3. **Second color (blue).** Mox Amber + Emry, Lurker of the Loch (mono-blue legendary creature) in
   play, Islands tapped out casting Emry → casting Consider ({U}) auto-taps **Mox Amber for 1(U)**.
   Together with test 1, demonstrates the producible set is the *union* over differently-colored
   legendaries.
4. **Legend rule intact.** Casting a second Mox Amber fires the legend rule ("choose one to keep");
   the extra copy is put into the graveyard.
- Scripted full game with Mox-bearing decks ran to a clean result (Player B decked) with no
  non-fatal errors / asserts / draws.
