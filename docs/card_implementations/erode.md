# Erode  (vocab index 132)

## Oracle text
Destroy target creature or planeswalker. Its controller may search their library for a basic
land card, put it onto the battlefield tapped, then shuffle.

## Forge script
- Source: fetched (Forge@master) → `bin/resources/cardsfolder/e/erode.txt`
- `ManaCost:W` / `Types:Instant` — a one-mana white instant.
- Key tags:
  - `A:SP$ Destroy | ValidTgts$ Creature,Planeswalker | SubAbility$ DBChangeZone` — the main
    spell ability destroys the chosen target, then chains its sub-ability.
  - `SVar:DBChangeZone:DB$ ChangeZone | Optional$ True | Origin$ Library | Destination$ Battlefield
    | Tapped$ True | ChangeType$ Land.Basic | DefinedPlayer$ TargetedController
    | ShuffleNonMandatory$ True | StackDescription$ None` — the targeted creature/planeswalker's
    controller may search their own library for a basic land and put it onto the battlefield
    tapped, then shuffles.
  - `ValidTgtsDesc$ creature or planeswalker`, `ChangeTypeDesc$ basic land`,
    `ShuffleNonMandatory$ True`, `StackDescription$ None` — cosmetic/display tags (ignored; the
    first three emit the pre-existing `WARNING: Unrecognized ability param`).

## Engine work
- **`DefinedPlayer$ TargetedController` on a search-based `ChangeZone`** — *new general handler.*
  The `defined_targeted_controller` flag was already parsed (`src/parse.cpp:836`) and honored by
  the same-name `ChangeZone` path, but the ordinary search-based `change_zone()` always searched
  and gave the fetched card to the *caster*. Added a small redirect at the top of
  `effects::change_zone` (`src/effects/effect_change_zone.cpp`): when
  `ab.defined_targeted_controller` is set and the ability has a target, the searching/owning
  player `owner` is taken from the **target's** `Zone.controller` instead of the spell's caster.
  This drives the library search (`search_zone`), the post-search shuffle, and the
  battlefield-entry controller assignment, so the basic land is fetched from and enters under the
  destroyed permanent's controller's account. The redirect only affects the search path; the
  targeted-move branch is skipped here because the sub-ability carries no target of its own
  (`ValidTgts$ N_A`).
- Everything else reuses existing handlers:
  - `SP$ Destroy` with `ValidTgts$ Creature,Planeswalker` → `effects::destroy` + existing target
    selection/legality.
  - `SubAbility$ DBChangeZone` on a top-level `A:` line → parsed by the existing
    `SubAbility$` branch (`src/parse.cpp:1185`) into `ability.subabilities`; `Ability::resolve`
    propagates `this->target` to the sub-ability (`src/components/ability.cpp:960`), so the
    `ChangeZone` knows which destroyed permanent's controller to act for.
  - `Origin$ Library | Destination$ Battlefield | ChangeType$ Land.Basic | Tapped$ True` →
    the existing fetch-land search path (shared with Edge of Autumn), including the
    `pending_enters_tapped` ETB-tapped mechanism.
  - `Optional$ True` — the search is non-mandatory by default (`ab.mandatory == false`), so
    `search_zone` already offers a "Fail to find" (decline) option, satisfying the "may search".

## Behavioral decisions (made in lieu of asking a human)
- **"Its controller" = the controller, not the owner (CR 109.5 / 608.2g).** Erode says *its
  controller* searches. The redirect reads `Zone.controller` of the target (the permanent's last
  battlefield controller), not `Zone.owner`, so a stolen creature's *current* controller — not
  its owner — gets the land. `add_to_zone` does not reset `Zone.controller` when the destroyed
  permanent moves to the graveyard, so the value is still valid during the sub-ability resolution.
- **Order of operations (CR 608.2c).** The `Destroy` resolves first (the permanent is already in
  the graveyard) and then the `ChangeZone` sub-ability resolves, exactly as a single spell that
  "destroys, then [its controller] may search" should — both happen during Erode's one resolution.
- **Land enters tapped (CR 614 / replacement-style ETB-tapped).** `Tapped$ True` routes through
  the existing `pending_enters_tapped` set, so the fetched basic enters tapped under the new
  controller.
- **Planeswalker target.** `ValidTgts$ Creature,Planeswalker` parses and is enforced by the
  shared target-legality system; no planeswalker-specific work is needed for Erode (a destroyed
  planeswalker takes the same destroy → graveyard path; its controller then fetches).

## Tests
- Isolation (test_harness, inline hands / preset battlefield, seed 1):
  - **Opponent's creature, fetch a land**: A casts Erode on B's Grizzly Bears → "Grizzly Bears is
    destroyed"; the search prompt was **Player B's** library; "Player B shuffles their library" →
    "Player B puts Island to the battlefield" → board showed `Opp BF: Island (T)` (entered tapped
    under B). PASS.
  - **Optional decline**: same line but B chooses "Fail to find" → creature destroyed, "Player B
    shuffles their library", "Player B fails to find" (no land), confirming the search is a *may*.
    PASS.
  - **Own creature (controller = caster)**: A casts Erode on A's own Grizzly Bears → "Searching
    Player A's library" — the redirect correctly routes the search to whoever controlled the
    destroyed permanent, including the caster. PASS.
- Regression (test_harness `--scripted`, 6 seeds, deck with 4 Erode + Grizzly Bears + Plains vs a
  Forest/Grizzly Bears deck): all 6 games decisive, no draws, no fatal or non-fatal errors (only
  the allowed cosmetic `ValidTgtsDesc$/ChangeTypeDesc$/ShuffleNonMandatory$` warnings). Erode was
  observed firing in real games — "Player A casts Erode targeting Grizzly Bears" → "Grizzly Bears
  is destroyed" → "Player B puts Forest to the battlefield". PASS.

## Result
implemented
