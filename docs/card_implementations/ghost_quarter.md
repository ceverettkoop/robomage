# Ghost Quarter  (vocab index 134)

## Oracle text
{T}: Add {C}.
{T}, Sacrifice Ghost Quarter: Destroy target land. Its controller may search their library for a
basic land card, put it onto the battlefield, then shuffle.

## Forge script
- Source: fetched (Forge@master) → `bin/resources/cardsfolder/g/ghost_quarter.txt`
- `Types:Land` / `ManaCost:no cost` — a colorless utility land.
- Key tags:
  - `A:AB$ Mana | Cost$ T | Produced$ C` — the {T}: Add {C} mana ability (identical to
    Wasteland's mana line).
  - `A:AB$ Destroy | Cost$ T Sac<1/CARDNAME> | ValidTgts$ Land | SubAbility$ DBChange` — the
    activated ability: tap and sacrifice Ghost Quarter to destroy a target land, then chain its
    sub-ability. `ValidTgts$ Land` allows **any** land (basic or nonbasic), unlike Wasteland's
    `ValidTgts$ Land.nonBasic`.
  - `SVar:DBChange:DB$ ChangeZone | Optional$ True | Origin$ Library | Destination$ Battlefield
    | ChangeType$ Land.Basic | DefinedPlayer$ TargetedController | ShuffleNonMandatory$ True` —
    the destroyed land's controller may search their own library for a basic land and put it onto
    the battlefield (untapped — **no** `Tapped$ True`), then shuffles.
  - `AILogic$ GhostQuarter`, `AITgts$ Land.nonBasic`, `AI:RemoveDeck:Random` — AI hints (ignored).
  - `ChangeTypeDesc$ basic land`, `ShuffleNonMandatory$ True` — cosmetic/display tags (ignored;
    they emit the pre-existing `WARNING: Unrecognized ability param`).

## Engine work
**None required** — Ghost Quarter is fully covered by existing handlers. It is the
activated-ability twin of **Erode** (vocab index 132), which already added the
`DefinedPlayer$ TargetedController` search redirect.

- `AB$ Mana | Produced$ C` → the existing mana-ability path (shared with Wasteland); offered when
  paying a cost, never goes on the stack.
- `AB$ Destroy | Cost$ T Sac<1/CARDNAME> | ValidTgts$ Land` → the existing tap-and-sacrifice-self
  activation cost (shared with Wasteland; `ability.sac_self` set in `src/parse.cpp` for
  `Sac<.../CARDNAME>`), the shared target-legality system (enforces `ValidTgts$ Land`), and
  `effects::destroy` (`src/effects/effect_destroy.cpp`) which moves the target to the graveyard.
  `destroy()` returns `true`, so the chained sub-ability runs.
- `SubAbility$ DBChange` on a top-level `A:` line → parsed by the existing `SubAbility$` branch
  into `ability.subabilities`; `Ability::resolve` (`src/components/ability.cpp:966`) propagates
  `this->target` and `this->controller` to the sub-ability before resolving it, so the
  `ChangeZone` knows which destroyed land's controller to act for. This propagation is identical
  for `AB$` (activated) and `SP$` (spell) parents — both produce an `Ability` with subabilities.
- `DefinedPlayer$ TargetedController` on the search `ChangeZone` → the existing redirect at the
  top of `effects::change_zone` (`src/effects/effect_change_zone.cpp:56-60`): when
  `ab.defined_targeted_controller` is set and the ability has a target, the searching/owning
  player `owner` is taken from the **target's** `Zone.controller` (which persists after the land
  is moved to the graveyard), not the spell's caster. This drives the library search
  (`search_zone`), the post-search shuffle, and the battlefield-entry controller assignment.
- `Origin$ Library | Destination$ Battlefield | ChangeType$ Land.Basic` (no `Tapped$`) → the
  existing fetch-land search path; because `ab.enters_tapped` is false (no `Tapped$ True`), the
  basic land enters **untapped** — the only behavioral difference from Erode, which fetches tapped.
- `Optional$ True` (and the absence of `Mandatory$`) → `ab.mandatory == false`, so `search_zone`
  offers a "Fail to find" (decline) option, satisfying the "may search".

## Behavioral decisions (made in lieu of asking a human)
- **"Its controller" = the controller, not the owner (CR 109.5 / 608.2g).** The redirect reads
  `Zone.controller` of the target (the land's last battlefield controller), so a stolen land's
  *current* controller — not its owner — gets to search. `add_to_zone` does not reset
  `Zone.controller` when the destroyed land moves to the graveyard, so the value is still valid
  during the sub-ability's resolution.
- **Order of operations (CR 608.2c).** The `Destroy` resolves first (the land is already in the
  graveyard) and then the `ChangeZone` sub-ability resolves, exactly as a single ability that
  "destroys, then [its controller] may search" should — both happen during Ghost Quarter's one
  resolution.
- **Search is a *may* (CR 701.18 / 701.18e).** `Optional$ True` keeps the search non-mandatory, so
  the controller may decline ("Fail to find"). The library is still shuffled afterward (the
  search occurred), matching Forge's `ShuffleNonMandatory$ True`.
- **Land enters untapped (CR 614 ETB).** Ghost Quarter has **no** `Tapped$ True`, so the fetched
  basic enters untapped under the destroyed land's controller — distinct from Erode (which uses
  `Tapped$ True`). The `change_zone` code only adds the entrant to `pending_enters_tapped` when
  `ab.enters_tapped` is true, so the default is untapped.
- **Any land is a legal target.** `ValidTgts$ Land` (not `Land.nonBasic`) lets Ghost Quarter
  destroy a basic land too, including the player's own; the shared target system enforces it.

## Tests
- Isolation (test_harness, preset battlefield, seed 1):
  - **Opponent's land, fetch a basic untapped**: A taps + sacrifices Ghost Quarter targeting B's
    Wasteland → "Player A sacrifices Ghost Quarter", "Wasteland is destroyed"; the search prompt
    was **Player B's** library; "Player B shuffles their library" → "Player B puts Island to the
    battlefield"; board showed `Opp BF: Island` with **no `(T)`** marker — the fetched basic
    entered **untapped** under B's control. PASS.
  - **Optional decline (fail to find)**: same line but B chooses "Fail to find" → "Wasteland is
    destroyed", "Player B shuffles their library", "Player B fails to find" (no land entered),
    confirming the search is a *may*. PASS.
  - **Mana ability**: `{T}: Add {C}` parses and is offered as a tappable mana source (shared with
    Wasteland's proven mana line). PASS.
- Regression (test_harness `--scripted`, 6 seeds, deck with 4 Ghost Quarter + Lightning Bolt +
  Grizzly Bears + Island/Mountain vs a Forest/Plains/Grizzly Bears deck): all 6 games decisive,
  no draws, no fatal or non-fatal errors (only the allowed cosmetic
  `ChangeTypeDesc$/ShuffleNonMandatory$/AITgts$` warnings). Ghost Quarter was observed firing in
  real games — "Player A played Ghost Quarter" → "Player A sacrifices Ghost Quarter" →
  ability on the stack targeting a land → resolved cleanly. PASS.

## Result
implemented
