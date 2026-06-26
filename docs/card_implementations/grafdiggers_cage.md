# Grafdigger's Cage  (vocab index 119)

## Oracle text
Creature cards in graveyards and libraries can't enter the battlefield.

Players can't cast spells from graveyards or libraries.

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/g/grafdiggers_cage.txt`
- Type: `Artifact`, mana cost `1`.
- Key tags:
  - `R:Event$ Moved | ActiveZones$ Battlefield | Origin$ Graveyard,Library | Destination$ Battlefield | ValidLKI$ Creature.Other | Prevent$ True | Layer$ CantHappen`
    — the prevention: a creature card moving from a graveyard or library onto the battlefield
    simply doesn't enter (`Prevent$ True` / `Layer$ CantHappen`). This is a *prevention*, not an
    exile-redirect — the card stays in its origin zone.
  - `S:Mode$ CantBeCast | Origin$ Graveyard,Library`
    — the cast restriction: no player can cast a spell from a graveyard or library (e.g.
    flashback, or "cast from library" effects).

## Engine work
Two distinct static abilities, each implemented as a real general mechanic keyed on the script
tag's intended meaning (not a card-specific shortcut).

### (a) Prevent-ETB-from-zones replacement (CR 614.13)
The existing replacement infrastructure (`src/systems/replacement_effects.{h,cpp}`) handled
`ENTERS_BATTLEFIELD` (enters-tapped / +1/+1 counters / Containment-Priest exile-instead) and
`MOVE_TO_ZONE` (Dauthi/Leyline graveyard→exile redirect). Neither tracked the **origin** zone of a
battlefield-bound move, and there was no "the move doesn't happen at all" outcome. Both were added
as general mechanics:

- `Effect::Replacement::PREVENT_ETB_FROM_ZONES` kind with `prevent_from_graveyard` /
  `prevent_from_library` flags (`src/components/effect.h`).
- `src/parse.cpp` parses the Cage R: line into that kind, keyed on `Event$ Moved` +
  `Destination$ Battlefield` + `Prevent$ True` + `ValidLKI$ Creature*` + `ActiveZones$ Battlefield`
  + an `Origin$` containing Graveyard and/or Library. This detection does not collide with the
  existing R: parse branches (ENTERS_TAPPED needs `Card.Self` + `ETBTapped`; Containment Priest
  needs `ReplaceWith$ Exile` + the uncast-creature filter; Dauthi/Leyline need
  `Destination$ Graveyard`).
- `ReplacementEvent` gained an `origin` input field and a `prevented` outcome field
  (`src/systems/replacement_effects.h`).
- `src/systems/orderer.cpp` (`add_to_zone`): the single chokepoint for every zone move now seeds
  `rev.origin` into the `MOVE_TO_ZONE` dispatch and, if `rev.prevented` comes back set, returns
  early — the card never leaves its origin zone. Because *all* battlefield entries
  (reanimation, Green Sun's Zenith search-to-battlefield, etc.) funnel through `add_to_zone`, this
  one guard covers every "from graveyard/library" entry path.
- `src/systems/replacement_effects.cpp` (`collect` / `apply_one`): on a `MOVE_TO_ZONE` event whose
  destination is the battlefield and whose origin is graveyard or library, if the moving object is
  a real card (has `CardData`, not a `Token`) and is a creature, it scans the battlefield for any
  permanent carrying a matching `PREVENT_ETB_FROM_ZONES` replacement and offers it as a candidate
  (CR 616.1). Applying it sets `prevented` and logs "… can't enter the battlefield from that zone."

### (b) Cast-from-zone restriction (CR 601.3)
The `CantBeCast` static (`rules_mod::cast_prohibited`) already enforced the nonCreature/per-turn
limit (Deafening Silence) and the opponent-lock (Voice of Victory). It is now origin-aware:

- `StaticAbility::cant_cast_from_graveyard` / `cant_cast_from_library` flags
  (`src/components/static_ability.h`), parsed from `Origin$ Graveyard,Library` on a `CantBeCast`
  static (`src/parse.cpp`).
- `rules_mod::cast_prohibited` gained a `cast_from` zone argument (default `HAND`)
  (`src/systems/rules_modifying.{h,cpp}`). When a static carries an origin restriction it applies
  **only** when the spell is cast from a matching zone; a normal hand cast (or any other zone) is
  unaffected by it.
- `src/systems/state_manager_actions.cpp`: the hand-cast site keeps the default (HAND, unaffected);
  the flashback (graveyard-cast) site now calls `cast_prohibited(..., Zone::GRAVEYARD)`, so a
  graveyard-cast restriction suppresses the "Cast … (flashback)" action.

## Behavioral decisions (made in lieu of asking a human)
- **Prevention, not exile.** Per the script (`Prevent$ True` / `Layer$ CantHappen`) and CR 614.13,
  a creature card that would enter the battlefield from a graveyard/library simply doesn't enter —
  it remains in its origin zone (distinct from Containment Priest, which exiles). Verified: the
  fetched creature is neither on the battlefield, in hand, nor in the graveyard afterward.
- **Controller-agnostic.** The prevention and the cast restriction apply to every player ("Players
  can't cast …", "Creature cards in graveyards and libraries can't enter") — the candidate scan and
  the cast check do not filter by controller.
- **Creatures only for the ETB clause** (`ValidLKI$ Creature.Other`): a non-creature card put onto
  the battlefield from a graveyard/library (e.g. a reanimated artifact/enchantment) is unaffected.
  The cast clause has no type filter and applies to spells of any type.
- **Hand casts and hand ETBs are untouched.** A creature hard-cast from hand resolves and enters
  normally; the cast restriction's origin gate means hand casts are never blocked.

## Tests
Isolation (`train/test_harness.py`, pre-set battlefields / stacked temp decks, `--no-shuffle`):
- **(a) ETB-from-library prevented.** A controls `Grafdigger's Cage` + Forests, casts
  `Green Sun's Zenith` for X=1 fetching `Birds of Paradise` → **"Birds of Paradise can't enter the
  battlefield from that zone."** Birds is not on the battlefield, not in hand, not in the graveyard
  (it stayed in the library). PASS.
  - Control (no Cage): same line → "Player A puts Birds of Paradise to the battlefield"; Birds
    appears on A's battlefield. PASS — confirms the prevention is caused by the Cage.
- **(b) Cast-from-graveyard (flashback) prohibited.** A casts `Deep Analysis` from hand (resolves,
  goes to graveyard); with `Grafdigger's Cage` in play the **"Cast Deep Analysis (flashback)"
  action never appears**. PASS.
  - Control (no Cage): the same setup *does* offer "Cast Deep Analysis (flashback)" once it is in
    the graveyard and 1U + 3 life are available. PASS — confirms the suppression is caused by the
    Cage.
- **(c) Hand cast/ETB unaffected.** A controls `Grafdigger's Cage` + Forest, hard-casts
  `Birds of Paradise` from hand → it enters the battlefield normally (no "can't enter" message),
  appearing on A's battlefield. PASS.

Regression (`train/test_harness.py --scripted`, 6 games, seeds 1–6): deck `temp/cage_regress`
(4 Grafdigger's Cage, 2 Green Sun's Zenith, 4 Birds of Paradise, 4 Grizzly Bears, 4 Noble Hierarch,
4 Scythecat Cub, 2 Endurance, GW lands) vs a RG `temp/cage_opp` deck. All 6 games finished
decisively (Player A won each), no draws, no fatal/non-fatal errors, no asserts/tracebacks, no
max-decision hangs. Grafdigger's Cage and Green Sun's Zenith were drawn and resolved in real games
with the engine stable. (Only the pre-existing cosmetic `WARNING: Unrecognized ability param:
AIXMax$ Y` for Green Sun's Zenith appeared.)

## Result
implemented
