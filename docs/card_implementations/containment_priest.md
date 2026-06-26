# Containment Priest  (vocab index 113)

## Oracle text
Flash

If a nontoken creature would enter the battlefield and it wasn't cast, exile it instead.

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/c/containment_priest.txt`
- Type: `Creature Human Cleric`, mana cost `1 W`, P/T `2/2`.
- Key tags:
  - `K:Flash` — already supported by the engine.
  - `R:Event$ Moved | ActiveZones$ Battlefield | Destination$ Battlefield | ValidCard$ Creature.!token+!wasCast | ReplaceWith$ Exile`
    — the replacement effect: a non-token creature entering the battlefield that wasn't cast is
    exiled instead.
  - `SVar:Exile:DB$ ChangeZone | Hidden$ True | Origin$ All | Destination$ Exile | Defined$ ReplacedCard`
    — the zone-change effect referenced by `ReplaceWith$` (a plain exile, no void counter).

(The Forge text reads "2/2"; the prompt's "2/1" was a misstatement. The script is authoritative — 2/2.)

## Engine work
This is a new shape of ETB replacement (CR 614.1a — uses "instead"): the *entering* event of a
permanent is replaced by sending the card to exile. The existing replacement infrastructure
(`src/systems/replacement_effects.{h,cpp}`) already had an `ENTERS_BATTLEFIELD` event with
`enters_tapped` / `etb_p1p1` outcomes; it had no "the permanent doesn't enter at all" outcome, and
nothing tracked the "wasn't cast" condition. Both were added as general mechanics keyed on the tag's
intended meaning.

New mechanics (general, not card-specific):
- `Effect::Replacement::EXILE_INSTEAD_OF_ETB` kind (`src/components/effect.h`).
- `src/parse.cpp` parses the Containment Priest R: line into that kind, detecting the
  `ValidCard$ Creature.!token+!wasCast` filter (`Event$ Moved`, `Destination$ Battlefield`,
  `ReplaceWith$ Exile`, `ActiveZones$ Battlefield`). The detection is keyed on the filter's
  meaning (non-token creature that wasn't cast); it does not collide with the existing
  `ENTERS_TAPPED` (needs `ValidCard$ Card.Self` + `ReplaceWith$ ETBTapped`) or
  `EXILE_INSTEAD_OF_GRAVEYARD` (needs `Destination$ Graveyard` + `OppOwn`) parse branches.
- **"wasn't cast" tracking.** A creature spell that resolves from the stack onto the battlefield
  *was cast* (CR 614.12 / the spell's resolution). `src/systems/stack_manager.cpp` marks the
  resolving permanent in a new one-shot set `cur_game.cast_to_battlefield` at the exact point it
  moves a permanent spell from the stack to the battlefield. Every *other* battlefield-entry path
  (reanimation / search-to-battlefield via `ChangeZone`, `Defined$ Self`/`Remembered`, tokens) does
  **not** set the flag, so those are "not cast" — which is exactly the rule. The set is consumed
  (erased) when the permanent's `Permanent` component is created.
- `ReplacementEvent::redirect_to_exile` outcome field (`src/systems/replacement_effects.h`).
- `src/systems/replacement_effects.cpp` (`collect` / `apply_one`): on an `ENTERS_BATTLEFIELD`
  event, if the entering object is a real card (has `CardData`, not a `Token`), is a creature, and
  is **not** in `cast_to_battlefield`, it scans the battlefield for any other permanent carrying the
  `EXILE_INSTEAD_OF_ETB` replacement and offers it as a candidate (CR 616.1 — affected player chooses
  one when several apply). Applying it sets `redirect_to_exile` and logs "… is exiled instead of
  entering the battlefield." A permanent never exiles itself as it enters (`e == ev.entity` skip).
- `src/systems/state_manager_statics.cpp` (`apply_permanent_components`): the function now takes the
  `Orderer` (threaded through from `state_based_effects`). After the ETB replacement dispatch, if
  `redirect_to_exile` is set, the card is moved to exile via `orderer->add_to_zone(..., EXILE)` and
  its `Permanent` component is **not** created (it never enters the battlefield). Otherwise the
  `cast_to_battlefield` marker is consumed.

## Behavioral decisions (made in lieu of asking a human)
- **Controller-agnostic.** The replacement applies to *any* non-token, uncast creature entering the
  battlefield, regardless of who controls it — including the Containment Priest controller's own
  reanimated/searched creatures and the opponent's. This matches the Oracle text ("a nontoken
  creature", not "a creature an opponent controls"). Verified both same-controller and cross-player.
- **"wasn't cast" = did not resolve as a cast spell.** Anything put onto the battlefield by an
  effect (Green Sun's Zenith search-to-battlefield, reanimation, etc.) is treated as not cast and is
  exiled; a creature that resolves as a cast spell from the stack enters normally. This is the
  intended CR reading and is what `cast_to_battlefield` encodes.
- **Tokens excluded** (`!token`): handled both by the script filter and because token entities go
  through a separate token branch in `apply_permanent_components` and lack `CardData`, so they are
  never offered as candidates.
- **Flash** is the printed keyword and already supported; no work needed (verified the card can be
  cast and behaves as a 2/2).

## Tests
Isolation (`train/test_harness.py`, pre-set battlefields, Green Sun's Zenith used as a non-cast
creature-entry effect — it searches a green creature and *puts it onto the battlefield*):
- **Non-cast creature exiled.** A controls `Containment Priest` + Forests, casts `Green Sun's Zenith`
  for X=1 fetching `Birds of Paradise` → **"Birds of Paradise is exiled instead of entering the
  battlefield."** Birds is not on the battlefield afterward. PASS.
- **Control (no Priest).** Same line without Containment Priest in play → "Player A puts Birds of
  Paradise to the battlefield"; Birds appears on A's battlefield. PASS — confirms the exile is caused
  by the Priest, not by GSZ.
- **Cast creature unaffected.** A controls `Containment Priest`, hard-casts `Birds of Paradise` from
  hand → "Birds of Paradise enters the battlefield" and appears on A's battlefield (no exile message).
  PASS — a cast creature is correctly let through.
- **Cross-player.** B controls `Containment Priest`; A casts `Green Sun's Zenith` fetching `Birds of
  Paradise` → **"Birds of Paradise is exiled instead of entering the battlefield."** PASS — the
  replacement applies to creatures entering under the opponent of the Priest's controller.

Regression (`train/test_harness.py --scripted`, 6 games, seeds 1–6): deck `temp/cp_test`
(4 Containment Priest, 4 Green Sun's Zenith, 4 Birds of Paradise, 3 Noble Hierarch, 4 Scythecat Cub,
2 Thalia, 3 Swords to Plowshares, 2 Knight of the Reliquary, GW lands/fetches) vs `delver`. All 6
games finished decisively (A 2 / B 4), no draws, no fatal/non-fatal errors, no asserts/tracebacks.
Containment Priest was drawn, cast, and resolved in real games with the engine stable. (Only the
pre-existing cosmetic `WARNING: Unrecognized ability param: AIXMax$ Y` for Green Sun's Zenith and
other unrelated cards appeared.)

## Result
implemented
