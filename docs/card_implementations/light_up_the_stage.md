# Light Up the Stage

## Oracle text
Spectacle {R} (You may cast this spell for its spectacle cost rather than its mana cost if an
opponent lost life this turn.)
Exile the top two cards of your library. Until the end of your next turn, you may play those
cards.

(`{2}{R}` sorcery)

## Forge script
- **Source:** pre-existing local (`bin/resources/cardsfolder/l/light_up_the_stage.txt`).
- **Key tags:**
  - `K:Spectacle:R`
  - `A:SP$ Dig | Defined$ You | DigNum$ 2 | ChangeNum$ All | DestinationZone$ Exile |
    RememberChanged$ True | SubAbility$ DBEffect`
  - `SVar:DBEffect:DB$ Effect | StaticAbilities$ StaticMayPlay |
    Duration$ UntilTheEndOfYourNextTurn | RememberObjects$ Remembered | ForgetOnMoved$ Exile |
    SubAbility$ DBCleanup`
  - `SVar:StaticMayPlay:Mode$ Continuous | Affected$ Card.IsRemembered | AffectedZone$ Exile |
    MayPlay$ True`

## Engine work
Reuses the **Spectacle** mechanic built for Skewer the Critics (`K:Spectacle:R` — the same
`AltCost::is_spectacle` parse + `can_afford_alt` opponent-lost-life gate). New mechanic added:
**impulse-play-paid** — a play-from-exile permission where the cards are played for their NORMAL
cost (not free), LANDS are permitted, and it persists until the end of the caster's next turn.

1. `src/effects/effect_dig.cpp` — the `SP$ Dig` handler now (a) accepts `DestinationZone$ Exile`
   (parse), and (b) honors `RememberChanged$ True` by pushing the chosen (exiled) cards into
   `cur_game.remembered_entities`, mirroring `ChangeZone`'s RememberChanged. This is what lets the
   paired `DB$ Effect` grant a permission on exactly those two cards.
2. `src/components/ability.h` — added `duration_until_end_of_your_next_turn` (Duration$
   UntilTheEndOfYourNextTurn) and `effect_grant_play_from_exile` (the plain-MayPlay grant flag).
3. `src/parse.cpp`:
   - `apply_param_to_ability` parses `Duration$ UntilTheEndOfYourNextTurn`.
   - The `DB$ Effect | StaticAbilities$` branch now sets `effect_grant_play_from_exile` when the
     named static is `MayPlay$ True` + `AffectedZone$ Exile` but WITHOUT
     `MayPlayWithoutManaCost$ True` (Ugin's free-cast form keeps its existing flag).
4. `src/classes/game.h` — `Game::ImpulseCastPermission` gained a `NORMAL` resource (play for the
   normal cost), an `allow_land` flag (a land among the exiled cards may be played), and
   `persist_until_end_of_next_turn` + `grant_turn` for the longer duration.
5. `src/effects/effect_grant_cast.cpp` — new `effect_grant_play_from_exile` branch: for each
   remembered exiled card, record a `NORMAL`-cost, land-allowed impulse permission carrying the
   duration and grant turn.
6. `src/systems/state_manager_actions.cpp` — the impulse-cast legal-action loop now: (a) offers a
   `PLAY_LAND` (from exile) for a land under a NORMAL+allow_land permission (sorcery timing, own
   main, empty stack, land drop remaining); (b) for a nonland NORMAL permission, gates
   affordability on the full `effective_base_cost` (hybrid-resolved) and labels it "(from exile)".
7. `src/action_processor.cpp` — the `impulse_cast` cast path pays the NORMAL base cost (deferred,
   after targets) for a NORMAL permission instead of a resource; the permission is erased when the
   card is cast, and also when a land is played from exile (`SPECIAL_ACTION` handler) —
   `ForgetOnMoved$ Exile`.
8. `src/classes/game.cpp` — cleanup no longer blanket-clears `impulse_cast_permission`; a
   `persist_until_end_of_next_turn` grant survives until the caster's NEXT turn's cleanup (detected
   as a later cleanup — `turn > grant_turn` — whose active player is the grant's caster).

CR 305.2 / 601.3e (playing from a non-hand zone under a permission), CR 118.9 (alternative /
permission-based play), CR "until the end of your next turn" duration, CR 702.107 (Spectacle).

Mechanics added (general): **impulse-play-paid** — any `DB$ Effect` granting plain `MayPlay$ True`
+ `AffectedZone$ Exile` on remembered exiled cards now lets them be PLAYED from exile for their
normal cost (lands included), honoring `Duration$ UntilTheEndOfYourNextTurn`.

## Behavioral decisions
- **Normal cost, not free:** plain `MayPlay$ True` (vs Ugin's `MayPlayWithoutManaCost$ True`) means
  the exiled cards cost their printed mana cost; the parse distinguishes the two forms.
- **Lands are playable:** the permission is "play those cards", so a land among the two is a normal
  land play (subject to the one-land-per-turn drop), unlike a free-CAST grant which can't play
  lands (CR 601.1).
- **Duration:** "until the end of your next turn" is one full turn cycle longer than
  `UntilYourNextTurn`; the permission is expired at the caster's next-turn cleanup, robust to extra
  turns (the first later cleanup on the caster's turn).
- **Scope:** only the two dug (remembered) cards get the permission — a card already in exile is
  unaffected.
- **X spells** played this way resolve with X = 0 (no X prompt on the impulse path) — a documented
  edge simplification.

## Tests (isolation)
- Cast Light Up the Stage for its normal `{2}{R}` (4 Mountains in play). Top two library cards
  (Mountain + Lightning Bolt) exiled; log "may play … from exile until the end of their next
  turn"; menu offered **"Play Mountain (from exile)"** and **"Cast Lightning Bolt (from exile)"**.
  PASS.
- Played the exile Mountain → entered the battlefield, permission consumed (no longer offered).
  PASS.
- Cast the exile Lightning Bolt for its normal `{R}` (one Mountain tapped) at Player B → 3 damage
  (20 → 17). PASS.
- **Scope:** a `Grizzly Bears` preset already in exile (not one of the dug cards) was NEVER offered
  as playable. PASS.
- Spectacle reuse: `K:Spectacle:R` parses identically to Skewer's (shared infra), verified in the
  Skewer isolation tests.
- CI gate: `ci_check.py --tier pygen,vocab,smoke` — see batch report.

## Result
Implemented.
