# Hide on the Ceiling  (vocab index 243)

## Oracle text
Exile X target artifacts and/or creatures. Return the exiled cards to the battlefield under their
owners' control at the beginning of the next end step.

(Instant, mana cost {X}{U}.  Universes Within of "Spectral Restitching".)

## Forge script
- Source: pre-existing local script — `bin/resources/cardsfolder/h/hide_on_the_ceiling.txt` (not edited).
- Key tags:
  - `A:SP$ ChangeZone | ValidTgts$ Artifact,Creature | TargetMin$ X | TargetMax$ X |
    Origin$ Battlefield | Destination$ Exile | RememberChanged$ True | SubAbility$ DBDelTrig`
  - `DBDelTrig:DB$ DelayedTrigger | Mode$ Phase | Phase$ End of Turn | Execute$ TrigReturn |
    RememberObjects$ RememberedLKI | SubAbility$ DBCleanup`
  - `TrigReturn:DB$ ChangeZone | Origin$ Exile | Destination$ Battlefield |
    Defined$ DelayTriggerRememberedLKI`
  - `DBCleanup:DB$ Cleanup | ClearRemembered$ True`
  - `SVar:X:Count$xPaid`
- Tags parsed as written; `TgtPrompt$`/`SpellDescription$`/`Variant:`/`TriggerDescription$` cosmetic.

## Engine work (builds on the shared "variable-X" unit; see toxic_deluge.md / candelabra_of_tawnos.md)

This card's mana cost is `{X}{U}`, so X is already prompted by the engine's existing `has_x_cost`
path (`cur_game.x_paid` set before targets — CR 601.2b). The **only** new requirement for this card
is **exactly-X targeting**, which the shared mechanic added for Candelabra provides:
- `TargetMin$ X | TargetMax$ X` (both `Count$xPaid`) → `target_min_from_xpaid` /
  `target_max_from_xpaid` via `resolve_xpaid_target_counts()` (`src/parse.cpp`); `select_target`'s
  multi-target loop then requires **exactly X** targets (no "Done", clamps at X) reading the X
  chosen by the mana-X prompt.
- The multi-target exile, the `RememberChanged$ True` snapshot of all exiled cards, the
  `DB$ DelayedTrigger` (RememberedLKI) firing at the next end step, and the `TrigReturn`
  ChangeZone returning the cards to the battlefield **under their owners' control** are all
  pre-existing machinery (`effect_change_zone.cpp` already iterates `ab.targets`, remembers each,
  and the Flickerwisp/Phelia exile-and-return delayed-trigger path snapshots and returns them). No
  new effect code was needed beyond the exactly-X targeting.

## Behavioral decisions (made in lieu of asking a human)
- **X from the mana cost** (`{X}{U}`) is the target count (`Count$xPaid`); exactly X
  artifacts/creatures are exiled.
- `ValidTgts$ Artifact,Creature` (no controller qualifier) lets either player's
  artifacts/creatures be targeted.
- Exiled cards **return at the next end step under their owners' control** — the engine reuses the
  same RememberedLKI delayed-return used by Flickerwisp/Phelia, so the cards re-enter as new
  objects under their owners.

## Tests (test_harness, seed 1; board A: 3 Islands; opponent B: Grizzly Bears, Birds of Paradise)
- **Cast X=2:** "Choose X value (0-2)" offered; targeting required exactly two — the menu offered
  **no** "Done" and stopped after two picks; both opponent permanents moved to exile; "Delayed
  trigger registered ... at next End of Turn"; at the end step both Grizzly Bears and Birds of
  Paradise returned to the battlefield. PASS.
- **Regression (`--scripted` full games, seeds 1-6):** the scripted agent cast Hide on the Ceiling
  in every game; all six decisive (no draws), no non-fatal errors.

## Result
implemented
