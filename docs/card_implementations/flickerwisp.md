# Flickerwisp (vocab index 221)

## Oracle text
Flying
When Flickerwisp enters, exile another target permanent. Return that card to the battlefield
under its owner's control at the beginning of the next end step.

`1 W W` — a 3/1 Elemental.

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/f/flickerwisp.txt`). Parsed as written —
no retag. Key tags:
- ETB: `T:Mode$ ChangesZone | ValidCard$ Card.Self | Origin$ Any | Destination$ Battlefield |
  Execute$ TrigExile`
- `SVar:TrigExile:DB$ ChangeZone | ValidTgts$ Permanent.Other | Mandatory$ True | Origin$
  Battlefield | Destination$ Exile | RememberChanged$ True | SubAbility$ DelTrig`
- `SVar:DelTrig:DB$ DelayedTrigger | Mode$ Phase | Phase$ End of Turn | Execute$ TrigBounce |
  RememberObjects$ RememberedLKI | SubAbility$ DBCleanup`
- `SVar:TrigBounce:DB$ ChangeZone | Origin$ Exile | Destination$ Battlefield | Defined$
  DelayTriggerRememberedLKI`
- `SVar:DBCleanup:DB$ Cleanup | ClearRemembered$ True`

## Engine work — the shared exile-return-delayed mechanic
The triggered exile and the targeted ChangeZone Battlefield→Exile (with `RememberChanged$ True`)
already existed (Skyclave Apparition). The delayed-trigger end-step framework
(`DB$ DelayedTrigger | Mode$ Phase | Phase$ End of Turn`) also existed (Mishra's Bauble). The gap
was returning the *specific exiled card* when the delayed trigger fires — `RememberObjects$
RememberedLKI` and `Defined$ DelayTriggerRememberedLKI` had no resolver, so the return did
nothing. This implements that as a general handler (CR 603.7a — a delayed triggered ability
references the objects it remembered when it was set up):

1. **`RememberObjects$ RememberedLKI` capture** (`src/parse.cpp`,
   `DelayedTriggerParams::remember_objects_lki` in `src/components/ability_params.h`): parsed into
   a flag. When `effects::delayed_trigger` (`src/effects/effect_delayed_trigger.cpp`) registers the
   trigger, it snapshots the objects the immediately-preceding `RememberChanged$` ChangeZone moved
   (`cur_game.remembered_entities`) onto the `DelayedTrigger` record
   (`DelayedTrigger::remembered_objects`, `src/classes/game.h`) and onto the fire ability's
   existing `restore_remembered_exiled_with` carrier.
2. **`Defined$ DelayTriggerRememberedLKI` resolution** (`src/parse.cpp`): routed to the existing
   `defined_remembered` flag. When the delayed trigger fires, `Ability::resolve` restores the
   captured objects into `cur_game.remembered_entities` (the pre-existing
   `restore_remembered_exiled_with` path, line ~946 of `src/components/ability.cpp`), so the
   `Defined$ Remembered` branch of `effects::change_zone` moves exactly those cards from Exile to
   the Battlefield under their owner's control.
3. **Execute sub-ability selection fix** (`effects::delayed_trigger`): the Execute$ sub-ability is
   now marked `from_delayed_execute` at parse (`src/parse.cpp`) and selected explicitly, with any
   trailing `SubAbility$` (the DBCleanup) chained after it — replacing the fragile
   `subabilities.back()` which broke once a DelayedTrigger had both an Execute$ and a SubAbility$.
4. **Defer the DelayedTrigger's subabilities** (`effects::delayed_trigger` now returns `false`):
   previously the handler returned `true`, so `Ability::resolve` chained the DB$ DelayedTrigger's
   subabilities *inline*, resolving the return immediately instead of at the scheduled phase. They
   are now deferred onto the delayed trigger and run only when it fires.
5. **`Phase$ End of Turn` → END_STEP_BEGAN** (`phase_string_to_event`): the parser keeps the raw
   `Phase$` value, which Forge writes as either `EndStep` or `End of Turn`; both now map to the
   beginning of the end step (CR 513). Previously only `EndStep` was recognized, so the trigger
   defaulted to upkeep and never returned the card at end of turn.
6. **`Mode$ Phase` on a DB$ DelayedTrigger** is consumed (informational) so it isn't flagged as an
   unrecognized param.

All edits are general: any exile-and-return-at-a-later-phase card (Flickerwisp, Phelia, and future
ones) reuses them; Mishra's Bauble's slowtrip (the other DelayedTrigger user) is preserved.

## Behavioral decisions (CR)
- `Permanent.Other` excludes Flickerwisp itself but allows **any other permanent** (your own or an
  opponent's; lands included) — temporary removal or a blink-to-reset.
- `Mandatory$ True` with a single required target: if a legal target exists it must be chosen
  (standard ChangesZone-with-target behavior).
- The card returns **under its owner's control** (CR 608.2g): the `Defined$ Remembered` ChangeZone
  sets the entering permanent's controller to its owner, so exiling an opponent's permanent returns
  it to the opponent.
- Return is at the beginning of **the next end step** (CR 513) — the end step of the turn it was
  exiled (the trigger is not `NextTurn$`). It fires on its controller's end step.
- A returned permanent enters as a new object (summoning sick, no counters/auras) — standard blink.
- `TriggerDescription$`/`TgtPrompt$` are cosmetic (ignored).

## Tests (test_harness.py, semantic `--play`, seed 1)
- **(a) exile opponent's permanent, return at end step** — A cast Flickerwisp targeting B's
  Scythecat Cub. `Scythecat Cub is moved to exile`; `Delayed trigger registered ... at next End of
  Turn`; gone from the battlefield through A's turn; at A's end step `Delayed trigger fires` →
  `Scythecat Cub is moved to the battlefield` (back under B's control). **Pass.**
- **(b) blink your own permanent** — A cast Flickerwisp targeting its own Collector Ouphe.
  `Collector Ouphe is moved to exile`, then at end step `Delayed trigger fires` → `Collector Ouphe
  is moved to the battlefield`. **Pass.**
- **Regression** — Mishra's Bauble (the other DelayedTrigger user) still defers its slowtrip draw
  to the next turn's upkeep (registered turn 0, drew at turn 2 — not immediately). Scripted full
  games `temp/flick_a` (4 Flickerwisp + white/green base) vs `temp/bolt_b` (Lightning Bolt /
  Mountain), seeds 1/2/3: all decisive (Player A wins each; no draws), zero non-fatal errors.

## Result
Implemented. Added the general exile-return-delayed mechanic (`RememberObjects$ RememberedLKI`
capture + `Defined$ DelayTriggerRememberedLKI` return + Execute-subability selection/deferral fix +
`End of Turn` phase mapping), shared with Phelia, Exuberant Shepherd.
