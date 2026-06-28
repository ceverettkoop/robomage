# Phelia, Exuberant Shepherd (vocab index 210)

## Oracle text
Flash
Whenever Phelia, Exuberant Shepherd attacks, exile up to one other target nonland permanent. At
the beginning of the next end step, return that card to the battlefield under its owner's control.
If it entered under your control, put a +1/+1 counter on Phelia.

`1 W` — a 2/2 Legendary Dog.

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/p/phelia_exuberant_shepherd.txt`). Parsed
as written — no retag. Key tags:
- `K:Flash`
- Attack trigger: `T:Mode$ Attacks | ValidCard$ Card.Self | TriggerZones$ Battlefield | Execute$
  TrigExile`
- `SVar:TrigExile:DB$ ChangeZone | ValidTgts$ Permanent.Other+nonLand | TargetMin$ 0 | TargetMax$
  1 | Origin$ Battlefield | Destination$ Exile | RememberChanged$ True | SubAbility$ DelTrig`
- `SVar:DelTrig:DB$ DelayedTrigger | Mode$ Phase | Phase$ End of Turn | Execute$ TrigBounce |
  RememberObjects$ RememberedLKI | SubAbility$ DBCleanup`
- `SVar:TrigBounce:DB$ ChangeZone | Origin$ Exile | Destination$ Battlefield | Defined$
  DelayTriggerRememberedLKI | Imprint$ True | SubAbility$ DBPutCounter`
- `SVar:DBPutCounter:DB$ PutCounter | Defined$ Self | CounterType$ P1P1 | CounterNum$ 1 |
  ConditionDefined$ Imprinted | ConditionPresent$ Card.YouCtrl+ThisTurnEntered | SubAbility$
  DBCleanup`
- `SVar:DBCleanup:DB$ Cleanup | ClearRemembered$ True | ClearImprinted$ True`

## Engine work
Phelia reuses the shared **exile-return-delayed mechanic** added with Flickerwisp (`RememberObjects$
RememberedLKI` capture + `Defined$ DelayTriggerRememberedLKI` return + Execute-subability
selection/deferral + `Phase$ End of Turn` → END_STEP_BEGAN; see `flickerwisp.md`). It needed three
additional general pieces:

1. **`Mode$ Attacks` trigger** (`src/parse.cpp`): "Whenever this creature attacks" now maps to the
   existing per-attacker `CREATURE_ATTACKED` event (emitted in `src/action_processor.cpp` at attack
   declaration, CR 508.2), with `ValidCard$ Card.Self` → `trigger_only_self` matching the attacking
   entity against the source. General over any `Mode$ Attacks` card; the generic trigger scan
   already handles `CREATURE_ATTACKED` + `trigger_only_self`.
2. **`ConditionDefined$ Imprinted`** (`src/parse.cpp`): routed to the existing
   `condition_on_remembered` path. The "imprinted" card here IS the returned card already held in
   `cur_game.remembered_entities` (via `RememberObjects$ RememberedLKI`/`Defined$
   DelayTriggerRememberedLKI`), so the imprint reuses the remembered-set condition machinery; the
   redundant `Imprint$ True` / `ClearImprinted$ True` tags are ignored (CLAUDE.md: ignoring a tag
   whose effect is already covered is acceptable).
3. **Filter-based remembered-set condition** (`evaluate_present_condition`,
   `src/systems/state_manager_actions.cpp`): `ConditionPresent$ Card.YouCtrl+ThisTurnEntered` now
   counts the remembered cards matching the filter via `permanent_matches_filter`. Because a card
   returned earlier in the SAME resolution has its `Zone` set to BATTLEFIELD with its controller
   assigned but its `Permanent` component created only by the deferred SBA pass, a Zone-based
   fallback checks the YouCtrl/OppCtrl clause against `Zone.controller` (entered-this-turn is
   definitionally true — it just returned). This is what lets "If it entered under your control"
   evaluate correctly within the delayed trigger's resolution.

`PutCounter | Defined$ Self` already targets the ability's source (Phelia), so the +1/+1 counter
goes on Phelia.

## Behavioral decisions (CR)
- `Permanent.Other+nonLand` + `TargetMin$ 0 | TargetMax$ 1`: "up to one OTHER target nonland
  permanent" — Phelia herself and lands are excluded; choosing "No target" exiles nothing (CR 115,
  the optional-target path).
- The card returns **under its owner's control** at the beginning of the next end step (CR 513 /
  608.2g): exiling an opponent's creature is temporary removal (gone during combat, back at end
  step under the opponent's control); exiling your own permanent blinks it back under your control.
- "If it entered under your control, put a +1/+1 counter on Phelia": the counter is placed only when
  the returned card re-enters under the ability's controller's control (you blinked your own
  permanent). Exiling an opponent's permanent grants no counter. Verified both ways.
- The trigger fires per attack declaration (CR 508.2), once for Phelia each time she attacks.
- `TriggerDescription$`/`TgtPrompt$`/`StackDescription$`/`HasAttackEffect`/`DeckHas` are cosmetic
  (ignored).

## Tests (test_harness.py, semantic `--play`, seed 1)
- **(a) exile opponent's creature, no counter** — Phelia (in play) attacks, exiles B's Scythecat
  Cub (`Scythecat Cub is moved to exile`; gone during combat); at the end step `Delayed trigger
  fires` → returned to the battlefield under **B's** control (SICK); Phelia stays **2/2** (no
  counter — it did not enter under A's control). **Pass.**
- **(b) blink your own permanent, get a counter** — Phelia attacks, exiles A's own Collector Ouphe;
  at the end step it returns under **A's** control and `Put 1 +1/+1 counter(s) on creature (now
  3/3)` — Phelia becomes **3/3**. **Pass.**
- **(c) up-to-one, no target** — Phelia attacks; the trigger offers `No target` (TargetMin 0);
  choosing it exiles nothing. **Pass.**
- **Regression** — scripted full games `temp/phelia_a` (4 Phelia + white/green base) vs
  `temp/bolt_b` (Lightning Bolt / Mountain), seeds 1/2/3: all decisive (Player A wins each; no
  draws), zero non-fatal errors; Phelia's attack trigger fired in real games (3× in seed 3).

## Result
Implemented on top of the shared exile-return-delayed mechanic. New reusable pieces: `Mode$
Attacks` trigger (→ CREATURE_ATTACKED), `ConditionDefined$ Imprinted` (→ remembered-set condition),
and the Zone-aware remembered-set filter condition for cards returned earlier in the same
resolution.
