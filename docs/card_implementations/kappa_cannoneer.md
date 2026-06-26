# Kappa Cannoneer  (vocab index 136)

## Oracle text
Improvise (Your artifacts can help cast this spell. Each artifact you tap after you're done
activating mana abilities pays for {1}.)

Ward {4} (Whenever this creature becomes the target of a spell or ability an opponent controls,
counter that spell or ability unless that player pays {4}.)

Whenever another artifact you control enters, put a +1/+1 counter on Kappa Cannoneer and it can't
be blocked this turn.

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/k/kappa_cannoneer.txt`
- Type: `Artifact Creature Turtle Warrior`, mana cost `5 U`, P/T `4/4`.
- Key tags:
  - `K:Improvise` — CR 702.126 cost-payment keyword: tap untapped artifacts you control to pay
    {1} of the generic cost each.
  - `K:Ward:4` — CR 702.21 protection: an opponent's spell/ability targeting Kappa is countered
    unless they pay {4}.
  - `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Artifact.YouCtrl |
    IsPresent$ Card.Self | TriggerZones$ Battlefield | Execute$ TrigPutCounter` — "whenever
    another artifact you control enters" (CR 603.2 zone-change trigger; the `IsPresent$ Card.Self`
    idiom expresses both the intervening-if "while Kappa is in play" and the Oracle's "another").
  - `SVar:TrigPutCounter:DB$ PutCounter | CounterType$ P1P1 | SubAbility$ DBUnblockable` — put a
    +1/+1 counter on Kappa (CR 702.9 / 122.1), then chain DBUnblockable.
  - `SVar:DBUnblockable:DB$ Effect | RememberObjects$ Self | ExileOnMoved$ Battlefield |
    StaticAbilities$ Unblockable` — a transient continuous effect making the source unblockable
    this turn (CR 509.1b).
  - `SVar:Unblockable:Mode$ CantBlockBy | ValidAttacker$ Card.IsRemembered` — the static the
    effect grants: the remembered creature (Kappa) can't be blocked.

No tags were retagged or repurposed; every mechanic below is keyed on the tag's intended meaning.
Cosmetic `ExileOnMoved$`/`Mode$ CantBlockBy` parameters that the engine does not model as a
standalone continuous-effect object surface only as a harmless `WARNING: Unrecognized ability
param` — the full behavior is realized from the other tags.

## Engine work
Three mechanics, none of which previously existed; each added as a general handler keyed on the
tag's meaning.

1. **Improvise** (CR 702.126), modeled exactly like the existing Delve cost-reduction but tapping
   battlefield artifacts instead of exiling graveyard cards:
   - `src/components/carddata.h`: new `bool has_improvise` (set in `src/parse.cpp` from
     `K:Improvise`).
   - `src/mana_system.{h,cpp}`: `can_pay_mana` / `prompt_mana_payment` / `auto_pay_mana` take a
     parallel `has_improvise` flag. New helpers `is_improvise_eligible` (untap artifact you
     control, not the spell being paid for) and `improvise_tap_one` (tap it, satisfy one
     `GENERIC`). Improvise is offered for generic pips in both the machine-mode auto-payer (after
     mana sources, so colored pips keep their producers) and the interactive payer (a "Tap X
     (Improvise)" option), with `tapped_entities` shared so an artifact-mana source is never both
     improvised and mana-tapped.
   - `src/systems/state_manager_actions.cpp` / `src/action_processor.cpp`: the cast legality check
     and the cast payment both thread `card_data.has_improvise`.

2. **Ward {N}** (CR 702.21) — a becomes-targeted trigger that counters the targeting object unless
   its controller pays {N}, reusing the existing "counter unless pay" machinery:
   - `src/components/carddata.h`: new `int ward_cost` (set from `K:Ward:N`, defaulting to 1 for a
     bare `K:Ward`).
   - `src/action_processor.cpp`: new `trigger_ward_for_targets(targeting_entity, controller,
     targets, orderer)`, called right after a spell or activated ability with chosen targets is
     put on the stack. For each target that is a battlefield permanent with a Ward cost controlled
     by an *opponent* of the targeting player, it pushes a `Counter` triggered ability (target =
     the targeting object, `unless_generic_cost = N`, controlled by the Ward permanent's
     controller) onto the stack — landing above the targeting object so Ward resolves first. The
     existing `Counter` effect (`effect_counter.cpp`) runs `run_unless_loop` (the Mana-Leak /
     PAY_UNLESS path) to give the targeting player the pay-or-be-countered choice, and handles
     both spells (→ graveyard) and standalone abilities (→ exile/destroy).

3. **The artifact-ETB trigger** (PutCounter on self + can't be blocked this turn):
   - `src/components/ability.h` + `src/parse.cpp` + `src/systems/state_manager_triggers.cpp`: new
     `trigger_valid_card_is_artifact` filter for `ValidCard$ Artifact.*` ChangesZone triggers
     (matches `CardData`/`Token` artifact types). Without it the trigger would fire on *any*
     permanent you control entering. The flag is also carried through the `Execute$` SVar copy.
   - `src/parse.cpp`: a `Destination$ Battlefield` trigger gated by `IsPresent$ Card.Self` is
     recognized as the Forge idiom for "another … enters" and marked `trigger_self_excluded`, so
     Kappa's own entry does not trigger it (Oracle: "another artifact you control"). The
     `IsPresent$ Card.Self` intervening-if is also given a real handler in
     `evaluate_present_condition` ("the source is on the battlefield").
   - `src/effects/effect_put_counter.cpp`: `CounterNum$` now defaults to 1 when omitted (Forge
     default), so "put a +1/+1 counter" actually adds one (previously a missing count meant 0).
   - `src/components/ability.h` + `src/parse.cpp`: `DB$ Effect | StaticAbilities$ Unblockable |
     RememberObjects$ Self` is parsed into `effect_static_ability` / `effect_remember_self`.
   - `src/components/creature.h`: new `bool cant_be_blocked_this_turn`.
   - `src/effects/effect_grant_cast.cpp` (the `Effect` category handler): when the effect's
     `StaticAbilities$` is `Unblockable`, it sets `cant_be_blocked_this_turn` on the remembered
     source rather than instantiating a continuous-effect object.
   - `src/action_processor.cpp`: `determine_blockable_attackers` removes a `cant_be_blocked_this_turn`
     attacker from every blocker's legal list (CR 509.1b).
   - `src/classes/game.cpp`: the flag is cleared at the cleanup step (CR 514.2 — "this turn").

## Behavioral decisions (made in lieu of asking a human)
- **"Another"** is enforced (self-excluded): Kappa entering does not trigger its own ability, even
  though the Forge script lacks `.Other` — the printed Oracle text is authoritative.
- **Ward fires per "becomes a target"** (CR 702.21): a single spell targeting one opponent's Ward
  permanent fires one Ward trigger; if the targeting object had multiple targets that are Ward
  permanents, each fires its own. Ward only fires for an *opponent's* spell/ability (a permanent's
  controller targeting its own Ward creature does not trigger it).
- **Ward resolves above the targeting object** and counters it unless {N} is paid by that object's
  controller — exactly the "counter unless pay" template (Mana Leak), so machine-mode RL answers it
  through the existing PAY_UNLESS action category.
- **Improvise pays only generic** ({1} each), after mana abilities, and never taps the spell being
  cast; an artifact already tapped for mana isn't reused. Castability and payment use the single
  shared `auto_pay_mana` algorithm, so a Kappa offered as castable can always be paid for.
- **"Can't be blocked this turn"** is a per-turn mark cleared at cleanup; it stacks across multiple
  artifact ETBs in a turn (idempotent) and lapses on Kappa's controller's next cleanup.

## Tests
Isolation (`train/test_harness.py`):
- **Artifact-ETB trigger.** Kappa in play, cast Mishra's Bauble → `Kappa Cannoneer triggered`,
  `Put 1 +1/+1 counter(s) on creature (now 5/5)`, `Kappa Cannoneer can't be blocked this turn`.
  PASS.
- **Unblockable enforced.** With the counter+unblockable applied and an opposing Grizzly Bears,
  combat prints `No creatures eligible to block` and Kappa deals 5 to the player. PASS.
- **No self-trigger.** Casting Kappa itself (via Improvise) → `Kappa Cannoneer enters the
  battlefield` with no trigger and Kappa stays 4/4. PASS — the "another" negative case.
- **Improvise.** Kappa (5U) cast with one Island + five untapped artifacts → five
  `… taps X for Improvise` lines + `Kappa Cannoneer enters the battlefield`. With only four
  artifacts (short one generic) the cast is correctly not offered. PASS.
- **Ward {4} — not paid.** Opponent's Lightning Bolt targets Kappa; controller declines →
  `Ward {4}: …`, then `Lightning Bolt is countered`; Kappa survives. PASS.
- **Ward {4} — paid.** With four extra lands the controller pays {4} → `Player A pays {4} — spell
  is not countered`, Bolt resolves (`Resolving ability (category: DealDamage, amount: 3)`). PASS.

Regression (`train/test_harness.py --scripted`, seeds 1–6): deck `temp/kappa_test` (4 Kappa
Cannoneer + Mishra's Bauble / Chalice / Lion's Eye Diamond / Null Rod / Grafdigger's Cage + 20
Island) vs `mav`. All 6 games finished decisively (A wins 3, B wins 3), no draws, zero error or
assert lines. A Kappa-mirror run (seeds 1–3) was also clean and exercised Improvise casting Kappa
in real games. Temp deck removed.

## Result
implemented
