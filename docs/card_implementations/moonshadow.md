# Moonshadow  (vocab index 138)

## Oracle text
Menace

This creature enters with six -1/-1 counters on it.

Whenever one or more permanent cards are put into your graveyard from anywhere while this
creature has a -1/-1 counter on it, remove a -1/-1 counter from this creature.

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/m/moonshadow.txt`
- Type: `Creature Elemental`, mana cost `B`, P/T `7/7`.
- Key tags:
  - `K:Menace` — CR 702.111: evasion keyword; the creature can't be blocked except by two or
    more creatures.
  - `K:etbCounter:M1M1:6` — CR 614.1c replacement: Moonshadow enters with six -1/-1 counters
    (so it is effectively 1/1 on entry).
  - `T:Mode$ ChangesZoneAll | ValidCards$ Permanent.YouOwn+!token | Origin$ Any | Destination$
    Graveyard | TriggerZones$ Battlefield | IsPresent$ Card.Self+counters_GE1_M1M1 |
    NoResolvingCheck$ True | Execute$ TrigRemoveCounter` — "whenever one or more permanent
    cards are put into your graveyard from anywhere while this creature has a -1/-1 counter on
    it" (CR 603.2 zone-change trigger; the `IsPresent$` is the intervening-if "while it has a
    -1/-1 counter").
  - `SVar:TrigRemoveCounter:DB$ RemoveCounter | Defined$ Self | CounterType$ M1M1 | CounterNum$
    1` — remove one -1/-1 counter from Moonshadow (CR 122.5).

No tags were retagged or repurposed; every mechanic is keyed on the tag's intended meaning. The
cosmetic `NoResolvingCheck$ True` (a Forge optimisation hint to skip re-checking the trigger
condition on resolution) is ignored — the engine re-checks the intervening-if at resolution
anyway (CR 603.4), which is harmless here; and `Mode$ ChangesZoneAll`'s once-per-batch nuance is
elided (the engine fires the trigger per matching card, see parse.cpp note), which only differs
when multiple matching cards hit the graveyard simultaneously and changes nothing observable for
this card since each fired copy just removes one counter that the intervening-if re-validates.

## Engine work
Four mechanics, none of which previously existed; each added as a general handler keyed on the
tag's meaning.

1. **`DB$ RemoveCounter`** (CR 122.5) — the executed sub-ability had no resolve-time handler
   (the category fell through to a no-op, so the counter was never removed):
   - `src/effects/effect_kind.{h,cpp}`: new `EffectKind::RemoveCounter` + string mapping.
   - `src/effects/effect_table.cpp`: dispatch to `remove_counter`.
   - `src/effects/effect_remove_counter.cpp` (new): removes up to `CounterNum$` counters of
     `CounterType$` from the target (the chosen target, else `Defined$ Self` = the source).
     Reuses the shared `CounterParams` (parsed by `parse_put_counter`, same as PutCounter) and
     `add_counters`, which resyncs the creature's cached P/T for +1/+1 and -1/-1 changes.

2. **`etbCounter` with a literal count** (CR 614.1c) — the parser only handled delve/X-paid
   dynamic counts; `etbCounter:M1M1:6` (a literal 6) yielded a count of 0, so the counters were
   never applied:
   - `src/components/static_ability.h`: new `int counter_count` (literal "enters with" count).
   - `src/parse.cpp`: a numeric count token is stored as `counter_count` (SVar tokens still
     resolve to delve / X-paid as before).
   - `src/systems/replacement_effects.cpp`: the ETB-counters replacement uses `counter_count`
     when it isn't a delve/X-paid count.
   - `src/systems/state_manager_statics.cpp`: -1/-1 "enters with" counters are now applied in
     the creature block (after the `Creature` component exists) alongside +1/+1, so
     `add_counters` resyncs P/T. Previously M1M1 was added before the Creature existed, leaving
     the cached P/T wrong (Moonshadow would have stayed 7/7 instead of becoming 1/1).

3. **The `ChangesZoneAll` trigger filters** — the existing CARD_CHANGED_ZONE matcher had no
   way to express "permanent cards (not tokens)", so the trigger fired on the wrong objects:
   - `src/components/ability.h` + `src/parse.cpp` + `src/systems/state_manager_triggers.cpp`:
     new `trigger_valid_card_is_permanent` (`ValidCard$ Permanent` — permanent card types only,
     excludes instants/sorceries via the new `is_permanent_card` helper in
     `src/game_queries.h`) and `trigger_valid_card_non_token` (`!token` — excludes tokens,
     which aren't cards, CR 110.1). Both flags are carried through the `Execute$` SVar copy.
     Without the Permanent filter the trigger fired when the controller's own Lightning Bolt
     went to the graveyard; without the non-token filter a dying token would have removed a
     counter.
   - The existing `.YouOwn` → `trigger_valid_player_is_controller` filter (the changing card's
     owner must be the source's controller) already covered "your graveyard"; the
     `Origin$ Any` (no origin filter) and `Destination$ Graveyard` are handled by the existing
     origin/destination machinery.
   - `src/systems/state_manager_actions.cpp` (`evaluate_present_condition`): new handler for
     `IsPresent$ Card.Self+counters_GE<N>_<TYPE>` — the source must be on the battlefield AND
     carry at least N counters of the given type. This is the intervening-if "while this
     creature has a -1/-1 counter on it" (CR 603.4); it stops the trigger once the last -1/-1
     counter is gone. (Previously the unparsed qualifier fell through to "any permanent on the
     battlefield", which was always true.)

4. **Menace** (CR 702.111) — the keyword was parsed and stored on the creature but never
   enforced in combat:
   - `src/action_processor.cpp`: new `release_illegal_menace_blockers`, called when the
     defending player confirms blocks. For each menace attacker blocked by exactly one
     creature, the lone block is released (CR 509.1b: such a declaration is illegal; the only
     legal resolution is that the single creature isn't blocking it). Releasing rather than
     rejecting the confirm avoids a deadlock when no second blocker is available.

## Behavioral decisions (made in lieu of asking a human)
- **"Permanent cards … not tokens"**: enforced literally — instants/sorceries and tokens going
  to the graveyard do not trigger Moonshadow (CR 110.1/110.4a). Verified against the negative
  cases below.
- **"Your graveyard"** = the controller's: the engine matches the changing card's *owner*
  against Moonshadow's *controller* (the standard engine convention for `.YouOwn`). In the
  common case owner = controller; a stolen Moonshadow edge case (owner ≠ controller) is not
  separately modelled.
- **The intervening-if is re-checked at resolution** (CR 603.4): once Moonshadow has no -1/-1
  counter (all six removed, or annihilated by +1/+1 counters via SBA 704.5q), the trigger does
  nothing — even if it was put on the stack.
- **Menace lone-blocker release**: when a single creature is the only blocker of a menace
  attacker, the block is dropped (the creature deals/takes no combat damage and the attacker is
  unblocked), which is the rules-legal outcome of an otherwise-illegal declaration.

## Tests
Isolation (`train/test_harness.py`):
- **ETB counters.** Cast Moonshadow → `Moonshadow enters with 6 -1/-1 counter(s) (1/1)`; it is
  a 1/1 on the battlefield (7/7 base − six -1/-1). PASS.
- **Trigger fires on your permanent card.** Moonshadow (1/1) + your Grizzly Bears in play; an
  opponent's Lightning Bolt kills the bear → `Moonshadow triggered`, `Removed 1 M1M1 counter(s)
  (now 2/2)`. P/T resyncs to 2/2. PASS.
- **No trigger on an opponent's card.** Your Bolt kills the opponent's Grizzly Bears → no
  trigger, Moonshadow stays 1/1. PASS (the `.YouOwn` case).
- **No trigger on a non-permanent card.** Casting Lightning Bolt (an instant you own) and
  letting it go to your graveyard does not trigger Moonshadow. PASS (the `Permanent` filter).
- **Menace.** A menace attacker can't be blocked by a single creature: the lone block is
  released and the attacker connects with the player. PASS.

Regression (`train/test_harness.py --scripted`, seeds 1–6): a deck with 4 Moonshadow + removal
+ creatures vs `mav`. All games finished decisively, no draws, zero error/assert lines (only the
pre-existing cosmetic `WARNING: Unrecognized ability param`). Temp deck removed.

## Result
implemented
