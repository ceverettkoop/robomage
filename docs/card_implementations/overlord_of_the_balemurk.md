# Overlord of the Balemurk

Vocab index: 293

## Oracle

```
Overlord of the Balemurk        {3}{B}{B}
Enchantment Creature — Avatar Horror
Impending 5—{1}{B} (If you cast this spell for its impending cost, it enters with five
time counters and isn't a creature until the last is removed. At the beginning of your
end step, remove a time counter from it.)
Whenever Overlord of the Balemurk enters or attacks, mill four cards, then you may return
a non-Avatar creature card or a planeswalker card from your graveyard to your hand.
5/5
```

## Forge source (pre-existing local script)

`bin/resources/cardsfolder/o/overlord_of_the_balemurk.txt` (unmodified). Key tags:

- `K:Impending:5:1 B` — Impending 5—{1}{B} (the new general mechanic; CR 702.175).
- `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigMill`
  — the "enters" half of the combined enters-or-attacks trigger.
- `T:Mode$ Attacks | ValidCard$ Card.Self | Secondary$ True | Execute$ TrigMill` — the "attacks"
  half (`Secondary$ True` is a Forge de-dup display hint; both triggers run the same `TrigMill`).
- `SVar:TrigMill:DB$ Mill | NumCards$ 4 | SubAbility$ DBChangeZone`
- `SVar:DBChangeZone:DB$ ChangeZone | Origin$ Graveyard | Destination$ Hand |
   ChangeType$ Creature.nonAvatar+YouOwn,Planeswalker.YouOwn | Hidden$ True | Optional$ True`

The mill + optional graveyard→hand return and the enters/attacks triggers were already supported
by the engine and required no new work (verified — see Tests). Only Impending was new.

## Engine work — Impending (new general mechanic, CR 702.175)

Impending is implemented as a reusable ALTERNATIVE casting cost on the existing `AltCost`
plumbing, plus a counter-driven creature-suppression that mirrors the Reconfigure ("isn't a
creature while attached") pattern and a built-in end-step counter shed that mirrors the Stun
untap-step shed (CR 122.1d). No card-specific code; any future Impending card reuses it.

Reusable pieces (function / file:line for the ledger):

- `AltCost::is_impending` + `AltCost::impending_count` — `src/components/carddata.h`. The mana
  portion `{1}{B}` lives on the shared `AltCost::mana_cost`, so the existing `can_afford_alt` /
  `pay_alternate_cost` mana paths are reused unchanged.
- Parse `K:Impending:<N>:<mana>` — `src/parse.cpp` (the K: keyword loop, just before Prowess):
  colon-split like Equip/Reconfigure with an extra leading count field; fills the `AltCost`.
- Legal action — `src/systems/state_manager_actions.cpp` (hand-cast loop, the `can_alt` block):
  the existing alt-cost `LegalAction` (`use_alt_cost`) is emitted with an "(impending)"
  description when `alt_cost.is_impending`.
- Cast marking — `Spell::cast_with_impending` (`src/components/spell.h`), set in
  `src/action_processor.cpp` when `use_alt_cost && alt_cost.is_impending`.
- Spell→pending→Permanent pipeline (same three-stage path evoke/escape/offspring use):
  `cur_game.pending_impending` (`src/classes/game.h`) is filled in `src/systems/stack_manager.cpp`
  from the resolving spell, then consumed in `StateManager::apply_permanent_components`
  (`src/systems/state_manager_statics.cpp`) which puts `impending_count` `"TIME"` counters on the
  new Permanent (CR 702.175d).
- Creature suppression (CR 702.175d) — `apply_permanent_components`
  (`src/systems/state_manager_statics.cpp`): an `impending_suppressed` permanent (one with a
  `"TIME"` counter) has its `Creature`/`Damage` components stripped each state-based pass, exactly
  like the reconfigure suppression; the creature block re-adds them on the pass after the last
  time counter sheds, so it "becomes a creature" through the normal recompute.
- End-step shed (CR 702.175e) — `shed_impending_time_counters(Zone::Ownership)` in
  `src/classes/game.cpp`, called from the `SECOND_MAIN`→`END_STEP` transition right after
  `END_STEP_BEGAN` is sent: removes one `"TIME"` counter from each impending permanent the active
  player controls (mirrors the Stun untap-step shed).

## Behavioral decisions

- The impending suppression is gated purely on the presence of a `"TIME"` counter
  (`get_counters(entity, "TIME") > 0`). In this two-player vocab only Impending produces TIME
  counters, and a normally-cast Overlord never gets one, so the gate is exactly "entered via
  impending and counters remain" — no extra Permanent flag needed.
- The shed is a built-in end-step action (like the Stun shed) rather than a stack-using inherent
  triggered ability. CR 702.175e models it as a triggered ability, but the prompt explicitly
  allows the built-in shed; it produces identical observable behavior in the two-player engine.
- Summoning sickness needs no special handling: the permanent's `has_summoning_sickness` clears
  during the controller's untap steps while it counts down, so by the time the last counter sheds
  (its controller's 5th end step) it is already a creature that can attack.
- The ETB ("enters") trigger fires whether the permanent enters as a creature (normal cast) or as
  a noncreature enchantment (impending) — it is a ChangesZone→Battlefield trigger, independent of
  creature-ness.

## Tests (train/test_harness.py, temp decks, semantic `--play`)

1. Impending cast `{1}{B}` (2 Swamps): "Cast Overlord of the Balemurk (impending)" offered and
   taken → "enters with 5 time counter(s) (impending)", board shows it WITHOUT P/T (no `[5/5]`,
   no Creature component) = not a creature; ETB trigger fired Mill 4 + optional return. PASS.
2. End-step countdown: one `"TIME"` removed at each of Player A's end steps — 5→4 (turn 1), 3
   (turn 3), 2 (turn 5), 1 (turn 7), and at turn 9 "loses its last time counter (it is now a
   creature)". PASS.
3. Becomes a creature: after the last shed the board shows `Overlord of the Balemurk [5/5]` with
   no SICK marker and it was successfully declared as an attacker on turn 11. PASS.
4. Normal cast `{3}{B}{B}` (5 Swamps): cast with no time-counter line, immediately `[5/5]`, ETB
   Mill 4 + return. PASS.
5. Return filter / Optional: with a milled Avatar (a second Overlord) and three Grizzly Bears in
   the graveyard, the optional return offered ONLY the three Grizzly Bears (non-Avatar creatures
   you own) plus "Fail to find" — the Avatar Overlord was correctly excluded; declining and
   actually returning ("Player A puts Grizzly Bears to hand", graveyard count drops) both work.
   PASS.
6. Attack trigger: Overlord placed on the battlefield (normal 5/5) attacking re-fired the
   mill 4 + return. PASS.
7. Scripted full-game regression, seeds 1/2/3/5/7: clean results, no draws, zero non-fatal
   errors. PASS.

## Result

Implemented Overlord of the Balemurk at full rules fidelity with a general, reusable Impending
mechanic (CR 702.175). All scenarios pass; build green; no non-fatal errors or draws.
