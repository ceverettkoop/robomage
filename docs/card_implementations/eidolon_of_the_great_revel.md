# Eidolon of the Great Revel  (vocab index 319)
## Oracle text
Whenever a player casts a spell with mana value 3 or less, Eidolon of the Great Revel deals 2 damage to that player.
## Forge script
- Source: pre-existing local
- Key tags: `T:Mode$ SpellCast | ValidCard$ Card.cmcLE3 | TriggerZones$ Battlefield | Execute$ TrigDamage`; `SVar:TrigDamage:DB$ DealDamage | Defined$ TriggeredActivator | NumDmg$ 2`
## Engine work
- **Fix 1 — literal-cmc-trigger-filter** (`src/parse.cpp`, SpellCast cmc filter parse ~line 2814): the `cmcLE/GE/EQ/...` filter parse only stored `trigger_cmc_expr` when the operand matched a known SVar key. A LITERAL numeric bound (`cmcLE3`) is not an SVar, so the filter was silently DROPPED and the trigger fired on every spell. Added an `else if` branch: when the operand is a bare integer, store the number string directly into `trigger_cmc_expr` (evaluated as an integer literal by `evaluate_sa_svar`, `state_manager_triggers.cpp` ~line 692) and keep the comparison op. Generalizes to any trigger with a literal cmc bound.
- **Fix 2 — DealDamage TriggeredActivator route** (`src/effects/effect_deal_damage.cpp` ~line 50): the `Defined$` player routing block handled `You`/`EachOpponent`/`TargetedController` but not `TriggeredActivator`, so `DealDamage Defined$ TriggeredActivator` fell through to the targeted-entity path with an unset target and tripped the non-fatal "Damage should have fizzled" guard. Added `ab.defined_triggered_activator` to the condition; `resolve_defined_player` already resolves it to the player captured at trigger-fire time.
- CR: 603.2 (triggered abilities), 202.3 (mana value comparison)
- Mechanics added (general): literal cmc bound on a SpellCast trigger filter; DealDamage to the triggered activator (the player who caused the trigger)
## Behavioral decisions
- none — behavior unambiguous. Fires on ANY player casting (self or opponent), damaging the caster; MV compared with `<= 3`.
## Tests
- Isolation (test_harness):
  - B casts Lightning Bolt (MV1) with A's Eidolon in play → Eidolon triggers, deals 2 to B (caster, 20→18), then Bolt resolves (A 20→17). No fizzle error.
  - B casts Deep Analysis (MV4) → NO Eidolon trigger, no damage (literal `cmcLE3` bound honored).
  - A casts Lightning Bolt (MV1) with A's own Eidolon in play → Eidolon deals 2 to A (caster, 20→18). Confirms it triggers on any player.
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
