# Maze of Ith

**Vocab index:** 343

## Oracle text

```
Maze of Ith
Land
{T}: Untap target attacked creature. Prevent all combat damage that would be
dealt to and dealt by that creature this turn.
```

(The engine's local script names the target `Creature.attacking`; the printed
Oracle "attacked creature" and the script's "attacking creature" resolve to the
same attacker in a two-player game.)

## Forge script

Source: pre-existing local script (`bin/resources/cardsfolder/m/maze_of_ith.txt`).
Key tags:

- `A:AB$ Untap | Cost$ T | ValidTgts$ Creature.attacking | SubAbility$ DBPump` —
  the {T} ability untaps the target attacking creature, then chains the Effect.
- `SVar:DBPump:DB$ Effect | ReplacementEffects$ RPrevent1,RPrevent2 |
  RememberObjects$ Targeted | ExileOnMoved$ Battlefield` — a transient
  until-end-of-turn Effect that remembers the targeted creature and carries two
  named replacement shields.
- `SVar:RPrevent1:Event$ DamageDone | Prevent$ True | IsCombat$ True |
  ValidSource$ Card.IsRemembered` — prevent combat damage dealt **by** the
  remembered creature.
- `SVar:RPrevent2:Event$ DamageDone | Prevent$ True | IsCombat$ True |
  ValidTarget$ Card.IsRemembered` — prevent combat damage dealt **to** the
  remembered creature.

Maze of Ith has no mana ability at all — only this activated ability.

## Engine work

**Mechanics added (general): `combat-damage-prevention`** — a floating,
turn-scoped, per-creature shield that prevents all combat damage a remembered
creature would deal and/or be dealt this turn (CR 615). Not a Maze-specific
hack; it is a registry keyed on remembered entities, reusable by future
fog/prevention cards.

Files touched:

- `src/components/ability.h` — two parse flags on `Ability`:
  `effect_prevent_combat_damage_by_remembered` (ValidSource$ Card.IsRemembered)
  and `effect_prevent_combat_damage_to_remembered` (ValidTarget$ Card.IsRemembered).
- `src/parse.cpp` — the `DB$ Effect | ReplacementEffects$` parse hook previously
  recognized only Veil of Summer's `CantHappen` counter. Extended to (a) split
  the comma-separated SVar list and (b) additionally recognize
  `Event$ DamageDone | Prevent$ True | IsCombat$ True | ValidSource$/ValidTarget$
  Card.IsRemembered` shields, setting the two flags. The Veil path is unchanged.
- `src/classes/game.h` — new turn-scoped registry
  `std::vector<CombatDamagePreventionShield> combat_damage_prevention_shields`
  (each shield: `creature`, `prevent_as_source`, `prevent_as_target`) plus the
  query `bool Game::combat_damage_prevented(Entity source, Entity target) const`.
- `src/classes/game.cpp` — `combat_damage_prevented` implementation (matches a
  source-shield against the damage source or a target-shield against the
  recipient) and the cleanup clear (`combat_damage_prevention_shields.clear()`
  alongside the other "this turn" grants, CR 514.2).
- `src/effects/effect_grant_cast.cpp` — the `Effect` handler registers a shield
  on the remembered creature (the ability's inherited target /
  `remembered_entities`) when either prevention flag is set.
- `src/systems/state_manager_combat.cpp` — `deal_combat_damage` consults
  `game.combat_damage_prevented(source, target)` at each damage assignment:
  unblocked attacker→player/planeswalker, blocker→attacker, attacker→blocker,
  and trample-over. Prevented damage marks nothing, causes no life loss, fires
  no combat-damage trigger, and grants no lifelink. Combat damage assignment
  (CR 510.1c) still occurs (the `remaining` counter is consumed); only the
  *dealing* is prevented (CR 615).

The `AB$ Untap` on `Creature.attacking` plus `SubAbility$` chaining was already
supported; the untap of the attacking creature works (untapping a
vigilance-less attacker does not remove it from combat, but the prevention shield
makes it deal/take no damage — the classic "Maze of Ith fog one attacker").

## Behavioral decisions

- **How shields are stored/matched.** A shield remembers exactly one creature
  entity and two booleans. `prevent_as_source` matches when that creature is the
  damage's *source* (RPrevent1 / ValidSource); `prevent_as_target` matches when
  it is the *recipient* (RPrevent2 / ValidTarget). Maze sets both, so the fogged
  attacker neither deals nor takes combat damage.
- **End-of-turn expiry.** The shields are a sourceless "this turn" grant (the
  Effect belongs to no permanent — the Untap ability resolves and is gone). They
  are cleared at cleanup with the other turn-scoped grants (CR 514.2), so the
  prevention does not persist into the next turn.
- **Scope of prevention.** Only combat damage is prevented (the shields carry
  `IsCombat$ True`); the registry is consulted solely in the combat-damage path.

## Tests

Isolation (test harness, `--play` with seat keys):

- Maze of Ith preset on A, Grizzly Bears (no summoning sickness) on B. B attacks
  A; A activates Maze targeting the attacker. Result: "Grizzly Bears untaps",
  "All combat damage dealt to and dealt by Grizzly Bears is prevented this turn",
  and at combat damage "2 combat damage from Grizzly Bears is prevented". A's
  life stays at 20.
- Two-attacker case: both Grizzly Bears attack, Maze fogs only one. Result: the
  targeted attacker's "2 combat damage ... is prevented" while the second
  "Grizzly Bears deals 2 damage to Player A" — Maze fogs only the ONE targeted
  attacker.

CI gate: `train/.venv/bin/python train/ci_check.py --tier pygen,vocab,smoke`
(0 errors, no draws).

## Result

Implemented.
