# Mai, Scornful Striker (vocab index 161)

## Oracle text
First strike
Whenever a player casts a noncreature spell, they lose 2 life.

(Legendary Creature — Human Noble Ally, 2/2, mana cost `{1}{B}`.)

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/m/mai_scornful_striker.txt`). Key tags:

- `K:First Strike` — already supported.
- `T:Mode$ SpellCast | TriggerZones$ Battlefield | ValidCard$ Card.nonCreature | ValidActivatingPlayer$ Player | Execute$ TrigLoseLife`
  — a noncreature-spell-cast trigger that fires for **any** player (`ValidActivatingPlayer$ Player`,
  not `You`).
- `SVar:TrigLoseLife:DB$ LoseLife | Defined$ TriggeredActivator | LifeAmount$ 2` — the player who
  caused the trigger (the spell's caster) loses 2 life.

## Engine work
The trigger side (`Mode$ SpellCast` + `ValidCard$ Card.nonCreature`) already parsed to
`Events::NONCREATURE_SPELL_CAST` and fired for any player. The gap was the effect side: `LoseLife`
always applied to `ab.controller` (the source's controller); there was no `Defined$ TriggeredActivator`
routing to hit the player who *caused* the trigger.

Implemented a **general** `Defined$ TriggeredActivator` resolution (any effect that reads a Defined
player can use it), keyed on the tag's real meaning — no retag:

- `src/components/ability.h`: added `bool defined_triggered_activator` (parse flag) and
  `Zone::Ownership triggered_activator` (the player bound at trigger-fire time; `UNKNOWN` until then).
- `src/parse.cpp` (`Defined`/`DefinedPlayer` handler): `Defined$ TriggeredActivator` sets
  `defined_triggered_activator`. Mirrors the existing `TriggeredSpellAbility`/`You`/`Player.Opponent`
  Defined branches.
- `src/systems/state_manager_triggers.cpp`: new static helper `bind_triggered_activator(ab, player_entity)`
  recurses the triggered ability's tree (itself + `subabilities` + `charm_choices`) and, on any ability
  flagged `defined_triggered_activator`, records the triggering event's `Params::PLAYER` as
  `triggered_activator`. Called at trigger-fire time in `check_triggered_abilities` whenever the event
  carries a `PLAYER` param. The `LoseLife` lives in a `DB$` subability under `Execute$`, so the recursion
  is required. General: it binds for any effect/subability in the tree, not just LoseLife.
- `src/effects/effect_lose_life.cpp`: when `defined_triggered_activator` is set and
  `triggered_activator != UNKNOWN`, the life loss applies to that player instead of `ab.controller`.

## Behavioral decisions
- **Who triggers it:** the script's `ValidActivatingPlayer$ Player` means *any* player casting a
  noncreature spell — including Mai's own controller — so the trigger is not gated to opponents
  (`valid_player_is_you` stays false → no controller gating). CR 603.x: the activator is the player
  who cast the triggering spell; the `NONCREATURE_SPELL_CAST` event carries that caster as
  `Params::PLAYER`.
- **Who loses life:** `Defined$ TriggeredActivator` (CR 603.x — "the event that triggered it") =
  the caster of the noncreature spell, captured at fire time rather than the source's controller.

## Tests
Test harness (`train/test_harness.py`, inline hands/battlefield + semantic `--play`):
- **(c) controller casts noncreature:** Mai + Mountains on A, A casts Lightning Bolt →
  "Player A loses 2 life (now at 18)". The caster (= controller) loses 2.
- **(b) creature spell:** Mai on A, A casts Delver of Secrets (a creature) → Mai does **not**
  trigger (no LoseLife; only Delver's own PeekAndReveal triggers).
- **(a) opponent casts noncreature:** Mai on A, B casts Lightning Bolt → "Player B loses 2 life
  (now at 18)". The *caster* B loses the life, proving the activator routing is real (not a
  hardcoded controller/opponent).
- **Regression:** scripted full games, seeds 1/2/3, mono-black decks featuring Mai + noncreature
  spells (Thoughtseize/Fatal Push/Cabal Therapy/Mishra's Bauble/Dark Ritual). All three games
  resolved with a clean winner (no draws, no non-fatal errors). Mai triggered 5 times in the seed-1
  game. Temp decks cleaned up.

## Result
Implemented. `make HEADLESS=TRUE` builds clean (zero compiler diagnostics). Registered
`{"Mai, Scornful Striker", 161}` in `src/card_vocab.h`; `train/card_costs.py` regenerated.
