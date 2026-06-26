# Voice of Victory  (vocab index 107)

## Oracle text
Mobilize 2 (Whenever this creature attacks, create two tapped and attacking 1/1 red Warrior
creature tokens. Sacrifice them at the beginning of the next end step.)

Your opponents can't cast spells during your turn.

(1/3 Human Soldier, mana cost {1}{W}.)

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/v/voice_of_victory.txt`
- Key tags:
  - `K:Mobilize:2` — the Mobilize keyword (rule 702.176) with count N = 2.
  - `S:Mode$ CantBeCast | ValidCard$ Card | Condition$ PlayerTurn | Caster$ Opponent` — static
    ability: the controller's opponents can't cast any spell during the controller's turn.

## Engine work
All changes are general (keyed on the tag's intended meaning), not card-specific.

- **Mobilize keyword (new mechanic).**
  - `src/ecs/events.h`: added `Events::CREATURE_ATTACKED` (id 12) — a per-attacker
    "whenever this creature attacks" event (the existing `CREATURE_ATTACKED_ALONE` only
    fires when exactly one creature attacks, which is wrong for an each-attacker trigger).
  - `src/action_processor.cpp`: fire `CREATURE_ATTACKED` (ENTITY = attacker, PLAYER =
    controller) once for **each** declared attacker, in the declare-attackers handler.
  - `src/systems/state_manager_statics.cpp` (`keyword_triggered_ability`): a `Mobilize:N`
    keyword now produces a TRIGGERED ability on `CREATURE_ATTACKED` with
    `trigger_only_self = true` (so it fires only when *this* creature attacks), category
    `"Mobilize"`, `amount = N`.
  - New effect `src/effects/effect_mobilize.cpp`:
    - `mobilize`: creates N tapped-and-attacking 1/1 red Warrior tokens (token script
      `r_1_1_warrior`), each attacking the same defender the Mobilize creature is attacking
      (508.4a — they are *put onto the battlefield attacking*, so no new attack is declared
      and no further "attacks" triggers fire). Registers a delayed trigger that fires at the
      next end step and sacrifices exactly those tokens.
    - `sacrifice_tokens`: the delayed end-step ability; sacrifices each token it created that
      is still on the battlefield.
  - `src/effects/effect_kind.{h,cpp}`, `effect_table.cpp`, `effects.h`: registered the two
    new effect kinds `Mobilize` and `SacrificeTokens`.
  - New resource `bin/resources/tokenscripts/r_1_1_warrior.txt` — the 1/1 red Warrior token
    (a token resource, not a card script).

- **CantBeCast "opponents can't cast during your turn" (extension of an existing mechanic).**
  - `src/components/static_ability.h`: added `cant_cast_by_opponent` (Caster$ Opponent).
  - `src/parse.cpp`: parse `Caster$ Opponent` on a CantBeCast static.
  - `src/systems/state_manager_statics.cpp` (`gather_active_statics`): evaluate the
    `Condition$ PlayerTurn` gate so `condition_met` is true only during the source
    controller's own turn.
  - `src/systems/rules_modifying.cpp` (`cast_prohibited`): a `cant_cast_by_opponent` static
    that is `condition_met` prohibits any caster who is not the static's controller (i.e.
    an opponent) from casting. This reuses the existing single cast-gate already called from
    `determine_legal_actions`, so it covers spells of every type at any priority window.

## Behavioral decisions (made in lieu of asking a human)
- **Mobilize tokens are created already attacking, not declared as attackers** (CR 508.4a):
  they are put onto the battlefield "tapped and attacking," so summoning sickness / vigilance
  / declare-attackers restrictions do not apply, and they do not themselves cause further
  "attacks" triggers. Implemented by setting `is_tapped`, `is_attacking`, and copying the
  source's `attack_target` directly rather than routing through attacker declaration.
- **"the next end step" is this turn's end step.** Attacks happen during the controller's own
  combat (CR 508.1), so the next end step is that same turn's end step, owned by the
  controller. The delayed trigger therefore registers with `fire_on_turn = cur_game.turn`,
  `owner_entity = controller`, firing on `END_STEP_BEGAN` (which carries the active player).
- **"can't cast spells" means all spells** (CR 601.3 / 603-independent cast restriction): the
  filter is `ValidCard$ Card`, so both creature and noncreature spells by opponents are
  blocked during the controller's turn. Verified a creature spell is also blocked.
- **The restriction is checked at every priority window**, including the opponent's responses
  on the controller's turn, because the single `cast_prohibited` gate runs inside
  `determine_legal_actions` for every cast option regardless of step.

## Tests
- Isolation (test_harness):
  - Mobilize attack trigger → Voice attacks: 2 tapped+attacking 1/1 Warrior tokens created;
    all three creatures deal combat damage (opponent 20 → 17); at the next end step both
    tokens are sacrificed/destroyed. Repeats correctly on subsequent turns.
  - CantBeCast → opponent holding Lightning Bolt with untapped Mountains is offered NO cast
    during the controller's turn (only Pass), but IS offered "Cast Lightning Bolt" on the
    opponent's own turn.
  - CantBeCast (creature) → opponent holding a castable creature is also blocked during the
    controller's turn (confirms all spell types are covered).
- Regression (test_harness --scripted, full games):
  - voice_test (mav with 4× Scythecat Cub → 4× Voice of Victory) vs mav, seeds 1-6: all
    decisive (no draws), no non-fatal errors.
  - voice_test mirror, seeds 1-10: Mobilize fires repeatedly in real games (up to 5 firings
    per game) with games completing cleanly.

## Result
implemented
