# Urza's Bauble (vocab index 146)

## Oracle text
{T}, Sacrifice Urza's Bauble: Look at a card at random in target player's hand. You draw a
card at the beginning of the next turn's upkeep.

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/u/urzas_bauble.txt`).
Key tags:
- `A:AB$ Reveal | Cost$ T Sac<1/CARDNAME> | ValidTgts$ Player | Random$ True | SubAbility$ DelTrigSlowtrip`
- `SVar:DelTrigSlowtrip:DB$ DelayedTrigger | NextTurn$ True | Mode$ Phase | Phase$ Upkeep | ValidPlayer$ Player | Execute$ DrawSlowtrip`
- `SVar:DrawSlowtrip:DB$ Draw | Defined$ You`

This is the same slowtrip-bauble template as Mishra's Bauble (vocab index 27), the proven
precedent. The only script difference: Mishra's uses `AB$ PeekAndReveal` (look at top of
target's *library*); Urza's uses `AB$ Reveal | Random$ True` (look at a *random card in the
target's hand*).

## Engine work
None new — covered.
- `Cost$ T Sac<1/CARDNAME>` (tap + sacrifice self), `ValidTgts$ Player`, and the
  `SubAbility$` chain are all handled by the activated-ability cost/target/subability
  machinery in `src/components/ability.cpp` and `src/action_processor.cpp`.
- `DB$ DelayedTrigger` → `EffectKind::DelayedTrigger` (`src/effects/effect_delayed_trigger.cpp`)
  registers the next-upkeep draw; `DB$ Draw` → `EffectKind::Draw`.
- `AB$ Reveal` maps to `EffectKind::None` (no registered handler), so it resolves as a
  no-op that still chains its subabilities (`Ability::resolve` runs subabilities when the
  handler is null). The "look at a random card in target player's hand" is hidden
  information with no game-state effect, so the no-op produces correct game state.

## Behavioral decisions
- The look-at-a-random-card-in-hand clause has no engine surface and is not visibly
  logged (unlike Mishra's `PeekAndReveal`, which logs the library peek). It has zero
  game-state effect, so this does not affect correctness of play. Flagged for review.
- `Random$ True`, `Mode$ Phase`, `TriggerDescription$` emit cosmetic "Unrecognized ability
  param" warnings — the same class of warning Mishra's Bauble already emits (`Mode$ Phase`).

## Tests
- Scenario: Urza's Bauble on Player A's battlefield, activated targeting Player B.
  Observed: "Player A sacrifices Urza's Bauble" (tap+sac cost), ability put on stack
  targeting Player B, resolved (category: Reveal, no-op), "Delayed trigger registered: Draw
  at next Upkeep." Then on Player A's next upkeep: "Delayed trigger fires." → "Player A
  draws" — the slowtrip card draw correctly goes to the Bauble's controller. Behavior is
  identical to the proven Mishra's Bauble template. Result: pass (game-relevant effect);
  the look-at-hand info clause is not demonstrable.

## Result
Implemented (registration only; mechanic pre-existing via the Mishra's Bauble slowtrip
template). needs_review: the hidden-information look-at-hand clause has no engine surface.
