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
- `Cost$ T Sac<1/CARDNAME>` (tap + sacrifice self), `ValidTgts$ Player`, and the
  `SubAbility$` chain are all handled by the activated-ability cost/target/subability
  machinery in `src/components/ability.cpp` and `src/action_processor.cpp`.
- `DB$ DelayedTrigger` → `EffectKind::DelayedTrigger` (`src/effects/effect_delayed_trigger.cpp`)
  registers the next-upkeep draw; `DB$ Draw` → `EffectKind::Draw`.
- **NEW — the `AB$ Reveal | Random$ True` "look at a random card in hand" clause is now
  fully implemented** (it was previously a `EffectKind::None` no-op). A new effect handler
  resolves it generally, mirroring how Mishra's Bauble models its private library peek:
  - `EffectKind::Reveal` was added (`src/effects/effect_kind.{h,cpp}`,
    `src/effects/effect_table.cpp`), dispatching to the new
    `effects::reveal()` in **`src/effects/effect_reveal.cpp`**.
  - The handler resolves the target/defined player (the `ValidTgts$ Player` target, else the
    controller), takes that player's **hand** via `Orderer::get_hand`, and if it is non-empty
    selects ONE card at random.
  - **Determinism:** the random index is drawn with `std::uniform_int_distribution` over the
    **game's seeded RNG `cur_game.gen`** — the same generator the orderer uses for shuffling
    and that Hymn to Tourach's random discard uses — so replays with a fixed `--seed` pick the
    same card every time.
  - **"Look at" modeling (mirrors Mishra's Bauble):** the look is **private to the ability's
    controller** (CR 701.16 "look at": the card is seen only by that player; it is NOT a public
    reveal). It is logged via `game_log_private(ab.controller, ...)` — exactly the channel
    Mishra's `PeekAndReveal | NoReveal$` uses for its top-of-library peek — so the effect is
    real and observable in perfect-information / narrative mode but does NOT leak into the
    public belief state (`mark_card_revealed` is deliberately NOT called, matching Mishra's).
  - The clause is implemented **generally** (any `AB$/DB$ Reveal` with `Random$ True` looking
    at a hand), reusing the existing `PeekParams` variant with a new `random_from_hand` flag
    and a `parse_reveal` hook that claims the `Random$` key. The `SubAbility$` slowtrip chain
    is untouched — `reveal()` returns `true` so the DelayedTrigger draw still chains.

## Behavioral decisions
- The look-at clause is now logged (no longer a silent no-op) but remains hidden information
  with zero game-state effect, consistent with CR 701.16 and with Mishra's Bauble's peek.
- `Mode$ Phase` and `TriggerDescription$` still emit cosmetic "Unrecognized ability param"
  warnings — the same class Mishra's Bauble already emits. `Random$` is now **recognized**
  (claimed by `parse_reveal`) and no longer warns.

## Tests
All via `train/test_harness.py` (HEADLESS build), card referred to as "Urzas Bauble".
- **(a) Look at opponent's hand.** Urza's Bauble on A's battlefield, B holds a known 4-card
  hand. Activating ({T}, Sac) targeting Player B printed:
  `Player A looks at a random card in Player B's hand: Forest`, `Player A sacrifices Urza's
  Bauble`, `Delayed trigger registered: Draw at next Upkeep.`, then on A's next upkeep the
  delayed trigger fired and A drew. The effect is no longer a no-op; the slowtrip draw still
  fires.
- **(b) Determinism.** Same scenario, seed 1 → "Forest" on two separate runs (identical);
  seed 5 → "Mountain"; seed 9 → "Forest". The pick is reproducible per seed and varies with
  the seed, confirming it uses `cur_game.gen`.
- **(c) Empty hand.** The handler guards `hand.empty()` before any indexing and logs
  "hand is empty" then returns true (slowtrip subability still chains) — structurally
  identical to Mishra's Bauble's empty-library guard, so no crash. (A naturally-empty hand is
  not readily constructible in the harness because every player auto-draws to a 7-card opening
  hand; verified by inspection of the guarded early-return.)
- **Regression.** Scripted full games, A = a deck with 4× Urza's Bauble + Bolt/Brainstorm/
  Ponder/lands vs B = `mav`, seeds 1/2/3: each game ended in a decisive winner (no draws),
  with no non-fatal errors / asserts / segfaults. Temp decks cleaned up.

## Result
Fully implemented. The look-at-random-card-in-hand peek is now a real, deterministic,
privately-logged effect (CR 701.16), modeled identically to Mishra's Bauble's private peek;
the slowtrip draw is unchanged.
