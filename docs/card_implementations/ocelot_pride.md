# Ocelot Pride  (vocab index 105)

## Oracle text
First strike, lifelink

Ascend

At the beginning of your end step, if you gained life this turn, create a 1/1 white Cat
creature token. Then if you have the city's blessing, for each token you control that
entered this turn, create a token that's a copy of it.

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/o/ocelot_pride.txt`
- Type: `Creature Cat`, mana cost `W`, P/T `1/1`.
- Key tags:
  - `K:First Strike`, `K:Lifelink` — already fully supported by combat/damage.
  - `K:Ascend` — city's blessing (702.131).
  - `T:Mode$ Phase | Phase$ End of Turn | ValidPlayer$ You | CheckSVar$ YouLifeGained |
    Execute$ TrigToken | TriggerZones$ Battlefield` — end-step trigger gated on
    "if you gained life this turn".
  - `SVar:YouLifeGained:Count$LifeYouGainedThisTurn` — the gate quantity.
  - `SVar:TrigToken:DB$ Token | TokenScript$ w_1_1_cat | TokenOwner$ You | SubAbility$ DBMassTokens`
    — make the base 1/1 white Cat.
  - `SVar:DBMassTokens:DB$ CopyPermanent | Condition$ Blessing |
    Defined$ Valid Card.token+YouCtrl+ThisTurnEntered` — the city's-blessing token-doubling clause.

## Engine work
All five mechanics were implemented as **general** handlers keyed on each tag's intended
meaning (no retagging / shortcutting). First Strike and Lifelink already worked.

1. **Life gained this turn** (`Count$LifeYouGainedThisTurn`).
   - `Player::life_gained_this_turn` added (`src/components/player.h`), reset for **both**
     players each turn during the CLEANUP step in `src/classes/game.cpp` — life can be gained
     on either player's turn, and the end-step trigger has already had its look by cleanup.
   - A single life-gain helper `player_gain_life(entity, amount)` (`src/game_queries.h`) raises
     `life_total` and accumulates `life_gained_this_turn`. **Every** life-gain site now routes
     through it so the counter cannot drift from life total: lifelink on noncombat damage
     (`src/components/damage.cpp`), combat lifelink (`src/systems/state_manager_combat.cpp`),
     and the `GainLife` effect (`src/effects/effect_gain_life.cpp`). (Life *loss* — PayLife,
     Sylvan Library, damage — is unchanged and never decrements the counter.)

2. **CheckSVar on a trigger = intervening-if** (CR 603.4). In `src/parse.cpp` the trigger-line
   `CheckSVar$`/`SVarCompare$` params now resolve the SVar to its `Count$` expression and store
   it as the intervening-if condition (`condition_present` + `intervening_if = true`), so a
   false condition removes the **whole** trigger from the stack (token AND copy do nothing) —
   the correct 603.4 behavior, distinct from a `ConditionCheckSVar` body-gate that would still
   chain subabilities. `evaluate_present_condition` (`src/systems/state_manager_actions.cpp`)
   gained a `Count$LifeYouGainedThisTurn` case (empty compare → GE1, i.e. "gained at least 1").

3. **Token** — `w_1_1_cat` already supported by the `Token` effect. (The token script file was
   fetched from Forge into the gitignored `bin/resources/tokenscripts/`.)

4. **Ascend / city's blessing** (CR 702.131). `Player::has_city_blessing` added; new
   `StateManager::update_city_blessing` (`src/systems/state_manager_statics.cpp`) runs each SBA
   pass: any controller of a permanent with the `Ascend` keyword who controls 10+ permanents and
   lacks the blessing gets it (a one-way latch, never lost — 702.131b/c). The keyword is read
   from source `CardData`, so this also covers non-creature Ascend permanents.

5. **CopyPermanent** — new effect (`src/effects/effect_copy_permanent.cpp`, registered in
   `effect_kind.{h,cpp}` / `effect_table.cpp` / `effects.h`). It snapshots the permanents
   matching its filter **before** creating any copy (so the freshly made copies — which also
   "entered this turn" — are not recursively re-copied; one copy per qualifying token), then
   creates a token copy of each under the controller's control. A token source is copied from
   its own `Token` component (the copyable snapshot, 707.2); a nontoken source is reconstructed
   from CardData + the Creature's *base* (printed) P/T.
   - `Defined$ Valid <filter>` is now parsed (`src/parse.cpp`) into the shared
     `valid_cards_filter`, the same place `ValidCards$` writes — not retagged.
   - `Condition$ Blessing` is parsed into `Ability::condition_city_blessing`; the resolve-time
     condition gate in `src/components/ability.cpp` skips the copy body (but still chains
     subabilities) unless the controller has the city's blessing.
   - The shared filter matcher `permanent_matches_cards_filter`
     (`src/effects/effect_put_counter_all.cpp`) gained two general qualifiers: `token`
     (`Permanent::is_token`) and `ThisTurnEntered` (`Permanent::entered_on_turn == cur_game.turn`).
   - `Permanent::entered_on_turn` added (`src/components/permanent.h`), set at both ETB sites
     (`state_manager_statics.cpp` for cards, `components/token.cpp` for tokens).

## Behavioral decisions (made in lieu of asking a human)
- **`TokenOwner$ You` ignored (cosmetic).** The `Token` effect already creates the token under
  the ability's controller, so `TokenOwner$ You` is redundant; it surfaces as the harmless
  pre-existing `WARNING: Unrecognized ability param: TokenOwner$ You`. No behavior change.
- **City's-blessing grant timing.** CR 702.131b grants the blessing "any time" the 10-permanent
  condition holds; the engine evaluates it inside the state-based-action fixpoint each pass, which
  is the engine's continuous-reapplication point — so it is granted as soon as the permanent count
  reaches 10 (verified the moment a 10th permanent is present). Once granted it is never revoked.
- **Copy fidelity.** A copied Cat token is reconstructed as another 1/1 white Cat. Token color is
  not modeled as a separate field anywhere in the engine (tokens carry name/types/P/T/keywords/
  abilities only), so copying those characteristics is a complete copy at the engine's fidelity.
- **No recursion / linear growth.** Per the snapshot-before-copy rule, each end step with the
  blessing yields exactly one base Cat plus one copy of each token that entered this turn — never
  exponential. Verified token count grows by +2 per qualifying end step, not by doubling.

## Tests
Isolation (`train/test_harness.py`, pre-set battlefields, seed 1):
- **Life gained → token; no blessing → no doubling.** A has `Ocelot Pride` (battlefield), attacks
  with it (lifelink, "Player A gains 1 life"). At A's end step the trigger fires:
  "Token created: 1/1 Cat Token", CopyPermanent resolves but makes no copy (only ~1 permanent, no
  city's blessing). Board: Ocelot Pride + exactly **1** Token. PASS.
- **No life gained → no token.** A has `Ocelot Pride`, declares no attackers (`attack:done`), gains
  no life. A's end step produces **no** "Token created" (intervening-if correctly fizzles the whole
  trigger). PASS.
- **City's blessing → doubling.** A has `Ocelot Pride` + 9 Plains (10 permanents): "Player A gets
  the city's blessing" fires. A attacks (lifelink, +1 life). At end step: "Token created: 1/1 Cat
  Token" **then** "Token copy created: 1/1 Cat Token" → board gains **2** Tokens. The copy did not
  re-copy itself. PASS.
- **Linear, terminating.** Long scripted game from the blessing board: each A end step makes exactly
  1 Cat + 1 copy (+2), game ends with a winner (no runaway, no draw). PASS.

Regression (`train/test_harness.py --scripted`, 6 games, seeds 1–6): deck `temp/ocelot_mav`
(the `mav` GW deck with 3 Ocelot Pride swapped in for 3 Thalia) vs `doomsday`. All 6 games finished
decisively (5 wins for the Ocelot deck, 1 loss to a Thassa's Oracle win), **no draws**, and no
non-fatal errors / asserts / tracebacks introduced. Only the pre-existing cosmetic
`Unrecognized ability param` warnings on unrelated cards (Green Sun's Zenith, Once Upon a Time,
Delver, Brainstorm, etc.) appear. (`train.py observe` could not be used — `torch` is not installed
in this environment — so the scripted regression was run directly through the harness.)

## Result
implemented
