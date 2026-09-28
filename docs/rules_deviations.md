# Rules deviations and known gaps

Where the engine knowingly departs from the Comprehensive Rules
([`mtg_comprehensive_rules.txt`](mtg_comprehensive_rules.txt)) or leaves a rules edge case
unimplemented. Every entry was checked against the code when written. **Update this file when you
knowingly deviate, leave a gap, or fix one listed here.**

Scope: the engine simulates two-player (1-v-1) games only (a project decision, see `CLAUDE.md`).
Multiplayer rules (CR 800+) are out of scope and are not listed per rule below.

Entry format: **CR** rule(s) · **Rules:** what the CR says · **Engine:** what the code does ·
**Effect:** consequence, with a vocab card where one is exposed · **Code:** where it lives.

---

## 1. Intentional deviations (design decisions)

### 1.1 Phyrexian life is paid when the pip is chosen
- **CR** 601.2h, 601.5, 733.1
- **Rules:** the total cost, Phyrexian life included, is paid together at 601.2h.
- **Engine:** the 2 life is paid as each Phyrexian pip is chosen, during cost determination. A cancelled or rewound cast refunds it (`phyrexian_life_paid`).
- **Effect:** none visible, since nothing can happen between the pip choice and the payment. The early payment keeps each decision observable (the snapshot tier's "choice has an observable effect" invariant) without adding a "committed life" observation field.
- **Code:** `src/action_processor.cpp`: `run_cast_flow` (PHYREXIAN_PIP step), `rewind_cast`.

### 1.2 Name-a-card offers a limited candidate set
- **CR** 201.4
- **Rules:** the player may name any card in the Oracle card reference.
- **Engine:** the candidates are the distinct vocab cards in the relevant deck(s), filtered by `ValidCards$`, capped at `MAX_ACTIONS`. `CHOOSER_ONLY` offers one player's whole deck (Cabal Therapy: the target player's deck; Disruptor Flute: the opponent's). `BOTH_PLAYERS` offers both decks for land naming (Petrified Hamlet).
- **Effect:** a card that is in neither deck can't be named. Naming such a card has no effect except bluffing.
- **Code:** `src/name_card_choices.cpp`: `build_name_card_choices`; `src/effects/effect_name_card.cpp`.

### 1.3 Drawn games in a bo3 match
- **CR** 104.4a, 103.1 (match procedure is MTR-style policy)
- **Rules:** a drawn game counts for neither player. After a draw, the player who chose the starting player in that game chooses again.
- **Engine:** a draw (`GAME_RESULT: <n> draw`) counts for neither player, and the match continues until one player has won two games. The drawn game's starting player starts again, which matches 103.1 under the forced-play policy (see 2.1). A match that reaches `MAX_MATCH_GAMES` (10) games is a fatal error.
- **Code:** `src/game_driver.cpp`: `play_bo3_match`; `MAX_MATCH_GAMES` in `src/game_driver.h`.

### 1.4 Automatic mana payment in machine mode
- **CR** 601.2g–h, 605, 107.4e
- **Rules:** the player chooses which mana abilities to activate, which pool mana to spend, and how each hybrid symbol is paid.
- **Engine:** in machine mode, the part of a cost the pool doesn't cover is paid by a greedy payer. It prefers painless and single-colour sources, and falls back to the highest-yield ability of each source. Hybrid pips are resolved to the first payable assignment, with no decision. A player can still float mana first through mana-ability actions at priority. Generic pips are paid from the pool in enum order (W, U, B, R, G, then C), not colourless first.
- **Effect:** there are no tapping decisions. With {W}{C} floating, paying {1} spends the {W}, which can strand a later {W} spell in the same step.
- **Code:** `src/mana_system.cpp`: `auto_pay_mana`, `auto_pay_mana_attempt`, `pay_from_pool`; `src/action_processor.cpp`: `run_cast_flow` (HYBRID step, `resolve_hybrid_cost`).

### 1.5 An illegal menace block is released, not rewound
- **CR** 509.1b, 702.111b, 733.1
- **Rules:** a block that breaks a restriction makes the whole declaration illegal, and it is rewound.
- **Engine:** when blockers are confirmed, a lone blocker on a menace attacker is removed from the block and the other blocks stand.
- **Code:** `src/action_processor.cpp`: `release_illegal_menace_blockers`.

### 1.6 Menus are capped at 64 actions, with interchangeable choices collapsed
- **CR** none (an action-encoding limit)
- **Engine:** `MAX_ACTIONS` is 64. Before a menu is emitted, entries that make the same choice on interchangeable objects are dropped (for example, two copies of a card in a graveyard). A menu with more than 64 distinct entries is a fatal error in debug builds and is silently truncated in release builds. The largest menu seen in fuzzing is 41, so this is theoretical. If it is ever hit, the user decides whether to raise `MAX_ACTIONS` or split the choice into two stages.
- **Code:** `src/input_logger.cpp`: `interchangeable_choices`, `distinct_choice_indices`; `src/machine_io.cpp`: `populate_query`; `src/classes/gamestate.h`.

### 1.7 Some prompts block and are never search roots
- **CR** none (a search limitation; the rules outcome is correct)
- **Engine:** a few prompts inside resolution or a zone move still read input inline and can't suspend:
  - choosing among two or more applicable replacement effects (CR 616.1);
  - the Mox Diamond discard;
  - an Aura's enchant pick when the Aura is put onto the battlefield by a library search, a same-name move, a Remembered/ExiledWith return, Dig, DigUntil or ChangeZoneAll;
  - dredge offered on a mulligan redraw (unreachable).

  The search driver sees them with `SEARCHINFO safe=0`, so MCTS never takes a snapshot there. The user accepted this as long as search doesn't break.
- **Code:** `src/systems/replacement_effects.cpp`: `choose_one`, the MOVE_TO_ZONE discard in `apply_one`; `src/effects/effect_change_zone.cpp` and `effect_dig*.cpp`, `effect_change_zone_all.cpp` (`FrameCtx::blocking()` call sites); `src/search_server.cpp`: `search_loop_safe`.

### 1.8 Rulings the engine follows that are easy to mistake for bugs
- **CR 614.12:** a nonbasic land entering under Blood Moon or Magus of the Moon enters **untapped**. Its own "enters tapped" replacement effect doesn't apply, because the land won't have that ability on the battlefield (the shockland ruling). The same rule means Moonshadow cast under Humility gets no -1/-1 counters. `src/systems/replacement_effects.cpp`: `collect`.
- **CR 613.1f / 614.12:** Containment Priest under Humility exiles nothing, because the replacement effect comes from an ability Humility removes. `replacement_effects.cpp` (the header comment above `collect`).
- **CR 702.21a / 601.2c:** a permanent targeted by several "target" words of one spell or ability becomes the target once, so each of its Ward instances (and each BecomesTarget trigger) triggers once. The CR doesn't address repeated targeting directly; this follows the published rulings for other "becomes the target" triggers (Heroic/Valiant trigger once per spell). `src/action_processor.cpp`: `chosen_targets_of`, `trigger_ward_for_targets`, `fire_targeting_hooks`.
- **CR 603.8:** Dark Depths' Marit Lage is still created after the sacrifice makes the "no ice counters" check read false. It is a state trigger, not an intervening-if. `src/resolution.cpp`: `resolve_ability`.

---

## 2. Known gaps and unimplemented edge cases

### 2.1 Pregame and match procedure
- **CR 103.2b — the companion is revealed after mulligans.** The rules put the reveal before shuffling and mulligans. The engine reveals it in the FIAT_SETUP pregame stage, after the keep/mulligan decisions, so the mulligan observation lacks the opponent's companion. `src/game_driver.cpp`: `pregame_fiat_setup` → `setup_companions` (`src/companion.cpp`).
- **CR 103.1 — the player who chooses always plays first.** There is no play/draw choice: the loser of the previous game is forced to play first. In game 1 the engine always puts A on the play. Python randomizes the seat instead (`cli_spec.resolve_on_the_play`), but the actor's bo1 loop does not. `src/game_driver.cpp`: `play_bo3_match`; `src/main.cpp`; `src/actor/az_actor_main.cpp`.
- **CR 100.2a / 100.4a — deck limits are not checked.** Minimum 60 cards, sideboard at most 15, and the four-copy limit are not validated, because harness temp decks are small. A malformed count in a deck file hits `std::stoul` under `-fno-exceptions` and terminates with no message. `src/classes/deck.cpp`: `Deck::parse_text`.

### 2.2 Combat
- **CR 510.1c — damage split among blockers is limited to lethal-sized picks.** The rules let the attacker divide damage among its blockers freely. The engine asks for a split only when the attacker can't assign lethal damage to every blocker. It then offers "assign exactly lethal to blocker X" one pick at a time, and the leftover goes on the last pick (or the first blocker if none was picked). When the power covers every blocker's lethal damage, the split is automatic: lethal to each, then the excess on the last blocker or trampling over. This is a documented ML simplification. `src/action_processor.cpp`: `attacker_needs_assignment`, `run_damage_assignment`, `finish_pending_attacker`; `src/systems/state_manager_combat.cpp`: `deal_combat_damage`.
- **CR 702.7b/c — the second damage step reads current keywords.** The rules say the creatures that deal damage in the second step are those that didn't deal first-strike damage, plus double strikers. The engine uses the creature's current First Strike/Double Strike. A creature that gains first strike after the first-strike step deals no damage. One that loses it after dealing first-strike damage deals damage again. `src/systems/state_manager_combat.cpp`: `should_deal_damage`.

### 2.3 Continuous effects and layers
- **CR 613.8 — dependency is implemented only for land-type setters.** The only dependency the engine handles: a land-type setter (Blood Moon, Magus of the Moon) suppresses the statics of the nonbasic lands it affects (Yavimaya, Cradle of Growth). Every other within-layer order is timestamp order (613.7), and `resolve_dependencies` is a documented no-op. `src/systems/state_manager_statics.cpp`: `apply_type_changing_effects`; `src/systems/state_manager_layers.cpp`: `resolve_dependencies`.
- **CR 613.7 — unearth's haste has no timestamp.** The haste is recorded as an Animate keyword with no Animate timestamp. An ability-removal effect (Humility) therefore always removes it, even when the removal is older. `state_manager_statics.cpp`: `mark_unearthed_permanent`, `regrant_later_keywords`.
- **CR 613.7c / 613.1f — keyword counters and keyword-derived triggers count as older than any removal.** Keyword counters and the triggered abilities that come from a granted keyword (Prowess) are treated as predating every ability-removal effect, so a later removal always strips them and they never come back after one. This limitation is documented in the code. `state_manager_statics.cpp`: `recompute_abilities`.
- **CR 614.12 — a token keeps its own entering replacements under ability removal.** The rules say a permanent entering under a "loses all abilities" effect doesn't apply its own entering replacement effects. The engine's check reads `CardData`, which tokens don't have, so a token still applies its own "enters tapped" or "enters with counters". Latent in the vocab. `state_manager_statics.cpp`: `entering_object_loses_abilities`.

### 2.4 Triggers and state-based actions
- **CR 603.2 / 603.3 — triggers are collected at the next SBA check.** Events are buffered and matched when the next state-based-action check starts, not at the moment each event happens. If a trigger source leaves within the same resolution, after its event but before the next check, it misses a trigger that doesn't look back. Leaves-the-battlefield look-back (603.10a) treats all departures in one drained batch as simultaneous, so a watcher that left earlier in the batch still sees later departures. `src/systems/state_manager.cpp`: `state_based_effects`; `src/systems/state_manager_triggers.cpp`: `collect_triggered_abilities`, `match_departed_watcher_triggers`.
- **CR 704.3 / 704.5j — one legend-rule conflict per SBA check.** Only the first conflict (APNAP, then by name) is resolved in each check. Any other conflict is resolved in the next loop iteration, so the two sets of legend-rule deaths are not one simultaneous event. `state_manager.cpp`: `choose_legend_rule_keep`.

### 2.5 Casting, targeting and resolution
- **CR 603.3c / 700.2 — modal triggered abilities choose their mode at resolution.** Modal spells announce modes and targets at cast (601.2b/c). A modal triggered ability (Knight of Autumn's ETB) reaches the stack without an announcement. Its mode and targets are chosen as it resolves, so the opponent responds without seeing them, and targeting hooks (Ward, BecomesTarget) don't fire for those targets. `src/effects/effect_charm.cpp`: `charm` (the resolution-time fallback); the trigger placement in `state_manager_triggers.cpp` announces no mode.
- **CR 608.2b — partial illegality isn't tracked across sub-abilities.** The engine applies 608.2b only within one ability's multi-target list. If the parent's target is illegal, the whole chain fizzles, including subs with their own legal targets. A sub whose own target is illegal also skips its nested subs. Vocab exposure is small (commands and charms with per-mode targets resolve each mode on its own). `src/resolution.cpp`: `resolve_ability`, marked with a TODO there.
- **CR 601.2c / 602.2b — an activated ability's X isn't capped by the number of legal targets.** Candelabra of Tawnos ("untap X target lands") can announce X above the number of lands. The target loop then stops early, with fewer targets than X, when the activation should be illegal. Spells have this cap (`spell_xpaid_target_cap`). `src/action_processor.cpp`: `run_activation_flow` (X_LADDER), `run_target_select`.
- **Multi-pip Phyrexian dead end.** Each pip's "Pay {C}" option is checked against the cost accumulated so far. A later pip can still find neither life ≥ 2 nor payable mana. The cast then fails payment and is rewound cleanly (601.5), so no illegal state results, but that menu path is wasted. `run_cast_flow` (PHYREXIAN_PIP), `fail_cast_payment`.
- **CR 733.1 — a rewind doesn't reverse sacrificed mana sources.** The payment rewind restores the pool, tapped state, activation counts, life and delve exiles, but not a sacrificed mana source (Lotus Petal). Not reversing a mana ability is permitted by 733.1, but the pool is still restored to its pre-payment contents, so that source's mana is lost too. Reachable only through the interactive CLI payer's "Cancel casting". The machine payer simulates a payment before committing it, so a machine cast never fails partway through paying. `src/mana_system.cpp`: `snapshot_mana_state`, `restore_mana_state`.

### 2.6 Zones, libraries and searches
- **CR 701.23 — search and shuffle sequencing.** Three problems, all in `src/effects/effect_change_zone.cpp` (`change_zone`, the search loop):
  1. A multi-card search (Buried Alive) shuffles once per pick. Only RNG consumption differs.
  2. A "may search" that is declined still shuffles: the decline is taken as fail-to-find (Stoneforge Mystic, Ghost Quarter, Erode, Flagstones of Trokair, Price of Freedom). The rulings say no search means no shuffle. The effect is that known-top information (after Brainstorm or Ponder) is lost. `ShuffleNonMandatory$` and `Shuffle$ False` are not read.
  3. The shuffle is keyed on the first origin zone only, so a multi-zone search that lists Library second would never shuffle. Latent: Doomsday's `Origin$ Graveyard,Library` has `Shuffle$ False`.
- **CR 111.8 — a token that left the battlefield can move again.** The rule is enforced only for linked exile-until-host-leaves returns (`register_exile_until_host_leaves`). `Orderer::add_to_zone` has no general guard, so a same-resolution exile-and-return of a token would bring it back. Latent: no vocab flicker returns in the same resolution. `src/systems/orderer.cpp`: `Orderer::add_to_zone`.
- **Bottom of an empty ordered zone lands at depth 1.** A card put on the bottom of an empty library gets `distance_from_top` 1 instead of 0. Draws still take the minimum depth, so the rules outcome is unchanged. The known-top-of-library tracker, which is keyed by depth, records the card at slot 1. `Orderer::add_to_zone` (`back + 1`).
- **CR 101.4 / 303.4f — Show and Tell puts cards in sequentially.** The active player's card is put onto the battlefield before the non-active player chooses, instead of both choosing in APNAP order and the cards entering together. As a result, an Aura put by the non-active player can enchant the active player's just-entered creature, and "enters" triggers see a sequential order. `src/effects/effect_change_zone.cpp`: `each_player_put_from_hand`.
- **Until-your-next-turn Animate reversal.** `revert_until_turn_animates` clears the rest-of-game animate fields (`animate_make_creature`, `animate_set_pt`, and the base P/T) and strips Creature/Damage whenever the printed card isn't a creature. This happens even if another Animate (earthbend, `Duration$ Permanent`, end of turn) still animates the permanent. Latent: it needs Karn, the Great Creator's +1 on an already-animated artifact. `src/effects/effect_animate.cpp`: `revert_until_turn_animates`.

---

## 3. Out of scope because no vocab card needs it

These are left unimplemented, or implemented but never exercised, because no card in `src/card_vocab.h` needs them. Implement them when such a card is added.

- **Toxic** (CR 702.164, 120.3g): not implemented. Poison counters and the 704.5c poison SBA exist.
- **Battles** (CR 310, 120.3h): "Battle" is recognised as a permanent card type only. Battles can't be damaged, attacked or defeated.
- **Infect / wither** (CR 702.90, 702.80): the damage-result paths exist (`src/components/damage.cpp`) but no vocab card has these keywords, so they are untested.
- **Handlers limited to the forms the vocab uses:**
  - `gain_life` evaluates only the `Targeted$CardPower` dynamic amount; any other `LifeAmount$` expression reads the literal amount (`effect_gain_life.cpp`).
  - `multiply_counter` always doubles +1/+1 counters and ignores `CounterType$` (`effect_multiply_counter.cpp`).
  - `parse_activation_cost` passes any unrecognised cost token to the mana parser (`src/parse.cpp`).
- **Mechanics with no vocab card:** mutate, banding, morph/manifest/disguise/cloak (face-down cast), splice, entwine, cumulative upkeep, dungeons/initiative. No rules machinery exists for these.
