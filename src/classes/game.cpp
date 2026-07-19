#include "game.h"

#include <algorithm>

#include "../cli_output.h"
#include "../day_night.h"
#include "../components/creature.h"
#include "../components/damage.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"
#include "../ecs/events.h"
#include "../effects/effects.h"
#include "../error.h"
#include "../game_queries.h"
#include "../mana_system.h"
#include "../saga.h"
#include "../systems/orderer.h"
#include "../systems/replacement_effects.h"
#include "../systems/stack_manager.h"
#include "../systems/state_manager.h"
#include "deck.h"

extern Coordinator global_coordinator;

bool Game::ready_to_resolve() {
    return a_has_passed && b_has_passed;
}

void Game::generate_players(const Deck &deck_a, const Deck &deck_b) {
    player_a_entity = gen_player(deck_a);
    player_b_entity = gen_player(deck_b);
}

Entity Game::gen_player(const Deck &deck) {
    Entity player_entity = global_coordinator.CreateEntity();
    Player player;
    player.life_total = 20;
    player.lands_played_this_turn = 0;
    global_coordinator.AddComponent(player_entity, player);
    return player_entity;
}

void Game::set_monarch(Entity player_entity) {
    if (monarch_entity == player_entity) return;  // already the monarch — no change (725.3)
    monarch_entity = player_entity;  // the previous monarch ceases to be the monarch (725.3)
    game_log("%s becomes the monarch.\n",
             player_name(player_entity == player_a_entity ? Zone::PLAYER_A : Zone::PLAYER_B).c_str());
}

void Game::record_action(int category, int card_vocab_idx, bool player_a) {
    action_history[action_history_write] = {category, card_vocab_idx, player_a, static_cast<int>(turn)};
    action_history_write = (action_history_write + 1) % ACTION_HISTORY_SIZE;
    if (action_history_count < ACTION_HISTORY_SIZE) action_history_count++;
}

void Game::clear_known_top_library(bool player_a_owner) {
    int *arr = player_a_owner ? known_top_library_a : known_top_library_b;
    for (int i = 0; i < KNOWN_TOP_LIBRARY_SIZE; i++) arr[i] = -1;
}

void Game::known_top_library_push(bool player_a_owner, int card_vocab_idx) {
    int *arr = player_a_owner ? known_top_library_a : known_top_library_b;
    for (int i = KNOWN_TOP_LIBRARY_SIZE - 1; i > 0; i--) arr[i] = arr[i - 1];
    arr[0] = card_vocab_idx;
}

void Game::known_top_library_remove_pos(bool player_a_owner, int pos) {
    if (pos < 0 || pos >= KNOWN_TOP_LIBRARY_SIZE) return;
    int *arr = player_a_owner ? known_top_library_a : known_top_library_b;
    for (int i = pos; i < KNOWN_TOP_LIBRARY_SIZE - 1; i++) arr[i] = arr[i + 1];
    arr[KNOWN_TOP_LIBRARY_SIZE - 1] = -1;
}

void Game::pass_priority() {
    if (player_a_has_priority) a_has_passed = true;
    if (!player_a_has_priority) b_has_passed = true;
    player_a_has_priority = !player_a_has_priority;
}

void Game::take_action() {
    // When a player takes an action, reset the pass tracking
    a_has_passed = false;
    b_has_passed = false;
    payment_fail_counts.clear();
}

bool Game::advance_step(std::shared_ptr<StackManager> stack_manager, std::shared_ptr<Orderer> orderer) {
    // will advance step and return true if step advanced
    // otherwise will resove stack or pass priority as needed
    // CR 502.4: no player receives priority during the untap step. The step's
    // turn-based actions (phasing, day/night check, untapping) run in the UNTAP
    // case below and the step advances straight to upkeep without a decision
    // window, regardless of pass tracking — this also covers the start of the
    // game, where cur_step begins at UNTAP with neither player having passed.
    // Nothing is resolved off the stack here either: an ability that triggers
    // during the untap step waits until a player would receive priority during
    // the upkeep (CR 603.3b), so it stays on the stack for the normal upkeep
    // priority round after the step change.
    if (ready_to_resolve() || cur_step == UNTAP) {
        if (!stack_manager->is_empty() && cur_step != UNTAP) {
            stack_manager->resolve_top(orderer);
            // A suspended resolution parked its decision for the loop top:
            // LEAVE the pass flags set, so the next iteration's advance_step
            // naturally re-enters resolve_top — the resume path. (Unreachable
            // until a handler is flipped suspendable.)
            if (resolution.active) return true;
            // reset pass tracking when something has resolved
            a_has_passed = false;
            b_has_passed = false;
            // remaining in current step
            return false;
        } else {
            // stack is empty and both players have passed
            //  step is changing
            Entity active_player_entity = player_a_turn ? player_a_entity : player_b_entity;
            Zone::Ownership active_player = player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;

            switch (cur_step) {
                case UNTAP: {
                    // Lapse any "until your next turn" Animate (Karn +1) this player created —
                    // its longer continuous-effect duration ends as their next turn begins.
                    effects::revert_until_turn_animates(active_player);
                    // Lapse any "until your next turn" player protection-from-everything grant
                    // protecting this player (The One Ring) — its duration ends as the protected
                    // player's next turn begins.
                    player_protection_from_everything.erase(
                        std::remove_if(player_protection_from_everything.begin(),
                                       player_protection_from_everything.end(),
                                       [active_player](const PlayerProtectionFromEverything &p) {
                                           return p.until_your_next_turn && p.player == active_player;
                                       }),
                        player_protection_from_everything.end());
                    // Lapse any "until your next turn" floating triggered ability this player
                    // created (Tamiyo, Seasoned Scholar's +2 "until your next turn, whenever ...")
                    // — its duration ends as the controller's next turn begins (CR 611.2).
                    floating_triggers.erase(
                        std::remove_if(floating_triggers.begin(), floating_triggers.end(),
                                       [active_player](const Ability &ft) {
                                           return ft.duration_until_your_next_turn &&
                                                  ft.controller == active_player;
                                       }),
                        floating_triggers.end());
                    // Phase in phased-out permanents controlled by active player
                    for (Entity entity = 0; entity < global_coordinator.GetMaxIssuedEntity(); ++entity) {
                        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
                        auto &perm_phase = global_coordinator.GetComponent<Permanent>(entity);
                        if (perm_phase.controller == active_player && perm_phase.is_phased_out) {
                            perm_phase.is_phased_out = false;
                            game_log("%s phases in\n", perm_phase.name.c_str());
                        }
                    }
                    // Second part of the untap step (CR 502.2 / 731.2): the day/night turn-based
                    // check, based on the turn that just ended. Runs after phasing, before untap.
                    day_night_untap_transition();
                    // Untap all permanents controlled by active player; reset per-turn counters
                    for (Entity entity = 0; entity < global_coordinator.GetMaxIssuedEntity(); ++entity) {
                        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;

                        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
                        if (permanent.controller == active_player) {
                            if (permanent.is_phased_out) continue;  // don't untap phased-out permanents
                            // Untap-prevention (Choke; rule 614.1d) is a replacement effect:
                            // dispatch an UNTAP event and skip untapping if it is replaced.
                            ReplacementEvent rev;
                            rev.type = ReplacementEvent::UNTAP;
                            rev.entity = entity;
                            rev.affected_player = active_player;
                            replacement::dispatch(rev);
                            // Stun counters (CR 122.1d): "If a permanent with a stun counter would
                            // become untapped, instead remove a stun counter from it." A tapped
                            // permanent with one or more STUN counters stays tapped and sheds one
                            // counter rather than untapping; otherwise it untaps normally.
                            if (!rev.skip_untap) {
                                // Counter type is stored verbatim from the script's CounterType$
                                // (Forge writes "Stun", CR 122.1d), so match that exact key.
                                if (permanent.is_tapped && get_counters(entity, "Stun") > 0) {
                                    add_counters(entity, "Stun", -1);
                                    game_log("%s has a stun counter removed instead of untapping.\n",
                                             permanent.name.c_str());
                                } else {
                                    permanent.is_tapped = false;
                                }
                            }
                            permanent.has_summoning_sickness = false;  // Clear summoning sickness
                            for (auto &ab : permanent.abilities) ab.activations_this_turn = 0;
                            permanent.loyalty_ability_activated_this_turn = false;  // 606.3 resets each of the controller's turns
                        }
                    }
                    cur_step = UPKEEP;
                    {
                        Event upkeep_event(Events::UPKEEP_BEGAN);
                        upkeep_event.SetParam(Params::PLAYER, active_player_entity);
                        global_coordinator.SendEvent(upkeep_event);
                    }
                    break;
                }
                case UPKEEP:
                    cur_step = DRAW;
                    // Fire DRAW_STEP_BEGAN before drawing
                    {
                        Event draw_step_event(Events::DRAW_STEP_BEGAN);
                        draw_step_event.SetParam(Params::PLAYER, active_player_entity);
                        global_coordinator.SendEvent(draw_step_event);
                    }
                    // first turn first player skips draw!
                    if (turn == 0 && player_a_turn == true) break;
                    // PLAYER_DREW_CARD is fired per-card inside the draw batch
                    // (with the first-card-in-draw-step flag), so no emit here.
                    // The turn-based draw runs as a resumable batch (pending_query
                    // tag TURN_DRAW): a dredge draw-replacement question (CR
                    // 702.52a) parks as a loop-top decision instead of blocking.
                    pending_draw.active = true;
                    pending_draw.player = active_player;
                    pending_draw.remaining = 1;
                    resume_pending_draws(*this, orderer);
                    // A dredge question parked the draw: return with the pass
                    // flags left true and the post-switch epilogue below DEFERRED
                    // — the parked query must be emitted against exactly the
                    // state the blocking prompt read (priority at the drawer,
                    // pass flags set, mana pools not yet emptied). The loop-top
                    // TURN_DRAW dispatch runs the epilogue via
                    // finish_suspended_turn_draw once the batch completes.
                    if (pending_draw.active) return true;
                    break;
                case DRAW:
                    cur_step = FIRST_MAIN;
                    {
                        Event first_main_event(Events::FIRST_MAIN_BEGAN);
                        first_main_event.SetParam(Params::PLAYER, active_player_entity);
                        global_coordinator.SendEvent(first_main_event);
                    }
                    // CR 714.3c turn-based action: as the active player's precombat main phase
                    // begins, put a lore counter on each Saga they control (firing the next
                    // chapter). Mirrors shed_impending_time_counters' built-in step hook.
                    saga_put_precombat_lore_counters(active_player);
                    break;
                case FIRST_MAIN:
                    cur_step = BEGIN_COMBAT;
                    {
                        Event begin_combat_event(Events::BEGIN_COMBAT_BEGAN);
                        begin_combat_event.SetParam(Params::PLAYER, active_player_entity);
                        global_coordinator.SendEvent(begin_combat_event);
                    }
                    break;
                case BEGIN_COMBAT:
                    cur_step = DECLARE_ATTACKERS;
                    attackers_declared = false;  // Reset for new combat
                    break;
                case DECLARE_ATTACKERS:
                    cur_step = DECLARE_BLOCKERS;
                    blockers_declared = false;  // Reset for new combat
                    break;
                case DECLARE_BLOCKERS: {
                    // Scan for first strikers / double strikers
                    has_first_strikers = false;
                    for (Entity e = 0; e < global_coordinator.GetMaxIssuedEntity(); ++e) {
                        if (!global_coordinator.entity_has_component<Creature>(e)) continue;
                        auto &cr = global_coordinator.GetComponent<Creature>(e);
                        if (!cr.is_attacking && !cr.is_blocking) continue;
                        if (creature_deals_first_strike_damage(cr)) {
                            has_first_strikers = true;
                            break;
                        }
                    }
                    if (has_first_strikers) {
                        cur_step = FIRST_STRIKE_DAMAGE;
                    } else {
                        cur_step = COMBAT_DAMAGE;
                    }
                    combat_damage_dealt = false;
                    break;
                }
                case FIRST_STRIKE_DAMAGE:
                    cur_step = COMBAT_DAMAGE;
                    combat_damage_dealt = false;
                    combat_damage_assignment.clear();  // T3.10: regular step re-decides for survivors
                    break;
                case COMBAT_DAMAGE:
                    cur_step = END_OF_COMBAT;
                    break;
                case END_OF_COMBAT:
                    // Clear all combat state from creatures
                    for (Entity entity = 0; entity < global_coordinator.GetMaxIssuedEntity(); ++entity) {
                        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
                        auto &creature = global_coordinator.GetComponent<Creature>(entity);
                        creature.is_attacking = false;
                        creature.attack_target = 0;
                        creature.is_blocking = false;
                        creature.blocking_target = 0;
                        creature.is_blocked = false;
                    }
                    combat_damage_assignment.clear();  // T3.10: drop any per-attacker assignments
                    cur_step = SECOND_MAIN;
                    {
                        Event second_main_event(Events::SECOND_MAIN_BEGAN);
                        second_main_event.SetParam(Params::PLAYER, active_player_entity);
                        global_coordinator.SendEvent(second_main_event);
                    }
                    break;
                case SECOND_MAIN:
                    cur_step = END_STEP;
                    {
                        Event end_step_event(Events::END_STEP_BEGAN);
                        end_step_event.SetParam(Params::PLAYER, active_player_entity);
                        global_coordinator.SendEvent(end_step_event);
                    }
                    // CR 702.175e: the "remove a time counter" impending shed is a triggered
                    // ability (produced from END_STEP_BEGAN in check_triggered_abilities and put
                    // on the stack), not a step side effect — so nothing is done inline here.
                    break;
                case END_STEP:
                    cur_step = CLEANUP;
                    {
                        Event cleanup_event(Events::CLEANUP_BEGAN);
                        cleanup_event.SetParam(Params::PLAYER, active_player_entity);
                        global_coordinator.SendEvent(cleanup_event);
                    }
                    break;
                case CLEANUP:
                    // Clear damage from all creatures; reset prowess bonus
                    for (Entity entity = 0; entity < global_coordinator.GetMaxIssuedEntity(); ++entity) {
                        if (global_coordinator.entity_has_component<Damage>(entity)) {
                            auto &damage = global_coordinator.GetComponent<Damage>(entity);
                            damage.damage_counters = 0;
                            damage.has_deathtouch_damage = false;
                        }
                        if (global_coordinator.entity_has_component<Creature>(entity)) {
                            auto &cr = global_coordinator.GetComponent<Creature>(entity);
                            if (cr.prowess_bonus != 0 || cr.eot_power_bonus != 0 ||
                                cr.eot_toughness_bonus != 0) {
                                cr.prowess_bonus = 0;
                                cr.eot_power_bonus = 0;
                                cr.eot_toughness_bonus = 0;
                                recompute_pt(cr);
                            }
                            // Drop "until end of turn" keyword grants (e.g. Haste); the
                            // static pass re-merges these onto cr.keywords each pass, so
                            // clearing the bucket here lets them lapse at cleanup (514.2).
                            cr.eot_keywords.clear();
                            // "Can't be blocked this turn" (Kappa Cannoneer) lapses at cleanup.
                            cr.cant_be_blocked_this_turn = false;
                        }
                        // "Loses <keyword> until end of turn" (Shadowspear's AB$ AnimateAll |
                        // RemoveKeywords$) lapses at cleanup (514.2). On a permanent (any type),
                        // so cleared outside the Creature branch above.
                        if (global_coordinator.entity_has_component<Permanent>(entity)) {
                            auto &perm = global_coordinator.GetComponent<Permanent>(entity);
                            perm.removed_keywords_eot.clear();
                            // "Until end of turn" Animate (CR 514.2) lapses now: erase the
                            // EOT-added types, and if this EOT animate is what made a noncreature
                            // permanent a creature (a crewed-by-trigger Vehicle like The
                            // Fantasticar), strip its bootstrapped Creature/Damage components so it
                            // stops being a creature — unless it is a creature by a permanent means.
                            if (!perm.animate_added_types_eot.empty() ||
                                perm.animate_make_creature_eot) {
                                for (const auto &t : perm.animate_added_types_eot)
                                    perm.types.erase(t);
                                perm.animate_added_types_eot.clear();
                                if (perm.animate_make_creature_eot) {
                                    perm.animate_make_creature_eot = false;
                                    bool still_creature = perm.animate_make_creature;
                                    if (!still_creature &&
                                        global_coordinator.entity_has_component<CardData>(entity))
                                        still_creature = is_creature_card(
                                            global_coordinator.GetComponent<CardData>(entity));
                                    if (!still_creature) {
                                        if (global_coordinator.entity_has_component<Creature>(entity))
                                            global_coordinator.RemoveComponent<Creature>(entity);
                                        if (global_coordinator.entity_has_component<Damage>(entity))
                                            global_coordinator.RemoveComponent<Damage>(entity);
                                    }
                                }
                            }
                        }
                    }

                    // Reset per-turn state
                    revolt_player_a = false;
                    revolt_player_b = false;
                    // "You may cast that card this turn" grants (Emry) expire at cleanup (601.3e).
                    may_cast_this_turn.clear();
                    // Floating "this turn" triggered abilities (Forth Eorlingas!'s become-monarch
                    // trigger, CR 603.7e) last only their turn of creation; drop them at cleanup.
                    // A Duration$ UntilYourNextTurn floating trigger (Tamiyo, Seasoned Scholar's +2)
                    // survives cleanup — it is removed at its controller's next untap step instead.
                    floating_triggers.erase(
                        std::remove_if(floating_triggers.begin(), floating_triggers.end(),
                                       [](const Ability &ft) { return !ft.duration_until_your_next_turn; }),
                        floating_triggers.end());
                    // Impulse-cast permissions (Amped Raptor) likewise last only "this turn".
                    impulse_cast_permission.clear();
                    auto &player = global_coordinator.GetComponent<Player>(active_player_entity);
                    player.lands_played_this_turn = 0;
                    // Snapshot this (the ending) turn's active player's OWN-TURN spell count before
                    // the per-turn reset, for the next turn's untap day/night check (CR 502.2 /
                    // 731.2). Both players' spells_cast_this_turn are reset to 0 each cleanup (the
                    // opponent's just below), so this counter holds only spells cast during this
                    // turn — no start-of-turn baseline subtraction is needed.
                    prev_turn_active_spell_count = static_cast<int>(player.spells_cast_this_turn);
                    player.spells_cast_this_turn = 0;
                    player.noncreature_spells_cast_this_turn = 0;
                    player.instant_sorcery_spells_cast_this_turn = 0;
                    player.spell_colors_cast_this_turn.clear();
                    player.cards_drawn_this_turn.clear();
                    player.cards_drawn_this_draw_step = 0;
                    // Also clear opponent's drawn-this-turn tracking
                    {
                        Entity opp_entity = player_a_turn ? player_b_entity : player_a_entity;
                        auto &opp = global_coordinator.GetComponent<Player>(opp_entity);
                        opp.cards_drawn_this_turn.clear();
                        opp.cards_drawn_this_draw_step = 0;
                        opp.spell_colors_cast_this_turn.clear();
                        // Reset the opponent's per-turn spell COUNTS too (CR 702.40a "cast before
                        // it this turn"): an instant the opponent cast during the active player's
                        // turn must not persist into the opponent's own next turn, or
                        // storm_count_this_turn (sum of both players' spells_cast_this_turn − 1)
                        // overcounts. "This turn" is the current turn for both players, so both
                        // counters are zero at the start of each new turn. The active player's
                        // own-turn count was already snapshotted into prev_turn_active_spell_count
                        // above for the day/night check before its reset, so that is unaffected.
                        opp.spells_cast_this_turn = 0;
                        opp.noncreature_spells_cast_this_turn = 0;
                        opp.instant_sorcery_spells_cast_this_turn = 0;
                    }
                    // Turn-long continuous effects created by an instant/sorcery (Veil of Summer:
                    // "Spells you control can't be countered this turn" + "hexproof from blue and
                    // from black until end of turn") lapse at cleanup (CR 514.2).
                    cant_counter_spells_of.clear();
                    hexproof_from_colors_this_turn.clear();
                    // An "until end of turn" player protection-from-everything grant lapses at
                    // cleanup; an "until your next turn" grant persists (reverted at that player's
                    // untap step instead — see the UNTAP case above).
                    player_protection_from_everything.erase(
                        std::remove_if(player_protection_from_everything.begin(),
                                       player_protection_from_everything.end(),
                                       [](const PlayerProtectionFromEverything &p) {
                                           return !p.until_your_next_turn;
                                       }),
                        player_protection_from_everything.end());
                    // "Life gained this turn" (Ocelot Pride) and "tokens entered this turn"
                    // reset for BOTH players each turn — life can be gained on either player's
                    // turn, and the end-step trigger above has already checked them. Done in
                    // cleanup so the just-fired end step still saw this turn's totals.
                    global_coordinator.GetComponent<Player>(player_a_entity).life_gained_this_turn = 0;
                    global_coordinator.GetComponent<Player>(player_b_entity).life_gained_this_turn = 0;

                    // Reset per-trigger resolution counts
                    ability_resolution_counts.clear();

                    // Empty mana pools
                    empty_mana_pool(Zone::PLAYER_A);
                    empty_mana_pool(Zone::PLAYER_B);

                    // End of turn, move to next turn. An extra turn (CR 500.7 / 720) takes
                    // priority over the normal active-player flip: if a player is owed an extra
                    // turn, that player (the most recently added — extra_turns is a LIFO stack)
                    // takes the next turn instead of passing to the opponent.
                    cur_step = UNTAP;
                    turn++;
                    if (!extra_turns.empty()) {
                        Zone::Ownership next_active = extra_turns.back();
                        extra_turns.pop_back();
                        player_a_turn = (next_active == Zone::PLAYER_A);
                    } else {
                        player_a_turn = !player_a_turn;
                    }
                    break;
            }
            // if the new step is untap or cleanup, we pretend both players passed
            // hacky
            if (cur_step == UNTAP || cur_step == CLEANUP) {
                a_has_passed = true;
                b_has_passed = true;
            } else {
                // otherwise we now get active player priority
                player_a_has_priority = player_a_turn;
                // Reset pass tracking
                a_has_passed = false;
                b_has_passed = false;
            }
            // any case where we are returning true, mana pool is now emptied
            empty_mana_pool(Zone::PLAYER_A);
            empty_mana_pool(Zone::PLAYER_B);
            return true;
        }
    } else {
        // return false if not ready to resolve, meaning someone has priority
        return false;
    }
}

bool Game::is_mandatory_choice_pending() const {
    return pending_choice != NONE;
}

void Game::finish_suspended_turn_draw() {
    // Exactly the post-switch epilogue for a priority-bearing step (cur_step
    // is DRAW here, never UNTAP/CLEANUP): active player gets priority, pass
    // tracking resets, mana pools empty across the step change.
    player_a_has_priority = player_a_turn;
    a_has_passed = false;
    b_has_passed = false;
    empty_mana_pool(Zone::PLAYER_A);
    empty_mana_pool(Zone::PLAYER_B);
}

void resume_pending_draws(Game &game, std::shared_ptr<Orderer> orderer) {
    Game::PendingDrawRT &pd = game.pending_draw;
    while (pd.active) {
        // A latched TURN_DRAW answer from the previous arm: apply it first —
        // draw normally (option 0) or one dredge — restoring the pre-arm
        // priority seat exactly as the blocking prompt's post-get_input
        // restore did.
        if (game.pending_query.active) {
            PendingQuery &pq = game.pending_query;
            if (pq.tag != PendingQuery::TURN_DRAW || !pq.answered)
                fatal_error("resume_pending_draws: foreign or unanswered pending query parked");
            std::vector<replacement::DrawReplacementOption> opts;
            std::vector<LegalAction> menu =
                replacement::collect_draw_replacements(pd.player, &opts);
            // Purity tripwire: nothing runs between suspend and resume, so the
            // re-derived menu must match the parked one.
            if (menu.size() != pq.menu.size())
                fatal_error("resume_pending_draws: menu size changed between arm and resume");
            int choice = pq.answer;
            game.player_a_has_priority = pq.prev_priority;
            pq = PendingQuery{};
            if (choice == 0)
                orderer->perform_draw(pd.player);
            else
                orderer->apply_dredge(pd.player, opts[static_cast<size_t>(choice) - 1].source,
                                      opts[static_cast<size_t>(choice) - 1].mill);
            pd.remaining--;
            continue;
        }
        // Per-draw ended bail, mirroring Orderer::draw's loop guard (a decked
        // draw ends the game mid-batch).
        if (pd.remaining <= 0 || game.ended) {
            pd = Game::PendingDrawRT{};
            return;
        }
        std::vector<replacement::DrawReplacementOption> opts;
        std::vector<LegalAction> menu = replacement::collect_draw_replacements(pd.player, &opts);
        if (menu.empty()) {
            // No dredge applies — the promptless common case, exactly today's
            // draw_one with an empty replacement dispatch.
            orderer->perform_draw(pd.player);
            pd.remaining--;
            continue;
        }
        // Park the dredge question for the loop top (tag TURN_DRAW): persist
        // priority at the drawing player (the blocking prompt's repoint) and
        // arm with the ambient pending-decision source (the blocking prompt
        // ran scope-less — source 0 at a turn-based draw).
        PendingQuery &pq = game.pending_query;
        pq = PendingQuery{};
        pq.tag = PendingQuery::TURN_DRAW;
        pq.active = true;
        pq.menu = std::move(menu);
        pq.chooser_is_a = (pd.player == Zone::PLAYER_A);
        pq.decision_source = game.pending_decision_source;
        pq.prev_priority = game.player_a_has_priority;
        game.player_a_has_priority = pq.chooser_is_a;
        return;
    }
}
