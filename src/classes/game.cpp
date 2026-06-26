#include "game.h"

#include "../cli_output.h"
#include "../components/creature.h"
#include "../components/damage.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"
#include "../ecs/events.h"
#include "../game_queries.h"
#include "../mana_system.h"
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
    player.poison_counters = 0;
    player.lands_played_this_turn = 0;
    global_coordinator.AddComponent(player_entity, player);
    return player_entity;
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
    if (ready_to_resolve()) {
        if (!stack_manager->is_empty()) {
            stack_manager->resolve_top(orderer);
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
                    // Phase in phased-out permanents controlled by active player
                    for (Entity entity = 0; entity < global_coordinator.GetMaxIssuedEntity(); ++entity) {
                        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
                        auto &perm_phase = global_coordinator.GetComponent<Permanent>(entity);
                        if (perm_phase.controller == active_player && perm_phase.is_phased_out) {
                            perm_phase.is_phased_out = false;
                            game_log("%s phases in\n", perm_phase.name.c_str());
                        }
                    }
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
                            if (!rev.skip_untap) permanent.is_tapped = false;
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
                    // PLAYER_DREW_CARD is fired per-card inside Orderer::draw_one
                    // (with the first-card-in-draw-step flag), so no emit here.
                    orderer->draw(active_player, 1);
                    break;
                case DRAW:
                    cur_step = FIRST_MAIN;
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
                    break;
                case SECOND_MAIN:
                    cur_step = END_STEP;
                    {
                        Event end_step_event(Events::END_STEP_BEGAN);
                        end_step_event.SetParam(Params::PLAYER, active_player_entity);
                        global_coordinator.SendEvent(end_step_event);
                    }
                    break;
                case END_STEP:
                    cur_step = CLEANUP;
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
                        }
                    }

                    // Reset per-turn state
                    revolt_player_a = false;
                    revolt_player_b = false;
                    auto &player = global_coordinator.GetComponent<Player>(active_player_entity);
                    player.lands_played_this_turn = 0;
                    player.spells_cast_this_turn = 0;
                    player.noncreature_spells_cast_this_turn = 0;
                    player.instant_sorcery_spells_cast_this_turn = 0;
                    player.cards_drawn_this_turn.clear();
                    player.cards_drawn_this_draw_step = 0;
                    // Also clear opponent's drawn-this-turn tracking
                    {
                        Entity opp_entity = player_a_turn ? player_b_entity : player_a_entity;
                        auto &opp = global_coordinator.GetComponent<Player>(opp_entity);
                        opp.cards_drawn_this_turn.clear();
                        opp.cards_drawn_this_draw_step = 0;
                    }
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

                    // End of turn, move to next turn
                    cur_step = UNTAP;
                    turn++;
                    player_a_turn = !player_a_turn;
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
