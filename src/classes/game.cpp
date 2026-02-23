#include "game.h"

#include "../components/damage.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"
#include "../mana_system.h"
#include "../systems/stack_manager.h"
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
    player.otp = false;  // Will be set properly for player A in caller
    player.life_total = 20;
    player.poison_counters = 0;
    player.energy_counters = 0;
    player.lands_played_this_turn = 0;
    global_coordinator.AddComponent(player_entity, player);
    return player_entity;
}

void Game::pass_priority() {
    if(player_a_has_priority) a_has_passed = true;
    if(!player_a_has_priority) b_has_passed = true;
    player_a_has_priority = !player_a_has_priority;
}

void Game::take_action() {
    // When a player takes an action, reset the pass tracking
    a_has_passed = false;
    b_has_passed = false;
}

bool Game::advance_step(std::shared_ptr<StackManager> stack_manager) {
    if(ready_to_resolve()){
        if(!stack_manager->is_empty()){
            stack_manager->resolve_top();
            //reset pass tracking when something has resolved
            a_has_passed = false;
            b_has_passed = false;
        }else{
        // Get active player entity
        Entity active_player_entity = player_a_turn ? player_a_entity : player_b_entity;
        Zone::Ownership active_player = player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;

        switch (cur_step) {
            case UNTAP:
                // Untap all permanents controlled by active player
                for (Entity entity = 0; entity < MAX_ENTITIES; ++entity) {
                    if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;

                    auto& permanent = global_coordinator.GetComponent<Permanent>(entity);
                    if (permanent.controller == active_player) {
                        permanent.is_tapped = false;
                        permanent.has_summoning_sickness = false;  // Clear summoning sickness
                    }
                }

                cur_step = UPKEEP;
                break;
            case UPKEEP:
                cur_step = DRAW;
                break;
            case DRAW:
                cur_step = FIRST_MAIN;
                break;
            case FIRST_MAIN:
                cur_step = BEGIN_COMBAT;
                break;
            case BEGIN_COMBAT:
                cur_step = DECLARE_ATTACKERS;
                attackers_declared = false;  // Reset for new combat
                break;
            case DECLARE_ATTACKERS:
                cur_step = DECLARE_BLOCKERS;
                blockers_declared = false;  // Reset for new combat
                break;
            case DECLARE_BLOCKERS:
                cur_step = COMBAT_DAMAGE;
                break;
            case COMBAT_DAMAGE:
                cur_step = END_OF_COMBAT;
                break;
            case END_OF_COMBAT:
                cur_step = SECOND_MAIN;
                break;
            case SECOND_MAIN:
                cur_step = END_STEP;
                break;
            case END_STEP:
                cur_step = CLEANUP;
                break;
            case CLEANUP:
                // Clear damage from all creatures
                for (Entity entity = 0; entity < MAX_ENTITIES; ++entity) {
                    if (global_coordinator.entity_has_component<Damage>(entity)) {
                        auto& damage = global_coordinator.GetComponent<Damage>(entity);
                        damage.damage_counters = 0;
                    }
                }

                // Reset lands played counter
                auto& player = global_coordinator.GetComponent<Player>(active_player_entity);
                player.lands_played_this_turn = 0;

                // Empty mana pools
                empty_mana_pool(Zone::PLAYER_A);
                empty_mana_pool(Zone::PLAYER_B);

                // End of turn, move to next turn
                cur_step = UNTAP;
                turn++;
                player_a_turn = !player_a_turn;
                break;
        }
        // Active player (whose turn it is) gets priority at start of new step
        player_a_has_priority = player_a_turn;
        // Reset pass tracking
        a_has_passed = false;
        b_has_passed = false;
        }
    }

    return false;
}

bool Game::is_mandatory_choice_pending() const {
    return pending_choice != NONE;
}
