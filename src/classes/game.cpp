#include "game.h"
#include "deck.h"
#include "../ecs/coordinator.h"
#include "../systems/stack_manager.h"


void Game::generate_players(const Deck &deck_a, const Deck &deck_b) {
    player_a_entity = gen_player(deck_a);
    player_b_entity = gen_player(deck_b);
}

Entity Game::gen_player(const Deck &deck) {
    return Entity();
}

void Game::pass_priority() {
    if (last_player_passed) {
        // Both players have now passed consecutively
        // This will be handled by advance_step
    }
    last_player_passed = true;
    player_a_has_priority = !player_a_has_priority;
}

void Game::take_action() {
    // When a player takes an action, reset the pass tracking
    last_player_passed = false;
}

bool Game::advance_step(std::shared_ptr<StackManager> stack_manager) {
    // Check if both players passed priority and stack is empty
    if (last_player_passed && stack_manager->is_empty()) {
        // Advance to next step
        last_player_passed = false;

        switch (cur_step) {
            case UNTAP:
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
                break;
            case DECLARE_ATTACKERS:
                cur_step = DECLARE_BLOCKERS;
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
                // End of turn, move to next turn
                cur_step = UNTAP;
                turn++;
                player_a_turn = !player_a_turn;
                break;
        }

        // Active player (whose turn it is) gets priority at start of new step
        player_a_has_priority = player_a_turn;

        return true;
    }

    return false;
}
