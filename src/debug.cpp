#include "debug.h"

#include "components/carddata.h"
#include "ecs/coordinator.h"
#include "error.h"
#include "systems/orderer.h"

const char *step_to_string(Step in_step) {
    switch (in_step) {
        case UNTAP:
            return "Untap";
            break;
        case UPKEEP:
            return "Upkeep";
            break;
        case DRAW:
            return "Draw";
            break;
        case FIRST_MAIN:
            return "First Main";
            break;
        case BEGIN_COMBAT:
            return "Begin Combat";
            break;
        case DECLARE_ATTACKERS:
            return "Declare Attackers";
            break;
        case DECLARE_BLOCKERS:
            return "Declare Blockers";
            break;
        case COMBAT_DAMAGE:
            return "Combat Damage";
            break;
        case END_OF_COMBAT:
            return "End of Combat";
            break;
        case SECOND_MAIN:
            return "Second Main";
            break;
        case END_STEP:
            return "End Step";
            break;
        case CLEANUP:
            return "Cleanup";
            break;
        default:
            return "ERROR UNREACHABLE";
            break;
    }
}

void print_step(const Game& cur_game) {
    printf("Active player is %s\n", cur_game.player_a_active ? "Player A" : "Player B");
    printf("Current step is %s, turn %zu, player %s's turn\n", step_to_string(cur_game.cur_step), cur_game.turn,
        cur_game.player_a_turn ? "A" : "B");
};

void print_library(std::shared_ptr<Orderer> orderer, Zone::Ownership owner) {
    auto library = orderer->get_library_contents(owner);
    size_t n = library.size();

    for (size_t i = 0; i < n; i++) {
        bool found = false;
        for (auto &&card_id : library) {
            auto &card = global_coordinator.GetComponent<CardData>(card_id);
            auto &zone = global_coordinator.GetComponent<Zone>(card_id);
            if (zone.distance_from_top == i) {
                printf("Card %s is %zu from top \n", card.uid.c_str(), i);
                found = true;
                break;
            }
        }
        if (!found) fatal_error("Missing card in library at index " + std::to_string(i));
    }
}

void print_hand(std::shared_ptr<Orderer> orderer, Zone::Ownership owner) {
    auto hand = orderer->get_hand(owner);

    printf("%s hand:\n", player_name(owner).c_str());
    for (auto &&card : hand) {
        auto &data = global_coordinator.GetComponent<CardData>(card);
        printf("%s\n", data.name.c_str());
    }
}

void print_stack(std::shared_ptr<Orderer> orderer) {
    auto stack = orderer->get_stack();
    for (size_t i = 0; i < stack.size(); i++){
        printf("%zu: %s", i, "TODO MAKE THIS A DESCRIPTION OF THE ABILITY");
    }
    return;
}

std::string player_name(Zone::Ownership owner) {
    if (owner == Zone::PLAYER_A)
        return "Player A";
    else
        return "Player B";
}
