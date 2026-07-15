#include "effects.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

bool sylvan_library(Ability &ab, std::shared_ptr<Orderer> orderer) {
    PendingDecisionScope pending_scope(ab.source);
    // Draw 2, then for each card drawn this turn still in hand, choose: pay 4 life or put on top
    Zone::Ownership ctrl = ab.controller;
    Entity ctrl_entity = (ctrl == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
    auto &pl = global_coordinator.GetComponent<Player>(ctrl_entity);

    // Draw 2 cards
    orderer->draw(ctrl, 2);
    game_log("%s draws 2 cards (Sylvan Library)\n", player_name(ctrl).c_str());

    // Get cards drawn this turn that are still in hand
    std::vector<Entity> drawn_in_hand;
    for (auto e : pl.cards_drawn_this_turn) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location == Zone::HAND && z.owner == ctrl) {
            drawn_in_hand.push_back(e);
        }
    }

    // Player chooses 2 cards (or fewer if not enough in hand)
    size_t to_choose = std::min(drawn_in_hand.size(), static_cast<size_t>(2));
    std::vector<Entity> chosen_cards;
    for (size_t pick = 0; pick < to_choose; pick++) {
        game_log("Choose a card drawn this turn (%zu remaining):\n", to_choose - pick);
        std::vector<LegalAction> choose_actions;
        for (auto e : drawn_in_hand) {
            // Skip already chosen
            bool already = false;
            for (auto c : chosen_cards) {
                if (c == e) {
                    already = true;
                    break;
                }
            }
            if (already) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            LegalAction la(PASS_PRIORITY, e, cd.name);
            la.category = ActionCategory::SYLVAN_CHOICE;
            choose_actions.push_back(la);
        }
        if (choose_actions.empty()) break;
        int choice = InputLogger::instance().get_input(choose_actions);
        chosen_cards.push_back(choose_actions[static_cast<size_t>(choice)].source_entity);
    }

    // For each chosen card: pay 4 life or put on top of library
    for (auto card : chosen_cards) {
        auto &cd = global_coordinator.GetComponent<CardData>(card);
        game_log("For %s: pay 4 life or put on top of library?\n", cd.name.c_str());
        std::vector<LegalAction> pay_actions = {
            LegalAction(PASS_PRIORITY, std::string("Pay 4 life")),
            LegalAction(PASS_PRIORITY, std::string("Put on top of library")),
        };
        pay_actions[0].category = ActionCategory::SYLVAN_CHOICE;
        pay_actions[0].option_ordinal = 1;  // 1 = pay 4 life
        pay_actions[1].category = ActionCategory::SYLVAN_CHOICE;
        pay_actions[1].option_ordinal = 0;  // 0 = put on top of library
        int choice = InputLogger::instance().get_input(pay_actions);
        if (choice == 0) {
            pl.life_total -= 4;
            game_log("%s pays 4 life (now at %d)\n", player_name(ctrl).c_str(), pl.life_total);
        } else {
            orderer->add_to_zone(false, card, Zone::LIBRARY);
            game_log("%s puts %s on top of library\n", player_name(ctrl).c_str(), cd.name.c_str());
        }
    }
    return true;
}

}  // namespace effects
