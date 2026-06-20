#include "effects.h"

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

bool discard(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // RevealYouChoose: target player reveals hand, caster picks a card matching filter
    Zone::Ownership tgt_owner = Zone::PLAYER_A;
    if (global_coordinator.entity_has_component<Player>(ab.target)) {
        tgt_owner = (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    }
    std::vector<Entity> hand = orderer->get_hand(tgt_owner);
    game_log("%s reveals their hand:\n", player_name(tgt_owner).c_str());
    for (auto e : hand) {
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        game_log("  %s\n", cd.name.c_str());
    }

    // Filter by DiscardValid$ — "Card.nonLand", "Card.nonCreature+nonLand"
    std::vector<Entity> valid;
    for (auto e : hand) {
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        bool passes = true;
        if (!ab.discard_valid.empty()) {
            std::string filter = ab.discard_valid;
            if (filter.rfind("Card.", 0) == 0) filter = filter.substr(5);
            size_t fp = 0;
            while (fp < filter.size()) {
                size_t plus = filter.find('+', fp);
                if (plus == std::string::npos) plus = filter.size();
                std::string constraint = filter.substr(fp, plus - fp);
                if (constraint.rfind("non", 0) == 0) {
                    std::string excluded_type = constraint.substr(3);
                    for (auto &t : cd.types) {
                        if (t.name == excluded_type) {
                            passes = false;
                            break;
                        }
                    }
                }
                if (!passes) break;
                fp = plus + 1;
            }
        }
        if (passes) valid.push_back(e);
    }

    if (valid.empty()) {
        game_log("No valid cards to discard.\n");
    } else {
        bool prev_priority = cur_game.player_a_has_priority;
        cur_game.player_a_has_priority = (ab.controller == Zone::PLAYER_A);
        game_log("%s chooses a card to discard:\n", player_name(ab.controller).c_str());
        std::vector<LegalAction> discard_actions;
        for (auto e : valid) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            LegalAction la(PASS_PRIORITY, e, cd.name);
            la.category = ActionCategory::OTHER_CHOICE;
            discard_actions.push_back(la);
        }
        int choice = InputLogger::instance().get_input(discard_actions);
        Entity chosen = valid[static_cast<size_t>(choice)];
        auto &cd = global_coordinator.GetComponent<CardData>(chosen);
        game_log("%s discards %s\n", player_name(tgt_owner).c_str(), cd.name.c_str());
        orderer->add_to_zone(false, chosen, Zone::GRAVEYARD);
        cur_game.player_a_has_priority = prev_priority;
    }
    return true;
}

}  // namespace effects
