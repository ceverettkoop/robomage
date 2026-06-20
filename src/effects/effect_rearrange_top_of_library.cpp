#include "effects.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

bool rearrange_top_of_library(Ability &ab, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership owner = global_coordinator.GetComponent<Zone>(ab.source).owner;

    size_t num_cards = ab.amount;
    if (!ab.dynamic_amount_expr.empty())
        num_cards = evaluate_dynamic_amount(ab.dynamic_amount_expr, owner, orderer, ab.target);

    std::vector<Entity> lib = orderer->get_library_contents(owner);
    // Sort by distance_from_top ascending so lib[0] is the actual top card
    std::sort(lib.begin(), lib.end(), [](Entity a, Entity b) {
        return global_coordinator.GetComponent<Zone>(a).distance_from_top <
               global_coordinator.GetComponent<Zone>(b).distance_from_top;
    });
    // looking at top n only
    if (lib.size() > num_cards) lib.resize(num_cards);
    size_t actual = lib.size();
    std::vector<Entity> remaining = lib;

    game_log("%s looks at the top %zu card(s) of their library.\n", player_name(owner).c_str(), actual);

    std::vector<Entity> chosen_order;
    // Player picks N-1 cards; the last is automatic
    for (size_t pick = 0; pick + 1 < actual; pick++) {
        game_log("Choose which card goes %zu from top:\n", actual - pick);
        std::vector<LegalAction> pick_actions;
        for (auto card : remaining) {
            auto &cd = global_coordinator.GetComponent<CardData>(card);
            LegalAction la(PASS_PRIORITY, card, cd.name);
            la.category = ActionCategory::TOP_LIBRARY;
            pick_actions.push_back(la);
        }
        int choice = InputLogger::instance().get_input(pick_actions);
        chosen_order.push_back(remaining[static_cast<size_t>(choice)]);
        remaining.erase(remaining.begin() + choice);
    }
    // Last card is forced
    if (!remaining.empty()) {
        chosen_order.push_back(remaining[0]);
    }

    // Put cards back: chosen_order[0] should end up on top, so place in reverse
    for (auto it : chosen_order) {
        orderer->add_to_zone(false, it, Zone::LIBRARY);
    }

    if (ab.may_shuffle) {
        std::vector<LegalAction> shuffle_actions = {
            LegalAction(PASS_PRIORITY, std::string("Don't shuffle")),
            LegalAction(PASS_PRIORITY, std::string("Shuffle")),
        };
        shuffle_actions[0].category = ActionCategory::SHUFFLE;
        shuffle_actions[1].category = ActionCategory::SHUFFLE;
        int shuffle_choice = InputLogger::instance().get_input(shuffle_actions);
        if (shuffle_choice == 1) {
            orderer->shuffle_library(owner);
            game_log("%s shuffles their library.\n", player_name(owner).c_str());
        }
    }
    return true;
}

}  // namespace effects
