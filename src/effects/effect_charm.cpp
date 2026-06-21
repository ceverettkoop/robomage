#include "effects.h"

#include <string>
#include <vector>

#include "../action_processor.h"
#include "../classes/action.h"
#include "../cli_output.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

namespace effects {

bool charm(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Modal spell: present choices to the player, then resolve the chosen sub-ability
    game_log("Choose mode:\n");
    std::vector<LegalAction> mode_actions;
    std::vector<size_t> mode_indices;  // map action index to charm_choices index
    for (size_t i = 0; i < ab.charm_choices.size(); i++) {
        // Skip modes that require targets but have none
        Ability &candidate = ab.charm_choices[i];
        candidate.source = ab.source;
        candidate.controller = ab.controller;
        if (candidate.valid_tgts != "N_A" && candidate.target_min > 0 &&
            !has_legal_targets(candidate, orderer)) {
            continue;
        }
        std::string desc = (i < ab.charm_choice_descriptions.size() && !ab.charm_choice_descriptions[i].empty())
            ? ab.charm_choice_descriptions[i]
            : ("Mode " + std::to_string(i + 1));
        LegalAction la(PASS_PRIORITY, desc);
        la.category = ActionCategory::OTHER_CHOICE;
        mode_actions.push_back(la);
        mode_indices.push_back(i);
    }
    if (mode_actions.empty()) {
        game_log("No valid modes — charm fizzles\n");
        return false;
    }
    int choice = InputLogger::instance().get_input(mode_actions);
    size_t chosen_idx = mode_indices[static_cast<size_t>(choice)];
    Ability &chosen = ab.charm_choices[chosen_idx];
    // Target selection for the chosen mode
    if (chosen.valid_tgts != "N_A") {
        select_target(chosen, orderer, ab.controller);
    }
    chosen.resolve(orderer);
    // Skip subabilities — charm handles its own resolution
    return false;
}

}  // namespace effects
