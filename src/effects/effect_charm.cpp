#include "effects.h"

#include <string>
#include <vector>

#include "../action_processor.h"
#include "../classes/action.h"
#include "../cli_output.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

namespace effects {

// Resolves a single already-chosen charm mode: select its target(s) (if any) then
// resolve it. Factored out so the choose-one and choose-two paths share one body.
static void resolve_chosen_mode(Ability &parent, Ability &chosen, std::shared_ptr<Orderer> orderer) {
    chosen.source = parent.source;
    chosen.controller = parent.controller;
    if (chosen.valid_tgts != "N_A") {
        select_target(chosen, orderer, parent.controller);
    }
    chosen.resolve(orderer);
}

// Modal spell (CR 700.2): "Choose one/two —". The number of modes to pick is CharmNum$
// (default 1). CR 601.2b requires the chosen modes to be different, so each pick is
// removed from the menu before the next. Forge resolves the modes top-to-bottom; matching
// that, each chosen mode has its targets selected and resolves immediately before the next
// mode is chosen. (This engine chooses modes at resolution rather than on cast — a
// simplification shared by every Charm card here; see docs/card_implementations.)
bool charm(Ability &ab, std::shared_ptr<Orderer> orderer) {
    int to_pick = ab.charm_num < 1 ? 1 : ab.charm_num;
    // Track which choice indices remain selectable.
    std::vector<bool> taken(ab.charm_choices.size(), false);

    for (int pick = 0; pick < to_pick; pick++) {
        game_log("Choose mode:\n");
        std::vector<LegalAction> mode_actions;
        std::vector<size_t> mode_indices;  // map action index -> charm_choices index
        for (size_t i = 0; i < ab.charm_choices.size(); i++) {
            if (taken[i]) continue;  // CR 601.2b: a mode can be chosen only once
            Ability &candidate = ab.charm_choices[i];
            candidate.source = ab.source;
            candidate.controller = ab.controller;
            // Skip modes that require targets but have none available.
            if (candidate.valid_tgts != "N_A" && candidate.target_min > 0 &&
                !has_legal_targets(candidate, orderer)) {
                continue;
            }
            std::string desc =
                (i < ab.charm_choice_descriptions.size() && !ab.charm_choice_descriptions[i].empty())
                    ? ab.charm_choice_descriptions[i]
                    : ("Mode " + std::to_string(i + 1));
            LegalAction la(PASS_PRIORITY, desc);
            la.category = ActionCategory::CHOOSE_MODE;
            mode_actions.push_back(la);
            mode_indices.push_back(i);
        }
        if (mode_actions.empty()) {
            // No further legal mode (all taken or none with legal targets). A modal
            // spell with too few legal modes simply resolves with what it could pick.
            if (pick == 0) game_log("No valid modes — charm fizzles\n");
            break;
        }
        int choice = InputLogger::instance().get_input(mode_actions);
        size_t chosen_idx = mode_indices[static_cast<size_t>(choice)];
        taken[chosen_idx] = true;
        resolve_chosen_mode(ab, ab.charm_choices[chosen_idx], orderer);
    }
    // Skip subabilities — charm handles its own resolution
    return false;
}

}  // namespace effects
