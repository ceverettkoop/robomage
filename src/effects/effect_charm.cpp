#include "effects.h"

#include <string>
#include <vector>

#include "../action_processor.h"
#include "../classes/action.h"
#include "../classes/game.h"
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

// Modal spell (CR 700.2): "Choose one/two —". The mode(s) and their targets were announced
// when the spell was CAST (CR 601.2b/c) — see announce_spell_targets — and recorded in
// charm_chosen. Resolution only replays those picks in order: each mode's own resolve()
// re-verifies its targets (CR 608.2b), so a mode whose targets became illegal fizzles
// individually without any prompting here. The choose-at-resolution loop below remains as a
// FALLBACK for a charm that reached the stack without an announcement (a cast path not
// routed through announce_spell_targets).
bool charm(Ability &ab, std::shared_ptr<Orderer> orderer) {
    PendingDecisionScope pending_scope(ab.source);
    if (!ab.charm_chosen.empty()) {
        for (int idx : ab.charm_chosen) {
            if (idx < 0 || static_cast<size_t>(idx) >= ab.charm_choices.size()) continue;
            Ability &chosen = ab.charm_choices[static_cast<size_t>(idx)];
            chosen.source = ab.source;
            chosen.controller = ab.controller;
            chosen.resolve(orderer);
        }
        // Skip subabilities — charm handles its own resolution
        return false;
    }

    game_log("(modes were not announced at cast — choosing at resolution)\n");
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
