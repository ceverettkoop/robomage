#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

// Surveil N (CR 701.25a): look at the top N cards of your library, then put any number of them
// into your graveyard and the rest on top of your library in any order. Modeled as one
// interactive loop over the N looked-at cards: at each step the player picks any remaining card
// and a destination — into the graveyard or onto the top of the library. The order in which
// cards are chosen "on top" fixes the final library order: the FIRST card sent to the top ends
// up as the topmost card (they are re-placed in reverse so the choice order reads top-down).
// The two per-card options share the same card entity, so they must differ by category to be
// distinguishable to the semantic action resolver (like scry's keep/bottom): on-top =
// TOP_LIBRARY, into-graveyard = CHOOSE_CARD (a library -> graveyard non-library zone change).
bool surveil(Ability &ab, std::shared_ptr<Orderer> orderer) {
    PendingDecisionScope pending_scope(ab.source);
    Zone::Ownership controller;
    if (global_coordinator.entity_has_component<Permanent>(ab.source)) {
        controller = global_coordinator.GetComponent<Permanent>(ab.source).controller;
    } else {
        controller = global_coordinator.GetComponent<Zone>(ab.source).owner;
    }

    size_t num = ab.amount;
    if (!ab.dynamic_amount_expr.empty())
        num = evaluate_dynamic_amount(ab.dynamic_amount_expr, controller, orderer, ab.target);
    if (num == 0) return true;  // CR 701.25c: surveil 0 is no event

    std::vector<Entity> looked = orderer->get_library_top(controller, num);
    if (looked.empty()) {
        game_log("%s's library is empty — nothing to surveil.\n", player_name(controller).c_str());
        return true;
    }

    game_log("%s surveils %zu.\n", player_name(controller).c_str(), looked.size());
    for (Entity card : looked) {
        auto &cd = global_coordinator.GetComponent<CardData>(card);
        game_log_private(controller, "Surveil: looking at %s\n", cd.name.c_str());
    }

    // remaining: the looked-at cards not yet assigned a destination (stable top-first order).
    // to_top: cards chosen to stay on top, in choice order (to_top[0] ends up topmost).
    std::vector<Entity> remaining = looked;
    std::vector<Entity> to_top;

    while (!remaining.empty()) {
        std::vector<LegalAction> actions;
        // First block: "put on top" for each remaining card; second block: "into graveyard".
        for (Entity card : remaining) {
            auto &cd = global_coordinator.GetComponent<CardData>(card);
            LegalAction la(PASS_PRIORITY, card, "Put " + cd.name + " on top of library");
            la.category = ActionCategory::TOP_LIBRARY;
            actions.push_back(la);
        }
        for (Entity card : remaining) {
            auto &cd = global_coordinator.GetComponent<CardData>(card);
            LegalAction la(PASS_PRIORITY, card, "Put " + cd.name + " into graveyard");
            la.category = ActionCategory::CHOOSE_CARD;
            actions.push_back(la);
        }

        int choice = InputLogger::instance().get_input(actions);
        size_t idx = static_cast<size_t>(choice);
        bool on_top = idx < remaining.size();
        size_t card_idx = on_top ? idx : idx - remaining.size();
        Entity card = remaining[card_idx];
        remaining.erase(remaining.begin() + static_cast<long>(card_idx));

        if (on_top) {
            to_top.push_back(card);
        } else {
            auto &cd = global_coordinator.GetComponent<CardData>(card);
            orderer->add_to_zone(false, card, Zone::GRAVEYARD);
            game_log("%s puts %s into their graveyard.\n", player_name(controller).c_str(), cd.name.c_str());
        }
    }

    // Re-place the kept cards on top. add_to_zone(false, ...) makes each the new top, so placing
    // in reverse leaves to_top[0] (the first choice) on top with the rest in the chosen order.
    for (size_t i = to_top.size(); i-- > 0;) {
        orderer->add_to_zone(false, to_top[i], Zone::LIBRARY);
    }
    return true;
}

}  // namespace effects
