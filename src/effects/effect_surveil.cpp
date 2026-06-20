#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

bool surveil(Ability &ab, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership controller;
    if (global_coordinator.entity_has_component<Permanent>(ab.source)) {
        controller = global_coordinator.GetComponent<Permanent>(ab.source).controller;
    } else {
        controller = global_coordinator.GetComponent<Zone>(ab.source).owner;
    }

    for (size_t i = 0; i < ab.amount; i++) {
        std::vector<Entity> lib = orderer->get_library_contents(controller);
        if (lib.empty()) {
            game_log("%s's library is empty — nothing to surveil.\n", player_name(controller).c_str());
            break;
        }

        // Find the top card (minimum distance_from_top)
        Entity top_card = lib[0];
        size_t min_dist = global_coordinator.GetComponent<Zone>(lib[0]).distance_from_top;
        for (auto e : lib) {
            size_t d = global_coordinator.GetComponent<Zone>(e).distance_from_top;
            if (d < min_dist) {
                min_dist = d;
                top_card = e;
            }
        }

        auto &top_cd = global_coordinator.GetComponent<CardData>(top_card);
        game_log_private(
            controller, "Top card of %s's library: %s\n", player_name(controller).c_str(), top_cd.name.c_str());
        std::vector<LegalAction> surveil_actions = {
            LegalAction(PASS_PRIORITY, top_card, std::string("Keep on top")),
            LegalAction(PASS_PRIORITY, top_card, std::string("Put in graveyard")),
        };
        int choice = InputLogger::instance().get_input(surveil_actions);

        if (choice == 1) {
            orderer->add_to_zone(false, top_card, Zone::GRAVEYARD);
            game_log("%s puts %s into the graveyard.\n", player_name(controller).c_str(), top_cd.name.c_str());
        }
    }
    return true;
}

}  // namespace effects
