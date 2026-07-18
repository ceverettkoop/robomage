#include "effects.h"

#include <string>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// SacrificeAll (Ajani's -4 after each opponent keeps one of each chosen type): every
// permanent matching the ValidCards$ filter is sacrificed by its controller. The filter
// (e.g. "Permanent.nonLand+OppCtrl+nonChosenCard") already excludes the kept permanents
// via nonChosenCard, which reads cur_game.chosen_cards.
HandlerResult sacrifice_all(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    std::vector<Entity> to_sac;
    for (auto e : orderer->mEntities)
        if (permanent_matches_filter(e, ab.valid_cards_filter, MatchCtx{ab.controller, ab.source}))
            to_sac.push_back(e);

    for (auto e : to_sac) {
        const std::string nm = global_coordinator.GetComponent<Permanent>(e).name;
        const Zone::Ownership owner = global_coordinator.GetComponent<Permanent>(e).controller;
        orderer->add_to_zone(false, e, Zone::GRAVEYARD);
        game_log("%s sacrifices %s.\n", player_name(owner).c_str(), nm.c_str());
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
