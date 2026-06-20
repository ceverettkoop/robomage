#include "effects.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

bool mill(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Move top N cards from target player's library to graveyard
    Zone::Ownership mill_owner = ab.controller;
    size_t mill_count = ab.amount_from_damage ? ab.trigger_damage_amount : ((ab.amount > 0) ? ab.amount : 1);
    std::vector<Entity> lib = orderer->get_library_contents(mill_owner);
    std::sort(lib.begin(), lib.end(), [](Entity a, Entity b) {
        return global_coordinator.GetComponent<Zone>(a).distance_from_top <
               global_coordinator.GetComponent<Zone>(b).distance_from_top;
    });
    if (ab.remember_milled) cur_game.remembered_entities.clear();
    for (size_t i = 0; i < mill_count && i < lib.size(); i++) {
        std::string cname = global_coordinator.entity_has_component<CardData>(lib[i])
                                ? global_coordinator.GetComponent<CardData>(lib[i]).name
                                : "card";
        orderer->add_to_zone(false, lib[i], Zone::GRAVEYARD);
        game_log("%s mills %s.\n", player_name(mill_owner).c_str(), cname.c_str());
        if (ab.remember_milled) cur_game.remembered_entities.push_back(lib[i]);
    }
    return true;
}

}  // namespace effects
