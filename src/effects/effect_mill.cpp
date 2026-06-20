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
    std::vector<Entity> milled = orderer->mill(mill_owner, mill_count);
    if (ab.remember_milled) {
        cur_game.remembered_entities.clear();
        for (auto e : milled) cur_game.remembered_entities.push_back(e);
    }
    return true;
}

bool parse_mill(Ability &ab, const std::string &key, const std::string &value) {
    if (key != "RememberMilled") return false;
    ab.remember_milled = (value == "True");
    return true;
}

}  // namespace effects
