#include "effects.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../queries/players.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

HandlerResult mill(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    // Move top N cards from target player's library to graveyard. A targeted mill
    // ("Target player mills three cards" — Witherbloom Command) mills the chosen
    // player; otherwise the effect's controller mills (Defined$ You / self-mill).
    Zone::Ownership mill_owner = ab.controller;
    if (ab.target.get() != 0 && global_coordinator.entity_has_component<Player>(ab.target.get()))
        mill_owner = seat_of_player(ab.target.get());
    size_t mill_count = ab.def->amount_from_damage ? ab.trigger_damage_amount : ((ab.def->amount > 0) ? ab.def->amount : 1);
    std::vector<Entity> milled = orderer->mill(mill_owner, mill_count);
    if (ab.def->remember_milled) {
        cur_game.remembered_entities.clear();
        for (auto e : milled) cur_game.remembered_entities.push_back(ObjectRef::of(e));
    }
    return HandlerResult::DONE_RUN_SUBS;
}

bool parse_mill(AbilityDef &ab, const std::string &key, const std::string &value) {
    if (key != "RememberMilled") return false;
    ab.remember_milled = (value == "True");
    return true;
}

}  // namespace effects
