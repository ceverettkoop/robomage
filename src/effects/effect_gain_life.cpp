#include "effects.h"

#include <cstdint>
#include <string>

#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../mana_system.h"

extern Coordinator global_coordinator;

namespace effects {

bool gain_life(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    Zone::Ownership gain_controller;
    if (ab.defined_targeted_controller && global_coordinator.entity_has_component<Zone>(ab.target)) {
        // Swords to Plowshares: gain life goes to the exiled creature's controller
        gain_controller = global_coordinator.GetComponent<Zone>(ab.target).controller;
        if (gain_controller == Zone::UNKNOWN && global_coordinator.entity_has_component<Permanent>(ab.target))
            gain_controller = global_coordinator.GetComponent<Permanent>(ab.target).controller;
    } else if (global_coordinator.entity_has_component<Permanent>(ab.source)) {
        gain_controller = global_coordinator.GetComponent<Permanent>(ab.source).controller;
    } else {
        gain_controller = global_coordinator.GetComponent<Zone>(ab.source).owner;
    }
    // Evaluate dynamic amount if set (e.g. "Targeted$CardPower"). effective_power gives the
    // creature's EFFECTIVE power (counters / continuous buffs included) read live while it is
    // still in play, or its last-known value once it has left — Swords to Plowshares exiles
    // the creature in its main effect before this sub-ability runs (CR 608.2h).
    size_t gain_amount = ab.amount;
    if (!ab.dynamic_amount_expr.empty() && ab.dynamic_amount_expr.find("Targeted$CardPower") != std::string::npos) {
        int p = effective_power(ab.target);
        gain_amount = static_cast<size_t>(p < 0 ? 0 : p);
    }
    Entity ctrl_entity = get_player_entity(gain_controller);
    player_gain_life(ctrl_entity, static_cast<int32_t>(gain_amount));
    auto &player = global_coordinator.GetComponent<Player>(ctrl_entity);
    game_log(
        "%s gains %zu life (now at %d)\n", player_name(gain_controller).c_str(), gain_amount, player.life_total);
    return true;
}

}  // namespace effects
