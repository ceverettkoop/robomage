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
    // Swords to Plowshares: gain life goes to the exiled creature's controller, read via
    // last-known info since the creature was exiled earlier this resolution (CR 608.2g/h).
    // Otherwise (and if that can't be resolved) the source's controller gains the life.
    Zone::Ownership gain_controller = Zone::UNKNOWN;
    if (ab.defined_targeted_controller && ab.target != 0)
        gain_controller = last_known_controller(ab.target);
    if (gain_controller == Zone::UNKNOWN)
        gain_controller = source_controller(ab.source);
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
