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
#include "../mana_system.h"
#include "../queries/characteristics.h"
#include "../queries/player_resources.h"
#include "../queries/players.h"

extern Coordinator global_coordinator;

namespace effects {

HandlerResult gain_life(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    // Swords to Plowshares: gain life goes to the exiled creature's controller, read via
    // last-known info since the creature was exiled earlier this resolution (CR 608.2g/h).
    // Otherwise (and if that can't be resolved) the ability's controller gains the life.
    Zone::Ownership gain_controller = Zone::UNKNOWN;
    if (ab.def.defined_targeted_controller && !ab.target.empty())
        gain_controller = last_known_controller(ab.target.lki_entity());
    if (gain_controller == Zone::UNKNOWN)
        gain_controller = ab.controller;  // "you gain" = the ability's controller (CR 109.5)
    // Evaluate dynamic amount if set (e.g. "Targeted$CardPower"). effective_power gives the
    // creature's EFFECTIVE power (counters / continuous buffs included) read live while it is
    // still in play, or its last-known value once it has left — Swords to Plowshares exiles
    // the creature in its main effect before this sub-ability runs (CR 608.2h).
    size_t gain_amount = ab.def.amount;
    if (!ab.def.dynamic_amount_expr.empty() && ab.def.dynamic_amount_expr.find("Targeted$CardPower") != std::string::npos) {
        int p = effective_power(ab.target.lki_entity());
        gain_amount = static_cast<size_t>(p < 0 ? 0 : p);
    }
    Entity ctrl_entity = get_player_entity(gain_controller);
    player_gain_life(ctrl_entity, static_cast<int32_t>(gain_amount));
    auto &player = global_coordinator.GetComponent<Player>(ctrl_entity);
    game_log(
        "%s gains %zu life (now at %d)\n", player_name(gain_controller).c_str(), gain_amount, player.life_total);
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
