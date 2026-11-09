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
#include "../queries/affected.h"
#include "../resolution_memory.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

HandlerResult mill(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    // Move top N cards from target player's library to graveyard. A targeted mill
    // ("Target player mills three cards" — Witherbloom Command) mills the chosen
    // player; otherwise the effect's controller mills (Defined$ You / self-mill).
    Zone::Ownership mill_owner = affected_player(ab);
    size_t mill_count = ab.def->amount_from_damage ? ab.trigger_damage_amount : ((ab.def->amount > 0) ? ab.def->amount : 1);
    std::vector<Entity> milled = orderer->mill(mill_owner, mill_count);
    report_effect(ab, EffectObjects::MILLED);
    for (auto e : milled) report_object(ab, EffectObjects::MILLED, e);
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
