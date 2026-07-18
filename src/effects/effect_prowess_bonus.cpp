#include "effects.h"

#include "../cli_output.h"
#include "../components/creature.h"
#include "../ecs/coordinator.h"

extern Coordinator global_coordinator;

namespace effects {

HandlerResult prowess_bonus(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    if (global_coordinator.entity_has_component<Creature>(ab.source)) {
        auto &cr = global_coordinator.GetComponent<Creature>(ab.source);
        cr.prowess_bonus += static_cast<int>(ab.amount);
        recompute_pt(cr);
        game_log("Prowess: creature gets +%zu/+%zu until end of turn.\n", ab.amount, ab.amount);
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
