#include "effects.h"

#include "../cli_output.h"
#include "../components/creature.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"

extern Coordinator global_coordinator;

namespace effects {

HandlerResult multiply_counter(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    // Double all P1P1 counters on target creature
    Entity tgt = (ab.target != 0) ? ab.target : ab.source;
    if (global_coordinator.entity_has_component<Creature>(tgt)) {
        auto &cr = global_coordinator.GetComponent<Creature>(tgt);
        int p1p1 = get_counters(tgt, "P1P1");
        if (p1p1 > 0) {
            add_counters(tgt, "P1P1", p1p1);  // doubling = add another copy of the current count
            game_log("MultiplyCounter: doubled +1/+1 counters on creature (now %u/%u).\n", cr.power, cr.toughness);
        }
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
