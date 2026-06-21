#include "effects.h"

#include "../cli_output.h"
#include "../components/creature.h"
#include "../ecs/coordinator.h"

extern Coordinator global_coordinator;

namespace effects {

bool multiply_counter(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    // Double all P1P1 counters on target creature
    Entity tgt = (ab.target != 0) ? ab.target : ab.source;
    if (global_coordinator.entity_has_component<Creature>(tgt)) {
        auto &cr = global_coordinator.GetComponent<Creature>(tgt);
        if (cr.plus_one_counters > 0) {
            cr.plus_one_counters *= 2;
            recompute_pt(cr);
            game_log("MultiplyCounter: doubled +1/+1 counters on creature (now %u/%u).\n", cr.power, cr.toughness);
        }
    }
    return true;
}

}  // namespace effects
