#include "effects.h"

#include <string>

#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"

extern Coordinator global_coordinator;

namespace effects {

HandlerResult exalted_bonus(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    if (ab.target != 0 && global_coordinator.entity_has_component<Creature>(ab.target)) {
        auto &cr = global_coordinator.GetComponent<Creature>(ab.target);
        cr.prowess_bonus += static_cast<int>(ab.amount);
        recompute_pt(cr);
        std::string tgt_name = global_coordinator.entity_has_component<CardData>(ab.target)
                                   ? global_coordinator.GetComponent<CardData>(ab.target).name
                                   : (global_coordinator.entity_has_component<Permanent>(ab.target)
                                             ? global_coordinator.GetComponent<Permanent>(ab.target).name
                                             : "creature");
        std::string src_name = global_coordinator.entity_has_component<CardData>(ab.source)
                                   ? global_coordinator.GetComponent<CardData>(ab.source).name
                                   : (global_coordinator.entity_has_component<Permanent>(ab.source)
                                             ? global_coordinator.GetComponent<Permanent>(ab.source).name
                                             : "permanent");
        game_log("Exalted (%s): %s gets +%zu/+%zu until end of turn.\n", src_name.c_str(), tgt_name.c_str(),
            ab.amount, ab.amount);
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
