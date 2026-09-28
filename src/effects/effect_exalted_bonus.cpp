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
    const Entity tgt = ab.target.get();
    const Entity src = ab.source.lki_entity();
    if (tgt != 0 && global_coordinator.entity_has_component<Creature>(tgt)) {
        auto &cr = global_coordinator.GetComponent<Creature>(tgt);
        cr.prowess_bonus += static_cast<int>(ab.amount);
        recompute_pt(cr);
        std::string tgt_name = global_coordinator.entity_has_component<CardData>(tgt)
                                   ? global_coordinator.GetComponent<CardData>(tgt).name
                                   : (global_coordinator.entity_has_component<Permanent>(tgt)
                                             ? global_coordinator.GetComponent<Permanent>(tgt).name
                                             : "creature");
        std::string src_name = global_coordinator.entity_has_component<CardData>(src)
                                   ? global_coordinator.GetComponent<CardData>(src).name
                                   : (global_coordinator.entity_has_component<Permanent>(src)
                                             ? global_coordinator.GetComponent<Permanent>(src).name
                                             : "permanent");
        game_log("Exalted (%s): %s gets +%zu/+%zu until end of turn.\n", src_name.c_str(), tgt_name.c_str(),
            ab.amount, ab.amount);
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
