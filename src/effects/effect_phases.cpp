#include "effects.h"

#include "../cli_output.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"

extern Coordinator global_coordinator;

namespace effects {

HandlerResult phases(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    // Phase out target permanent
    if (ab.target != 0 && global_coordinator.entity_has_component<Permanent>(ab.target)) {
        auto &tgt_perm = global_coordinator.GetComponent<Permanent>(ab.target);
        tgt_perm.is_phased_out = true;
        game_log("%s phases out\n", tgt_perm.name.c_str());
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
