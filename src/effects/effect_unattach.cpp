#include "effects.h"

#include "../cli_output.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"

extern Coordinator global_coordinator;

namespace effects {

// Reconfigure (CR 702.151a): "Unattach this permanent." The ability's source stops being attached
// to the creature it equips; the continuous-effects pass then makes it a creature again (CR
// 702.151b). A source that left the battlefield or is no longer attached is unaffected.
HandlerResult unattach(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    (void)ctx;
    if (!is_battlefield_permanent(ab.source)) return HandlerResult::DONE_RUN_SUBS;
    auto &perm = global_coordinator.GetComponent<Permanent>(ab.source);
    if (perm.equipped_to == 0) return HandlerResult::DONE_RUN_SUBS;
    perm.equipped_to = 0;
    game_log("%s unattaches.\n", perm.name.c_str());
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
