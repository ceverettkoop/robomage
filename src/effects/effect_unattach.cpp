#include "effects.h"

#include "../cli_output.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../queries/battlefield.h"

extern Coordinator global_coordinator;

namespace effects {

// Reconfigure (CR 702.151a): "Unattach this permanent." The ability's source stops being attached
// to the creature it equips; the continuous-effects pass then makes it a creature again (CR
// 702.151b). A source that left the battlefield or is no longer attached is unaffected.
HandlerResult unattach(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    (void)ctx;
    const Entity self = ab.source.get();
    if (self == 0 || !is_battlefield_permanent(self)) return HandlerResult::DONE_RUN_SUBS;
    auto &perm = global_coordinator.GetComponent<Permanent>(self);
    if (perm.equipped_to.get() == 0) return HandlerResult::DONE_RUN_SUBS;
    perm.equipped_to = ObjectRef{};
    game_log("%s unattaches.\n", perm.name.c_str());
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
