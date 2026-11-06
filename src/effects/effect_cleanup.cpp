#include "effects.h"

#include "../resolution_memory.h"

namespace effects {

HandlerResult cleanup(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    report_effect(ab, EffectObjects::CLEANED);
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
