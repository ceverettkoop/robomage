#include "effects.h"

#include "../cli_output.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"

extern Coordinator global_coordinator;

namespace effects {

// DB$ Tap (generic): tap Defined$ Self (the source) or the chosen target. Ba Sing Se's
// "enters tapped unless you control a basic land" is realized as a conditional ENTERS_TAPPED
// replacement effect parsed from its R: line, so the ETB$ True path never reaches resolution;
// this handler covers a Tap that does resolve (e.g. a future "tap target permanent" ability).
HandlerResult tap(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    Entity e = ab.defined_self ? ab.source : ab.target;
    if (e == 0 || !is_battlefield_permanent(e)) return HandlerResult::DONE_RUN_SUBS;
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    if (!perm.is_tapped) {
        perm.is_tapped = true;
        game_log("%s is tapped.\n", perm.name.c_str());
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
