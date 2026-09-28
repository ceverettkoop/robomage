#include "effects.h"

#include <string>

#include "../cli_output.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../queries/battlefield.h"

extern Coordinator global_coordinator;

namespace effects {

HandlerResult untap(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    // Untap every chosen target. A single-target Untap uses ab.target; a multi-target Untap
    // (Candelabra of Tawnos: "Untap X target lands") populates ab.targets — untap each. An
    // UNtargeted Untap (Grim Monolith: "{4}: Untap this artifact.", no ValidTgts$) untaps its
    // own source.
    std::vector<Entity> targets = live_entities(ab.targets);
    if (ab.targets.empty()) {
        if (!ab.target.empty()) {
            targets.push_back(ab.target.get());  // single chosen target (0 = its object is gone)
        } else if (ab.valid_tgts == "N_A") {
            // Untargeted Untap (Grim Monolith) untaps its own source. A TARGETED untap that
            // resolved with NO chosen target (e.g. Candelabra's "Untap X target lands" with
            // X=0) untaps NOTHING — it must NOT fall back to its source. Doing so would untap
            // Candelabra itself, undoing the {T} it paid as an activation cost and making the
            // ability infinitely re-activatable in one priority window (a non-terminating loop).
            targets.push_back(ab.source.get());
        }
    }
    for (Entity t : targets) {
        if (!is_battlefield_permanent(t)) continue;
        auto &tperm = global_coordinator.GetComponent<Permanent>(t);
        tperm.is_tapped = false;
        game_log("%s untaps\n", tperm.name.c_str());
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
