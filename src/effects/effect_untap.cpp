#include "effects.h"

#include <string>

#include "../cli_output.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"

extern Coordinator global_coordinator;

namespace effects {

bool untap(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    // Untap every chosen target. A single-target Untap uses ab.target; a multi-target Untap
    // (Candelabra of Tawnos: "Untap X target lands") populates ab.targets — untap each. An
    // UNtargeted Untap (Grim Monolith: "{4}: Untap this artifact.", no ValidTgts$) untaps its
    // own source.
    std::vector<Entity> targets = ab.targets;
    if (targets.empty()) {
        if (ab.target != 0) {
            targets.push_back(ab.target);  // single chosen target
        } else if (ab.valid_tgts == "N_A") {
            // Untargeted Untap (Grim Monolith) untaps its own source. A TARGETED untap that
            // resolved with NO chosen target (e.g. Candelabra's "Untap X target lands" with
            // X=0) untaps NOTHING — it must NOT fall back to its source. Doing so would untap
            // Candelabra itself, undoing the {T} it paid as an activation cost and making the
            // ability infinitely re-activatable in one priority window (a non-terminating loop).
            targets.push_back(ab.source);
        }
    }
    for (Entity t : targets) {
        if (!global_coordinator.entity_has_component<Permanent>(t)) continue;
        auto &tperm = global_coordinator.GetComponent<Permanent>(t);
        tperm.is_tapped = false;
        game_log("%s untaps\n", tperm.name.c_str());
    }
    return true;
}

}  // namespace effects
