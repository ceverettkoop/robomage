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
    if (targets.empty()) targets.push_back(ab.target != 0 ? ab.target : ab.source);
    for (Entity t : targets) {
        if (!global_coordinator.entity_has_component<Permanent>(t)) continue;
        auto &tperm = global_coordinator.GetComponent<Permanent>(t);
        tperm.is_tapped = false;
        game_log("%s untaps\n", tperm.name.c_str());
    }
    return true;
}

}  // namespace effects
