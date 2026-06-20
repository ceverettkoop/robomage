#include "effects.h"

#include <string>

#include "../cli_output.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"

extern Coordinator global_coordinator;

namespace effects {

bool untap(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    if (global_coordinator.entity_has_component<Permanent>(ab.target)) {
        auto &tperm = global_coordinator.GetComponent<Permanent>(ab.target);
        tperm.is_tapped = false;
        std::string tname = tperm.name;
        game_log("%s untaps\n", tname.c_str());
    }
    return true;
}

}  // namespace effects
