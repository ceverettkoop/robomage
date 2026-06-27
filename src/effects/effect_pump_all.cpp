#include "effects.h"

#include <string>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// DB$ PumpAll (Lorehold Charm: "Creatures you control get +1/+1 and gain trample until end of
// turn"). The mass counterpart of single-target Pump: every battlefield permanent matching
// ValidCards$ gets the +P/+T (NumAtt$/NumDef$, static or count-SVar) and any granted KW$,
// until end of turn (CR 611.3 — the bonus is applied once and reverted at cleanup, 514.2). The
// ValidCards$ filter is matched through the shared permanent_matches_filter so YouCtrl/OppCtrl
// controller scoping and the full qualifier grammar (subtypes, colors, P/T, …) come for free,
// and the per-creature application reuses apply_pump_to_creature so the EOT bucket / keyword /
// logging logic is identical to single-target Pump.
bool pump_all(Ability &ab, std::shared_ptr<Orderer> orderer) {
    const PumpParams *pp = std::get_if<PumpParams>(&ab.params);

    std::vector<Entity> targets;
    for (auto e : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Creature>(e)) continue;
        if (permanent_matches_filter(e, ab.valid_cards_filter, MatchCtx{ab.controller, ab.source}))
            targets.push_back(e);
    }
    if (targets.empty()) {
        game_log("PumpAll: no matching creatures.\n");
        return true;
    }
    // NumAtt$/NumDef$ count-SVars are evaluated per-creature (the magnitude can depend on the
    // permanent in flexible filters); for a static +N/+N this is a constant across the loop.
    for (auto e : targets) {
        int pump_att = 0, pump_def = 0;
        resolve_pump_amounts(pp, ab.controller, orderer, e, pump_att, pump_def);
        apply_pump_to_creature(e, pump_att, pump_def, pp);
    }
    return true;
}

}  // namespace effects
