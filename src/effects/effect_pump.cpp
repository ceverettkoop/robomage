#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../cli_output.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"
#include "../input_logger.h"

extern Coordinator global_coordinator;

namespace effects {

bool pump(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    // Present target selection, then chain subabilities with that target
    Zone::Ownership ctrl = ab.controller;
    std::vector<Entity> pump_targets;
    for (Entity e = 0; e < MAX_ENTITIES; ++e) {
        if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
        if (!global_coordinator.entity_has_component<Creature>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location != Zone::BATTLEFIELD) continue;
        auto &p = global_coordinator.GetComponent<Permanent>(e);
        if (ab.valid_tgts.find("YouCtrl") != std::string::npos && p.controller != ctrl) continue;
        pump_targets.push_back(e);
    }
    if (pump_targets.empty()) {
        game_log("Pump: no valid targets.\n");
        // still chain subabilities with no target
    } else {
        game_log("Choose a creature for Pump:\n");
        std::vector<LegalAction> tgt_actions;
        for (auto te : pump_targets) {
            std::string ename = global_coordinator.GetComponent<Permanent>(te).name;
            auto &tcr = global_coordinator.GetComponent<Creature>(te);
            LegalAction la(PASS_PRIORITY, te,
                ename + " [" + std::to_string(tcr.power) + "/" + std::to_string(tcr.toughness) + "]");
            la.category = ActionCategory::SELECT_TARGET;
            tgt_actions.push_back(la);
        }
        int choice = InputLogger::instance().get_input(tgt_actions);
        if (choice >= 0 && choice < static_cast<int>(pump_targets.size()))
            ab.target = pump_targets[static_cast<size_t>(choice)];
    }
    // Apply P/T modification if NumAtt$/NumDef$ were set
    if ((ab.pump_att != 0 || ab.pump_def != 0) && ab.target != 0 &&
        global_coordinator.entity_has_component<Creature>(ab.target)) {
        auto &cr = global_coordinator.GetComponent<Creature>(ab.target);
        // Pump modifies the base characteristic (this engine does not auto-revert
        // pumps at end of turn). Signed + floored at 0 by recompute_pt so a -X/-X
        // pump can never underflow the uint32_t effective field.
        cr.base_power += ab.pump_att;
        cr.base_toughness += ab.pump_def;
        recompute_pt(cr);
        std::string tname = global_coordinator.entity_has_component<Permanent>(ab.target)
            ? global_coordinator.GetComponent<Permanent>(ab.target).name : "<unknown>";
        game_log("%s gets %+d/%+d (now %u/%u)\n", tname.c_str(), ab.pump_att, ab.pump_def, cr.power, cr.toughness);
    }
    return true;
}

}  // namespace effects
