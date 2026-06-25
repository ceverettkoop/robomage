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
#include "../game_queries.h"
#include "../input_logger.h"

extern Coordinator global_coordinator;

namespace effects {

bool pump(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    // Pump used purely as a targeting vehicle for a graveyard card (Surgical Extraction's
    // SP$ Pump | TgtZone$ Graveyard): the target was already chosen at cast and the
    // subabilities do the work — don't re-pick a battlefield creature here.
    if (ab.target_in_graveyard) return true;

    // Present target selection, then chain subabilities with that target
    Zone::Ownership ctrl = ab.controller;
    std::vector<Entity> pump_targets;
    for (Entity e = 0; e < global_coordinator.GetMaxIssuedEntity(); ++e) {
        if (!is_battlefield_permanent(e)) continue;
        if (!global_coordinator.entity_has_component<Creature>(e)) continue;
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
    const PumpParams *pp = std::get_if<PumpParams>(&ab.params);
    int pump_att = pp ? pp->att : 0;
    int pump_def = pp ? pp->def : 0;
    if ((pump_att != 0 || pump_def != 0) && ab.target != 0 &&
        global_coordinator.entity_has_component<Creature>(ab.target)) {
        auto &cr = global_coordinator.GetComponent<Creature>(ab.target);
        // "Until end of turn" pump: store in the EOT bonus bucket so cleanup (514.2/611.2b)
        // reverts it. Signed + floored at 0 by recompute_pt so a -X/-X pump (e.g. Dismember)
        // can never underflow the uint32_t effective field.
        cr.eot_power_bonus += pump_att;
        cr.eot_toughness_bonus += pump_def;
        recompute_pt(cr);
        std::string tname = global_coordinator.entity_has_component<Permanent>(ab.target)
            ? global_coordinator.GetComponent<Permanent>(ab.target).name : "<unknown>";
        game_log("%s gets %+d/%+d (now %u/%u)\n", tname.c_str(), pump_att, pump_def, cr.power, cr.toughness);
    }
    return true;
}

bool parse_pump(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "NumAtt") { effect_params<PumpParams>(ab).att = std::stoi(value); return true; }
    if (key == "NumDef") { effect_params<PumpParams>(ab).def = std::stoi(value); return true; }
    return false;
}

}  // namespace effects
