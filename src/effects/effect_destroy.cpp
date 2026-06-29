#include "effects.h"

#include <string>

#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

static void destroy_single(Entity tgt, std::shared_ptr<Orderer> orderer) {
    if (!global_coordinator.entity_has_component<Zone>(tgt)) {
        game_log("Destroy: target is no longer in play\n");
        return;
    }
    auto &tz = global_coordinator.GetComponent<Zone>(tgt);
    if (tz.location != Zone::BATTLEFIELD) {
        game_log("Destroy: target is no longer on the battlefield\n");
        return;
    }
    std::string name = global_coordinator.entity_has_component<Permanent>(tgt)
                           ? global_coordinator.GetComponent<Permanent>(tgt).name
                           : "<unknown>";
    // CR 702.12b: a permanent with indestructible can't be destroyed. The effect still
    // resolves; the permanent stays on the battlefield.
    if (is_indestructible(tgt)) {
        game_log("%s is indestructible — not destroyed\n", name.c_str());
        return;
    }
    orderer->add_to_zone(false, tgt, Zone::GRAVEYARD);
    game_log("%s is destroyed\n", name.c_str());
}

bool destroy(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Pyroblast/Hydroblast destroy mode: only destroy if the target is the required
    // color. The spell still resolves (doing nothing) against a wrong-color permanent.
    if (!target_color_condition_met(ab, ab.target)) {
        std::string tname = global_coordinator.entity_has_component<Permanent>(ab.target)
                                ? global_coordinator.GetComponent<Permanent>(ab.target).name
                                : "<unknown>";
        game_log("%s is not the required color — not destroyed\n", tname.c_str());
        return true;
    }

    // Conditional destroy (Fatal Push): check target CMC against threshold
    if (!ab.condition_present.empty() && ab.condition_present.find("cmcLEX") != std::string::npos) {
        // Evaluate X from dynamic_amount_expr (resolved at parse time to e.g. "Count$Revolt.4.2")
        int threshold = 2;  // default fallback
        if (!ab.dynamic_amount_expr.empty()) {
            threshold = static_cast<int>(evaluate_dynamic_amount(ab.dynamic_amount_expr, ab.controller, orderer, ab.target));
        }
        Entity tgt = ab.target;
        if (global_coordinator.entity_has_component<CardData>(tgt)) {
            int tgt_cmc = card_mana_value(global_coordinator.GetComponent<CardData>(tgt));
            if (tgt_cmc > threshold) {
                std::string tname = global_coordinator.entity_has_component<Permanent>(tgt)
                    ? global_coordinator.GetComponent<Permanent>(tgt).name : "<unknown>";
                game_log("%s has mana value %d (threshold %d) — not destroyed\n", tname.c_str(), tgt_cmc, threshold);
                return true;
            }
        }
    }

    if (!ab.targets.empty()) {
        for (auto tgt : ab.targets) destroy_single(tgt, orderer);
    } else {
        destroy_single(ab.target, orderer);
    }
    return true;
}

}  // namespace effects
