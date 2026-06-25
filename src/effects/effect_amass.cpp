#include "effects.h"

#include <algorithm>
#include <cctype>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../components/token.h"
#include "../components/types.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../parse.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Returns true if the permanent has a subtype/type matching `name`.
static bool permanent_has_type(const Permanent &perm, const std::string &name) {
    for (const auto &t : perm.types)
        if (t.name == name) return true;
    return false;
}

// "Amass <subtype> N" (rule 701.46): if you control an Army, put N +1/+1 counters
// on it and it becomes the amassed creature type in addition to its other types.
// If you control no Army, first create a 0/0 black Army creature token of the
// amassed type, then put the counters on it.
bool amass(Ability &ab, std::shared_ptr<Orderer> orderer) {
    int n = static_cast<int>(ab.amount);
    const AmassParams *ap = std::get_if<AmassParams>(&ab.params);
    std::string subtype = (ap && !ap->subtype.empty()) ? ap->subtype : "Orc";

    Zone::Ownership ctrl = global_coordinator.entity_has_component<Permanent>(ab.source)
                               ? global_coordinator.GetComponent<Permanent>(ab.source).controller
                               : global_coordinator.GetComponent<Zone>(ab.source).owner;

    // Find an Army the controller already controls.
    Entity army = 0;
    for (auto e : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        if (global_coordinator.GetComponent<Zone>(e).location != Zone::BATTLEFIELD) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.controller != ctrl) continue;
        if (permanent_has_type(perm, "Army")) { army = e; break; }
    }

    if (army == 0) {
        // No Army: create a 0/0 black <subtype> Army token (token script named
        // "b_0_0_<subtype>_army", matching the existing orc/zombie/sliver scripts).
        std::string lc = subtype;
        std::transform(lc.begin(), lc.end(), lc.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        Token tok = parse_token_script("b_0_0_" + lc + "_army");
        if (tok.name.empty()) {
            game_log("Amass: no token script for '%s' army.\n", subtype.c_str());
            return true;
        }
        army = global_coordinator.CreateEntity();
        global_coordinator.AddComponent(army, Zone(Zone::HAND, ctrl, ctrl));
        global_coordinator.AddComponent(army, tok);
        orderer->add_to_zone(false, army, Zone::BATTLEFIELD);
        bootstrap_token_components(army, tok, ctrl, cur_game.timestamp);
    } else {
        // Existing Army gains the amassed creature type in addition to its types.
        auto &perm = global_coordinator.GetComponent<Permanent>(army);
        if (!permanent_has_type(perm, subtype))
            perm.types.insert(Type{SUBTYPE, subtype});
    }

    if (n > 0 && global_coordinator.entity_has_component<Creature>(army)) {
        add_counters(army, "P1P1", n);
        auto &cr = global_coordinator.GetComponent<Creature>(army);
        game_log("Amass %s %d: %s Army is now %u/%u.\n", subtype.c_str(), n,
                 player_name(ctrl).c_str(), cr.power, cr.toughness);
    }
    return true;
}

bool parse_amass(Ability &ab, const std::string &key, const std::string &value) {
    if (ab.category != "Amass") return false;
    if (key == "Type") { effect_params<AmassParams>(ab).subtype = value; return true; }
    if (key == "Num")  { ab.amount = static_cast<size_t>(std::stoi(value)); return true; }
    return false;
}

}  // namespace effects
