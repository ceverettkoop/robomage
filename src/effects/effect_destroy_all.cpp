#include "effects.h"

#include <string>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/types.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

bool destroy_all(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Destroy all permanents matching the filter (e.g. Meltdown: "Artifact.cmcLEX")
    const DestroyAllParams *dp = std::get_if<DestroyAllParams>(&ab.params);
    std::string filter = dp ? dp->filter : "";
    bool filter_artifact = filter.find("Artifact") != std::string::npos;
    bool filter_creature = filter.find("Creature") != std::string::npos;
    bool filter_enchantment = filter.find("Enchantment") != std::string::npos;
    // Parse CMC filter: cmcLEX means CMC <= X paid
    int cmc_le = -1;
    if (filter.find("cmcLEX") != std::string::npos) {
        cmc_le = static_cast<int>(cur_game.x_paid);
    }
    std::vector<Entity> to_destroy;
    for (auto e : orderer->mEntities) {
        if (!is_battlefield_permanent(e)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        // Type filter
        bool type_match = (!filter_artifact && !filter_creature && !filter_enchantment);
        for (auto &t : perm.types) {
            if (filter_artifact && t.kind == TYPE && t.name == "Artifact") type_match = true;
            if (filter_creature && t.kind == TYPE && t.name == "Creature") type_match = true;
            if (filter_enchantment && t.kind == TYPE && t.name == "Enchantment") type_match = true;
        }
        if (!type_match) continue;
        // CMC filter
        if (cmc_le >= 0 && global_coordinator.entity_has_component<CardData>(e)) {
            int cmc = static_cast<int>(global_coordinator.GetComponent<CardData>(e).mana_cost.size());
            if (cmc > cmc_le) continue;
        }
        to_destroy.push_back(e);
    }
    for (auto e : to_destroy) {
        std::string ename = global_coordinator.entity_has_component<Permanent>(e)
            ? global_coordinator.GetComponent<Permanent>(e).name : "<unknown>";
        orderer->add_to_zone(false, e, Zone::GRAVEYARD);
        game_log("%s is destroyed\n", ename.c_str());
    }
    return true;
}

bool parse_destroy_all(Ability &ab, const std::string &key, const std::string &value) {
    if (key != "ValidCards") return false;
    effect_params<DestroyAllParams>(ab).filter = value;
    return true;
}

}  // namespace effects
