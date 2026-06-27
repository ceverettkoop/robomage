#include "effects.h"

#include <string>
#include <vector>

#include "../cli_output.h"
#include "../components/damage.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

// DamageAll (Whipflare, Pyroclasm, ...): deal NumDmg$ damage to every battlefield
// permanent matching the ValidCards$ filter (e.g. "Creature.nonArtifact"). The damage
// is dealt simultaneously to all matching creatures (CR 701.x / 119), so we mark damage
// on each first; the lethal-damage state-based action then destroys any creature with
// lethal damage marked (respecting Indestructible). Mirrors destroy_all/sacrifice_all in
// structure, but marks damage rather than moving the permanent itself.
bool damage_all(Ability &ab, std::shared_ptr<Orderer> orderer) {
    size_t dmg = ab.amount;
    std::vector<Entity> targets;
    for (auto e : orderer->mEntities)
        if (permanent_matches_filter(e, ab.valid_cards_filter, MatchCtx{ab.controller, ab.source}))
            targets.push_back(e);

    for (auto e : targets) {
        if (::deal_damage(ab.source, e, dmg)) {
            game_log("Dealt %zu damage to %s\n", dmg,
                     global_coordinator.GetComponent<Permanent>(e).name.c_str());
        }
    }
    return true;
}

}  // namespace effects
