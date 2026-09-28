#include "attachments.h"

#include "../components/carddata.h"
#include "../components/creature.h"
#include "../ecs/coordinator.h"

bool equipment_can_equip(Entity equipment, Entity host) {
    if (host == 0 || host == equipment) return false;
    if (!is_battlefield_permanent(host) || !global_coordinator.entity_has_component<Creature>(host))
        return false;
    if (global_coordinator.entity_has_component<Creature>(equipment) &&
        !(global_coordinator.entity_has_component<CardData>(equipment) &&
          global_coordinator.GetComponent<CardData>(equipment).is_reconfigure))
        return false;
    return true;
}

std::vector<Entity> equip_candidates(Entity equipment, Zone::Ownership player,
                                     const std::set<Entity> &entities) {
    std::vector<Entity> out;
    for (auto e : entities)
        if (is_equip_candidate(equipment, e, player)) out.push_back(e);
    return out;
}
