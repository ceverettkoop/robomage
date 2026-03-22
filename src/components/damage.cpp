#include "damage.h"
#include "creature.h"
#include "../ecs/coordinator.h"

//returns "was damage dealt this way"
bool deal_damage(Entity source, Entity target, size_t amount) {

    Coordinator& coordinator = Coordinator::global();

    //does nothing if target cannot be damaged and/or no longer is valid
    if(!coordinator.entity_has_component<Damage>(target)){
        return false;
    }

    if (coordinator.entity_has_component<Creature>(target)) {
        if (has_protection_from(coordinator.GetComponent<Creature>(target), source))
            return false;
    }

    coordinator.GetComponent<Damage>(target).damage_counters += amount;
    return true;
}