#include "damage.h"
#include "../ecs/coordinator.h"

//returns "was damage dealt this way"
bool deal_damage(Entity source, Entity target, size_t amount) {

    //TODO check for damage prevention, protection, other nonsense
    Coordinator& coordinator = Coordinator::global();
    
    //does nothing if target cannot be damaged and/or no longer is valid
    if(!coordinator.entity_has_component<Damage>(target)){
        return false;
    }

    coordinator.GetComponent<Damage>(target).damage_counters += amount;
}