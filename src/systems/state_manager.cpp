#include "state_manager.h"

#include "../components/effect.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../error.h"

void StateManager::init() {
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>());
    signature.set(global_coordinator.GetComponentType<Effect>());
    global_coordinator.SetSystemSignature<StateManager>(signature);
}

//layers / timestamps would be implemented here; for now order is arbitrary
void StateManager::state_based_effects() {
    for (auto &&entity : mEntities) {
        // if we are dealing with an effect
        if (global_coordinator.entity_has_component<Effect>(entity)) {

        }
    }
}
