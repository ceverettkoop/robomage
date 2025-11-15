#include "state_manager.h"
#include "../ecs/coordinator.h"
#include "../error.h"

// orderer cares about anything that has a zone, also anything that has an effect

void StateManager::init() {
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>(), global_coordinator.GetComponentType<Effect>());
    global_coordinator.SetSystemSignature<StateManager>(signature);
}