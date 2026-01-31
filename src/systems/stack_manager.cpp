#include "stack_manager.h"

#include <cstddef>
#include "../components/ability.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"

void StackManager::init() {
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>());
    signature.set(global_coordinator.GetComponentType<Ability>());
    global_coordinator.SetSystemSignature<StackManager>(signature);
}

bool StackManager::is_empty() {
    for (auto &&entity : mEntities) {
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location == Zone::STACK) {
            return false;
        }
    }
    return true;
}

void StackManager::resolve_top() {
    Entity top_entity = 0;
    size_t min_distance = SIZE_MAX;
    bool found = false;

    // Find the entity on top of the stack (closest to top, distance_from_top == 0)
    for (auto &&entity : mEntities) {
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location == Zone::STACK && zone.distance_from_top < min_distance) {
            top_entity = entity;
            min_distance = zone.distance_from_top;
            found = true;
        }
    }

    if (found && global_coordinator.entity_has_component<Ability>(top_entity)) {
        auto &ability = global_coordinator.GetComponent<Ability>(top_entity);
        ability.resolve();
    }
}

std::vector<Entity> StackManager::get_stack_contents() {
    std::vector<Entity> stack_entities;

    for (auto &&entity : mEntities) {
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location == Zone::STACK) {
            stack_entities.push_back(entity);
        }
    }

    return stack_entities;
}
