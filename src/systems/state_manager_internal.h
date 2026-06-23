#ifndef STATE_MANAGER_INTERNAL_H
#define STATE_MANAGER_INTERNAL_H

#include <string>

#include "../ecs/entity.h"

// Helpers shared across the StateManager translation units (state_manager.cpp
// and its state_manager_*.cpp siblings). Not part of the public System API.

// Human-readable name for an entity — permanent, card, or token — used in
// game_log output. Defined in state_manager_statics.cpp.
std::string entity_name(Entity e);

#endif /* STATE_MANAGER_INTERNAL_H */
