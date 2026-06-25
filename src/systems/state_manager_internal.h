#ifndef STATE_MANAGER_INTERNAL_H
#define STATE_MANAGER_INTERNAL_H

#include <string>

#include "../ecs/entity.h"

struct Creature;

// Helpers shared across the StateManager translation units (state_manager.cpp
// and its state_manager_*.cpp siblings). Not part of the public System API.

// Human-readable name for an entity — permanent, card, or token — used in
// game_log output. Defined in state_manager_statics.cpp.
std::string entity_name(Entity e);

// Should this creature deal damage during the current combat damage step?
// (first-strike step → only First/Double Strike; regular step → everyone except
// pure first-strikers.) Defined in state_manager_combat.cpp; reused by the T3.10
// damage-assignment handler in action_processor.cpp.
bool should_deal_damage(const Creature &cr, bool first_strike_only);

#endif /* STATE_MANAGER_INTERNAL_H */
