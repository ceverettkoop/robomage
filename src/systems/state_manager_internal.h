#ifndef STATE_MANAGER_INTERNAL_H
#define STATE_MANAGER_INTERNAL_H

#include <string>

#include "../ecs/entity.h"

struct Creature;
struct Game;

// Helpers shared across the StateManager translation units (state_manager.cpp
// and its state_manager_*.cpp siblings). Not part of the public System API.

// entity_name — the shared display-name resolver — lives in game_queries.h.

// Display name for a spell/ability target or attack target: "Player A"/"Player B"
// when the entity is a player, otherwise entity_name(). Single source for the
// "name a target" idiom used in narrative logs and the observer. Defined in
// state_manager_statics.cpp.
std::string target_display_name(const Game &game, Entity tgt);

// Should this creature deal damage during the current combat damage step?
// (first-strike step → only First/Double Strike; regular step → everyone except
// pure first-strikers.) Defined in state_manager_combat.cpp; reused by the T3.10
// damage-assignment handler in action_processor.cpp.
bool should_deal_damage(const Creature &cr, bool first_strike_only);

#endif /* STATE_MANAGER_INTERNAL_H */
