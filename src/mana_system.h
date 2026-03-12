#ifndef MANA_SYSTEM_H
#define MANA_SYSTEM_H

#include <cstddef>
#include <set>

#include "classes/colors.h"
#include "components/zone.h"
#include "ecs/entity.h"

// Get player entity from ownership
Entity get_player_entity(Zone::Ownership player);

// Check if a given mana pool can afford a cost (does not read player state)
bool can_afford_pool(const std::multiset<Colors>& pool, const std::multiset<Colors>& cost);

// Check if player can afford a mana cost
bool can_afford(Zone::Ownership player, const std::multiset<Colors>& cost);

// Spend mana from player's pool (assumes can_afford was checked)
void spend_mana(Zone::Ownership player, const std::multiset<Colors>& cost);

// Add mana to player's pool
void add_mana(Zone::Ownership player, Colors mana_color, size_t amount);

// Pay as much of cost as possible from player's pool, return the unpaid remainder
ManaValue pay_partial(Zone::Ownership player, const ManaValue& cost);

// Empty player's mana pool (called at step transitions)
void empty_mana_pool(Zone::Ownership player);

#endif
