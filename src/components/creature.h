#ifndef CREATURE_H
#define CREATURE_H

#include <cstdint>
#include "../ecs/entity.h"

struct Creature {
    uint32_t power = 0;
    uint32_t toughness = 0;
    bool is_attacking = false;
    Entity attack_target = 0;  // Entity of player or planeswalker being attacked (0 = none)
};

#endif /* CREATURE_H */
