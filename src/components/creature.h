#ifndef CREATURE_H
#define CREATURE_H

#include <cstdint>
#include <string>
#include <vector>
#include "../ecs/entity.h"

struct Creature {
    uint32_t power = 0;
    uint32_t toughness = 0;
    bool is_attacking = false;
    Entity attack_target = 0;  // Entity of player or planeswalker being attacked (0 = none)
    bool is_blocking = false;
    Entity blocking_target = 0;  // Entity of attacker being blocked (0 = none)
    std::vector<std::string> keywords;
};

#endif /* CREATURE_H */
