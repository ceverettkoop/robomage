#ifndef CREATURE_H
#define CREATURE_H

#include <cstdint>
#include <set>
#include <string>
#include <vector>
#include "../ecs/entity.h"
#include "../classes/colors.h"

struct Creature {
    uint32_t power = 0;
    uint32_t toughness = 0;
    bool is_attacking = false;
    Entity attack_target = 0;  // Entity of player or planeswalker being attacked (0 = none)
    bool is_blocking = false;
    Entity blocking_target = 0;  // Entity of attacker being blocked (0 = none)
    std::vector<std::string> keywords;
    bool must_attack = false;        // set by MustAttack static ability; enforced in declare_attackers
    int plus_one_counters = 0;       // +1/+1 counters; power/toughness adjusted when added/removed
    int prowess_bonus = 0;           // temporary +1/+1 from prowess; cleared at cleanup step
};

std::set<Colors> get_protection_colors(const Creature &cr);
bool has_protection_from(const Creature &cr, Entity source);

#endif /* CREATURE_H */
