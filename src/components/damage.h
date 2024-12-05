#ifndef DAMAGE_H
#define DAMAGE_H

#include <cstdint>
#include <cstddef> 
#include "../ecs/entity.h"

struct Damage{
    uint32_t damage_counters;
};

bool deal_damage(Entity source, Entity target, size_t amount);

#endif /* DAMAGE_H */
