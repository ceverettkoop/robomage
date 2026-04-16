#ifndef DAMAGE_H
#define DAMAGE_H

#include <cstdint>
#include <cstddef> 
#include "../ecs/entity.h"

struct Damage{
    size_t damage_counters;
    // 702.2b: any nonzero damage from a deathtouch source is lethal. Track whether
    // any damage marked on this creature came from such a source this turn.
    bool has_deathtouch_damage = false;
};

bool deal_damage(Entity source, Entity target, size_t amount);

#endif /* DAMAGE_H */