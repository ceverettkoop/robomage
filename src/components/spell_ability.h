#ifndef SPELL_ABILITY_H
#define SPELL_ABILITY_H

#include "../ecs/entity.h"
#include <string>

//effect of spell itself when it resolves, distinct from other kinds of abilities which case effects
struct SpellAbility{
    std::string type;
    Entity source;
    size_t amount;


    void deal_damage(Entity target, size_t amount);
};

#endif /* SPELL_ABILITY_H */
