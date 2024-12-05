#ifndef SPELL_ABILITY_H
#define SPELL_ABILITY_H

#include "../ecs/entity.h"
#include <string>

//effect of spell itself when it resolves, distinct from other kinds of abilities which cause effects
struct SpellAbility{
    std::string type;
    Entity source;
    Entity target;
    size_t amount;

    void resolve();
};

#endif /* SPELL_ABILITY_H */
