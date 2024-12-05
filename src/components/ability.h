#ifndef ABILITY_H
#define ABILITY_H

#include "../ecs/entity.h"
#include <string>

struct Ability{

    enum AbilityType{
        TRIGGERED,
        ACTIVATED,
        SPELL
    };

    AbilityType ability_type;
    std::string category;
    Entity source;
    Entity target;
    size_t amount;

    void resolve();
};

#endif /* ABILITY_H */
