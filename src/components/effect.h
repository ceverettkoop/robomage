#ifndef EFFECT_H
#define EFFECT_H

#include "zone.h"
#include <set>
#include <string>
#include "types.h"

struct Effect {
    std::set<Zone::ZoneValue> affected_zones;
    std::set<Type> affected_types;
    std::string category;
    int amount;
};


#endif /* EFFECT_H */
