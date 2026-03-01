#ifndef PERMANENT_H
#define PERMANENT_H

#include "../ecs/entity.h"
#include "zone.h"
#include "ability.h"
#include "static_ability.h"
#include <vector>

struct Permanent {
    bool is_token = false;
    bool is_tapped = false;
    bool has_summoning_sickness = true; 
    std::vector<Ability> abilities;
    std::vector<StaticAbility> static_abilities;
    Zone::Ownership controller = Zone::UNKNOWN;
    size_t timestamp_entered_battlefield = 0;  // For ordering simultaneous ETBs
    bool transformed = false;  // true when DFC is showing its back face
};

#endif /* PERMANENT_H */
