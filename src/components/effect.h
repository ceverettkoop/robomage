#ifndef EFFECT_H
#define EFFECT_H

#include "zone.h"
#include <set>
#include <string>
#include <vector>
#include "types.h"

// Replacement effects modify how an event occurs (e.g. entering tapped instead of untapped).
// Each Replacement is a one-time effect consumed when it fires; `applied` is set true afterwards.
struct Effect {
    struct Replacement {
        enum Kind {
            ENTERS_TAPPED,  // permanent enters the battlefield tapped
        };
        Kind kind = ENTERS_TAPPED;
        bool applies_to_self_only = false;  // only fires when the affected entity is the source itself
        bool applied = false;               // true once consumed; prevents re-application
    };

    std::vector<Replacement> replacements;

    // Reserved for future continuous effects (stat modifications, keyword grants, etc.)
    std::set<Zone::ZoneValue> affected_zones;
    std::set<Type> affected_types;
    std::string category;
    int amount = 0;
};



#endif /* EFFECT_H */
