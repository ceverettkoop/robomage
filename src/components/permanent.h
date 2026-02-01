#ifndef PERMANENT_H
#define PERMANENT_H

#include "../ecs/entity.h"
#include "zone.h"

struct Permanent {
    bool is_token = false;
    bool is_tapped = false;
    bool has_summoning_sickness = true;  // True when entering battlefield
    Zone::Ownership controller = Zone::UNKNOWN;
    size_t timestamp_entered_battlefield = 0;  // For ordering simultaneous ETBs
};

#endif /* PERMANENT_H */
