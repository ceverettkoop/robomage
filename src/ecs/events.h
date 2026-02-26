#ifndef EVENTS_H
#define EVENTS_H

#include "event.h"

// Event IDs
namespace Events {
    constexpr EventId CREATURE_DIED    = 1;
    constexpr EventId PLAYER_DREW_CARD = 2;
}

// Param IDs used across events
namespace Params {
    constexpr ParamId ENTITY = 1;  // The primary entity involved in an event
    constexpr ParamId PLAYER = 2;  // The player entity involved in an event
}

#endif /* EVENTS_H */
