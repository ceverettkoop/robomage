#ifndef EVENTS_H
#define EVENTS_H

#include "event.h"

// Event IDs
namespace Events {
    constexpr EventId CREATURE_DIED = 1;
}

// Param IDs used across events
namespace Params {
    constexpr ParamId ENTITY = 1;  // The primary entity involved in an event
}

#endif /* EVENTS_H */
