#ifndef EVENTS_H
#define EVENTS_H

#include "event.h"

// Event IDs
// Each constant is annotated with the card-script phrase it corresponds to.
namespace Events {
    constexpr EventId CREATURE_DIED          = 1;  // "when a creature dies" / ChangesZone Origin$Battlefield Destination$Graveyard ValidCard$Creature
    constexpr EventId PLAYER_DREW_CARD       = 2;  // "whenever a player draws a card"
    constexpr EventId CREATURE_ENTERED       = 3;  // "when a creature enters" / ChangesZone Destination$Battlefield ValidCard$Creature (or .Other)
    constexpr EventId UPKEEP_BEGAN           = 4;  // "at the beginning of [your] upkeep" / Phase$ Upkeep
    constexpr EventId NONCREATURE_SPELL_CAST = 5;  // "whenever [you cast] a noncreature spell" / SpellCast ValidCard$nonCreature
    constexpr EventId END_STEP_BEGAN         = 6;  // "at the beginning of [your] end step" / Phase$ EndStep
    constexpr EventId PERMANENT_ENTERED      = 7;  // "when this permanent enters" / ChangesZone Destination$Battlefield ValidCard$Card.Self
}

// Param IDs used across events
namespace Params {
    constexpr ParamId ENTITY = 1;  // The primary entity involved in an event
    constexpr ParamId PLAYER = 2;  // The player entity involved in an event
}

#endif /* EVENTS_H */
