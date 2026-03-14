#ifndef EVENTS_H
#define EVENTS_H

#include "event.h"

// Event IDs
// Each constant is annotated with the card-script phrase it corresponds to.
namespace Events {
    constexpr EventId CARD_CHANGED_ZONE      = 1;  // any card changes zone; Params: ENTITY=card, PLAYER=owner, ORIGIN=old zone, DESTINATION=new zone
    constexpr EventId PLAYER_DREW_CARD       = 2;  // "whenever a player draws a card"
    constexpr EventId UPKEEP_BEGAN           = 4;  // "at the beginning of [your] upkeep" / Phase$ Upkeep
    constexpr EventId NONCREATURE_SPELL_CAST = 5;  // "whenever [you cast] a noncreature spell" / SpellCast ValidCard$nonCreature
    constexpr EventId END_STEP_BEGAN         = 6;  // "at the beginning of [your] end step" / Phase$ EndStep
    constexpr EventId DRAW_STEP_BEGAN        = 7;  // "at the beginning of [your] draw step" / Phase$ Draw
    constexpr EventId SPELL_CAST             = 9;  // every spell cast; Params: PLAYER=caster
}

// Param IDs used across events
namespace Params {
    constexpr ParamId ENTITY      = 1;  // The primary entity involved in an event
    constexpr ParamId PLAYER      = 2;  // The player entity involved in an event
    constexpr ParamId ORIGIN      = 3;  // Zone::ZoneValue before the move (CARD_CHANGED_ZONE)
    constexpr ParamId DESTINATION = 4;  // Zone::ZoneValue after the move (CARD_CHANGED_ZONE)
}

#endif /* EVENTS_H */
