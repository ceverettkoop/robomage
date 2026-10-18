#ifndef REVEALS_H
#define REVEALS_H

#include "../components/zone.h"
#include "../ecs/entity.h"

// Records a card of `owner`'s revealed to both players (CR 701.20a). A card revealed in a library
// is remembered as revealed for the rest of the effect (Game::revealed_in_library: an effect that
// then moves it moves it publicly) and known to its owner's opponent at its place until it moves
// or the library is shuffled (Game::OppKnownLibraryTop, which the menu collapse reads). A card
// revealed anywhere else leaves nothing to record. No-op for a copy of a spell (not a card, CR
// 707.10) and for an owner that is not a seat.
void mark_card_revealed(Entity e, Zone::Ownership owner);

#endif /* REVEALS_H */
