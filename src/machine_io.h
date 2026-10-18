#ifndef MACHINE_IO_H
#define MACHINE_IO_H

#include "classes/gamestate.h"
#include "components/zone.h"
#include "ecs/entity.h"

// The CLI board's source (until plan step 4b) and the card-identity read of an action's or a stack
// object's source.

// Fills `gs` with the board as `viewer` sees it (Zone::UNKNOWN: the priority player): life, the
// poison and energy counters and mana pools of both players, the stack top first, each side's
// permanents in ascending entity order (phased-out ones included) and the viewer's hand.
void populate_gamestate(GameState *gs, Zone::Ownership viewer = Zone::UNKNOWN);

// Card vocab index for an action's source entity (or a stack entity): the card's vocab index, the
// token-band index (else TOKEN_SENTINEL) for a token, or -1 for a non-card source. Walks the
// Permanent(token) -> CardData -> Ability.source(Permanent(token)/CardData) chain, then the
// object's last-known information for one that ceased to exist.
int action_card_vocab_idx(Entity e);

#endif /* MACHINE_IO_H */
