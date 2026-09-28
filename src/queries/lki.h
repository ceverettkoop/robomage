#ifndef QUERIES_LKI_H
#define QUERIES_LKI_H

#include <cstdint>
#include <string>
#include "../ecs/entity.h"

// ── Object identity (CR 400.7) and last-known information (CR 608.2h) ────────

// CR 400.7 object identity: the entity's current Zone::obj_gen, assigning a fresh one first if it
// has none (an object placed without a zone move — a preset battlefield, a token created in play —
// reads 0), so the returned stamp is always nonzero for an entity with a Zone. 0 for an entity
// without one (a player). Defined in lki.cpp (needs cur_game).
uint64_t stamp_object_gen(Entity e);

// The leaving-the-battlefield snapshot (CR 608.2h) captured for `e` as it last left play, or
// null if none was captured or it is superseded. A card's snapshot describes the object that
// left, and only the resolving spell or ability that moved it reads it as that object (CR
// 608.2h: Swords to Plowshares' life gain); to every later reader the card is a new object with
// its printed characteristics (CR 400.7). So a card's snapshot is superseded when that
// resolution ends, at birth when the card left outside a resolution (a state-based death, a
// cost), and at the card's next zone change. A token keeps its snapshot after it ceases to exist
// (CR 111.7) — no new object ever takes its place — while its entity id is not issued again.
// Defined in lki.cpp.
struct LastKnownInfo;
const LastKnownInfo *lki_for(Entity e);

// The snapshot of the object that left the battlefield as `e`, superseded or not. For
// look-backs that refer to that departed object rather than to whatever the card is now: its
// own leaves/dies trigger collection (CR 603.10) and an ability whose source it was, resolving
// later (Blast Zone's counters, Amped Raptor's cast-from-hand gate; CR 608.2h).
const LastKnownInfo *departed_lki_for(Entity e);

// Called as `e` makes a zone change that does not start on the battlefield: its snapshot from
// an earlier battlefield exit now describes an older object (CR 400.7), so mark it superseded.
void supersede_last_known_info(Entity e);

// Supersede every card's snapshot (tokens excepted). Called as a resolution ends: the effects
// that read a card's last-known information as the object that just left (a sub-ability of the
// spell that moved it, CR 608.2h) have all run.
void supersede_departed_cards();

// The display name `e` had as it last left the battlefield ("Construct token" for a token),
// from its last-known information; empty when none was captured.
std::string last_known_name(Entity e);

#endif /* QUERIES_LKI_H */
