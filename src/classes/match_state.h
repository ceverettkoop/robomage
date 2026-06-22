#ifndef MATCH_STATE_H
#define MATCH_STATE_H

#include "../components/zone.h"
#include "../ecs/entity.h"

// Persistent, match-scoped accumulator of every opponent card revealed so far.
// Stored as a binary multi-hot keyed by card owner ("have they ever shown card
// X this match"). Lives outside the per-game ECS so it survives init_ecs()
// across the games of a bo3 — entity IDs reset each game, so a binary set is the
// only unambiguous representation across the game boundary.
constexpr int REVEALED_SIZE = 1024;  // mirror N_CARD_TYPES in machine_io.h

extern unsigned char g_revealed_by_a[REVEALED_SIZE];
extern unsigned char g_revealed_by_b[REVEALED_SIZE];

// Zero both arrays. Call at match start (before game 1 of a bo3, and at
// single-game init), never between bo3 games (persistence is the point).
void match_reset_revealed();

// Set the revealed bit for `owner` corresponding to entity `e`'s card identity.
// Idempotent; ignores tokens and entities with no registered card identity.
void mark_card_revealed(Entity e, Zone::Ownership owner);

#endif /* MATCH_STATE_H */
