#ifndef QUERIES_COPIES_H
#define QUERIES_COPIES_H

#include "../components/copy_overlay.h"
#include "../components/token.h"
#include "../ecs/entity.h"

// ── Copies of permanents (CR 707) ─────────────────────────────────────────────
// A copy takes the copiable values of the original (CR 707.2) with the effect's exceptions
// (CopyOverlay, CR 707.9). Both forms a copy takes go through here.

// The token characteristics of a copy of permanent `original`: a token source's own Token; a
// nontoken permanent's printed card, on the face it shows (CR 712.8e) — name, types, colors, mana
// value, P/T, keywords, and its triggered, activated and static abilities and replacement effects.
// Counters, pumps and type- or ability-changing effects are not copiable, so the live Permanent /
// Creature state is not read. An empty name: `original` has no card or token to copy.
Token copied_token(Entity original, const CopyOverlay &overlay);

// CR 707.2 / 707.9b in-place copy: `permanent` becomes a copy of `original` while staying the SAME
// object, overwriting its copiable characteristics (name, types, abilities, keywords, static
// abilities) and leaving non-copiable state untouched (707.2): same entity, counters, tapped state
// and attachments. So the next state-based pass re-derives its activated, mana, triggered and static
// abilities from the new characteristics (apply_permanent_components + apply_land_abilities). Its
// printed CardData is stashed on Permanent::printed_card (first copy only, so a re-copy keeps the
// original printed identity) for end_copy_effect.
void become_copy(Entity permanent, Entity original, const CopyOverlay &overlay);

// Undo an in-place copy effect on a permanent leaving the battlefield: put the stashed printed
// CardData back on the card and its printed name/types on the (about-to-be-stripped) Permanent,
// so every reader between now and the state-based strip sees the card itself. No-op for a
// permanent that is not a copy.
void end_copy_effect(Entity permanent);

#endif /* QUERIES_COPIES_H */
