#ifndef QUERIES_COPIES_H
#define QUERIES_COPIES_H

#include "../components/carddata.h"
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

// The card values of a copy of `original`, card or token (CR 707.2, 707.9): the face it shows (CR
// 707.8, a copy of a double-faced permanent has that face alone) and a token's values as a card.
CardData copied_card(Entity original, const CopyOverlay &overlay);

// CR 707.5: the card `entering`, on the battlefield with no Permanent yet, has `original`'s copiable
// values as it enters, so its Permanent is built from them and every "as this enters" of theirs
// applies. Its own card goes on the entry (EntryInfo::printed_card), for the Permanent.
void enter_as_copy(Entity entering, Entity original, const CopyOverlay &overlay);

// The permanent `permanent` has `values` now, the same object as before: its name, types and
// Creature follow them at once, its abilities at the next state-based pass.
void permanent_takes_values(Entity permanent, const CardData &values);

// CR 707.2 / 707.9b in-place copy: `permanent` becomes a copy of `original` while staying the SAME
// object, overwriting its copiable characteristics (name, types, abilities, keywords, static
// abilities) and leaving non-copiable state untouched (707.2): same entity, counters, tapped state
// and attachments. So the next state-based pass re-derives its activated, mana, triggered and static
// abilities from the new characteristics (apply_permanent_components + apply_land_abilities). Its
// printed CardData is stashed on Permanent::printed_card (first copy only, so a re-copy keeps the
// original printed identity) for end_copy_effect. A token becomes the copy through its Token, keeping
// its identity (Token::script_name). `until_eot` (Clone's Duration$ UntilEndOfTurn): the copy ends at
// cleanup (end_copy_until_eot). Its Creature follows the copied values at once: base P/T, or none.
void become_copy(Entity permanent, Entity original, const CopyOverlay &overlay, bool until_eot = false);

// CR 514.2: what a copy effect until end of turn replaced comes back: `permanent`'s values, and its
// Creature with them. No-op for a permanent with none.
void end_copy_until_eot(Entity permanent);

// Undo an in-place copy effect on a permanent leaving the battlefield: put the stashed printed
// CardData back on the card and its printed name/types on the (about-to-be-stripped) Permanent,
// so every reader between now and the state-based strip sees the card itself. No-op for a
// permanent that is not a copy.
void end_copy_effect(Entity permanent);

#endif /* QUERIES_COPIES_H */
