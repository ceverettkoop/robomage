#ifndef QUERIES_FACE_DOWN_H
#define QUERIES_FACE_DOWN_H

#include "../components/cast_option.h"
#include "../components/face_down_card.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"

struct CardData;

// ── Face-down objects (CR 708, 406.3) ─────────────────────────────────────────
//
// What a face-down permanent or spell is, the CR 708.2 characteristics, is its entity's CardData:
// the card is turned over, its face is the one a row of kSpellOverlayRows gives (face_down_face),
// and the card itself waits on a FaceDownCard. So nothing that reads an object's characteristics
// knows or asks whether it is face down, and a card with no face-down state pays for none of this.
// Who it is, a question for the state format and nothing else, is face_down_card.

// Whether `e` lies face down where it is: the one read of Zone::is_face_down.
inline bool is_face_down(Entity e) {
    return global_coordinator.entity_has_component<Zone>(e) && global_coordinator.GetComponent<Zone>(e).is_face_down;
}

// The card under the face-down permanent or spell `e`; null for anything else, a face-down exiled
// card included.
inline const CardData *face_down_card(Entity e) {
    return global_coordinator.entity_has_component<FaceDownCard>(e)
               ? global_coordinator.GetComponent<FaceDownCard>(e).card.get()
               : nullptr;
}

// CR 708.3, 708.4: the card `card`, not yet a permanent, lies face down as `how` says (a face-down
// row of kSpellOverlayRows) before it enters, so its Permanent is built from that face, it enters
// as nothing else and its Permanent keeps `how` (EntryInfo::overlay).
void enter_face_down(Entity card, SpellOverlay how);

// CR 708.2a: `e` is turned face down where it is, and true. A permanent stays the same object and
// has the characteristics of `how` from now on; a card in exile is hidden by its Zone. False, and
// nothing happens, for what is already face down (CR 708.2b) and for a token
// (docs/rules_deviations.md).
bool turn_face_down(Entity e, SpellOverlay how);

// CR 708.8: `e` is turned face up where it is. A permanent has its card's copiable values again,
// the same object; nothing about it entering applies.
void turn_face_up(Entity e);

// CR 708.9: the face-down permanent or spell `e` is moving to a zone it won't be face down in. It
// is the card it is again from here, so the move's event shows that card to every viewer (the
// reveal). Nothing happens to anything else.
void reveal_as_it_leaves(Entity e);

#endif /* QUERIES_FACE_DOWN_H */
