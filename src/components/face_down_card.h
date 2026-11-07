#ifndef FACE_DOWN_CARD_H
#define FACE_DOWN_CARD_H

#include <memory>

struct CardData;

// The card under a face-down permanent or spell (CR 708). The entity's own CardData is the face it
// lies on (face_down_face, queries/overlays.h), so every read of what the object is finds that; this
// is who it is, read through face_down_card (queries/face_down.h) and by nothing that a viewer's
// frame shows unless identity_visible says so. A face-down exiled card has none: it stays the card
// it is, and only its Zone::is_face_down hides it (CR 406.3).
struct FaceDownCard {
        std::shared_ptr<const CardData> card;
};

#endif /* FACE_DOWN_CARD_H */
