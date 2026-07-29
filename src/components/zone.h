#ifndef ZONE_H
#define ZONE_H

#include <cstddef>

struct Zone {
        enum ZoneValue: int { LIBRARY, BATTLEFIELD, HAND, STACK, GRAVEYARD, EXILE, SIDEBOARD };
        enum Ownership: int {UNKNOWN, PLAYER_A, PLAYER_B};

        Zone();
        Zone(ZoneValue in_loc, Ownership in_owner, Ownership in_controller);

        ZoneValue location;
        size_t distance_from_top = 0; //0 is top, stored for all zones but only relevant in the ordered zones: library, graveyard, and exile (all per-owner, recency-ordered so 0 is newest/top). Not meaningful for hand, battlefield, or sideboard.
        Ownership owner = UNKNOWN;
        Ownership controller = UNKNOWN; //only relevant for battlefield
        // The card's identity is known to its non-owner while in a hidden zone
        // (e.g. revealed in hand by Duress/Thoughtseize, or a revealed tutor target
        // placed into hand). Reset on every zone change; only meaningful in HAND.
        bool identity_known = false;
        // A card exiled FACE DOWN (CR 708 / 701.35 — e.g. The Creation of Avacyn chapter I
        // "exile it face down"). While face down in exile its characteristics are hidden; a
        // later effect (a SetState$ TurnFaceUp) turns it face up by clearing this flag. Reset
        // on every zone change (a card that leaves exile is no longer the face-down object).
        bool is_face_down = false;
};

#endif /* ZONE_H */
