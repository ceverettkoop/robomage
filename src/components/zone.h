#ifndef ZONE_H
#define ZONE_H

#include <cstddef>
#include <cstdint>

struct Zone {
        enum ZoneValue: int { LIBRARY, BATTLEFIELD, HAND, STACK, GRAVEYARD, EXILE, SIDEBOARD };
        enum Ownership: int {UNKNOWN, PLAYER_A, PLAYER_B};

        Zone();
        Zone(ZoneValue in_loc, Ownership in_owner, Ownership in_controller);

        ZoneValue location;
        size_t distance_from_top = 0; //0 is top, stored for all zones but only relevant in the ordered zones: library, graveyard, and exile (all per-owner, recency-ordered so 0 is newest/top). Not meaningful for hand, battlefield, or sideboard.
        // CR 400.7 object identity: a globally-unique generation stamped every time this entity
        // ENTERS a zone (assigned in Orderer::add_to_zone from Game::next_obj_gen). A card that
        // leaves and re-enters a zone becomes a NEW object; because the engine reuses the same
        // entity id across that move (Tamiyo/Ajani exile-and-return-transformed, any flicker), a
        // stamp is the only way a spell/ability that targeted the OLD object can tell that the
        // object it chose no longer exists — even when a same-id, same-type incarnation reoccupies
        // the old zone. Snapshotted at target selection and re-checked at resolution (CR 608.2b).
        // 0 = never stamped (a target with no snapshot skips the check).
        uint64_t obj_gen = 0;
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
