#ifndef ZONE_H
#define ZONE_H

#include <cstddef>

struct Zone {
        enum ZoneValue: int { LIBRARY, BATTLEFIELD, HAND, STACK, GRAVEYARD, EXILE, SIDEBOARD };
        enum Ownership: int {UNKNOWN, PLAYER_A, PLAYER_B};

        Zone();
        Zone(ZoneValue in_loc, Ownership in_owner, Ownership in_controller);

        ZoneValue location;
        size_t distance_from_top = 0; //0 is top, stored for all zones but only relevant in library and graveyard (maybe exile?)
        Ownership owner = UNKNOWN;
        Ownership controller = UNKNOWN; //only relevant for battlefield
        // The card's identity is known to its non-owner while in a hidden zone
        // (e.g. revealed in hand by Duress/Thoughtseize, or a revealed tutor target
        // placed into hand). Reset on every zone change; only meaningful in HAND.
        bool identity_known = false;
};

#endif /* ZONE_H */
