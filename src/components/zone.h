#ifndef ZONE_H
#define ZONE_H

struct Zone {
        enum ZoneValue { LIBRARY, BATTLEFIELD, HAND, STACK, GRAVEYARD, EXILE, SIDEBOARD };

        Zone(ZoneValue in_value) { value = in_value; };
        ZoneValue value;
        unsigned distance_from_top = 0; //0 is top
};

#endif /* ZONE_H */
