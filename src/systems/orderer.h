#ifndef ORDERER_H
#define ORDERER_H

#include "../ecs/system.h"
#include "../ecs/entity.h"

struct Deck;
struct Zone {
    enum ZoneValue : int;
};

class Orderer : public System {

public:
    static void init();
    void add_to_zone(bool on_bottom, Entity target, Zone::ZoneValue destination);
    void shuffle_zone(Zone::ZoneValue target);
    void generate_libraries(const Deck &deck_a, const Deck &deck_b);
};

#endif /* ORDERER_H */
