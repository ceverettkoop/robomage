#ifndef ORDERER_H
#define ORDERER_H

#include "../ecs/system.h"
#include "../ecs/entity.h"
#include "../components/zone.h"
#include <vector>
#include <memory>

struct Deck;

class Orderer : public System {

public:
    static void init();
    void add_to_zone(bool on_bottom, Entity target, Zone::ZoneValue destination);
    std::vector<Entity> get_library_contents(Zone::Ownership owner);
    void shuffle_library(Zone::Ownership owner);
    void generate_libraries(const Deck &deck_a, const Deck &deck_b);
};

#endif /* ORDERER_H */
