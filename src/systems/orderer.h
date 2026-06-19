#ifndef ORDERER_H
#define ORDERER_H

#include "../ecs/system.h"
#include "../ecs/entity.h"
#include "../components/zone.h"
#include <vector>
#include <memory>

struct Deck;
struct Ability;

class Orderer : public System, public std::enable_shared_from_this<Orderer>{

public:
    static void init();
    void add_to_zone(bool on_bottom, Entity target, Zone::ZoneValue destination);
    // Create a standalone ability entity and place it on the stack. `ability` must
    // already have source/controller/target populated. Returns the new entity.
    Entity push_ability_onto_stack(const Ability &ability, Zone::Ownership controller);
    std::vector<Entity> get_library_contents(Zone::Ownership owner);
    std::vector<Entity> get_hand(Zone::Ownership owner);
    void shuffle_library(Zone::Ownership owner);
    void generate_libraries(const Deck &deck_a, const Deck &deck_b);
    void draw_hands();
    void draw(Zone::Ownership player, size_t ct);
    std::vector<Entity> get_stack();
    void do_london_mulligan();
    std::vector<Entity> place_on_battlefield(const std::vector<std::string> &card_names,
                                             Zone::Ownership owner);

};

#endif /* ORDERER_H */
