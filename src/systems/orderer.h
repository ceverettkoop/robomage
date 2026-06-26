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
    // fire_draw_event=false suppresses the PLAYER_DREW_CARD trigger event (used for
    // opening-hand and mulligan draws, which are not "draws" that trigger abilities).
    void draw(Zone::Ownership player, size_t ct, bool fire_draw_event = true);
    // Move the top `ct` cards of a player's library to their graveyard. Returns the
    // milled entities in mill order (top first).
    std::vector<Entity> mill(Zone::Ownership player, size_t ct);
    std::vector<Entity> get_graveyard(Zone::Ownership owner);
    std::vector<Entity> get_stack();
    void do_london_mulligan();
    std::vector<Entity> place_on_battlefield(const std::vector<std::string> &card_names,
                                             Zone::Ownership owner);
    // Test-harness helper: start cards already in a player's graveyard (mirrors
    // place_on_battlefield) so graveyard-functioning cards can be exercised in isolation.
    std::vector<Entity> place_in_graveyard(const std::vector<std::string> &card_names,
                                           Zone::Ownership owner);

private:
    // Draw a single card for `player`, first offering any available dredge
    // replacement (rule 702.52a) via replacement::dispatch. Sets the decked-out
    // loss if the library is empty.
    void draw_one(Zone::Ownership player, bool fire_draw_event = true);

};

#endif /* ORDERER_H */
