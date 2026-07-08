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
    // Place a freshly-created entity directly on top of the stack (CR 707.10: a spell copy is
    // *created* on the stack, it does not move there from another zone). Adds a STACK Zone owned
    // by `controller`, sets it as the new top (distance_from_top 0, shifting the rest down), and
    // marks it publicly revealed — but fires NO CARD_CHANGED_ZONE event and NO MOVE_TO_ZONE
    // replacement, so an Origin$ Hand / leaves-a-zone trigger or move replacement can't mis-match
    // an object that was never in that zone. Use for copies/tokens that come into existence on the
    // stack rather than add_to_zone (which models a genuine zone transition).
    void place_created_on_stack(Entity target, Zone::Ownership controller);
    // Create a standalone ability entity and place it on the stack. `ability` must
    // already have source/controller/target populated. Returns the new entity.
    Entity push_ability_onto_stack(const Ability &ability, Zone::Ownership controller);
    std::vector<Entity> get_library_contents(Zone::Ownership owner);
    // The top `n` cards of `owner`'s library, ordered so result[0] is the actual top card
    // (sorted by Zone::distance_from_top ascending); fewer than `n` if the library is smaller.
    // The single top-of-library accessor for scry/dig/surveil/rearrange/peek — get_library_contents
    // returns cards in arbitrary entity order, so reading the top must go through here rather than
    // re-sorting inline at each call site.
    std::vector<Entity> get_library_top(Zone::Ownership owner, size_t n);
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
    // CR 103.5: keep/mulligan decisions are announced in turn order each round,
    // starting player first (matters in bo3 games 2-3 where B may be on the play).
    void do_london_mulligan(bool player_a_goes_first);
    // CR 103.6/103.6b opening-hand actions: after mulligans resolve (each player has kept and
    // bottomed), each player in APNAP order — starting player first — may run the
    // MayEffectFromOpeningHand ability of each such card in their kept hand (Leyline of the
    // Void: begin the game with it on the battlefield). Each offer is an OPTIONAL_YESNO
    // decision seated on the deciding player. Returns true if any ability ran, so the caller
    // can re-run state-based effects and a card put onto the battlefield gets its Permanent
    // component before the first turn begins.
    bool do_opening_hand_actions(bool player_a_goes_first);
    std::vector<Entity> place_on_battlefield(const std::vector<std::string> &card_names,
                                             Zone::Ownership owner);
    // Test-harness helper: start cards already in a player's graveyard (mirrors
    // place_on_battlefield) so graveyard-functioning cards can be exercised in isolation.
    std::vector<Entity> place_in_graveyard(const std::vector<std::string> &card_names,
                                           Zone::Ownership owner);
    // Test-harness helper: start cards already in a non-battlefield zone owned by a
    // player (graveyard/exile/sideboard) so zone-change effects that pull from those
    // zones (e.g. Karn's "from exile or outside the game") can be exercised in isolation.
    // No Permanent/Creature components are attached.
    std::vector<Entity> place_in_zone(const std::vector<std::string> &card_names,
                                      Zone::Ownership owner, Zone::ZoneValue zone);

private:
    // Draw a single card for `player`, first offering any available dredge
    // replacement (rule 702.52a) via replacement::dispatch. Sets the decked-out
    // loss if the library is empty.
    void draw_one(Zone::Ownership player, bool fire_draw_event = true);

};

#endif /* ORDERER_H */
