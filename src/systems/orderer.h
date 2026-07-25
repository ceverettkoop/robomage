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
    // top_seen_by_owner (only meaningful for a non-bottom LIBRARY placement): does the library's
    // OWNER see the identity of the card being placed on top? True for self-facing effects
    // (Brainstorm/Ponder put-backs, surveil, own dig) where the actor IS the owner. False for a
    // fateseal-class dig on an OPPONENT's library (Jace +2), where the looker is the controller and
    // the owner must NOT learn the card. When false the owner's known-top cache still shifts down
    // (positions stay honest) but records an UNKNOWN marker at slot 0 instead of the real identity.
    // exile_face_down (CR 708): when true and destination is EXILE, the card is exiled FACE DOWN —
    // its Zone::is_face_down flag is set and it is NOT accumulated into the owner's public revealed
    // multi-hot (its identity stays hidden from the opponent until an effect turns it face up).
    void add_to_zone(bool on_bottom, Entity target, Zone::ZoneValue destination,
                     bool top_seen_by_owner = true, bool exile_face_down = false);
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
    // The post-replacement half of a single draw: move the top library card to hand
    // (or set the decked-out loss on an empty library), with the draw logs and the
    // PLAYER_DREW_CARD event. Called directly by the suspendable draw loops
    // (resume_pending_draws, effects::draw_n_with_replacements) once the dredge
    // question is settled; draw_one routes through it after its blocking dispatch.
    void perform_draw(Zone::Ownership player, bool fire_draw_event = true);
    // Performs the base draw plus any additive-draw-replacement bonus cards (CR 614.1/614.5,
    // Quantum Riddler) as one replaced event. Single owner of the additive-draw application:
    // every draw path routes its "the draw actually happens" case through here (never the
    // dredge-replaced case), so the bonus is applied in exactly one place. The bonus is read
    // immediately before the base draw, so its "one or fewer cards in hand" gate sees the
    // pre-draw hand.
    void perform_draw_with_bonus(Zone::Ownership player, bool fire_draw_event = true);
    // Apply a chosen dredge (CR 702.52a): mill `mill_ct`, return `source` from the
    // graveyard to hand, and log — the replaced draw's outcome. Shared by the
    // blocking draw_one path and the suspendable draw loops.
    void apply_dredge(Zone::Ownership player, Entity source, int mill_ct);
    // Move the top `ct` cards of a player's library to their graveyard. Returns the
    // milled entities in mill order (top first).
    std::vector<Entity> mill(Zone::Ownership player, size_t ct);
    std::vector<Entity> get_graveyard(Zone::Ownership owner);
    std::vector<Entity> get_stack();
    // NOTE: the London mulligan and CR 103.6b opening-hand decision loops live in
    // game_driver.cpp's pregame gate (run_pregame_step) — loop-top, snapshot-safe
    // decisions driven by PregameState. Only the zone operations they use
    // (get_hand / add_to_zone / shuffle_library / draw) belong to the Orderer.
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
