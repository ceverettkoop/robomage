#ifndef QUERIES_ZONES_H
#define QUERIES_ZONES_H

#include <set>
#include <vector>
#include "../components/zone.h"
#include "../ecs/entity.h"

// ── Zoned objects ─────────────────────────────────────────────────────────────

// Every zoned object: each entity carrying a Zone component — cards in every zone (sideboard
// included), tokens, and the ability objects on the stack — in entity-id order. It is the
// systems' shared entity set (Orderer, StateManager and StackManager all have the {Zone}
// signature), exposed for free functions that have no system's mEntities in reach (the SVar
// evaluator, the replacement dispatcher, the observation serializer, search determinization).
// Code in a system, or handed an orderer, passes its own mEntities instead. Player entities and
// the card_db template entities carry no Zone and are not in it (reach a player through
// cur_game's player entities). Bound by init_ecs (bind_zoned_entities); defined in zones.cpp.
const std::set<Entity> &zoned_entities();

// Makes `entities` (a {Zone}-signature system's mEntities) the set zoned_entities() returns.
// Called once per init_ecs, as each game's ECS registers its systems.
void bind_zoned_entities(const std::set<Entity> &entities);

// The objects `owner` has in `zone` (any owner when UNKNOWN), among `entities`, in entity-id
// order — not the zone's own order (Orderer::get_library_contents / get_graveyard give that).
// For the battlefield use battlefield_permanents, which applies the phasing rule.
std::vector<Entity> zone_objects(const std::set<Entity> &entities, Zone::ZoneValue zone,
                                 Zone::Ownership owner = Zone::UNKNOWN);

// ── Cards outside the battlefield ─────────────────────────────────────────────

// CR 400.2: the graveyard, battlefield, stack and exile are public zones; the library and hand
// (and the sideboard, outside the game) are hidden.
inline bool is_public_zone(Zone::ZoneValue z) {
    return z == Zone::BATTLEFIELD || z == Zone::GRAVEYARD || z == Zone::STACK || z == Zone::EXILE;
}

// Distinct card types (CR 205.2) among cards in `owner`'s graveyard, excluding `except`
// (pass 0 to count every card). Single source for delirium / Escape's ExileFromGrave
// group-type constraint / any "card types in your graveyard" count over a live entity set.
int graveyard_card_types(Zone::Ownership owner, const std::set<Entity> &entities,
                         Entity except = 0);

// Returns true when the given player has 4+ card types among cards in their graveyard.
inline bool check_delirium(Zone::Ownership owner, const std::set<Entity> &entities) {
    return graveyard_card_types(owner, entities) >= 4;
}

// Number of cards in `owner`'s graveyard, excluding `except` (pass 0 to count every card).
// Single source for the literal-count Escape ExileFromGrave cost (Uro: "exile five other
// cards") legality check and payment loop.
int graveyard_card_count(Zone::Ownership owner, const std::set<Entity> &entities,
                         Entity except = 0);

// ── Play permissions from the graveyard and exile ────────────────────────────
// Whether `player` has a permission to play (cast, or play as a land) the card `card`
// from the zone it is in, IGNORING timing, cost affordability, targets and cast
// prohibitions: those stay in the legal-action enumeration. The single source for
// "which play routes exist for this card": the enumeration's graveyard/exile loops gate on
// its `sources` bits, and the ML observation's graveyard/exile playable flags read it, so
// the two cannot disagree about which cards are playable from those zones.
// Sources covered:
//   FLASHBACK         a graveyard card its owner may cast with flashback (CR 702.34)
//   ESCAPE            a graveyard card its owner may cast with escape (CR 702.139)
//   GRAVEYARD_CAST    a nonland graveyard card in Game::resolved_effects.may_cast_this_turn (Emry's grant;
//                     the owner, this turn)
//   GRAVEYARD_LAND    a land card in its owner's graveyard while a static lets the owner play
//                     lands from the graveyard (Icetill Explorer, Mole Man)
//   EXILE_GRANT       an exiled card with a Game::resolved_effects.impulse_cast_permission whose caster is
//                     `player` (Light Up the Stage, Ugin's -11, Dauthi Voidwalker, warp); a land
//                     only under a "play" grant that allows lands
// Not covered: a suspended card still carrying time counters (no permission until the last
// counter is removed), a card cast during a resolution (Amped Raptor, suspend's last counter:
// no standing permission exists), graveyard-activated abilities such as unearth (they activate
// an ability, not play the card), and hand casts.
// expires_this_turn is true when the card is playable and EVERY source covering it lapses at
// this turn's cleanup (Emry's grant, a "this turn" exile grant, or a Light Up the Stage grant
// during the caster's next turn); false for a static or keyword source or a grant that
// outlives this turn. Defined in zones.cpp (needs cur_game and rules_mod).
struct CardPlayPermission {
    enum Source : unsigned {
        FLASHBACK      = 1u << 0,
        ESCAPE         = 1u << 1,
        GRAVEYARD_CAST = 1u << 2,
        GRAVEYARD_LAND = 1u << 3,
        EXILE_GRANT    = 1u << 4,
    };
    unsigned sources = 0;
    bool expires_this_turn = false;
    bool playable() const { return sources != 0; }
};
CardPlayPermission card_play_permission(Entity card, Zone::Ownership player);

// Whether `a` and `b` are interchangeable cards to choose between: two copies of the same card in
// the same player's graveyard, playable from there by the same players on the same terms
// (card_play_permission), or in the same player's library (a search, a Doomsday pile), neither of
// them revealed there or known to the opponent at its place (Game::KnownLibraryTop). Choosing one
// or the other makes no difference to the game, and the observation encodes the choices
// identically, so a choice menu offers one of them. Defined in zones.cpp.
bool interchangeable_cards(Entity a, Entity b);

// The card `source` exiled and still tracks via Permanent::exiled_with — the association a Saga
// records at chapter I so its later chapters can act on "the card exiled with this" (Defined$
// ExiledWith / ExiledWith$CardManaCost, The Creation of Avacyn). Returns the most-recently exiled
// entry that is still the object it exiled (skipping any that have since become new objects or
// left the game), or 0 if none.
Entity exiled_with_card(Entity source);

// The most recently exiled card linked to `host` (Permanent::exiled_with) that STILL has a live
// "return it from exile" path scheduled in cur_game.delayed_triggers, or 0 if none. This
// distinguishes an exile-with-return host (a Static Prison holding a real Murktide) from a
// permanent exiler with no return (Skyclave Apparition) or one holding a token. Two return
// shapes exist (see effect_change_zone.cpp / effect_delayed_trigger.cpp):
//   (A) "exile until host leaves" (Static Prison, Sheltered by Ghosts): a fire_on_leave_battlefield
//       trigger watching `host`, whose ChangeZone fire ability moves the card EXILE -> its origin,
//       with the card in the fire ability's restore_remembered_exiled_with.
//   (B) end-of-turn blink (Flickerwisp / Phelia): a phase-based trigger whose ChangeZone fire
//       ability moves the card EXILE -> battlefield, with the card in the trigger's
//       remembered_objects (also mirrored into restore_remembered_exiled_with).
// Skyclave Apparition / Keen-Eyed Curator populate exiled_with but register NO such trigger, so
// they return 0. Conservative: any ChangeZone delayed trigger that would move the card OUT of exile
// (origin EXILE, destination != EXILE) and references that card counts as a return path. Defined in
// zones.cpp (needs cur_game.delayed_triggers).
Entity returnable_exiled_card(Entity host);

#endif /* QUERIES_ZONES_H */
