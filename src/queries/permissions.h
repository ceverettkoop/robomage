#ifndef QUERIES_PERMISSIONS_H
#define QUERIES_PERMISSIONS_H

#include <cstdint>
#include <utility>
#include <vector>

#include "../components/zone.h"
#include "../ecs/entity.h"
#include "../ecs/entity_set.h"

// ── Permissions to play a card from the zone it is in ────────────────────────────
// Whether `player` has a permission to play (cast, or play as a land) the card `card`
// from the zone it is in, IGNORING timing, cost affordability, targets and cast
// prohibitions: those stay in the legal-action enumeration. The single source for
// "which play routes exist for this card": the enumeration's graveyard/exile rows gate on
// its `sources` bits, and the state format's play_permission rows read it, so
// the two cannot disagree about which cards are playable from those zones.
//
// A permission is a row of the registry in permissions.cpp: its kind, the zone it reaches, and what
// covers the card. A row reads whatever gives it: the card itself (flashback), an effect (Emry's
// grant), a static or an emblem's static (Icetill Explorer). Any zone can have rows; a zone with
// none is not looked at.
enum class Permission : uint8_t {
    FLASHBACK,       // a graveyard card its owner may cast with flashback (CR 702.34)
    ESCAPE,          // a graveyard card its owner may cast with escape (CR 702.139)
    // a nonland graveyard card in Game::resolved_effects.may_cast_this_turn (Emry's grant; the
    // owner, this turn), or while an Effect's static lets the owner cast spells from their
    // graveyard (Gaea's Will)
    GRAVEYARD_CAST,
    // a land card in its owner's graveyard while a static lets the owner play lands from the
    // graveyard (Icetill Explorer, Mole Man, Gaea's Will)
    GRAVEYARD_LAND,
    // an exiled card with a Game::resolved_effects.impulse_cast_permission whose caster is
    // `player` (Light Up the Stage, Ugin's -11, Dauthi Voidwalker, warp); a land only under a
    // "play" grant that allows lands
    EXILE_GRANT,
    // the exiled copy of a prepared permanent's prepare spell, for that permanent's controller
    // (CR 722.3c)
    PREPARED,
    // a card in its owner's graveyard whose own static says it may be cast from there (Hogaak)
    GRAVEYARD_SELF,
    // the top card of its owner's library while a static lets the owner cast it from there, as
    // itself or face down (Mystic Forge, CR 601.3e), read live: the permission is on whatever card
    // is on top now
    LIBRARY,
    NONE,  // no permission is needed: a card in its owner's hand
};

constexpr unsigned permission_bit(Permission p) { return 1u << static_cast<unsigned>(p); }

// Not covered: a suspended card still carrying time counters (no permission until the last
// counter is removed), a card cast during a resolution (Amped Raptor, suspend's last counter:
// no standing permission exists), graveyard-activated abilities such as unearth (they activate
// an ability, not play the card), and hand casts.
// expires_this_turn is true when the card is playable and EVERY source covering it lapses at
// this turn's cleanup (Emry's grant, a "this turn" exile grant, or a Light Up the Stage grant
// during the caster's next turn); false for a static or keyword source or a grant that
// outlives this turn.
struct PlayPermissions {
    unsigned sources = 0;  // permission_bit of each Permission covering the card
    bool expires_this_turn = false;
    bool playable() const { return sources != 0; }
    bool has(Permission p) const { return (sources & permission_bit(p)) != 0; }
};
PlayPermissions play_permissions(Entity card, Zone::Ownership player);

// The cast `card` was allowed by `permission` is made: a permission that lasts only until it is used
// is gone (an exile grant; it lapses once the card has left exile).
void spend_permission(Permission permission, Entity card);

// The cards among `entities` that `player` has a permission for, in entity-id order, with the
// bits of each.
std::vector<std::pair<Entity, unsigned>> play_routes(const EntitySet &entities, Zone::Ownership player);

#endif /* QUERIES_PERMISSIONS_H */
