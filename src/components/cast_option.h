#ifndef CAST_OPTION_H
#define CAST_OPTION_H

#include <cstdint>

#include "../queries/permissions.h"
#include "zone.h"

// Which face of the card is cast or played. BACK is a modal DFC's back face or a split card's
// second half: the entity's CardData is the front face, but on the stack the spell has only that
// face's characteristics (CR 712.8f, 709.3; read through active_face). A land played as its back
// face enters showing it (EntryInfo::enters_transformed). BOTH is a fused split spell: the two
// halves' combined characteristics (CR 702.102b, CardData::combined).
enum class CastFace : uint8_t { FRONT, BACK, BOTH };

// What a cast pays as its cost, each a row of kCastCostRows (action_processor.cpp), in this order. From
// ALTERNATIVE on, an alternative cost (CR 118.9).
enum class CastCost : uint8_t {
    MANA_COST,     // its mana cost, wherever it is cast from
    // Without paying its mana cost under a free-cast static (CR 118.9): the regular branch
    // with a {0} mana cost — additional costs, cost increases and the SetCost floor still apply, X
    // is 0 (CR 107.3b), the mana cost's hybrid and Phyrexian pips unpaid.
    NO_MANA_COST,
    GRANT,         // what its exile grant puts in place of its mana cost: a resource, or nothing
    FLASHBACK,     // exiled when it leaves the stack
    ESCAPE,        // from the graveyard for its escape cost: the resulting permanent "escaped"
                   // (CR 702.138), read by an "if it escaped" clause (Uro)
    ALTERNATIVE,   // for its alternative cost, and:
    EVOKE,         // sacrifices itself when it enters
    IMPENDING,     // CR 702.175: enters with time counters, and isn't a creature until they're gone
    WARP,          // exiled at the next end step (a delayed trigger), then castable from exile later
                   // for its normal cost
};

// What a cast changes on the spell beyond its face and its cost: a kind and a row of
// kSpellOverlayRows (queries/overlays.cpp), and a column of the cast table.
enum class SpellOverlay : uint8_t {
    NONE,
    OVERLOAD,  // CR 702.96: no targets, "target" reads as "each"
    ADVENTURE,  // CR 715: exiled as it resolves, and its caster may play the card from exile
};

// One way of casting a card: a row of a menu holds it, and the cast proceeds from it. The rows are
// cast_options.cpp's table; a row that is no cast holds the default, and PLAY_LAND rows their face.
struct CastOption {
        CastFace face = CastFace::FRONT;
        Zone::ZoneValue zone = Zone::HAND;      // where it is cast from
        CastCost cost = CastCost::MANA_COST;
        Permission permission = Permission::NONE;  // what lets it be cast from there
        bool offspring = false;  // Offspring additional cost paid: makes a 1/1 token copy on ETB
        // Where it goes as it leaves the stack whatever else would send it elsewhere, read by
        // spell_destination; GRAVEYARD leaves that to what moves it.
        Zone::ZoneValue destination = Zone::GRAVEYARD;
        const char *row = nullptr;  // the name of its row, its --coverage key
        SpellOverlay overlay = SpellOverlay::NONE;  // what it changes on the spell, read through cast_face

        bool alternative() const { return cost >= CastCost::ALTERNATIVE; }
        bool operator==(const CastOption &o) const {
            return face == o.face && zone == o.zone && cost == o.cost && permission == o.permission &&
                   offspring == o.offspring && destination == o.destination && overlay == o.overlay;
        }
};

#endif /* CAST_OPTION_H */
