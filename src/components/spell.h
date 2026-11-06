#ifndef SPELL_H
#define SPELL_H

#include "chosen_cost.h"
#include "zone.h"

// Present only while the entity is on the stack as a spell.
// Removed when the spell resolves, is countered, or otherwise leaves the stack.
struct Spell {
    Zone::Ownership caster = Zone::UNKNOWN;
    ChosenCost chosen;  // how it was cast and what was paid
    bool cant_be_countered = false;
    // A COPY of a spell on the stack (CR 707.10): a copy is not a card. When it resolves (or is
    // countered) it ceases to exist rather than moving to a graveyard/library — the stack
    // manager destroys it instead of sending it to a zone. Set by run_copy_spell.
    bool is_copy = false;
    // Where the spell goes when it finishes resolving (CR 608.2n): its graveyard, unless an
    // instruction of its own moves it elsewhere (Burning Wish: "Exile Burning Wish").
    Zone::ZoneValue resolves_to = Zone::GRAVEYARD;
};

#endif /* SPELL_H */
