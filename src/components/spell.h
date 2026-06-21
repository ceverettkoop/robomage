#ifndef SPELL_H
#define SPELL_H

#include "zone.h"

// Present only while the entity is on the stack as a spell.
// Removed when the spell resolves, is countered, or otherwise leaves the stack.
struct Spell {
    Zone::Ownership caster = Zone::UNKNOWN;
    bool cast_with_flashback = false;
    bool cast_with_evoke = false;  // cast for its evoke cost — sacrifice itself when it enters
    bool cant_be_countered = false;
};

#endif /* SPELL_H */
