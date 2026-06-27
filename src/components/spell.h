#ifndef SPELL_H
#define SPELL_H

#include <vector>

#include "zone.h"

// Present only while the entity is on the stack as a spell.
// Removed when the spell resolves, is countered, or otherwise leaves the stack.
struct Spell {
    Zone::Ownership caster = Zone::UNKNOWN;
    bool cast_with_flashback = false;
    bool cast_with_evoke = false;  // cast for its evoke cost — sacrifice itself when it enters
    bool cast_with_offspring = false;  // Offspring additional cost paid — make a 1/1 token copy on ETB
    bool cant_be_countered = false;
    int x_paid = 0;  // value chosen for {X} at cast time (Chalice of the Void enters with X charge counters)
    // Kicker (CR 702.33): one flag per CardData::kicker_costs entry — kicked[i] is true iff the
    // (i+1)th kicker's additional cost was paid as this spell was cast. The spell "has been
    // kicked" if any flag is set (CR 702.33d). A linked "if it was kicked with its [N] kicker"
    // SpellCast trigger reads kicked[N-1]. Empty for a spell with no kicker.
    std::vector<bool> kicked;
    // Replicate (CR 702.x, Consign to Memory): how many times the replicate additional cost was
    // paid as this spell was cast. The on-cast copy effect makes this many copies of the spell
    // on the stack (each may choose new targets). 0 (or no Replicate) = no copies. A COPY of the
    // spell is not "cast", so it carries replicate_count = 0 and replicates nothing further.
    int replicate_count = 0;
    // A COPY of a spell on the stack (CR 707.10): a copy is not a card. When it resolves (or is
    // countered) it ceases to exist rather than moving to a graveyard/library — the stack
    // manager destroys it instead of sending it to a zone. Set by copy_spell_on_stack.
    bool is_copy = false;
};

#endif /* SPELL_H */
