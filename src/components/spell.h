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
    bool cast_with_escape = false;  // cast from the graveyard for its escape cost — the resulting permanent "escaped" (CR 702.139), read by an "if it escaped" clause (Uro)
    bool cast_with_offspring = false;  // Offspring additional cost paid — make a 1/1 token copy on ETB
    bool cast_with_impending = false;  // cast for its Impending alternate cost (CR 702.175) — the resulting permanent enters with time counters and isn't a creature until they're gone
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
    // Gift (CR 702.176): true iff the spell's controller PROMISED the gift to the opponent as
    // this spell was cast (an optional choice, not a cost — CR 702.176b). Read at resolution to
    // give the opponent the gift, and (via cur_game.pending_gift_promised at cast time) to switch
    // a Count$PromisedGift-driven effect (e.g. Into the Flood Maw widening its bounce target).
    bool gift_promised = false;
    // A COPY of a spell on the stack (CR 707.10): a copy is not a card. When it resolves (or is
    // countered) it ceases to exist rather than moving to a graveyard/library — the stack
    // manager destroys it instead of sending it to a zone. Set by run_copy_spell.
    bool is_copy = false;
    // Total mana actually spent to cast this spell (the pip count of the mana cost paid, CR
    // 106/601.2g). 0 for a spell cast for free or via an alternative cost that pays no mana
    // (Force of Will's pitch+life, a 0-cost spell). Set at cast time in action_processor and read
    // by a SpellCast trigger's ValidSA$ Spell.ManaSpent <op><n> filter (Roiling Vortex: "if no
    // mana was spent"). Shared, general — Lavinia, Azorius Renegade reads the same field.
    int mana_spent = 0;
};

#endif /* SPELL_H */
