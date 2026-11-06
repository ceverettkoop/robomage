#ifndef CHOSEN_COST_H
#define CHOSEN_COST_H

#include <cstdint>
#include <set>
#include <vector>

#include "../classes/colors.h"
#include "../object_ref.h"
#include "cast_option.h"
#include "cost_part.h"
#include "mana_unit.h"

// What a cast committed to (CR 601.2b-h): how the spell was cast and what was paid. Built on
// Game::PendingCast as the cast is proposed, and recorded on the Spell as it becomes cast
// (CR 601.2i) for as long as it is on the stack; until then a Spell holds only the face and overlay of its option.
struct ChosenCost {
        CastOption option;  // how it is cast
        // The X announced (CR 601.2b / 107.3a; a variable life cost's X too): Chalice of the Void enters
        // with X charge counters, the resolving spell's Count$xPaid reads it.
        int x = 0;
        // Kicker (CR 702.33): one flag per CardData::kicker_costs entry — kicked[i] is true iff the
        // (i+1)th kicker's additional cost was paid as this spell was cast. The spell "has been
        // kicked" if any flag is set (CR 702.33d). A linked "if it was kicked with its [N] kicker"
        // SpellCast trigger reads kicked[N-1]. Empty for a spell with no kicker.
        std::vector<bool> kicked;
        // Replicate (CR 702.x, Consign to Memory): how many times the replicate additional cost was
        // paid as this spell was cast. The on-cast copy effect makes this many copies of the spell
        // on the stack (each may choose new targets). 0 (or no Replicate) = no copies. A COPY of the
        // spell is not "cast", so it carries 0 and replicates nothing further.
        int replicate_count = 0;
        // Escalate (CR 702.120a): the modes chosen beyond the first, each paying the escalate cost.
        int escalate_count = 0;
        // Which of the card's additional_choices was announced (CR 601.2b); -1 with none.
        int additional_choice = -1;
        // Gift (CR 702.176): true iff the spell's controller PROMISED the gift to the opponent as
        // this spell was cast (an optional choice, not a cost — CR 702.176b). Read at resolution to
        // give the opponent the gift, and (via current_gift_promised() at cast time) to switch
        // a Count$PromisedGift-driven effect (e.g. Into the Flood Maw widening its bounce target).
        bool gift_promised = false;
        // The mana actually spent to cast this spell, what left the pool to pay it (CR 106/601.2g).
        // Empty for a spell cast for free or via an alternative cost that pays no mana (Force of Will's
        // pitch+life, a 0-cost spell); a pip a delve exile or a tap paid (TapToPay) is no mana spent,
        // so a spell can owe mana and spend none.
        ManaPaid mana_paid;
        // What the cost moved, as the objects they became where they went: sacrificed, returned,
        // discarded or exiled, in the order they moved.
        struct Moved {
                CostPart::Kind kind;
                ObjectRef object;
        };
        std::vector<Moved> moved;

        // How much mana was spent: read by a SpellCast trigger's ValidSA$ Spell.ManaSpent <op><n> filter
        // (Roiling Vortex: "if no mana was spent"). Shared, general — Lavinia, Azorius Renegade reads
        // the same count.
        int mana_spent() const { return static_cast<int>(mana_paid.size()); }
        // Converge (CR 702.90): the distinct colors of mana spent. Its size is the Converge count, taken
        // by the resolution frame as it resolves and read by a Count$Converge amount/condition bound
        // (Prismatic Ending).
        std::set<Colors> colors_spent() const { return colors_of(mana_paid); }

        bool flashback() const { return option.cost == CastCost::FLASHBACK; }
        bool escape() const { return option.cost == CastCost::ESCAPE; }
        bool alternative() const { return option.alternative(); }
        bool back_face() const { return option.face == CastFace::BACK; }
};

#endif /* CHOSEN_COST_H */
