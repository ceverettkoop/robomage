#ifndef QUERIES_PENDING_COST_H
#define QUERIES_PENDING_COST_H

#include <vector>

#include "../classes/colors.h"
#include "../classes/game.h"
#include "../components/carddata.h"
#include "../ecs/entity.h"
#include "../state_format/schema.h"

// ── The cost being paid at the current decision (docs/state_format.md register 33) ───────────
//
// What a cost-payment decision is paying for and what is still owed, read from the flow that is
// paying it: the cast state machine (Game::PendingCast, CR 601.2b-h), the activation state machine
// (Game::PendingActivation, CR 602.2b), or a payment that keeps no state machine of its own (the
// interactive mana payment loop, an unless-cost offer), which publishes itself through
// CostPaymentScope.

struct PendingCost {
        Entity paid_for = 0;  // the spell or ability being paid for (an unless offer: the asking source)
        rmstate::CostPart part = rmstate::CostPart::MANA;
        ManaValue remaining;            // plain pips still owed (WUBRGC and generic)
        int x_pips = 0;                 // {X} pips whose value is still to be announced
        std::vector<HybridPip> hybrid;  // hybrid pips not yet resolved to a color
        std::vector<Colors> phyrexian;  // Phyrexian pips not yet resolved to mana or life
        int life = 0;                   // life still to pay
        int energy = 0;                 // energy still to pay
        int discard = 0;                // cards still to discard
        int paid = 0;                   // mana already paid toward it
};

// The cost being paid right now, if a payment is in progress at this decision: false outside every
// payment (and while a cast or activation is choosing modes or targets, which are not costs).
bool pending_cost(PendingCost &out);

// Publishes a payment that keeps no state machine of its own as Game::pending.cost_payment for
// the span of the scope, restoring what was there before when it ends. A payment that parks its
// decision (a suspended unless offer, `*parked` set as the scope ends) leaves the state in place
// for the decision the main loop emits, and the resumed payment's scope takes it over.
class CostPaymentScope {
    public:
        CostPaymentScope(const Game::PendingDecisions::CostPayment &state, const bool *parked = nullptr);
        ~CostPaymentScope();
        CostPaymentScope(const CostPaymentScope &) = delete;
        CostPaymentScope &operator=(const CostPaymentScope &) = delete;
        // The mana still owed changed (a source was tapped, a card delved).
        void set_remaining(const ManaValue &remaining, int paid);

    private:
        Game::PendingDecisions::CostPayment prev_;
        const bool *parked_;
};

#endif /* QUERIES_PENDING_COST_H */
