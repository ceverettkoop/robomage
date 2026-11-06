#include "unless_payment.h"

#include "cli_output.h"
#include "components/player.h"
#include "cost_payment.h"
#include "ecs/coordinator.h"
#include "mana_system.h"
#include "queries/pending_cost.h"
#include "queries/player_resources.h"
#include "queries/players.h"
#include "resolution_frame.h"
#include "svar_eval.h"

extern Coordinator global_coordinator;

UnlessPayKind unless_pay_kind(const CostParts &cost) {
    if (cost.has(CostPart::ENERGY)) return UnlessPayKind::ENERGY;
    if (cost.has(CostPart::DISCARD)) return UnlessPayKind::DISCARD;
    return cost.has(CostPart::LIFE) ? UnlessPayKind::LIFE : UnlessPayKind::MANA;
}

size_t unless_amount(const CostParts &cost) {
    switch (unless_pay_kind(cost)) {
        case UnlessPayKind::ENERGY:
            return static_cast<size_t>(cost.energy());
        case UnlessPayKind::DISCARD:
            return static_cast<size_t>(cost.count(CostPart::DISCARD));
        case UnlessPayKind::LIFE:
            return static_cast<size_t>(cost.life());
        default:
            return cost.mana().size();
    }
}

bool pay_switched_energy(const Ability &ab) {
    const int n = static_cast<int>(evaluate_amount(ab.def->unless_energy_expr, ab.controller, 0, ab.target.get()));
    auto &pl = global_coordinator.GetComponent<Player>(get_player_entity(ab.controller));
    const bool paid = pay_energy(pl, n);  // n <= 0 is a trivially-payable no-op (returns true)
    if (n > 0) game_log("%s pays %d energy.\n", player_name(ab.controller).c_str(), n);
    return paid;
}

bool run_unless_loop(
    size_t cost, Zone::Ownership controller, std::shared_ptr<Orderer> orderer, Entity paid_for,
    Entity decision_source, FrameCtx &ctx, bool &suspended, const UnlessSubject &subject,
    UnlessPayKind kind, const ManaValue *cost_pips) {
    suspended = false;

    // The offer in progress, for the state format's pending_cost (queries/pending_cost.h); it
    // stays published while an ask is parked.
    Game::PendingDecisions::CostPayment offer;
    offer.unless = true;
    offer.paid_for = paid_for != 0 ? paid_for : decision_source;
    switch (kind) {
        case UnlessPayKind::LIFE:
            offer.life = static_cast<int>(cost);
            break;
        case UnlessPayKind::ENERGY:
            offer.energy = static_cast<int>(cost);
            break;
        case UnlessPayKind::DISCARD:
            offer.discard = static_cast<int>(cost);
            break;
        default:
            if (cost_pips && !cost_pips->empty())
                offer.remaining = *cost_pips;
            else
                for (size_t i = 0; i < cost; i++) offer.remaining.insert(GENERIC);
            break;
    }
    CostPaymentScope offer_scope(offer, &suspended);

    const UnlessTerms terms{cost, kind, cost_pips, subject};
    CostPayer who(CostPath::UNLESS, controller, decision_source, ManaPayee(paid_for, ManaPurpose::OTHER), nullptr, nullptr);
    who.unless = &terms;
    const PayStatus paid = run_cost_parts(who, ctx, orderer);
    suspended = paid == PayStatus::SUSPENDED;
    return paid == PayStatus::DECLINED;
}
