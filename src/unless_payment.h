#ifndef UNLESS_PAYMENT_H
#define UNLESS_PAYMENT_H

#include <cstddef>
#include <memory>

#include "choice_labels.h"
#include "classes/colors.h"
#include "components/cost_part.h"
#include "components/zone.h"
#include "ecs/entity.h"

class Orderer;
class FrameCtx;
struct Ability;

// What an unless-cost offer pays and what it governs (run_unless_loop's terms).
struct UnlessTerms {
    size_t amount;
    UnlessPayKind kind;
    const ManaValue *pips;
    const UnlessSubject &subject;
    const FilterSpec *sacrifice;  // what a SACRIFICE may take; null for any other kind
};

// An UnlessCost$ as run_unless_loop takes it: how it is paid, and how much of that (0 for no cost).
UnlessPayKind unless_pay_kind(const CostParts &cost);
size_t unless_amount(const CostParts &cost);

// Offer an unless-cost (CR 118.12) to `controller`: pay `cost` of `kind` — {cost} generic mana or
// the exact `cost_pips` (Chain Lightning: {R}{R}), `cost` life (Ward—Pay life, CR 702.21), discard
// `cost` card(s) from hand (Reality Smasher, CR 701.8), `cost` energy ({E}, CR 122.1c — Static
// Prison), or sacrifice a permanent matching `sacrifice`, a menu of them, with nothing asked when
// there is none (Grist) — or decline. Returns true if the payer declined or couldn't pay. `subject` names the
// governed effect and its object, which the shared choice_labels builders word into the pay and
// decline entries. The payer decides whether to pay, not the ability's controller: run_cost_parts
// seats every ask on `controller` through `ctx`, which may suspend (`suspended` set, return value
// meaningless — check it FIRST). `decision_source` (the resolving ability's source) is the
// pending-decision context of every ask.
bool run_unless_loop(size_t cost, Zone::Ownership controller, std::shared_ptr<Orderer> orderer, Entity paid_for,
                     Entity decision_source, FrameCtx &ctx, bool &suspended, const UnlessSubject &subject,
                     UnlessPayKind kind = UnlessPayKind::MANA, const ManaValue *cost_pips = nullptr,
                     const FilterSpec *sacrifice = nullptr);

// UnlessCost$ PayEnergy<SVar> with UnlessSwitched$ True: `ab`'s controller pays the amount read now
// as a single cost of resolution, an amount they chose earlier in it (Count$ChosenNumber), so
// nothing is asked. True if it is paid, and the effect then happens.
bool pay_switched_energy(const Ability &ab);

#endif /* UNLESS_PAYMENT_H */
