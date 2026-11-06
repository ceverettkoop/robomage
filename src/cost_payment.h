#ifndef COST_PAYMENT_H
#define COST_PAYMENT_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "components/cost_part.h"
#include "components/zone.h"
#include "ecs/entity.h"
#include "mana_system.h"

class FrameCtx;
class Orderer;
struct Ability;

// ── Paying a cost (CR 118, 601.2f-h, 602.2b) ─────────────────────────────────────────────────
//
// Every cost is paid by run_cost_parts: a payment path is a table of steps in the order that path
// pays (cost_payment.cpp), each step paying a part of the cost or choosing what a part moves. Any
// step may ask a choice through its FrameCtx and park; all progress lives in the path's CostRun,
// on Game, so a snapshot of a parked payment covers it and the re-entered run lands on the step
// that asked.

// The paths that pay a cost, each with its own order of steps.
enum class CostPath { ACTIVATION };

// A permanent or card chosen to pay a cost: it moves only once the mana has committed, so a payment
// that fails has moved nothing (CR 733.1) and the mana can still be taken from a permanent on its
// way out (CR 601.2g).
struct CostItem {
    Entity entity = 0;
    Zone::ZoneValue dest = Zone::GRAVEYARD;
    CostPart::Kind kind = CostPart::SACRIFICE;  // the part it pays
    // The narrative line, formatted as it is chosen (while the card is still where it was) and
    // emitted as it moves.
    std::string log;
};

// How far a payment has got and what it still owes: persisted by the flow that pays it.
struct CostRun {
    size_t step = 0;  // the next step of the path's table
    int picks = 0;    // choices made at this step
    std::vector<CostItem> items;  // chosen and not yet moved
    // The mana owed and how it is being paid.
    ManaValue mana;
    bool mana_pending = false;
    ManaPaymentRT mana_payment;
    // Snapshot of mana state taken as the payment first can activate a mana ability, restored when
    // it is reversed (mana_snap_taken).
    ManaPaymentSnapshot mana_snap;
    bool mana_snap_taken = false;
    // Life and energy owed, paid once the mana has committed.
    int life = 0;
    int energy = 0;
    bool committed = false;  // the mana is paid and the rest of the cost is being paid
};

// Who pays what along a path. Rebuilt by the flow at every entry; the flow's CostRun carries what
// persists.
struct CostPayer {
    CostPayer(CostPath path_, Zone::Ownership who, Entity paying_for, ManaPayee to, const CostParts *cost_, CostRun *run_)
        : path(path_), payer(who), source(paying_for), payee(to), cost(cost_), run(run_) {}

    CostPath path;
    Zone::Ownership payer;
    Entity source;  // the spell or the ability's source: the object {T} taps, `Other` excludes and asks each choice
    ManaPayee payee;
    const CostParts *cost;  // the cost being paid
    CostRun *run;

    // An activation: the ability activated, its copy going on the stack and the X announced.
    const Ability *ability = nullptr;
    Ability *announced = nullptr;
    size_t x = 0;
};

// Pay `who`'s cost, from where its CostRun stands. PAID: all of it, in the path's order. FAILED: it
// can't be paid, and the caller reverses the proposal. SUSPENDED: a choice is parked, and the caller
// returns until it is answered. An answered choice comes back through `ctx`.
PayStatus run_cost_parts(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer);

#endif /* COST_PAYMENT_H */
