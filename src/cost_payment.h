#ifndef COST_PAYMENT_H
#define COST_PAYMENT_H

#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "components/chosen_cost.h"
#include "components/cost_part.h"
#include "components/zone.h"
#include "ecs/entity.h"
#include "mana_system.h"

class FrameCtx;
class Orderer;
struct Ability;
struct Player;
struct UnlessTerms;

// ── Paying a cost (CR 118, 601.2f-h, 602.2b) ─────────────────────────────────────────────────
//
// Every cost is paid by run_cost_parts: a payment path is a table of steps in the order that path
// pays (cost_payment.cpp), each step paying a part of the cost or choosing what a part moves. Any
// step may ask a choice through its FrameCtx and park; all progress lives in the path's CostRun,
// on Game, so a snapshot of a parked payment covers it and the re-entered run lands on the step
// that asked. A path that can't park between its steps, whose question re-derives from the live
// state, keeps no CostRun.

// The paths that pay a cost, each with its own order of steps.
enum class CostPath {
    CAST,
    ACTIVATION,
    MANA_ABILITY,
    TRIGGER,
    IMMEDIATE_TRIGGER,
    UNLESS
};

// A permanent or card chosen to pay a cost: it moves only once the mana has committed, so a payment
// that fails has moved nothing (CR 733.1) and the mana can still be taken from a permanent on its
// way out (CR 601.2g).
struct CostItem {
        Entity entity = 0;
        Zone::ZoneValue dest = Zone::GRAVEYARD;
        CostPart::Kind kind = CostPart::SACRIFICE;  // the part it pays
        bool delve = false;                         // a delve exile (CR 702.66a): recorded in cur_game.delve_exiled
        // The narrative line, formatted as it is chosen (while the card is still where it was) and
        // emitted as it moves.
        std::string log;
};

// How far a payment has got and what it still owes: persisted by the flow that pays it.
struct CostRun {
        size_t step = 0;              // the next step of the path's table
        int picks = 0;                // choices made at this step
        std::vector<CostItem> items;  // chosen and not yet moved
        // The mana owed and how it is being paid. A cast's is determined as it is announced, paid after
        // its targets are chosen (CR 601.2c before 601.2g/h): a sacrifice-for-mana source spent first
        // could remove a spell's only legal target (Lotus Petal, Lion's Eye Diamond). With the payment
        // here the target is chosen while the source is still on the battlefield; sacrificing it for
        // mana then merely makes the target illegal, and the spell fizzles (CR 608.2b).
        ManaValue mana;
        bool mana_pending = false;
        bool delve = false;      // the spell's Delve applies to `mana`
        bool improvise = false;  // the spell's Improvise applies to `mana`
        ManaPaymentRT mana_payment;
        // The exact colors of mana spent, accumulated by the payment (pay_mana_cost's spent-sink) as
        // pips leave the pool. Empty for a free / no-mana cast.
        ManaValue spent_colors;
        // Snapshot of mana state taken as the payment first can activate a mana ability, restored when
        // it is reversed (mana_snap_taken).
        ManaPaymentSnapshot mana_snap;
        bool mana_snap_taken = false;
        // Life and energy owed, paid once the mana has committed: a cancelled payment never costs any.
        // An exile grant's LIFE or ENERGY resource cost (CR 118.9, 107.14: Amped Raptor) adds to them.
        int life = 0;
        int energy = 0;
        // Variable life X (Toxic Deluge's PayLife<X>): the value announced (CR 601.2b). Kept apart from
        // `life` only so the narrative can still say "pays N life (X = N)". -1 = no such cost.
        int life_x = -1;
        bool committed = false;  // the mana is paid and the rest of the cost is being paid
        // Delve (CR 702.66): the chosen exile count, and the pre-delve priority seat. Both delve steps
        // are seated on the caster; delve_seat_held says the seat is still to be restored.
        size_t delve_count = 0;
        bool delve_prev_priority_a = true;
        bool delve_seat_held = false;
        // Escape's ExileFromGrave<X/...> (CR 702.138): the distinct card types chosen so far.
        std::set<std::string> exiled_types;
        int discards_chosen = 0;  // Escalate's discards chosen so far (CR 702.120a)
};

// Who pays what along a path. Rebuilt by the flow at every entry; the flow's CostRun carries what
// persists.
struct CostPayer {
        CostPayer(
            CostPath path_, Zone::Ownership who, Entity paying_for, ManaPayee to, const CostParts *cost_, CostRun *run_)
            : path(path_), payer(who), source(paying_for), payee(to), cost(cost_), run(run_) {}

        CostPath path;
        Zone::Ownership payer;
        Entity source;  // the spell or the ability's source: the object {T} taps, `Other` excludes and asks each choice
        ManaPayee payee;
        const CostParts *cost;  // the cost being paid
        CostRun *run;           // null on a path that keeps no progress

        // An activation: the ability activated, its copy going on the stack and the X announced.
        const Ability *ability = nullptr;
        Ability *announced = nullptr;
        size_t x = 0;

        // A cast: what it pays besides its mana, each null when this cast doesn't. `cost` is unused.
        const CostParts *alternative = nullptr;  // the alternative cost it was cast for
        const CostParts *additional = nullptr;   // a spell's additional sacrifice
        const CostParts *escalate = nullptr;     // the Escalate cost, paid `escalate_times` times
        int escalate_times = 0;
        const CostParts *flashback = nullptr;
        const CostParts *escape = nullptr;
        bool spell_has_delve = false;
        ChosenCost *chosen = nullptr;  // what the cast has committed to, recording what is paid

        // A mana ability paying through activate_mana_source: the pool and the player it pays from, the
        // real ones or a simulated payment's copies (commit false: no tap, no sacrifice, no narrative).
        ManaValue *pool = nullptr;
        Player *working = nullptr;
        bool commit = true;
        int life_reserve = 0;

        bool mandatory = false;     // a triggered ability's cost its controller doesn't choose (CR 603.8)
        bool source_gone = false;   // a triggered ability's source is no longer the object that triggered
        const UnlessTerms *unless = nullptr;  // an unless-cost offered (CR 118.12)
};

// A triggered ability's cost, paid as it resolves (CR 603.2, 603.5). Its CostRun is the ability's resolving level's, so a parked question resumes there.
// PAID: go on to the effect. DECLINED: the ability does nothing more. SUSPENDED: a choice is parked.
PayStatus pay_trigger_cost(const Ability &ab, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer);

// Which part of a cast's cost its payment has reached, for the state format's pending_cost.
enum class CastCostStage {
    ALTERNATIVE,
    ADDITIONAL,
    MANA
};
CastCostStage cast_cost_stage(const CostRun &run);

// Pay `who`'s cost, from where its CostRun stands. PAID: all of it, in the path's order. DECLINED: an
// optional cost not paid, by choice or because it couldn't be. FAILED: it can't be paid, and the
// caller reverses the proposal. SUSPENDED: a choice is parked, and the caller returns until it is
// answered. An answered choice comes back through `ctx`.
PayStatus run_cost_parts(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer);

#endif /* COST_PAYMENT_H */
