#include "pending_cost.h"

#include <algorithm>

#include "../components/ability.h"
#include "../ecs/coordinator.h"
#include "../mana_system.h"

using Cast = Game::PendingCast;
using Activation = Game::PendingActivation;
using StatePart = rmstate::CostPart;

static bool cast_cost(const Cast &pc, PendingCost &out);
static bool activation_cost(const Activation &pa, PendingCost &out);
static bool cast_step_is_payment(Cast::Step step);
static StatePart cast_step_part(const Cast &pc);
static const CardData *cast_face(const Cast &pc);

// The steps of the cast flow that decide or pay a cost (CR 601.2b's cost announcements and
// 601.2f-h's payment); the mode/target announcement steps and the copy retargeting are not.
static bool cast_step_is_payment(Cast::Step step) {
    switch (step) {
        case Cast::MODES_TARGETS:
        case Cast::AURA_TARGET:
        case Cast::ANNOUNCE:
        case Cast::COPY_TARGETS:
        case Cast::FINISH:
            return false;
        default:
            return true;
    }
}

static StatePart cast_step_part(const Cast &pc) {
    switch (pc.step) {
        case Cast::PAY_COSTS:
            switch (cast_cost_stage(pc.pay)) {
                case CastCostStage::ALTERNATIVE:
                    return StatePart::ALTERNATIVE;
                case CastCostStage::ADDITIONAL:
                    return StatePart::ADDITIONAL;
                default:
                    return StatePart::MANA;
            }
        case Cast::ALT_HYBRID:
            return StatePart::ALTERNATIVE;
        case Cast::KICKER:
        case Cast::REPLICATE:
        case Cast::ADDITIONAL:
        case Cast::GIFT:
            return StatePart::ADDITIONAL;
        case Cast::CHOOSE_X:
        case Cast::LIFE_X:
            return StatePart::X;
        default:
            return StatePart::MANA;
    }
}

// The face being cast (a modal DFC's back face or a split card's second half when cast as such).
static const CardData *cast_face(const Cast &pc) {
    if (pc.spell_entity == 0 || !global_coordinator.entity_has_component<CardData>(pc.spell_entity)) return nullptr;
    const CardData &cd = global_coordinator.GetComponent<CardData>(pc.spell_entity);
    return pc.chosen.back_face ? cd.backside.get() : &cd;
}

static bool cast_cost(const Cast &pc, PendingCost &out) {
    if (!pc.active || !cast_step_is_payment(pc.step)) return false;
    out.paid_for = pc.spell_entity;
    out.part = cast_step_part(pc);
    out.remaining = pc.pay.mana_pending ? pc.pay.mana : pc.cost_to_pay;
    out.life = pc.pay.life + std::max(0, pc.pay.life_x);
    out.energy = pc.pay.energy;
    // The discards still to pick: Escalate's (CR 702.120a), owed once the modes are announced, and
    // those of the form of an additional cost announced.
    if (const CardData *face = cast_face(pc)) {
        int discards = face->escalate_cost.count(CostPart::DISCARD) * pc.chosen.escalate_count;
        if (pc.chosen.additional_choice >= 0)
            discards += face->additional_choices[static_cast<size_t>(pc.chosen.additional_choice)].count(CostPart::DISCARD);
        out.discard = std::max(0, discards - pc.pay.discards_chosen);
    }
    // The regular cost branch folds X, hybrid and Phyrexian pips into cost_to_pay one step at a
    // time (KICKER through PHYREXIAN_PIP; the alternative branches skip them): the pips of the
    // steps not yet reached are still owed as printed.
    const CardData *face = cast_face(pc);
    const bool regular = pc.step >= Cast::KICKER && pc.step <= Cast::PHYREXIAN_PIP;
    if (face && regular) {
        if (pc.step <= Cast::CHOOSE_X && face->has_x_cost) out.x_pips = face->x_pip_count;
        if (pc.step <= Cast::HYBRID) out.hybrid = face->hybrid_mana;
        for (size_t i = pc.phyrexian_idx; i < face->phyrexian_mana.size(); ++i)
            out.phyrexian.push_back(face->phyrexian_mana[i]);
    }
    if (face && pc.step == Cast::ALT_HYBRID) {
        const std::vector<HybridPip> &pips = face->alt_cost.cost.hybrid();
        out.hybrid.assign(pips.begin() + static_cast<std::ptrdiff_t>(pc.hybrid_idx), pips.end());
    }
    return true;
}

static bool activation_cost(const Activation &pa, PendingCost &out) {
    if (!pa.active) return false;
    if (pa.step == Activation::ZONE_TARGET || pa.step == Activation::TARGET || pa.step == Activation::FINISH)
        return false;
    out.paid_for = pa.stack_entity != 0 ? pa.stack_entity : pa.source_entity;
    out.part =
        (pa.step == Activation::X_LADDER || pa.step == Activation::LOYALTY_X) ? StatePart::X : StatePart::ACTIVATION;
    const AbilityDef &def = *pa.ability.def;
    const Zone::Ownership controller = pa.activator_is_a ? Zone::PLAYER_A : Zone::PLAYER_B;
    if (pa.pay.mana_announced) {
        out.remaining = pa.pay.mana;
    } else {
        out.remaining = effective_activation_mana_cost(pa.ability, controller, nullptr);
        if (pa.step == Activation::X_LADDER && def.cost.has(CostPart::X_MANA))
            out.x_pips = std::max(1, def.cost.x_pips());
        else
            for (size_t i = 0; i < pa.x_activation * static_cast<size_t>(std::max(1, def.cost.x_pips())); ++i)
                out.remaining.insert(GENERIC);
    }
    if (pa.step <= Activation::HYBRID)
        out.hybrid.assign(def.cost.hybrid().begin() + static_cast<std::ptrdiff_t>(pa.hybrid_idx), def.cost.hybrid().end());
    out.life = def.cost.life();
    out.energy = def.cost.energy();
    return true;
}

bool pending_cost(PendingCost &out) {
    out = PendingCost();
    const Game::PendingDecisions &pd = cur_game.pending;
    if (pd.cost_payment.active) {
        const Game::PendingDecisions::CostPayment &cp = pd.cost_payment;
        out.paid_for = cp.paid_for;
        out.part = cp.unless ? StatePart::UNLESS : cp.trigger ? StatePart::TRIGGER : StatePart::MANA;
        out.remaining = cp.remaining;
        out.life = cp.life;
        out.energy = cp.energy;
        out.discard = cp.discard;
        out.paid = cp.paid;
        return true;
    }
    return cast_cost(pd.cast, out) || activation_cost(pd.activation, out);
}

CostPaymentScope::CostPaymentScope(const Game::PendingDecisions::CostPayment &state, const bool *parked)
    : prev_(cur_game.pending.cost_payment), parked_(parked) {
    // A resumed payment takes over the state its parked decision kept.
    if (prev_.active && prev_.paid_for == state.paid_for && prev_.unless == state.unless)
        prev_ = Game::PendingDecisions::CostPayment();
    cur_game.pending.cost_payment = state;
    cur_game.pending.cost_payment.active = true;
}

CostPaymentScope::~CostPaymentScope() {
    if (!parked_ || !*parked_) cur_game.pending.cost_payment = prev_;
}

void CostPaymentScope::set_remaining(const ManaValue &remaining, int paid) {
    cur_game.pending.cost_payment.remaining = remaining;
    cur_game.pending.cost_payment.paid = paid;
}
