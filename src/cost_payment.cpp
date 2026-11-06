#include "cost_payment.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "choice_labels.h"
#include "classes/action.h"
#include "classes/game.h"
#include "cli_output.h"
#include "components/ability.h"
#include "components/carddata.h"
#include "components/creature.h"
#include "components/permanent.h"
#include "components/player.h"
#include "coverage.h"
#include "ecs/coordinator.h"
#include "game_driver.h"
#include "queries/battlefield.h"
#include "queries/characteristics.h"
#include "queries/costs.h"
#include "queries/counters.h"
#include "queries/filters.h"
#include "queries/pending_cost.h"
#include "queries/player_resources.h"
#include "queries/players.h"
#include "resolution_frame.h"
#include "systems/orderer.h"
#include "unless_payment.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace {

// The cost a pick reads, for a path that pays more than one.
enum class CostSlot {
    COST,
    ALTERNATIVE,
    ADDITIONAL,
    ADDITIONAL_CHOICE,
    ESCALATE,
    FLASHBACK,
    ESCAPE
};

// How one choice of what a part moves is asked and recorded.
struct PickRule {
        CostSlot slot;
        CostPart::Kind kind;
        const char *label;  // the menu row: label + name + label_tail
        const char *label_tail;
        ActionCategory category;
        Zone::ZoneValue dest;
        const char *log;  // the narrative line: player + log + name + log_tail
        const char *log_tail;
        // How many it takes. ONE_IF_ANY: a single pick when the filter names any and a candidate exists,
        // whatever the part's count. BY_COUNT: the part's count. BY_TYPES: until the cards chosen cover
        // its distinct card types.
        enum Amount {
            ONE_IF_ANY,
            BY_COUNT,
            BY_TYPES
        } amount;
        bool takes_forced;  // nothing is asked when every candidate must be taken
};

// One step of a payment path.
struct CostStep {
        enum Op {
            PICK,            // choose what `pick` moves; the move waits for the mana
            DELVE_COUNT,     // how many cards Delve exiles
            DELVE_PICK,      // which ones
            SNAPSHOT,        // the snapshot a reversed payment restores
            TAP,             // {T}: the source taps
            OWE_ACTIVATION,  // what an activation owes
            FLOAT_LEAVING,   // CR 601.2g: mana from a permanent about to leave to pay the cost
            RECORD_MANA_SPENT,
            PAY_MANA,
            CHECK_RESOURCES,  // the life and energy can still be paid
            COUNT_TAP,
            LOYALTY,
            LIFE,
            LIFE_X,
            ENERGY,
            SACRIFICE_SELF,
            EXILE_SELF,
            MOVE_ITEMS,  // what the picks chose moves
            DISCARD_SELF,
            DISCARD_HAND,
            // A mana ability paying through activate_mana_source, from its working pool and player.
            MANA_ABILITY_GATE,  // the life it costs and loses can be paid
            MANA_ABILITY_POOL,  // its activation mana, from the pool alone
            MANA_ABILITY_SACRIFICE,
            MANA_ABILITY_LIFE,
            TRIGGER_SACRIFICE,  // a triggered ability's Sac<1/CARDNAME>, paid as it resolves
            TRIGGER_OFFER,      // a triggered ability's "you may" cost, offered if it can be paid
            OWE_TRIGGER,        // what the triggered ability owes
            TRIGGER_MANA,       // its mana; a payment given up declines the offer
            TRIGGER_ENERGY,     // an ImmediateTrigger's optional PayEnergy<N>
            UNLESS_OFFER,       // an unless-cost, offered to its payer
        };
        Op op;
        const PickRule *pick = nullptr;
};

struct CostSteps {
        const CostStep *rows;
        size_t size;
};

}  // namespace

template <size_t N>
static constexpr CostSteps table(const CostStep (&rows)[N]);
static CostSteps path_steps(CostPath path);
static PayStatus pay_step(
    const CostStep &step, const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer);
static PayStatus pick_items(
    const PickRule &rule, const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer);
static std::vector<Entity> pick_pool(
    const PickRule &rule, const CostPayer &who, const std::shared_ptr<Orderer> &orderer);
static const CostParts *pick_cost(const PickRule &rule, const CostPayer &who);
static int picks_owed(const PickRule &rule, const CostParts &cost, const CostPayer &who);
static std::string item_name(const PickRule &rule, Entity e);
static bool chosen_as_cost(const CostRun &run, Entity e);
static void record_pick(const PickRule &rule, Entity chosen, const CostPayer &who);
static PayStatus choose_delve_count(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer);
static PayStatus choose_delve_cards(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer);
static void choose_delve_exile(const CostPayer &who, Entity e);
static void owe_activation(const CostPayer &who, const std::shared_ptr<Orderer> &orderer);
static PayStatus pay_mana(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer);
static const char *path_form(CostPath path);
static bool resources_payable(const CostPayer &who);
static void pay_loyalty(const CostPayer &who);
static void move_items(const CostPayer &who, const std::shared_ptr<Orderer> &orderer);
static void pay_life_x(const CostPayer &who);
static void sacrifice_self(const CostPayer &who, const std::shared_ptr<Orderer> &orderer);
static void exile_self(const CostPayer &who, const std::shared_ptr<Orderer> &orderer);
static void discard_self(const CostPayer &who, const std::shared_ptr<Orderer> &orderer);
static void discard_hand(const CostPayer &who, const std::shared_ptr<Orderer> &orderer);
static PayStatus pay_from_working_pool(const CostPayer &who);
static PayStatus pay_trigger_sacrifice(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer);
static PayStatus offer_trigger_cost(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer);
static std::string cost_offer_text(const CostParts &cost);
static PayStatus pay_trigger_mana(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer);
static PayStatus offer_trigger_energy(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer);
static PayStatus offer_unless_discard(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer);
static PayStatus offer_unless_resource(const CostPayer &who, FrameCtx &ctx);
static PayStatus offer_unless_mana(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer);

// ── What a pick chooses ──────────────────────────────────────────────────────────────────────
// Type-based sacrifice cost (Cycling "Sac a land", Knight of the Reliquary).
static const PickRule kActivationSacrifice = {CostSlot::COST, CostPart::SACRIFICE, "Sacrifice ", "",
    ActionCategory::SACRIFICE_PERMANENT, Zone::GRAVEYARD, " sacrifices ", "", PickRule::ONE_IF_ANY, false};
// Return-to-hand cost (Scryb Ranger: return a Forest to hand).
static const PickRule kActivationReturn = {CostSlot::COST, CostPart::RETURN, "Return ", " to hand",
    ActionCategory::RETURN_PERMANENT, Zone::HAND, " returns ", " to hand", PickRule::ONE_IF_ANY, false};
// Alt-cost pitch (Force of Will: "exile a blue card from your hand"). The alternative cost's picks
// are skipped entirely on a normal cast of the same card (CR 601.2b): Daze cast for {1}{U} does not
// also return an Island.
static const PickRule kAltPitch = {CostSlot::ALTERNATIVE, CostPart::EXILE_FROM_HAND, "Exile ", "",
    ActionCategory::PAYING_COSTS, Zone::EXILE, " exiles ", "", PickRule::BY_COUNT, false};
// Alt-cost return-to-hand (Daze: "return an Island you control to its owner's hand").
static const PickRule kAltReturn = {CostSlot::ALTERNATIVE, CostPart::RETURN, "Return ", "",
    ActionCategory::RETURN_PERMANENT, Zone::HAND, " returns ", " to hand", PickRule::BY_COUNT, false};
// Alt-cost sacrifice (Fireblast: "sacrifice two Mountains rather than pay this spell's mana cost",
// CR 118.9).
static const PickRule kAltSacrifice = {CostSlot::ALTERNATIVE, CostPart::SACRIFICE, "Sacrifice ", "",
    ActionCategory::SACRIFICE_PERMANENT, Zone::GRAVEYARD, " sacrifices ", "", PickRule::BY_COUNT, false};
// Additional sacrifice on the spell itself (CR 601.2f / 118.x): Natural Order — "As an additional
// cost to cast this spell, sacrifice a green creature."
static const PickRule kAdditionalSacrifice = {CostSlot::ADDITIONAL, CostPart::SACRIFICE, "Sacrifice ", "",
    ActionCategory::SACRIFICE_PERMANENT, Zone::GRAVEYARD, " sacrifices ", "", PickRule::ONE_IF_ANY, false};
// The discard of an additional cost announced in that form (Bitter Triumph, CR 601.2b, 601.2h).
static const PickRule kAdditionalDiscard = {CostSlot::ADDITIONAL_CHOICE, CostPart::DISCARD, "Discard ", "",
    ActionCategory::DISCARD, Zone::GRAVEYARD, " discards ", "", PickRule::BY_COUNT, false};
// Escalate's discards (CR 702.120a / 601.2h): the cost's N cards for each mode beyond the first. The
// mode menu offered each extra mode only while enough cards were left.
static const PickRule kEscalateDiscard = {CostSlot::ESCALATE, CostPart::DISCARD, "Discard ", "",
    ActionCategory::DISCARD, Zone::GRAVEYARD, " discards ", " (escalate)", PickRule::BY_COUNT, true};
// Flashback's sacrifice (CR 702.34, Cabal Therapy).
static const PickRule kFlashbackSacrifice = {CostSlot::FLASHBACK, CostPart::SACRIFICE, "Sacrifice ", "",
    ActionCategory::SACRIFICE_PERMANENT, Zone::GRAVEYARD, " sacrifices ", "", PickRule::ONE_IF_ANY, false};
// Escape's ExileFromGrave (CR 702.138 / 601.2f), mandatory until met: OTHER cards of the graveyard
// until the set collectively has at least min_types distinct card types (CR 205.2), or exactly
// `count` of them (Uro: "Exile five other cards from your graveyard").
static const PickRule kEscapeExileTypes = {CostSlot::ESCAPE, CostPart::EXILE_FROM_GRAVEYARD, "Exile ",
    " from graveyard", ActionCategory::EXILE_FROM_YARD, Zone::EXILE, " exiles ", " from their graveyard",
    PickRule::BY_TYPES, false};
static const PickRule kEscapeExileCount = {CostSlot::ESCAPE, CostPart::EXILE_FROM_GRAVEYARD, "Exile ",
    " from graveyard", ActionCategory::EXILE_FROM_YARD, Zone::EXILE, " exiles ", " from their graveyard",
    PickRule::BY_COUNT, false};
// A triggered ability's discard ("you may discard a card. If you do, ...": Formidable Speaker).
static const PickRule kTriggerDiscard = {CostSlot::COST, CostPart::DISCARD, "Discard ", "", ActionCategory::DISCARD,
    Zone::GRAVEYARD, " discards ", "", PickRule::BY_COUNT, false};

// ── The paths ────────────────────────────────────────────────────────────────────────────────
// CR 601.2f-h, after the targets: first every non-mana cost item is chosen, then the mana is paid,
// then the life and energy and the moves. Nothing irreversible happens before the mana commits, so a
// proposal reversed then (CR 601.5 / 733.1) is undone by rewind_cast.
static const CostStep kCastSteps[] = {
    {CostStep::PICK, &kAltPitch},
    {CostStep::PICK, &kAltReturn},
    {CostStep::PICK, &kAltSacrifice},
    {CostStep::PICK, &kAdditionalSacrifice},
    {CostStep::PICK, &kAdditionalDiscard},
    {CostStep::PICK, &kEscalateDiscard},
    {CostStep::DELVE_COUNT},
    {CostStep::DELVE_PICK},
    {CostStep::PICK, &kFlashbackSacrifice},
    {CostStep::PICK, &kEscapeExileTypes},
    {CostStep::PICK, &kEscapeExileCount},
    {CostStep::SNAPSHOT},
    {CostStep::FLOAT_LEAVING},
    {CostStep::PAY_MANA},
    {CostStep::RECORD_MANA_SPENT},
    {CostStep::CHECK_RESOURCES},
    {CostStep::LIFE},
    {CostStep::LIFE_X},
    {CostStep::ENERGY},
    {CostStep::MOVE_ITEMS},
};

// CR 602.2b: choose what the cost moves, then pay it (CR 601.2g-h). The tap comes first, recorded by
// the snapshot, so a reversed activation untaps; the life and energy after the mana, which a painful
// source can have made unpayable.
static const CostStep kActivationSteps[] = {
    {CostStep::PICK, &kActivationSacrifice},
    {CostStep::PICK, &kActivationReturn},
    {CostStep::SNAPSHOT},
    {CostStep::TAP},
    {CostStep::OWE_ACTIVATION},
    {CostStep::FLOAT_LEAVING},
    {CostStep::PAY_MANA},
    {CostStep::CHECK_RESOURCES},
    {CostStep::COUNT_TAP},
    {CostStep::LOYALTY},
    {CostStep::LIFE},
    {CostStep::ENERGY},
    {CostStep::SACRIFICE_SELF},
    {CostStep::EXILE_SELF},
    {CostStep::MOVE_ITEMS},
    {CostStep::DISCARD_SELF},
    {CostStep::DISCARD_HAND},
};

// A mana ability activated while paying for something else, off the stack (CR 605.3): a refusal
// happens before anything else, so it cancels cleanly; the auto-payer pre-covers its activation mana
// by tapping other sources into the pool, and the interactive payer and the unless offer rely on
// the refusal.
static const CostStep kManaAbilitySteps[] = {
    {CostStep::MANA_ABILITY_GATE},
    {CostStep::MANA_ABILITY_POOL},
    {CostStep::TAP},
    {CostStep::MANA_ABILITY_SACRIFICE},
    {CostStep::EXILE_SELF},
    {CostStep::MANA_ABILITY_LIFE},
};

// A triggered ability pays its cost as it resolves (CR 603.2): what its controller may decline is
// offered, chosen, paid for in mana and only then moves, as on an activation.
static const CostStep kTriggerSteps[] = {
    {CostStep::TRIGGER_SACRIFICE},
    {CostStep::TRIGGER_OFFER},
    {CostStep::PICK, &kTriggerDiscard},
    {CostStep::OWE_TRIGGER},
    {CostStep::SNAPSHOT},
    {CostStep::TRIGGER_MANA},
    {CostStep::MOVE_ITEMS},
};
static const CostStep kImmediateTriggerSteps[] = {{CostStep::TRIGGER_ENERGY}};
static const CostStep kUnlessSteps[] = {{CostStep::UNLESS_OFFER}};

template <size_t N>
static constexpr CostSteps table(const CostStep (&rows)[N]) {
    return {rows, N};
}

static CostSteps path_steps(CostPath path) {
    switch (path) {
        case CostPath::CAST:
            return table(kCastSteps);
        case CostPath::MANA_ABILITY:
            return table(kManaAbilitySteps);
        case CostPath::TRIGGER:
            return table(kTriggerSteps);
        case CostPath::IMMEDIATE_TRIGGER:
            return table(kImmediateTriggerSteps);
        case CostPath::UNLESS:
            return table(kUnlessSteps);
        case CostPath::ACTIVATION:
            break;
    }
    return table(kActivationSteps);
}

CastCostStage cast_cost_stage(const CostRun &run) {
    if (run.step >= table(kCastSteps).size) return CastCostStage::MANA;
    const CostStep &step = kCastSteps[run.step];
    if (step.op == CostStep::PICK)
        return step.pick->slot == CostSlot::ALTERNATIVE ? CastCostStage::ALTERNATIVE : CastCostStage::ADDITIONAL;
    return step.op == CostStep::DELVE_COUNT || step.op == CostStep::DELVE_PICK ? CastCostStage::ADDITIONAL
                                                                               : CastCostStage::MANA;
}

PayStatus announce_hybrid_pips(const HybridAnnouncement &a, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    const std::vector<HybridPip> &pips = *a.pips;
    if (pips.empty()) return PayStatus::PAID;
    if (*a.next == 0) coverage_cost(CostPart::MANA, "hybrid");
    if (auto_mana && !a.colors_matter) {
        ManaValue resolved;
        if (resolve_hybrid_cost(a.payer, *a.owed, pips, a.payee, orderer, a.delve, a.tap_to_pay, &resolved, 0, a.exclude)) {
            *a.owed = resolved;
            return PayStatus::PAID;
        }
        // No payable assignment: add the colored option (else the generic alternative) so payment
        // fails cleanly through the normal path.
        for (const HybridPip &pip : pips) {
            if (!pip.colors.empty()) a.owed->insert(pip.colors.front());
            else for (int i = 0; i < pip.generic_alt; i++) a.owed->insert(GENERIC);
        }
        return PayStatus::PAID;
    }
    for (; *a.next < pips.size(); ++*a.next) {
        const HybridPip &pip = pips[*a.next];
        const std::vector<HybridPip> later(pips.begin() + static_cast<std::ptrdiff_t>(*a.next) + 1, pips.end());
        // What each row adds to `owed`, offered under auto-mana only if the rest can be paid with it.
        std::vector<LegalAction> menu;
        std::vector<ManaValue> adds;
        auto offer = [&](const std::string &label, const ManaValue &add) {
            ManaValue with_it = *a.owed;
            with_it.insert(add.begin(), add.end());
            if (auto_mana && !resolve_hybrid_cost(a.payer, with_it, later, a.payee, orderer, a.delve, a.tap_to_pay,
                                                  nullptr, 0, a.exclude))
                return;
            LegalAction row(PASS_PRIORITY, label);
            row.category = ActionCategory::PAYING_COSTS;
            menu.push_back(row);
            adds.push_back(add);
        };
        for (Colors c : pip.colors) offer(std::string("Pay {") + mana_symbol_str(c) + "}", ManaValue{c});
        if (pip.generic_alt > 0) {
            ManaValue generic;
            for (int i = 0; i < pip.generic_alt; i++) generic.insert(GENERIC);
            offer("Pay {" + std::to_string(pip.generic_alt) + "} generic", generic);
        }
        if (menu.empty()) return PayStatus::FAILED;
        const int choice = ctx.ask(std::move(menu), a.payer, a.source, AskKind::MID_PAYMENT);
        if (choice < 0 && decision_suspended()) return PayStatus::SUSPENDED;
        const ManaValue &add = adds[static_cast<size_t>(choice)];
        a.owed->insert(add.begin(), add.end());
    }
    return PayStatus::PAID;
}

bool trigger_cost_offered(const CostParts &cost) { return cost.has(CostPart::MANA) || cost.has(CostPart::DISCARD); }

PayStatus pay_trigger_cost(const Ability &ab, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    CostRun local;
    CostRun &run = ctx.can_suspend() ? ctx.rt<CostRun>() : local;
    const Entity named = ab.source.lki_entity();
    CostPayer who(CostPath::TRIGGER, ab.controller, named, ManaPayee(named, ManaPurpose::OTHER), &ab.def->cost, &run);
    who.mandatory = ab.def->mandatory;
    who.source_gone = ab.source.get() == 0;
    // What is owed, for the state format's pending_cost, while the offer is open.
    Game::PendingDecisions::CostPayment offer;
    offer.trigger = true;
    offer.paid_for = named;
    offer.remaining = ab.def->cost.mana();
    offer.discard = ab.def->cost.count(CostPart::DISCARD) - run.discards_chosen;
    bool parked = false;
    std::optional<CostPaymentScope> scope;
    if (trigger_cost_offered(ab.def->cost)) scope.emplace(offer, &parked);
    const PayStatus paid = run_cost_parts(who, ctx, orderer);
    parked = paid == PayStatus::SUSPENDED;
    return paid;
}

PayStatus run_cost_parts(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    const CostSteps steps = path_steps(who.path);
    if (!who.run) {
        for (size_t i = 0; i < steps.size; ++i) {
            const PayStatus status = pay_step(steps.rows[i], who, ctx, orderer);
            if (status != PayStatus::PAID) return status;
        }
        return PayStatus::PAID;
    }
    CostRun &run = *who.run;
    for (; run.step < steps.size; ++run.step, run.picks = 0) {
        const PayStatus status = pay_step(steps.rows[run.step], who, ctx, orderer);
        if (status != PayStatus::PAID) return status;
    }
    return PayStatus::PAID;
}

static PayStatus pay_step(
    const CostStep &step, const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    switch (step.op) {
        case CostStep::PICK:
            return pick_items(*step.pick, who, ctx, orderer);
        case CostStep::DELVE_COUNT:
            return choose_delve_count(who, ctx, orderer);
        case CostStep::DELVE_PICK:
            return choose_delve_cards(who, ctx, orderer);
        case CostStep::SNAPSHOT:
            // The first step that can activate a mana ability (CR 733.1).
            who.run->mana_snap = snapshot_mana_state(who.payer, orderer);
            who.run->mana_snap_taken = true;
            break;
        case CostStep::TAP:
            if (who.commit && who.cost->taps() && global_coordinator.entity_has_component<Permanent>(who.source))
                global_coordinator.GetComponent<Permanent>(who.source).is_tapped = true;
            break;
        case CostStep::OWE_ACTIVATION:
            owe_activation(who, orderer);
            break;
        case CostStep::FLOAT_LEAVING:
            if (who.run->mana_pending)
                for (const CostItem &item : who.run->items)
                    float_mana_before_cost_removal(item.entity, who.payer, orderer, who.run->mana, who.payee);
            break;
        case CostStep::RECORD_MANA_SPENT:
            // What left the pool (CR 106, 601.2g), for a ValidSA$ Spell.ManaSpent trigger to read at
            // cast time: a pip paid by a delve exile or a tap spends no mana.
            who.chosen->mana_spent = static_cast<int>(who.run->spent_colors.size());
            break;
        case CostStep::PAY_MANA:
            return pay_mana(who, ctx, orderer);
        case CostStep::CHECK_RESOURCES:
            if (!resources_payable(who)) return PayStatus::FAILED;
            who.run->committed = true;
            break;
        case CostStep::COUNT_TAP:
            if (who.cost->taps()) coverage_cost(CostPart::TAP);
            break;
        case CostStep::LOYALTY:
            pay_loyalty(who);
            break;
        case CostStep::LIFE:
            if (who.run->life > 0) {
                pay_life(global_coordinator.GetComponent<Player>(get_player_entity(who.payer)), who.run->life);
                game_log("%s pays %d life\n", player_name(who.payer).c_str(), who.run->life);
                coverage_cost(CostPart::LIFE);
            }
            break;
        case CostStep::LIFE_X:
            pay_life_x(who);
            break;
        case CostStep::ENERGY:
            // PayEnergy<N>, CR 122.1c.
            if (who.run->energy > 0) {
                pay_energy(global_coordinator.GetComponent<Player>(get_player_entity(who.payer)), who.run->energy);
                game_log("%s pays %d energy\n", player_name(who.payer).c_str(), who.run->energy);
                coverage_cost(CostPart::ENERGY);
            }
            break;
        case CostStep::SACRIFICE_SELF:
            sacrifice_self(who, orderer);
            break;
        case CostStep::EXILE_SELF:
            exile_self(who, orderer);
            break;
        case CostStep::MOVE_ITEMS:
            move_items(who, orderer);
            break;
        case CostStep::DISCARD_SELF:
            discard_self(who, orderer);
            break;
        case CostStep::DISCARD_HAND:
            discard_hand(who, orderer);
            break;
        case CostStep::MANA_ABILITY_GATE:
            // A life cost the player can't pay (CR 119.4) refuses the activation before any effect.
            if (!mana_ability_life_payable(*who.working, who.payer, *who.ability, who.life_reserve))
                return PayStatus::FAILED;
            break;
        case CostStep::MANA_ABILITY_POOL:
            return pay_from_working_pool(who);
        case CostStep::MANA_ABILITY_SACRIFICE:
            if (who.commit && who.cost->self(CostPart::SACRIFICE)) {
                game_log("%s sacrifices %s\n", player_name(who.payer).c_str(),
                    global_coordinator.GetComponent<Permanent>(who.source).name.c_str());
                orderer->add_to_zone(false, who.source, Zone::GRAVEYARD);
            }
            break;
        case CostStep::MANA_ABILITY_LIFE:
            // Paid on the working player in both modes: a simulated payment (auto_pay_mana_attempt)
            // pays from a copy, so a second life-costing source sees the life already spent.
            if (who.cost->life() > 0) {
                pay_life(*who.working, who.cost->life());
                if (who.commit) game_log("%s pays %d life\n", player_name(who.payer).c_str(), who.cost->life());
            }
            break;
        case CostStep::TRIGGER_SACRIFICE:
            return pay_trigger_sacrifice(who, ctx, orderer);
        case CostStep::TRIGGER_OFFER:
            return offer_trigger_cost(who, ctx, orderer);
        case CostStep::OWE_TRIGGER:
            who.run->mana = who.cost->mana();
            who.run->mana_pending = !who.run->mana.empty();
            who.run->tap_to_pay = who.cost->tap_to_pay();
            break;
        case CostStep::TRIGGER_MANA:
            return pay_trigger_mana(who, ctx, orderer);
        case CostStep::TRIGGER_ENERGY:
            return offer_trigger_energy(who, ctx, orderer);
        case CostStep::UNLESS_OFFER:
            switch (who.unless->kind) {
                case UnlessPayKind::DISCARD:
                    return offer_unless_discard(who, ctx, orderer);
                case UnlessPayKind::LIFE:
                case UnlessPayKind::ENERGY:
                    return offer_unless_resource(who, ctx);
                case UnlessPayKind::MANA:
                    return offer_unless_mana(who, ctx, orderer);
            }
            break;
    }
    return PayStatus::PAID;
}

// Choose what `rule`'s part moves, one pick at a time, the menu re-derived from the live zones at
// each pass: a pick stays where it is until the mana has committed.
static PayStatus pick_items(
    const PickRule &rule, const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    CostRun &run = *who.run;
    const CostParts *cost = pick_cost(rule, who);
    if (!cost) return PayStatus::PAID;
    for (int left = picks_owed(rule, *cost, who); left > 0; left = picks_owed(rule, *cost, who)) {
        const std::vector<Entity> pool = pick_pool(rule, who, orderer);
        // The offer held the player to the whole cost; a pick with nothing left to take means the cost
        // can't be paid in full, and a partial payment isn't allowed (CR 118.3, 601.2h).
        if (pool.empty()) return PayStatus::FAILED;
        size_t pick = 0;
        if (!rule.takes_forced || pool.size() > static_cast<size_t>(left)) {
            std::vector<LegalAction> menu;
            for (Entity e : pool) {
                LegalAction la(PASS_PRIORITY, e, rule.label + item_name(rule, e) + rule.label_tail);
                la.category = rule.category;
                menu.push_back(la);
            }
            const int choice = ctx.ask(std::move(menu), who.payer, who.source);
            if (choice < 0 && decision_suspended()) return PayStatus::SUSPENDED;
            pick = static_cast<size_t>(choice);
        }
        record_pick(rule, pool[pick], who);
        run.picks++;
    }
    return PayStatus::PAID;
}

// The cost `rule` reads for this payment; null when this payment doesn't pay it.
static const CostParts *pick_cost(const PickRule &rule, const CostPayer &who) {
    switch (rule.slot) {
        case CostSlot::COST:
            return who.cost;
        case CostSlot::ALTERNATIVE:
            return who.alternative;
        case CostSlot::ADDITIONAL:
            return who.additional;
        case CostSlot::ADDITIONAL_CHOICE:
            return who.additional_choice;
        case CostSlot::ESCALATE:
            return who.escalate;
        case CostSlot::FLASHBACK:
            return who.flashback;
        case CostSlot::ESCAPE:
            return who.escape;
    }
    return nullptr;
}

// How many more picks `rule` is owed.
static int picks_owed(const PickRule &rule, const CostParts &cost, const CostPayer &who) {
    const CostRun &run = *who.run;
    switch (rule.amount) {
        case PickRule::ONE_IF_ANY:
            return cost.filter(rule.kind).empty() ? 0 : 1 - run.picks;
        case PickRule::BY_TYPES:
            return cost.exile_min_types() - static_cast<int>(run.exiled_types.size());
        case PickRule::BY_COUNT:
            break;
    }
    const int times = rule.slot == CostSlot::ESCALATE ? who.escalate_times : 1;
    return cost.count(rule.kind) * times - run.picks;
}

// The candidates of the next pick, in zone order.
static std::vector<Entity> pick_pool(
    const PickRule &rule, const CostPayer &who, const std::shared_ptr<Orderer> &orderer) {
    const CostParts &cost = *pick_cost(rule, who);
    const CostRun &run = *who.run;
    std::vector<Entity> pool;
    switch (rule.kind) {
        case CostPart::SACRIFICE:
            // A .Other in the filter excludes the source ("another creature").
            pool = controlled_permanents_matching(who.payer, cost.filter(rule.kind), orderer->mEntities, who.source);
            break;
        case CostPart::RETURN:
            pool = controlled_permanents_matching(who.payer, cost.filter(rule.kind), orderer->mEntities);
            break;
        case CostPart::EXILE_FROM_HAND: {
            MatchCtx ctx;
            ctx.source = who.source;
            for (Entity e : orderer->get_hand(who.payer))
                if (card_matches_filter(e, cost.filter(rule.kind), ctx)) pool.push_back(e);
            break;
        }
        case CostPart::DISCARD:
            for (Entity e : orderer->get_hand(who.payer))
                if (e != who.source && card_matches_filter(e, cost.filter(rule.kind))) pool.push_back(e);
            break;
        default:  // EXILE_FROM_GRAVEYARD: OTHER cards of the payer's graveyard
            for (Entity e : orderer->mEntities) {
                if (e == who.source || !global_coordinator.entity_has_component<Zone>(e)) continue;
                const Zone &zone = global_coordinator.GetComponent<Zone>(e);
                if (zone.location == Zone::GRAVEYARD && zone.owner == who.payer &&
                    global_coordinator.entity_has_component<CardData>(e))
                    pool.push_back(e);
            }
            break;
    }
    // One object pays one cost (CR 118.3): what an earlier pick chose is spoken for.
    pool.erase(
        std::remove_if(pool.begin(), pool.end(), [&run](Entity e) { return chosen_as_cost(run, e); }), pool.end());
    return pool;
}

static std::string item_name(const PickRule &rule, Entity e) {
    return rule.kind == CostPart::SACRIFICE || rule.kind == CostPart::RETURN
               ? global_coordinator.GetComponent<Permanent>(e).name
               : global_coordinator.GetComponent<CardData>(e).name;
}

// Has `e` already been chosen as a cost item? A pick is still where it was, so the next menu
// filters on this.
static bool chosen_as_cost(const CostRun &run, Entity e) {
    for (const CostItem &item : run.items)
        if (item.entity == e) return true;
    return false;
}

// Record a choice. Its narrative line is formatted now, while the card is still where it is.
static void record_pick(const PickRule &rule, Entity chosen, const CostPayer &who) {
    CostRun &run = *who.run;
    CostItem item;
    item.entity = chosen;
    item.dest = rule.dest;
    item.kind = rule.kind;
    item.log = player_name(who.payer) + rule.log + item_name(rule, chosen) + rule.log_tail;
    run.items.push_back(std::move(item));
    if (rule.amount == PickRule::BY_TYPES)
        for (const auto &t : global_coordinator.GetComponent<CardData>(chosen).types)
            if (t.kind == TYPE) run.exiled_types.insert(t.name);
    if (rule.kind == CostPart::DISCARD) run.discards_chosen++;
    // Ninjutsu's ninja attacks what the returned creature was attacking (CR 702.49c), read while
    // the creature is still in combat.
    if (rule.kind == CostPart::RETURN && who.ability && who.ability->def->is_ninjutsu)
        who.announced->ninjutsu_attack_target = global_coordinator.GetComponent<Creature>(chosen).attack_target;
}

// Delve (CR 702.66 / 601.2h): the caster chooses how many graveyard cards to exile and which ones,
// one pick at a time; each exile pays one generic pip of the cost. Runs after targets are locked in
// (601.2c), like every other cost. The remainder is then paid WITHOUT delve — the count menu is
// constrained to counts whose remaining cost is payable. Stage 1 here: the count menu. The count
// range [min_needed .. max_exiles] is computed at arm and re-derived identically at apply (nothing
// runs between arm and resume): each exile removes one GENERIC pip, so payability is monotone in the
// count and the legal counts form a contiguous range; min_needed is found with the shared
// simulate-mode payer (can_pay_mana, has_delve=false), so the offered menu and the eventual payment
// can never disagree — cast legality (which allowed the maximum delve) guarantees the range is
// non-empty.
static PayStatus choose_delve_count(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    CostRun &run = *who.run;
    if (!run.mana_pending || !run.delve) return PayStatus::PAID;
    size_t eligible_ct = 0;
    for (auto e : orderer->mEntities)
        if (is_delve_eligible(e, who.payer)) eligible_ct++;
    size_t max_exiles = std::min(eligible_ct, run.mana.count(GENERIC));
    if (max_exiles == 0) return PayStatus::PAID;
    ManaValue reduced = run.mana;
    size_t min_needed = 0;
    while (min_needed < max_exiles &&
           !can_pay_mana(who.payer, reduced, who.payee, orderer, /*has_delve=*/false, run.tap_to_pay)) {
        reduced.erase(reduced.find(GENERIC));
        min_needed++;
    }
    if (!ctx.resuming()) {
        // Seat both delve stages on the casting player: the prev seat lives in the run and
        // choose_delve_cards' completion (or a rewind) restores it.
        run.delve_prev_priority_a = cur_game.priority.player_a_has_priority;
        run.delve_seat_held = true;
        cur_game.priority.player_a_has_priority = (who.payer == Zone::PLAYER_A);
    }
    // The count prompt is skipped when only one count is legal. The actions carry the delve spell as
    // their source entity, so the machine protocol emits its card id (a plain X-cost ladder emits
    // the null sentinel — this is how observers tell the two CHOOSE_X menus apart).
    if (max_exiles == min_needed) {
        run.delve_count = min_needed;
        return PayStatus::PAID;
    }
    std::vector<LegalAction> count_menu;
    for (size_t n = min_needed; n <= max_exiles; n++) {
        LegalAction la(PASS_PRIORITY, who.source,
            "Exile " + std::to_string(n) + (n == 1 ? " card" : " cards") + " from graveyard (Delve)");
        la.category = ActionCategory::CHOOSE_X;
        la.option_ordinal = static_cast<int>(n);  // the delve exile count
        la.card_is_public = true;                 // the spell being cast is public
        count_menu.push_back(la);
    }
    if (!ctx.resuming()) game_log("Choose how many cards to exile via Delve (%zu-%zu):\n", min_needed, max_exiles);
    const int choice = ctx.ask(std::move(count_menu), who.payer, who.source);
    if (choice < 0 && decision_suspended()) return PayStatus::SUSPENDED;
    run.delve_count = min_needed + static_cast<size_t>(choice);
    return PayStatus::PAID;
}

// Delve stage 2 — which cards, one pick at a time; the candidate menu is re-derived from the live
// graveyard each pass minus the cards already picked. When the remaining candidates are all going to
// be exiled anyway (candidates <= picks left, including the single-candidate case) there is no real
// choice, so they are taken without a prompt. Each pick pays one GENERIC pip of the cost (CR 702.66a)
// and is a chosen cost item: the card is exiled, and recorded in ChosenCost::moved, as the items move.
static PayStatus choose_delve_cards(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    CostRun &run = *who.run;
    if (!run.delve_seat_held) return PayStatus::PAID;
    while (static_cast<size_t>(run.picks) < run.delve_count) {
        const size_t done = static_cast<size_t>(run.picks);
        std::vector<LegalAction> picks;
        for (auto e : orderer->mEntities) {
            if (!is_delve_eligible(e, who.payer) || chosen_as_cost(run, e)) continue;
            auto &ecd = global_coordinator.GetComponent<CardData>(e);
            LegalAction la(PASS_PRIORITY, e, "Exile " + ecd.name + " (Delve)");
            la.category = ActionCategory::CHOOSE_CARD;
            la.card_is_public = true;  // graveyards are public zones
            picks.push_back(la);
        }
        if (picks.empty()) break;  // defensive: eligibility was counted above
        size_t pick = 0;
        if (picks.size() > run.delve_count - done) {
            if (!ctx.resuming())
                game_log("Choose a card to exile via Delve (%zu of %zu):\n", done + 1, run.delve_count);
            const int choice = ctx.ask(picks, who.payer, who.source);
            if (choice < 0 && decision_suspended()) return PayStatus::SUSPENDED;
            pick = static_cast<size_t>(choice);
        }
        choose_delve_exile(who, picks[pick].source_entity);
        run.picks++;
    }
    // Restore the pre-delve seat (persisted by choose_delve_count).
    cur_game.priority.player_a_has_priority = run.delve_prev_priority_a;
    run.delve_seat_held = false;
    return PayStatus::PAID;
}

// A delve exile pays one generic pip now and moves with the other items.
static void choose_delve_exile(const CostPayer &who, Entity e) {
    CostRun &run = *who.run;
    auto git = run.mana.find(GENERIC);
    if (git != run.mana.end()) run.mana.erase(git);
    CostItem item;
    item.entity = e;
    item.dest = Zone::EXILE;
    item.kind = CostPart::EXILE_FROM_GRAVEYARD;
    item.delve = true;
    item.log = player_name(who.payer) + " exiles " + global_coordinator.GetComponent<CardData>(e).name + " via Delve.";
    run.items.push_back(std::move(item));
}

// The mana an activation owes (after ReduceCost$ — CR 601.2f; reduces generic only; Eiganjo's
// Channel is cheaper per legendary creature). For an X-cost ability the chosen X is added as generic
// mana on top of the base cost — once per {X} pip in the cost, so Blast Zone's {X}{X} charges 2X,
// not X.
static void owe_activation(const CostPayer &who, const std::shared_ptr<Orderer> &orderer) {
    CostRun &run = *who.run;
    if (!run.mana_announced) {
        run.mana = effective_activation_mana_cost(*who.ability, who.payer, orderer);
        const size_t x_pips = std::max<size_t>(1, who.cost->x_pips());
        for (size_t i = 0; i < who.x * x_pips; i++) run.mana.insert(GENERIC);
    }
    run.mana_pending = true;
    run.tap_to_pay = who.cost->tap_to_pay();
    run.life = who.cost->life();
    run.energy = who.cost->energy();
}

// The --coverage form of a part paid along `path`, when the path has one of its own.
static const char *path_form(CostPath path) { return path == CostPath::TRIGGER ? "trigger" : nullptr; }

// The mana, last of what a payment can fail on.
static PayStatus pay_mana(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    CostRun &run = *who.run;
    if (!run.mana_pending || run.mana.empty()) return PayStatus::PAID;
    const PayStatus paid = pay_mana_cost(run.mana_payment, who.payer, run.mana, who.payee, ctx, orderer,
        run.tap_to_pay, &run.spent_colors);
    if (paid == PayStatus::PAID) coverage_cost(CostPart::MANA, path_form(who.path));
    return paid;
}

// The life and energy are checked before any of them is paid: a painful mana source the payment
// used can have left too little life (the gate reserves it, so only a wrongly offered proposal gets
// here), and then the whole cost can't be paid and the proposal is reversed (CR 601.5).
static bool resources_payable(const CostPayer &who) {
    const CostRun &run = *who.run;
    const auto &player = global_coordinator.GetComponent<Player>(get_player_entity(who.payer));
    return can_pay_life(player, run.life + std::max(0, run.life_x)) && player_energy(player) >= run.energy;
}

// Loyalty cost (606.4/606.5): pay by adding/removing loyalty counters on the source planeswalker, and
// mark the per-permanent once-per-turn gate (606.3). The ability still goes on the stack and resolves
// later; the loyalty change is the cost, paid now.
static void pay_loyalty(const CostPayer &who) {
    const AbilityDef &def = *who.ability->def;
    if (!def.is_loyalty_ability || !global_coordinator.entity_has_component<Permanent>(who.source)) return;
    auto &perm = global_coordinator.GetComponent<Permanent>(who.source);
    // An X loyalty cost (Chandra, Flamecaller's [-X]) removes/adds the X chosen at activation; the
    // part carries only the sign. A fixed cost uses its count as-is.
    const int loyalty_delta = def.cost.loyalty_is_x()
                                  ? (def.cost.loyalty() < 0 ? -who.announced->x_paid : who.announced->x_paid)
                                  : def.cost.loyalty();
    const int loyalty = add_counters(who.source, "LOYALTY", loyalty_delta);
    perm.loyalty_ability_activated_this_turn = true;
    game_log("%s activates a loyalty ability (%+d, loyalty now %d)\n", entity_name(who.source).c_str(), loyalty_delta,
        loyalty);
    coverage_cost(CostPart::LOYALTY);
}

// Everything the picks chose moves, in the order chosen. Nothing here can fail: the choices were
// made against the live board and only this step moves anything. A permanent leaving here may make a
// chosen target illegal; the spell then fizzles at resolution (CR 608.2b), matching paper rules.
static void move_items(const CostPayer &who, const std::shared_ptr<Orderer> &orderer) {
    CostRun &run = *who.run;
    for (const CostItem &item : run.items) {
        coverage_cost(item.kind, item.delve ? "delve" : path_form(who.path));
        if (who.path == CostPath::CAST) game_log("%s\n", item.log.c_str());
        orderer->add_to_zone(false, item.entity, item.dest);
        if (who.path != CostPath::CAST) game_log("%s\n", item.log.c_str());
        if (who.chosen) who.chosen->moved.push_back({item.kind, ObjectRef::of(item.entity)});
    }
    run.items.clear();
}

// A variable life cost's announced X, paid with the rest of the life (Toxic Deluge).
static void pay_life_x(const CostPayer &who) {
    const CostRun &run = *who.run;
    if (run.life_x < 0) return;
    pay_life(global_coordinator.GetComponent<Player>(get_player_entity(who.payer)), run.life_x);
    game_log("%s pays %d life (X = %d)\n", player_name(who.payer).c_str(), run.life_x, run.life_x);
    coverage_cost(CostPart::LIFE, "x");
}

// Sacrifice self: move to graveyard; apply_permanent_components SBA removes Permanent next pass.
static void sacrifice_self(const CostPayer &who, const std::shared_ptr<Orderer> &orderer) {
    if (!who.cost->self(CostPart::SACRIFICE)) return;
    std::string sname = entity_name(who.source);
    orderer->add_to_zone(false, who.source, Zone::GRAVEYARD);
    game_log("%s sacrifices %s\n", player_name(who.payer).c_str(), sname.c_str());
    coverage_cost(CostPart::SACRIFICE, "self");
}

// The source exiles itself: from the battlefield (Exile<1/CARDNAME>: The Echoverse Fulcrum) or the
// zone it is activated in (ExileFromGrave<1/CARDNAME>: Runehorn Hellkite; ExileFromHand<1/CARDNAME>:
// Simian Spirit Guide, a mana ability).
static void exile_self(const CostPayer &who, const std::shared_ptr<Orderer> &orderer) {
    if (!who.commit) return;
    for (CostPart::Kind kind : {CostPart::EXILE, CostPart::EXILE_FROM_HAND, CostPart::EXILE_FROM_GRAVEYARD}) {
        if (!who.cost->self(kind)) continue;
        const std::string name = entity_name(who.source);
        orderer->add_to_zone(false, who.source, Zone::EXILE);
        game_log("%s exiles %s\n", player_name(who.payer).c_str(), name.c_str());
        coverage_cost(kind, "self");
    }
}

// Discard self from hand cost (Faerie Macabre).
static void discard_self(const CostPayer &who, const std::shared_ptr<Orderer> &orderer) {
    if (!who.cost->self(CostPart::DISCARD)) return;
    std::string cname = global_coordinator.entity_has_component<CardData>(who.source)
                            ? global_coordinator.GetComponent<CardData>(who.source).name
                            : "card";
    orderer->add_to_zone(false, who.source, Zone::GRAVEYARD);
    game_log("%s discards %s\n", player_name(who.payer).c_str(), cname.c_str());
    coverage_cost(CostPart::DISCARD, "self");
}

// Discard hand cost (Lion's Eye Diamond).
static void discard_hand(const CostPayer &who, const std::shared_ptr<Orderer> &orderer) {
    if (!who.cost->discards_hand()) return;
    coverage_cost(CostPart::DISCARD, "hand");
    for (auto card : orderer->get_hand(who.payer)) {
        std::string cname = global_coordinator.entity_has_component<CardData>(card)
                                ? global_coordinator.GetComponent<CardData>(card).name
                                : "card";
        orderer->add_to_zone(false, card, Zone::GRAVEYARD);
        game_log("%s discards %s\n", player_name(who.payer).c_str(), cname.c_str());
    }
}

// The ability's activation mana, from the working pool. pay_from_pool returns the unpayable remainder
// and drains the pool even on a partial payment, so snapshot the pool and restore it when the cost
// bounces.
static PayStatus pay_from_working_pool(const CostPayer &who) {
    if (who.cost->mana().empty()) return PayStatus::PAID;
    ManaPool pool_before = *who.pool;
    ManaValue unpaid = pay_from_pool(*who.pool, who.cost->mana());
    if (unpaid.empty()) return PayStatus::PAID;
    *who.pool = pool_before;
    return PayStatus::FAILED;
}

// Reflexive "you may sacrifice CARDNAME. If you do, ..." cost on a triggered ability (The
// Fantasticar's fourth-noncreature-spell trigger: Execute AB$ Token | Cost$ Sac<1/CARDNAME>). The
// Sac<.../CARDNAME> cost makes the whole effect optional: prompt the controller, sacrifice the source
// on accept, and do nothing on decline. With Cost$ Mandatory (Dark Depths: "sacrifice it. If you do,
// create Marit Lage.") the sacrifice is not optional and nothing is asked. Either way a source no
// longer on the battlefield is not the permanent that triggered (CR 400.7): it can't be sacrificed,
// so the "if you do" effect doesn't happen.
static PayStatus pay_trigger_sacrifice(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    if (!who.cost->self(CostPart::SACRIFICE)) return PayStatus::PAID;
    const Entity src = who.source;
    const std::string sname = !who.source_gone && global_coordinator.entity_has_component<Permanent>(src)
                                  ? global_coordinator.GetComponent<Permanent>(src).name
                                  : std::string("it");
    if (who.source_gone || !is_battlefield_permanent(src)) {
        game_log("%s is already gone; nothing is sacrificed.\n", sname.c_str());
        return PayStatus::DECLINED;
    }
    if (!who.mandatory) {
        const int yc = ctx.ask(yesno_menu("Don't sacrifice " + sname, "Sacrifice " + sname), who.payer, src);
        if (yc < 0 && decision_suspended()) return PayStatus::SUSPENDED;
        if (yc == 0) {
            game_log("%s declines to sacrifice %s.\n", player_name(who.payer).c_str(), sname.c_str());
            return PayStatus::DECLINED;
        }
    }
    orderer->add_to_zone(false, src, Zone::GRAVEYARD);
    game_log("%s sacrifices %s.\n", player_name(who.payer).c_str(), sname.c_str());
    return PayStatus::PAID;
}

// "You may discard a card", "you may pay {B}" (CR 603.5): offered only when the controller can pay
// (CR 118.3), and on no answer of yes the rest of the ability doesn't happen.
static PayStatus offer_trigger_cost(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    if (!trigger_cost_offered(*who.cost)) return PayStatus::PAID;
    CostCheck check(who.payer, who.source, orderer);
    check.payee = who.payee;
    if (!can_pay(*who.cost, check)) return PayStatus::DECLINED;
    const int yc = ctx.ask(optional_yesno_menu(cost_offer_text(*who.cost)), who.payer, who.source);
    if (yc < 0 && decision_suspended()) return PayStatus::SUSPENDED;
    if (yc == 1) return PayStatus::PAID;
    game_log("%s declines to %s.\n", player_name(who.payer).c_str(), cost_offer_text(*who.cost).c_str());
    return PayStatus::DECLINED;
}

// "discard a card", "pay {B}", "pay 3 life".
static std::string cost_offer_text(const CostParts &cost) {
    std::string text;
    if (!cost.mana().empty()) text = "pay " + mana_value_text(cost.mana());
    if (cost.life() > 0) text += (text.empty() ? "pay " : " and ") + std::to_string(cost.life()) + " life";
    const int discards = cost.count(CostPart::DISCARD);
    if (discards > 0) {
        if (!text.empty()) text += " and ";
        text += "discard " + (discards == 1 ? std::string("a card") : std::to_string(discards) + " cards");
    }
    return text;
}

std::string cost_label(const CostParts &cost) {
    std::string text = cost_offer_text(cost);
    if (!text.empty()) text[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(text[0])));
    return text;
}

// A payment given up partway (the last row of the interactive payment) gives its mana abilities
// back (CR 733.1), and the offer stands declined.
static PayStatus pay_trigger_mana(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    const PayStatus paid = pay_mana(who, ctx, orderer);
    if (paid != PayStatus::FAILED) return paid;
    restore_mana_state(who.payer, who.run->mana_snap, orderer);
    game_log("%s declines to %s.\n", player_name(who.payer).c_str(), cost_offer_text(*who.cost).c_str());
    return PayStatus::DECLINED;
}

// An ImmediateTrigger's optional PayEnergy<N> ("you may pay {E}. When you do, ..."): offered only
// when the controller can pay it (CR 122.1c).
static PayStatus offer_trigger_energy(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    if (!can_pay(*who.cost, CostCheck(who.payer, who.source, orderer))) return PayStatus::DECLINED;
    const int energy = who.cost->energy();
    const int yc = ctx.ask(optional_yesno_menu("Pay " + std::to_string(energy) + " energy"), who.payer, who.source);
    if (yc < 0 && decision_suspended()) return PayStatus::SUSPENDED;
    auto &payer = global_coordinator.GetComponent<Player>(get_player_entity(who.payer));
    if (yc != 1 || !pay_energy(payer, energy)) return PayStatus::DECLINED;
    game_log("%s pays %d energy.\n", player_name(who.payer).c_str(), energy);
    return PayStatus::PAID;
}

// The "unless they discard N cards" flavor (CR 701.8; Reality Smasher): the payer may discard
// `amount` cards of their choice from hand, and declines if they have too few. The chosen cards go to
// the graveyard (a public zone — every viewer sees them). The yes/no answer and the discard-count
// progress persist in the level's UnlessRt; the per-discard menu is intentionally rebuilt from the
// LIVE hand each ask — between asks the hand only shrinks by the discards themselves.
static PayStatus offer_unless_discard(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    const UnlessTerms &terms = *who.unless;
    UnlessRt local_rt;
    UnlessRt &rt = ctx.can_suspend() ? ctx.rt<UnlessRt>() : local_rt;

    if (!rt.pay_answered) {
        const bool payable = orderer->get_hand(who.payer).size() >= terms.amount;  // the full count or none
        std::vector<LegalAction> actions;
        if (payable) actions.push_back(unless_pay_action(terms.subject, terms.kind, terms.amount, nullptr));
        const size_t decline_idx = actions.size();
        actions.push_back(unless_decline_action(terms.subject, terms.kind));

        const int choice = ctx.ask(std::move(actions), who.payer, who.source);
        if (choice < 0 && decision_suspended()) return PayStatus::SUSPENDED;
        if (!payable || choice == static_cast<int>(decline_idx)) return PayStatus::DECLINED;
        rt.pay_answered = true;
    }

    // Pay: the payer chooses `amount` distinct cards from hand to discard.
    for (; rt.discards_done < terms.amount; ++rt.discards_done) {
        std::vector<Entity> cur_hand = orderer->get_hand(who.payer);
        if (cur_hand.empty()) break;
        std::vector<LegalAction> dactions;
        for (auto e : cur_hand) {
            LegalAction la(PASS_PRIORITY, e, global_coordinator.GetComponent<CardData>(e).name);
            la.category = ActionCategory::DISCARD;
            dactions.push_back(la);
        }
        const int dchoice = ctx.ask(dactions, who.payer, who.source);
        if (dchoice < 0 && decision_suspended()) return PayStatus::SUSPENDED;
        const Entity chosen = dactions[static_cast<size_t>(dchoice)].source_entity;
        game_log("%s discards %s — %s\n", player_name(who.payer).c_str(),
            global_coordinator.GetComponent<CardData>(chosen).name.c_str(),
            unless_outcome_text(terms.subject, /*paid=*/true).c_str());
        orderer->add_to_zone(false, chosen, Zone::GRAVEYARD);
    }
    return PayStatus::PAID;
}

// Pay `amount` life (Ward—Pay N life, CR 702.21) or energy ({E}, CR 122.1c: Static Prison's "sacrifice
// CARDNAME unless you pay {E}"), or decline. Offered only to a payer who has it (CR 119.4; energy
// lives as an "ENERGY" counter on the payer).
static PayStatus offer_unless_resource(const CostPayer &who, FrameCtx &ctx) {
    const UnlessTerms &terms = *who.unless;
    const bool life = terms.kind == UnlessPayKind::LIFE;
    const int amount = static_cast<int>(terms.amount);
    auto &payer = global_coordinator.GetComponent<Player>(get_player_entity(who.payer));
    const bool payable = life ? can_pay_life(payer, amount) : player_energy(payer) >= amount;

    std::vector<LegalAction> unless_actions;
    if (payable) unless_actions.push_back(unless_pay_action(terms.subject, terms.kind, terms.amount, nullptr));
    unless_actions.push_back(unless_decline_action(terms.subject, terms.kind));

    const int choice = ctx.ask(std::move(unless_actions), who.payer, who.source);
    if (choice < 0 && decision_suspended()) return PayStatus::SUSPENDED;
    if (!payable || choice != 0) return PayStatus::DECLINED;
    if (life)
        pay_life(payer, amount);
    else
        pay_energy(payer, amount);
    game_log("%s pays %zu %s — %s\n", player_name(who.payer).c_str(), terms.amount, life ? "life" : "energy",
        unless_outcome_text(terms.subject, /*paid=*/true).c_str());
    return PayStatus::PAID;
}

// Daze / Mana Leak / Ward {N}: pay-{N}-unless with tap-for-mana sub-choices. A live-menu loop (Shape
// C) with no persisted progress: the menu is rebuilt from live state every pass — floated mana
// (Player.mana, snapshot-covered) changes which cost-bearing sources are offered and whether "Pay"
// is affordable — so a resume re-enters, rebuilds the identical menu, and the ask consumes the
// latched answer (its size assert guards drift). A chosen tap action executes its mana ability at
// consume time (mana floats), then the loop re-arms the rebuilt menu. Exact colored pips (Chain
// Lightning: {R}{R}) when supplied; else `amount` generic pips.
static PayStatus offer_unless_mana(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    const UnlessTerms &terms = *who.unless;
    std::multiset<Colors> cond_cost;
    if (terms.pips && !terms.pips->empty())
        cond_cost = *terms.pips;
    else
        for (size_t i = 0; i < terms.amount; i++) cond_cost.insert(GENERIC);

    while (true) {
        std::vector<LegalAction> unless_actions = collect_mana_legal_actions(who.payer, orderer, who.payee);
        // Drop cost-bearing mana sources (Talon Gates' {1}{T}) whose activation cost the
        // FLOATING pool can't cover right now: the tap-a-source loop below pays activation
        // costs only from mana already floating, so offering such a source would either
        // produce free mana or (refused) re-offer the same menu forever to a deterministic
        // machine-mode agent. The menu is rebuilt each iteration, so the source reappears
        // as soon as enough mana has been floated.
        unless_actions.erase(std::remove_if(unless_actions.begin(), unless_actions.end(),
                                 [&](const LegalAction &la) {
                                     return !la.ability.def->cost.mana().empty() &&
                                            !can_afford(who.payer, la.ability.def->cost.mana());
                                 }),
            unless_actions.end());

        const bool payable = can_afford(who.payer, cond_cost);
        const size_t pay_idx = unless_actions.size();
        if (payable) unless_actions.push_back(unless_pay_action(terms.subject, terms.kind, terms.amount, terms.pips));
        const size_t decline_idx = unless_actions.size();
        unless_actions.push_back(unless_decline_action(terms.subject, terms.kind));

        // Passed as an lvalue: the tap branch below still needs the menu to map
        // the chosen action.
        const int choice = ctx.ask(unless_actions, who.payer, who.source);
        if (choice < 0 && decision_suspended()) return PayStatus::SUSPENDED;

        if (choice == static_cast<int>(decline_idx)) return PayStatus::DECLINED;

        if (payable && choice == static_cast<int>(pay_idx)) {
            spend_mana(who.payer, cond_cost, who.payee.entity);
            game_log("%s pays %s — %s\n", player_name(who.payer).c_str(),
                unless_cost_text(terms.kind, terms.amount, terms.pips).c_str(),
                unless_outcome_text(terms.subject, /*paid=*/true).c_str());
            return PayStatus::PAID;
        }

        if (choice >= 0 && choice < static_cast<int>(pay_idx)) {
            auto &chosen = unless_actions[static_cast<size_t>(choice)];
            auto &pl = global_coordinator.GetComponent<Player>(get_player_entity(who.payer));
            // Canonical activation against the player's real (floating) pool: pays a
            // cost-bearing source's activation cost (Talon Gates' {1}{T}) from the pool —
            // refusing cleanly (no tap, no mana) while the pool can't cover it, so the
            // re-prompt lets the player float mana from a cost-free source first — and
            // applies the full mana-ability semantics: tap only tap-cost sources, a self-sacrifice
            // (Lotus Petal), life costs, dynamic amounts, SubAbility$ riders (Ancient
            // Tomb's damage), ProduceMana replacements, and activation counters.
            if (!activate_mana_source(chosen.source_entity, chosen.ability, who.payer, orderer, pl.mana, pl,
                    /*commit=*/true, ManaLogStyle::TAPPED_SYMBOL))
                game_log("Cannot pay that ability's activation cost — tap another source for mana first.\n");
        }
    }
}
