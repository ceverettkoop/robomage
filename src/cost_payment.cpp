#include "cost_payment.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

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
#include "queries/battlefield.h"
#include "queries/characteristics.h"
#include "queries/counters.h"
#include "queries/player_resources.h"
#include "queries/players.h"
#include "resolution_frame.h"
#include "systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace {

// How one choice of what a part moves is asked and recorded.
struct PickRule {
    CostPart::Kind kind;
    const char *label;       // the menu row: label + name + label_tail
    const char *label_tail;
    ActionCategory category;
    Zone::ZoneValue dest;
    const char *log;  // the narrative line: player + log + name + log_tail
    const char *log_tail;
};

// One step of a payment path.
struct CostStep {
    enum Op {
        PICK,               // choose what `pick` moves; the move waits for the mana
        SNAPSHOT,           // the snapshot a reversed payment restores
        TAP,                // {T}: the source taps
        OWE_ACTIVATION,     // what an activation owes
        FLOAT_LEAVING,      // CR 601.2g: mana from a permanent about to leave to pay the cost
        PAY_MANA,
        CHECK_RESOURCES,    // the life and energy can still be paid
        COUNT_TAP,
        LOYALTY,
        LIFE,
        ENERGY,
        SACRIFICE_SELF,
        MOVE_ITEMS,         // what the picks chose moves
        DISCARD_SELF,
        DISCARD_HAND,
    };
    Op op;
    const PickRule *pick = nullptr;
};

struct CostSteps {
    const CostStep *rows;
    size_t size;
};

}  // namespace

static CostSteps path_steps(CostPath path);
static PayStatus pay_step(const CostStep &step, const CostPayer &who, FrameCtx &ctx,
                          const std::shared_ptr<Orderer> &orderer);
static PayStatus pick_items(const PickRule &rule, const CostPayer &who, FrameCtx &ctx,
                            const std::shared_ptr<Orderer> &orderer);
static std::vector<Entity> pick_pool(const PickRule &rule, const CostPayer &who,
                                     const std::shared_ptr<Orderer> &orderer);
static std::string item_name(const PickRule &rule, Entity e);
static void owe_activation(const CostPayer &who, const std::shared_ptr<Orderer> &orderer);
static PayStatus pay_mana(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer);
static bool resources_payable(const CostPayer &who);
static void pay_loyalty(const CostPayer &who);
static void move_items(const CostPayer &who, const std::shared_ptr<Orderer> &orderer);
static const char *cost_item_part(const CostItem &item);
static void sacrifice_self(const CostPayer &who, const std::shared_ptr<Orderer> &orderer);
static void discard_self(const CostPayer &who, const std::shared_ptr<Orderer> &orderer);
static void discard_hand(const CostPayer &who, const std::shared_ptr<Orderer> &orderer);

// ── What a pick chooses ──────────────────────────────────────────────────────────────────────
// Type-based sacrifice cost (Cycling "Sac a land", Knight of the Reliquary).
static const PickRule kSacrificeOne = {CostPart::SACRIFICE,  "Sacrifice ",   "", ActionCategory::SACRIFICE_PERMANENT,
                                       Zone::GRAVEYARD, " sacrifices ", ""};
// Return-to-hand cost (Scryb Ranger: return a Forest to hand).
static const PickRule kReturnOne = {CostPart::RETURN, "Return ",   " to hand", ActionCategory::RETURN_PERMANENT,
                                    Zone::HAND,       " returns ", " to hand"};

// ── The paths ────────────────────────────────────────────────────────────────────────────────
// CR 602.2b: choose what the cost moves, then pay it (CR 601.2g-h). The tap comes first, recorded by
// the snapshot, so a reversed activation untaps; the life and energy after the mana, which a painful
// source can have made unpayable.
static const CostStep kActivationSteps[] = {
    {CostStep::PICK, &kSacrificeOne},
    {CostStep::PICK, &kReturnOne},
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
    {CostStep::MOVE_ITEMS},
    {CostStep::DISCARD_SELF},
    {CostStep::DISCARD_HAND},
};

static CostSteps path_steps(CostPath path) {
    switch (path) {
        case CostPath::ACTIVATION:
            break;
    }
    return {kActivationSteps, sizeof(kActivationSteps) / sizeof(kActivationSteps[0])};
}

PayStatus run_cost_parts(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    const CostSteps steps = path_steps(who.path);
    CostRun &run = *who.run;
    for (; run.step < steps.size; ++run.step, run.picks = 0) {
        const PayStatus status = pay_step(steps.rows[run.step], who, ctx, orderer);
        if (status != PayStatus::PAID) return status;
    }
    return PayStatus::PAID;
}

static PayStatus pay_step(const CostStep &step, const CostPayer &who, FrameCtx &ctx,
                          const std::shared_ptr<Orderer> &orderer) {
    CostRun &run = *who.run;
    auto &player = global_coordinator.GetComponent<Player>(get_player_entity(who.payer));
    switch (step.op) {
        case CostStep::PICK:
            return pick_items(*step.pick, who, ctx, orderer);
        case CostStep::SNAPSHOT:
            // The first step that can activate a mana ability (CR 733.1).
            run.mana_snap = snapshot_mana_state(who.payer, orderer);
            run.mana_snap_taken = true;
            break;
        case CostStep::TAP:
            if (who.cost->taps() && global_coordinator.entity_has_component<Permanent>(who.source))
                global_coordinator.GetComponent<Permanent>(who.source).is_tapped = true;
            break;
        case CostStep::OWE_ACTIVATION:
            owe_activation(who, orderer);
            break;
        case CostStep::FLOAT_LEAVING:
            if (run.mana_pending)
                for (const CostItem &item : run.items)
                    float_mana_before_cost_removal(item.entity, who.payer, orderer, run.mana, who.payee);
            break;
        case CostStep::PAY_MANA:
            return pay_mana(who, ctx, orderer);
        case CostStep::CHECK_RESOURCES:
            if (!resources_payable(who)) return PayStatus::FAILED;
            run.committed = true;
            break;
        case CostStep::COUNT_TAP:
            if (who.cost->taps()) coverage_cost("Tap");
            break;
        case CostStep::LOYALTY:
            pay_loyalty(who);
            break;
        case CostStep::LIFE:
            if (run.life > 0) {
                pay_life(player, run.life);
                game_log("%s pays %d life\n", player_name(who.payer).c_str(), run.life);
                coverage_cost("PayLife");
            }
            break;
        case CostStep::ENERGY:
            // PayEnergy<N>, CR 122.1c.
            if (run.energy > 0) {
                pay_energy(player, run.energy);
                game_log("%s pays %d energy\n", player_name(who.payer).c_str(), run.energy);
                coverage_cost("PayEnergy");
            }
            break;
        case CostStep::SACRIFICE_SELF:
            sacrifice_self(who, orderer);
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
    }
    return PayStatus::PAID;
}

// Choose what `rule`'s part moves, the menu rebuilt from the live zones at each pass.
static PayStatus pick_items(const PickRule &rule, const CostPayer &who, FrameCtx &ctx,
                            const std::shared_ptr<Orderer> &orderer) {
    CostRun &run = *who.run;
    if (who.cost->filter(rule.kind).empty()) return PayStatus::PAID;
    std::vector<Entity> pool = pick_pool(rule, who, orderer);
    if (pool.empty()) return PayStatus::PAID;
    std::vector<LegalAction> menu;
    for (Entity e : pool) {
        LegalAction la(PASS_PRIORITY, e, rule.label + item_name(rule, e) + rule.label_tail);
        la.category = rule.category;
        menu.push_back(la);
    }
    const int choice = ctx.ask(std::move(menu), who.payer, who.source);
    if (choice < 0 && decision_suspended()) return PayStatus::SUSPENDED;
    const Entity chosen = pool[static_cast<size_t>(choice)];
    CostItem item;
    item.entity = chosen;
    item.dest = rule.dest;
    item.kind = rule.kind;
    item.log = player_name(who.payer) + rule.log + item_name(rule, chosen) + rule.log_tail;
    run.items.push_back(std::move(item));
    run.picks++;
    // Ninjutsu's ninja attacks what the returned creature was attacking (CR 702.49c), read while
    // the creature is still in combat.
    if (rule.kind == CostPart::RETURN && who.ability && who.ability->def->is_ninjutsu)
        who.announced->ninjutsu_attack_target = global_coordinator.GetComponent<Creature>(chosen).attack_target;
    return PayStatus::PAID;
}

static std::vector<Entity> pick_pool(const PickRule &rule, const CostPayer &who,
                                     const std::shared_ptr<Orderer> &orderer) {
    const FilterSpec &filter = who.cost->filter(rule.kind);
    // A .Other in a sacrifice's filter excludes the source ("another creature").
    return controlled_permanents_matching(who.payer, filter, orderer->mEntities,
                                          rule.kind == CostPart::SACRIFICE ? who.source : 0);
}

static std::string item_name(const PickRule &, Entity e) { return global_coordinator.GetComponent<Permanent>(e).name; }

// The mana an activation owes (after ReduceCost$ — CR 601.2f; reduces generic only; Eiganjo's
// Channel is cheaper per legendary creature). For an X-cost ability the chosen X is added as generic
// mana on top of the base cost — once per {X} pip in the cost, so Blast Zone's {X}{X} charges 2X,
// not X.
static void owe_activation(const CostPayer &who, const std::shared_ptr<Orderer> &orderer) {
    CostRun &run = *who.run;
    run.mana = effective_activation_mana_cost(*who.ability, who.payer, orderer);
    const size_t x_pips = std::max<size_t>(1, who.cost->x_pips());
    for (size_t i = 0; i < who.x * x_pips; i++) run.mana.insert(GENERIC);
    run.mana_pending = true;
    run.life = who.cost->life();
    run.energy = who.cost->energy();
}

// The mana, last of what a payment can fail on.
static PayStatus pay_mana(const CostPayer &who, FrameCtx &ctx, const std::shared_ptr<Orderer> &orderer) {
    CostRun &run = *who.run;
    if (!run.mana_pending || run.mana.empty()) return PayStatus::PAID;
    const PayStatus paid = pay_mana_cost(run.mana_payment, who.payer, run.mana, who.payee, ctx, orderer);
    if (paid == PayStatus::PAID) coverage_cost("Mana");
    return paid;
}

// The life and energy are checked before any of them is paid: a painful mana source the payment
// used can have left too little life (the gate reserves it, so only a wrongly offered proposal gets
// here), and then the whole cost can't be paid and the proposal is reversed (CR 601.5).
static bool resources_payable(const CostPayer &who) {
    const CostRun &run = *who.run;
    const auto &player = global_coordinator.GetComponent<Player>(get_player_entity(who.payer));
    return can_pay_life(player, run.life) && player_energy(player) >= run.energy;
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
    coverage_cost("Loyalty");
}

// Everything the picks chose moves, in the order chosen. Nothing here can fail: the choices were
// made against the live board and only this step moves anything.
static void move_items(const CostPayer &who, const std::shared_ptr<Orderer> &orderer) {
    CostRun &run = *who.run;
    for (const CostItem &item : run.items) {
        orderer->add_to_zone(false, item.entity, item.dest);
        game_log("%s\n", item.log.c_str());
        coverage_cost(cost_item_part(item));
    }
    run.items.clear();
}

// The --coverage cost part a chosen cost item pays.
static const char *cost_item_part(const CostItem &item) {
    return item.kind == CostPart::RETURN ? "Return" : "Sac";
}

// Sacrifice self: move to graveyard; apply_permanent_components SBA removes Permanent next pass.
static void sacrifice_self(const CostPayer &who, const std::shared_ptr<Orderer> &orderer) {
    if (!who.cost->self(CostPart::SACRIFICE)) return;
    std::string sname = entity_name(who.source);
    orderer->add_to_zone(false, who.source, Zone::GRAVEYARD);
    game_log("%s sacrifices %s\n", player_name(who.payer).c_str(), sname.c_str());
    coverage_cost("SacSelf");
}

// Discard self from hand cost (Faerie Macabre).
static void discard_self(const CostPayer &who, const std::shared_ptr<Orderer> &orderer) {
    if (!who.cost->self(CostPart::DISCARD)) return;
    std::string cname = global_coordinator.entity_has_component<CardData>(who.source)
                            ? global_coordinator.GetComponent<CardData>(who.source).name
                            : "card";
    orderer->add_to_zone(false, who.source, Zone::GRAVEYARD);
    game_log("%s discards %s\n", player_name(who.payer).c_str(), cname.c_str());
    coverage_cost("DiscardSelf");
}

// Discard hand cost (Lion's Eye Diamond).
static void discard_hand(const CostPayer &who, const std::shared_ptr<Orderer> &orderer) {
    if (!who.cost->discards_hand()) return;
    coverage_cost("DiscardHand");
    for (auto card : orderer->get_hand(who.payer)) {
        std::string cname = global_coordinator.entity_has_component<CardData>(card)
                                ? global_coordinator.GetComponent<CardData>(card).name
                                : "card";
        orderer->add_to_zone(false, card, Zone::GRAVEYARD);
        game_log("%s discards %s\n", player_name(who.payer).c_str(), cname.c_str());
    }
}
