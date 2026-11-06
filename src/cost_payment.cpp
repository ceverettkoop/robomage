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
#include "queries/filters.h"
#include "queries/player_resources.h"
#include "queries/players.h"
#include "resolution_frame.h"
#include "systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace {

// The cost a pick reads, for a path that pays more than one.
enum class CostSlot { COST, ALTERNATIVE, ADDITIONAL, ESCALATE, FLASHBACK, ESCAPE };

// How one choice of what a part moves is asked and recorded.
struct PickRule {
    CostSlot slot;
    CostPart::Kind kind;
    const char *label;       // the menu row: label + name + label_tail
    const char *label_tail;
    ActionCategory category;
    Zone::ZoneValue dest;
    const char *log;  // the narrative line: player + log + name + log_tail
    const char *log_tail;
    // How many it takes. ONE_IF_ANY: a single pick when the filter names any and a candidate exists,
    // whatever the part's count. BY_COUNT: the part's count. BY_TYPES: until the cards chosen cover
    // its distinct card types.
    enum Amount { ONE_IF_ANY, BY_COUNT, BY_TYPES } amount;
    bool repeats_chosen;    // what is already chosen stays in the pool
    bool stops_when_empty;  // no candidate left ends the picks (a cost legality guaranteed)
    bool takes_forced;      // nothing is asked when every candidate must be taken
};

// One step of a payment path.
struct CostStep {
    enum Op {
        PICK,               // choose what `pick` moves; the move waits for the mana
        DELVE_COUNT,        // how many cards Delve exiles
        DELVE_PICK,         // which ones
        SNAPSHOT,           // the snapshot a reversed payment restores
        TAP,                // {T}: the source taps
        OWE_ACTIVATION,     // what an activation owes
        FLOAT_LEAVING,      // CR 601.2g: mana from a permanent about to leave to pay the cost
        RECORD_MANA_SPENT,
        PAY_MANA,
        CHECK_RESOURCES,    // the life and energy can still be paid
        COUNT_TAP,
        LOYALTY,
        LIFE,
        LIFE_X,
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
static bool resources_payable(const CostPayer &who);
static void pay_loyalty(const CostPayer &who);
static void move_items(const CostPayer &who, const std::shared_ptr<Orderer> &orderer);
static const char *cost_item_part(const CostItem &item);
static void pay_life_x(const CostPayer &who);
static void sacrifice_self(const CostPayer &who, const std::shared_ptr<Orderer> &orderer);
static void discard_self(const CostPayer &who, const std::shared_ptr<Orderer> &orderer);
static void discard_hand(const CostPayer &who, const std::shared_ptr<Orderer> &orderer);

// ── What a pick chooses ──────────────────────────────────────────────────────────────────────
// Type-based sacrifice cost (Cycling "Sac a land", Knight of the Reliquary).
static const PickRule kActivationSacrifice = {CostSlot::COST, CostPart::SACRIFICE, "Sacrifice ", "",
    ActionCategory::SACRIFICE_PERMANENT, Zone::GRAVEYARD, " sacrifices ", "", PickRule::ONE_IF_ANY, true, true, false};
// Return-to-hand cost (Scryb Ranger: return a Forest to hand).
static const PickRule kActivationReturn = {CostSlot::COST, CostPart::RETURN, "Return ", " to hand",
    ActionCategory::RETURN_PERMANENT, Zone::HAND, " returns ", " to hand", PickRule::ONE_IF_ANY, true, true, false};
// Alt-cost pitch (Force of Will: "exile a blue card from your hand"). The alternative cost's picks
// are skipped entirely on a normal cast of the same card (CR 601.2b): Daze cast for {1}{U} does not
// also return an Island.
static const PickRule kAltPitch = {CostSlot::ALTERNATIVE, CostPart::EXILE_FROM_HAND, "Exile ", "",
    ActionCategory::PAYING_COSTS, Zone::EXILE, " exiles ", "", PickRule::BY_COUNT, false, false, false};
// Alt-cost return-to-hand (Daze: "return an Island you control to its owner's hand").
static const PickRule kAltReturn = {CostSlot::ALTERNATIVE, CostPart::RETURN, "Return ", "",
    ActionCategory::RETURN_PERMANENT, Zone::HAND, " returns ", " to hand", PickRule::BY_COUNT, false, false, false};
// Alt-cost sacrifice (Fireblast: "sacrifice two Mountains rather than pay this spell's mana cost",
// CR 118.9).
static const PickRule kAltSacrifice = {CostSlot::ALTERNATIVE, CostPart::SACRIFICE, "Sacrifice ", "",
    ActionCategory::SACRIFICE_PERMANENT, Zone::GRAVEYARD, " sacrifices ", "", PickRule::BY_COUNT, false, true, false};
// Additional sacrifice on the spell itself (CR 601.2f / 118.x): Natural Order — "As an additional
// cost to cast this spell, sacrifice a green creature."
static const PickRule kAdditionalSacrifice = {CostSlot::ADDITIONAL, CostPart::SACRIFICE, "Sacrifice ", "",
    ActionCategory::SACRIFICE_PERMANENT, Zone::GRAVEYARD, " sacrifices ", "", PickRule::ONE_IF_ANY, true, true, false};
// Escalate's discards (CR 702.120a / 601.2h): the cost's N cards for each mode beyond the first. The
// mode menu offered each extra mode only while enough cards were left.
static const PickRule kEscalateDiscard = {CostSlot::ESCALATE, CostPart::DISCARD, "Discard ", "",
    ActionCategory::DISCARD, Zone::GRAVEYARD, " discards ", " (escalate)", PickRule::BY_COUNT, false, true, true};
// Flashback's sacrifice (CR 702.34, Cabal Therapy).
static const PickRule kFlashbackSacrifice = {CostSlot::FLASHBACK, CostPart::SACRIFICE, "Sacrifice ", "",
    ActionCategory::SACRIFICE_PERMANENT, Zone::GRAVEYARD, " sacrifices ", "", PickRule::ONE_IF_ANY, true, true, false};
// Escape's ExileFromGrave (CR 702.139 / 601.2f), mandatory until met: OTHER cards of the graveyard
// until the set collectively has at least min_types distinct card types (CR 205.2), or exactly
// `count` of them (Uro: "Exile five other cards from your graveyard").
static const PickRule kEscapeExileTypes = {CostSlot::ESCAPE, CostPart::EXILE_FROM_GRAVEYARD, "Exile ", " from graveyard",
    ActionCategory::EXILE_FROM_YARD, Zone::EXILE, " exiles ", " from their graveyard", PickRule::BY_TYPES, false, true, false};
static const PickRule kEscapeExileCount = {CostSlot::ESCAPE, CostPart::EXILE_FROM_GRAVEYARD, "Exile ", " from graveyard",
    ActionCategory::EXILE_FROM_YARD, Zone::EXILE, " exiles ", " from their graveyard", PickRule::BY_COUNT, false, true, false};

// ── The paths ────────────────────────────────────────────────────────────────────────────────
// CR 601.2f-h, after the targets: first every non-mana cost item is chosen, then the mana is paid,
// then the life and energy and the moves. Nothing irreversible happens before the mana commits, so a
// proposal reversed then (CR 601.5 / 733.1) is undone by rewind_cast.
static const CostStep kCastSteps[] = {
    {CostStep::PICK, &kAltPitch},
    {CostStep::PICK, &kAltReturn},
    {CostStep::PICK, &kAltSacrifice},
    {CostStep::PICK, &kAdditionalSacrifice},
    {CostStep::PICK, &kEscalateDiscard},
    {CostStep::DELVE_COUNT},
    {CostStep::DELVE_PICK},
    {CostStep::PICK, &kFlashbackSacrifice},
    {CostStep::PICK, &kEscapeExileTypes},
    {CostStep::PICK, &kEscapeExileCount},
    {CostStep::SNAPSHOT},
    {CostStep::FLOAT_LEAVING},
    {CostStep::RECORD_MANA_SPENT},
    {CostStep::PAY_MANA},
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
    {CostStep::MOVE_ITEMS},
    {CostStep::DISCARD_SELF},
    {CostStep::DISCARD_HAND},
};

static CostSteps path_steps(CostPath path) {
    switch (path) {
        case CostPath::CAST:
            return {kCastSteps, sizeof(kCastSteps) / sizeof(kCastSteps[0])};
        case CostPath::ACTIVATION:
            break;
    }
    return {kActivationSteps, sizeof(kActivationSteps) / sizeof(kActivationSteps[0])};
}

CastCostStage cast_cost_stage(const CostRun &run) {
    if (run.step >= sizeof(kCastSteps) / sizeof(kCastSteps[0])) return CastCostStage::MANA;
    const CostStep &step = kCastSteps[run.step];
    if (step.op == CostStep::PICK)
        return step.pick->slot == CostSlot::ALTERNATIVE ? CastCostStage::ALTERNATIVE : CastCostStage::ADDITIONAL;
    return step.op == CostStep::DELVE_COUNT || step.op == CostStep::DELVE_PICK ? CastCostStage::ADDITIONAL
                                                                               : CastCostStage::MANA;
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
        case CostStep::DELVE_COUNT:
            return choose_delve_count(who, ctx, orderer);
        case CostStep::DELVE_PICK:
            return choose_delve_cards(who, ctx, orderer);
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
        case CostStep::RECORD_MANA_SPENT:
            // The cost's pip count is the total mana paid, after any delve/improvise reduction
            // already removed pips from it, for a ValidSA$ Spell.ManaSpent trigger to read at cast
            // time. A no-mana alternative cost leaves the 0 that defer_alternate_cost recorded.
            if (run.mana_pending) *who.mana_spent = static_cast<int>(run.mana.size());
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
        case CostStep::LIFE_X:
            pay_life_x(who);
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

// Choose what `rule`'s part moves, one pick at a time, the menu re-derived from the live zones at
// each pass: a pick stays where it is until the mana has committed.
static PayStatus pick_items(const PickRule &rule, const CostPayer &who, FrameCtx &ctx,
                            const std::shared_ptr<Orderer> &orderer) {
    CostRun &run = *who.run;
    const CostParts *cost = pick_cost(rule, who);
    if (!cost) return PayStatus::PAID;
    for (int left = picks_owed(rule, *cost, who); left > 0; left = picks_owed(rule, *cost, who)) {
        const std::vector<Entity> pool = pick_pool(rule, who, orderer);
        if (pool.empty() && rule.stops_when_empty) break;
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
static std::vector<Entity> pick_pool(const PickRule &rule, const CostPayer &who,
                                     const std::shared_ptr<Orderer> &orderer) {
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
        case CostPart::EXILE_FROM_HAND:
            // The color is read through effective_colors, which a color-setting static reaches.
            for (Entity e : orderer->get_hand(who.payer))
                if (e != who.source &&
                    (cost.pitch_color() == NO_COLOR || effective_colors(e).count(cost.pitch_color())))
                    pool.push_back(e);
            break;
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
    if (!rule.repeats_chosen)
        pool.erase(std::remove_if(pool.begin(), pool.end(), [&run](Entity e) { return chosen_as_cost(run, e); }),
                   pool.end());
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
           !can_pay_mana(who.payer, reduced, who.payee, orderer, /*has_delve=*/false, run.improvise)) {
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
        la.card_is_public = true;  // the spell being cast is public
        count_menu.push_back(la);
    }
    if (!ctx.resuming())
        game_log("Choose how many cards to exile via Delve (%zu-%zu):\n", min_needed, max_exiles);
    const int choice = ctx.ask(std::move(count_menu), who.payer, who.source);
    if (choice < 0 && decision_suspended()) return PayStatus::SUSPENDED;
    run.delve_count = min_needed + static_cast<size_t>(choice);
    return PayStatus::PAID;
}

// Delve stage 2 — which cards, one pick at a time; the candidate menu is re-derived from the live
// graveyard each pass minus the cards already picked. When the remaining candidates are all going to
// be exiled anyway (candidates <= picks left, including the single-candidate case) there is no real
// choice, so they are taken without a prompt. Each pick pays one GENERIC pip of the cost (CR 702.66a)
// and is a chosen cost item: the card is exiled, and recorded in cur_game.delve_exiled, as the items
// move.
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
    const PayStatus paid = pay_mana_cost(run.mana_payment, who.payer, run.mana, who.payee, ctx, orderer,
                                         /*has_delve=*/false, run.improvise, who.spent_colors);
    if (paid == PayStatus::PAID) coverage_cost("Mana");
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
    coverage_cost("Loyalty");
}

// Everything the picks chose moves, in the order chosen. Nothing here can fail: the choices were
// made against the live board and only this step moves anything. A permanent leaving here may make a
// chosen target illegal; the spell then fizzles at resolution (CR 608.2b), matching paper rules.
static void move_items(const CostPayer &who, const std::shared_ptr<Orderer> &orderer) {
    CostRun &run = *who.run;
    // A delve cast's exiles replace any a previous delve spell recorded (delve_exiled persists until
    // an etbCounter replacement consumes it).
    if (who.spell_has_delve) cur_game.delve_exiled.clear();
    for (const CostItem &item : run.items) {
        coverage_cost(cost_item_part(item));
        if (who.path == CostPath::CAST) game_log("%s\n", item.log.c_str());
        orderer->add_to_zone(false, item.entity, item.dest);
        if (who.path != CostPath::CAST) game_log("%s\n", item.log.c_str());
        if (item.delve) cur_game.delve_exiled.push_back(ObjectRef::of(item.entity));
    }
    run.items.clear();
}

// The --coverage cost part a chosen cost item pays.
static const char *cost_item_part(const CostItem &item) {
    if (item.delve) return "Delve";
    switch (item.kind) {
        case CostPart::RETURN:
            return "Return";
        case CostPart::EXILE_FROM_GRAVEYARD:
            return "ExileFromGrave";
        case CostPart::EXILE_FROM_HAND:
            return "ExileFromHand";
        case CostPart::DISCARD:
            return "Discard";
        default:
            return "Sac";
    }
}

// A variable life cost's announced X, paid with the rest of the life (Toxic Deluge).
static void pay_life_x(const CostPayer &who) {
    const CostRun &run = *who.run;
    if (run.life_x < 0) return;
    pay_life(global_coordinator.GetComponent<Player>(get_player_entity(who.payer)), run.life_x);
    game_log("%s pays %d life (X = %d)\n", player_name(who.payer).c_str(), run.life_x, run.life_x);
    coverage_cost("PayLifeX");
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
