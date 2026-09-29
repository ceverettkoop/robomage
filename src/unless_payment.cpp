#include "unless_payment.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "classes/action.h"
#include "classes/match_state.h"
#include "cli_output.h"
#include "components/carddata.h"
#include "components/player.h"
#include "ecs/coordinator.h"
#include "input_logger.h"
#include "mana_system.h"
#include "queries/player_resources.h"
#include "queries/players.h"
#include "resolution_frame.h"
#include "systems/orderer.h"

extern Coordinator global_coordinator;

// The "unless they discard N cards" flavor of run_unless_loop (CR 701.8), suspendable via
// FrameCtx. Forward-declared per CLAUDE.md.
static bool run_discard_unless(size_t count, Zone::Ownership controller,
                               std::shared_ptr<Orderer> orderer, Entity decision_source,
                               FrameCtx &ctx, bool &suspended, const UnlessSubject &subject);

// Offer `controller` the choice to discard `count` card(s) of their choice from hand to pay an
// unless-cost (CR 701.8). Returns true if the cost was NOT paid (they declined or have too few
// cards), i.e. the spell should be countered. Reusable by any "unless they discard N cards"
// effect; the chosen cards go to the graveyard (a public zone — record the reveal in the belief
// state). Asks are seated on `controller` through ctx.ask (which repoints and restores priority
// around each prompt, reproducing the old caller-side swap). May suspend: `suspended` is set and
// the return value is meaningless (check it FIRST). The yes/no answer and the discard-count
// progress persist in the level's UnlessRt; the per-discard menu is intentionally rebuilt from
// the LIVE hand each ask — between asks the hand only shrinks by the discards themselves.
static bool run_discard_unless(size_t count, Zone::Ownership controller,
                               std::shared_ptr<Orderer> orderer, Entity decision_source,
                               FrameCtx &ctx, bool &suspended, const UnlessSubject &subject) {
    suspended = false;
    UnlessRt local_rt;
    UnlessRt &rt = ctx.can_suspend() ? ctx.rt<UnlessRt>() : local_rt;

    if (!rt.pay_answered) {
        std::vector<Entity> hand = orderer->get_hand(controller);
        bool can_pay = hand.size() >= count;  // CR 701.8: must discard the full count or none

        std::vector<LegalAction> actions;
        size_t pay_idx = actions.size();
        if (can_pay) actions.push_back(unless_pay_action(subject, UnlessPayKind::DISCARD, count, nullptr));
        size_t decline_idx = actions.size();
        actions.push_back(unless_decline_action(subject, UnlessPayKind::DISCARD));

        int choice = ctx.ask(actions, controller, decision_source);
        if (choice < 0 && decision_suspended()) {
            suspended = true;
            return false;
        }
        if (!can_pay || choice == static_cast<int>(decline_idx)) {
            (void)pay_idx;
            return true;  // countered
        }
        rt.pay_answered = true;
    }

    // Pay: the payer chooses `count` distinct cards from hand to discard.
    for (; rt.discards_done < count; ++rt.discards_done) {
        std::vector<Entity> cur_hand = orderer->get_hand(controller);
        if (cur_hand.empty()) break;
        std::vector<LegalAction> dactions;
        for (auto e : cur_hand) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            LegalAction la(PASS_PRIORITY, e, cd.name);
            la.category = ActionCategory::DISCARD;
            dactions.push_back(la);
        }
        int dchoice = ctx.ask(dactions, controller, decision_source);
        if (dchoice < 0 && decision_suspended()) {
            suspended = true;
            return false;
        }
        Entity chosen = dactions[static_cast<size_t>(dchoice)].source_entity;
        auto &cd = global_coordinator.GetComponent<CardData>(chosen);
        game_log("%s discards %s — %s\n", player_name(controller).c_str(), cd.name.c_str(),
                 unless_outcome_text(subject, /*paid=*/true).c_str());
        // The discarded card enters a public zone — record its identity in the belief state.
        mark_card_revealed(chosen, controller);
        orderer->add_to_zone(false, chosen, Zone::GRAVEYARD);
    }
    return false;  // paid
}

// Returns true if the unless-cost was not paid (controller declined or couldn't pay).
// kind: how the unless-cost is paid — {cost} generic mana or exact pips (default), `cost` life
// (Ward—Pay N life, CR 702.21), discard `cost` card(s) (Reality Smasher, CR 701.8), or `cost`
// energy. A life payment is only offered when the payer's life total >= cost (CR 119.4 — a player
// can't pay more life than they have). Every pay/decline entry is worded by the shared
// choice_labels builders from `subject` (the governed effect and its object).
// The payer decides whether to pay, not the ability's controller: every kind seats its asks
// on `controller` through ctx.ask (which repoints and restores priority around the prompt exactly
// as the old manual swap did) and may suspend — `suspended` is set and the return value is
// meaningless (check it FIRST). The MANA kind is a live-menu loop (Shape C): its pay/decline menu
// interleaves per-source tap-for-mana actions rebuilt from live state each pass, so a chosen tap
// executes at consume time, floats mana, and the loop re-arms the rebuilt menu.
bool run_unless_loop(
    size_t cost, Zone::Ownership controller, std::shared_ptr<Orderer> orderer, Entity paid_for,
    Entity decision_source, FrameCtx &ctx, bool &suspended, const UnlessSubject &subject,
    UnlessPayKind kind, const ManaValue *cost_pips) {
    suspended = false;

    if (kind == UnlessPayKind::DISCARD) {
        return run_discard_unless(cost, controller, orderer, decision_source, ctx, suspended, subject);
    }

    if (kind == UnlessPayKind::LIFE) {
        auto &payer = global_coordinator.GetComponent<Player>(get_player_entity(controller));
        bool can_pay = can_pay_life(payer, static_cast<int>(cost));

        std::vector<LegalAction> unless_actions;
        size_t pay_idx = unless_actions.size();
        if (can_pay) unless_actions.push_back(unless_pay_action(subject, kind, cost, nullptr));
        size_t decline_idx = unless_actions.size();
        unless_actions.push_back(unless_decline_action(subject, kind));

        int choice = ctx.ask(std::move(unless_actions), controller, decision_source);
        if (choice < 0 && decision_suspended()) {
            suspended = true;
            return false;
        }
        if (can_pay && choice == static_cast<int>(pay_idx)) {
            pay_life(payer, static_cast<int>(cost));
            game_log("%s pays %zu life — %s\n", player_name(controller).c_str(), cost,
                     unless_outcome_text(subject, /*paid=*/true).c_str());
            return false;
        }
        (void)decline_idx;
        return true;
    }

    if (kind == UnlessPayKind::ENERGY) {
        // Pay N energy ({E}, CR 122.1c) or the prevented effect happens (Static Prison's
        // "sacrifice CARDNAME unless you pay {E}"). Energy lives as an "ENERGY" counter on the
        // payer; pay_energy gates on having enough (CR 119.4-analogue for a resource cost).
        auto &payer = global_coordinator.GetComponent<Player>(get_player_entity(controller));
        bool can_pay = player_energy(payer) >= static_cast<int>(cost);

        std::vector<LegalAction> unless_actions;
        size_t pay_idx = unless_actions.size();
        if (can_pay) unless_actions.push_back(unless_pay_action(subject, kind, cost, nullptr));
        size_t decline_idx = unless_actions.size();
        unless_actions.push_back(unless_decline_action(subject, kind));

        int choice = ctx.ask(std::move(unless_actions), controller, decision_source);
        if (choice < 0 && decision_suspended()) {
            suspended = true;
            return false;
        }
        if (can_pay && choice == static_cast<int>(pay_idx)) {
            pay_energy(payer, static_cast<int>(cost));
            game_log("%s pays %zu energy — %s\n", player_name(controller).c_str(), cost,
                     unless_outcome_text(subject, /*paid=*/true).c_str());
            return false;
        }
        (void)decline_idx;
        return true;
    }

    // MANA kind (Daze / Mana Leak / Ward {N}): pay-{N}-unless with tap-for-mana
    // sub-choices. A live-menu loop (Shape C) with no persisted progress: the
    // menu is rebuilt from live state every pass — floated mana (Player.mana,
    // snapshot-covered) changes which cost-bearing sources are offered and
    // whether "Pay" is affordable — so a resume re-enters, rebuilds the
    // identical menu, and the ask consumes the latched answer (its size assert
    // guards drift). A chosen tap action executes its mana ability at consume
    // time (mana floats), then the loop re-arms the rebuilt menu. Asks are
    // seated on `controller` through ctx.ask, replacing the old manual seat
    // swap around the whole loop.
    // Exact colored pips (Chain Lightning: {R}{R}) when supplied; else `cost` generic pips.
    std::multiset<Colors> cond_cost;
    if (cost_pips && !cost_pips->empty())
        cond_cost = *cost_pips;
    else
        for (size_t i = 0; i < cost; i++) cond_cost.insert(GENERIC);

    while (true) {
        std::vector<LegalAction> unless_actions = collect_mana_legal_actions(controller, orderer);
        // Drop cost-bearing mana sources (Talon Gates' {1}{T}) whose activation cost the
        // FLOATING pool can't cover right now: the tap-a-source loop below pays activation
        // costs only from mana already floating, so offering such a source would either
        // produce free mana or (refused) re-offer the same menu forever to a deterministic
        // machine-mode agent. The menu is rebuilt each iteration, so the source reappears
        // as soon as enough mana has been floated.
        unless_actions.erase(
            std::remove_if(unless_actions.begin(), unless_actions.end(),
                [&](const LegalAction &la) {
                    return !la.ability.def->activation_mana_cost.empty() &&
                           !can_afford(controller, la.ability.def->activation_mana_cost);
                }),
            unless_actions.end());

        bool can_pay = can_afford(controller, cond_cost);
        size_t pay_idx = unless_actions.size();
        if (can_pay) unless_actions.push_back(unless_pay_action(subject, kind, cost, cost_pips));
        size_t decline_idx = unless_actions.size();
        unless_actions.push_back(unless_decline_action(subject, kind));

        // Passed as an lvalue: the tap branch below still needs the menu to map
        // the chosen action.
        int choice = ctx.ask(unless_actions, controller, decision_source);
        if (choice < 0 && decision_suspended()) {
            suspended = true;
            return false;
        }

        if (choice == static_cast<int>(decline_idx)) {
            return true;
        }

        if (can_pay && choice == static_cast<int>(pay_idx)) {
            spend_mana(controller, cond_cost, paid_for);
            game_log("%s pays %s — %s\n", player_name(controller).c_str(),
                     unless_cost_text(kind, cost, cost_pips).c_str(),
                     unless_outcome_text(subject, /*paid=*/true).c_str());
            return false;
        }

        if (choice >= 0 && choice < static_cast<int>(pay_idx)) {
            auto &chosen = unless_actions[static_cast<size_t>(choice)];
            Entity land = chosen.source_entity;
            auto &pl = global_coordinator.GetComponent<Player>(get_player_entity(controller));
            // Canonical activation against the player's real (floating) pool: pays a
            // cost-bearing source's activation cost (Talon Gates' {1}{T}) from the pool —
            // refusing cleanly (no tap, no mana) while the pool can't cover it, so the
            // re-prompt lets the player float mana from a cost-free source first — and
            // applies the full mana-ability semantics: tap only tap-cost sources, sac_self
            // (Lotus Petal), life costs, dynamic amounts, SubAbility$ riders (Ancient
            // Tomb's damage), ProduceMana replacements, and activation counters.
            if (!activate_mana_source(land, chosen.ability, controller, orderer, pl.mana, pl,
                                      /*commit=*/true, ManaLogStyle::TAPPED_SYMBOL)) {
                game_log("Cannot pay that ability's activation cost — tap another source for mana first.\n");
                continue;
            }
        }
    }
}
