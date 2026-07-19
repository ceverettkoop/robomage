#include "effects.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

HandlerResult sylvan_library(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    PendingDecisionScope pending_scope(ab.source);
    // Draw 2, then for each card drawn this turn still in hand, choose: pay 4 life or put on top
    Zone::Ownership ctrl = ab.controller;
    Entity ctrl_entity = (ctrl == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
    auto &pl = global_coordinator.GetComponent<Player>(ctrl_entity);

    // The draw-2 and the drawn-this-turn scan run once; the candidate set, the
    // picks so far, and both loop indices persist in the frame rt (candidates +
    // chosen cards pinned against determinize — the chosen ones may still be
    // put back on top).
    SylvanRt local_rt;
    SylvanRt &rt = ctx.can_suspend() ? ctx.rt<SylvanRt>() : local_rt;
    if (!rt.init) {
        // Draw 2 cards — each offering the dredge draw-replacement through ctx
        // (suspendable; rt.draws_done persists the batch progress). The dredge
        // ask arms with the ambient pending-decision source, which is
        // ab.source here (the handler-wide scope above), exactly what the
        // blocking prompt inside Orderer::draw saw.
        if (!draw_n_with_replacements(ctx, orderer, ctrl, rt.draws_done, 2))
            return HandlerResult::SUSPENDED;
        game_log("%s draws 2 cards (Sylvan Library)\n", player_name(ctrl).c_str());

        // Get cards drawn this turn that are still in hand
        for (auto e : pl.cards_drawn_this_turn) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location == Zone::HAND && z.owner == ctrl) {
                rt.drawn_in_hand.push_back(e);
            }
        }

        // Player chooses 2 cards (or fewer if not enough in hand)
        rt.to_choose = std::min(rt.drawn_in_hand.size(), static_cast<size_t>(2));
        rt.init = true;
    }

    for (; rt.pick < rt.to_choose; rt.pick++) {
        // Arm-only pick header (resuming-guarded like every pre-ask log).
        if (!ctx.resuming())
            game_log("Choose a card drawn this turn (%zu remaining):\n", rt.to_choose - rt.pick);
        std::vector<LegalAction> choose_actions;
        for (auto e : rt.drawn_in_hand) {
            // Skip already chosen
            bool already = false;
            for (auto c : rt.chosen) {
                if (c == e) {
                    already = true;
                    break;
                }
            }
            if (already) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            LegalAction la(PASS_PRIORITY, e, cd.name);
            la.category = ActionCategory::SYLVAN_CHOICE;
            choose_actions.push_back(la);
        }
        if (choose_actions.empty()) break;
        // No priority repoint existed here — the resolving seat is ab.controller,
        // so seating the ask there is a no-op swap.
        int choice = ctx.ask(choose_actions, ctrl, ab.source);
        if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
        rt.chosen.push_back(choose_actions[static_cast<size_t>(choice)].source_entity);
    }

    // For each chosen card: pay 4 life or put on top of library
    for (; rt.resolved < rt.chosen.size(); rt.resolved++) {
        Entity card = rt.chosen[rt.resolved];
        auto &cd = global_coordinator.GetComponent<CardData>(card);
        if (!ctx.resuming())
            game_log("For %s: pay 4 life or put on top of library?\n", cd.name.c_str());
        std::vector<LegalAction> pay_actions = {
            LegalAction(PASS_PRIORITY, std::string("Pay 4 life")),
            LegalAction(PASS_PRIORITY, std::string("Put on top of library")),
        };
        pay_actions[0].category = ActionCategory::SYLVAN_CHOICE;
        pay_actions[0].option_ordinal = 1;  // 1 = pay 4 life
        pay_actions[1].category = ActionCategory::SYLVAN_CHOICE;
        pay_actions[1].option_ordinal = 0;  // 0 = put on top of library
        int choice = ctx.ask(std::move(pay_actions), ctrl, ab.source);
        if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
        if (choice == 0) {
            pl.life_total -= 4;
            game_log("%s pays 4 life (now at %d)\n", player_name(ctrl).c_str(), pl.life_total);
        } else {
            orderer->add_to_zone(false, card, Zone::LIBRARY);
            game_log("%s puts %s on top of library\n", player_name(ctrl).c_str(), cd.name.c_str());
        }
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
