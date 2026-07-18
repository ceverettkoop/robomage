#include "effects.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

HandlerResult rearrange_top_of_library(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    PendingDecisionScope pending_scope(ab.source);
    Zone::Ownership owner = global_coordinator.GetComponent<Zone>(ab.source).owner;

    // The looked-at slice is frozen once into the frame rt (pinned against
    // determinize — the already-slotted cards stay in the library until the
    // one-shot put-back below); the slot-pick loop index and the shuffle-y/n
    // stage persist so a resume re-enters the suspended decision.
    RearrangeRt local_rt;
    RearrangeRt &rt = ctx.can_suspend() ? ctx.rt<RearrangeRt>() : local_rt;
    if (!rt.init) {
        size_t num_cards = ab.amount;
        if (!ab.dynamic_amount_expr.empty())
            num_cards = evaluate_dynamic_amount(ab.dynamic_amount_expr, owner, orderer, ab.target);

        // looking at top n only
        rt.lib = orderer->get_library_top(owner, num_cards);
        rt.remaining = rt.lib;

        game_log("%s looks at the top %zu card(s) of their library.\n", player_name(owner).c_str(), rt.lib.size());
        rt.init = true;
    }
    size_t actual = rt.lib.size();

    // Player picks N-1 cards; the last is automatic
    for (; rt.pick + 1 < actual; rt.pick++) {
        // Arm-only slot header: a resume consumes the parked answer for this
        // slot without re-logging.
        if (!ctx.resuming())
            game_log("Choose which card goes %zu from top:\n", actual - rt.pick);
        std::vector<LegalAction> pick_actions;
        // Depth (0-indexed from the top) at which the card chosen this pick will
        // sit: the game_log above places it "actual - pick" from the top, so the
        // top card is depth 0 and the 3rd-from-top is depth 2 (e.g. Ponder). The
        // whole menu shares this depth — it is decision CONTEXT (which slot are we
        // filling), not intra-menu disambiguation (the cards already differ).
        int place_depth = static_cast<int>(actual - rt.pick - 1);
        for (auto card : rt.remaining) {
            auto &cd = global_coordinator.GetComponent<CardData>(card);
            LegalAction la(PASS_PRIORITY, card, cd.name);
            la.category = ActionCategory::TOP_LIBRARY;
            la.option_ordinal = place_depth;
            pick_actions.push_back(la);
        }
        // No priority repoint existed here — the resolving seat is ab.controller,
        // so seating the ask there is a no-op swap.
        int choice = ctx.ask(std::move(pick_actions), ab.controller, ab.source);
        if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
        rt.chosen_order.push_back(rt.remaining[static_cast<size_t>(choice)]);
        rt.remaining.erase(rt.remaining.begin() + choice);
    }

    // Put the cards back — exactly once, even if the shuffle prompt below
    // suspends and this handler is re-entered.
    if (!rt.placed) {
        // Last card is forced
        if (!rt.remaining.empty()) {
            rt.chosen_order.push_back(rt.remaining[0]);
        }
        // Put cards back: chosen_order[0] should end up on top, so place in reverse
        for (auto it : rt.chosen_order) {
            orderer->add_to_zone(false, it, Zone::LIBRARY);
        }
        rt.placed = true;
    }

    if (ab.may_shuffle) {
        std::vector<LegalAction> shuffle_actions = {
            LegalAction(PASS_PRIORITY, std::string("Don't shuffle")),
            LegalAction(PASS_PRIORITY, std::string("Shuffle")),
        };
        shuffle_actions[0].category = ActionCategory::DONT_SHUFFLE;
        shuffle_actions[1].category = ActionCategory::SHUFFLE;
        int shuffle_choice = ctx.ask(std::move(shuffle_actions), ab.controller, ab.source);
        if (shuffle_choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
        if (shuffle_choice == 1) {
            orderer->shuffle_library(owner);
            game_log("%s shuffles their library.\n", player_name(owner).c_str());
        }
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
