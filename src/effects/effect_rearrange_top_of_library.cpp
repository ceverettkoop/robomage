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

static void place_rearranged_card(RearrangeRt &rt, size_t remaining_idx, std::shared_ptr<Orderer> orderer);

// Slots rt.remaining[remaining_idx]: appends it to rt.chosen_order (deepest
// first, so chosen_order[0] ends up deepest and the last entry on top), removes
// it from the candidates, and puts it on top of its owner's library, which also
// pushes it onto the known-top cache.
static void place_rearranged_card(RearrangeRt &rt, size_t remaining_idx, std::shared_ptr<Orderer> orderer) {
    Entity card = rt.remaining[remaining_idx];
    rt.chosen_order.push_back(card);
    rt.remaining.erase(rt.remaining.begin() + static_cast<long>(remaining_idx));
    orderer->add_to_zone(false, card, Zone::LIBRARY);
}

namespace effects {

HandlerResult rearrange_top_of_library(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    PendingDecisionScope pending_scope(ab.source);
    Zone::Ownership owner = global_coordinator.GetComponent<Zone>(ab.source).owner;

    // The looked-at slice is frozen once into the frame rt (pinned against
    // determinize). Slots are filled deepest first and each chosen card is put
    // on top of the library the moment it is chosen, so every later placement
    // lands above it and the final order is the slot order; the chosen card is
    // on the known-top cache (and so in the observation) for the remaining
    // picks. The unchosen cards stay in the library below the placed ones and
    // are addressed by entity through rt.remaining, never by position. The
    // slot-pick loop index, the forced-last placement, and the shuffle-y/n
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

        game_log("%s looks at the top %zu card(s) of %s library.\n", player_name(ab.controller).c_str(),
                 rt.lib.size(), owner_possessive(ab.controller, owner).c_str());
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
        // Record, drop from the candidates, and place in one step with no
        // suspension point between them: a resume re-enters at the next pick
        // (rt.pick advances in the loop increment), so a card is never placed twice.
        place_rearranged_card(rt, static_cast<size_t>(choice), orderer);
    }

    // The last card is forced onto the top slot — exactly once, even if the
    // shuffle prompt below suspends and this handler is re-entered.
    if (!rt.placed) {
        if (!rt.remaining.empty()) place_rearranged_card(rt, 0, orderer);
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
            game_log("%s shuffles %s library.\n", player_name(ab.controller).c_str(),
                     owner_possessive(ab.controller, owner).c_str());
        }
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
