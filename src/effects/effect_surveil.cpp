#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"
#include "../svar_eval.h"

extern Coordinator global_coordinator;

namespace effects {

// Surveil N (CR 701.25a): look at the top N cards of your library, then put any number of them
// into your graveyard and the rest on top of your library in any order — look_and_split with the
// graveyard as the other pile.
HandlerResult surveil(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    PendingDecisionScope pending_scope(ab.source.lki_entity());
    Zone::Ownership controller = ab.controller;  // "you" = the ability's controller (CR 109.5)

    // The looked-at slice is frozen once; the shrinking `remaining` pool and the
    // `to_top` list of already-placed kept cards persist in the frame rt (both
    // pinned against determinize) so a suspended pick resumes against the
    // identical pool.
    LookSplitRt local_rt;
    LookSplitRt &rt = ctx.can_suspend() ? ctx.rt<LookSplitRt>() : local_rt;
    if (!rt.init) {
        size_t num = ab.def->amount;
        if (!ab.def->dynamic_amount_expr.empty())
            num = evaluate_dynamic_amount(ab.def->dynamic_amount_expr, controller, orderer, ab.target.get());
        if (num == 0) return HandlerResult::DONE_RUN_SUBS;  // CR 701.25c: surveil 0 is no event

        std::vector<Entity> looked = orderer->get_library_top(controller, num);
        if (looked.empty()) {
            game_log("%s's library is empty — nothing to surveil.\n", player_name(controller).c_str());
            return HandlerResult::DONE_RUN_SUBS;
        }

        game_log("%s surveils %zu.\n", player_name(controller).c_str(), looked.size());
        for (Entity card : looked) {
            auto &cd = global_coordinator.GetComponent<CardData>(card);
            game_log_private(controller, "Surveil: looking at %s\n", cd.name.c_str());
        }
        rt.remaining = looked;
        rt.init = true;
    }
    return look_and_split(rt, controller, LookSplitRest::GRAVEYARD, orderer, ctx, ab.source.lki_entity());
}

// See declaration in effects.h.
HandlerResult look_and_split(LookSplitRt &rt, Zone::Ownership looker, LookSplitRest rest,
                             std::shared_ptr<Orderer> orderer, FrameCtx &ctx, Entity source) {
    bool to_bottom = rest == LookSplitRest::LIBRARY_BOTTOM;
    while (!rt.remaining.empty()) {
        std::vector<LegalAction> actions;
        // First block: "put on top" for each remaining card; second block: the other pile.
        // Depth (0-indexed from the top) the card kept on top this round will sit
        // at: to_top[0] is topmost, so it is the count already kept. Shared by
        // the whole "put on top" block — decision context, not disambiguation.
        int place_depth = static_cast<int>(rt.to_top.size());
        for (Entity card : rt.remaining) {
            auto &cd = global_coordinator.GetComponent<CardData>(card);
            LegalAction la(PASS_PRIORITY, card, "Put " + cd.name + " on top of library");
            la.category = ActionCategory::TOP_LIBRARY;
            la.option_ordinal = place_depth;
            actions.push_back(la);
        }
        for (Entity card : rt.remaining) {
            auto &cd = global_coordinator.GetComponent<CardData>(card);
            LegalAction la(PASS_PRIORITY, card,
                           "Put " + cd.name + (to_bottom ? " on the bottom of library" : " into graveyard"));
            la.category = to_bottom ? ActionCategory::BOTTOM_DECK_CARD : ActionCategory::CHOOSE_CARD;
            actions.push_back(la);
        }

        // Asked of the looking player, who need not be the resolving seat (Kozilek's Command
        // makes a targeted opponent scry); the ask repoints priority at them for the decision only.
        int choice = ctx.ask(std::move(actions), looker, source);
        if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
        size_t idx = static_cast<size_t>(choice);
        bool on_top = idx < rt.remaining.size();
        size_t card_idx = on_top ? idx : idx - rt.remaining.size();
        Entity card = rt.remaining[card_idx];
        rt.remaining.erase(rt.remaining.begin() + static_cast<long>(card_idx));

        // Drop from the pool, place, and record in one step with no suspension point
        // between them: a resume re-asks only while the card is still in `remaining`,
        // so a card is never placed twice.
        if (on_top) {
            orderer->put_in_library_at_depth(card, rt.to_top.size());
            rt.to_top.push_back(card);
        } else if (to_bottom) {
            orderer->add_to_zone(true, card, Zone::LIBRARY);
            game_log("%s puts a card on the bottom of their library.\n", player_name(looker).c_str());
        } else {
            auto &cd = global_coordinator.GetComponent<CardData>(card);
            orderer->add_to_zone(false, card, Zone::GRAVEYARD);
            game_log("%s puts %s into their graveyard.\n", player_name(looker).c_str(), cd.name.c_str());
        }
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
