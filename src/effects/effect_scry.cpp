#include "effects.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Scry N (CR 701.18): the chosen player looks at the top N cards of their library, then
// may put any number of them on the bottom of their library and the rest back on top in
// any order. Modeled as a per-card top-or-bottom choice from the top down; cards left on
// top keep their relative order (the optional reorder-among-kept is omitted as a
// simplification). The player is ValidTgts$ Player (ab.target); absent a target the
// source's controller scries. After scrying, any SubAbility$ chains with the same target
// (Kozilek's Command: "scries X, then draws a card" — DBDraw with Defined$ ParentTarget).
HandlerResult scry(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    PendingDecisionScope pending_scope(ab.source);
    Zone::Ownership owner;
    if (ab.target != 0 && global_coordinator.entity_has_component<Player>(ab.target))
        owner = (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    else if (global_coordinator.entity_has_component<Permanent>(ab.source))
        owner = global_coordinator.GetComponent<Permanent>(ab.source).controller;
    else
        owner = global_coordinator.GetComponent<Zone>(ab.source).owner;

    // The looked-at slice is frozen once into the frame rt (pinned against
    // determinize by pinned_entities()); the per-card loop index persists so a
    // resume re-enters the suspended keep/bottom decision. Bottom moves happen
    // per answer, exactly as before.
    ScryRt local_rt;
    ScryRt &rt = ctx.can_suspend() ? ctx.rt<ScryRt>() : local_rt;
    if (!rt.init) {
        size_t num = ab.amount;
        if (!ab.dynamic_amount_expr.empty())
            num = evaluate_dynamic_amount(ab.dynamic_amount_expr, owner, orderer, ab.target);
        if (num == 0) return HandlerResult::DONE_RUN_SUBS;

        rt.lib = orderer->get_library_top(owner, num);
        if (rt.lib.empty()) {
            game_log("%s's library is empty — nothing to scry.\n", player_name(owner).c_str());
            return HandlerResult::DONE_RUN_SUBS;
        }
        game_log("%s scries %zu.\n", player_name(owner).c_str(), rt.lib.size());
        rt.init = true;
    }

    // Decide top-to-bottom per card. Bottomed cards move to the library bottom; cards left
    // on top stay in place (their distance_from_top compacts as bottomed cards leave).
    for (; rt.idx < rt.lib.size(); ++rt.idx) {
        Entity card = rt.lib[rt.idx];
        auto &cd = global_coordinator.GetComponent<CardData>(card);
        // Arm-only per-card look line: a resume consumes the parked answer for
        // this card without re-logging.
        if (!ctx.resuming())
            game_log_private(owner, "Scry: top card is %s\n", cd.name.c_str());
        std::vector<LegalAction> scry_actions = {
            LegalAction(PASS_PRIORITY, card, std::string("Keep on top")),
            LegalAction(PASS_PRIORITY, card, std::string("Put on bottom")),
        };
        // Both options reference the same card, so they must differ by category to be
        // distinguishable to the semantic action resolver (a `top:`/`bottom:` --play spec
        // keys on category + card): keep = TOP_LIBRARY, bottom = BOTTOM_DECK_CARD.
        scry_actions[0].category = ActionCategory::TOP_LIBRARY;
        scry_actions[1].category = ActionCategory::BOTTOM_DECK_CARD;
        // No priority repoint existed here — the resolving seat is ab.controller,
        // so seating the ask there is a no-op swap.
        int choice = ctx.ask(std::move(scry_actions), ab.controller, ab.source);
        if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
        if (choice == 1) {
            orderer->add_to_zone(true, card, Zone::LIBRARY);
            game_log("%s puts a card on the bottom of their library.\n", player_name(owner).c_str());
        }
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
