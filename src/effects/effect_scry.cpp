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
#include "../queries/players.h"
#include "../systems/orderer.h"
#include "../svar_eval.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Scry N (CR 701.22a): the chosen player looks at the top N cards of their library, then puts
// any number of them on the bottom of their library in any order and the rest back on top in any
// order — look_and_split with the library bottom as the other pile. The player is ValidTgts$
// Player (ab.target); absent a target the ability's controller scries. The scrying player sees
// the cards and makes every choice, so a targeted opponent decides for their own library. After
// scrying, any SubAbility$ chains with the same target (Kozilek's Command: "scries X, then draws
// a card" — DBDraw with Defined$ ParentTarget).
HandlerResult scry(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    PendingDecisionScope pending_scope(ab.source.lki_entity());
    Zone::Ownership owner;
    if (ab.target.get() != 0 && global_coordinator.entity_has_component<Player>(ab.target.get()))
        owner = seat_of_player(ab.target.get());
    else
        owner = ab.controller;  // "you" = the ability's controller (CR 109.5)

    // The looked-at slice is frozen once into the frame rt (pinned against determinize by
    // pinned_entities()), with the cards already placed, so a resume re-enters the suspended pick.
    LookSplitRt local_rt;
    LookSplitRt &rt = ctx.can_suspend() ? ctx.rt<LookSplitRt>() : local_rt;
    if (!rt.init) {
        size_t num = ab.def->amount;
        if (!ab.def->dynamic_amount_expr.empty())
            num = evaluate_dynamic_amount(ab.def->dynamic_amount_expr, owner, orderer, ab.target.get());
        if (num == 0) return HandlerResult::DONE_RUN_SUBS;

        std::vector<Entity> looked = orderer->get_library_top(owner, num);
        if (looked.empty()) {
            game_log("%s's library is empty — nothing to scry.\n", player_name(owner).c_str());
            return HandlerResult::DONE_RUN_SUBS;
        }
        game_log("%s scries %zu.\n", player_name(owner).c_str(), looked.size());
        for (Entity card : looked) {
            auto &cd = global_coordinator.GetComponent<CardData>(card);
            game_log_private(owner, "Scry: looking at %s\n", cd.name.c_str());
        }
        rt.remaining = looked;
        rt.init = true;
    }
    return look_and_split(rt, owner, LookSplitRest::LIBRARY_BOTTOM, orderer, ctx, ab.source.lki_entity());
}

}  // namespace effects
