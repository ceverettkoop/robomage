#include "effects.h"

#include <vector>

#include "../classes/game.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../systems/orderer.h"
#include "../systems/replacement_effects.h"

extern Coordinator global_coordinator;

namespace effects {

bool draw_n_with_replacements(FrameCtx &ctx, std::shared_ptr<Orderer> orderer,
                              Zone::Ownership owner, size_t &done, size_t total) {
    for (; done < total; done++) {
        // Per-draw ended bail, mirroring Orderer::draw's loop guard (a decked
        // draw ends the game mid-batch).
        if (cur_game.ended) return true;
        std::vector<replacement::DrawReplacementOption> opts;
        std::vector<LegalAction> menu = replacement::collect_draw_replacements(owner, &opts);
        if (menu.empty()) {
            // No dredge applies — the promptless common case, exactly today's
            // draw_one with an empty replacement dispatch.
            orderer->perform_draw(owner);
            continue;
        }
        // One dredge question per draw (CR 702.52a / 614.1a), asked on the
        // DRAWING player (who may differ from the resolving controller — a
        // draw can be forced by an opponent's effect) with the ambient
        // pending-decision source — the blocking dispatch_draw prompt ran
        // under whatever scope the calling handler held (sylvan's ab.source
        // scope; none for a plain Draw resolution).
        int choice = ctx.ask(menu, owner, cur_game.pending_decision_source);
        if (choice < 0 && decision_suspended()) return false;
        if (choice == 0)
            orderer->perform_draw(owner);
        else
            orderer->apply_dredge(owner, opts[static_cast<size_t>(choice) - 1].source,
                                  opts[static_cast<size_t>(choice) - 1].mill);
    }
    return true;
}

HandlerResult draw(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    // The drawing player and the draw count resolve ONCE into the frame rt: a
    // dynamic NumCards$ (The One Ring's per-BURDEN-counter count) must not be
    // re-evaluated after a dredge mill mutates the counted state, and the
    // resumed batch re-enters at the parked draw.
    DrawRt local_rt;
    DrawRt &rt = ctx.can_suspend() ? ctx.rt<DrawRt>() : local_rt;
    if (!rt.init) {
        // "Target player draws" (e.g. Deep Analysis) draws for the chosen target
        // player; otherwise the effect's controller (source owner) draws.
        // Redirect to the targeted player ONLY when this Draw itself declared the target —
        // its own ValidTgts$, or an explicit Defined$ naming the parent's target. A DB$ Draw
        // with no Defined$ means Forge's default of "You" (the controller) even though
        // sub-ability chaining copies parent.target into ab.target — Archon of Cruelty's
        // DBDraw ("You draw a card") must draw for the caster, not the sacrifice/discard
        // target. Mirrors the same guard in effect_lose_life.cpp.
        bool targets_player = (ab.valid_tgts != "N_A") || ab.defined == "Targeted" ||
                              ab.defined == "ParentTarget" || ab.defined == "Parent";
        Zone::Ownership owner;
        if (targets_player && ab.target != 0 && global_coordinator.entity_has_component<Player>(ab.target))
            owner = (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
        else if (ab.source != 0 && global_coordinator.entity_has_component<Zone>(ab.source))
            owner = global_coordinator.GetComponent<Zone>(ab.source).owner;
        else
            // The source is gone (e.g. a Clue token sacrificed as part of the activation cost before
            // this Draw resolves): fall back to the ability's controller, captured when it went on the
            // stack and stable even after the source leaves play (CR 608.2g). For "draw a card" the
            // drawing player is the ability's controller, which equals the source owner in every case
            // where the source still exists.
            owner = ab.controller;
        // A Draw with no NumCards$ draws a single card (Forge default), e.g. Kozilek's
        // Command's "then draws a card" rider (DB$ Draw | Defined$ ParentTarget).
        size_t count = ab.amount > 0 ? ab.amount : 1;
        // A dynamic NumCards$ (The One Ring: NumCards$ X, X = Count$CardCounters.BURDEN — "draw a card
        // for each burden counter on it") is evaluated at resolution against the source permanent.
        if (!ab.dynamic_amount_expr.empty())
            count = evaluate_dynamic_amount(ab.dynamic_amount_expr, owner, orderer, ab.target, ab.source);
        rt.owner = owner;
        rt.total = count;
        rt.init = true;
    }
    if (!draw_n_with_replacements(ctx, orderer, rt.owner, rt.done, rt.total))
        return HandlerResult::SUSPENDED;
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
