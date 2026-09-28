#include "effects.h"

#include <vector>

#include "../classes/game.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../queries/players.h"
#include "../systems/orderer.h"
#include "../systems/replacement_effects.h"

extern Coordinator global_coordinator;

namespace effects {

bool draw_n_with_replacements(FrameCtx &ctx, std::shared_ptr<Orderer> orderer,
                              Zone::Ownership owner, size_t &done, size_t total,
                              Entity decision_source) {
    for (; done < total; done++) {
        // Per-draw ended bail, mirroring Orderer::draw's loop guard (a decked
        // draw ends the game mid-batch).
        if (cur_game.ended) return true;
        std::vector<replacement::DrawReplacementOption> opts;
        std::vector<LegalAction> menu = replacement::collect_draw_replacements(owner, &opts);
        if (menu.empty()) {
            // No dredge applies — the promptless common case, exactly today's
            // draw_one with an empty replacement dispatch. The additive draw replacement
            // (CR 614.1/614.5, Quantum Riddler) bonus is applied inside the helper.
            orderer->perform_draw_with_bonus(owner);
            continue;
        }
        // One dredge question per draw (CR 702.52a / 614.1a), asked on the
        // DRAWING player (who may differ from the resolving controller — a
        // draw can be forced by an opponent's effect), with the resolving
        // ability's source as the pending-decision source.
        int choice = ctx.ask(menu, owner, decision_source);
        if (choice < 0 && decision_suspended()) return false;
        if (choice == 0) {
            orderer->perform_draw_with_bonus(owner);
        } else {
            orderer->apply_dredge(owner, opts[static_cast<size_t>(choice) - 1].source,
                                  opts[static_cast<size_t>(choice) - 1].mill);
        }
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
        // player; otherwise the ability's controller draws ("you", CR 109.5).
        // Redirect to the targeted player ONLY when this Draw itself declared the target —
        // its own ValidTgts$, or an explicit Defined$ naming the parent's target. A DB$ Draw
        // with no Defined$ means Forge's default of "You" (the controller) even though
        // sub-ability chaining copies parent.target into ab.target — Archon of Cruelty's
        // DBDraw ("You draw a card") must draw for the caster, not the sacrifice/discard
        // target. Mirrors the same guard in effect_lose_life.cpp.
        bool targets_player = (ab.def->valid_tgts != "N_A") || ab.def->defined == "Targeted" ||
                              ab.def->defined == "ParentTarget" || ab.def->defined == "Parent";
        Zone::Ownership owner;
        if (targets_player && ab.target.get() != 0 &&
            global_coordinator.entity_has_component<Player>(ab.target.get()))
            owner = seat_of_player(ab.target.get());
        else
            // The ability's controller, captured when it went on the stack and stable after the
            // source changes control or leaves play (CR 608.2g) — a reanimated Uro's draw goes to
            // the reanimating player, not the card's owner.
            owner = ab.controller;
        // A Draw with no NumCards$ draws a single card (Forge default), e.g. Kozilek's
        // Command's "then draws a card" rider (DB$ Draw | Defined$ ParentTarget).
        size_t count = ab.def->amount > 0 ? ab.def->amount : 1;
        // A dynamic NumCards$ (The One Ring: NumCards$ X, X = Count$CardCounters.BURDEN — "draw a card
        // for each burden counter on it") is evaluated at resolution against the source permanent.
        if (!ab.def->dynamic_amount_expr.empty())
            count = evaluate_dynamic_amount(ab.def->dynamic_amount_expr, owner, orderer, ab.target.get(), ab.source.lki_entity());
        rt.owner = owner;
        rt.total = count;
        rt.init = true;
    }
    if (!draw_n_with_replacements(ctx, orderer, rt.owner, rt.done, rt.total, ab.source.lki_entity()))
        return HandlerResult::SUSPENDED;
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
