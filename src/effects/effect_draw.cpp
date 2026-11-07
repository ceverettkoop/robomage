#include "effects.h"

#include <vector>

#include "../classes/game.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../queries/players.h"
#include "../resolution.h"
#include "../systems/orderer.h"
#include "../systems/replacement_effects.h"
#include "../svar_eval.h"
#include "../queries/affected.h"

extern Coordinator global_coordinator;

namespace effects {

bool draw_n_with_replacements(FrameCtx &ctx, std::shared_ptr<Orderer> orderer,
                              Zone::Ownership owner, DrawBatch &batch, size_t total,
                              Entity decision_source) {
    if (!batch.started) {
        batch.started = true;
        batch.remaining = total + static_cast<size_t>(replacement::draw_count_bonus(owner));
    }
    while (batch.body || batch.remaining > 0) {
        if (batch.body) {
            // What replaces the card resolves as a level of this frame, so a question it asks parks
            // like any other here (CR 614.6: no object is resolving, but the effect still runs).
            const Ability body = replacement::draw_body(batch);
            if (ctx.can_suspend()) {
                if (ctx.resolve_child(body, FrameLevel::REPLACEMENT, 0, static_cast<int>(batch.remaining),
                                      [](Ability &) {}, orderer) == ResolveStatus::SUSPENDED)
                    return false;
            } else {
                Ability blocking = body;
                resolve_ability(blocking, orderer);
            }
            batch.body = nullptr;
            batch.remaining--;
            continue;
        }
        // Per-draw ended bail, mirroring Orderer::draw's loop guard (a decked
        // draw ends the game mid-batch).
        if (cur_game.ended) return true;
        std::vector<replacement::DrawReplacementOption> opts;
        std::vector<LegalAction> menu = replacement::collect_draw_replacements(owner, &opts);
        if (menu.empty()) {
            orderer->perform_draw(owner);
            batch.remaining--;
            continue;
        }
        // One question per draw (CR 616.1, 702.52a), asked on the DRAWING player (who may differ
        // from the resolving controller — a draw can be forced by an opponent's effect), with the
        // resolving ability's source as the pending-decision source. A lone row is no question.
        int choice = 0;
        if (menu.size() > 1) {
            choice = ctx.ask(menu, owner, decision_source);
            if (choice < 0 && decision_suspended()) return false;
        }
        replacement::take_draw_option(*orderer, owner, opts[static_cast<size_t>(choice)], batch);
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
        // "Target player draws" (e.g. Deep Analysis) draws for the chosen target player;
        // otherwise the ability's controller draws ("you", CR 109.5) — captured when it went on
        // the stack and stable after the source changes control or leaves play (CR 608.2g), so a
        // reanimated Uro's draw goes to the reanimating player, not the card's owner. A DB$ Draw
        // with no Defined$ draws for "you" even though sub-ability chaining copies parent.target
        // into ab.target — Archon of Cruelty's DBDraw ("You draw a card") must draw for the
        // caster, not the sacrifice/discard target (affected_player).
        Zone::Ownership owner = affected_player(ab);
        // A Draw with no NumCards$ draws a single card (Forge default), e.g. Kozilek's
        // Command's "then draws a card" rider (DB$ Draw | Defined$ ParentTarget).
        size_t count = ab.def->amount > 0 ? ab.def->amount : 1;
        // A dynamic NumCards$ (The One Ring: NumCards$ X, X = Count$CardCounters.BURDEN — "draw a card
        // for each burden counter on it") is evaluated at resolution against the source permanent.
        if (!ab.def->dynamic_amount_expr.empty())
            count = evaluate_amount(ab.def->dynamic_amount_expr, owner, ab.source.lki_entity(), ab.target.get());
        rt.owners = affected_players(ab);
        rt.total = count;
        rt.init = true;
    }
    // The active player performs all of their draws first (CR 121.2c).
    for (; rt.owner_idx < rt.owners.size(); ++rt.owner_idx, rt.batch = DrawBatch{})
        if (!draw_n_with_replacements(ctx, orderer, rt.owners[rt.owner_idx], rt.batch, rt.total, ab.source.lki_entity()))
            return HandlerResult::SUSPENDED;
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
