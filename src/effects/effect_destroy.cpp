#include "effects.h"

#include <string>

#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

static void destroy_single(Entity tgt, std::shared_ptr<Orderer> orderer) {
    // An "up to one target" destroy resolving with NO target chosen (tgt == 0) is a
    // deliberate no-op, not a vanished target — say so instead of the misleading
    // "no longer in play" line (reserved below for a real target that has gone away).
    if (tgt == 0) {
        game_log("Destroy: no target chosen\n");
        return;
    }
    if (!global_coordinator.entity_has_component<Zone>(tgt)) {
        game_log("Destroy: target is no longer in play\n");
        return;
    }
    auto &tz = global_coordinator.GetComponent<Zone>(tgt);
    if (tz.location != Zone::BATTLEFIELD) {
        game_log("Destroy: target is no longer on the battlefield\n");
        return;
    }
    std::string name = entity_name(tgt);
    // CR 702.12b: a permanent with indestructible can't be destroyed. The effect still
    // resolves; the permanent stays on the battlefield.
    if (is_indestructible(tgt)) {
        game_log("%s is indestructible — not destroyed\n", name.c_str());
        return;
    }
    orderer->add_to_zone(false, tgt, Zone::GRAVEYARD);
    game_log("%s is destroyed\n", name.c_str());
}

HandlerResult destroy(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    // Defined$ Self (The Tabernacle at Pendrell Vale's granted upkeep trigger: "destroy this
    // creature unless you pay {1}"): the effect acts on its own source. Bind it as the target so
    // the shared destroy/unless path below operates on the creature. Idempotent on a suspend/resume.
    if (ab.defined_self && ab.target == 0 && ab.targets.empty())
        ab.target = ab.source;

    // Pyroblast/Hydroblast destroy mode: only destroy if the target is the required
    // color. The spell still resolves (doing nothing) against a wrong-color permanent.
    if (!target_color_condition_met(ab, ab.target)) {
        std::string tname = entity_name(ab.target);
        game_log("%s is not the required color — not destroyed\n", tname.c_str());
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Conditional destroy (Fatal Push): check target CMC against threshold
    if (!ab.condition_present.empty() && ab.condition_present.find("cmcLEX") != std::string::npos) {
        // Evaluate X from dynamic_amount_expr (resolved at parse time to e.g. "Count$Revolt.4.2")
        int threshold = 2;  // default fallback
        if (!ab.dynamic_amount_expr.empty()) {
            threshold = static_cast<int>(evaluate_dynamic_amount(ab.dynamic_amount_expr, ab.controller, orderer, ab.target));
        }
        Entity tgt = ab.target;
        if (global_coordinator.entity_has_component<CardData>(tgt)) {
            int tgt_cmc = card_mana_value(global_coordinator.GetComponent<CardData>(tgt));
            if (tgt_cmc > threshold) {
                std::string tname = entity_name(tgt);
                game_log("%s has mana value %d (threshold %d) — not destroyed\n", tname.c_str(), tgt_cmc, threshold);
                return HandlerResult::DONE_RUN_SUBS;
            }
        }
    }

    // Unless-cost (The Tabernacle: "destroy this creature unless you pay {1}", CR 118.5 /
    // 603.2). The payer (UnlessPayer$ You ⇒ the ability's controller = the creature's controller)
    // may pay to prevent the destruction; run_unless_loop returns false when paid (don't destroy)
    // and true when declined / unaffordable. Handled for the single-target / defined_self form. The
    // MANA kind may suspend on a payment decision (checked before the return value); the arm-only
    // announcement is resume-guarded so a resume never re-logs it.
    if (ab.unless_generic_cost > 0) {
        Entity tgt = !ab.targets.empty() ? ab.targets[0] : ab.target;
        Zone::Ownership payer = (ab.unless_payer != Zone::UNKNOWN) ? ab.unless_payer : ab.controller;
        UnlessPayKind kind = ab.unless_cost_is_energy   ? UnlessPayKind::ENERGY
                           : ab.unless_cost_is_discard  ? UnlessPayKind::DISCARD
                           : ab.unless_cost_is_life      ? UnlessPayKind::LIFE
                                                         : UnlessPayKind::MANA;
        if (!ctx.resuming())
            game_log("%s may pay to prevent %s from being destroyed:\n",
                     player_name(payer).c_str(), entity_name(tgt).c_str());
        bool suspended = false;
        bool do_destroy = run_unless_loop(ab.unless_generic_cost, payer, orderer, tgt,
                                          ctx, suspended, kind);
        if (suspended) return HandlerResult::SUSPENDED;
        if (!do_destroy) return HandlerResult::DONE_RUN_SUBS;  // paid — nothing is destroyed
    }

    if (!ab.targets.empty()) {
        for (auto tgt : ab.targets) destroy_single(tgt, orderer);
    } else {
        destroy_single(ab.target, orderer);
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
