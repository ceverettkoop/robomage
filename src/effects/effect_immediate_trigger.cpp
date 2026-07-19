#include "effects.h"

#include <string>
#include <vector>

#include "../action_processor.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/player.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// ImmediateTrigger ("when you do ...") is a reflexive trigger created mid-resolution of
// its parent ability (Ajani's [0]: create a token, then if you control a red permanent
// other than Ajani, deal damage). We resolve it inline: gate on the ConditionPresent$
// filter, then run the Execute$ sub-ability (selecting a target if it needs one). Any
// Cleanup sub-ability runs regardless so remembered objects are cleared. Returns
// DONE_NO_SUBS so resolve() does not also chain the sub-abilities (we ran them here).
//
// An optional Cost$ (PayEnergy<N>) turns the trigger into a reflexive "you MAY pay {cost}.
// When you do, [Execute]" ability (CR 603.2c, Guide of Souls). The cost is offered only when
// the controller can actually pay it; on accept the cost is paid and the Execute chain runs,
// on decline (or when it can't be paid) the reflexive effect is skipped.
//
// Suspendable (Batch 8): the fire decision (condition scan + energy yes/no) resolves once
// into ImmediateRt — the scan is pure so re-running it before the energy answer latches is
// identical, but it must never re-run after a sub mutates the board — and each sub resolves
// as a persisted IMMEDIATE FrameLevel, its per-sub target selection on the shared
// run_target_select machine (RESOLUTION asker, selection held on the persisted parent's
// stored sub entry so a suspended pick survives).
HandlerResult immediate_trigger(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    ImmediateRt local_rt;
    ImmediateRt &rt = ctx.can_suspend() ? ctx.rt<ImmediateRt>() : local_rt;

    if (!rt.init) {
        bool fire = ab.condition_present.empty();
        if (!fire) {
            for (auto e : orderer->mEntities) {
                if (permanent_matches_filter(e, ab.condition_present, MatchCtx{ab.controller, ab.source})) {
                    fire = true;
                    break;
                }
            }
        }

        // Optional PayEnergy<N> cost: only fire the reflexive effect if the controller chooses to
        // pay and has the energy to do so (CR 122.1c). Decline / insufficient ⇒ skip Execute.
        if (fire && ab.energy_cost > 0) {
            Entity ctrl_entity =
                (ab.controller == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
            auto &pl = global_coordinator.GetComponent<Player>(ctrl_entity);
            if (player_energy(pl) < ab.energy_cost) {
                fire = false;  // can't pay — not offered
            } else {
                // The request_optional_yesno menu, asked through ctx so it can
                // suspend (same "Decline:/Accept:" entries, same chooser
                // repoint-and-restore, ambient pending-decision source).
                std::string prompt = "Pay " + std::to_string(ab.energy_cost) + " energy";
                std::vector<LegalAction> yn;
                LegalAction decline(PASS_PRIORITY, std::string("Decline: ") + prompt);
                decline.category = ActionCategory::OPTIONAL_YESNO;
                decline.option_ordinal = 0;  // 0 = decline
                yn.push_back(decline);
                LegalAction accept(PASS_PRIORITY, std::string("Accept: ") + prompt);
                accept.category = ActionCategory::OPTIONAL_YESNO;
                accept.option_ordinal = 1;  // 1 = accept
                yn.push_back(accept);
                int yc = ctx.ask(std::move(yn), ab.controller, cur_game.pending_decision_source);
                if (yc < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
                if (yc == 1 && pay_energy(pl, ab.energy_cost)) {
                    game_log("%s pays %d energy.\n", player_name(ab.controller).c_str(), ab.energy_cost);
                } else {
                    fire = false;  // declined
                }
            }
        }
        rt.fire = fire;
        rt.init = true;
    }

    for (; rt.sub_idx < static_cast<int>(ab.subabilities.size());
         ++rt.sub_idx, rt.tsel_done = false, rt.tsel = TargetSelectRT{}) {
        Ability &stored = ab.subabilities[static_cast<size_t>(rt.sub_idx)];
        bool is_cleanup = (stored.category == "Cleanup");
        if (!is_cleanup && !rt.fire) continue;  // condition not met: skip the reflexive effect
        stored.source = ab.source;
        stored.controller = ab.controller;
        // A DB$ Pump sub with no pre-chosen target picks its own target inside the pump
        // handler (which also knows the ControlledBy-ParentTarget filter select_target
        // doesn't), so leave its target unset here and let the handler prompt (Guide of
        // Souls' put-counters-on-target-attacker, Cloak and Dagger's optional pump).
        // Selection runs on the STORED sub entry (the persisted parent), so a suspended
        // pick resumes against the same in-flight ability; tsel_done gates it off once
        // complete so a suspension inside the sub's own resolve never re-selects.
        if (!is_cleanup && !rt.tsel_done && stored.valid_tgts != "N_A" &&
            stored.category != "Pump" && has_legal_targets(stored, orderer)) {
            if (ctx.can_suspend()) {
                ResolutionTargetAsker asker(ctx);
                if (run_target_select(stored, rt.tsel, asker, orderer, ab.controller) ==
                    TargetStatus::SUSPENDED)
                    return HandlerResult::SUSPENDED;
            } else {
                select_target(stored, orderer, ab.controller);
            }
        }
        rt.tsel_done = true;
        if (ctx.can_suspend()) {
            Ability *parent = &ab;
            auto bind = [parent](Ability &sub) {
                sub.source = parent->source;
                sub.controller = parent->controller;
            };
            if (ctx.resolve_child(stored, FrameLevel::IMMEDIATE, rt.sub_idx, -1, bind,
                                  orderer) == ResolveStatus::SUSPENDED)
                return HandlerResult::SUSPENDED;
        } else {
            Ability sub = stored;
            sub.resolve(orderer);
        }
    }
    return HandlerResult::DONE_NO_SUBS;
}

}  // namespace effects
