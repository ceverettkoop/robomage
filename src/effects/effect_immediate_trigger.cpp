#include "effects.h"

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
// Cleanup sub-ability runs regardless so remembered objects are cleared. Returns false so
// resolve() does not also chain the sub-abilities (we ran them here).
//
// An optional Cost$ (PayEnergy<N>) turns the trigger into a reflexive "you MAY pay {cost}.
// When you do, [Execute]" ability (CR 603.2c, Guide of Souls). The cost is offered only when
// the controller can actually pay it; on accept the cost is paid and the Execute chain runs,
// on decline (or when it can't be paid) the reflexive effect is skipped.
HandlerResult immediate_trigger(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
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
            std::string prompt = "Pay " + std::to_string(ab.energy_cost) + " energy";
            if (request_optional_yesno(ab.controller, prompt) && pay_energy(pl, ab.energy_cost)) {
                game_log("%s pays %d energy.\n", player_name(ab.controller).c_str(), ab.energy_cost);
            } else {
                fire = false;  // declined
            }
        }
    }

    for (auto sub : ab.subabilities) {
        sub.source = ab.source;
        sub.controller = ab.controller;
        if (sub.category == "Cleanup") {
            sub.resolve(orderer);
            continue;
        }
        if (!fire) continue;  // condition not met: skip the reflexive effect
        // A DB$ Pump sub with no pre-chosen target picks its own target inside the pump
        // handler (which also knows the ControlledBy-ParentTarget filter select_target
        // doesn't), so leave its target unset here and let the handler prompt (Guide of
        // Souls' put-counters-on-target-attacker, Cloak and Dagger's optional pump).
        if (sub.valid_tgts != "N_A" && sub.category != "Pump" && has_legal_targets(sub, orderer))
            select_target(sub, orderer, ab.controller);
        sub.resolve(orderer);
    }
    return HandlerResult::DONE_NO_SUBS;
}

}  // namespace effects
