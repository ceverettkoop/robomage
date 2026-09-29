#include "effects.h"

#include <string>
#include <vector>

#include "../action_processor.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/player.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../queries/characteristics.h"
#include "../queries/filters.h"
#include "../queries/player_resources.h"
#include "../queries/players.h"
#include "../queries/spells.h"
#include "../systems/orderer.h"
#include "../resolution.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static void queue_reflexive_trigger(const Ability &parent, const Ability &execute);

namespace effects {

// ImmediateTrigger ("when you do ...") creates a reflexive triggered ability mid-resolution of
// its parent ability (Ajani's [0]: create a token; when you do, if you control a red permanent
// other than Ajani, deal damage). Reflexive triggers follow the rules for delayed triggered
// abilities (CR 603.12): the Execute$ ability triggers now if the ConditionPresent$ filter holds,
// and is put on the stack the next time a player would receive priority, with its targets chosen
// then (CR 603.3d) — so opponents can respond to it and Ward can counter it. The parent's own
// SubAbility$ chain (a Cleanup clearing remembered objects) runs here, as part of the parent.
//
// An optional Cost$ (PayEnergy<N>) turns the trigger into a reflexive "you MAY pay {cost}.
// When you do, [Execute]" ability (Guide of Souls). The cost is offered only when the
// controller can actually pay it, and paid during the parent's resolution; on decline (or when
// it can't be paid) nothing triggers.
//
// Suspendable: the fire decision (condition scan + energy yes/no) resolves once into
// ImmediateRt, and each parent-chain sub resolves as a persisted IMMEDIATE FrameLevel.
HandlerResult immediate_trigger(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    ImmediateRt local_rt;
    ImmediateRt &rt = ctx.can_suspend() ? ctx.rt<ImmediateRt>() : local_rt;

    if (!rt.init) {
        bool fire = ab.def->condition_present.empty();
        if (!fire) {
            for (auto e : orderer->mEntities) {
                if (permanent_matches_filter(e, ab.def->condition_present, MatchCtx{ab.controller, ab.source.lki_entity()})) {
                    fire = true;
                    break;
                }
            }
        }

        // Optional PayEnergy<N> cost: only fire the reflexive effect if the controller chooses to
        // pay and has the energy to do so (CR 122.1c). Decline / insufficient ⇒ skip Execute.
        if (fire && ab.def->energy_cost > 0) {
            Entity ctrl_entity = get_player_entity(ab.controller);
            auto &pl = global_coordinator.GetComponent<Player>(ctrl_entity);
            if (player_energy(pl) < ab.def->energy_cost) {
                fire = false;  // can't pay — not offered
            } else {
                // The request_optional_yesno menu, asked through ctx so it can
                // suspend (same "Decline:/Accept:" entries, same chooser
                // repoint-and-restore), with the ability's source as the
                // pending-decision source.
                std::string prompt = "Pay " + std::to_string(ab.def->energy_cost) + " energy";
                std::vector<LegalAction> yn = optional_yesno_menu(prompt);
                int yc = ctx.ask(std::move(yn), ab.controller, ab.source.lki_entity());
                if (yc < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
                if (yc == 1 && pay_energy(pl, ab.def->energy_cost)) {
                    game_log("%s pays %d energy.\n", player_name(ab.controller).c_str(), ab.def->energy_cost);
                } else {
                    fire = false;  // declined
                }
            }
        }
        rt.fire = fire;
        rt.init = true;
    }

    for (; rt.sub_idx < static_cast<int>(ab.subabilities.size()); ++rt.sub_idx) {
        Ability &stored = ab.subabilities[static_cast<size_t>(rt.sub_idx)];
        if (stored.def->from_delayed_execute) {
            if (rt.fire) queue_reflexive_trigger(ab, stored);
            continue;
        }
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
            sub.source = ab.source;
            sub.controller = ab.controller;
            resolve_ability(sub, orderer);
        }
    }
    return HandlerResult::DONE_NO_SUBS;
}

}  // namespace effects

// The Execute$ ability of `parent` as the reflexive triggered ability it creates: controlled by
// the parent's controller, with the parent's source (CR 603.12, 603.7d), queued for the next
// trigger placement.
static void queue_reflexive_trigger(const Ability &parent, const Ability &execute) {
    Ability reflexive = execute;
    reflexive.source = parent.source;
    reflexive.controller = parent.controller;
    reflexive.targeted_player = parent.player_target_for_subs();
    // X of the resolving parent, as a delayed trigger it creates would use (CR 107.3n).
    reflexive.x_paid = current_x_paid();
    cur_game.queue_trigger(reflexive, entity_name(parent.source.lki_entity()) + " reflexive trigger");
}
