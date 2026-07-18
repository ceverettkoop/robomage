#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../input_logger.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

HandlerResult attach(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    // Equip the source equipment to the remembered entity
    Entity equip_entity = ab.source;
    Entity target_creature = (ab.defined_remembered && !cur_game.remembered_entities.empty())
                                 ? cur_game.remembered_entities[0]
                                 : ab.target;

    if (ab.optional_choice && target_creature != 0) {
        // Optional$ True — "you MAY attach ..." (Cori-Steel Cutter's DBAttach). The ability's
        // controller decides at resolution through the shared yes/no prompt, which seats the
        // decision on them (like every other resolution-time "you may" confirmation).
        if (!request_optional_yesno(ab.controller, "attach " + entity_name(equip_entity) +
                                                       " to " + entity_name(target_creature)))
            goto attach_done;
    }
    // A reanimation-then-attach chain (Pre-War Formalwear: ChangeZone Graveyard→Battlefield then
    // DB$ Attach Defined$ Remembered) resolves before the next state-based pass adds the moved
    // creature's Permanent component, so the target is on the battlefield (Zone) but has no
    // Permanent yet. Defer the attach: record it as a pending link consumed by
    // apply_permanent_components once the creature's Permanent is created (mirroring
    // pending_enters_tapped). The equipment already has its Permanent (it entered earlier).
    if (target_creature != 0 && global_coordinator.entity_has_component<Permanent>(equip_entity) &&
        !global_coordinator.entity_has_component<Permanent>(target_creature) &&
        global_coordinator.entity_has_component<Zone>(target_creature) &&
        global_coordinator.GetComponent<Zone>(target_creature).location == Zone::BATTLEFIELD) {
        cur_game.pending_attach[target_creature] = equip_entity;
        game_log("Equipment will attach once the creature finishes entering.\n");
        goto attach_done;
    }
    if (target_creature != 0 && global_coordinator.entity_has_component<Permanent>(equip_entity) &&
        global_coordinator.entity_has_component<Permanent>(target_creature)) {
        auto &eq_perm = global_coordinator.GetComponent<Permanent>(equip_entity);
        // Detach from previous creature
        if (eq_perm.equipped_to != 0 && global_coordinator.entity_has_component<Permanent>(eq_perm.equipped_to)) {
            global_coordinator.GetComponent<Permanent>(eq_perm.equipped_to).equipped_by = 0;
        }
        eq_perm.equipped_to = target_creature;
        global_coordinator.GetComponent<Permanent>(target_creature).equipped_by = equip_entity;
        game_log("Equipment attached.\n");
    }
attach_done:;
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
