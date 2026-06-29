#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

bool attach(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    // Equip the source equipment to the remembered entity
    Entity equip_entity = ab.source;
    Entity target_creature = (ab.defined_remembered && !cur_game.remembered_entities.empty())
                                 ? cur_game.remembered_entities[0]
                                 : ab.target;

    if (ab.optional) {
        // Ask the controller whether to attach
        game_log("Attach equipment to token? (0=No 1=Yes)\n");
        std::vector<LegalAction> attach_actions = {
            LegalAction(PASS_PRIORITY, std::string("No")),
            LegalAction(PASS_PRIORITY, std::string("Yes")),
        };
        attach_actions[0].category = ActionCategory::OPTIONAL_YESNO;
        attach_actions[1].category = ActionCategory::OPTIONAL_YESNO;
        int choice = InputLogger::instance().get_input(attach_actions);
        if (choice == 0) goto attach_done;
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
    return true;
}

}  // namespace effects
