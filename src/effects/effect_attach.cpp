#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/entry_info.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../queries/attachments.h"
#include "../queries/battlefield.h"
#include "../queries/characteristics.h"
#include "../queries/entry.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

HandlerResult attach(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    // Equip the source equipment to the remembered entity
    Entity equip_entity = ab.source.get();
    Entity target_creature = (ab.defined_remembered && !cur_game.remembered_entities.empty())
                                 ? cur_game.remembered_entities[0].get()
                                 : ab.target.get();

    if (ab.optional_choice && target_creature != 0) {
        // Optional$ True — "you MAY attach ..." (Cori-Steel Cutter's DBAttach). The ability's
        // controller decides at resolution through the shared yes/no menu, asked through ctx
        // so it can suspend (same "Decline:/Accept:" entries, same chooser repoint-and-
        // restore, the ability's source as the pending-decision source). A
        // single Shape A prompt: no rt — a resume re-derives the pure prelude and the next
        // ask consumes the latched answer.
        std::string prompt = "attach " + entity_name(equip_entity) + " to " +
                             entity_name(target_creature);
        std::vector<LegalAction> yn = optional_yesno_menu(prompt);
        int yc = ctx.ask(std::move(yn), ab.controller, ab.source.lki_entity());
        if (yc < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
        if (yc != 1) goto attach_done;
    }
    // A reanimation-then-attach chain (Pre-War Formalwear: ChangeZone Graveyard→Battlefield then
    // DB$ Attach Defined$ Remembered) resolves before the next state-based pass adds the moved
    // creature's Permanent component, so the target is on the battlefield (Zone) but has no
    // Permanent yet. Defer the attach: record it as a pending link consumed by
    // apply_permanent_components once the creature's Permanent is created (mirroring
    // EntryInfo::enters_tapped). The equipment already has its Permanent (it entered earlier).
    if (target_creature != 0 && global_coordinator.entity_has_component<Permanent>(equip_entity) &&
        !global_coordinator.entity_has_component<Permanent>(target_creature) &&
        global_coordinator.entity_has_component<Zone>(target_creature) &&
        global_coordinator.GetComponent<Zone>(target_creature).location == Zone::BATTLEFIELD) {
        entry_info(target_creature).attach_equipment = ObjectRef::of(equip_entity);
        game_log("Equipment will attach once the creature finishes entering.\n");
        goto attach_done;
    }
    if (target_creature != 0 && is_battlefield_permanent(equip_entity) &&
        global_coordinator.entity_has_component<Permanent>(target_creature)) {
        // An Equipment attaches only to something it can equip; otherwise it doesn't move
        // (CR 301.5b/301.5c). One that left the battlefield before an equip ability resolved
        // stays where it is.
        bool is_equipment = global_coordinator.entity_has_component<CardData>(equip_entity) &&
                            global_coordinator.GetComponent<CardData>(equip_entity).is_equipment;
        if (is_equipment && !equipment_can_equip(equip_entity, target_creature)) goto attach_done;
        global_coordinator.GetComponent<Permanent>(equip_entity).equipped_to = ObjectRef::of(target_creature);
        game_log("Equipment attached.\n");
    }
attach_done:;
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
