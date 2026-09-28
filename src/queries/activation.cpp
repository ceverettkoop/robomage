#include "activation.h"

#include "../classes/game.h"
#include "../ecs/coordinator.h"
#include "attachments.h"
#include "battlefield.h"
#include "combat.h"
#include "types.h"

int ability_resolutions_this_turn(Entity source) {
    const int *n = cur_game.ability_resolution_counts.find(source);
    return n ? *n : 0;
}

bool controller_has_metalcraft(Zone::Ownership controller, const std::set<Entity> &entities) {
    int artifacts = 0;
    for (auto e : battlefield_permanents(entities, controller))
        if (type_set_has(global_coordinator.GetComponent<Permanent>(e).types, "Artifact"))
            if (++artifacts >= 3) return true;
    return false;
}

bool activation_condition_met(const Ability &ab, Zone::Ownership controller,
                              const std::set<Entity> &entities, Entity source) {
    if (ab.def.activation_condition.empty()) return true;
    if (ab.def.activation_condition == "Metalcraft")
        return controller_has_metalcraft(controller, entities);
    // NotMonstrous (CR 701.37a): a monstrosity ability is legal only while its source isn't
    // already monstrous. Keyed on the source permanent's is_monstrous designation. Installed
    // automatically by parse_put_counter for any Monstrosity$ ability.
    if (ab.def.activation_condition == "NotMonstrous")
        return source != 0 &&
               global_coordinator.entity_has_component<Permanent>(source) &&
               !global_coordinator.GetComponent<Permanent>(source).is_monstrous;
    // CanEquip (CR 702.6a, 301.5c): an equip ability (K:Equip / K:Reconfigure) is activatable
    // only while its Equipment can equip some creature its controller controls.
    if (ab.def.activation_condition == "CanEquip")
        return source != 0 && !equip_candidates(source, controller, entities).empty();
    // Attached (CR 702.151a): reconfigure's unattach ability — "Activate only if this permanent
    // is attached to a creature".
    if (ab.def.activation_condition == "Attached")
        return source != 0 && global_coordinator.entity_has_component<Permanent>(source) &&
               global_coordinator.GetComponent<Permanent>(source).equipped_to.get() != 0;
    return false;
}

bool activation_source_ready(const Ability &ab, Entity source, Zone::Ownership controller,
                             const std::set<Entity> &entities) {
    if (!activation_condition_met(ab, controller, entities, source)) return false;
    if (ab.def.tap_cost) {
        if (!global_coordinator.entity_has_component<Permanent>(source)) return false;
        if (global_coordinator.GetComponent<Permanent>(source).is_tapped) return false;
        if (is_summoning_sick(source)) return false;
    }
    return ab.def.activation_limit <= 0 || ab.activations_this_turn < ab.def.activation_limit;
}
