#include "transform.h"

#include <string>

#include "cli_output.h"
#include "components/carddata.h"
#include "components/creature.h"
#include "components/damage.h"
#include "components/permanent.h"
#include "components/types.h"
#include "ecs/coordinator.h"
#include "game_queries.h"

extern Coordinator global_coordinator;

static bool face_has_type(const CardData &cd, const char *type) {
    for (auto &t : cd.types)
        if (t.kind == TYPE && t.name == type) return true;
    return false;
}

void set_permanent_face(Entity e, bool show_back) {
    if (!global_coordinator.entity_has_component<Permanent>(e)) return;
    if (!global_coordinator.entity_has_component<CardData>(e)) return;

    const CardData &front = global_coordinator.GetComponent<CardData>(e);
    const CardData *active = show_back ? front.backside.get() : &front;
    if (!active) return;  // requested back face but this card is single-faced

    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    const std::string old_name = perm.name;
    perm.transformed = show_back;
    perm.name = active->name;
    perm.types = active->types;

    const bool is_creature = face_has_type(*active, "Creature");
    const bool is_planeswalker = face_has_type(*active, "Planeswalker");

    // Rebuild the Creature/Damage components for the active face. Removing and
    // re-adding keeps cached P/T and damage consistent with the new face; +1/+1 and
    // other counters live on Permanent and survive the swap. recompute_pt folds in
    // any counter_pt_bonus on the next state-based pass.
    if (global_coordinator.entity_has_component<Creature>(e))
        global_coordinator.RemoveComponent<Creature>(e);
    if (global_coordinator.entity_has_component<Damage>(e))
        global_coordinator.RemoveComponent<Damage>(e);
    if (is_creature) {
        Creature cr;
        cr.base_power = static_cast<int>(active->power);
        cr.base_toughness = static_cast<int>(active->toughness);
        cr.keywords = active->keywords;
        recompute_pt(cr);
        global_coordinator.AddComponent(e, cr);
        Damage dmg;
        dmg.damage_counters = 0;
        global_coordinator.AddComponent(e, dmg);
    }

    // A planeswalker face enters with its printed loyalty (306.5b); a non-planeswalker
    // face carries no loyalty counters.
    if (is_planeswalker)
        perm.counters["LOYALTY"] = active->starting_loyalty;
    else
        perm.counters.erase("LOYALTY");

    // Reinstall the active face's activated abilities (e.g. the back-face loyalty
    // abilities) and static abilities, keyed to this entity. Triggered abilities are
    // matched from the active face elsewhere; only activated/static state lives on the
    // permanent.
    perm.abilities.clear();
    for (auto ab : active->abilities) {
        if (ab.ability_type != Ability::ACTIVATED) continue;
        ab.source = e;
        perm.abilities.push_back(ab);
    }
    perm.static_abilities.clear();
    for (auto &sa : active->static_abilities) perm.static_abilities.push_back(sa);

    game_log("%s transforms into %s!\n", old_name.c_str(), active->name.c_str());
}

void transform_permanent(Entity e) {
    if (!global_coordinator.entity_has_component<Permanent>(e)) return;
    const bool currently_back = global_coordinator.GetComponent<Permanent>(e).transformed;
    set_permanent_face(e, !currently_back);
}
