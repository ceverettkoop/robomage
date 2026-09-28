#include "transform.h"

#include <algorithm>
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

static bool face_has_type(const CardData &cd, const char *type);
static void sync_face_creature_state(Entity e, const CardData &face);
static void swap_face_abilities(Entity e, Permanent &perm, const CardData &old_face,
                                const CardData &new_face);

static bool face_has_type(const CardData &cd, const char *type) {
    for (auto &t : cd.types)
        if (t.kind == TYPE && t.name == type) return true;
    return false;
}

// The permanent stays the same object across a transform (CR 712.18), so its Creature and
// Damage components are kept: marked damage, combat status, until-end-of-turn pumps and
// keyword grants, and set-P/T effects all carry over. Only the printed base P/T changes to the
// new face's (CR 712.8d/e); the keyword list is rebuilt from the new face on the next static
// pass (gather_active_statics). A face that isn't a creature drops the components, which also
// removes the permanent from combat (CR 506.4); a noncreature that transforms into a creature
// gets fresh ones.
static void sync_face_creature_state(Entity e, const CardData &face) {
    if (!face_has_type(face, "Creature")) {
        if (global_coordinator.entity_has_component<Creature>(e))
            global_coordinator.RemoveComponent<Creature>(e);
        if (global_coordinator.entity_has_component<Damage>(e))
            global_coordinator.RemoveComponent<Damage>(e);
        return;
    }
    if (!global_coordinator.entity_has_component<Creature>(e)) {
        Creature cr;
        cr.keywords = face.keywords;
        global_coordinator.AddComponent(e, cr);
    }
    if (!global_coordinator.entity_has_component<Damage>(e)) {
        Damage dmg;
        dmg.damage_counters = 0;
        global_coordinator.AddComponent(e, dmg);
    }
    auto &cr = global_coordinator.GetComponent<Creature>(e);
    cr.base_power = static_cast<int>(face.power);
    cr.base_toughness = static_cast<int>(face.toughness);
    recompute_pt(cr);
}

// Replace the old face's printed activated abilities with the new face's (e.g. the back-face
// loyalty abilities), keyed to this entity. Abilities from any other source — a static grant,
// an Animate grant, a land type's intrinsic mana ability — are left in place. Triggered
// abilities are matched from the active face at trigger time; statics are the face's own.
static void swap_face_abilities(Entity e, Permanent &perm, const CardData &old_face,
                                const CardData &new_face) {
    auto from_face = [](const CardData &face, Ability &ab) {
        if (ab.ability_type != Ability::ACTIVATED || ab.granted_by_static != 0) return false;
        for (const auto &printed : face.abilities)
            if (printed.ability_type == Ability::ACTIVATED && ab.identical_activated_ability(printed))
                return true;
        return false;
    };
    perm.abilities.erase(std::remove_if(perm.abilities.begin(), perm.abilities.end(),
                                        [&](Ability &ab) { return from_face(old_face, ab); }),
                         perm.abilities.end());
    for (auto ab : new_face.abilities) {
        if (ab.ability_type != Ability::ACTIVATED) continue;
        ab.source = ObjectRef::of(e);
        perm.abilities.push_back(ab);
    }
    perm.static_abilities = new_face.static_abilities;
}

void set_permanent_face(Entity e, bool show_back) {
    if (!global_coordinator.entity_has_component<Permanent>(e)) return;
    if (!global_coordinator.entity_has_component<CardData>(e)) return;

    const CardData &front = global_coordinator.GetComponent<CardData>(e);
    const CardData *active = show_back ? front.backside.get() : &front;
    if (!active) return;  // requested back face but this card is single-faced

    const CardData &old_face = active_face(e, front);
    if (&old_face == active) return;  // already showing that face

    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    const std::string old_name = perm.name;
    perm.transformed = show_back;
    perm.times_transformed++;
    perm.name = active->name;
    perm.types = active->types;

    sync_face_creature_state(e, *active);
    // Counters stay on a transforming permanent (CR 712.18), loyalty included: a planeswalker
    // gets loyalty counters only as it enters (CR 306.5b), so the loyalty map is left alone.
    swap_face_abilities(e, perm, old_face, *active);

    game_log("%s transforms into %s!\n", old_name.c_str(), active->name.c_str());
}

void transform_permanent(Entity e) {
    if (!global_coordinator.entity_has_component<Permanent>(e)) return;
    const bool currently_back = global_coordinator.GetComponent<Permanent>(e).transformed;
    set_permanent_face(e, !currently_back);
}

bool ability_may_transform_source(const Ability &ab) {
    if (ab.source_transforms < 0) return true;
    const Entity self = ab.source.get();
    if (self == 0 || !global_coordinator.entity_has_component<Permanent>(self)) return true;
    return global_coordinator.GetComponent<Permanent>(self).times_transformed ==
           static_cast<uint32_t>(ab.source_transforms);
}

void stamp_source_transforms(Ability &ab) {
    if (ab.source_transforms >= 0) return;
    const Entity self = ab.source.get();
    if (self == 0 || !global_coordinator.entity_has_component<Permanent>(self)) return;
    ab.source_transforms = global_coordinator.GetComponent<Permanent>(self).times_transformed;
}
