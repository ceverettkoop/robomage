#include "affected.h"

#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/player.h"
#include "../ecs/coordinator.h"
#include "players.h"
#include "zones.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static bool names_parent_target(const Ability &ab);
static void append_chosen_targets(const Ability &ab, std::vector<Entity> &out);

// Defined$ Targeted / ParentTarget / Parent: the ability refers to the target its parent chose.
static bool names_parent_target(const Ability &ab) {
    const std::string &d = ab.def->defined;
    return d == "Targeted" || d == "ParentTarget" || d == "Parent";
}

static void append_chosen_targets(const Ability &ab, std::vector<Entity> &out) {
    if (!ab.targets.empty()) {
        for (Entity e : live_entities(ab.targets)) out.push_back(e);
    } else if (Entity t = ab.target.get()) {
        out.push_back(t);
    }
}

bool names_defined_player(const Ability &ab) {
    const AbilityDef &d = *ab.def;
    return d.defined_you || d.defined_each_opponent || d.defined_targeted_controller ||
           d.defined_triggered_activator || d.defined_triggered_player ||
           d.defined_triggered_card_controller;
}

Zone::Ownership affected_player(const Ability &ab) {
    if (names_defined_player(ab)) {
        Zone::Ownership who = resolve_defined_player(ab);
        if (who != Zone::UNKNOWN) return who;
    }
    if (ab.def->valid_tgts != "N_A" || names_parent_target(ab)) {
        const Entity t = ab.target.get();
        if (t != 0 && global_coordinator.entity_has_component<Player>(t)) return seat_of_player(t);
    }
    return ab.controller;
}

std::vector<Entity> affected_objects(const Ability &ab) {
    std::vector<Entity> out;
    const AbilityDef &d = *ab.def;
    if (d.valid_tgts != "N_A" || names_parent_target(ab)) {
        append_chosen_targets(ab, out);
    } else if (d.defined_triggered_attacker_lki || d.defined_triggered_spell ||
               d.defined_triggered_source_sa) {
        // Bound as the ability's target when it triggered.
        append_chosen_targets(ab, out);
    } else if (d.defined_remembered) {
        for (Entity e : live_entities(cur_game.resolution.memory.remembered)) out.push_back(e);
    } else if (d.defined_exiled_with) {
        if (Entity e = exiled_with_card(ab.source.get())) out.push_back(e);
    } else if (d.defined.empty() || d.defined_self) {
        if (Entity e = ab.source.get()) out.push_back(e);
    }
    return out;
}
