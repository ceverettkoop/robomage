#include "activation.h"

#include "../classes/game.h"
#include "../ecs/coordinator.h"
#include "attachments.h"
#include "battlefield.h"
#include "combat.h"
#include "types.h"

// What a rule reads of one activation: who activates, from what, and whether a sorcery could be
// cast now. `source` is the object the ability is activated from, 0 if unknown.
struct ActivationAttempt {
    Entity source;
    Zone::Ownership controller;
    const EntitySet &entities;
    bool sorcery_window;
};

struct ActivationCondition {
    const char *name;
    bool (*met)(const ActivationAttempt &attempt);
};

static bool condition_metalcraft(const ActivationAttempt &attempt);
static bool condition_not_monstrous(const ActivationAttempt &attempt);
static bool condition_can_equip(const ActivationAttempt &attempt);
static bool condition_attached(const ActivationAttempt &attempt);
static bool condition_never(const ActivationAttempt &attempt);
static bool activation_rule_allows(const ActivationRule &rule, const ActivationAttempt &attempt);

static const ActivationCondition kActivationConditions[] = {
    {"Metalcraft", condition_metalcraft},
    {"NotMonstrous", condition_not_monstrous},
    {"CanEquip", condition_can_equip},
    {"Attached", condition_attached},
};
static const ActivationCondition kNoSuchCondition = {"", condition_never};

// NotMonstrous (CR 701.37a): a monstrosity ability is legal only while its source isn't
// already monstrous. Keyed on the source permanent's is_monstrous designation. Installed
// automatically by parse_put_counter for any Monstrosity$ ability.
static bool condition_not_monstrous(const ActivationAttempt &attempt) {
    return attempt.source != 0 && global_coordinator.entity_has_component<Permanent>(attempt.source) &&
           !global_coordinator.GetComponent<Permanent>(attempt.source).is_monstrous;
}

// CanEquip (CR 702.6a, 301.5c): an equip ability (K:Equip / K:Reconfigure) is activatable
// only while its Equipment can equip some creature its controller controls.
static bool condition_can_equip(const ActivationAttempt &attempt) {
    return attempt.source != 0 && !equip_candidates(attempt.source, attempt.controller, attempt.entities).empty();
}

// Attached (CR 702.151a): reconfigure's unattach ability — "Activate only if this permanent
// is attached to a creature".
static bool condition_attached(const ActivationAttempt &attempt) {
    return attempt.source != 0 && global_coordinator.entity_has_component<Permanent>(attempt.source) &&
           global_coordinator.GetComponent<Permanent>(attempt.source).equipped_to.get() != 0;
}

static bool condition_metalcraft(const ActivationAttempt &attempt) {
    return controller_has_metalcraft(attempt.controller, attempt.entities);
}

static bool condition_never(const ActivationAttempt &) { return false; }

int ability_resolutions_this_turn(Entity source) {
    const int *n = cur_game.ability_resolution_counts.find(source);
    return n ? *n : 0;
}

bool trigger_resolved_limit_reached(const Ability &ab) {
    if (ab.def->trigger_resolved_limit <= 0) return false;
    const auto *counts = cur_game.limited_resolution_counts.find(ab.source.get());
    if (!counts) return false;
    for (const Game::LimitedResolution &lr : *counts)
        if (lr.def == ab.def && lr.controller == ab.controller) return lr.count >= ab.def->trigger_resolved_limit;
    return false;
}

int triggers_this_turn(const AbilityDef *def, Entity source) {
    const auto *counts = cur_game.limited_trigger_counts.find(source);
    if (!counts) return 0;
    for (const Game::LimitedTrigger &lt : *counts)
        if (lt.def == def) return lt.count;
    return 0;
}

void count_limited_trigger(const AbilityDef *def, Entity source) {
    std::vector<Game::LimitedTrigger> &counts = cur_game.limited_trigger_counts[source];
    for (Game::LimitedTrigger &lt : counts)
        if (lt.def == def) {
            ++lt.count;
            return;
        }
    counts.push_back({def, 1});
}

void count_resolution(const Ability &ab, bool from_stack) {
    if (from_stack && ab.def->ability_type == AbilityDef::TRIGGERED) cur_game.ability_resolution_counts[ab.source]++;
    if (ab.def->trigger_resolved_limit <= 0) return;
    const Entity source = ab.source.get();
    if (source == 0) return;
    std::vector<Game::LimitedResolution> &counts = cur_game.limited_resolution_counts[source];
    for (Game::LimitedResolution &lr : counts)
        if (lr.def == ab.def && lr.controller == ab.controller) {
            ++lr.count;
            return;
        }
    counts.push_back(Game::LimitedResolution{ab.def, ab.controller, 1});
}

bool controller_has_metalcraft(Zone::Ownership controller, const EntitySet &entities) {
    int artifacts = 0;
    for (auto e : battlefield_permanents(entities, controller))
        if (type_set_has(global_coordinator.GetComponent<Permanent>(e).types, "Artifact"))
            if (++artifacts >= 3) return true;
    return false;
}

const ActivationCondition *activation_condition_named(const std::string &name) {
    for (const ActivationCondition &row : kActivationConditions)
        if (name == row.name) return &row;
    return &kNoSuchCondition;
}

bool activation_condition_known(const std::string &name) {
    return activation_condition_named(name) != &kNoSuchCondition;
}

bool activates_as_sorcery(const AbilityDef &def) {
    for (const ActivationRule &rule : def.activation_rules)
        if (rule.kind == ActivationRule::TIMING) return true;
    return false;
}

static bool activation_rule_allows(const ActivationRule &rule, const ActivationAttempt &attempt) {
    switch (rule.kind) {
        case ActivationRule::TIMING:
            return attempt.sorcery_window;
        case ActivationRule::CONDITION:
            return rule.condition->met(attempt);
    }
    return false;
}

bool activation_source_ready(const Ability &ab, Entity source, Zone::Ownership controller, const EntitySet &entities,
    bool sorcery_window) {
    const ActivationAttempt attempt{source, controller, entities, sorcery_window};
    for (const ActivationRule &rule : ab.def->activation_rules)
        if (!activation_rule_allows(rule, attempt)) return false;
    return ab.def->activation_limit <= 0 || ab.activations_this_turn < ab.def->activation_limit;
}

bool source_can_tap(Entity source) {
    if (!global_coordinator.entity_has_component<Permanent>(source)) return false;
    if (global_coordinator.GetComponent<Permanent>(source).is_tapped) return false;
    return !is_summoning_sick(source);
}
