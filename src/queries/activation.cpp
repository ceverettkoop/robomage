#include "activation.h"

#include <algorithm>

#include "../classes/game.h"
#include "../coverage.h"
#include "../ecs/coordinator.h"
#include "attachments.h"
#include "battlefield.h"
#include "combat.h"
#include "types.h"

// What a rule reads of one activation: who activates, from what, and whether a sorcery could be
// cast now. `source` is the object the ability is activated from, 0 if unknown.
struct ActivationAttempt {
    const Ability &ab;
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
static bool condition_attacked_this_turn(const ActivationAttempt &attempt);
static bool condition_never(const ActivationAttempt &attempt);
static bool activation_rule_allows(const ActivationRule &rule, const ActivationAttempt &attempt);
static const ActivationCount *find_count(const Permanent &perm, const AbilityDef *def, Entity granted_by_static);
static const ActivationCount *rule_count(const ActivationRule &rule, const ActivationAttempt &attempt);
static void bump_count(Permanent &perm, const AbilityDef *def, Entity granted_by_static);

static const ActivationCondition kActivationConditions[] = {
    {"Metalcraft", condition_metalcraft},
    {"NotMonstrous", condition_not_monstrous},
    {"CanEquip", condition_can_equip},
    {"Attached", condition_attached},
    {"AttackedThisTurn", condition_attacked_this_turn},
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

// AttackedThisTurn (CR 702.142a): boast's "only if this creature attacked this turn".
static bool condition_attacked_this_turn(const ActivationAttempt &attempt) {
    return attempt.source != 0 && global_coordinator.entity_has_component<Permanent>(attempt.source) &&
           attacked_this_turn(global_coordinator.GetComponent<Permanent>(attempt.source));
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

bool reads_attacked_this_turn(const Permanent &perm) {
    const ActivationCondition *attacked = activation_condition_named("AttackedThisTurn");
    for (const Ability &ab : perm.abilities)
        for (const ActivationRule &rule : ab.def->activation_rules)
            if (rule.kind == ActivationRule::CONDITION && rule.condition == attacked) return true;
    return false;
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
        case ActivationRule::PER_TURN: {
            const ActivationCount *count = rule_count(rule, attempt);
            return !count || count->this_turn < rule.limit;
        }
        case ActivationRule::PER_LIFETIME: {
            const ActivationCount *count = rule_count(rule, attempt);
            return !count || count->lifetime < rule.limit;
        }
    }
    return false;
}

bool activation_source_ready(const Ability &ab, Entity source, Zone::Ownership controller, const EntitySet &entities,
    bool sorcery_window) {
    const ActivationAttempt attempt{ab, source, controller, entities, sorcery_window};
    for (const ActivationRule &rule : ab.def->activation_rules)
        if (!activation_rule_allows(rule, attempt)) return false;
    return true;
}

static const ActivationCount *find_count(const Permanent &perm, const AbilityDef *def, Entity granted_by_static) {
    for (const ActivationCount &count : perm.activation_counts)
        if (count.def == def && count.granted_by_static == granted_by_static) return &count;
    return nullptr;
}

// The count `rule` limits, for the object the attempt activates from; null when there is none yet,
// or no permanent to hold one (a card activated from its hand).
static const ActivationCount *rule_count(const ActivationRule &rule, const ActivationAttempt &attempt) {
    if (!global_coordinator.entity_has_component<Permanent>(attempt.source)) return nullptr;
    const Permanent &perm = global_coordinator.GetComponent<Permanent>(attempt.source);
    return rule.counted == ActivationRule::LOYALTY_ABILITIES
               ? find_count(perm, nullptr, 0)
               : find_count(perm, attempt.ab.def, attempt.ab.granted_by_static);
}

static void bump_count(Permanent &perm, const AbilityDef *def, Entity granted_by_static) {
    for (ActivationCount &count : perm.activation_counts)
        if (count.def == def && count.granted_by_static == granted_by_static) {
            ++count.this_turn;
            ++count.lifetime;
            return;
        }
    perm.activation_counts.push_back(ActivationCount{def, granted_by_static, 1, 1});
}

void reset_permanent_activations_this_turn(Permanent &perm) {
    std::vector<ActivationCount> &counts = perm.activation_counts;
    if (counts.empty()) return;
    for (ActivationCount &count : counts) count.this_turn = 0;
    counts.erase(std::remove_if(counts.begin(), counts.end(),
                     [](const ActivationCount &count) {
                         if (!count.def) return true;
                         for (const ActivationRule &rule : count.def->activation_rules)
                             if (rule.kind == ActivationRule::PER_LIFETIME) return false;
                         return true;
                     }),
        counts.end());
}

int activations_this_turn(const Permanent &perm, const Ability &ab) {
    const ActivationCount *count = find_count(perm, ab.def, ab.granted_by_static);
    return count ? count->this_turn : 0;
}

int activation_turn_limit(const AbilityDef &def) {
    for (const ActivationRule &rule : def.activation_rules)
        if (rule.kind == ActivationRule::PER_TURN && rule.counted == ActivationRule::THIS_ABILITY) return rule.limit;
    return 0;
}

bool activation_limit_reached(const Permanent &perm, const Ability &ab) {
    const ActivationCount *count = find_count(perm, ab.def, ab.granted_by_static);
    if (!count) return false;
    for (const ActivationRule &rule : ab.def->activation_rules) {
        if (rule.counted != ActivationRule::THIS_ABILITY) continue;
        if (rule.kind == ActivationRule::PER_TURN && count->this_turn >= rule.limit) return true;
        if (rule.kind == ActivationRule::PER_LIFETIME && count->lifetime >= rule.limit) return true;
    }
    return false;
}

void count_activation(Permanent &perm, const Ability &ab) {
    bool own = false, loyalty = false;
    for (const ActivationRule &rule : ab.def->activation_rules) {
        if (rule.kind == ActivationRule::CONDITION) coverage_row(rule.condition->name);
        if (rule.kind == ActivationRule::PER_LIFETIME) coverage_row("activation_per_lifetime");
        if (rule.kind != ActivationRule::PER_TURN && rule.kind != ActivationRule::PER_LIFETIME) continue;
        (rule.counted == ActivationRule::LOYALTY_ABILITIES ? loyalty : own) = true;
    }
    if (own) bump_count(perm, ab.def, ab.granted_by_static);
    if (loyalty) bump_count(perm, nullptr, 0);
}


bool source_can_tap(Entity source) {
    if (!global_coordinator.entity_has_component<Permanent>(source)) return false;
    if (global_coordinator.GetComponent<Permanent>(source).is_tapped) return false;
    return !is_summoning_sick(source);
}
