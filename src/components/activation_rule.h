#ifndef ACTIVATION_RULE_H
#define ACTIVATION_RULE_H

#include <cstdint>

#include "../ecs/entity.h"

struct AbilityDef;

// A named state an ability's activation needs: a row of queries/activation.cpp's table.
struct ActivationCondition;

// One restriction on activating an ability (CR 602.5). An ability's rules are a list on its
// definition (AbilityDef::activation_rules), and every one must allow an activation
// (activation_source_ready, queries/activation.h). A keyword that restricts activation is rows
// of this list, not a check of its own.
struct ActivationRule {
    enum Kind : uint8_t {
        TIMING,     // only as a sorcery (CR 602.5d)
        CONDITION,  // only while `condition` holds
        PER_TURN,   // at most `limit` activations each turn (CR 602.5b)
        PER_LIFETIME,  // at most `limit` activations for as long as the object exists (CR 400.7)
    };
    // What a limit counts.
    enum Counted : uint8_t {
        THIS_ABILITY,       // this ability, as its object acquired it (CR 602.5c)
        LOYALTY_ABILITIES,  // every loyalty ability of its object together (CR 606.3)
    };
    Kind kind = TIMING;
    const ActivationCondition *condition = nullptr;
    int limit = 0;
    Counted counted = THIS_ABILITY;

    static ActivationRule as_sorcery() { return ActivationRule{}; }
    static ActivationRule only_if(const ActivationCondition *condition) {
        ActivationRule rule;
        rule.kind = CONDITION;
        rule.condition = condition;
        return rule;
    }
    static ActivationRule per_turn(int limit, Counted counted = THIS_ABILITY) {
        ActivationRule rule;
        rule.kind = PER_TURN;
        rule.limit = limit;
        rule.counted = counted;
        return rule;
    }
    static ActivationRule per_lifetime(int limit) {
        ActivationRule rule;
        rule.kind = PER_LIFETIME;
        rule.limit = limit;
        return rule;
    }
};

// How often one limited ability of an object was activated, on its Permanent.
struct ActivationCount {
    const AbilityDef *def = nullptr;  // null: the object's loyalty abilities together (CR 606.3)
    Entity granted_by_static = 0;     // the static it was acquired through (CR 602.5c); 0 for its own
    int this_turn = 0;
    int lifetime = 0;
};

#endif /* ACTIVATION_RULE_H */
