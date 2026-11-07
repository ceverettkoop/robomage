#ifndef ACTIVATION_RULE_H
#define ACTIVATION_RULE_H

#include <cstdint>

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
    };
    Kind kind = TIMING;
    const ActivationCondition *condition = nullptr;

    static ActivationRule as_sorcery() { return ActivationRule{}; }
    static ActivationRule only_if(const ActivationCondition *condition) { return {CONDITION, condition}; }
};

#endif /* ACTIVATION_RULE_H */
