#ifndef ACTION_H
#define ACTION_H

#include "../ecs/entity.h"
#include <string>

enum ActionType {
    PASS_PRIORITY,
    CAST_SPELL,
    ACTIVATE_ABILITY,
    SPECIAL_ACTION  // Includes: play land, turn face-up morph, etc.
};

struct LegalAction {
    ActionType type;
    Entity source_entity;  // Card/ability being used (if applicable)
    Entity target_entity;  // Target of the action (if applicable)
    std::string description;  // Human-readable description

    LegalAction(ActionType t, const std::string& desc)
        : type(t), source_entity(0), target_entity(0), description(desc) {}

    LegalAction(ActionType t, Entity source, const std::string& desc)
        : type(t), source_entity(source), target_entity(0), description(desc) {}

    LegalAction(ActionType t, Entity source, Entity target, const std::string& desc)
        : type(t), source_entity(source), target_entity(target), description(desc) {}
};

#endif /* ACTION_H */
