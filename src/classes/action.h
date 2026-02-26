#ifndef ACTION_H
#define ACTION_H

#include "../ecs/entity.h"
#include "../components/ability.h"
#include <string>

enum ActionType {
    PASS_PRIORITY,
    CAST_SPELL,
    ACTIVATE_ABILITY,
    SPECIAL_ACTION  // Includes: play land, turn face-up morph, etc.
};

// Semantic category of each legal action, emitted per-action in machine mode
// so the model can learn action semantics across varying game states.
enum class ActionCategory {
    PASS_PRIORITY     = 0,
    MANA_ABILITY      = 1,
    SELECT_ATTACKER   = 2,
    CONFIRM_ATTACKERS = 3,
    SELECT_BLOCKER    = 4,
    CONFIRM_BLOCKERS  = 5,
    ACTIVATE_ABILITY  = 6,
    CAST_SPELL        = 7,
    SELECT_TARGET     = 8,
    PLAY_LAND         = 9,
    OTHER_CHOICE      = 10,
};

static constexpr int ACTION_CATEGORY_MAX = 10;  // highest ActionCategory value

struct LegalAction {
    ActionType type;
    Entity source_entity;  // Card/permanent being used (if applicable)
    Entity target_entity;  // Target entity (if applicable)
    Ability ability;       // Ability being activated (ACTIVATE_ABILITY only)
    std::string description;
    ActionCategory category = ActionCategory::OTHER_CHOICE;

    LegalAction(ActionType t, const std::string& desc)
        : type(t), source_entity(0), target_entity(0), description(desc) {}

    LegalAction(ActionType t, Entity source, const std::string& desc)
        : type(t), source_entity(source), target_entity(0), description(desc) {}

    LegalAction(ActionType t, Entity source, Entity target, const std::string& desc)
        : type(t), source_entity(source), target_entity(target), description(desc) {}

    LegalAction(ActionType t, Entity source, const Ability& ab, const std::string& desc)
        : type(t), source_entity(source), target_entity(0), ability(ab), description(desc) {}
};

#endif /* ACTION_H */
