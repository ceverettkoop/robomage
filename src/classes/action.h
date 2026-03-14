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
    MANA_ABILITY      = 1,  // legacy, unused — color-specific categories below are emitted instead
    SELECT_ATTACKER   = 2,
    CONFIRM_ATTACKERS = 3,
    SELECT_BLOCKER    = 4,
    CONFIRM_BLOCKERS  = 5,
    ACTIVATE_ABILITY  = 6,
    CAST_SPELL        = 7,
    SELECT_TARGET     = 8,
    PLAY_LAND         = 9,
    OTHER_CHOICE      = 10,
    MULLIGAN          = 11,  // binary: 0=keep, 1=take mulligan
    BOTTOM_DECK_CARD  = 12,  // select card index from hand to put on library bottom
    MANA_W            = 13,  // tap for white mana
    MANA_U            = 14,  // tap for blue mana
    MANA_B            = 15,  // tap for black mana
    MANA_R            = 16,  // tap for red mana
    MANA_G            = 17,  // tap for green mana
    MANA_C            = 18,  // tap for colorless mana
    SEARCH_LIBRARY    = 19,  // select a card from a library search (index 0 = fail to find)
    TOP_LIBRARY       = 20,  // select a card to place on top of library
    SHUFFLE           = 21,  // shuffle a library
    PAYING_COSTS      = 22,  // delve exile or pitch card from hand to pay costs
    DIG_CHOICE        = 23,  // choose a card from a dig (look at top N) ability
};

static constexpr int ACTION_CATEGORY_MAX = 23;  // highest ActionCategory value

struct LegalAction {
    ActionType type;
    Entity source_entity;  // Card/permanent being used (if applicable)
    Entity target_entity;  // Target entity (if applicable)
    Ability ability;       // Ability being activated (ACTIVATE_ABILITY only)
    std::string description;
    ActionCategory category = ActionCategory::OTHER_CHOICE;
    bool use_alt_cost = false;

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
