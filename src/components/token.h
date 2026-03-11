#ifndef TOKEN_H
#define TOKEN_H

#include <cstdint>
#include <set>
#include <string>
#include <vector>
#include "ability.h"
#include "types.h"

// Token entities have Zone + Permanent + Creature + Damage + Token on the battlefield.
// They have no CardData. When a token leaves the battlefield it is destroyed entirely.
struct Token {
    std::string name = "";
    std::set<Type> types;
    std::vector<Ability> abilities;    // triggered abilities (e.g. Prowess)
    std::vector<std::string> keywords; // informational; copied to Creature on creation
    uint32_t power = 0;
    uint32_t toughness = 0;
};

#endif /* TOKEN_H */
