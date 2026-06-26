#ifndef TOKEN_H
#define TOKEN_H

#include <cstdint>
#include <set>
#include <string>
#include <vector>
#include "ability.h"
#include "types.h"
#include "zone.h"
#include "../classes/colors.h"
#include "../ecs/entity.h"

// Token entities have Zone + Permanent + Creature + Damage + Token on the battlefield.
// They have no CardData. When a token leaves the battlefield it is destroyed entirely.
struct Token {
    std::string name = "";
    std::set<Type> types;
    std::vector<Ability> abilities;    // triggered abilities (e.g. Prowess)
    std::vector<std::string> keywords; // informational; copied to Creature on creation
    uint32_t power = 0;
    uint32_t toughness = 0;
    // Token colors from the token script's Colors: line (Forge). Empty = colorless (CR 105.2c)
    // — a token has no mana cost, so its color comes solely from this indicator. Mirrors
    // CardData::explicit_colors so colorlessness is computed the same way for tokens and cards.
    std::set<Colors> explicit_colors;
};

// Attach the Permanent + Creature + Damage components a token needs on the battlefield,
// deriving P/T/keywords/controller from the Token. The single source of this bootstrap,
// called both immediately by Ability::resolve_token (so subabilities see the components)
// and by the StateManager SBE pass for any token entity still missing them. The timestamp
// counter is consumed (post-incremented) only when the Permanent is actually created, so
// repeated SBE passes over an already-bootstrapped token do not advance it.
void bootstrap_token_components(Entity tok_entity, const Token &tok,
                                Zone::Ownership controller, size_t &timestamp);

#endif /* TOKEN_H */
