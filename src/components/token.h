#ifndef TOKEN_H
#define TOKEN_H

#include <cstdint>
#include <set>
#include <string>
#include <vector>
#include "ability.h"
#include "static_ability.h"
#include "types.h"
#include "zone.h"
#include "../classes/colors.h"
#include "../ecs/entity.h"

// Token entities have Zone + Permanent + Creature + Damage + Token on the battlefield.
// They have no CardData. When a token leaves the battlefield it is destroyed entirely.
struct Token {
    std::string name = "";
    // Token SCRIPT filename stem (e.g. "w_1_1_cat", "c_a_clue_draw"), set by
    // parse_token_script. Keys the ML token-identity lookup (token_script_to_index in
    // card_vocab.h) — display names collide across scripts, script stems do not.
    std::string script_name = "";
    std::set<Type> types;
    std::vector<Ability> abilities;    // triggered abilities (e.g. Prowess)
    // Continuous static abilities from the token script's S: lines (e.g. the Urza's Saga
    // Construct token's "This creature gets +1/+1 for each artifact you control."). Copied onto
    // the Permanent at bootstrap so gather_active_statics applies them like a real card's statics.
    std::vector<StaticAbility> static_abilities;
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
