#ifndef CARD_H
#define CARD_H

#include <string>
#include <set>
#include <cstdint>
#include "types.h"
#include "colors.h"
#include "../ecs/entity.h"

//this is the underlying card, a token or copy would not have these traits
struct Card{
    std::string uid;
    std::string name;
    std::string oracle_text;
    std::set<Type> types;
    std::multiset<Colors> mana_cost;
    uint32_t power = 0;
    uint32_t toughness = 0;
    //starting loyalty etc described as a static ability
    std::set<Entity> abilities;
};

#endif /* CARD_H */
