#ifndef CARDDATA_H
#define CARDDATA_H

#include <string>
#include <set>
#include <cstdint>
#include <vector>
#include "../components/types.h"
#include "../classes/colors.h"
#include "ability.h"

struct AltCost {
    bool has_alt_cost = false;
    int life_cost = 0;
    int exile_blue_from_hand = 0;
};

//this is the underlying card, not a permanent or spell
struct CardData{
    std::string uid;
    std::string name;
    std::string oracle_text;
    std::set<Type> types;
    std::multiset<Colors> mana_cost;
    uint32_t power = 0;
    uint32_t toughness = 0;
    //starting loyalty etc described as a static ability
    std::vector<Ability> abilities;
    AltCost alt_cost;
};

#endif /* CARD_H */
