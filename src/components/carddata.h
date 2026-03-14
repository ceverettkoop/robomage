#ifndef CARDDATA_H
#define CARDDATA_H

#include <memory>
#include <string>
#include <set>
#include <cstdint>
#include <vector>
#include <unordered_set>
#include "../components/types.h"
#include "../classes/colors.h"
#include "ability.h"
#include "effect.h"
#include "static_ability.h"

struct AltCost {
    bool has_alt_cost = false;
    int life_cost = 0;
    int exile_blue_from_hand = 0;
    int return_to_hand_count = 0;
    std::string return_to_hand_type = "";
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
    std::vector<std::string> keywords;
    std::vector<StaticAbility> static_abilities;
    std::vector<Effect::Replacement> replacement_effects;  // parsed from R: lines TODO expand this to parse SVAR below
    std::shared_ptr<CardData> backside;  // populated for DFCs; nullptr for normal cards
    bool has_delve = false;              // K:Delve — exile from graveyard to reduce generic cost
    bool has_x_cost = false;             // ManaCost contains X — variable generic cost chosen at cast time
    bool shuffle_into_library = false;   // card shuffles into library instead of going to graveyard on resolution
    bool is_equipment = false;           // has K:Equip line
    ManaValue equip_cost;                // parsed from K:Equip:cost
    std::set<Colors> explicit_colors;    // Colors: field override (e.g. Dryad Arbor)
};

#endif /* CARD_H */
