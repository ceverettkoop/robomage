#ifndef PLAYER_H
#define PLAYER_H

#include <cstdint>
#include <set>
#include <vector>
#include "../classes/colors.h"
#include "../ecs/entity.h"

struct Player {
    bool otp;
    int32_t life_total = 20;
    uint8_t poison_counters = 0;
    uint8_t energy_counters = 0;
    std::multiset<Colors> mana;
    uint8_t lands_played_this_turn = 0;
    size_t spells_cast_this_turn = 0;
    size_t spells_cast_this_game = 0;
    std::vector<Entity> cards_drawn_this_turn;
    // creature subtypes in this player's deck: pair<list_index, all_subtypes_index>
    std::vector<std::pair<int, int>> creature_subtypes;
};

#endif /* PLAYER_H */
