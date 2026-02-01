#ifndef PLAYER_H
#define PLAYER_H

#include <cstdint>
#include <set>
#include "../classes/colors.h"

struct Player {
    bool otp;
    int32_t life_total = 20;
    uint8_t poison_counters = 0;
    uint8_t energy_counters = 0;
    std::multiset<Colors> mana;
    uint8_t lands_played_this_turn = 0;
};

#endif /* PLAYER_H */
