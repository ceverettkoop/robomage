#ifndef PLAYER_H
#define PLAYER_H

#include <cstdint>

struct Player{
    bool on_the_play;
    int32_t life_total = 20;
    uint8_t poison_counters = 0;
    uint8_t energy_counters = 0;
};

#endif /* PLAYER_H */
