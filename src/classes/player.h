#ifndef PLAYER_H
#define PLAYER_H

#include <cstdint>

struct Player{
    Player(bool _otp){
        otp = _otp;
    }
    bool otp;
    int32_t life_total = 20;
    uint8_t poison_counters = 0;
    uint8_t energy_counters = 0;
};

#endif /* PLAYER_H */
