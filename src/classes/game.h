#ifndef GAME_H
#define GAME_H

#include <cstdint>

enum Step{
    UNTAP,
    UPKEEP,
    DRAW,
    FIRST_MAIN,
    BEGIN_COMBAT,
    DECLARE_ATTACKERS,
    DECLARE_BLOCKERS,
    COMBAT_DAMAGE,
    END_OF_COMBAT,
    SECOND_MAIN,
    END_STEP,
    CLEANUP
};


struct Game{
    uint32_t timestamp;
    uint16_t turn;
    Step cur_step;
};

#endif /* GAME_H */
