#ifndef GAME_H
#define GAME_H

#include <cstdint>
#include "../ecs/entity.h"

struct Deck;

enum Step {
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

struct Game {
        Game(unsigned int _seed){
            seed = _seed;
        };
        unsigned int seed;
        uint32_t timestamp = 0;
        uint16_t turn = 0;
        Step cur_step = UNTAP;
        Entity player_a_entity;
        Entity player_b_entity;

        void generate_players(const Deck& deck_a, const Deck& deck_b);
        void generate_libraries(const Deck& deck_a, const Deck& deck_b);

    private:
        Entity gen_player(const Deck& deck);
};

#endif /* GAME_H */
