#ifndef GAME_H
#define GAME_H

#include <cstdint>
#include <memory>
#include <random>
#include "../ecs/entity.h"

struct Deck;
struct Game;

extern Game cur_game;

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
        Game(){};
        Game(size_t _seed){
            seed = _seed;
            gen = std::mt19937(seed);
        };
        size_t seed;
        size_t timestamp = 0;
        size_t turn = 0;
        Step cur_step = UNTAP;
        Entity player_a_entity;
        Entity player_b_entity;
        std::mt19937 gen;
        bool ended = false;
        bool player_a_active = true;
        bool player_a_turn = true;
        bool player_a_has_priority = true;
        bool last_player_passed = false; // tracks if the player who just had priority passed

        void generate_players(const Deck& deck_a, const Deck& deck_b);
        bool advance_step(std::shared_ptr<class StackManager> stack_manager);
        void pass_priority();
        void take_action(); // resets last_player_passed since an action was taken

    private:
        Entity gen_player(const Deck& deck);
};

#endif /* GAME_H */
