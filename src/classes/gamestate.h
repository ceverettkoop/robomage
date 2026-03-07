#ifndef GAMESTATE_H
#define GAMESTATE_H

#ifdef __cplusplus
#include "game.h"
extern "C" {
#else
#include "stdbool.h"
#include "game.h"
#endif

typedef struct PlayerState_tag {
        int life;
        int poison_counters;
        int mana[6];  // WUBRGC
        int lands_played_this_turn;
} PlayerState;

typedef struct PermanentState_tag {
        int card_vocab_idx;  // -1 = empty slot
        bool controller_is_self;
        bool is_tapped;
        int power;  // 0 for lands
        int toughness;
        bool is_attacking;
        bool is_blocking;
        bool has_summoning_sickness;
        int damage;
} PermanentState;

typedef struct StackEntry_tag {
        int card_vocab_idx;
        bool controller_is_self;
} StackEntry;

typedef struct GameState_tag {
        PlayerState self;
        PlayerState opponent;
        Step cur_step;
        bool is_active_player;  // essentially is it our turn, we always have priority when state is transmitted to us
        bool self_is_player_a;
        int stack_size;

        PermanentState self_creatures[MAX_BATTLEFIELD_SLOTS];
        PermanentState opp_creatures[MAX_BATTLEFIELD_SLOTS];
        PermanentState self_lands[MAX_LAND_SLOTS];
        PermanentState opp_lands[MAX_LAND_SLOTS];

        StackEntry stack[MAX_STACK_DISPLAY];

        int self_graveyard[MAX_GY_SLOTS];  // card_vocab_idx, -1 = empty
        int opp_graveyard[MAX_GY_SLOTS];

        int self_hand[MAX_HAND_SLOTS];  // card_vocab_idx, -1 = empty
} GameState;

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif /* GAMESTATE_H */
