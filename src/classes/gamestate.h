#ifndef GAMESTATE_H
#define GAMESTATE_H

#include "game.h"

// GameState: the board the CLI shows (print_game_state, cli_output.cpp) from one seat's view,
// filled from the engine state by populate_gamestate (machine_io.h). Plan step 4b replaces it with
// the deciding seat's RMSTATE frame.

#define MAX_BATTLEFIELD_SLOTS 48  // permanents shown per player
#define MAX_STACK_DISPLAY 12      // stack objects shown, top first
#define MAX_HAND_SLOTS 10         // hand cards shown
// The widest sideboard menu (game_driver.cpp, fatal past it).
#define MAX_ACTIONS 64
#define PERM_COUNTERS_LEN 64  // PermanentState.counters summary width

struct PlayerState {
        int life;
        int poison_counters;
        int energy;   // energy ({E}) counters (CR 122.1c)
        int mana[6];  // the mana pool by color: WUBRGC
};

struct PermanentState {
        int card_vocab_idx;  // -1 = empty slot
        bool is_tapped;
        bool is_creature;
        bool has_summoning_sickness;
        int power;  // 0 for non-creatures
        int toughness;
        int damage;
        char counters[PERM_COUNTERS_LEN];  // compact typed-counter summary ("charge:2, +1/+1:3"), empty = none
};

struct StackEntry {
        int card_vocab_idx;  // -1 = unknown
};

struct GameState {
        PlayerState self;
        PlayerState opponent;
        int turn;
        Step cur_step;
        bool is_active_player;                // the viewer (self) is the active player
        bool self_is_player_a;                // the viewer is Player A
        int stack_size;                       // stack objects, all of them (the first MAX_STACK_DISPLAY are shown)
        StackEntry stack[MAX_STACK_DISPLAY];  // top of stack first
        PermanentState self_permanents[MAX_BATTLEFIELD_SLOTS];  // ascending entity id
        PermanentState opp_permanents[MAX_BATTLEFIELD_SLOTS];
        int self_hand[MAX_HAND_SLOTS];  // the viewer's hand by vocab id, -1 = empty slot
};

#endif /* GAMESTATE_H */
