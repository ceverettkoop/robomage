#ifndef GAMESTATE_H
#define GAMESTATE_H

#ifdef __cplusplus
#include "game.h"
extern "C" {
#else
#include "stdbool.h"
#include "game.h"
#endif

// relevant limits, also used in machine_io.h
#define MAX_BATTLEFIELD_SLOTS 48  // all permanents (creatures + lands + other) per player
#define MAX_STACK_DISPLAY 12
#define MAX_GY_SLOTS 64  // per player
#define MAX_HAND_SLOTS 10
#define MAX_ACTIONS 64
#define MAX_CHOICE_DESC 128

typedef struct PlayerState_tag {
    int life;
    int poison_counters;
    int mana[6];               // WUBRGC
    int lands_played_this_turn;
    int hand_ct;               // actual hand size
} PlayerState;

typedef struct PermanentState_tag {
    int  card_vocab_idx;         // -1 = empty slot
    bool controller_is_self;
    bool is_tapped;
    bool is_creature;
    bool is_land;
    int  power;                  // 0 for non-creatures
    int  toughness;
    bool is_attacking;
    bool is_blocking;
    bool has_summoning_sickness;
    int  damage;
    char token_name[32];         // non-empty for tokens (card_vocab_idx == TOKEN_SENTINEL)
} PermanentState;

typedef struct StackEntry_tag {
    int  card_vocab_idx;  // -1 = unknown/empty
    bool controller_is_self;
    bool is_spell;        // true = card spell on stack; false = triggered/activated ability
    char target_name[48]; // display name of target, empty = no target
} StackEntry;

typedef enum ActionRefZone_tag {
    REF_NONE = 0,
    REF_SELF_BATTLEFIELD,
    REF_OPP_BATTLEFIELD,
    REF_SELF_HAND,
    REF_STACK,
    REF_SELF_GY,
    REF_OPP_GY,
    REF_SELF_EXILE,
    REF_OPP_EXILE,
    REF_PLAYER_SELF,
    REF_PLAYER_OPP,
} ActionRefZone;

typedef struct ActionChoice_tag {
    int           category;                    // ActionCategory value
    int           card_vocab_idx;              // -1 = null sentinel
    bool          controller_is_self;
    ActionRefZone zone_ref;
    int           slot_idx;                    // index into zone array (-1 = N/A)
    char          description[MAX_CHOICE_DESC]; //NOT SERIALIZED TO ML
} ActionChoice;

typedef struct Query_tag {
    int          num_choices;
    ActionChoice choices[MAX_ACTIONS];
} Query;

typedef struct GameState_tag {
    PlayerState self;
    PlayerState opponent;
    int         turn;
    Step        cur_step;
    bool        is_active_player;  // true when the viewer (self) is the active player
    bool        viewer_has_priority; // true when the viewer (self) currently holds priority
    bool        self_is_player_a;
    int         stack_size;

    PermanentState self_permanents[MAX_BATTLEFIELD_SLOTS];
    PermanentState opp_permanents[MAX_BATTLEFIELD_SLOTS];

    StackEntry  stack[MAX_STACK_DISPLAY];

    int  self_graveyard[MAX_GY_SLOTS];   // card_vocab_idx, -1 = empty
    int  opp_graveyard[MAX_GY_SLOTS];
    int  self_exile[MAX_GY_SLOTS]; //NOT SERIALIZED TO ML FOR NOW -too expensive -TODO Revisit when exile matters 
    int  opp_exile[MAX_GY_SLOTS];//NOT SERIALIZED TO ML FOR NOW -too expensive -TODO Revisit when exile matters

    int  self_hand[MAX_HAND_SLOTS];      // card_vocab_idx, -1 = empty
    int  opp_hand_ct;                    // == opponent.hand_ct (backwards compat)
    int  self_library_ct;
    int  opp_library_ct;

    // Recent action history (newest first), 3 floats per entry:
    //   category / ACTION_CATEGORY_MAX, card_vocab_idx / N_CARD_TYPES, is_self
    float action_history[ACTION_HISTORY_SIZE * 3];
    int   action_history_len;  // valid entries (0 to ACTION_HISTORY_SIZE)
} GameState;

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif /* GAMESTATE_H */
