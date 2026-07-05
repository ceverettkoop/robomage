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
#define MAX_STACK_MODES 6  // chosen-mode multi-hot width per stack entry
#define MAX_STACK_TGTS 4   // announced targets serialized per stack entry (truncated)
#define MAX_GY_SLOTS 64  // per player
#define MAX_HAND_SLOTS 10
#define MAX_ACTIONS 64
#define MAX_CHOICE_DESC 128
#define REVEALED_CARD_TYPES 1024  // mirror N_CARD_TYPES in machine_io.h / REVEALED_SIZE in match_state.h

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
    int  loyalty;                // loyalty counters for planeswalkers (0 for non-planeswalkers)
    char token_name[32];         // non-empty for tokens (card_vocab_idx == TOKEN_SENTINEL)
    char counters[64];           // compact typed-counter summary ("charge:2, +1/+1:3"), empty = none
                                 // (display only — NOT serialized to the ML state vector)
} PermanentState;

// One announced target of a stack object (public info, chosen at cast — CR 601.2c).
typedef struct StackTarget_tag {
    bool present;             // a target occupies this sub-slot
    bool is_player;           // the target is a player (card_vocab_idx = -1 then)
    bool controller_is_self;  // controller of the targeted permanent / the targeted player
                              // himself (owner for a non-permanent card)
    int  card_vocab_idx;      // -1 = player / unknown
} StackTarget;

typedef struct StackEntry_tag {
    int  card_vocab_idx;  // -1 = unknown/empty
    bool controller_is_self;
    bool is_spell;        // true = card spell on stack; false = triggered/activated ability
    bool chosen_modes[MAX_STACK_MODES];  // multi-hot of modal modes announced at cast
                                         // (CR 601.2b); all false when not modal
    StackTarget targets[MAX_STACK_TGTS]; // announced targets in announcement order:
                                         // primary, sub-abilities', chosen modes'
    char target_name[48]; // display name of first target, empty = no target (display only)
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
    bool          card_is_public;              // card identity is public (revealed) even in a hidden zone
    ActionRefZone zone_ref;
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
    // Opponent-hand cards whose identity the viewer knows (revealed in hand by
    // Duress/Thoughtseize/tutor, and not yet moved to another zone). card_vocab_idx,
    // -1 = empty/unknown slot. Tracks the specific card, unlike opp_revealed which
    // is only a match-scoped "ever seen" multi-hot.
    int  opp_known_hand[MAX_HAND_SLOTS];
    int  self_library_ct;
    int  opp_library_ct;

    // Recent action history (newest first), 4 floats per entry:
    //   category / ACTION_CATEGORY_MAX, card_vocab_idx / N_CARD_TYPES, is_self,
    //   turn / 50.0
    float action_history[ACTION_HISTORY_SIZE * 4];
    int   action_history_len;  // valid entries (0 to ACTION_HISTORY_SIZE)

    // Known top-of-library cards (viewer's library only). Index 0 = top.
    // -1 = unknown.
    int known_top_library_self[KNOWN_TOP_LIBRARY_SIZE];

    // Opponent's revealed-cards multi-hot, accumulated across the match (bo3).
    // opp_revealed[i] = 1 if the opponent has ever revealed card vocab index i.
    unsigned char opp_revealed[REVEALED_CARD_TYPES];

    // bo3 match state
    int  match_game_number;  // -1 = single game, 0-2 = bo3 game index
    int  match_wins_self;
    int  match_wins_opp;
    bool is_sideboard_phase;

    // Pending decision context: the spell/ability currently making a mid-resolution
    // choice (target select, dig/scry/surveil pick, search, discard, modal, ...).
    // The source may not be on the stack yet — targets are announced before the
    // spell moves there (CR 601.2b/c) — so without this the observation cannot show
    // WHAT is asking for the current choice. card_vocab_idx, -1 = no pending source.
    int  pending_decision_card;
    bool pending_decision_ctrl_is_self;  // pending source's controller == viewer
} GameState;

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif /* GAMESTATE_H */
