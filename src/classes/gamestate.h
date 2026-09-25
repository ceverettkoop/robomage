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
#define N_OBS_KEYWORDS 16  // keyword multi-hot width per permanent slot (OBS_KEYWORDS in machine_io.h)
#define MAX_STACK_MODES 6  // chosen-mode multi-hot width per stack entry
#define MAX_STACK_TGTS 4   // announced targets serialized per stack entry (truncated)
#define MAX_DELAYED_TRIGGER_SLOTS 16  // pending delayed triggers serialized (truncated, seq order)
#define N_DELAYED_FIRE_KINDS 4  // fire_on one-hot: upkeep, end step, end of combat, leaves battlefield
#define MAX_GY_SLOTS 64  // per player
#define MAX_HAND_SLOTS 10
#define DECKLIST_MAIN_SLOTS 48  // distinct-name slots: self live library + opp maindeck
// Distinct-name slots for a sideboard block. A legal sideboard is 15 CARDS, so a
// balanced one never exceeds 15 distinct names — but the sideboard phase serializes
// the sideboarding player's own live deck after EVERY single-card move, including
// the mid-swap state where a card has been cut into a still-full sideboard (16
// cards). Hence 16, not 15: the extra slot is that transient cut card.
#define DECKLIST_SIDE_SLOTS 16
#define MAX_ACTIONS 64
#define MAX_CHOICE_DESC 128
#define PERM_COUNTERS_LEN 64  // PermanentState.counters summary width (mirrored in train/env.py)
#define PERM_TOKEN_NAME_LEN 32  // PermanentState.token_name width (mirrored in train/env.py)

typedef struct PlayerState_tag {
    int life;
    int poison_counters;
    int energy;                // energy ({E}) counters (CR 122.1c)
    int mana[6];               // WUBRGC
    int lands_played_this_turn;
    int hand_ct;               // actual hand size
    bool is_monarch;           // this player is the monarch (CR 725)
    bool city_blessing;        // this player has the city's blessing (CR 702.131c)
    bool revolt;               // a permanent this player controlled left the battlefield this turn
    int  extra_turns_pending;  // extra turns this player has queued (CR 500.7)
    // ── Mana development (serialized as the state vector's MANA DEVELOPMENT block;
    // field meanings and normalizers are documented in machine_io.h) ──
    int  mana_potential[6];    // WUBRGC: untapped sources that could produce that color NOW
    int  mana_potential_total; // distinct untapped mana sources + floating pool size
    int  lands_in_play;        // battlefield lands this player controls
    int  lands_in_hand;        // land cards in hand — VIEWER ONLY (0 for the opponent)
    int  land_drops_remaining; // lands still playable this turn (rules_mod, clamped at 0)
    // ── Per-turn counters (serialized as the state vector's PER-TURN COUNTERS block;
    // field meanings and normalizers are documented in machine_io.h) ──
    int  spells_cast_this_turn;
    int  noncreature_spells_cast_this_turn;
    int  instant_sorcery_spells_cast_this_turn;
    int  cards_drawn_this_turn;
    int  life_gained_this_turn;
    int  life_lost_this_turn;
    bool spell_colors_cast_this_turn[5]; // W, U, B, R, G
} PlayerState;

typedef struct PermanentState_tag {
    int  card_vocab_idx;         // -1 = empty slot
    int  chosen_name_idx;        // vocab idx of Permanent::chosen_name (Pithing Needle /
                                 // Disruptor Flute named card, Petrified Hamlet named land);
                                 // -1 = no name chosen
    int  returnable_exile_idx;   // vocab idx of the most recently exiled card linked to this
                                 // permanent that STILL has a return path (Static Prison holding a
                                 // real card, Flickerwisp/Phelia EOT blink); -1 = none (no return,
                                 // e.g. Skyclave Apparition). See returnable_exiled_card().
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
    int  p1p1_net;               // net +1/+1 minus -1/-1 counters (signed)
    int  other_counters;         // total counters of every other kind (excl. P1P1/M1M1/LOYALTY)
    // Entity-reference slots into the unified viewer-relative slot space (0-47 self
    // permanents, 48-95 opp permanents, 96-107 stack top-first; -1 = none). See the
    // entity->slot map in machine_io.cpp and norm_ref in machine_io.h.
    int  attached_to_ref;        // for equipment/auras: slot of the permanent this is attached to
    int  attached_by_ref;        // for creatures: slot of the equipment/aura attached to this
    int  attack_target_ref;      // attacked planeswalker's slot (-1 while attacking a player)
    int  blocking_target_ref;    // for blockers: slot of the attacker this creature blocks
    bool is_blocked;             // attacker was blocked at declare-blockers (CR 509.1h)
    bool is_phased_out;          // phased out (CR 702.26); serialized so the slot stays visible
    bool entered_this_turn;      // entered the battlefield this turn (ThisTurnEntered)
    int  ability_resolutions_this_turn; // triggered-ability resolutions from it this turn
                                 // (Count$ResolvedThisTurn)
    int  activations_this_turn;  // activations counted against its once-per-turn gates
                                 // (ActivationLimit$ counters summed + loyalty activation)
    bool cant_be_blocked_this_turn; // a "can't be blocked this turn" effect applies
    bool combat_damage_prevented;   // it is the creature of a combat-damage prevention shield
    bool pending_delayed_subject;   // watched by / a subject of a WAITING delayed trigger
                                    // (is_waiting_delayed_trigger_subject)
    bool keywords[N_OBS_KEYWORDS];  // effective keyword multi-hot (OBS_KEYWORDS order)
    char token_name[PERM_TOKEN_NAME_LEN]; // non-empty for tokens (card_vocab_idx == TOKEN_SENTINEL)
    char counters[PERM_COUNTERS_LEN]; // compact typed-counter summary ("charge:2, +1/+1:3"), empty = none
                                 // (display only — NOT serialized to the ML state vector; emitted as a
                                 // narrative-mode BQUERY side block for observers, like descriptions)
} PermanentState;

// One announced target of a stack object (public info, chosen at cast — CR 601.2c).
typedef struct StackTarget_tag {
    bool present;             // a target occupies this sub-slot
    bool is_player;           // the target is a player (card_vocab_idx = -1 then)
    bool controller_is_self;  // controller of the targeted permanent / the targeted player
                              // himself (owner for a non-permanent card)
    int  slot_ref;            // target's slot in the unified entity-ref space (-1 = none/player)
    int  card_vocab_idx;      // -1 = player / unknown
} StackTarget;

typedef struct StackEntry_tag {
    int  card_vocab_idx;  // -1 = unknown/empty
    bool controller_is_self;
    bool is_spell;        // true = card spell on stack; false = triggered/activated ability
    int  x_or_amount;     // spell: X paid at cast (Spell::x_paid); ability: Ability::amount
    // Cast qualifiers (public info announced at cast, CR 601.2b/f); all false for abilities.
    bool is_copy;               // a copy of a spell on the stack (CR 707.10)
    bool kicked_any;            // any kicker additional cost was paid (CR 702.33d)
    bool cast_with_flashback;
    bool cast_with_evoke;
    bool cast_with_escape;
    bool cast_with_offspring;
    bool cast_with_impending;
    bool chosen_modes[MAX_STACK_MODES];  // multi-hot of modal modes announced at cast
                                         // (CR 601.2b); all false when not modal
    StackTarget targets[MAX_STACK_TGTS]; // announced targets in announcement order:
                                         // primary, sub-abilities', chosen modes'
    char target_name[48]; // display name of first target, empty = no target (display only)
} StackEntry;

// One pending delayed trigger (CR 603.7): a record still waiting in
// Game::delayed_triggers, or its fired ability on the stack (DelayedTriggerLink::seq != 0).
typedef struct DelayedTriggerEntry_tag {
    bool present;
    bool controller_is_self;
    bool on_stack;             // false = waiting to fire, true = fired, its ability on the stack
    int  stack_ref;            // the stack object's slot ref when on_stack (-1 = none/truncated)
    int  creator_card_idx;     // vocab idx of the card whose ability set it up (-1 = none)
    int  creator_ref;          // creator's battlefield/stack slot ref (-1 = elsewhere)
    int  subject_ref;          // first watched / affected permanent's battlefield slot ref (-1 = none)
    int  subject_card_idx;     // vocab idx of the first subject (-1 = none)
    int  fire_kind;            // DelayedTriggerLink::FireKind (-1 = no one-hot column)
    bool fires_this_turn;      // a waiting phase trigger scheduled later this turn
} DelayedTriggerEntry;

typedef enum ActionRefZone_tag {
    REF_NONE = 0,
    REF_SELF_BATTLEFIELD,
    REF_OPP_BATTLEFIELD,
    REF_SELF_HAND,
    REF_OPP_HAND,
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
    int           slot_ref;                    // source entity's slot in the unified entity-ref
                                               // space (see machine_io.cpp map; -1 = none)
    int           option_ordinal;              // per-action ordinal/value scalar (mode index, X
                                               // value, color index, cast variant, depth, ...);
                                               // -1 = not applicable. See LegalAction::option_ordinal
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
    bool        self_is_player_a;
    int         stack_size;

    PermanentState self_permanents[MAX_BATTLEFIELD_SLOTS];
    PermanentState opp_permanents[MAX_BATTLEFIELD_SLOTS];

    StackEntry  stack[MAX_STACK_DISPLAY];

    // Pending delayed triggers, ascending by registration seq (packed, no holes).
    DelayedTriggerEntry delayed[MAX_DELAYED_TRIGGER_SLOTS];

    int  self_graveyard[MAX_GY_SLOTS];   // card_vocab_idx, -1 = empty
    int  opp_graveyard[MAX_GY_SLOTS];
    // Exile is collected + serialized in RECENCY order (slot 0 = most recent
    // arrival, sorted by Zone::distance_from_top). All exile is public in this
    // engine, so both zones are fully visible. card_vocab_idx, -1 = empty.
    int  self_exile[MAX_GY_SLOTS];
    int  opp_exile[MAX_GY_SLOTS];

    int  self_hand[MAX_HAND_SLOTS];      // card_vocab_idx, -1 = empty
    // Opponent-hand cards whose identity the viewer knows (revealed in hand by
    // Duress/Thoughtseize/tutor, and not yet moved to another zone). card_vocab_idx,
    // -1 = empty/unknown slot. Tracks the specific card, unlike the opp decklist
    // revealed bits, which are a match-scoped "ever seen" flag per card name.
    int  opp_known_hand[MAX_HAND_SLOTS];
    int  self_library_ct;
    int  opp_library_ct;

    // Known top-of-library cards (viewer's library only). Index 0 = top.
    // -1 = unknown.
    int known_top_library_self[KNOWN_TOP_LIBRARY_SIZE];

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

    // Global extras (serialized before the deck-identity tail blocks)
    bool is_day;               // game designation is day (CR 731.1)
    bool is_night;             // game designation is night
    int  pending_choice_kind;  // MandatoryChoice enum value (NONE = 0)
    // Priority-window context: the viewer's and the other seat's pass flags, and
    // whether the current decision is an ordinary priority window. All false unless
    // it is (see priority_window_open in game_driver.h).
    bool self_has_passed;
    bool opp_has_passed;
    bool is_priority_window;
    // Mulligan state (Game::pregame): mulligans each player has taken this game, and
    // the cards the viewer still has to bottom (0 outside the viewer's bottoming).
    // All 0 during a bo3 sideboard phase.
    int  self_mulligans_taken;
    int  opp_mulligans_taken;
    int  self_bottom_remaining;
    // The viewer is the starting player of the game this observation pertains to:
    // the current game in-game, the UPCOMING game during a bo3 sideboard phase.
    bool self_plays_first;
    // Sideboard-phase progress: swaps completed so far this phase, and the maindeck
    // drift from its size at phase start (0 or +1 — the IN-FIRST menu never cuts
    // first, so the deck is only ever oversized). Both 0 outside the phase.
    int  sideboard_swaps_made;
    int  sideboard_delta;

    // ── Deck-identity tail blocks (serialized last; see machine_io.h) ──────────
    // Each block is a list of (vocab id, count) slots, packed ascending by vocab
    // id with no holes; id = -1 marks an empty slot (count 0). counts are raw
    // integers (serialize_state normalizes by /4).
    //
    // Self LIVE library contents: the viewer's LIBRARY zone tallied live at
    // serialization time (cards leaving/returning to the library are reflected).
    // Viewer-only — the opponent's live library stays hidden.
    int self_live_library_id[DECKLIST_MAIN_SLOTS];
    int self_live_library_ct[DECKLIST_MAIN_SLOTS];
    // Viewer's OWN current 75 (deck_state's LIVE store): the deck CONFIGURATION,
    // every card regardless of zone, tracking each sideboard swap as it lands.
    // Distinct from the live-library block above (the LIBRARY ZONE, which shrinks
    // as you draw and is stale between games).
    int self_deck_main_id[DECKLIST_MAIN_SLOTS];
    int self_deck_main_ct[DECKLIST_MAIN_SLOTS];
    int self_deck_side_id[DECKLIST_SIDE_SLOTS];
    int self_deck_side_ct[DECKLIST_SIDE_SLOTS];
    // Opponent-of-viewer's REGISTERED decklist, frozen at the match's registered 75
    // (see deck_state.h); maindeck then sideboard.
    int opp_deck_main_id[DECKLIST_MAIN_SLOTS];
    int opp_deck_main_ct[DECKLIST_MAIN_SLOTS];
    int opp_deck_side_id[DECKLIST_SIDE_SLOTS];
    int opp_deck_side_ct[DECKLIST_SIDE_SLOTS];
    // Per opp decklist slot: 1 if the opponent has revealed that card this match
    // (match_state's reveal set, accumulated across the games of a bo3; a
    // double-faced card also counts when its back face was revealed).
    unsigned char opp_deck_main_revealed[DECKLIST_MAIN_SLOTS];
    unsigned char opp_deck_side_revealed[DECKLIST_SIDE_SLOTS];
} GameState;

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif /* GAMESTATE_H */
