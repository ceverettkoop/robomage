#ifndef ACTION_H
#define ACTION_H

#include <string>

#include "../components/ability.h"
#include "../ecs/entity.h"

enum ActionType {
    PASS_PRIORITY,
    CAST_SPELL,
    ACTIVATE_ABILITY,
    SPECIAL_ACTION  // Includes: play land, turn face-up morph, etc.
};

// Semantic category of each legal action, emitted per-action in machine mode
// so the model can learn action semantics across varying game states.
enum class ActionCategory {
    PASS_PRIORITY = 0,
    SELECT_ATTACKER = 1,
    CONFIRM_ATTACKERS = 2,
    SELECT_BLOCKER = 3,
    CONFIRM_BLOCKERS = 4,
    ACTIVATE_ABILITY = 5,
    CAST_SPELL = 6,
    SELECT_TARGET = 7,
    PLAY_LAND = 8,
    OTHER_CHOICE = 9,
    MULLIGAN = 10,          // binary: 0=keep, 1=take mulligan
    BOTTOM_DECK_CARD = 11,  // select card index from hand to put on library bottom
    MANA_W = 12,            // tap for white mana
    MANA_U = 13,            // tap for blue mana
    MANA_B = 14,            // tap for black mana
    MANA_R = 15,            // tap for red mana
    MANA_G = 16,            // tap for green mana
    MANA_C = 17,            // tap for colorless mana
    SEARCH_LIBRARY = 18,    // select a card from a library search (index 0 = fail to find)
    TOP_LIBRARY = 19,       // select a card to place on top of library
    SHUFFLE = 20,           // shuffle a library
    PAYING_COSTS = 21,      // tap a source / pitch a card to pay costs (interactive payment)
    DIG_CHOICE = 22,        // choose a card from a dig (look at top N) ability
    SIDEBOARD_IN = 23,      // choose a card from sideboard to add to main deck
    SIDEBOARD_OUT = 24,     // choose a card from main deck to move to sideboard
    SIDEBOARD_DONE = 25,    // finish sideboarding
    // --- Dedicated choice categories, so the model sees a distinct semantic per
    //     decision. OTHER_CHOICE remains the default/fallback for any choice not
    //     specifically classified. ---
    SACRIFICE_PERMANENT = 26,  // choose a permanent to sacrifice (cost or effect)
    RETURN_PERMANENT = 27,     // choose a permanent to return to its owner's hand
    CHOOSE_X = 28,             // choose the value of X for an X cost, or a delve exile count
                               // (delve count actions carry the spell as source entity;
                               //  an X ladder emits the null card-id sentinel)
    DISCARD = 29,              // choose a card to discard (cost, effect, or cleanup)
    CHOOSE_MODE = 30,          // choose a modal/charm mode
    CHOOSE_MANA_COLOR = 31,    // choose the color of a flexible mana producer
    PAY_UNLESS = 32,           // pay-or-decline of a "counter unless pay" cost
    NAME_CARD = 33,            // name a card
    CHOOSE_TYPE = 34,          // choose a creature type
    KEEP_LEGEND = 35,          // legend rule: choose which duplicate to keep
    ORDER_TRIGGERS = 36,       // order simultaneous triggers onto the stack
    CHOOSE_REPLACEMENT = 37,   // choose which replacement effect / dredge-or-draw to apply
    ATTACK_TARGET = 38,        // choose what a creature attacks (player or planeswalker)
    BLOCK_TARGET = 39,         // choose which attacker a blocker blocks
    OPTIONAL_YESNO = 40,       // optional yes/no confirmation
    SYLVAN_CHOICE = 41,        // Sylvan Library card pick / pay-4-life-or-top choice
    CHOOSE_CARD = 42,          // choose a card from a zone for a non-library zone-change
    ASSIGN_DAMAGE = 43,        // T3.10: attacker assigns lethal combat damage to a chosen blocker
    COMPANION = 44,            // CR 702.139: pay {3} to put your chosen companion from the sideboard into hand
    DONT_SHUFFLE = 45,         // decline the optional shuffle after a rearrange-top effect (Ponder etc.);
                               // paired with SHUFFLE so the two choices are distinct to the model
    KEEP_HAND = 46,            // opening mulligan: keep the current hand (paired with MULLIGAN's take-a-mulligan)
    EXILE_FROM_YARD = 47,      // exile a card from the graveyard to pay a cost (Escape); distinct from
                               // SACRIFICE_PERMANENT (a permanent leaving the battlefield)
};

static constexpr int ACTION_CATEGORY_MAX = 47;  // highest ActionCategory value

struct LegalAction {
        ActionType type;
        Entity source_entity;  // Card/permanent being used (if applicable)
        Entity target_entity;  // Target entity (if applicable)
        Ability ability;       // Ability being activated (ACTIVATE_ABILITY only)
        std::string description;
        ActionCategory category = ActionCategory::OTHER_CHOICE;
        bool use_alt_cost = false;
        bool use_flashback = false;
        bool use_offspring = false;  // cast paying the Offspring additional cost (CR 702.171)
        bool use_escape = false;     // cast from graveyard paying the Escape cost (CR 702.139)
        bool impulse_cast = false;   // cast from exile under a cur_game.impulse_cast_permission, paying its alternative RESOURCE cost (energy/life) instead of mana (Amped Raptor)
        // PLAY_LAND of a modal DFC's BACK face: the source entity is the combined card (whose
        // CardData is the front face), but it is being played as its back face (a land). The
        // processor marks it pending_enters_transformed so it enters showing the back face.
        bool play_back_face = false;
        // CAST_SPELL of a modal DFC's BACK face when that back face is a NONLAND spell
        // (Tergrid, God of Fright // Tergrid's Lantern). The source entity is the combined
        // card (whose CardData is the front face); it is cast paying the BACK face's mana cost
        // and using the back face's characteristics/abilities (CR 712.8). If the back is a
        // permanent the processor marks it pending_enters_transformed so it enters showing the
        // back face (reusing the transform machinery, parallel to play_back_face for lands).
        bool cast_back_face = false;
        // SPECIAL_ACTION that puts the player's chosen Companion (CR 702.139) from the sideboard
        // into their hand for {3}. Disambiguates the companion special action from the play-land
        // special action; the processor pays {3} and moves the source entity Sideboard -> Hand.
        bool companion_to_hand = false;
        // SPECIAL_ACTION that begins the Suspend process (CR 702.62) for the source card in hand:
        // pay its suspend cost and exile it with N time counters. Disambiguates the suspend special
        // action from the play-land / companion special actions; the processor pays the cost, moves
        // the source entity Hand -> Exile, and records the time counters.
        bool suspend_action = false;
        // True when this choice's card identity is public knowledge to all players
        // (e.g. a revealed tutor like Personal Tutor). Lets observers show the card
        // name even for an otherwise-private choice (search/top-of-library).
        bool card_is_public = false;
        // Per-action ordinal/value scalar, serialized to ML ALONGSIDE `category`.
        // Disambiguates options that share a category and reference no distinct
        // entity (so they'd otherwise serialize identically): the mode index of a
        // modal spell, the chosen value of an X ladder, a flexible producer's color
        // index, a cast variant (normal/alt-cost/offspring/...), a top-of-library
        // placement depth, a binary pay/decline, etc. -1 = not applicable.
        int option_ordinal = -1;

        LegalAction(ActionType t, const std::string &desc)
            : type(t), source_entity(0), target_entity(0), description(desc) {}

        LegalAction(ActionType t, Entity source, const std::string &desc)
            : type(t), source_entity(source), target_entity(0), description(desc) {}

        LegalAction(ActionType t, Entity source, Entity target, const std::string &desc)
            : type(t), source_entity(source), target_entity(target), description(desc) {}

        LegalAction(ActionType t, Entity source, const Ability &ab, const std::string &desc)
            : type(t), source_entity(source), target_entity(0), ability(ab), description(desc) {}
};

#endif /* ACTION_H */
