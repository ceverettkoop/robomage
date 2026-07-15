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
    MANA_ABILITY = 1,  // legacy, unused — color-specific categories below are emitted instead
    SELECT_ATTACKER = 2,
    CONFIRM_ATTACKERS = 3,
    SELECT_BLOCKER = 4,
    CONFIRM_BLOCKERS = 5,
    ACTIVATE_ABILITY = 6,
    CAST_SPELL = 7,
    SELECT_TARGET = 8,
    PLAY_LAND = 9,
    OTHER_CHOICE = 10,
    MULLIGAN = 11,          // binary: 0=keep, 1=take mulligan
    BOTTOM_DECK_CARD = 12,  // select card index from hand to put on library bottom
    MANA_W = 13,            // tap for white mana
    MANA_U = 14,            // tap for blue mana
    MANA_B = 15,            // tap for black mana
    MANA_R = 16,            // tap for red mana
    MANA_G = 17,            // tap for green mana
    MANA_C = 18,            // tap for colorless mana
    SEARCH_LIBRARY = 19,    // select a card from a library search (index 0 = fail to find)
    TOP_LIBRARY = 20,       // select a card to place on top of library
    SHUFFLE = 21,           // shuffle a library
    PAYING_COSTS = 22,      // tap a source / pitch a card to pay costs (interactive payment)
    DIG_CHOICE = 23,        // choose a card from a dig (look at top N) ability
    SIDEBOARD_IN = 24,      // choose a card from sideboard to add to main deck
    SIDEBOARD_OUT = 25,     // choose a card from main deck to move to sideboard
    SIDEBOARD_DONE = 26,    // finish sideboarding
    // --- Categories below split out the former OTHER_CHOICE (10) catch-all so the
    //     model sees a distinct semantic per decision. OTHER_CHOICE remains the
    //     default/fallback for any choice not specifically classified. ---
    SACRIFICE_PERMANENT = 27,  // choose a permanent to sacrifice (cost or effect)
    RETURN_PERMANENT = 28,     // choose a permanent to return to its owner's hand
    CHOOSE_X = 29,             // choose the value of X for an X cost, or a delve exile count
                               // (delve count actions carry the spell as source entity;
                               //  an X ladder emits the null card-id sentinel)
    DISCARD = 30,              // choose a card to discard (cost, effect, or cleanup)
    CHOOSE_MODE = 31,          // choose a modal/charm mode
    CHOOSE_MANA_COLOR = 32,    // choose the color of a flexible mana producer
    PAY_UNLESS = 33,           // pay-or-decline of a "counter unless pay" cost
    NAME_CARD = 34,            // name a card
    CHOOSE_TYPE = 35,          // choose a creature type
    KEEP_LEGEND = 36,          // legend rule: choose which duplicate to keep
    ORDER_TRIGGERS = 37,       // order simultaneous triggers onto the stack
    CHOOSE_REPLACEMENT = 38,   // choose which replacement effect / dredge-or-draw to apply
    ATTACK_TARGET = 39,        // choose what a creature attacks (player or planeswalker)
    BLOCK_TARGET = 40,         // choose which attacker a blocker blocks
    OPTIONAL_YESNO = 41,       // optional yes/no confirmation
    PLAY_FREE = 42,            // play a card for free (e.g. cast from exile)
    SYLVAN_CHOICE = 43,        // Sylvan Library card pick / pay-4-life-or-top choice
    CHOOSE_CARD = 44,          // choose a card from a zone for a non-library zone-change
    ASSIGN_DAMAGE = 45,        // T3.10: attacker assigns lethal combat damage to a chosen blocker
    COMPANION = 46,            // CR 702.139: pay {3} to put your chosen companion from the sideboard into hand
    DONT_SHUFFLE = 47,         // decline the optional shuffle after a rearrange-top effect (Ponder etc.);
                               // paired with SHUFFLE so the two choices are distinct to the model
    KEEP_HAND = 48,            // opening mulligan: keep the current hand (paired with MULLIGAN's take-a-mulligan)
    EXILE_FROM_YARD = 49,      // exile a card from the graveyard to pay a cost (Escape); distinct from
                               // SACRIFICE_PERMANENT (a permanent leaving the battlefield)
};

static constexpr int ACTION_CATEGORY_MAX = 49;  // highest ActionCategory value

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
