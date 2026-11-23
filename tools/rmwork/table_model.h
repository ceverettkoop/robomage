#ifndef RMWORK_TABLE_MODEL_H
#define RMWORK_TABLE_MODEL_H

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../../src/agents/seat_frames.h"
#include "card_book.h"

// ── The Table's model: what one viewer's STATE frame puts on the table ─────────────────────────
//
// Built from one viewer's decoded STATE frame and that viewer's stream only (its CARDS frame through
// a CardBook), so the Table shows exactly what the viewer may see: an object's name comes from its
// own `card` column, an object the frame shows face down to the viewer has none. The bottom seat
// is the viewer (seat A for the omniscient viewer), the top seat its opponent.
//
// Battlefield objects go into four rows: the top seat's lands (the far edge), the top seat's other
// permanents, the bottom seat's other permanents and the bottom seat's lands. A land is an object
// whose post-layer types include Land and not Creature (an animated land stands with the
// creatures). A row holds piles, in entry (timestamp) order: a permanent with the Auras and
// Equipment attached to it (an attachment joins its host's pile, wherever its controller sits).
// Piles of one permanent that look the same at a glance share a `group` (interchangeable: the same
// card and face, or the same token; tapped or not alike; no counters, damage, loyalty or
// designations, or the same ones; the same P/T, colours, subtypes and summoning sickness; nothing
// attached; not face down, phased out, attacking or blocking). The model never stacks them: the
// layout (table_layout.h) stacks a group only when its row is too wide to show each. The bottom
// seat's hand is listed when the frame identifies its cards (a card it shows face down is a card
// back); the top seat's hand is a count.

namespace rmwork {

struct TableCounter {
        std::string kind;  // the label: "+1/+1", "-1/-1", "Lore", "Time"
        int count = 0;
};

struct TableObject {
        uint64_t handle = 0;
        int controller = 0;  // 0 A, 1 B
        int owner = 0;
        bool in_hand = false;
        // The face shown: card id and face (a token's card, a copy's copied card, a transformed
        // card's back face); -1 when the frame does not identify the card.
        int card = -1;
        bool back = false;
        std::string name;  // "" when not identified
        bool token = false, face_down = false;
        bool tapped = false, sick = false, phased_out = false;
        bool attacking = false, blocking = false, blocked = false;
        bool land = false, creature = false, planeswalker = false;
        uint8_t colors = 0;                 // post-layer, WUBRG bits
        std::vector<std::string> subtypes;  // post-layer
        bool has_pt = false;                // a creature (post-layer): power and toughness are shown
        int power = 0, toughness = 0;
        int printed_power = 0, printed_toughness = 0;  // the face's (a face-down permanent's 2/2)
        bool pt_differs = false;                       // from the printed
        int damage = 0;
        bool deathtouch_damage = false;
        int loyalty = -1;                    // loyalty counters, -1 for none
        std::vector<TableCounter> counters;  // every other kind
        std::vector<std::string> tags;       // designations: "Monstrous", "Exerted", "Named: Island"
        uint64_t attached_to = 0;
        // Play: a creature the human plans to attack with (table_combat.h), shown standing forward like
        // an attacker; never set by build_table_model.
        bool planned_attack = false;
};

// What one place in a row holds: `objects[0]` is the host (or a stack's front card); the rest are
// its attachments (in handle order) or, in a stack the layout made, the group's other members.
struct TablePile {
        std::vector<int> objects;  // indices into TableModel::objects
        bool stacked = false;      // identical permanents shown as one pile with a ×n count (layouts only)
        int group = -1;            // the model's: piles of one row with the same group are interchangeable; -1 none
};

enum TableRowId {
    ROW_TOP_LANDS,
    ROW_TOP_OTHER,
    ROW_BOTTOM_OTHER,
    ROW_BOTTOM_LANDS,
    ROW_COUNT
};

struct TablePlayer {
        int seat = 0;      // 0 A, 1 B
        std::string name;  // "Player A"
        std::string deck;  // the decklist's name, "" when the frame has none
        int64_t life = 0;
        int hand = 0, library = 0, graveyard = 0, exile = 0;
        int poison = 0, energy = 0;
        std::vector<TableCounter> counters;  // other player counters
        bool monarch = false, city_blessing = false;
        int speed = -1;                               // -1 none
        std::array<int, 6> pool{{0, 0, 0, 0, 0, 0}};  // floating mana: W, U, B, R, G, C
        // Player effects and emblems, a chip each: "Emblem: <card>", "Hexproof from blue", …
        std::vector<std::string> effects;
};

// The floating mana as symbols ("{U}{R}{R}"; "{R}×5" past four of one colour), "" when the pool is empty.
std::string pool_text(const TablePlayer &p);

struct StackTarget {
        bool player = false;
        uint64_t handle = 0;  // an object's (0: it left its zone)
        int seat = -1;        // a player's: 0 A, 1 B
        std::string label;    // "Player B", "Grizzly Bears", "Brainstorm (gone)", "a face-down card"
};

struct TableStackItem {
        uint64_t handle = 0;
        int controller = 0;
        int card = -1;  // the thumbnail's face; -1 none (a face-down spell, a trigger with no source)
        bool back = false;
        std::string name;  // "Lightning Bolt", "Ragavan (triggered ability)", "spell"
        std::string text;  // "Instant · X=2", "activated ability"
        std::vector<StackTarget> targets;
};

struct CombatBlock {
        uint64_t blocker = 0, attacker = 0;
        int assigned = 0;  // the attacker's combat damage assigned to the blocker so far
};

// What an attacking creature attacks: a player (target 0) or a planeswalker.
struct CombatAttack {
        uint64_t attacker = 0;
        uint64_t target = 0;   // a planeswalker's handle, 0 for a player
        int target_seat = -1;  // the player's: 0 A, 1 B
};

// One menu row's links: the object or player it acts with (source) and on (target).
struct MenuLink {
        uint64_t source = 0, target = 0;
        int source_seat = -1, target_seat = -1;  // 0 A, 1 B, -1 none
};

struct TableModel {
        int bottom = 0, top = 1;  // seats
        int viewer = 0;           // 0 A, 1 B, 2 omniscient
        std::vector<TableObject> objects;
        std::array<std::vector<TablePile>, ROW_COUNT> rows;
        std::vector<int> hand;  // the bottom seat's hand, indices into objects
        std::array<TablePlayer, 2> players;
        // The middle strip.
        bool pregame = false;               // the opening hands
        bool sideboarding = false;          // between games
        bool terminal = false;              // the match is over
        int turn = 0;                       // from 1; 0 before the first turn
        int active = -1;                    // 0 A, 1 B
        int decider = -1;                   // the deciding seat, -1 none
        int step = -1;                      // the stop step (settings.h's twelve), -1 none
        std::string step_name;              // "Declare blockers", "First-strike damage"
        std::string day_night;              // "Day", "Night", "" for neither
        std::vector<TableStackItem> stack;  // top first
        std::vector<CombatAttack> attacks;  // one per attacking creature, in the combat table's order
        std::vector<CombatBlock> blocks;
        std::vector<MenuLink> menu;  // one per menu row of the frame (none in a seat frame without the menu)

        // The object of `handle`, -1 when it is not on the table.
        int find(uint64_t handle) const;
        // What attacker `handle` attacks, nullptr when it is not attacking.
        const CombatAttack *attack_of(uint64_t handle) const;
        // Sorts the handle index (build_table_model does; a hand-built model calls it once filled).
        void index();

    private:
        std::vector<std::pair<uint64_t, int>> by_handle_;
};

// Builds the model of `frame`, a STATE frame of viewer `viewer` (0 A, 1 B, 2 omniscient) read with
// `stream`, whose CARDS frame `book` was built from; `decider` is the decision's deciding seat (-1
// none) and `pregame` says it is an opening-hand decision.
TableModel build_table_model(const agents::SeatStream &stream, const CardBook &book, const rmstate::DecodedFrame &frame,
    int viewer, int decider, bool pregame);

}  // namespace rmwork

#endif /* RMWORK_TABLE_MODEL_H */
