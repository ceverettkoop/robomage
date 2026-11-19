#ifndef RMWORK_STATE_TREE_H
#define RMWORK_STATE_TREE_H

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "decision_frame.h"

// ── The State Tree's model: one viewer's STATE frame as a tree, generic over its SCHEMA ──────────
//
// Like ImHex's Pattern Data: every node has a name, a colour swatch (an object's colours: one, two
// split, gold for more, colourless grey, a land tan), a zone, a type and a value. At the top, nodes
// shaped for reading: "game" (the header's and turn's columns), "player A" / "player B" (the player
// row's columns, the player's counters, then a node per zone with its objects: the battlefield by
// controller, the other zones by owner), "stack" (top first); then "frame tables": every table the
// frame holds (its row count), each row a child, each column of a row a child with its schema
// name, its storage and semantic tag as the type, and its value read by the tag — an enum or mask
// by the schema's string table, a card id by the CARDS frame's name, a handle as "#h" and the name
// of the object the viewer's own `object` table identifies with it (none otherwise), a row index,
// a seat, text. An object's node holds its object row's columns and its rows in the object's child
// tables (subtypes, keywords, counters, abilities, …); a stack object's its stack row's columns and
// its modes and targets.
//
// Children are made when a node is expanded (children()), so a frame is never formatted whole; a
// node's key is its path, the same for the same thing at every decision, so the view keeps what is
// expanded across decisions.

namespace rmwork {

struct TreeRow {
        enum class Kind {
            GAME,
            PLAYER,
            ZONE,
            OBJECT,
            STACK,
            STACK_ITEM,
            TABLES,
            TABLE,
            ROW,
            FIELD
        };
        Kind kind = Kind::FIELD;
        std::string key;  // the node's path
        std::string name;
        int depth = 0;
        bool expandable = false;
        std::string zone, type, value;
        bool card_type = false;  // the type is a card's type line
        bool swatch = false;
        uint8_t colors = 0;  // WUBRG bits of the swatch
        bool land = false;
        uint64_t handle = 0;  // the object the node stands for (an OBJECT or STACK_ITEM node, an object row)
        int seat = -1;        // the player a PLAYER node stands for
        // What the node's children are made from.
        int table = -1;  // index into the frame's tables
        int row = -1;    // a row of that table
        int zone_id = -1;
};

class StateTreeModel {
    public:
        // Reads `f`'s frame (which must outlive the model); `table` is its Table model (the stack's
        // names).
        void build(const DecisionFrame &f, const TableModel &table);
        bool ready() const { return frame_ != nullptr; }
        const std::vector<TreeRow> &roots() const { return roots_; }
        std::vector<TreeRow> children(const TreeRow &r) const;
        // The rows shown with the nodes in `open` expanded, depth first.
        std::vector<TreeRow> flatten(const std::set<std::string> &open) const;
        // The name the frame gives the object of `handle` ("" when it does not identify it).
        std::string object_name(uint64_t handle) const;
        // The keys from a root down to the node of the object of `handle` (a stack object's under
        // "stack", any other under its player's zone: the battlefield's controller, another zone's
        // owner), the node's own key last; empty when the frame has no such object.
        std::vector<std::string> object_path(uint64_t handle) const;
        // The key of player `seat`'s node (0 A, 1 B).
        static std::string player_key(int seat);
        // A cell's value as the tree shows it.
        std::string cell_text(const rmstate::DecodedTable &t, size_t row, size_t col) const;

    private:
        void add_fields(const rmstate::DecodedTable &t, size_t row, const std::string &key, int depth,
            std::vector<TreeRow> *out) const;
        void add_child_rows(const char *const *tables, size_t n, uint64_t handle, const std::string &key, int depth,
            std::vector<TreeRow> *out) const;
        std::vector<TreeRow> player_children(const TreeRow &r) const;
        std::vector<TreeRow> zone_children(const TreeRow &r) const;
        TreeRow object_row(size_t obj_row, const std::string &parent_key, int depth) const;
        std::string object_summary(size_t obj_row) const;
        int table_index(const char *name) const;
        const rmstate::DecodedTable *table(const char *name) const;
        int col(const rmstate::DecodedTable &t, const char *name) const;
        uint64_t cell(const rmstate::DecodedTable &t, size_t row, const char *name) const;

        const rmstate::DecodedFrame *frame_ = nullptr;
        const agents::SeatStream *stream_ = nullptr;
        const CardBook *book_ = nullptr;
        int viewer_ = VIEW_OMNI;
        std::vector<TreeRow> roots_;
        std::vector<std::pair<uint64_t, size_t>> objects_;  // handle -> object row, sorted
        std::vector<std::pair<uint64_t, std::string>> stack_names_;
};

}  // namespace rmwork

#endif /* RMWORK_STATE_TREE_H */
