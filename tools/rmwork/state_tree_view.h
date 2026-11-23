#ifndef RMWORK_STATE_TREE_VIEW_H
#define RMWORK_STATE_TREE_VIEW_H

#include <set>
#include <string>
#include <vector>

#include "attribution.h"
#include "links.h"
#include "state_tree.h"
#include "tool_data.h"

// ── The State Tree tool ──────────────────────────────────────────────────────────────────────────
//
// The current frame's tree (state_tree.h) as an ImHex-style table: Name (indented, ▶ / ▼ on a node
// with children; a click on a row expands or collapses it), Col (a colour swatch for objects), Zone,
// Type, Value. Which nodes are expanded is kept by their keys across decisions and viewers. Hovering
// an object's row links it (the Table outlines it, Choices shades its rows); a row whose object or
// player is linked from elsewhere is shaded. When the links' item (the hover, else the selection)
// changes to one another tool reported, the tree reveals its row: an object's (a menu row's source,
// else its target), a player's; the nodes above it are expanded and, when the row is off-screen,
// the view scrolls to centre it. The tree's own hover reveals nothing, and the user's scrolling is
// left alone until the item changes again. With an attribution of the decision (attribution.h), the
// Impact column shows each node's value for the current target as a heat swatch and a number: an
// object's or stack object's node its token's (or its card's), a player's node the player's, and in the
// attributed seat's own frame a table row's token and an object field's cell; "sort children by
// impact" orders each node's children by |value|, the nodes with none after.

namespace rmwork {

class StateTreeView {
    public:
        StateTreeView();
        void draw(ToolData &data, LinkState *links);
        const StateTreeModel &model() const { return model_; }
        // Builds the model of the current decision now (the checks read it without drawing).
        void refresh(const ToolData &data);
        // Reveals the row of the links' item when it changed since the last call (draw calls it).
        void follow_links(const LinkState &links);
        // The key of the row the next draw scrolls to ("" for none), and of the last row revealed
        // with whether it was on screen once drawn.
        const std::string &scroll_target() const { return scroll_key_; }
        const std::string &revealed() const { return revealed_key_; }
        bool revealed_on_screen() const { return revealed_on_screen_; }
        // The checks: scroll the tree to its end at the next draw.
        void force_scroll_to_end() { scroll_to_end_ = true; }
        std::set<std::string> &expanded() { return open_; }
        // Screenshots: expand the players' zones, the stack, the frame's tables and its object table.
        void expand_overview();
        // The rows shown (the last refresh with the current expansion; sorted by impact when asked).
        const std::vector<TreeRow> &rows();
        // The attribution the Impact column shows (null: none), for the next draw.
        void set_impact(const AttrTints *t);
        // Each node's children sorted by |impact|.
        void set_sort_by_impact(bool on);
        bool sort_by_impact() const { return sort_impact_; }
        // A row's impact (false when it has none).
        bool impact_of(const TreeRow &r, double *v) const;

    private:
        std::vector<std::string> link_path(const LinkItem &item, const std::vector<MenuLink> &menu) const;
        static bool row_on_screen();

        void sort_rows();

        StateTreeModel model_;
        AttrTints impact_;
        bool sort_impact_ = false;
        std::vector<uint16_t> table_ids_;  // per frame table index its id
        LinkItem followed_;   // the links' item the tree last followed
        LinkItem own_hover_;  // what the tree hovered in its last draw
        std::string scroll_key_, revealed_key_;
        bool revealed_on_screen_ = false;
        bool scroll_to_end_ = false;
        unsigned generation_ = ~0u;
        std::set<std::string> open_;
        std::vector<TreeRow> rows_;
        bool rows_dirty_ = true;
};

}  // namespace rmwork

#endif /* RMWORK_STATE_TREE_VIEW_H */
