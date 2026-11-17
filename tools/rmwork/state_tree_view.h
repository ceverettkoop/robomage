#ifndef RMWORK_STATE_TREE_VIEW_H
#define RMWORK_STATE_TREE_VIEW_H

#include <set>
#include <string>
#include <vector>

#include "links.h"
#include "state_tree.h"
#include "tool_data.h"

// ── The State Tree tool ──────────────────────────────────────────────────────────────────────────
//
// The current frame's tree (state_tree.h) as an ImHex-style table: Name (indented, ▶ / ▼ on a node
// with children; a click on a row expands or collapses it), Col (a colour swatch for objects), Zone,
// Type, Value. Which nodes are expanded is kept by their keys across decisions and viewers. Hovering
// an object's row links it (the Table outlines it, Choices shades its rows); a row whose object or
// player is linked from elsewhere is shaded.

namespace rmwork {

class StateTreeView {
    public:
        StateTreeView();
        void draw(ToolData &data, LinkState *links);
        const StateTreeModel &model() const { return model_; }
        // Builds the model of the current decision now (the checks read it without drawing).
        void refresh(const ToolData &data);
        std::set<std::string> &expanded() { return open_; }
        // Screenshots: expand the players' zones, the stack, the frame's tables and its object table.
        void expand_overview();
        // The rows shown (the last refresh with the current expansion).
        const std::vector<TreeRow> &rows();

    private:
        StateTreeModel model_;
        unsigned generation_ = ~0u;
        std::set<std::string> open_;
        std::vector<TreeRow> rows_;
        bool rows_dirty_ = true;
};

}  // namespace rmwork

#endif /* RMWORK_STATE_TREE_VIEW_H */
