#ifndef RMWORK_REPLAY_TOOLS_H
#define RMWORK_REPLAY_TOOLS_H

#include <string>

#include "choices_view.h"
#include "inspector_view.h"
#include "layout.h"
#include "links.h"
#include "players_view.h"
#include "sideboard_view.h"
#include "state_tree_view.h"
#include "tool_data.h"
#include "value_views.h"

// ── The tools beside the Table and the Game Log ─────────────────────────────────────────────────
//
// Players, Choices, Decision Inspector, State Tree, Swings, Value Graph, Model's Read and Sideboard,
// and the zone windows the Players tool opens; each reads the cursor's decision in the current viewer's
// frame (ToolData) of the session it is given: Replay's, or Play's live one (whose viewer is the
// human's seat). While the human chooses (set_choosing), Choices submits rows. begin_frame, before the tools draw,
// reads the decision and hands the links its menu and stack targets, so the Table, Choices, the State Tree and the zone
// windows link the same handles whichever of them are open. Also the toolbar's next swing and the status bar's V.

namespace rmwork {

class ReplayTools {
    public:
        void begin_frame(ReplaySession &s, TableView &table, LinkState &links);
        // Draws tool `t`'s body (one of the tools above) for a loaded match.
        void draw(Tool t, ReplaySession &s, LinkState *links);
        // Choices submits the row chosen (Play, the human's pending decision at the cursor).
        void set_choosing(bool on) { choosing_ = on; }
        bool choosing() const { return choosing_; }
        // The magnify chord the Sideboard tool's cards follow.
        void set_magnify(ImGuiKeyChord k) { magnify_ = k; }
        // The zone windows (after the tools).
        void draw_windows(LinkState *links, ImGuiKeyChord magnify);

        // The decision the next swing the viewer sees goes to after the cursor, -1 none.
        long next_swing_target(const ReplaySession &s) const;
        bool next_swing(ReplaySession &s);
        // The status bar's V: the deciding seat's value of the shown policy ("V +0.42 (A)") when
        // the viewer sees it, else "V —".
        std::string status_value() const;

        ToolData &data() { return data_; }
        PlayersView &players() { return players_; }
        ChoicesView &choices() { return choices_; }
        InspectorView &inspector() { return inspector_; }
        StateTreeView &tree() { return tree_; }
        ValueGraphView &graph() { return graph_; }
        SideboardView &sideboard() { return sideboard_; }

    private:
        ToolData data_;
        PlayersView players_;
        ChoicesView choices_;
        InspectorView inspector_;
        StateTreeView tree_;
        ValueGraphView graph_;
        SwingsView swings_;
        ModelReadView model_;
        SideboardView sideboard_;
        bool choosing_ = false;
        ImGuiKeyChord magnify_ = 0;
};

}  // namespace rmwork

#endif /* RMWORK_REPLAY_TOOLS_H */
