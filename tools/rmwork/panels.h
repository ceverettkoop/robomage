#ifndef RMWORK_PANELS_H
#define RMWORK_PANELS_H

#include <string>

#include "game_log.h"
#include "layout.h"
#include "links.h"
#include "replay_session.h"
#include "replay_tools.h"
#include "settings.h"
#include "table_view.h"

// ── The tools' contents ─────────────────────────────────────────────────────────────────────────
//
// With a match in the mode's session (Replay's loaded match, or Play's live one) the Table shows the
// decision as the viewer saw it (table_view.h), the Game Log the viewer's log (game_log.h) and the
// other tools theirs (replay_tools.h); otherwise each tool shows what it will show, the Value Graph
// its empty axes, and Play's Table a New Game button.

namespace rmwork {

struct ToolContext {
        Mode mode = Mode::REPLAY;
        ReplaySession *replay = nullptr;  // the mode's session: Replay's, or Play's live one
        GameLogView *log = nullptr;
        TableView *table = nullptr;
        LinkState *links = nullptr;
        ReplayTools *tools = nullptr;
        const Settings *settings = nullptr;
        StopSettings *stops = nullptr;  // Play's stops, which the Table's step bar toggles
        std::string loading;       // what the loader (or Play's engine before its first frame) is doing, "" when idle
        bool *new_game = nullptr;  // Play's empty Table: set when its New Game button is pressed
};

// Draws tool `t`'s body inside its window (between Begin and End).
void draw_tool_body(Tool t, const ToolContext &ctx);

}  // namespace rmwork

#endif /* RMWORK_PANELS_H */
