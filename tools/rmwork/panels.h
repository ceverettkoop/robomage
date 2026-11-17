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
// In Replay with a game loaded the Table shows the decision as the viewer saw it (table_view.h), the
// Game Log the viewer's log (game_log.h) and the other tools theirs (replay_tools.h); otherwise each
// tool shows what it will show, and the Value Graph its empty axes.

namespace rmwork {

struct ToolContext {
        Mode mode = Mode::REPLAY;
        ReplaySession *replay = nullptr;  // Replay's session (nullptr in Play)
        GameLogView *log = nullptr;
        TableView *table = nullptr;
        LinkState *links = nullptr;
        ReplayTools *tools = nullptr;
        const Settings *settings = nullptr;
        std::string loading;  // what the loader is doing, "" when idle
};

// Draws tool `t`'s body inside its window (between Begin and End).
void draw_tool_body(Tool t, const ToolContext &ctx);

}  // namespace rmwork

#endif /* RMWORK_PANELS_H */
