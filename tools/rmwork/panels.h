#ifndef RMWORK_PANELS_H
#define RMWORK_PANELS_H

#include <string>

#include "game_log.h"
#include "layout.h"
#include "replay_session.h"

// ── The tools' contents ─────────────────────────────────────────────────────────────────────────
//
// In Replay with a game loaded the Game Log shows the viewer's log (game_log.h) and the other tools
// say which game and decision they will show (their renderers are later stages); otherwise each
// tool shows what it will show. The Value Graph draws its empty axes.

namespace rmwork {

struct ToolContext {
        Mode mode = Mode::REPLAY;
        ReplaySession *replay = nullptr;  // Replay's session (nullptr in Play)
        GameLogView *log = nullptr;
        std::string loading;  // what the loader is doing, "" when idle
};

// Draws tool `t`'s body inside its window (between Begin and End).
void draw_tool_body(Tool t, const ToolContext &ctx);

}  // namespace rmwork

#endif /* RMWORK_PANELS_H */
