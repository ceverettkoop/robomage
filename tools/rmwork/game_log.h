#ifndef RMWORK_GAME_LOG_H
#define RMWORK_GAME_LOG_H

#include <string>

#include "replay_session.h"

// ── The Game Log tool in Replay ──────────────────────────────────────────────────────────────────
//
// The viewer's log (match.h): the omniscient narrative, or in a seat view only that seat's own log
// (its frames' event tables), never the narrative. The current decision's lines are highlighted,
// the lines before it drawn normally and the later ones dimmed (Replay is analysis: what comes
// later stays readable, and a seat's later lines are still only what that seat saw). Clicking a
// line moves to its decision; the view scrolls to the current decision whenever the decision or
// the viewer changes.

namespace rmwork {

class GameLogView {
    public:
        void draw(ReplaySession &s);
        // The lines the view shows for `view` (the match's log of that viewer).
        static const std::vector<LogLine> &lines(const Match &m, int view);
        // The caption above the lines ("Seat A's log: …").
        static std::string caption(const Match &m, int view);

    private:
        unsigned drawn_generation_ = ~0u;
};

}  // namespace rmwork

#endif /* RMWORK_GAME_LOG_H */
