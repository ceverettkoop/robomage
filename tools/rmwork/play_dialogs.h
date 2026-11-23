#ifndef RMWORK_PLAY_DIALOGS_H
#define RMWORK_PLAY_DIALOGS_H

#include <string>
#include <utility>
#include <vector>

#include "match.h"

// ── Play's modal dialogs: the concession's confirmation and the match result ─────────────────────
//
// Each is opened by a request and drawn every frame at the frame's top level, like every dialog.
//
// Concede: Game → Concede Game (-2) or Concede Match (-3), or the toolbar's Concede…: the choice of
// the two, Concede or Cancel.
//
// Match over (plan §4.3, item 15): the winner and, in best of three, the score and each game's
// winner; your values through the match as Model's Read gave them (when it ran); Open in Replay (the
// game's decision log, through Replay's loader), New Game and Close.

namespace rmwork {

class ConcedeDialog {
    public:
        // Opens with Concede Match chosen (`match`) or Concede Game.
        void open(bool match);
        // True in the frame Concede was pressed (`*match` the choice).
        bool draw(bool *match);
        bool is_open() const { return open_ || want_open_; }

    private:
        bool want_open_ = false;
        bool open_ = false;
        bool match_ = false;
};

// What the match result screen shows.
struct MatchSummary {
        std::string title;                              // "You win", "You lose", "A draw"
        std::string score;                              // "2-1" (best of three), "" otherwise
        std::vector<std::string> games;                 // "Game 1: you won", …
        std::string log_path;                           // the decision log
        std::vector<std::pair<double, double>> values;  // (your decision, your V) in order
        std::string values_note;                        // why there is no curve
};

// The summary of finished live match `m` for the human at seat `seat`.
MatchSummary summarize_match(const Match &m, int seat);

class GameOverDialog {
    public:
        enum class Choice {
            NONE,
            OPEN_REPLAY,
            NEW_GAME
        };
        void open(const MatchSummary &s);
        // The button pressed this frame (the dialog closes on any).
        Choice draw();
        bool is_open() const { return open_ || want_open_; }
        const MatchSummary &summary() const { return s_; }

    private:
        bool want_open_ = false;
        bool open_ = false;
        MatchSummary s_;
};

}  // namespace rmwork

#endif /* RMWORK_PLAY_DIALOGS_H */
