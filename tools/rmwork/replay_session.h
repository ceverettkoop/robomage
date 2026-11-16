#ifndef RMWORK_REPLAY_SESSION_H
#define RMWORK_REPLAY_SESSION_H

#include <memory>
#include <string>

#include "match.h"

// ── Replay's loaded match, its cursor and viewer ────────────────────────────────────────────────
//
// The decision the tools show (the cursor, 0-based) and the viewer whose frames they read (A, B or
// omniscient). Autoplay steps the cursor every `interval` seconds and stops at the last decision.
// The status bar's fields come from the cursor's decision (its omniscient frame's turn and step,
// public facts every viewer shares).

namespace rmwork {

class ReplaySession {
    public:
        inline static constexpr double DEFAULT_INTERVAL = 0.6;

        bool loaded() const { return match_ != nullptr; }
        const Match *match() const { return match_.get(); }
        Match *match() { return match_.get(); }
        // Shows `m` at decision `cursor` (clamped); autoplay stops.
        void set_match(std::unique_ptr<Match> m, size_t cursor);
        void clear();

        size_t cursor() const { return cursor_; }
        size_t count() const { return match_ ? match_->decisions.size() : 0; }
        const DecisionInfo *current() const;
        // Moves the cursor (clamped); false when nothing is loaded.
        bool go_to(size_t index);
        bool step(int delta);
        bool first() { return go_to(0); }
        bool last() { return go_to(count() == 0 ? 0 : count() - 1); }

        int view() const { return view_; }
        void set_view(int view);

        bool playing() const { return playing_; }
        void set_playing(bool on, double now);
        // Advances autoplay to `now` (seconds).
        void tick(double now);
        double interval = DEFAULT_INTERVAL;

        // The status bar's fields: "Decision 12 / 89", "Game 1 · Turn 3 · First main",
        // "Player A decides"; "—" forms when nothing is loaded.
        std::string decision_text() const;
        std::string turn_text() const;
        std::string decider_text() const;
        // The game combo's text for the cursor: "Game 2", "Sideboarding 2", "No game loaded".
        std::string game_label() const;

        // Bumped whenever the cursor or the viewer changes (the Game Log follows it).
        unsigned generation() const { return generation_; }

    private:
        std::unique_ptr<Match> match_;
        size_t cursor_ = 0;
        int view_ = VIEW_OMNI;
        bool playing_ = false;
        double next_step_at_ = 0;
        unsigned generation_ = 0;
};

}  // namespace rmwork

#endif /* RMWORK_REPLAY_SESSION_H */
