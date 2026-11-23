#ifndef RMWORK_SIMULATION_VIEW_H
#define RMWORK_SIMULATION_VIEW_H

#include <string>
#include <vector>

#include "simulation.h"

// ── The Simulations window: a batch's progress and results (plan §2.9) ──────────────────────────
//
// A floating window (not a docked tool: it belongs to no mode's layout, and it stays over Replay while
// a match chosen from it is looked at), opened when a batch of several matches starts, from File →
// Simulations and File → Recent Simulations. At the top a combo box of the past simulations (the
// simulations directory's newest first; choosing one opens it), New… (the Simulate dialog), Open
// Folder… (the file picker on directories) and, while a batch runs, Cancel. Then the entrants, the
// format and settings, the directory, a progress bar ("7 / 10 matches · 3 running · 0:42"), the summary
// (per entrant the matches and games won; draws, failures; with alternating seats entrant 1's score
// and its 95% Wilson interval, without it the wins by seat) and a row per match: number, seed, seat A's
// and seat B's entrant, result, games, decisions, time. A click on a finished match's row opens its
// decision log in Replay; on a failed one, the error dialog with the match's error and stderr tail.

namespace rmwork {

// What the window asks the App to do.
struct SimulationRequest {
        enum class Kind {
            NONE,
            OPEN_LOG,   // `path` in Replay
            FAILURE,    // match `match`'s failure in the error dialog
            NEW,        // the Simulate dialog
            OPEN_PAST,  // the simulation in directory `path`
            PICK,       // the file picker on directories
            CANCEL      // cancel the running batch
        };
        Kind kind = Kind::NONE;
        std::string path;
        size_t match = 0;
};

class SimulationView {
    public:
        // Opens it (brought to the front) and lists the past simulations under `root` again.
        void open(const std::string &root);
        void close() { open_ = false; }
        bool is_open() const { return open_; }
        // Lists the past simulations again (after a batch ended).
        void refresh();
        const std::vector<SimulationEntry> &past() const { return past_; }
        // Draws it while open; what was asked for this frame.
        SimulationRequest draw(const Simulation &sim);
        // The summary's text lines as drawn (for the checks).
        static std::vector<std::string> summary_lines(const Simulation &sim);

    private:
        void draw_header(const Simulation &sim, SimulationRequest *req);
        void draw_summary(const Simulation &sim);
        void draw_matches(const Simulation &sim, SimulationRequest *req);
        bool open_ = false;
        bool appearing_ = false;
        std::string root_;
        std::vector<SimulationEntry> past_;
};

}  // namespace rmwork

#endif /* RMWORK_SIMULATION_VIEW_H */
