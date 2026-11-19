#ifndef RMWORK_SELFTEST_TOOLS_H
#define RMWORK_SELFTEST_TOOLS_H

#include <array>
#include <functional>
#include <string>

#include "app.h"
#include "selftest_table.h"

// ── The checks of the tools beside the Table, shared by --selftest and --selftest-data ──────────

namespace rmwork {

// What the checks met, so a run can say what it covered.
struct ToolsCoverage {
        int frames = 0;
        int menus = 0;     // frames whose menu the viewer sees
        int withheld = 0;  // a non-deciding seat's frames (nothing of the decision shown)
        int zone_cards = 0;
        int known_hand = 0;     // opponent's hand cards a seat knows without seeing them
        int known_library = 0;  // library cards a seat knows by its knowledge alone
        int linked_exile = 0;   // exiled cards with a noted source
        int pending = 0, pending_costs = 0;
        int trees = 0, tree_handles = 0;  // frames whose State Tree was walked; handle cells checked
        int links = 0;                    // menu rows linked both ways
        int tree_reveals = 0;             // links followed to their State Tree rows
        int tree_scrolls = 0;             // State Tree rows scrolled into view while drawn
        int drawn = 0;                    // decisions drawn with every tool open
        int episode_decisions = 0;        // decisions checked against their evaluations
};

// Every decision of the loaded match in every view, against the viewer's frame: the Players' counts,
// the zone windows' cards (exactly the frame's and its knowledge's), Choices (the menu rows, or none
// in a non-deciding seat's view; the chosen row the log's reply), the Inspector's fields, the State
// Tree (every table and row of the frame; a handle named only where the frame identifies it), the
// links between Choices, the Table and the State Tree (the same handles).
void check_tools_frames(const CheckFn &check, App &app, const std::string &at, ToolsCoverage *cov);

// Draws a sample of decisions in every view with every tool open (headless), hovers a Choices row
// (its objects lit for the Table), and opens, magnifies in and closes (Esc) zone windows.
void check_tools_drawing(
    const CheckFn &check, App &app, const std::function<void(int)> &frames, const std::string &at, ToolsCoverage *cov);

// What check_tools_values expects of the loaded match's evaluations.
struct ValuesExpect {
        size_t policies = 0;                // how many policies the episodes evaluate
        std::array<std::string, 2> agents;  // the agents the episodes name, per seat
        // Recorded by nn: agents on both seats: every decision evaluated by every policy, each π a
        // distribution over the menu rows.
        bool every_decision = false;
};

// The loaded match with evaluations in its episodes (`episode_files`, read again here): the Value
// Graph's series are the episodes' values (on Player A's scale; a seat view only its own seat's),
// Swings and seat disagreement recomputed independently, the toolbar's next swing, the
// Inspector's values and π, Choices' π bars (the shown policy's, else the first, and each policy's),
// the Model's Read; with `want.every_decision`, a series point and a π summing to 1 at every
// decision.
void check_tools_values(const CheckFn &check, App &app, const std::function<void(int)> &frames, const std::string &at,
    const ValuesExpect &want, ToolsCoverage *cov);

// Engine-free: the evaluations' series, swings, disagreements, turn bands and value text on a
// synthetic match.
void check_tools_units(const CheckFn &check);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_TOOLS_H */
