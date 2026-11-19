#ifndef RMWORK_SELFTEST_TABLE_H
#define RMWORK_SELFTEST_TABLE_H

#include <functional>
#include <string>

#include "app.h"
#include "match.h"

// ── The Table's checks, shared by --selftest (engine-free) and --selftest-data ──────────────────

namespace rmwork {

// Records one check: `ok`, else the failure `what`.
using CheckFn = std::function<void(bool ok, const std::string &what)>;

// What the frame checks met, so a run can say it covered attachments, stacks, blocks …
struct TableCoverage {
        int frames = 0;
        int identical = 0;  // groups of two or more interchangeable piles in a model
        int stacked = 0;    // piles stacked by a layout (the narrow panel)
        int wrapped = 0;    // rows a layout wrapped
        int attachments = 0;
        int stack_targets = 0;
        int blocks = 0;
        int counters = 0;
        int tapped = 0;
        int hidden = 0;            // battlefield or hand objects a seat frame shows face down
        int hover_checks = 0;      // stack objects hovered (check_table_drawing)
        int magnifier_checks = 0;  // cards magnified and menu rows lit
        int step_checks = 0;       // Replay's step bar clicked (it toggles nothing)
};

// Every decision of `m` (and its terminal frame) in every view: the Table model of the viewer's
// frame against the frame's own tables (every battlefield object placed exactly once, attachments
// in their host's pile; tapped, counters, P/T, damage carried; the stack in order with its targets
// resolved; no name the frame does not give), and the layout inside the panel at four panel sizes
// (a narrow one among them): a stack's members interchangeable and identified, a stack's count its
// size, stacks and wrapped rows only where the full-detail cards would be narrower than
// LEGIBLE_CARD_W, a row wrapped only when it has no group left to stack.
void check_table_frames(
    const CheckFn &check, TableView &view, const Match &m, const std::string &at, TableCoverage *cov);

// Draws the loaded match's every decision in every view (headless), then the interactions: hovering
// a stack object lights its targets, Q over a card opens the magnifier until released, a right-click
// one closes when the mouse leaves, a click on the step bar toggles no stop in Replay (each where
// the match has a decision to try it at: counted in `cov`). `frames(n)` runs n UI frames.
void check_table_drawing(
    const CheckFn &check, App &app, const std::function<void(int)> &frames, const std::string &at, TableCoverage *cov);

// Play's step bar on a live game (the Table in Play, at a decision of the human's): a click on a
// step toggles its stop for the side whose turn it is, the settings file `settings_path` (the App
// persists) has it, a second click turns it back.
void check_step_bar_clicks(
    const CheckFn &check, App &app, const std::function<void(int)> &frames, const std::string &settings_path);

// Engine-free: the layout and fit of synthetic models (stacking identical permanents and wrapping
// rows only when a narrow panel needs it), the step bar's cells, the magnifier's rectangles, the
// stops a step toggles, the links, the background tiles.
void check_table_units(const CheckFn &check);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_TABLE_H */
