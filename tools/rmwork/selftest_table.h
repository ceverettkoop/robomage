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
        int land_stacks = 0;
        int attachments = 0;
        int stack_targets = 0;
        int blocks = 0;
        int counters = 0;
        int tapped = 0;
        int hidden = 0;            // battlefield or hand objects a seat frame shows face down
        int hover_checks = 0;      // stack objects hovered (check_table_drawing)
        int magnifier_checks = 0;  // cards magnified and menu rows lit
};

// Every decision of `m` (and its terminal frame) in every view: the Table model of the viewer's
// frame against the frame's own tables (every battlefield object placed exactly once, attachments
// in their host's pile, stacked lands counted; tapped, counters, P/T, damage carried; the stack in
// order with its targets resolved; no name the frame does not give), and the layout inside the
// panel at three panel sizes.
void check_table_frames(
    const CheckFn &check, TableView &view, const Match &m, const std::string &at, TableCoverage *cov);

// Draws the loaded match's every decision in every view (headless), then the interactions: hovering
// a stack object lights its targets, Q over a card opens the magnifier until released, a right-click
// one closes when the mouse leaves (each where the match has a decision to try it at: counted in
// `cov`). `frames(n)` runs n UI frames.
void check_table_drawing(
    const CheckFn &check, App &app, const std::function<void(int)> &frames, const std::string &at, TableCoverage *cov);

// Engine-free: the layout and fit of synthetic models, the links, the background tiles.
void check_table_units(const CheckFn &check);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_TABLE_H */
