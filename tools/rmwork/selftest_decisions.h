#ifndef RMWORK_SELFTEST_DECISIONS_H
#define RMWORK_SELFTEST_DECISIONS_H

#include <functional>

#include "app.h"
#include "selftest_table.h"

// ── rmwork --selftest-data's checks of the decisions answered on the Table (table_decision.h) ────
//
// Through the App as the UI drives it (headless frames, simulated mouse clicks found from the Table's
// layout), on two sculpted live games as seat A against explore (the engine's --hand-a, --battlefield-a
// … flags through PlayConfig::setup; seat B holds only Islands):
//   - targeting (auto-mana on): Lightning Bolt's target answered by a click on a creature, Prismari
//     Charm's first target by a click on the opponent's chip, its second offered with the first marked
//     chosen and answered by the prompt line's Done button; each reply the row naming what was clicked;
//   - mana payment (--no-auto-mana): the cost still to pay as pips, a click on a dual land opening the
//     popup of its two rows and the popup's item answering, Dismember's Phyrexian pips answered by the
//     prompt's life and mana buttons, a click on a land paying, a click on an untapped land at a
//     priority decision tapping it for mana and the pool shown on the player;
//   - at every decision the Table answers, its picks only objects the human's frame shows on the Table
//     (its battlefield and hand, the stack) or players;
//   - the targeting game in Replay: the same highlights in seat A's view, none in seat B's, and a click on
//     a target selects it and answers nothing.

namespace rmwork {

void check_table_decisions(const CheckFn &check, App &app, const std::function<void(int)> &frames);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_DECISIONS_H */
