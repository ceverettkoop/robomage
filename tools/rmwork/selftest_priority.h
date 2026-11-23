#ifndef RMWORK_SELFTEST_PRIORITY_H
#define RMWORK_SELFTEST_PRIORITY_H

#include <functional>

#include "app.h"
#include "selftest_table.h"

// ── The checks of priority decisions answered on the Table (table_decision.h's PRIORITY) ─────────
//
// rmwork --selftest (check_priority_units): the costs priority_costs.h reads, on script costs and
// oracle text (a fetchland's, a token's sacrifice, loyalty, a cycling cost, Uro's escape, a flashback
// with and without mana, Daze's and Force of Will's alternative costs, suspend), and the labels.
//
// rmwork --selftest-data (check_table_priority), through the App as the UI drives it (headless frames,
// simulated clicks found from the Table's layout) on sculpted live games as seat A against explore
// (selftest_live.h); each reply is checked to be the row naming what was clicked:
//   - a Mountain played by a click on it in the hand (the hand's lands and castable cards lit, the
//     prompt saying what may be done, Pass on the prompt line); Lightning Bolt cast by a click ({R}
//     its cost), its target then clicked on the Table; Pass by the prompt line's button with the Bolt on
//     the stack ("Respond to your Lightning Bolt"); a later priority decision passed by the Space key;
//   - Daze: its two ways in a popup (the normal cast {1}{U}, the alternative cost's "return an Island
//     …"), the alternative chosen there, its Island return the next decision;
//   - Polluted Delta's ability activated by a click ({T}, pay 1 life, sacrifice it), its search once it
//     resolves;
//   - Wasteland under --no-auto-mana: its mana row and its ability in one popup, the ability chosen,
//     its target clicked on the Table;
//   - Cabal Therapy's flashback from the graveyard by the prompt line's button (its cost: sacrifice a
//     creature);
//   - the first game in Replay: the same lights read-only, the Bolt marked chosen, Pass marked chosen
//     where it was, a click answering nothing; seat B's view without them.

namespace rmwork {

void check_priority_units(const CheckFn &check);
void check_table_priority(const CheckFn &check, App &app, const std::function<void(int)> &frames);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_PRIORITY_H */
