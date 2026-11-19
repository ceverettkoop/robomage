#ifndef RMWORK_SELFTEST_CHOICES_H
#define RMWORK_SELFTEST_CHOICES_H

#include <functional>

#include "app.h"
#include "selftest_table.h"

// ── rmwork --selftest-data's checks of modes, numbers, names, costs and pickers on the Table ─────
//
// docs/workbench_plan.md §4.2, items 9, 10, 12 and 13. Through the App as the UI drives it (headless
// frames, simulated clicks found from the Table's layout, the choice panel's and the card picker's
// items), on sculpted live games as seat A against explore (selftest_live.h). Each answer's reply is
// checked to be the row naming what was clicked, and a frame after it to show the effect:
//   - Kozilek's Command: X by the prompt line's stepper (+ twice, then "X = 2"; the pips the value
//     leaves), two modes clicked in the choice panel (the second decision's panel marking the first
//     chosen, with its target), the spell with X 2 on the stack, Player B's two Spawn tokens once it
//     resolves;
//   - Collective Brutality: a mode, then "Done choosing modes" by the prompt line's button (the bears die);
//   - Fury: its evoke cost's exile by a click on a hand card, its two triggers ordered in the panel,
//     the targets clicked, a point of divided damage given by a click (the next decision's count);
//   - Cabal Therapy: the name chosen in the panel after typing in its search box (the opponent
//     discards it), then its flashback's sacrifice by a click on a creature;
//   - Lightning Bolt then Daze: the Island returned by a click, the unless cost declined by the prompt
//     line's button; a second Bolt then Force of Will: the blue card exiled by a click on the hand;
//   - Murktide Regent: the delve count by the stepper, a graveyard card exiled in the card picker;
//   - Lórien Revealed, then the cleanup discard of two cards planned on the hand and sent;
//   - Thoughtseize: the opponent's revealed hand in the card picker (its lands dimmed), a card picked;
//   - Watery Grave's optional life payment accepted by the prompt line's button (life 18, untapped),
//     then the legend rule's choice by a click on the Thalia kept;
//   - the Kozilek's Command game in Replay: the stepper at the X chosen, the panel read-only with the
//     mode chosen marked, a click answering nothing; seat B's view without them.

namespace rmwork {

void check_table_choices(const CheckFn &check, App &app, const std::function<void(int)> &frames);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_CHOICES_H */
