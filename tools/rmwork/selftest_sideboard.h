#ifndef RMWORK_SELFTEST_SIDEBOARD_H
#define RMWORK_SELFTEST_SIDEBOARD_H

#include <functional>

#include "app.h"
#include "selftest_play.h"

// ── rmwork --selftest-data's Sideboard tool checks (selftest_play's Play checks run them) ────────
//
//   - the plan's rules (the phase's locks, balance, the swap cap, undo) and next_sideboard_step on
//     made-up decisions: the row by category and card, Done at the target, a menu without the row the
//     plan needs refused with its error;
//   - the guides file read from the repository against its JSON: a known matchup's row, a deck's draw
//     land, a held-out pair, a non-league deck;
//   - a live bo3 as seat A (league/burn against explore's league/bug) to its first sideboarding
//     decision: the tool brought to the front with the seat's 75, the guide's plan for the matchup
//     (on the play or the draw as the frame says) and Use This Plan; a target the menu cannot reach
//     stopped before anything is sent; a planned swap (a sideboard card in, a deck card out)
//     confirmed and sent through the session (IN, OUT and Done rows), the next game's frame showing
//     the planned deck; the match played to its end;
//   - that match in Replay: the read-only tool at the phase's decision in seat A's view (the phase's
//     swaps from the frames), and in seat B's view (seat B's own 75 and a note, never seat A's).

namespace rmwork {

void check_sideboard(const CheckFn &check, App &app, const std::function<void(int)> &frames, const PlayTestEnv &env);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_SIDEBOARD_H */
