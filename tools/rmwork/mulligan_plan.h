#ifndef RMWORK_MULLIGAN_PLAN_H
#define RMWORK_MULLIGAN_PLAN_H

#include <cstdint>

#include "../../src/agents/decision_view.h"
#include "card_plan.h"

// ── The London mulligan's bottom picks, planned on the Table, then sent ─────────────────────────
//
// CR 103.5: a player who keeps a hand after N mulligans puts N cards from it on the bottom of their
// library. The engine asks one BOTTOM_DECK_CARD decision per card (no pending flow: a row per hand
// card, hand cards never collapsed), the player's `to_bottom` the cards still owed, and no row takes
// a card back. So on the Table (docs/workbench_plan.md §4.2, item 5) the human plans them as a
// CardPlan (card_plan.h) and "Put N on the bottom" sends it one reply per decision (next_bottom_row).

namespace rmwork {

// The reply that carries `plan` on at `v`, the human's decision: at a London BOTTOM_DECK_CARD decision
// the row naming the first planned card it offers (its card in `*card`); -1 when the decision is another
// or offers none (the plan is sent).
int next_bottom_row(const CardPlan &plan, const agents::DecisionView &v, uint64_t *card);

// Whether `v`'s menu is the London mulligan's bottom pick: every row BOTTOM_DECK_CARD, no pending flow.
bool london_bottom_menu(const agents::DecisionView &v);

}  // namespace rmwork

#endif /* RMWORK_MULLIGAN_PLAN_H */
