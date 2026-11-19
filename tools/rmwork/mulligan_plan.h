#ifndef RMWORK_MULLIGAN_PLAN_H
#define RMWORK_MULLIGAN_PLAN_H

#include <cstdint>
#include <vector>

#include "../../src/agents/decision_view.h"

// ── The London mulligan's bottom picks, planned on the Table, then sent ─────────────────────────
//
// CR 103.5: a player who keeps a hand after N mulligans puts N cards from it on the bottom of their
// library. The engine asks one BOTTOM_DECK_CARD decision per card (no pending flow: a row per hand
// card, hand cards never collapsed), the player's `to_bottom` the cards still owed, and no row takes
// a card back. So on the Table (docs/workbench_plan.md §4.2, item 5) the human plans them: a click on
// a hand card adds it (in order), a click on a planned card takes it back, Undo the last, Clear all;
// once the plan holds every card owed, "Put N on the bottom" sends it one reply per decision
// (next_bottom_row: the row naming the first planned card still in the hand).

namespace rmwork {

class BottomPlan {
    public:
        // Keeps the plan when it is game `game`'s, else starts an empty one.
        void follow(int game);
        void clear();
        bool empty() const { return cards_.empty(); }
        const std::vector<uint64_t> &cards() const { return cards_; }
        // Moves at every change.
        unsigned generation() const { return generation_; }
        // Card `h`'s place in the plan (from 1), 0 when it is not planned.
        int order_of(uint64_t h) const;
        // Plans card `h` (false when `owed` are planned already), or takes it back when it is planned.
        bool toggle(uint64_t h, int owed);
        // Takes the last planned card back; false when none is.
        bool undo();

        bool sending() const { return sending_; }
        void start_sending() { sending_ = true; }
        // The decision the last reply answered (-1 none).
        long answered() const { return answered_; }
        void record(long index) { answered_ = index; }

    private:
        int game_ = -1;
        std::vector<uint64_t> cards_;
        bool sending_ = false;
        long answered_ = -1;
        unsigned generation_ = 0;
};

// The reply that carries `plan` on at `v`, the human's decision: at a London BOTTOM_DECK_CARD decision
// the row naming the first planned card it offers (its card in `*card`); -1 when the decision is another
// or offers none (the plan is sent).
int next_bottom_row(const BottomPlan &plan, const agents::DecisionView &v, uint64_t *card);

// Whether `v`'s menu is the London mulligan's bottom pick: every row BOTTOM_DECK_CARD, no pending flow.
bool london_bottom_menu(const agents::DecisionView &v);

}  // namespace rmwork

#endif /* RMWORK_MULLIGAN_PLAN_H */
