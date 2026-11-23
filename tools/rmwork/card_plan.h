#ifndef RMWORK_CARD_PLAN_H
#define RMWORK_CARD_PLAN_H

#include <cstdint>
#include <string>
#include <vector>

#include "../../src/agents/decision_view.h"

// ── Cards picked on the Table as a plan, then sent one reply per decision ──────────────────────
//
// Some decisions pick several cards one decision at a time, a row per card and no row taking a card
// back: the London mulligan's bottom picks (CR 103.5, one BOTTOM_DECK_CARD decision per card;
// mulligan_plan.h) and a discard of several cards (CR 514.1's cleanup discard down to the maximum hand
// size, an escalate cost's discards: one DISCARD decision per card). On the Table
// (docs/workbench_plan.md §4.2, items 5 and 13) the human plans them: a click on a lit card adds it (in
// order), a click on a planned card takes it back, Undo the last, Clear all; once the plan holds every
// card owed, the prompt line's send button sends it one reply per decision (next_planned_row: the row
// naming the first planned card still offered).

namespace rmwork {

// The decision sequence a plan belongs to: the game, the turn (-1 for the opening hand) and what it
// picks for.
struct CardPlanKey {
        int game = -1;
        long turn = -1;
        std::string purpose;  // "bottom", "discard"
        bool operator==(const CardPlanKey &o) const { return game == o.game && turn == o.turn && purpose == o.purpose; }
        bool operator!=(const CardPlanKey &o) const { return !(*this == o); }
};

class CardPlan {
    public:
        // Keeps the plan when it is `key`'s, else starts an empty one for it.
        void follow(const CardPlanKey &key);
        const CardPlanKey &key() const { return key_; }
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
        CardPlanKey key_;
        std::vector<uint64_t> cards_;
        bool sending_ = false;
        long answered_ = -1;
        unsigned generation_ = 0;
};

// The reply that carries `plan` on at `v`, the human's decision, when every row of its menu is of
// `category`: the row naming the first planned card it offers (its card in `*card`); -1 when the decision
// is another or offers none (the plan is sent).
int next_planned_row(const CardPlan &plan, const agents::DecisionView &v, const char *category, uint64_t *card);

}  // namespace rmwork

#endif /* RMWORK_CARD_PLAN_H */
