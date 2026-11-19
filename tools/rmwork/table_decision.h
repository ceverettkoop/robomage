#ifndef RMWORK_TABLE_DECISION_H
#define RMWORK_TABLE_DECISION_H

#include <cstdint>
#include <string>
#include <vector>

#include "../../src/agents/decision_view.h"
#include "match.h"
#include "table_model.h"

// ── What a click on the Table answers: a decision's menu rows by the object or player they name ──
//
// Direct manipulation on the Table (docs/workbench_plan.md §4.2, items 6 and 11). Read from the
// viewer's STATE frame only (its menu, pending and pending_cost tables), so a seat view of the other
// seat's decision has none. Three kinds of decision are answered on the Table:
//   - TARGET: every row is SELECT_TARGET or DIVIDE_DAMAGE (a spell's or ability's targets, a trigger's
//     as it is put on the stack, one more point of divided damage). A target row names its candidate
//     as its source (a player as its source player);
//   - PAY: a PAYING_COSTS row in the menu: the interactive mana payment of --no-auto-mana (a mana
//     ability's row names its permanent) and a Phyrexian pip paid with life or mana;
//   - MANA: a priority decision offering mana abilities (MANA_* rows, --no-auto-mana): only those rows
//     are picks, so a click on a land taps it for mana.
// A row naming an object shown on the Table (a battlefield or hand card, a stack object) or a player
// is a pick of that object or player: a click on it answers with the row, or, when it names several
// (a dual land's two colours), offers them in a popup. A TARGET or PAY row naming nothing (Done
// selecting targets, Cancel casting, Pay 2 life, Pay {B}) is a button on the decision's prompt line in
// the middle strip, beside the prompt and the cost still to pay; a row naming an object the Table does
// not show (a card in a graveyard) is answered in Choices only.
//
// The frame lists no target chosen so far while a spell is being cast or an ability activated or put
// on the stack (its `stack_target` rows are written once the targets are announced), so the targets
// already chosen are read from the earlier decisions of the same targeting (chosen_targets).

namespace rmwork {

enum class DecisionKind {
    NONE,
    TARGET,
    PAY,
    MANA
};

struct DecisionPick {
        uint64_t handle = 0;    // an object's (a card on the Table, a stack object), 0 for a player
        int seat = -1;          // a player's: 0 A, 1 B
        std::vector<int> rows;  // the menu rows naming it, in menu order
        int so_far = -1;        // DIVIDE_DAMAGE: the points it has been given so far
};

// A target chosen earlier in the same targeting (`order` from 1).
struct ChosenTarget {
        uint64_t handle = 0;
        int seat = -1;
        int order = 0;
        std::string label;  // the chosen row's description
};

struct TableDecision {
        DecisionKind kind = DecisionKind::NONE;
        std::string prompt;                 // "Casting Prismari Charm: choose a target"
        uint64_t source = 0;                // the spell or ability chosen or paid for (`pending.source`), 0 none
        std::vector<std::string> row_text;  // every menu row's description
        std::vector<DecisionPick> picks;    // in menu order of their first row
        std::vector<int> buttons;           // the rows the prompt line offers as buttons, in menu order
        std::vector<std::string> pips;      // the cost still to pay as mana symbols: "2", "W", "B/P", "U/R", "X", "S"
        std::string owed;                   // the rest of it and what is paid: "2 life · 1 mana paid"
        std::vector<ChosenTarget> chosen;   // TARGET: the targets chosen so far (chosen_targets)

        // The prompt line shows (a TARGET or PAY decision).
        bool prompted() const { return kind == DecisionKind::TARGET || kind == DecisionKind::PAY; }
        // The pick of object `handle` / player `seat`, nullptr when none.
        const DecisionPick *pick_of_object(uint64_t handle) const;
        const DecisionPick *pick_of_seat(int seat) const;
        // Object `handle` / player `seat` was chosen earlier in this targeting: its order, else 0.
        int chosen_order_of_object(uint64_t handle) const;
        int chosen_order_of_seat(int seat) const;
};

// The decision of `v`, a STATE frame of the Table's viewer, against the Table's model of it (the
// objects and stack objects it shows); NONE when the frame has no menu or the menu is of another kind.
TableDecision read_table_decision(const agents::DecisionView &v, const TableModel &m);

// The targets chosen so far in the targeting decision `index` of `m` (in `view`'s frames): the rows
// chosen at the decisions just before it of the same seat for the same source (`source`), as long as
// they chose a target (SELECT_TARGET) or a point of divided damage (DIVIDE_DAMAGE, skipped), oldest
// first; a row naming nothing (Done) is no target.
std::vector<ChosenTarget> chosen_targets(const Match &m, size_t index, int view, uint64_t source);

}  // namespace rmwork

#endif /* RMWORK_TABLE_DECISION_H */
