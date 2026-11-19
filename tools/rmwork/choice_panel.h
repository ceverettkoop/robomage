#ifndef RMWORK_CHOICE_PANEL_H
#define RMWORK_CHOICE_PANEL_H

#include <string>
#include <vector>

#include "../../src/agents/decision_view.h"
#include "card_book.h"
#include "match.h"
#include "table_decision.h"

// ── The choice panel's model: a decision's rows that are neither a card nor a button ─────────────
//
// docs/workbench_plan.md §4.2, items 9 and 10 (mock-ups decisions.html#modes-numbers, -order, #name-card).
// The rows of a decision the Table answers (table_decision.h) that its picks and the prompt line do not
// hold (TableDecision::panel_rows), as a list in a window over the opponent's half, as the card
// picker's is:
//   - MODES: a modal spell's modes, in the card's order (each row's `ordinal`, its mode index). The
//     engine asks one mode per decision and leaves the modes chosen out of the next menu, so the modes
//     chosen earlier in this cast are read from the deciding seat's decisions just before (the same
//     pending source) and listed, marked chosen with the targets chosen for them. "Done choosing modes"
//     is a button of the prompt line and of the panel's foot;
//   - ORDER: the player's simultaneous triggers waiting to go on the stack (CR 603.3b), each with its
//     source card; the one chosen goes on the stack now, above those put there before (the stack shows
//     them), so the last put there resolves first;
//   - NAME: a name-a-card decision's candidates (the engine's limited set, src/name_card_choices.h, or
//     every vocab card under --closed-decklist), grouped by the DECKLISTS frame's lists (the
//     opponent's main deck and sideboard, your deck, others), searchable, the card's face shown on
//     hover;
//   - a decision with more buttons than the prompt line offers (MAX_PROMPT_ROWS: a creature type to
//     choose, targets the Table does not show): its rows, searchable.

namespace rmwork {

struct ChoiceItem {
        int row = -1;  // the row it answers; -1 shown only (a mode chosen earlier)
        std::string text;
        int card = -1;  // the card drawn beside it (a mode's spell, a trigger's source, the card named)
        bool back = false;
        std::string type_line;  // NAME: the card's
        int group = -1;         // NAME: index into ChoicePanel::groups
        int order = 0;          // MODES: the mode's index on the card
        std::string note;       // "chosen 1st → Player B", "row 3"
        bool chosen_before = false;
};

struct ChoicePanel {
        bool active = false;
        DecisionKind kind = DecisionKind::NONE;
        std::string title;  // "Collective Brutality · choose a mode"
        int source_card = -1;
        bool searchable = false;
        std::vector<std::string> groups;  // NAME: the groups' titles, in order
        std::vector<ChoiceItem> items;
        std::vector<int> buttons;  // the rows offered at the panel's foot (Done choosing modes)
        std::vector<std::string> row_text;
        std::string note;     // how the choice works ("The trigger chosen goes on the stack now …")
        int chosen_row = -1;  // the row the decision chose (Replay, an earlier decision), -1 none
        // The item answering row `row`, nullptr when none.
        const ChoiceItem *item_of_row(int row) const;
};

// The panel of decision `index` of `m` in viewer `view`'s frame `v` (read with the viewer's book), whose
// Table decision is `d`: inactive unless `d` has panel rows. `chosen_row` is the reply made there (-1
// none: the human's pending decision).
ChoicePanel read_choice_panel(const Match &m, size_t index, int view, const agents::DecisionView &v,
    const TableDecision &d, const CardBook &book, int chosen_row);

// Whether `it` matches `filter` (a case-insensitive substring of its text or type line; "" matches all).
bool choice_item_matches(const ChoiceItem &it, const std::string &filter);

}  // namespace rmwork

#endif /* RMWORK_CHOICE_PANEL_H */
