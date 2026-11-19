#ifndef RMWORK_CHOICE_PANEL_VIEW_H
#define RMWORK_CHOICE_PANEL_VIEW_H

#include <string>
#include <vector>

#include "card_book.h"
#include "card_picker_view.h"
#include "card_renderer.h"
#include "choice_panel.h"
#include "imgui.h"
#include "links.h"

// ── The choice panel: a window over the Table for a decision's modes, triggers or names ────────
//
// docs/workbench_plan.md §4.2, items 9 and 10 (mock-ups decisions.html#modes-numbers, -order,
// #name-card). Drawn by the Table inside its panel where the card picker goes (over the opponent's
// half, below its chip, as tall as its rows need down to the prompt line), for the same reason: the
// decision's context (the stack, the prompt line, the board) stays in view and nothing is brought to
// the front. Its title bar's button collapses it.
//
// What it shows (choice_panel.h): MODES and ORDER as a column of rows, each with its card (the spell,
// the trigger's source) small at the left, the text wrapped beside it and its note at the right (a mode
// chosen earlier marked, with its targets); NAME and the overflowing list as a filter box over a
// scrolling list (the row number, the name, the type line; NAME grouped by decklist), the card's face
// magnified beside the panel while a name is hovered. At its foot the hint and the rows offered there
// (Done choosing modes).
//
// While the human chooses (`answering`) a click on a row answers it; otherwise (Replay, an earlier
// decision, the opponent thinking) nothing answers and the row chosen there is marked (outlined in the
// accent colour, "Chosen"). Hovering a row links it (its Choices row).

namespace rmwork {

class ChoicePanelView {
    public:
        // A thing a click acts on, as last drawn (screen coordinates): an item (its row) or a foot button.
        struct Item {
                int row = -1;
                bool button = false;
                std::string label;
                ImVec2 min, max;
        };

        // Draws `p` (cards from `book`) inside the current window at `place` (the card picker's places);
        // the row a click answered (`answering`), else -1.
        int draw(const ChoicePanel &p, const CardBook &book, const CardPickerView::Place &place, bool answering,
            LinkState *links);
        // The renderer cards are drawn with (nullptr: the text renderer); it must outlive its use.
        void set_renderer(const CardRenderer *r) { renderer_ = r ? r : &text_renderer_; }

        const std::string &filter() const { return filter_; }
        void set_filter(const std::string &f);
        bool collapsed() const { return collapsed_; }
        void set_collapsed(bool on) { collapsed_ = on; }

        // As last drawn: whether it was, its rectangle, the clickable things, the items the filter let
        // through and the filter box's rectangle.
        bool shown() const { return shown_; }
        ImVec2 min() const { return min_; }
        ImVec2 max() const { return max_; }
        const std::vector<Item> &items() const { return items_; }
        size_t items_shown() const { return items_shown_; }
        ImVec2 filter_min() const { return filter_min_; }
        ImVec2 filter_max() const { return filter_max_; }
        // Screenshots: filter by `f` at each new decision ("" off).
        void force_filter(const std::string &f) { force_filter_ = f; }
        // Marks the panel not drawn (a frame without one).
        void hide() { shown_ = false; }

    private:
        float rows_height(const ChoicePanel &p, float width) const;
        void title_bar(const ChoicePanel &p, float width);
        int draw_rows(const ChoicePanel &p, const CardBook &book, ImVec2 size, bool answering, LinkState *links);
        int draw_list(const ChoicePanel &p, const CardBook &book, ImVec2 size, bool answering, LinkState *links);
        int draw_footer(const ChoicePanel &p, bool answering);
        CardLook look_of(const CardBook &book, int card, bool back) const;
        void reset_for(const ChoicePanel &p);

        TextCardRenderer text_renderer_;
        const CardRenderer *renderer_ = &text_renderer_;
        std::string filter_;
        char filter_buf_[96] = {0};
        bool collapsed_ = false;
        std::string force_filter_;
        std::string key_;  // the decision the filter and the collapse belong to (its rows' text)
        bool shown_ = false;
        ImVec2 min_, max_, filter_min_, filter_max_;
        std::vector<Item> items_;
        size_t items_shown_ = 0;
};

}  // namespace rmwork

#endif /* RMWORK_CHOICE_PANEL_VIEW_H */
