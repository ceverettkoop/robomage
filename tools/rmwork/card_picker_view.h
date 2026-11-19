#ifndef RMWORK_CARD_PICKER_VIEW_H
#define RMWORK_CARD_PICKER_VIEW_H

#include <cstdint>
#include <string>
#include <vector>

#include "card_book.h"
#include "card_picker.h"
#include "card_renderer.h"
#include "imgui.h"
#include "links.h"
#include "magnifier.h"

// ── The card picker: a window over the Table for the cards a decision names off it ─────────────
//
// docs/workbench_plan.md §4.2, item 8 (mock-ups decisions.html#card-picker, -ponder, -brainstorm,
// -surveil). Drawn by the Table inside its own panel, as the mock-up has it, rather than as a docked
// tool: the decision's context (the stack object resolving, the prompt line, the board) stays where it
// is, and nothing has to be brought to the front and given back. It sits over the opponent's half,
// from below the opponent's chip down to the middle strip, or, when its cards need more room, down to
// the prompt line (over the strip's step bar and stack, the title naming what resolves), the prompt line
// visible; a whole-zone pick too large for that (Doomsday's library and graveyard) takes the Table down
// to its foot, the prompt then at the picker's own foot. Its title bar's button collapses it to that bar
// to show the board under it.
//
// What it shows (card_picker.h): the cards by zone, identical cards one entry with ×n, a library's in
// order from the top with each one's place; a card some row names glows, a card shown only (the rest of
// a searched library) is dimmed. A filter box (name or type line) and a Cards / List switch above them
// (the list: one line per entry with its count, type line and place); "Unpickable" hides or shows the
// dimmed cards. Beside them the pile being built, top first, the place the next pick fills marked, the
// cards picked so far in it (the frame no longer shows a card put back), and what went to the bottom or
// the graveyard. Under them the rows naming no card (Fail to find, Take nothing) as buttons. A
// decision whose picks are all on the Table (Brainstorm's put-back from the hand) shows the pile alone,
// narrow, at the right.
//
// While the human chooses (`answering`), a click on a card with one row answers it; a card with several
// (scry's Top / Bottom, surveil's Top / Graveyard) has a button per row under it. Otherwise (Replay, an
// earlier decision, the opponent thinking) nothing answers; the row chosen there is marked (the card
// outlined in the accent colour, "Chosen"; the button highlighted). Hovering a card links it (its
// Choices row) and says what a click does; Q held (or a right-click) magnifies it.

namespace rmwork {

class CardPickerView {
    public:
        // Where the picker may go, in screen coordinates: its left, top and right edges, and the bottom
        // of the short band (to the middle strip), of the long one (to the prompt line) and of the tall one
        // (to the Table's foot: over the prompt line, whose text `prompt` the picker's foot then says).
        struct Place {
                float x0 = 0, y0 = 0, x1 = 0;
                float y_short = 0, y_long = 0, y_tall = 0;
                std::string prompt;
        };
        // A thing a click acts on, as last drawn (screen coordinates): a card (`card`), a placement or
        // footer button; `row` is the row it answers (-1: a card with several rows, or none).
        struct Item {
                int row = -1;
                bool card = false;
                size_t group = 0, entry = 0;
                std::string label;
                ImVec2 min, max;
        };

        // Draws `p` (cards from `book`) inside the current window at `place`; the row a click answered
        // (`answering`), else -1.
        int draw(const CardPicker &p, const CardBook &book, const Place &place, bool answering, ImGuiKeyChord magnify,
            LinkState *links);
        // The renderer cards are drawn with (nullptr: the text renderer); it must outlive its use.
        void set_renderer(const CardRenderer *r) { renderer_ = r ? r : &text_renderer_; }

        const std::string &filter() const { return filter_; }
        void set_filter(const std::string &f);
        bool list() const { return list_; }
        void set_list(bool on) { list_ = on; }
        bool collapsed() const { return collapsed_; }
        void set_collapsed(bool on) { collapsed_ = on; }
        bool show_unpickable() const { return unpickable_; }
        void set_show_unpickable(bool on) { unpickable_ = on; }

        // As last drawn: whether it was, its rectangle, the clickable things, the entries the filter let
        // through and the card width of the grid.
        bool shown() const { return shown_; }
        // It took the tall band (over the prompt line).
        bool tall() const { return tall_; }
        ImVec2 min() const { return min_; }
        ImVec2 max() const { return max_; }
        const std::vector<Item> &items() const { return items_; }
        size_t entries_shown() const { return entries_shown_; }
        float card_width() const { return card_w_; }
        // The filter box's rectangle (screen coordinates).
        ImVec2 filter_min() const { return filter_min_; }
        ImVec2 filter_max() const { return filter_max_; }
        uint64_t magnified() const { return mag_.handle(); }
        // Screenshots: filter by `f` at the start of each resolution ("" off).
        void force_filter(const std::string &f) { force_filter_ = f; }
        // At the next draw, scrolls the cards to the one the decision chose (a new decision drawn).
        void request_scroll() { scroll_ = true; }
        // Marks the picker not drawn (a frame without one).
        void hide() { shown_ = false; }

    private:
        struct Cell {
                size_t group = 0, entry = 0;
                ImVec2 at;  // relative to the grid's origin
        };
        struct Flow {
                std::vector<Cell> cells;
                std::vector<ImVec2> labels;  // each group's heading, relative
                float height = 0;
                float width = 0;  // the widest line's
        };
        // `buttons`: the placement buttons stacked under a card (placement_rows).
        Flow flow(const CardPicker &p, float cw, float width, int buttons) const;
        int placement_rows(const CardPicker &p, float cw) const;
        bool buttons_fit(const CardPicker &p, const CardPickEntry &e, float cw) const;
        float list_height(const CardPicker &p) const;
        bool visible(const CardPickEntry &e) const;
        CardLook look_of(const CardBook &book, const CardPickEntry &e) const;
        CardLook look_of(const CardBook &book, const PileCard &c) const;
        void title_bar(const CardPicker &p, float width);
        void toolbar(const CardPicker &p);
        int draw_grid(const CardPicker &p, const CardBook &book, ImVec2 size, bool answering, ImGuiKeyChord magnify,
            LinkState *links);
        int draw_list(const CardPicker &p, const CardBook &book, ImVec2 size, bool answering, LinkState *links);
        void draw_pile(const CardPicker &p, const CardBook &book, ImVec2 size);
        int draw_footer(const CardPicker &p, bool answering, const std::string &prompt);
        // An entry's placement buttons, `width` wide each stacked from `at`, else side by side in `width`.
        int placement_buttons(const CardPicker &p, const CardPickEntry &e, size_t g, size_t i, bool answering,
            float width, const ImVec2 *at);
        std::string click_text(const CardPicker &p, const CardPickEntry &e) const;
        void reset_for(const CardPicker &p);

        TextCardRenderer text_renderer_;
        const CardRenderer *renderer_ = &text_renderer_;
        MagnifierState mag_;
        std::string filter_;
        char filter_buf_[96] = {0};
        bool list_ = false;
        bool collapsed_ = false;
        bool unpickable_ = true;
        uint64_t source_ = ~0ull;  // the resolution the filter and the collapse belong to
        std::string force_filter_;
        bool shown_ = false;
        bool tall_ = false;
        bool scroll_ = false;
        ImVec2 min_, max_, filter_min_, filter_max_;
        std::vector<Item> items_;
        size_t entries_shown_ = 0;
        float card_w_ = 0;
};

}  // namespace rmwork

#endif /* RMWORK_CARD_PICKER_VIEW_H */
