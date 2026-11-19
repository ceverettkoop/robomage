#ifndef RMWORK_TABLE_VIEW_H
#define RMWORK_TABLE_VIEW_H

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "card_book.h"
#include "card_renderer.h"
#include "keys.h"
#include "links.h"
#include "magnifier.h"
#include "replay_session.h"
#include "settings.h"
#include "table_backdrop.h"
#include "table_decision.h"
#include "table_layout.h"
#include "table_model.h"

// ── The Table tool: the current decision as the viewer saw it ───────────────────────────────────
//
// Reads the ReplaySession's decision and viewer, builds the TableModel of that viewer's frame (its
// CardBook built once per match and viewer) and draws it MTGO-style over the background from
// Settings → Table, filling the panel edge to edge (table_layout.h's fit, every frame, so it follows
// a resize). On the cards, over what the CardRenderer draws: the post-layer P/T (green above the
// printed, red below, blue for one of each), loyalty, counters by kind, marked damage, summoning
// sickness, the designations, the ×n count of a stack of identical permanents (which the layout
// makes only when the board is too wide to show each); a phased-out permanent is faded. The
// bottom seat's hand lies on a shaded band. Attackers
// stand forward; a line runs from each blocker to the attacker it blocks. The player chips (name,
// deck, life, hand, poison, energy, monarch, speed) and their effect and emblem chips sit at the
// far top and bottom left. The middle strip: turn, active player, step, deciding seat, day or
// night, the twelve-step bar, and the stack when it is not empty. In Play the step bar has a stop
// dot under each step (filled where Settings → Stops stops for the side whose turn it is), and a
// click on a step toggles that stop (a tooltip says which); Replay's has neither.
//
// Linking (links.h): the Table reports the card, stack object or player chip under the mouse and
// outlines whatever the hovered (else the selected) item links to: a stack object, hovered here or
// in another tool, lights itself and its targets (cards and player chips), with no line between
// them; a menu row its source and target. A left click selects a card or chip (again: unselects). When the
// viewer is the deciding seat, the cards some menu row acts with glow. A stack's front card stands
// for all of it: it glows and is outlined when any of its members would be, and hovering it says
// its count.
//
// Decisions on the Table (table_decision.h): at a targeting or payment decision the viewer sees, the
// objects and players a row names glow and everything else on the battlefield and in the hand is
// dimmed; the spell or ability being cast, activated or put on the stack is lit (its stack object, or a
// "Source" tag on its permanent); targets chosen earlier in the same targeting are outlined in the
// accent colour with their number, a point of divided damage given so far is counted on its target; and
// the middle strip's prompt line says what is decided ("YOUR DECISION" in Play), the cost still to pay
// as pips (life, energy, mana paid after it) and the rows naming nothing as buttons. At a priority
// decision offering mana abilities the lands and sources glow as for any decision. While the human
// chooses (TableDraw::answer: Play, at the pending decision), a click on a pick or a button answers
// with its row (take_answer); a pick naming several rows (a dual land's two colours) opens a popup of
// them; hovering a pick says what a click does. Otherwise (Replay, an earlier decision, the opponent
// thinking) the same highlights show and a click only selects.
//
// The magnifier: holding the Magnify key (Q) over a card, or right-clicking it, shows it large
// beside it (a double-faced card's both faces, the face up first), with its state on the table
// listed under it. Releasing the key hides a key magnifier;
// a right-click one hides when the mouse leaves the card, at a click or at Esc.

namespace rmwork {

struct TableDraw {
        ReplaySession *replay = nullptr;
        const TableBackground *background = nullptr;
        LinkState *links = nullptr;
        ImGuiKeyChord magnify = 0;
        bool play = false;              // Play's Table (stop dots shown, a click on a step toggles its stop)
        StopSettings *stops = nullptr;  // Play's stops (Settings → Stops), changed by the step bar's clicks
        bool answer = false;            // the human chooses here: a click on a pick or a button answers
};

class TableView {
    public:
        TableView();
        // The renderer cards are drawn with (nullptr: the text renderer); it must outlive its use.
        void set_renderer(const CardRenderer *r);
        // Draws the Table into the current window (between Begin and End).
        void draw(const TableDraw &d);
        // A message for the status bar, "" when none.
        std::string take_status();
        // The menu row a click on the Table answered (TableDraw::answer), -1 none; taken once.
        int take_answer();
        // The decision the Table answers, as last drawn (NONE when the viewer does not see its menu).
        const TableDecision &decision() const { return decision_; }
        // A player chip's rectangle as last drawn (panel coordinates), by seat.
        const TRect &chip_rect(int seat) const { return chip_rects_[seat == 1 ? 1 : 0]; }
        // The popup offering one object's rows: whether it is open, its rows, and its items' rectangles
        // as last drawn (screen coordinates), in the rows' order.
        bool pick_popup_open() const { return popup_shown_; }
        const std::vector<int> &pick_popup_rows() const { return popup_rows_; }
        const std::vector<TRect> &pick_popup_items() const { return popup_items_; }

        // The model and layout last drawn (nullptr / empty before a game is drawn).
        const TableModel *model() const { return have_model_ ? &model_ : nullptr; }
        const TableLayout &layout() const { return layout_; }
        // Whether the magnifier is open, and on which object.
        bool magnifier_open() const { return mag_.handle() != 0; }
        uint64_t magnified() const { return mag_.handle(); }

        // Debugging and screenshots: hover stack object `index` (0 = top; -1 off) every frame, and
        // open the magnifier on the first card whose name contains `name` ("" off).
        void force_stack_hover(int index) { force_stack_ = index; }
        // In Play: hover step `index` of the step bar (0 = untap; -1 off) every frame, its tooltip shown.
        void force_step_hover(int index) { force_step_ = index; }
        void force_magnify(const std::string &name) { force_mag_ = name; }
        // While the human chooses: open the pick popup of the first card whose name contains `name` and
        // that a decision's rows name more than once, once per decision ("" off).
        void force_pick(const std::string &name) { force_pick_ = name; }

        // The book of `view`'s CARDS frame for `m` (built on first use).
        const CardBook &book(const Match &m, int view);

    private:
        struct Hit {
                int card = -1;    // index into layout_.cards
                int stack = -1;   // index into model_.stack
                int seat = -1;    // a player chip
                int step = -1;    // a step bar's cell (Play)
                int button = -1;  // a button of the prompt line (index into layout_.prompt_buttons)
        };
        bool refresh(ReplaySession &s);
        TableMetrics measure(bool stop_dots) const;
        // The pile a placed battlefield card stands in (nullptr for a hand card).
        const TablePile *pile_of(const PlacedCard &c) const;
        CardLook look_of(const TableObject &o) const;
        // `steps`: the step bar's cells are hit too (Play).
        Hit hit_test(ImVec2 local, bool steps) const;
        void draw_cards(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_card_state(ImDrawList *dl, const TableObject &o, const PlacedCard &c, ImVec2 o2);
        void draw_combat(ImDrawList *dl, ImVec2 o);
        void draw_mid(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_steps(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_stack(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_chips(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_prompt(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_decision_marks(ImDrawList *dl, const PlacedCard &c, ImVec2 a, ImVec2 b, float cw);
        // The pick a hit names (a card, a stack's member, a stack object, a player chip), nullptr none.
        const DecisionPick *pick_at(const Hit &hit) const;
        // What a click answers at `hit` while the human chooses: a row, or a popup of a pick's rows;
        // false when it answers nothing (the click selects).
        bool answer_click(const Hit &hit);
        void open_pick_popup(const DecisionPick &p);
        void apply_force_pick(const TableDraw &d);
        void hover_hint(const Hit &hit, const TableDraw &d);
        void draw_pick_popup(const TableDraw &d);
        void draw_magnifier(const TableObject &o, const TRect &at, ImVec2 origin, int stacked);
        void update_magnifier(const Hit &hit, const TableDraw &d, bool over);

        TextCardRenderer text_renderer_;
        const CardRenderer *renderer_;
        TableBackdrop backdrop_;
        const Match *book_match_ = nullptr;
        std::string book_dump_;  // the match's dump: a match at a freed one's address is another
        std::array<CardBook, 3> books_;
        std::array<bool, 3> book_built_{{false, false, false}};
        bool have_model_ = false;
        const Match *model_match_ = nullptr;
        unsigned model_generation_ = 0;
        int model_view_ = -1;
        TableModel model_;
        TableDecision decision_;
        TableLayout layout_;
        std::string error_;
        std::string status_;
        std::array<TRect, 2> chip_rects_;  // by seat
        uint64_t hovered_handle_ = 0;
        int hovered_stack_ = -1;
        int hovered_step_ = -1;
        int hovered_button_ = -1;
        int answer_ = -1;
        bool open_popup_ = false;  // open the pick popup at this frame
        bool popup_shown_ = false;
        unsigned popup_generation_ = 0;
        std::string popup_title_;
        std::vector<int> popup_rows_;
        std::vector<TRect> popup_items_;
        MagnifierState mag_;
        int force_stack_ = -1;
        int force_step_ = -1;
        std::string force_mag_;
        std::string force_pick_;
        bool force_picked_ = false;  // force_pick_ opened the popup at this decision
        unsigned force_pick_generation_ = 0;
};

}  // namespace rmwork

#endif /* RMWORK_TABLE_VIEW_H */
