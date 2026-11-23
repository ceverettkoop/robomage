#ifndef RMWORK_TABLE_VIEW_H
#define RMWORK_TABLE_VIEW_H

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "attribution.h"
#include "card_book.h"
#include "card_picker.h"
#include "card_picker_view.h"
#include "card_plan.h"
#include "card_renderer.h"
#include "choice_panel.h"
#include "choice_panel_view.h"
#include "keys.h"
#include "links.h"
#include "magnifier.h"
#include "mulligan_plan.h"
#include "replay_session.h"
#include "settings.h"
#include "table_backdrop.h"
#include "table_combat.h"
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
// Combat (table_decision.h's combat kinds, table_combat.h): at declare attackers the creatures that can
// attack glow; at declare blockers the creatures that can block, with the attackers kept lit. Every
// attacker is labelled with what it attacks ("→ Player B", "→ Kaito") during the declarations and
// whenever it attacks a planeswalker. While the human chooses, a click plans instead of answering: a
// creature that can attack is planned to attack the defending player (it stands forward, outlined in the
// accent colour) and, when the defending player has a planeswalker, stays armed so that a click on a
// planeswalker (or the player's chip) changes what it attacks; a click on a planned attacker takes it
// back. A click on a creature that can block arms it (a line follows the mouse), a click on an attacker
// (or a drag from the blocker onto it) plans the block, drawn as an accent arrow blocker → attacker; a
// click on a planned blocker takes the block back. The prompt line's buttons: "Attack with N" / "No
// attack" (or "Block with N" / "No blocks") sends the plan (CombatPlan, one reply per decision through
// plan_reply), "Attack with all" plans every creature that can attack, "Clear" drops the plan. At
// ATTACK_TARGET and BLOCK_TARGET (asked when a row was answered in Choices, or when sending stopped) a
// click on the player chip, a planeswalker or an attacker answers, the attacker under the mouse joined
// to the blocker by an arrow; at DAMAGE the blockers that can still be given lethal damage glow with
// "lethal N", the damage given so far is counted on each blocker ("N assigned"), the prompt says how much
// is left, and a click gives a blocker its lethal damage.
//
// The opening hand (table_decision.h's MULLIGAN and BOTTOM, mulligan_plan.h): the hand is drawn large
// (table_layout.h's PREGAME_HAND_SCALE) and never dimmed; the prompt line says how many mulligans were
// taken, what keeping costs, play or draw, and offers Keep and Mulligan. At the London mulligan's bottom
// picks a click on a hand card plans it ("Bottom 1", "Bottom 2" in the accent colour; a click on a
// planned card takes it back), the prompt line's Undo and Clear change the plan, and "Put N on the
// bottom", once the plan holds every card owed, sends it one reply per decision (plan_reply). A discard
// of several cards (the cleanup discard, an escalate cost's) is planned the same way ("Discard 1", …;
// "Discard N" sends it; card_plan.h).
//
// Modes, triggers to order and names (table_decision.h's MODES, ORDER, NAME, and a decision with more
// buttons than the prompt line holds): the choice panel (choice_panel_view.h), a window where the card
// picker goes; a click on a row answers it. A number (NUMBER: an X, a delve count): a stepper on the
// prompt line ("−", the value's button, "+"), bounded by the rows, the cost the value leaves as pips
// with the mana it comes to; the value's button sends its row. A yes/no, an unless cost and every other
// decision's rows: buttons on the prompt line, the objects a row names lit.
//
// The card picker (card_picker.h, card_picker_view.h): at a decision among cards the Table does not
// show (a library search, Ponder's and Doomsday's ordering, scry, surveil, a dig, a graveyard card for a
// cost), a window over the opponent's half shows them, the pile being built and the rows naming no card;
// a click there answers as a click on the Table does. Rows naming a hand card (Brainstorm's put-back)
// are picked on the hand, the picker showing the pile beside them.
//
// A priority decision (table_decision.h's PRIORITY): the hand cards that can be cast or played and the
// permanents with an ability to activate (or, under --no-auto-mana, to tap for mana) have a thin outline,
// thicker under the mouse; nothing is dimmed. While the human chooses a click casts, plays or activates
// (answers the card's row), and a card with several ways (Daze's two casts, a land's mana and its other
// ability, a cycling card) opens a popup naming each with its cost (priority_costs.h); hovering says what
// a click does with the cost ("Cast Lightning Bolt {R}"). The prompt line says what may be done and has
// the rows naming a card off the Table (a flashback or escape from the graveyard, a companion) as buttons
// (more than MAX_PROMPT_ROWS − 1 of them: one button opening their popup) and Pass, the pass row.
//
// In Replay and at an earlier decision of Play the row chosen there (the log's reply) is marked: its
// card outlined in the accent colour with "Chosen", its button highlighted.
//
// The magnifier: holding the Magnify key (Q) over a card, or right-clicking it, shows it large
// beside it (a double-faced card's both faces, the face up first), with its state on the table
// listed under it. Releasing the key hides a key magnifier;
// a right-click one hides when the mouse leaves the card, at a click or at Esc.

namespace rmwork {

// A button of the prompt line: a menu row (Done selecting targets, Confirm attackers …) or, while the
// human plans a combat declaration, the plan's.
struct PromptButton {
        enum class Act {
            ROW,    // answers `row`
            SEND,   // sends the plan (its confirm row `row` when nothing planned is left to send)
            ALL,    // plans every creature that can attack
            CLEAR,  // drops the plan
            UNDO,   // takes the last card planned back
            DEC,    // the number stepper's value down
            INC,    // the number stepper's value up
            POPUP   // opens the popup of `rows` (a priority decision's ways off the Table, when many)
        };
        std::string label;
        int row = -1;
        Act act = Act::ROW;
        std::vector<int> rows;  // POPUP: the rows it offers
};

struct TableDraw {
        ReplaySession *replay = nullptr;
        const TableBackground *background = nullptr;
        LinkState *links = nullptr;
        ImGuiKeyChord magnify = 0;
        bool play = false;                 // Play's Table (stop dots shown, a click on a step toggles its stop)
        StopSettings *stops = nullptr;     // Play's stops (Settings → Stops), changed by the step bar's clicks
        bool answer = false;               // the human chooses here: a click on a pick or a button answers
        ImGuiKeyChord pass = 0;            // Play: the key that passes priority (named in the Pass button's tooltip)
        ImGuiKeyChord to_end_of_turn = 0;  // Play: F2's key (named there too)
        // Attribution's Table layer (attribution.h): each card, stack object and player chip tinted by
        // its unit's value, with a legend; null when the decision has no attribution. `attr_layer`
        // shows it, and the legend's toggle flips it.
        const AttrTints *tints = nullptr;
        bool *attr_layer = nullptr;
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
        // The prompt line's buttons as last drawn, in order (their rectangles: layout().prompt_buttons).
        const std::vector<PromptButton> &prompt_buttons() const { return buttons_; }
        // The combat declaration planned (Play, while the human chooses), and whether it is being sent.
        const CombatPlan &plan() const { return plan_; }
        bool plan_sending() const { return plan_.sending() || card_plan_.sending(); }
        // The cards planned (Play, while the human chooses them): the London mulligan's bottom picks, a
        // discard of several cards.
        const CardPlan &card_plan() const { return card_plan_; }
        // The choice panel of the decision drawn (inactive when none), and its window.
        const ChoicePanel &panel() const { return panel_; }
        ChoicePanelView &panel_view() { return panel_view_; }
        // NUMBER: the stepper's value as last drawn (-1 none).
        int number_value() const;
        // The card picker of the decision drawn (inactive when none), and its window.
        const CardPicker &picker() const { return picker_; }
        CardPickerView &picker_view() { return picker_view_; }
        // While a plan (a combat declaration, the bottom picks) is being sent: the reply to decision `index`
        // of `m` (the human's pending one, in seat `seat`'s frames), -1 none; a stop or the end of the plan
        // is said in the status bar.
        int plan_reply(const Match &m, long index, int seat);
        // The Attribution layer's tints as last drawn: per object handle and per player chip (by seat) the
        // colour it was tinted with (empty when the layer is off).
        const std::map<uint64_t, ImU32> &tinted_objects() const { return tinted_; }
        const std::array<ImU32, 2> &tinted_players() const { return tinted_players_; }
        // The legend's rectangle as last drawn (panel coordinates; empty when none).
        const TRect &attr_legend() const { return attr_legend_; }
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
        // Hover the first card on the Table whose name contains `name` every frame, its hover hint shown
        // ("" off).
        void force_card_hover(const std::string &name) { force_card_ = name; }
        // While the human chooses: open the pick popup of the first card whose name contains `name` and
        // that a decision's rows name more than once, once per decision ("" off).
        void force_pick(const std::string &name) { force_pick_ = name; }
        // While the human plans a combat declaration: plan `spec` once per declaration, comma-separated
        // items NAME (attacks the player / is armed to block), NAME>TARGET (attacks the planeswalker
        // TARGET, or "player"; blocks the attacker TARGET) or NAME>? (planned and armed); names are
        // substrings, the first creature that can attack or block ("" off). At the London mulligan's bottom
        // pick and a discard of several cards: the hand cards to plan, comma-separated name substrings.
        void force_plan(const std::string &spec) { force_plan_ = spec; }

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
        bool refresh(ReplaySession &s, bool answer);
        TableMetrics measure(bool stop_dots) const;
        // The pile a placed battlefield card stands in (nullptr for a hand card).
        const TablePile *pile_of(const PlacedCard &c) const;
        CardLook look_of(const TableObject &o) const;
        // `steps`: the step bar's cells are hit too (Play).
        Hit hit_test(ImVec2 local, bool steps) const;
        void draw_cards(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_card_state(ImDrawList *dl, const TableObject &o, const PlacedCard &c, ImVec2 o2);
        void draw_combat(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_attack_labels(ImDrawList *dl, ImVec2 o);
        void draw_combat_marks(ImDrawList *dl, const PlacedCard &c, ImVec2 a, ImVec2 b, float cw);
        // An arrow from object `from_h`'s card to object `to_h`'s (their facing edges), or to the point
        // `to` when `to_h` is 0.
        void block_arrow(ImDrawList *dl, ImVec2 o, uint64_t from_h, uint64_t to_h, ImVec2 to, ImU32 color, float t);
        // The placed card of object `h` (its pile's front card when stacked), nullptr when not drawn.
        const PlacedCard *placed(uint64_t h) const;
        // The object of a hit card (a stack's member that is a pick, else its front card), 0 none.
        uint64_t hit_object(const Hit &hit) const;
        std::string name_of(uint64_t h) const;
        // ── Combat planning (table_combat.h) ──
        // The human plans a declaration here (answering, at ATTACKERS or BLOCKERS).
        bool planning() const;
        // Object `h` stays lit at a combat decision (an attacker, a blocker, a planeswalker attacked or
        // planned to be) / is what the armed creature's second click may name.
        bool combat_kept(uint64_t h) const;
        bool armed_target(uint64_t h) const;
        // The defending player's planeswalkers (the top seat's).
        std::vector<uint64_t> planeswalkers() const;
        void apply_plan_to_model();
        void build_buttons();
        std::string prompt_text() const;
        bool press_button(const PromptButton &b);
        bool plan_press(const Hit &hit);
        bool card_plan_press(const Hit &hit);
        // The human plans cards here (answering, at the London mulligan's bottom pick or a discard of
        // several cards), and how many the plan holds.
        bool card_planning() const;
        int card_plan_owed() const;
        // The prompt line's cost: the cost still to pay, or (NUMBER) what the stepper's value leaves.
        std::vector<std::string> prompt_pips() const;
        void draw_picker(const TableDraw &d);
        void draw_panel(const TableDraw &d);
        void plan_release(const Hit &hit);
        void plan_all();
        void apply_force_plan();
        void draw_mid(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_steps(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_stack(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_chips(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_attribution(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_prompt(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_decision_marks(ImDrawList *dl, const PlacedCard &c, ImVec2 a, ImVec2 b, float cw);
        // The pick a hit names (a card, a stack's member, a stack object, a player chip), nullptr none.
        const DecisionPick *pick_at(const Hit &hit) const;
        // What a click answers at `hit` while the human chooses: a row, or a popup of a pick's rows;
        // false when it answers nothing (the click selects).
        bool answer_click(const Hit &hit);
        void open_pick_popup(const DecisionPick &p);
        void open_rows_popup(const std::vector<int> &rows, const std::string &title);
        // A row's text as the popup and the hover say it: at a priority decision with its cost.
        std::string row_label(int row) const;
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
        CombatPlan plan_;
        CardPlan card_plan_;
        CardPicker picker_;
        CardPickerView picker_view_;
        ChoicePanel panel_;
        ChoicePanelView panel_view_;
        int number_index_ = 0;    // NUMBER: the stepper's row
        std::string number_key_;  // the decision the stepper's row belongs to
        std::vector<PromptButton> buttons_;
        bool answering_ = false;  // the human chooses at the decision drawn (TableDraw::answer)
        unsigned model_plan_generation_ = 0;
        bool model_answer_ = false;
        uint64_t press_handle_ = 0;     // the blocker a press armed (a drag from it may plan its block)
        bool press_was_armed_ = false;  // it was armed before the press (a click without a drag disarms it)
        TableLayout layout_;
        std::string error_;
        std::string status_;
        std::array<TRect, 2> chip_rects_;  // by seat
        std::map<uint64_t, ImU32> tinted_;
        std::array<ImU32, 2> tinted_players_{{0, 0}};
        TRect attr_legend_;
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
        std::string force_card_;
        std::string force_pick_;
        bool force_picked_ = false;  // force_pick_ opened the popup at this decision
        unsigned force_pick_generation_ = 0;
        std::string force_plan_;
        PlanKey force_planned_;  // the declaration force_plan_ was applied to
};

}  // namespace rmwork

#endif /* RMWORK_TABLE_VIEW_H */
