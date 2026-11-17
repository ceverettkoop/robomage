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
#include "table_layout.h"
#include "table_model.h"

// ── The Table tool: the current decision as the viewer saw it ───────────────────────────────────
//
// Reads the ReplaySession's decision and viewer, builds the TableModel of that viewer's frame (its
// CardBook built once per match and viewer) and draws it MTGO-style over the background from
// Settings → Table, filling the panel edge to edge (table_layout.h's fit, every frame, so it follows
// a resize). On the cards, over what the CardRenderer draws: the post-layer P/T (green above the
// printed, red below, blue for one of each), loyalty, counters by kind, marked damage, summoning
// sickness, the designations, a land stack's count; a phased-out permanent is faded. The
// bottom seat's hand lies on a shaded band. Attackers
// stand forward; a line runs from each blocker to the attacker it blocks. The player chips (name,
// deck, life, hand, poison, energy, monarch, speed) and their effect and emblem chips sit at the
// far top and bottom left. The middle strip: turn, active player, step, deciding seat, day or
// night, the twelve-step bar (its stop dots only in Play), and the stack when it is not empty.
//
// Linking (links.h): the Table reports the card, stack object or player chip under the mouse and
// outlines whatever the hovered (else the selected) item links to: a stack object, hovered here or
// in another tool, lights itself and its targets (cards and player chips), with no line between
// them; a menu row its source and target. A left click selects a card or chip (again: unselects). When the
// viewer is the deciding seat, the cards some menu row acts with glow.
//
// The magnifier: holding the Magnify key (Q) over a card, or right-clicking it, shows it large
// beside it, with its state on the table listed under it. Releasing the key hides a key magnifier;
// a right-click one hides when the mouse leaves the card, at a click or at Esc.

namespace rmwork {

struct TableDraw {
        ReplaySession *replay = nullptr;
        const TableBackground *background = nullptr;
        LinkState *links = nullptr;
        ImGuiKeyChord magnify = 0;
        bool play = false;  // Play's Table (stop dots shown)
        const StopSettings *stops = nullptr;
};

class TableView {
    public:
        TableView();
        // Draws the Table into the current window (between Begin and End).
        void draw(const TableDraw &d);
        // A message for the status bar, "" when none.
        std::string take_status();

        // The model and layout last drawn (nullptr / empty before a game is drawn).
        const TableModel *model() const { return have_model_ ? &model_ : nullptr; }
        const TableLayout &layout() const { return layout_; }
        // Whether the magnifier is open, and on which object.
        bool magnifier_open() const { return mag_.handle() != 0; }
        uint64_t magnified() const { return mag_.handle(); }

        // Debugging and screenshots: hover stack object `index` (0 = top; -1 off) every frame, and
        // open the magnifier on the first card whose name contains `name` ("" off).
        void force_stack_hover(int index) { force_stack_ = index; }
        void force_magnify(const std::string &name) { force_mag_ = name; }

        // The book of `view`'s CARDS frame for `m` (built on first use).
        const CardBook &book(const Match &m, int view);

    private:
        struct Hit {
                int card = -1;   // index into layout_.cards
                int stack = -1;  // index into model_.stack
                int seat = -1;   // a player chip
        };
        bool refresh(ReplaySession &s);
        TableMetrics measure() const;
        CardLook look_of(const TableObject &o) const;
        Hit hit_test(ImVec2 local) const;
        void draw_cards(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_card_state(ImDrawList *dl, const TableObject &o, const PlacedCard &c, ImVec2 o2);
        void draw_combat(ImDrawList *dl, ImVec2 o);
        void draw_mid(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_steps(ImDrawList *dl, ImVec2 pos, const TableDraw &d);
        void draw_stack(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_chips(ImDrawList *dl, ImVec2 o, const TableDraw &d);
        void draw_magnifier(const TableObject &o, const TRect &at, ImVec2 origin);
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
        TableLayout layout_;
        std::string error_;
        std::string status_;
        std::array<TRect, 2> chip_rects_;  // by seat
        uint64_t hovered_handle_ = 0;
        int hovered_stack_ = -1;
        MagnifierState mag_;
        int force_stack_ = -1;
        std::string force_mag_;
};

}  // namespace rmwork

#endif /* RMWORK_TABLE_VIEW_H */
