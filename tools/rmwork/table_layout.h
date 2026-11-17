#ifndef RMWORK_TABLE_LAYOUT_H
#define RMWORK_TABLE_LAYOUT_H

#include <vector>

#include "table_model.h"

// ── Where everything on the Table goes, and the cards' size ─────────────────────────────────────
//
// Pure geometry (no ImGui), in the panel's own coordinates ((0, 0) its top-left corner). From top
// to bottom: the top seat's chip band, its lands row (the far edge) and other-permanents row, the
// middle strip (the turn, step, deciding seat and step bar, and the stack when it is not empty),
// the bottom seat's other-permanents and lands rows, then its hand and its chip band. Rows are
// centred; the two halves share the height left over, their rows centred in it.
//
// The fit (fit_card_width, the mock-up's fitTable): one card width for the four battlefield rows
// (the hand's is HAND_SCALE times it) such that the rows and the hand fill the panel's height and
// the widest row fits its width, never above MAX_CARD_W. A tapped card lies on its side (as wide as
// a card is tall); an attachment peeks ATTACH_PEEK of a card's height above the one before it; a
// stack of identical lands shows up to three more cards STACK_STEP of a width apart; an attacking
// permanent's edge facing the middle stands ATTACK_SHIFT of a card's height beyond an upright
// card's (a tapped attacker too, so it shows forward of its untapped neighbours). A row is never shorter than
// one upright card, so the halves keep their place when a row empties.

namespace rmwork {

inline constexpr float CARD_ASPECT = 1.3934f;  // height / width
inline constexpr float HAND_SCALE = 1.15f;
inline constexpr float ATTACH_PEEK = 0.17f;
inline constexpr float STACK_STEP = 0.10f;
inline constexpr int STACK_VISIBLE = 4;
inline constexpr float ATTACK_SHIFT = 0.25f;
inline constexpr float MAX_CARD_W = 220.0f;
inline constexpr float MIN_CARD_W = 8.0f;
inline constexpr int MAX_STACK_LINES = 3;

struct TRect {
        float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        float w() const { return x1 - x0; }
        float h() const { return y1 - y0; }
        bool contains(float x, float y) const { return x >= x0 && x < x1 && y >= y0 && y < y1; }
        // Inside `o`, allowing `eps` of rounding.
        bool inside(const TRect &o, float eps = 0.5f) const {
            return x0 >= o.x0 - eps && y0 >= o.y0 - eps && x1 <= o.x1 + eps && y1 <= o.y1 + eps;
        }
};

// The text-dependent sizes the view measures with its font.
struct TableMetrics {
        float line_h = 17;   // a line of UI text
        float header_w = 0;  // the middle strip's turn, step and decider texts, side by side
        float steps_w = 0;   // the step bar
        float steps_h = 0;
};

struct PlacedCard {
        int object = -1;  // index into TableModel::objects
        TRect rect;       // as drawn: a tapped card's is its sideways rectangle
        bool rotated = false;
        bool in_hand = false;
        int row = -1;         // TableRowId, -1 in the hand
        int pile = -1;        // index in its row
        bool front = true;    // the pile's front card (its host, or a stack's front land)
        int stack_count = 0;  // on a land stack's front card: the stack's size
};

struct TableLayout {
        float card_w = 0;  // a battlefield card's width
        float hand_w = 0;
        bool fits = false;  // every card and strip inside the panel at a card width of at least MIN_CARD_W
        TRect panel;
        TRect top_band, bottom_band;  // the chip bands
        TRect rows[ROW_COUNT];
        TRect hand;
        TRect mid;                       // the middle strip
        TRect mid_header;                // its first line(s): the turn, step, decider, step bar
        bool steps_wrapped = false;      // the step bar on a line of its own
        TRect stack_label;               // "Stack · 3 objects, top first"
        std::vector<TRect> stack_items;  // the stack objects shown, top first
        std::vector<PlacedCard> cards;   // in drawing order (behind first)
};

// The card width that fits `m`'s rows and hand into a `w` × `h` panel whose fixed parts (chip bands,
// middle strip, gaps) take `fixed_h` (fit_card_width's own computation when called by layout_table).
float fit_card_width(const TableModel &m, float w, float h, float fixed_h);
// The whole layout of `m` in a `w` × `h` panel.
TableLayout layout_table(const TableModel &m, float w, float h, const TableMetrics &mx);

}  // namespace rmwork

#endif /* RMWORK_TABLE_LAYOUT_H */
