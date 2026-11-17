#include "table_layout.h"

#include <algorithm>
#include <cmath>

namespace {

// Pixels.
const float GAP = 6;       // between piles in a row
const float HAND_GAP = 8;  // between hand cards
const float ROW_GAP = 8;   // between rows and strips
const float SIDE = 10;     // left and right of the rows
const float PAD = 6;       // around a chip and below the hand
const float MID_MARGIN = 8;
const float MID_PAD = 5;
const float STACK_GAP = 6;

// A pile's size in card widths; `shift` is the attack shift (toward the middle).
struct PileShape {
        float w = 1, h = rmwork::CARD_ASPECT, shift = 0;
};

// The middle strip's parts' heights for a panel width.
struct MidShape {
        float header_h = 0;
        bool wrapped = false;
        int per_line = 1, lines = 0, shown = 0;
        float item_w = 0, item_h = 0, label_h = 0;
        float h = 0;
};

}  // namespace

static PileShape pile_shape(const rmwork::TableModel &m, const rmwork::TablePile &p);
static float row_height(const rmwork::TableModel &m, int row);
static float row_width(const rmwork::TableModel &m, int row);
static bool shows_steps(const rmwork::TableModel &m);
static MidShape mid_shape(const rmwork::TableModel &m, float w, const rmwork::TableMetrics &mx);
static float chip_band(const rmwork::TableMetrics &mx);
static float fixed_height(float mid_h, float band);
static void place_row(const rmwork::TableModel &m, int row, const rmwork::TRect &r, float bw, bool top,
    std::vector<rmwork::PlacedCard> *out);
static rmwork::TRect card_rect(float x, float y, float bw, bool tapped);

static PileShape pile_shape(const rmwork::TableModel &m, const rmwork::TablePile &p) {
    using namespace rmwork;
    PileShape s;
    if (p.objects.empty()) return s;
    const TableObject &host = m.objects[static_cast<size_t>(p.objects[0])];
    if (p.land_stack) {
        const int extra = std::min(static_cast<int>(p.objects.size()) - 1, STACK_VISIBLE - 1);
        s.w = 1 + static_cast<float>(extra) * STACK_STEP;
        s.h = CARD_ASPECT;
        return s;
    }
    const size_t k = p.objects.size() - 1;
    const float peek = ATTACH_PEEK * CARD_ASPECT;
    s.w = host.tapped ? CARD_ASPECT : 1;
    s.h = static_cast<float>(k) * peek + (host.tapped ? 1 : CARD_ASPECT);
    for (size_t i = 1; i <= k; ++i) {
        const TableObject &a = m.objects[static_cast<size_t>(p.objects[i])];
        s.w = std::max(s.w, a.tapped ? CARD_ASPECT : 1.0f);
        s.h = std::max(s.h, static_cast<float>(k - i) * peek + (a.tapped ? 1 : CARD_ASPECT));
    }
    if (host.attacking) s.shift = ATTACK_SHIFT * CARD_ASPECT;
    return s;
}

// A row's height in card widths: its tallest pile, never less than an upright card; an attacker's
// pile is as tall as an upright card at least, plus its shift.
static float row_height(const rmwork::TableModel &m, int row) {
    float h = rmwork::CARD_ASPECT;
    for (const rmwork::TablePile &p : m.rows[static_cast<size_t>(row)]) {
        const PileShape s = pile_shape(m, p);
        h = std::max(h, std::max(s.h, rmwork::CARD_ASPECT) + s.shift);
    }
    return h;
}

// A row's width in card widths (gaps not counted).
static float row_width(const rmwork::TableModel &m, int row) {
    float w = 0;
    for (const rmwork::TablePile &p : m.rows[static_cast<size_t>(row)]) w += pile_shape(m, p).w;
    return w;
}

static bool shows_steps(const rmwork::TableModel &m) {
    return m.turn > 0 && !m.pregame && !m.sideboarding && !m.terminal && m.step >= 0;
}

static MidShape mid_shape(const rmwork::TableModel &m, float w, const rmwork::TableMetrics &mx) {
    MidShape s;
    const float inner = std::max(1.0f, w - 2 * MID_MARGIN - 2 * MID_PAD);
    const float steps_w = shows_steps(m) ? mx.steps_w : 0;
    const float steps_h = shows_steps(m) ? mx.steps_h : 0;
    if (steps_w <= 0 || mx.header_w + 14 + steps_w <= inner) {
        s.header_h = std::max(mx.line_h, steps_h);
    } else {
        s.wrapped = true;
        s.header_h = mx.line_h + 4 + steps_h;
    }
    s.h = 2 * MID_PAD + s.header_h;
    if (m.stack.empty()) return s;
    const float thumb = std::round(2.0f * mx.line_h);
    s.item_h = thumb * rmwork::CARD_ASPECT + 6;
    const float min_w = 14 * mx.line_h, max_w = 27 * mx.line_h;
    s.per_line = std::max(1, static_cast<int>(std::floor((inner + STACK_GAP) / (min_w + STACK_GAP))));
    s.item_w =
        std::min(max_w, (inner - STACK_GAP * static_cast<float>(s.per_line - 1)) / static_cast<float>(s.per_line));
    const int count = static_cast<int>(m.stack.size());
    s.lines = std::min((count + s.per_line - 1) / s.per_line, rmwork::MAX_STACK_LINES);
    s.shown = std::min(count, s.lines * s.per_line);
    s.label_h = mx.line_h + 4;
    s.h += 5 + s.label_h + static_cast<float>(s.lines) * s.item_h + static_cast<float>(s.lines - 1) * STACK_GAP;
    return s;
}

// A chip band: a chip (a line of text with 2 px above and below) with PAD around it.
static float chip_band(const rmwork::TableMetrics &mx) {
    return mx.line_h + 4 + 2 * PAD;
}

// Everything but the four rows and the hand: two chip bands, five gaps, the middle strip, the
// space below the hand.
static float fixed_height(float mid_h, float band) {
    return 2 * band + 5 * ROW_GAP + mid_h + PAD;
}

static rmwork::TRect card_rect(float x, float y, float bw, bool tapped) {
    const float w = tapped ? bw * rmwork::CARD_ASPECT : bw;
    const float h = tapped ? bw : bw * rmwork::CARD_ASPECT;
    return rmwork::TRect{x, y, x + w, y + h};
}

// Row `row`'s piles, centred in `r`; a top row's piles hang from its top edge, a bottom row's stand
// on its bottom edge, an attacker moved toward the middle.
static void place_row(const rmwork::TableModel &m, int row, const rmwork::TRect &r, float bw, bool top,
    std::vector<rmwork::PlacedCard> *out) {
    using namespace rmwork;
    const std::vector<TablePile> &piles = m.rows[static_cast<size_t>(row)];
    float total = 0;
    for (const TablePile &p : piles) total += pile_shape(m, p).w * bw;
    total += GAP * static_cast<float>(piles.empty() ? 0 : piles.size() - 1);
    float x = (r.x0 + r.x1) / 2 - total / 2;
    for (size_t pi = 0; pi < piles.size(); ++pi) {
        const TablePile &p = piles[pi];
        const PileShape s = pile_shape(m, p);
        const float pw = s.w * bw, ph = s.h * bw;
        // An attacker's edge facing the middle stands `shift` beyond an upright card's.
        const float reach = (std::max(s.h, CARD_ASPECT) + s.shift) * bw;
        const float y = s.shift == 0 ? (top ? r.y0 : r.y1 - ph) : (top ? r.y0 + reach - ph : r.y1 - reach);
        const int n = static_cast<int>(p.objects.size());
        if (p.land_stack) {
            for (int j = n - 1; j >= 0; --j) {
                PlacedCard c;
                c.object = p.objects[static_cast<size_t>(j)];
                c.rect =
                    card_rect(x + static_cast<float>(std::min(j, STACK_VISIBLE - 1)) * STACK_STEP * bw, y, bw, false);
                c.row = row;
                c.pile = static_cast<int>(pi);
                c.front = j == 0;
                c.stack_count = j == 0 ? n : 0;
                out->push_back(c);
            }
        } else {
            const int k = n - 1;
            const float peek = ATTACH_PEEK * CARD_ASPECT * bw;
            for (int i = k; i >= 0; --i) {
                const TableObject &o = m.objects[static_cast<size_t>(p.objects[static_cast<size_t>(i)])];
                PlacedCard c;
                c.object = p.objects[static_cast<size_t>(i)];
                const float cw = o.tapped ? bw * CARD_ASPECT : bw;
                const float oy = i == 0 ? static_cast<float>(k) * peek : static_cast<float>(k - i) * peek;
                c.rect = card_rect(x + (pw - cw) / 2, y + oy, bw, o.tapped);
                c.rotated = o.tapped;
                c.row = row;
                c.pile = static_cast<int>(pi);
                c.front = i == 0;
                out->push_back(c);
            }
        }
        x += pw + GAP;
    }
}

namespace rmwork {

float fit_card_width(const TableModel &m, float w, float h, float fixed_h) {
    float units_h = HAND_SCALE * CARD_ASPECT;
    for (int r = 0; r < ROW_COUNT; ++r) units_h += row_height(m, r);
    float bw = (h - fixed_h) / units_h;
    const float avail = w - 2 * SIDE;
    for (int r = 0; r < ROW_COUNT; ++r) {
        const size_t n = m.rows[static_cast<size_t>(r)].size();
        if (n == 0) continue;
        bw = std::min(bw, (avail - GAP * static_cast<float>(n - 1)) / row_width(m, r));
    }
    if (!m.hand.empty()) {
        const float n = static_cast<float>(m.hand.size());
        bw = std::min(bw, (avail - HAND_GAP * (n - 1)) / (n * HAND_SCALE));
    }
    return std::max(MIN_CARD_W, std::min(bw, MAX_CARD_W));
}

TableLayout layout_table(const TableModel &m, float w, float h, const TableMetrics &mx) {
    TableLayout L;
    L.panel = TRect{0, 0, w, h};
    const MidShape mid = mid_shape(m, w, mx);
    const float band = chip_band(mx);
    const float fixed = fixed_height(mid.h, band);
    const float bw = fit_card_width(m, w, h, fixed);
    L.card_w = bw;
    L.hand_w = bw * HAND_SCALE;
    float rows_h[ROW_COUNT];
    float content = fixed + L.hand_w * CARD_ASPECT;
    for (int r = 0; r < ROW_COUNT; ++r) {
        rows_h[r] = row_height(m, r) * bw;
        content += rows_h[r];
    }
    const float spare = std::max(0.0f, h - content) / 4;  // above and below each half's rows
    float y = 0;
    L.top_band = TRect{0, y, w, y + band};
    y += band + spare;
    L.rows[ROW_TOP_LANDS] = TRect{SIDE, y, w - SIDE, y + rows_h[ROW_TOP_LANDS]};
    y += rows_h[ROW_TOP_LANDS] + ROW_GAP;
    L.rows[ROW_TOP_OTHER] = TRect{SIDE, y, w - SIDE, y + rows_h[ROW_TOP_OTHER]};
    y += rows_h[ROW_TOP_OTHER] + spare + ROW_GAP;
    L.mid = TRect{MID_MARGIN, y, w - MID_MARGIN, y + mid.h};
    L.mid_header = TRect{L.mid.x0 + MID_PAD, y + MID_PAD, L.mid.x1 - MID_PAD, y + MID_PAD + mid.header_h};
    L.steps_wrapped = mid.wrapped;
    if (!m.stack.empty()) {
        float sy = L.mid_header.y1 + 5;
        L.stack_label = TRect{L.mid_header.x0, sy, L.mid_header.x1, sy + mid.label_h};
        sy += mid.label_h;
        for (int i = 0; i < mid.shown; ++i) {
            const int line = i / mid.per_line, col = i % mid.per_line;
            const float ix = L.mid_header.x0 + static_cast<float>(col) * (mid.item_w + STACK_GAP);
            const float iy = sy + static_cast<float>(line) * (mid.item_h + STACK_GAP);
            L.stack_items.push_back(TRect{ix, iy, ix + mid.item_w, iy + mid.item_h});
        }
    }
    y += mid.h + ROW_GAP + spare;
    L.rows[ROW_BOTTOM_OTHER] = TRect{SIDE, y, w - SIDE, y + rows_h[ROW_BOTTOM_OTHER]};
    y += rows_h[ROW_BOTTOM_OTHER] + ROW_GAP;
    L.rows[ROW_BOTTOM_LANDS] = TRect{SIDE, y, w - SIDE, y + rows_h[ROW_BOTTOM_LANDS]};
    y += rows_h[ROW_BOTTOM_LANDS] + spare + ROW_GAP;
    L.hand = TRect{SIDE, y, w - SIDE, y + L.hand_w * CARD_ASPECT};
    y += L.hand_w * CARD_ASPECT + PAD;
    L.bottom_band = TRect{0, y, w, y + band};

    place_row(m, ROW_TOP_LANDS, L.rows[ROW_TOP_LANDS], bw, true, &L.cards);
    place_row(m, ROW_TOP_OTHER, L.rows[ROW_TOP_OTHER], bw, true, &L.cards);
    place_row(m, ROW_BOTTOM_OTHER, L.rows[ROW_BOTTOM_OTHER], bw, false, &L.cards);
    place_row(m, ROW_BOTTOM_LANDS, L.rows[ROW_BOTTOM_LANDS], bw, false, &L.cards);
    const float n = static_cast<float>(m.hand.size());
    float hx = w / 2 - (n * L.hand_w + HAND_GAP * std::max(0.0f, n - 1)) / 2;
    for (int i : m.hand) {
        PlacedCard c;
        c.object = i;
        c.rect = TRect{hx, L.hand.y0, hx + L.hand_w, L.hand.y1};
        c.in_hand = true;
        L.cards.push_back(c);
        hx += L.hand_w + HAND_GAP;
    }

    L.fits = bw >= MIN_CARD_W && L.bottom_band.y1 <= h + 0.5f && L.mid.inside(L.panel);
    for (const PlacedCard &c : L.cards) L.fits = L.fits && c.rect.inside(L.panel);
    for (const TRect &r : L.stack_items) L.fits = L.fits && r.inside(L.panel);
    return L;
}

}  // namespace rmwork
