#include "table_layout.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <map>

#include "settings.h"

namespace {

// Pixels.
const float GAP = 6;       // between piles in a row
const float LINE_GAP = 6;  // between a wrapped row's lines
const float HAND_GAP = 8;  // between hand cards
const float ROW_GAP = 8;   // between rows and strips
const float SIDE = 10;     // left and right of the rows
const float PAD = 6;       // around a chip and below the hand
const float MID_MARGIN = 8;
const float MID_PAD = 5;
const float STACK_GAP = 6;
const float BUTTON_GAP = 6;  // between the prompt line's buttons, and between its lines

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
        float prompt_h = 0;  // the prompt line with its buttons' lines, 0 none
        float h = 0;
};

// The rows as the layout shows them: each row's piles (groups stacked) and each line's first pile.
struct RowPlan {
        std::array<std::vector<rmwork::TablePile>, rmwork::ROW_COUNT> piles;
        std::array<std::vector<size_t>, rmwork::ROW_COUNT> starts;
};

// The card widths the panel allows: by its height, by each row's width (FLT_MAX for an empty row),
// by the hand's width.
struct FitLimits {
        float height = FLT_MAX;
        std::array<float, rmwork::ROW_COUNT> rows{{FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX}};
        float hand = FLT_MAX;
};

}  // namespace

static PileShape pile_shape(const rmwork::TableModel &m, const rmwork::TablePile &p);
static size_t line_end(const RowPlan &plan, int row, size_t line);
static float line_units_w(const rmwork::TableModel &m, const RowPlan &plan, int row, size_t line);
static float line_units_h(const rmwork::TableModel &m, const RowPlan &plan, int row, size_t line);
static float row_units_h(const rmwork::TableModel &m, const RowPlan &plan, int row);
static float row_gaps_h(const RowPlan &plan, int row);
static FitLimits fit_limits(const rmwork::TableModel &m, const RowPlan &plan, float w, float h, float fixed_h);
static float limited_width(const FitLimits &lim);
static int count_lines(const std::vector<float> &widths, float limit, std::vector<size_t> *starts);
static void split_lines(const rmwork::TableModel &m, RowPlan *plan, int row, int lines);
static bool stack_largest_group(const rmwork::TableModel &m, RowPlan *plan, int row);
static float give_rows_room(const rmwork::TableModel &m, RowPlan *plan, float w, float h, float fixed_h);
static bool shows_steps(const rmwork::TableModel &m);
static MidShape mid_shape(const rmwork::TableModel &m, float w, const rmwork::TableMetrics &mx);
static float chip_band(const rmwork::TableMetrics &mx);
static float fixed_height(float mid_h, float band);
static void place_row(const rmwork::TableModel &m, const RowPlan &plan, int row, const rmwork::TRect &r, float bw,
    bool top, std::vector<rmwork::PlacedCard> *out);
static void place_line(const rmwork::TableModel &m, const RowPlan &plan, int row, size_t line, const rmwork::TRect &r,
    float bw, bool top, std::vector<rmwork::PlacedCard> *out);
static void place_steps(
    const rmwork::TableModel &m, const MidShape &mid, const rmwork::TableMetrics &mx, rmwork::TableLayout *L);
static int place_buttons(const rmwork::TableMetrics &mx, float x0, float x1, float y, std::vector<rmwork::TRect> *out);
static rmwork::TRect card_rect(float x, float y, float bw, bool tapped);

static PileShape pile_shape(const rmwork::TableModel &m, const rmwork::TablePile &p) {
    using namespace rmwork;
    PileShape s;
    if (p.objects.empty()) return s;
    const TableObject &host = m.objects[static_cast<size_t>(p.objects[0])];
    if (p.stacked) {
        const int extra = std::min(static_cast<int>(p.objects.size()) - 1, STACK_VISIBLE - 1);
        s.w = (host.tapped ? CARD_ASPECT : 1) + static_cast<float>(extra) * STACK_STEP;
        s.h = host.tapped ? 1 : CARD_ASPECT;
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

// One past line `line`'s last pile.
static size_t line_end(const RowPlan &plan, int row, size_t line) {
    const std::vector<size_t> &st = plan.starts[static_cast<size_t>(row)];
    return line + 1 < st.size() ? st[line + 1] : plan.piles[static_cast<size_t>(row)].size();
}

// A line's width in card widths (gaps not counted).
static float line_units_w(const rmwork::TableModel &m, const RowPlan &plan, int row, size_t line) {
    float w = 0;
    const std::vector<rmwork::TablePile> &piles = plan.piles[static_cast<size_t>(row)];
    for (size_t i = plan.starts[static_cast<size_t>(row)][line]; i < line_end(plan, row, line); ++i)
        w += pile_shape(m, piles[i]).w;
    return w;
}

// A line's height in card widths: its tallest pile, never less than an upright card; an attacker's
// pile is as tall as an upright card at least, plus its shift.
static float line_units_h(const rmwork::TableModel &m, const RowPlan &plan, int row, size_t line) {
    float h = rmwork::CARD_ASPECT;
    const std::vector<rmwork::TablePile> &piles = plan.piles[static_cast<size_t>(row)];
    for (size_t i = plan.starts[static_cast<size_t>(row)][line]; i < line_end(plan, row, line); ++i) {
        const PileShape s = pile_shape(m, piles[i]);
        h = std::max(h, std::max(s.h, rmwork::CARD_ASPECT) + s.shift);
    }
    return h;
}

// A row's height in card widths: its lines' (the gaps between them not counted).
static float row_units_h(const rmwork::TableModel &m, const RowPlan &plan, int row) {
    float h = 0;
    for (size_t j = 0; j < plan.starts[static_cast<size_t>(row)].size(); ++j) h += line_units_h(m, plan, row, j);
    return h;
}

// The gaps between a row's lines, in pixels.
static float row_gaps_h(const RowPlan &plan, int row) {
    return LINE_GAP * static_cast<float>(plan.starts[static_cast<size_t>(row)].size() - 1);
}

static FitLimits fit_limits(const rmwork::TableModel &m, const RowPlan &plan, float w, float h, float fixed_h) {
    using namespace rmwork;
    FitLimits lim;
    float units_h = HAND_SCALE * CARD_ASPECT, gaps_h = 0;
    for (int r = 0; r < ROW_COUNT; ++r) {
        units_h += row_units_h(m, plan, r);
        gaps_h += row_gaps_h(plan, r);
    }
    lim.height = (h - fixed_h - gaps_h) / units_h;
    const float avail = w - 2 * SIDE;
    for (int r = 0; r < ROW_COUNT; ++r)
        for (size_t j = 0; j < plan.starts[static_cast<size_t>(r)].size(); ++j) {
            const size_t n = line_end(plan, r, j) - plan.starts[static_cast<size_t>(r)][j];
            if (n == 0) continue;
            lim.rows[static_cast<size_t>(r)] = std::min(lim.rows[static_cast<size_t>(r)],
                (avail - GAP * static_cast<float>(n - 1)) / line_units_w(m, plan, r, j));
        }
    if (!m.hand.empty()) {
        const float n = static_cast<float>(m.hand.size());
        lim.hand = (avail - HAND_GAP * (n - 1)) / (n * HAND_SCALE);
    }
    return lim;
}

// The card width the limits give, between MIN_CARD_W and MAX_CARD_W.
static float limited_width(const FitLimits &lim) {
    float bw = std::min(lim.height, lim.hand);
    for (float r : lim.rows) bw = std::min(bw, r);
    return std::max(rmwork::MIN_CARD_W, std::min(bw, rmwork::MAX_CARD_W));
}

// Greedy lines of at most `limit` card widths each (a wider pile alone); returns their count and
// fills `starts` when given.
static int count_lines(const std::vector<float> &widths, float limit, std::vector<size_t> *starts) {
    if (starts) starts->assign(1, 0);
    int lines = 1;
    float acc = 0;
    for (size_t i = 0; i < widths.size(); ++i) {
        if (acc > 0 && acc + widths[i] > limit + 1e-4f) {
            ++lines;
            acc = 0;
            if (starts) starts->push_back(i);
        }
        acc += widths[i];
    }
    return lines;
}

// Row `row`'s piles, in order, onto at most `lines` lines of as even a width as can be.
static void split_lines(const rmwork::TableModel &m, RowPlan *plan, int row, int lines) {
    std::vector<float> widths;
    float lo = 0, hi = 0;
    for (const rmwork::TablePile &p : plan->piles[static_cast<size_t>(row)]) {
        widths.push_back(pile_shape(m, p).w);
        lo = std::max(lo, widths.back());
        hi += widths.back();
    }
    for (int i = 0; i < 40 && hi - lo > 1e-3f; ++i) {
        const float mid = (lo + hi) / 2;
        if (count_lines(widths, mid, nullptr) <= lines) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    count_lines(widths, hi, &plan->starts[static_cast<size_t>(row)]);
}

// Stacks row `row`'s largest group of interchangeable piles (two or more) into one pile at the
// place of its first; false when the row has none. The row's lines are split again as many.
static bool stack_largest_group(const rmwork::TableModel &m, RowPlan *plan, int row) {
    std::vector<rmwork::TablePile> &piles = plan->piles[static_cast<size_t>(row)];
    std::map<int, int> sizes;
    for (const rmwork::TablePile &p : piles)
        if (p.group >= 0 && !p.stacked) ++sizes[p.group];
    int best = -1, most = 1;
    for (const auto &g : sizes)
        if (g.second > most) {
            best = g.first;
            most = g.second;
        }
    if (best < 0) return false;
    std::vector<rmwork::TablePile> out;
    size_t at = 0;  // the stack's place in `out`, from its first member
    bool started = false;
    for (const rmwork::TablePile &p : piles) {
        if (p.group != best || p.stacked) {
            out.push_back(p);
        } else if (!started) {
            out.push_back(p);
            out.back().stacked = true;
            at = out.size() - 1;
            started = true;
        } else {
            out[at].objects.push_back(p.objects[0]);
        }
    }
    piles.swap(out);
    split_lines(m, plan, row, static_cast<int>(plan->starts[static_cast<size_t>(row)].size()));
    return true;
}

// Gives the widest rows room while the cards are narrower than LEGIBLE_CARD_W and a row's width
// is what holds them down: stacks the row's largest group, else wraps it onto one more line when
// that widens the row's cards and narrows no other's. Returns the card width.
static float give_rows_room(const rmwork::TableModel &m, RowPlan *plan, float w, float h, float fixed_h) {
    using namespace rmwork;
    FitLimits lim = fit_limits(m, *plan, w, h, fixed_h);
    float bw = limited_width(lim);
    while (bw < LEGIBLE_CARD_W) {
        int row = -1;
        for (int r = 0; r < ROW_COUNT; ++r)
            if (row < 0 || lim.rows[static_cast<size_t>(r)] < lim.rows[static_cast<size_t>(row)]) row = r;
        if (lim.rows[static_cast<size_t>(row)] >= std::min(lim.height, lim.hand)) break;
        if (!stack_largest_group(m, plan, row)) {
            const int lines = static_cast<int>(plan->starts[static_cast<size_t>(row)].size());
            if (lines >= MAX_ROW_LINES) break;
            RowPlan trial = *plan;
            split_lines(m, &trial, row, lines + 1);
            if (static_cast<int>(trial.starts[static_cast<size_t>(row)].size()) <= lines) break;
            // Kept when it widens the row's cards without narrowing the others (another row as wide
            // may still hold them down: it is given room next).
            const FitLimits tl = fit_limits(m, trial, w, h, fixed_h);
            if (limited_width(tl) < bw - 1e-3f ||
                tl.rows[static_cast<size_t>(row)] <= lim.rows[static_cast<size_t>(row)])
                break;
            *plan = trial;
        }
        lim = fit_limits(m, *plan, w, h, fixed_h);
        bw = limited_width(lim);
    }
    return bw;
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
    if (mx.prompt_h > 0) {
        const int lines = place_buttons(mx, 0, inner, 0, nullptr);
        s.prompt_h = mx.prompt_h + static_cast<float>(lines) * (mx.prompt_h + BUTTON_GAP);
        s.h += 5 + s.prompt_h;
    }
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

// The prompt line's buttons in [x0, x1] from `y` (the prompt line's top): right-aligned on it when they
// leave the prompt at least half of it, else right-aligned on lines of their own below it, as many as
// they need; returns those lines' count (0: on the prompt line). `out` (when given) gets their rects.
static int place_buttons(const rmwork::TableMetrics &mx, float x0, float x1, float y, std::vector<rmwork::TRect> *out) {
    using rmwork::TRect;
    float total = 0;
    for (float w : mx.button_w) total += w + BUTTON_GAP;
    const float h = mx.prompt_h;
    if (total <= (x1 - x0) / 2) {
        float x = x1 - (total - BUTTON_GAP);
        for (float w : mx.button_w) {
            if (out) out->push_back(TRect{x, y, x + w, y + h});
            x += w + BUTTON_GAP;
        }
        return 0;
    }
    // Lines of their own: as many buttons a line as fit, each line right-aligned.
    int lines = 0;
    size_t i = 0;
    while (i < mx.button_w.size()) {
        size_t j = i;
        float line_w = 0;
        while (j < mx.button_w.size() && (j == i || line_w + BUTTON_GAP + mx.button_w[j] <= x1 - x0)) {
            line_w += (j == i ? 0 : BUTTON_GAP) + mx.button_w[j];
            ++j;
        }
        ++lines;
        const float ly = y + static_cast<float>(lines) * (h + BUTTON_GAP);
        float x = std::max(x0, x1 - line_w);
        for (size_t k = i; k < j; ++k) {
            const float w = std::min(mx.button_w[k], x1 - x0);
            if (out) out->push_back(TRect{x, ly, x + w, ly + h});
            x += w + BUTTON_GAP;
        }
        i = j;
    }
    return lines;
}

static rmwork::TRect card_rect(float x, float y, float bw, bool tapped) {
    const float w = tapped ? bw * rmwork::CARD_ASPECT : bw;
    const float h = tapped ? bw : bw * rmwork::CARD_ASPECT;
    return rmwork::TRect{x, y, x + w, y + h};
}

// Row `row`'s lines from the top of `r` down, each `LINE_GAP` from the last.
static void place_row(const rmwork::TableModel &m, const RowPlan &plan, int row, const rmwork::TRect &r, float bw,
    bool top, std::vector<rmwork::PlacedCard> *out) {
    float y = r.y0;
    for (size_t j = 0; j < plan.starts[static_cast<size_t>(row)].size(); ++j) {
        const float lh = line_units_h(m, plan, row, j) * bw;
        place_line(m, plan, row, j, rmwork::TRect{r.x0, y, r.x1, y + lh}, bw, top, out);
        y += lh + LINE_GAP;
    }
}

// Line `line` of row `row`, its piles centred in `r`; a top row's piles hang from its top edge, a
// bottom row's stand on its bottom edge, an attacker moved toward the middle.
static void place_line(const rmwork::TableModel &m, const RowPlan &plan, int row, size_t line, const rmwork::TRect &r,
    float bw, bool top, std::vector<rmwork::PlacedCard> *out) {
    using namespace rmwork;
    const std::vector<TablePile> &piles = plan.piles[static_cast<size_t>(row)];
    const size_t first = plan.starts[static_cast<size_t>(row)][line], last = line_end(plan, row, line);
    float total = line_units_w(m, plan, row, line) * bw;
    total += GAP * static_cast<float>(last > first ? last - first - 1 : 0);
    float x = (r.x0 + r.x1) / 2 - total / 2;
    for (size_t pi = first; pi < last; ++pi) {
        const TablePile &p = piles[pi];
        const PileShape s = pile_shape(m, p);
        const float pw = s.w * bw, ph = s.h * bw;
        // An attacker's edge facing the middle stands `shift` beyond an upright card's.
        const float reach = (std::max(s.h, CARD_ASPECT) + s.shift) * bw;
        const float y = s.shift == 0 ? (top ? r.y0 : r.y1 - ph) : (top ? r.y0 + reach - ph : r.y1 - reach);
        const int n = static_cast<int>(p.objects.size());
        PlacedCard c;
        c.row = row;
        c.pile = static_cast<int>(pi);
        c.line = static_cast<int>(line);
        if (p.stacked) {
            const bool tapped = m.objects[static_cast<size_t>(p.objects[0])].tapped;
            for (int j = n - 1; j >= 0; --j) {
                c.object = p.objects[static_cast<size_t>(j)];
                c.rect =
                    card_rect(x + static_cast<float>(std::min(j, STACK_VISIBLE - 1)) * STACK_STEP * bw, y, bw, tapped);
                c.rotated = tapped;
                c.front = j == 0;
                c.stack_count = j == 0 ? n : 0;
                out->push_back(c);
            }
        } else {
            const int k = n - 1;
            const float peek = ATTACH_PEEK * CARD_ASPECT * bw;
            for (int i = k; i >= 0; --i) {
                const TableObject &o = m.objects[static_cast<size_t>(p.objects[static_cast<size_t>(i)])];
                c.object = p.objects[static_cast<size_t>(i)];
                const float cw = o.tapped ? bw * CARD_ASPECT : bw;
                const float oy = i == 0 ? static_cast<float>(k) * peek : static_cast<float>(k - i) * peek;
                c.rect = card_rect(x + (pw - cw) / 2, y + oy, bw, o.tapped);
                c.rotated = o.tapped;
                c.front = i == 0;
                out->push_back(c);
            }
        }
        x += pw + GAP;
    }
}

// The step bar's cells: right-aligned on the header's line, or on a line of their own at its left,
// narrowed to the header's width when it is narrower than the bar.
static void place_steps(
    const rmwork::TableModel &m, const MidShape &mid, const rmwork::TableMetrics &mx, rmwork::TableLayout *L) {
    using namespace rmwork;
    if (!shows_steps(m) || mx.steps_w <= 0) return;
    const float n = static_cast<float>(STOP_STEP_COUNT);
    const float bar_w = std::min(mx.steps_w, L->mid_header.w());
    const float cw = std::max(1.0f, (bar_w - 4 * STEP_GAP) / n - 1);
    float x = mid.wrapped ? L->mid_header.x0 : L->mid_header.x1 - (n * (cw + 1) + 4 * STEP_GAP);
    const float y = mid.wrapped ? L->mid_header.y0 + mx.line_h + 4 : L->mid_header.y0;
    for (int i = 0; i < STOP_STEP_COUNT; ++i) {
        if (step_gap_before(i)) x += STEP_GAP;
        L->step_cells.push_back(TRect{x, y, x + cw, y + mx.steps_h});
        x += cw + 1;
    }
}

namespace rmwork {

bool step_gap_before(int i) {
    return i == 3 || i == 4 || i == 9 || i == 10;
}

TableLayout layout_table(const TableModel &m, float w, float h, const TableMetrics &mx) {
    TableLayout L;
    L.panel = TRect{0, 0, w, h};
    const MidShape mid = mid_shape(m, w, mx);
    const float band = chip_band(mx);
    const float fixed = fixed_height(mid.h, band);
    RowPlan plan;
    for (int r = 0; r < ROW_COUNT; ++r) {
        plan.piles[static_cast<size_t>(r)] = m.rows[static_cast<size_t>(r)];
        plan.starts[static_cast<size_t>(r)].assign(1, 0);
    }
    L.detail_w = limited_width(fit_limits(m, plan, w, h, fixed));
    const float bw = give_rows_room(m, &plan, w, h, fixed);
    L.card_w = bw;
    L.hand_w = bw * HAND_SCALE;
    float rows_h[ROW_COUNT];
    float content = fixed + L.hand_w * CARD_ASPECT;
    for (int r = 0; r < ROW_COUNT; ++r) {
        rows_h[r] = row_units_h(m, plan, r) * bw + row_gaps_h(plan, r);
        content += rows_h[r];
        L.piles[static_cast<size_t>(r)] = plan.piles[static_cast<size_t>(r)];
        L.lines[static_cast<size_t>(r)] = static_cast<int>(plan.starts[static_cast<size_t>(r)].size());
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
    place_steps(m, mid, mx, &L);
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
    if (mid.prompt_h > 0) {
        const float py = L.mid.y1 - MID_PAD - mid.prompt_h;
        L.prompt = TRect{L.mid_header.x0, py, L.mid_header.x1, py + mid.prompt_h};
        place_buttons(mx, L.prompt.x0, L.prompt.x1, py, &L.prompt_buttons);
    }
    y += mid.h + ROW_GAP + spare;
    L.rows[ROW_BOTTOM_OTHER] = TRect{SIDE, y, w - SIDE, y + rows_h[ROW_BOTTOM_OTHER]};
    y += rows_h[ROW_BOTTOM_OTHER] + ROW_GAP;
    L.rows[ROW_BOTTOM_LANDS] = TRect{SIDE, y, w - SIDE, y + rows_h[ROW_BOTTOM_LANDS]};
    y += rows_h[ROW_BOTTOM_LANDS] + spare + ROW_GAP;
    L.hand = TRect{SIDE, y, w - SIDE, y + L.hand_w * CARD_ASPECT};
    y += L.hand_w * CARD_ASPECT + PAD;
    L.bottom_band = TRect{0, y, w, y + band};

    place_row(m, plan, ROW_TOP_LANDS, L.rows[ROW_TOP_LANDS], bw, true, &L.cards);
    place_row(m, plan, ROW_TOP_OTHER, L.rows[ROW_TOP_OTHER], bw, true, &L.cards);
    place_row(m, plan, ROW_BOTTOM_OTHER, L.rows[ROW_BOTTOM_OTHER], bw, false, &L.cards);
    place_row(m, plan, ROW_BOTTOM_LANDS, L.rows[ROW_BOTTOM_LANDS], bw, false, &L.cards);
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
    for (const TRect &r : L.step_cells) L.fits = L.fits && r.inside(L.mid);
    for (const TRect &r : L.prompt_buttons) L.fits = L.fits && r.inside(L.mid);
    return L;
}

}  // namespace rmwork
