#include "selftest_table.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>

#include "imgui_internal.h"
#include "magnifier.h"
#include "settings.h"
#include "table_backdrop.h"
#include "table_layout.h"
#include "table_model.h"

namespace {

// The panel sizes the fit is checked at (the last a narrow, tall one: a Table docked beside other
// tools), and the text sizes of a 14 px UI font (Replay's step bar; Play's with its stop dots).
const float SIZES[][2] = {{640, 480}, {900, 700}, {1600, 1000}, {480, 1000}};
const rmwork::TableMetrics METRICS = {17, 420, 430, 19};
const rmwork::TableMetrics PLAY_METRICS = {17, 420, 430, 28};

// Where a handle was placed in a layout.
struct Place {
        int count = 0;
        int row = -1, pile = -1;
        bool in_hand = false;
};

}  // namespace

static int seat_of(uint64_t seat);
static std::string where(const std::string &at, size_t decision, int view);
static void check_layout(const rmwork::CheckFn &check, const rmwork::TableModel &model,
    const std::set<uint64_t> &battlefield, const std::set<uint64_t> &hand, const std::string &at,
    rmwork::TableCoverage *cov);
static void check_crowding_rules(const rmwork::CheckFn &check, const rmwork::TableModel &model,
    const rmwork::TableLayout &L, const std::string &here, rmwork::TableCoverage *cov);
static void check_objects(const rmwork::CheckFn &check, const agents::SeatStream &st, const rmstate::DecodedFrame &f,
    const rmwork::TableModel &model, int view, const std::string &at, rmwork::TableCoverage *cov);
static void check_stack(const rmwork::CheckFn &check, const agents::SeatStream &st, const rmstate::DecodedFrame &f,
    const rmwork::TableModel &model, const std::string &at, rmwork::TableCoverage *cov);
static rmwork::TableObject synthetic(uint64_t handle, int controller, int card, bool land);
static rmwork::TableModel synthetic_board(int extra_per_row, int hand);
static rmwork::TableModel crowded_board(int lands, int tokens, int distinct);
static void check_synthetic_layout(const rmwork::CheckFn &check, const rmwork::TableModel &m, const std::string &at);
static int stacked_count(const rmwork::TableLayout &L, int row);
static void check_crowding(const rmwork::CheckFn &check);
static void check_step_cells(const rmwork::CheckFn &check);
static void check_stop_flags(const rmwork::CheckFn &check);
static void check_prompt_line(const rmwork::CheckFn &check);
static void check_combat_plan(const rmwork::CheckFn &check);
static void check_magnifier_rects(const rmwork::CheckFn &check);
static bool rect_in(ImVec2 a, ImVec2 b, ImVec2 lo, ImVec2 hi);
static void check_links(const rmwork::CheckFn &check);
static void check_tiles(const rmwork::CheckFn &check);
static ImVec2 table_origin(rmwork::Mode m);
static void move_mouse(ImVec2 p, const std::function<void(int)> &frames);
static void click(ImVec2 p, const std::function<void(int)> &frames);

static int seat_of(uint64_t seat) {
    return seat == 1 ? 0 : seat == 2 ? 1 : -1;
}

static std::string where(const std::string &at, size_t decision, int view) {
    return at + ": decision " + std::to_string(decision + 1) + ", " + rmwork::view_name(view) + ": ";
}

// At each panel size: inside the panel, every battlefield and hand object placed once, each
// attachment in its host's pile, each stack's count its pile's size, and the crowding rules.
static void check_layout(const rmwork::CheckFn &check, const rmwork::TableModel &model,
    const std::set<uint64_t> &battlefield, const std::set<uint64_t> &hand, const std::string &at,
    rmwork::TableCoverage *cov) {
    using namespace rmwork;
    for (const auto &size : SIZES) {
        const TableLayout L = layout_table(model, size[0], size[1], METRICS);
        const std::string here =
            at + std::to_string(static_cast<int>(size[0])) + "x" + std::to_string(static_cast<int>(size[1])) + ": ";
        check(L.fits, here + "the layout does not fit the panel (card width " + std::to_string(L.card_w) + ")");
        std::map<uint64_t, Place> placed;
        std::map<std::pair<int, int>, int> pile_sizes;
        for (const PlacedCard &c : L.cards) {
            const TableObject &o = model.objects[static_cast<size_t>(c.object)];
            Place &p = placed[o.handle];
            ++p.count;
            p.row = c.row;
            p.pile = c.pile;
            p.in_hand = c.in_hand;
            if (!c.in_hand) ++pile_sizes[{c.row, c.pile}];
        }
        bool once = true, all = true, hands = true;
        for (const auto &p : placed) {
            if (p.second.count != 1) once = false;
            if (p.second.in_hand ? !hand.count(p.first) : !battlefield.count(p.first)) all = false;
        }
        for (uint64_t h : battlefield)
            if (!placed.count(h) || placed[h].in_hand) all = false;
        for (uint64_t h : hand)
            if (!placed.count(h) || !placed[h].in_hand) hands = false;
        check(once, here + "an object is placed more than once");
        check(all, here + "the placed objects are not the frame's battlefield objects");
        check(hands, here + "the hand placed is not the bottom seat's hand");
        bool hosts = true, stacks = true;
        for (const PlacedCard &c : L.cards) {
            const TableObject &o = model.objects[static_cast<size_t>(c.object)];
            if (o.attached_to && battlefield.count(o.attached_to) && !c.in_hand) {
                // The pile's front card is the host (or the host of the host).
                const auto host = placed.find(o.attached_to);
                if (host == placed.end() || host->second.row != c.row || host->second.pile != c.pile) hosts = false;
            }
            if (c.stack_count > 1 && pile_sizes[{c.row, c.pile}] != c.stack_count) stacks = false;
        }
        check(hosts, here + "an attachment is not in its host's pile");
        check(stacks, here + "a stack's count is not its pile's size");
        check_crowding_rules(check, model, L, here, cov);
    }
}

// A stack holds two or more identified, face-up members of one model group, each a pile of its own
// in the model; stacks and wrapped rows only where the full-detail cards are narrower than
// LEGIBLE_CARD_W, never narrowing them; a row wrapped only with no group left to stack; each card
// inside its row.
static void check_crowding_rules(const rmwork::CheckFn &check, const rmwork::TableModel &model,
    const rmwork::TableLayout &L, const std::string &here, rmwork::TableCoverage *cov) {
    using namespace rmwork;
    std::map<int, int> group_of;  // object -> its model pile's group (-1 none; absent: not alone in its pile)
    for (const auto &row : model.rows)
        for (const TablePile &p : row)
            if (p.objects.size() == 1) group_of[p.objects[0]] = p.group;
    int stacks = 0, wraps = 0;
    bool members = true, wrap_ok = true;
    for (int r = 0; r < ROW_COUNT; ++r) {
        std::map<int, int> left;  // unstacked piles per group
        for (const TablePile &p : L.piles[static_cast<size_t>(r)]) {
            if (!p.stacked) {
                if (p.group >= 0) ++left[p.group];
                continue;
            }
            ++stacks;
            const auto g0 = group_of.find(p.objects[0]);
            if (p.objects.size() < 2 || g0 == group_of.end() || g0->second < 0) members = false;
            for (int o : p.objects) {
                const auto g = group_of.find(o);
                const TableObject &obj = model.objects[static_cast<size_t>(o)];
                if (g == group_of.end() || g0 == group_of.end() || g->second != g0->second || obj.card < 0 ||
                    obj.face_down || obj.name.empty())
                    members = false;
            }
        }
        if (L.lines[static_cast<size_t>(r)] > 1) {
            ++wraps;
            for (const auto &g : left)
                if (g.second > 1) wrap_ok = false;
        }
    }
    check(members, here + "a stack's members are not interchangeable, identified piles of their own");
    check(wrap_ok, here + "a row is wrapped though it has identical piles left to stack");
    if (stacks || wraps)
        check(L.detail_w < LEGIBLE_CARD_W,
            here + "permanents are stacked or a row wrapped though every card is legible on its own (" +
                std::to_string(L.detail_w) + " px)");
    check(L.card_w + 1e-3f >= L.detail_w, here + "stacking or wrapping made the cards smaller");
    bool in_row = true;
    for (const PlacedCard &c : L.cards)
        if (!c.in_hand && !c.rect.inside(L.rows[c.row], 1.0f)) in_row = false;
    check(in_row, here + "a card stands outside its row");
    if (cov) {
        cov->stacked += stacks;
        cov->wrapped += wraps;
    }
}

// Each battlefield object's state as its frame row has it; a card the frame does not identify is
// not named; only the bottom seat's hand is listed.
static void check_objects(const rmwork::CheckFn &check, const agents::SeatStream &st, const rmstate::DecodedFrame &f,
    const rmwork::TableModel &model, int view, const std::string &at, rmwork::TableCoverage *cov) {
    using namespace rmwork;
    const agents::Table obj(f, st.schema(), "object");
    const agents::Table ctr(f, st.schema(), "object_counter");
    const int z_bf = st.enum_value("zone", "BATTLEFIELD"), z_hand = st.enum_value("zone", "HAND");
    const int creature = st.enum_value("card_type", "Creature");
    std::map<uint64_t, std::map<std::string, int>> counters;
    for (size_t r = 0; r < ctr.rows(); ++r)
        counters[ctr.u(r, ctr.col("obj"))][ctr.text(r, ctr.col("kind"))] +=
            static_cast<int>(ctr.u(r, ctr.col("count")));
    std::set<uint64_t> battlefield, hand;
    bool state = true, names = true, pt = true, counts = true;
    std::string first;
    for (size_t r = 0; r < obj.rows(); ++r) {
        const uint64_t h = obj.u(r, obj.col("handle"));
        const int zone = static_cast<int>(obj.u(r, obj.col("zone")));
        const bool in_hand = zone == z_hand && seat_of(obj.u(r, obj.col("owner"))) == model.bottom;
        if (zone == z_bf) battlefield.insert(h);
        if (in_hand) hand.insert(h);
        if (zone != z_bf && !in_hand) {
            if (model.find(h) >= 0) {
                names = false;
                first = "an object off the battlefield and the bottom hand is on the table";
            }
            continue;
        }
        const int i = model.find(h);
        if (i < 0) {
            state = false;
            continue;
        }
        const TableObject &o = model.objects[static_cast<size_t>(i)];
        const bool hidden = obj.none(r, obj.col("card"));
        if (hidden) ++cov->hidden;
        if (hidden != (o.card < 0) || hidden != o.name.empty()) {
            names = false;
            first = "object #" + std::to_string(h) + (hidden ? " is named though its frame hides it" : " is not named");
        }
        if (zone != z_bf) continue;
        if (o.tapped != (obj.u(r, obj.col("tapped")) != 0) ||
            o.damage != static_cast<int>(obj.u(r, obj.col("damage"))) ||
            o.phased_out != (obj.u(r, obj.col("phased_out")) != 0) || o.attached_to != obj.u(r, obj.col("attached_to")))
            state = false;
        if (o.tapped) ++cov->tapped;
        const bool is_creature = creature >= 0 && (obj.u(r, obj.col("types")) & (1ull << creature));
        if (is_creature && (!o.has_pt || o.power != static_cast<int>(obj.s(r, obj.col("power"))) ||
                               o.toughness != static_cast<int>(obj.s(r, obj.col("toughness")))))
            pt = false;
        int kinds = 0, loyalty = -1;
        for (const auto &k : counters[h]) {
            if (k.first == "LOYALTY") {
                loyalty = k.second;
            } else if (k.second > 0) {
                ++kinds;
            }
        }
        int got = 0;
        for (const TableCounter &c : o.counters) got += c.count;
        int want = 0;
        for (const auto &k : counters[h])
            if (k.first != "LOYALTY") want += k.second;
        if (static_cast<int>(o.counters.size()) != kinds || got != want || o.loyalty != loyalty) counts = false;
        cov->counters += kinds;
    }
    check(state, at + "an object's tapped, damage, phasing or attachment is not its frame's");
    check(names, at + first);
    check(pt, at + "a creature's P/T is not its frame's");
    check(counts, at + "an object's counters are not its frame's");
    for (const TableObject &o : model.objects)
        if (o.in_hand && o.owner != model.bottom) check(false, at + "the top seat's hand is listed");
    if (view < 2) {
        const agents::Table menu(f, st.schema(), "menu");
        check(model.menu.size() == menu.rows(), at + "the menu links are not one per menu row");
    }
    for (const auto &row : model.rows) {
        std::map<int, int> groups;
        for (const TablePile &p : row) {
            cov->attachments += static_cast<int>(p.objects.size()) - 1;
            if (p.group >= 0 && ++groups[p.group] == 2) ++cov->identical;
        }
    }
    check_layout(check, model, battlefield, hand, at, cov);
}

// The stack top first, each object's targets in announcement order, resolved to the table's
// objects, the other stack objects or the frame's other objects, players by seat.
static void check_stack(const rmwork::CheckFn &check, const agents::SeatStream &st, const rmstate::DecodedFrame &f,
    const rmwork::TableModel &model, const std::string &at, rmwork::TableCoverage *cov) {
    const agents::Table stack(f, st.schema(), "stack");
    const agents::Table targets(f, st.schema(), "stack_target");
    const agents::Table obj(f, st.schema(), "object");
    std::set<uint64_t> handles;
    for (size_t r = 0; r < obj.rows(); ++r) handles.insert(obj.u(r, obj.col("handle")));
    check(model.stack.size() == stack.rows(), at + "the stack's size is not the frame's");
    bool order = true, resolved = true;
    for (size_t r = 0; r < stack.rows(); ++r) {
        const size_t pos = static_cast<size_t>(stack.u(r, stack.col("pos")));
        const uint64_t h = stack.u(r, stack.col("handle"));
        if (pos >= model.stack.size() || model.stack[pos].handle != h) {
            order = false;
            continue;
        }
        std::vector<std::pair<uint64_t, size_t>> rows;
        for (size_t t = 0; t < targets.rows(); ++t)
            if (targets.u(t, targets.col("stack")) == h) rows.emplace_back(targets.u(t, targets.col("order")), t);
        std::sort(rows.begin(), rows.end());
        const rmwork::TableStackItem &item = model.stack[pos];
        if (item.targets.size() != rows.size()) {
            resolved = false;
            continue;
        }
        for (size_t k = 0; k < rows.size(); ++k) {
            const size_t t = rows[k].second;
            const rmwork::StackTarget &got = item.targets[k];
            ++cov->stack_targets;
            const bool player = st.enum_name("target_kind", targets.u(t, targets.col("kind"))) == "PLAYER";
            if (got.player != player || got.label.empty()) resolved = false;
            if (player && got.seat != seat_of(targets.u(t, targets.col("target_player")))) resolved = false;
            if (!player && got.handle != targets.u(t, targets.col("target"))) resolved = false;
            if (!player && got.handle && !handles.count(got.handle)) resolved = false;
        }
    }
    check(order, at + "the stack is not top first");
    check(resolved, at + "a stack object's targets are not the frame's");
    const agents::Table combat(f, st.schema(), "combat");
    size_t blocks = 0;
    for (size_t r = 0; r < combat.rows(); ++r)
        if (combat.u(r, combat.col("blocker"))) ++blocks;
    check(model.blocks.size() == blocks, at + "the blocks are not the frame's");
    cov->blocks += static_cast<int>(blocks);
}

static rmwork::TableObject synthetic(uint64_t handle, int controller, int card, bool land) {
    rmwork::TableObject o;
    o.handle = handle;
    o.controller = o.owner = controller;
    o.card = card;
    o.name = "Card " + std::to_string(card);
    o.land = land;
    o.creature = !land;
    o.has_pt = !land;
    return o;
}

// A board: per row three identical lands, a tapped land, a creature wearing two attachments,
// an attacking creature and `extra_per_row` more; `hand` cards in the bottom hand.
static rmwork::TableModel synthetic_board(int extra_per_row, int hand) {
    using namespace rmwork;
    TableModel m;
    uint64_t h = 1;
    for (int seat = 0; seat < 2; ++seat) {
        const bool top = seat == m.top;
        std::vector<TablePile> &lands = m.rows[top ? ROW_TOP_LANDS : ROW_BOTTOM_LANDS];
        std::vector<TablePile> &other = m.rows[top ? ROW_TOP_OTHER : ROW_BOTTOM_OTHER];
        for (int k = 0; k < 3; ++k) {
            m.objects.push_back(synthetic(h++, seat, 1, true));
            TablePile p;
            p.objects.push_back(static_cast<int>(m.objects.size()) - 1);
            p.group = seat;
            lands.push_back(p);
        }
        m.objects.push_back(synthetic(h++, seat, 2, true));
        m.objects.back().tapped = true;
        lands.push_back({{static_cast<int>(m.objects.size()) - 1}, false, -1});
        TablePile worn;
        m.objects.push_back(synthetic(h++, seat, 3, false));
        const uint64_t host = m.objects.back().handle;
        worn.objects.push_back(static_cast<int>(m.objects.size()) - 1);
        for (int k = 0; k < 2; ++k) {
            m.objects.push_back(synthetic(h++, seat, 4, false));
            m.objects.back().attached_to = host;
            m.objects.back().tapped = k == 1;
            worn.objects.push_back(static_cast<int>(m.objects.size()) - 1);
        }
        other.push_back(worn);
        m.objects.push_back(synthetic(h++, seat, 5, false));
        m.objects.back().attacking = m.objects.back().tapped = true;
        other.push_back({{static_cast<int>(m.objects.size()) - 1}, false, -1});
        for (int k = 0; k < extra_per_row; ++k) {
            m.objects.push_back(synthetic(h++, seat, 6 + k, k % 2 == 0));
            (k % 2 == 0 ? lands : other).push_back({{static_cast<int>(m.objects.size()) - 1}, false, -1});
        }
    }
    for (int k = 0; k < hand; ++k) {
        m.objects.push_back(synthetic(h++, m.bottom, 40 + k, false));
        m.objects.back().in_hand = true;
        m.hand.push_back(static_cast<int>(m.objects.size()) - 1);
    }
    m.turn = 5;
    m.step = 5;
    m.active = 0;
    m.decider = 1;
    for (int k = 0; k < 7; ++k) {
        TableStackItem it;
        it.handle = 1000 + static_cast<uint64_t>(k);
        it.name = "Spell";
        m.stack.push_back(it);
    }
    m.index();
    return m;
}

static void check_synthetic_layout(const rmwork::CheckFn &check, const rmwork::TableModel &m, const std::string &at) {
    using namespace rmwork;
    std::set<uint64_t> battlefield, hand;
    for (const TableObject &o : m.objects) (o.in_hand ? hand : battlefield).insert(o.handle);
    check_layout(check, m, battlefield, hand, at, nullptr);
    const TableLayout small = layout_table(m, 640, 480, METRICS), big = layout_table(m, 1600, 1000, METRICS);
    check(big.card_w > small.card_w, at + "a larger panel does not give larger cards");
    check(small.stack_items.size() <= m.stack.size() && small.stack_items.empty() == m.stack.empty(),
        at + "the stack shows no object, or more than it has");
    // The attacker stands toward the middle: below its row's top for the top seat.
    for (const PlacedCard &c : big.cards) {
        const TableObject &o = m.objects[static_cast<size_t>(c.object)];
        if (!o.attacking) continue;
        const TRect &row = big.rows[c.row];
        const bool top = c.row == ROW_TOP_OTHER;
        check(top ? c.rect.y0 > row.y0 + 1 : c.rect.y1 < row.y1 - 1, at + "an attacker does not stand forward");
        check(c.rotated && c.rect.w() > c.rect.h(), at + "a tapped attacker is not on its side");
    }
    for (const PlacedCard &c : big.cards) check(c.stack_count == 0, at + "a wide panel stacks identical permanents");
}

// Per seat `lands` identical untapped lands, `tokens` identical tokens and `distinct` creatures of
// different cards, the tokens and creatures in one row, in turns; a turn in progress.
static rmwork::TableModel crowded_board(int lands, int tokens, int distinct) {
    using namespace rmwork;
    TableModel m;
    uint64_t h = 1;
    for (int seat = 0; seat < 2; ++seat) {
        const bool top = seat == m.top;
        std::vector<TablePile> &land_row = m.rows[top ? ROW_TOP_LANDS : ROW_BOTTOM_LANDS];
        std::vector<TablePile> &other = m.rows[top ? ROW_TOP_OTHER : ROW_BOTTOM_OTHER];
        for (int k = 0; k < lands; ++k) {
            m.objects.push_back(synthetic(h++, seat, 1, true));
            land_row.push_back({{static_cast<int>(m.objects.size()) - 1}, false, 10 * seat});
        }
        for (int k = 0; k < std::max(tokens, distinct); ++k) {
            if (k < tokens) {
                m.objects.push_back(synthetic(h++, seat, 900, false));
                m.objects.back().token = true;
                other.push_back({{static_cast<int>(m.objects.size()) - 1}, false, 10 * seat + 1});
            }
            if (k < distinct) {
                m.objects.push_back(synthetic(h++, seat, 100 + k, false));
                other.push_back({{static_cast<int>(m.objects.size()) - 1}, false, -1});
            }
        }
    }
    m.turn = 7;
    m.step = 3;
    m.active = 0;
    m.decider = 0;
    m.index();
    return m;
}

// How many piles of row `row` are stacks.
static int stacked_count(const rmwork::TableLayout &L, int row) {
    int n = 0;
    for (const rmwork::TablePile &p : L.piles[static_cast<size_t>(row)]) n += p.stacked ? 1 : 0;
    return n;
}

// Identical lands and tokens stack only on a narrow panel, each row's group into one stack; a row
// of different creatures wraps; a row with both stacks its tokens first, then wraps; the
// general rules (check_layout) on each.
static void check_crowding(const rmwork::CheckFn &check) {
    using namespace rmwork;
    const TableModel crowd = crowded_board(12, 10, 0);
    std::set<uint64_t> battlefield;
    for (const TableObject &o : crowd.objects) battlefield.insert(o.handle);
    check_layout(check, crowd, battlefield, {}, "crowding, lands and tokens: ", nullptr);
    const TableLayout wide = layout_table(crowd, 1600, 1000, METRICS);
    bool none = true;
    for (int r = 0; r < ROW_COUNT; ++r) none = none && stacked_count(wide, r) == 0 && wide.lines[r] == 1;
    check(none && wide.card_w >= LEGIBLE_CARD_W && wide.card_w == wide.detail_w,
        "crowding: a wide panel stacks or wraps 12 lands and 10 tokens");
    const TableLayout narrow = layout_table(crowd, 640, 1000, METRICS);
    bool stacked = narrow.detail_w < LEGIBLE_CARD_W && narrow.card_w >= LEGIBLE_CARD_W && narrow.fits;
    for (int r = 0; r < ROW_COUNT; ++r) {
        const bool lands = r == ROW_TOP_LANDS || r == ROW_BOTTOM_LANDS;
        stacked = stacked && stacked_count(narrow, r) == 1 && narrow.piles[r].size() == 1 && narrow.lines[r] == 1 &&
                  narrow.piles[r][0].objects.size() == static_cast<size_t>(lands ? 12 : 10);
    }
    check(stacked, "crowding: a narrow panel does not stack 12 lands and 10 tokens into one pile a row (card width " +
                       std::to_string(narrow.card_w) + ", at full detail " + std::to_string(narrow.detail_w) + ")");
    int counted = 0;
    for (const PlacedCard &c : narrow.cards) counted += c.stack_count;
    check(counted == 44, "crowding: the stacks' counts are not their sizes");

    const TableModel creatures = crowded_board(0, 0, 14);
    battlefield.clear();
    for (const TableObject &o : creatures.objects) battlefield.insert(o.handle);
    check_layout(check, creatures, battlefield, {}, "crowding, creatures: ", nullptr);
    const TableLayout wrapped = layout_table(creatures, 900, 1000, METRICS);
    check(wrapped.lines[ROW_TOP_OTHER] == 2 && wrapped.lines[ROW_BOTTOM_OTHER] == 2 &&
              stacked_count(wrapped, ROW_TOP_OTHER) == 0 && wrapped.card_w >= LEGIBLE_CARD_W &&
              wrapped.detail_w < LEGIBLE_CARD_W && wrapped.fits,
        "crowding: 14 different creatures on a narrow panel do not wrap onto two lines (card width " +
            std::to_string(wrapped.card_w) + ")");
    bool lines_ok = true;
    for (const PlacedCard &c : wrapped.cards)
        if (c.row == ROW_TOP_OTHER) lines_ok = lines_ok && (c.line == (c.pile < 7 ? 0 : 1));
    check(lines_ok, "crowding: a wrapped row's piles are not in order, seven a line");
    check(layout_table(creatures, 1600, 1000, METRICS).lines[ROW_TOP_OTHER] == 1,
        "crowding: a wide panel wraps 14 creatures");

    const TableModel both = crowded_board(0, 10, 12);
    battlefield.clear();
    for (const TableObject &o : both.objects) battlefield.insert(o.handle);
    check_layout(check, both, battlefield, {}, "crowding, tokens and creatures: ", nullptr);
    const TableLayout mixed = layout_table(both, 900, 1000, METRICS);
    check(stacked_count(mixed, ROW_BOTTOM_OTHER) == 1 && mixed.lines[ROW_BOTTOM_OTHER] == 2 &&
              mixed.piles[ROW_BOTTOM_OTHER].size() == 13 && mixed.card_w >= LEGIBLE_CARD_W,
        "crowding: 10 tokens and 12 creatures do not stack the tokens, then wrap (" +
            std::to_string(mixed.piles[ROW_BOTTOM_OTHER].size()) + " piles, " +
            std::to_string(mixed.lines[ROW_BOTTOM_OTHER]) + " lines)");
    // A short panel: the height holds the cards down, so nothing wraps.
    const TableLayout short_panel = layout_table(both, 900, 480, METRICS);
    check(short_panel.lines[ROW_BOTTOM_OTHER] == 1, "crowding: a row wraps on a panel too short for it");
}

// The step bar: twelve cells left to right, apart, inside the middle strip's header (narrowed on a
// panel narrower than the bar), right-aligned on a wide panel and on their own line at its left on a
// narrow one; taller in Play (the stop dots); none before the first turn.
static void check_step_cells(const rmwork::CheckFn &check) {
    using namespace rmwork;
    const TableModel m = synthetic_board(2, 3);
    for (const TableMetrics &mx : {METRICS, PLAY_METRICS})
        for (float w : {1600.0f, 600.0f, 340.0f}) {
            const TableLayout L = layout_table(m, w, 1000, mx);
            const std::string at = std::string("step bar, ") + (mx.steps_h > METRICS.steps_h ? "Play" : "Replay") +
                                   ", " + std::to_string(static_cast<int>(w)) + " px: ";
            check(L.step_cells.size() == STOP_STEP_COUNT, at + "not twelve cells");
            bool ordered = true, inside = true, tall = true;
            for (size_t i = 0; i < L.step_cells.size(); ++i) {
                const TRect &c = L.step_cells[i];
                inside = inside && c.inside(L.mid_header);
                tall = tall && std::fabs(c.h() - mx.steps_h) < 0.01f;
                if (i > 0) ordered = ordered && c.x0 > L.step_cells[i - 1].x1;
            }
            check(ordered && inside && tall, at + "the cells overlap, leave the header or are not the bar's height");
            if (!L.step_cells.empty())
                check(L.steps_wrapped ? std::fabs(L.step_cells[0].x0 - L.mid_header.x0) < 0.5f
                                      : std::fabs(L.step_cells.back().x1 - L.mid_header.x1) < 1.5f,
                    at + "the bar is not where its line puts it");
        }
    TableModel pregame = m;
    pregame.pregame = true;
    check(layout_table(pregame, 1600, 1000, PLAY_METRICS).step_cells.empty(), "step bar: cells before the first turn");
}

// A combat plan's bookkeeping (table_combat.h), and a planned attacker standing forward like an attacker.
static void check_combat_plan(const rmwork::CheckFn &check) {
    using namespace rmwork;
    CombatPlan plan;
    PlanKey attack;
    attack.game = 0;
    attack.turn = 3;
    plan.follow(attack);
    const unsigned g0 = plan.generation();
    plan.plan_attack(PlannedAttack{7, 0, 1});
    plan.arm(7);
    plan.plan_attack(PlannedAttack{7, 20, -1});
    plan.plan_attack(PlannedAttack{8, 0, 1});
    check(plan.attacks().size() == 2 && plan.attack_of(7)->target == 20 && plan.armed() == 7 && plan.generation() != g0,
        "combat plan: a second plan of a creature does not replace its first");
    plan.drop(7);
    check(!plan.attack_of(7) && plan.armed() == 0 && plan.attacks().size() == 1,
        "combat plan: a dropped attacker stays planned or armed");
    plan.follow(attack);
    check(plan.attacks().size() == 1, "combat plan: following its own declaration drops it");
    PlanKey block = attack;
    block.blockers = true;
    plan.follow(block);
    check(plan.empty() && plan.key() == block, "combat plan: another declaration keeps the old plan");
    plan.plan_block(PlannedBlock{5, 9});
    plan.start_sending();
    PlanStep step;
    step.row = 0;
    step.creature = 5;
    plan.record(step, 12);
    check(plan.sending() && plan.answered() == 12 && plan.sent(5) && !plan.sent(9),
        "combat plan: a sent reply is not recorded");
    plan.clear();
    check(plan.empty() && !plan.sending() && plan.answered() == -1 && !plan.sent(5) && plan.key() == block,
        "combat plan: clear leaves something planned or sending");

    TableModel m = synthetic_board(2, 3);
    int creature = -1;
    for (size_t i = 0; i < m.objects.size() && creature < 0; ++i)
        if (m.objects[i].creature && !m.objects[i].in_hand && m.objects[i].controller == m.bottom)
            creature = static_cast<int>(i);
    check(creature >= 0, "combat plan: the synthetic board has no creature of the bottom seat");
    if (creature < 0) return;
    const auto top_of = [creature](const TableLayout &L) {
        for (const PlacedCard &c : L.cards)
            if (c.object == creature) return c.rect.y0;
        return -1.0f;
    };
    const float home = top_of(layout_table(m, 1600, 1000, PLAY_METRICS));
    m.objects[static_cast<size_t>(creature)].planned_attack = true;
    const float planned = top_of(layout_table(m, 1600, 1000, PLAY_METRICS));
    m.objects[static_cast<size_t>(creature)].planned_attack = false;
    m.objects[static_cast<size_t>(creature)].attacking = true;
    const float attacking = top_of(layout_table(m, 1600, 1000, PLAY_METRICS));
    check(home >= 0 && planned < home && std::fabs(planned - attacking) < 0.5f,
        "combat plan: a planned attacker does not stand forward as an attacker does");
}

// The decision's prompt line at the middle strip's foot, below the stack: its buttons in order, right-
// aligned on it while they leave the prompt half the line, else on lines of their own below it; all inside.
static void check_prompt_line(const rmwork::CheckFn &check) {
    using namespace rmwork;
    TableModel m = synthetic_board(2, 3);
    TableStackItem it;
    it.handle = 999;
    it.name = "Lightning Bolt";
    m.stack.push_back(it);
    const TableLayout none = layout_table(m, 1600, 1000, PLAY_METRICS);
    check(none.prompt.w() == 0 && none.prompt_buttons.empty(), "prompt line: one without a decision");
    for (const std::vector<float> &buttons :
        {std::vector<float>{}, std::vector<float>{180}, std::vector<float>{220, 90}, std::vector<float>{500, 500, 500}})
        for (float w : {1600.0f, 700.0f}) {
            TableMetrics mx = PLAY_METRICS;
            mx.prompt_h = mx.line_h + 8;
            mx.button_w = buttons;
            const TableLayout L = layout_table(m, w, 1000, mx);
            const std::string at = "prompt line, " + std::to_string(buttons.size()) + " buttons, " +
                                   std::to_string(static_cast<int>(w)) + " px: ";
            check(L.fits && L.prompt.w() > 0 && L.prompt.inside(L.mid) && !L.stack_items.empty() &&
                      L.prompt.y0 >= L.stack_items.back().y1,
                at + "not inside the strip below the stack");
            check(L.prompt_buttons.size() == buttons.size(), at + "a button missing");
            float total = 0;
            for (float b : buttons) total += b + 6;
            const bool inline_ok = total <= L.prompt.w() / 2;
            bool ordered = true, placed = true;
            for (size_t i = 0; i < L.prompt_buttons.size(); ++i) {
                const TRect &b = L.prompt_buttons[i];
                placed = placed && b.inside(L.prompt) &&
                         (inline_ok ? std::fabs(b.y0 - L.prompt.y0) < 0.5f : b.y0 >= L.prompt.y0 + mx.prompt_h);
                if (i > 0 && std::fabs(b.y0 - L.prompt_buttons[i - 1].y0) < 0.5f)
                    ordered = ordered && b.x0 > L.prompt_buttons[i - 1].x1;
            }
            check(ordered && placed, at + "the buttons overlap, leave the line or are not where their width puts them");
            if (inline_ok && !L.prompt_buttons.empty())
                check(std::fabs(L.prompt_buttons.back().x1 - L.prompt.x1) < 0.5f,
                    at + "the buttons are not right-aligned");
        }
}

// A step's stop toggles in the turn it names only, and the settings file keeps it.
static void check_stop_flags(const rmwork::CheckFn &check) {
    using namespace rmwork;
    Settings s = default_settings();
    const StopSettings before = s.stops;
    bool *flag = stop_flag(&s.stops, false, 2);
    *flag = !*flag;
    check(s.stops.opponent[2] != before.opponent[2] && s.stops.own == before.own,
        "stops: toggling the opponent's draw step changes another stop");
    *stop_flag(&s.stops, true, 8) = !before.own[8];
    Settings back;
    std::string err;
    check(settings_from_json(settings_to_json(s), &back, &err) && back.stops.own == s.stops.own &&
              back.stops.opponent == s.stops.opponent,
        "stops: toggled stops do not round-trip through the settings file " + err);
}

static bool rect_in(ImVec2 a, ImVec2 b, ImVec2 lo, ImVec2 hi) {
    return a.x >= lo.x - 0.5f && a.y >= lo.y - 0.5f && b.x <= hi.x + 0.5f && b.y <= hi.y + 0.5f;
}

// The magnifier: a double-faced card's two faces side by side, the face up first, as large, with the
// box under both, on the screen, beside the card when there is room; shrunk on a small screen.
static void check_magnifier_rects(const rmwork::CheckFn &check) {
    using namespace rmwork;
    struct Case {
            ImVec2 vp, card_min, card_max;
            int faces;
            const char *what;
    };
    const Case cases[] = {{ImVec2(1600, 1000), ImVec2(300, 400), ImVec2(380, 510), 2, "two faces, card at the left"},
        {ImVec2(1600, 1000), ImVec2(1450, 800), ImVec2(1530, 910), 2, "two faces, card at the bottom right"},
        {ImVec2(1600, 1000), ImVec2(300, 400), ImVec2(380, 510), 1, "one face"},
        {ImVec2(520, 420), ImVec2(200, 150), ImVec2(260, 230), 2, "two faces, small screen"}};
    for (const Case &c : cases) {
        const std::string at = std::string("magnifier, ") + c.what + ": ";
        const MagnifierRects r = magnifier_rects(c.card_min, c.card_max, ImVec2(0, 0), c.vp, c.faces, 21, 80);
        check(r.faces == c.faces, at + "the number of faces");
        bool on_screen = rect_in(r.box_min, r.box_max, ImVec2(0, 0), c.vp);
        for (int i = 0; i < r.faces; ++i)
            on_screen = on_screen && rect_in(r.face_min[i], r.face_max[i], ImVec2(0, 0), c.vp);
        check(on_screen, at + "a face or the box is off the screen");
        const float w0 = r.face_max[0].x - r.face_min[0].x, h0 = r.face_max[0].y - r.face_min[0].y;
        check(std::fabs(h0 / w0 - CARD_ASPECT) < 0.01f, at + "a face is not card-shaped");
        if (r.faces == 2) {
            const float w1 = r.face_max[1].x - r.face_min[1].x;
            check(std::fabs(w0 - w1) < 0.01f && r.face_min[1].x > r.face_max[0].x &&
                      std::fabs(r.face_min[1].y - r.face_min[0].y) < 0.01f,
                at + "the other face is not as large, to the right of the face up");
            check(r.caption_min[0].y < r.face_min[0].y, at + "no caption above the faces");
            check(std::fabs(r.box_min.x - r.face_min[0].x) < 0.01f && std::fabs(r.box_max.x - r.face_max[1].x) < 0.01f,
                at + "the box is not under both faces");
        }
        check(r.box_min.y >= r.face_max[0].y, at + "the box is not under the faces");
        const bool beside = r.box_max.x <= c.card_min.x || r.box_min.x >= c.card_max.x;
        if (c.vp.x >= 1600) check(beside && w0 >= 200, at + "not beside the card, at least 200 px wide");
    }
}

static void check_links(const rmwork::CheckFn &check) {
    using namespace rmwork;
    LinkState l;
    l.set_context(1);
    l.set_menu({{10, 20, -1, -1}, {0, 0, 1, -1}, {30, 0, -1, 0}});
    StackTarget a, b;
    a.handle = 40;
    b.player = true;
    b.seat = 1;
    l.set_stack_targets({{50, {a, b}}});
    l.hover_object(10);
    l.new_frame();
    check(l.object_linked(10) && l.row_linked(0) && !l.row_linked(1), "links: a hovered source does not light its row");
    l.hover_object(50);
    l.new_frame();
    check(l.object_linked(40) && l.player_linked(1) && !l.object_linked(10),
        "links: a stack object's targets are not lit");
    l.hover_menu_row(2);
    l.new_frame();
    check(l.object_linked(30) && l.player_linked(0), "links: a menu row does not light its source and target");
    l.hover_player(1);
    l.new_frame();
    check(l.row_linked(1) && !l.row_linked(0), "links: a player does not light its rows");
    LinkItem sel;
    sel.kind = LinkItem::Kind::OBJECT;
    sel.handle = 20;
    l.select(sel);
    l.new_frame();
    check(l.row_linked(0) && l.object_linked(20), "links: the selection does not link with nothing hovered");
    l.set_context(2);
    l.new_frame();
    check(
        !l.object_linked(20) && l.selected().kind == LinkItem::Kind::NONE, "links: a new decision keeps the selection");
    check(l.is_source(30) && !l.is_source(20), "links: is_source");
}

static void check_tiles(const rmwork::CheckFn &check) {
    for (int i = 0; i < rmwork::TEXTURE_COUNT; ++i) {
        const std::string key = rmwork::TABLE_TEXTURES[i].key;
        int w = 0, h = 0;
        const std::vector<uint32_t> px = rmwork::TableBackdrop::make_tile(key, &w, &h);
        check(w > 0 && h > 0 && px.size() == static_cast<size_t>(w) * static_cast<size_t>(h),
            "texture " + key + ": the tile's size");
        int w2 = 0, h2 = 0;
        check(rmwork::TableBackdrop::make_tile(key, &w2, &h2) == px, "texture " + key + " is not deterministic");
        std::set<uint32_t> distinct(px.begin(), px.end());
        check(distinct.size() > 1, "texture " + key + " is one flat colour");
        bool opaque = true;
        for (uint32_t p : px) opaque = opaque && (p >> IM_COL32_A_SHIFT) == 255;
        check(key == "slate" || opaque, "texture " + key + " is not opaque");
    }
}

// The Table window's inner top-left corner in mode `m`.
static ImVec2 table_origin(rmwork::Mode m) {
    const ImGuiWindow *w = ImGui::FindWindowByName(rmwork::tool_window_name(m, rmwork::Tool::TABLE).c_str());
    return w ? w->InnerRect.Min : ImVec2(0, 0);
}

static void move_mouse(ImVec2 p, const std::function<void(int)> &frames) {
    ImGui::GetIO().AddMousePosEvent(p.x, p.y);
    frames(2);
}

// A left click at `p`.
static void click(ImVec2 p, const std::function<void(int)> &frames) {
    move_mouse(p, frames);
    ImGuiIO &io = ImGui::GetIO();
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    frames(1);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    frames(1);
}

namespace rmwork {

ImVec2 table_window_origin(Mode m) {
    return table_origin(m);
}

void mouse_to(ImVec2 p, const std::function<void(int)> &frames) {
    move_mouse(p, frames);
}

void mouse_click(ImVec2 p, const std::function<void(int)> &frames) {
    click(p, frames);
}

void mouse_drag(ImVec2 from, ImVec2 to, const std::function<void(int)> &frames) {
    move_mouse(from, frames);
    ImGuiIO &io = ImGui::GetIO();
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    frames(1);
    for (int k = 1; k <= 4; ++k) {
        const float t = static_cast<float>(k) / 4;
        io.AddMousePosEvent(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t);
        frames(1);
    }
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    frames(2);
}

void check_table_frames(
    const CheckFn &check, TableView &view, const Match &m, const std::string &at, TableCoverage *cov) {
    const size_t frames = m.decisions.size() + (m.has_terminal() ? 1 : 0);
    for (size_t i = 0; i < frames; ++i) {
        for (int v = 0; v < VIEW_COUNT; ++v) {
            std::string err;
            const auto f = m.frame(i, v, &err);
            const std::string here = where(at, i, v);
            if (!f) {
                check(false, here + "the frame does not decode: " + err);
                continue;
            }
            const DecisionInfo *d = i < m.decisions.size() ? &m.decisions[i] : nullptr;
            const TableModel model = build_table_model(
                m.stream(v), view.book(m, v), *f, v, d ? d->decider : -1, d && d->phase == "Opening hand");
            check(model.bottom == (v == VIEW_B ? 1 : 0), here + "the bottom seat is not the viewer's");
            check_objects(check, m.stream(v), *f, model, v, here, cov);
            check_stack(check, m.stream(v), *f, model, here, cov);
            ++cov->frames;
        }
    }
}

void check_table_drawing(
    const CheckFn &check, App &app, const std::function<void(int)> &frames, const std::string &at, TableCoverage *cov) {
    ReplaySession &s = app.replay();
    if (!s.loaded()) return;
    app.set_tool_open(Mode::REPLAY, Tool::TABLE, true);
    // Every decision in every view, drawn; the first with a stack target on the table and the first
    // where a seat's playable card links menu rows are kept for the interactions.
    int stack_at = -1, stack_view = -1, source_at = -1;
    for (size_t i = 0; i < s.count(); ++i) {
        for (int v = 0; v < VIEW_COUNT; ++v) {
            s.go_to(i);
            s.set_view(v);
            frames(1);
            const TableModel *model = app.table().model();
            check(model != nullptr, at + ": decision " + std::to_string(i + 1) + " draws no table");
            if (!model) continue;
            if (stack_at < 0)
                for (const TableStackItem &it : model->stack)
                    for (const StackTarget &t : it.targets)
                        if (!t.player && model->find(t.handle) >= 0 &&
                            app.table().layout().stack_items.size() == model->stack.size()) {
                            stack_at = static_cast<int>(i);
                            stack_view = v;
                        }
            if (source_at < 0 && v == model->decider && v < 2)
                for (const MenuLink &l : model->menu)
                    if (l.source && model->find(l.source) >= 0) source_at = static_cast<int>(i);
        }
    }
    check(app.table().model() != nullptr, at + ": the Table draws nothing");
    if (stack_at >= 0) {
        s.go_to(static_cast<size_t>(stack_at));
        s.set_view(stack_view);
        frames(2);
        const TableModel &model = *app.table().model();
        const TableLayout &L = app.table().layout();
        const ImVec2 o = table_origin(Mode::REPLAY);
        size_t k = 0;
        uint64_t target = 0;
        for (size_t j = 0; j < model.stack.size() && !target; ++j)
            for (const StackTarget &t : model.stack[j].targets)
                if (!t.player && model.find(t.handle) >= 0) {
                    k = j;
                    target = t.handle;
                }
        const TRect &r = L.stack_items[k];
        move_mouse(ImVec2(o.x + (r.x0 + r.x1) / 2, o.y + (r.y0 + r.y1) / 2), frames);
        check(app.links().hovered().handle == model.stack[k].handle && app.links().object_linked(target),
            at + ": hovering a stack object does not light its target");
        ++cov->hover_checks;
    }
    if (source_at >= 0) {
        s.go_to(static_cast<size_t>(source_at));
        s.set_view(s.current()->decider);
        frames(2);
        const TableModel &model = *app.table().model();
        const TableLayout &L = app.table().layout();
        const ImVec2 o = table_origin(Mode::REPLAY);
        for (const PlacedCard &c : L.cards) {
            const uint64_t h = model.objects[static_cast<size_t>(c.object)].handle;
            int row = -1;
            for (size_t r = 0; r < model.menu.size() && row < 0; ++r)
                if (model.menu[r].source == h) row = static_cast<int>(r);
            if (row < 0) continue;
            const ImVec2 p(o.x + (c.rect.x0 + c.rect.x1) / 2, o.y + (c.rect.y0 + c.rect.y1) / 2);
            move_mouse(p, frames);
            check(app.links().row_linked(row), at + ": hovering a playable card does not light its menu row");
            // Q held over the card magnifies it until released.
            ImGuiIO &io = ImGui::GetIO();
            io.AddKeyEvent(ImGuiKey_Q, true);
            frames(2);
            check(app.table().magnifier_open() && app.table().magnified() == h, at + ": Q does not magnify the card");
            io.AddKeyEvent(ImGuiKey_Q, false);
            frames(2);
            check(!app.table().magnifier_open(), at + ": releasing Q does not close the magnifier");
            io.AddMouseButtonEvent(ImGuiMouseButton_Right, true);
            frames(1);
            io.AddMouseButtonEvent(ImGuiMouseButton_Right, false);
            frames(1);
            check(app.table().magnifier_open() && app.table().magnified() == h,
                at + ": a right-click does not magnify the card");
            move_mouse(ImVec2(o.x + L.mid.x1 - 3, o.y + L.mid.y0 + 3), frames);
            check(!app.table().magnifier_open(), at + ": leaving the card does not close a right-click magnifier");
            ++cov->magnifier_checks;
            break;
        }
    }
    // Replay's step bar shows no stops; a click on it toggles none.
    if (app.table().model() && !app.table().layout().step_cells.empty()) {
        const StopSettings before = app.settings().stops;
        const TRect &r = app.table().layout().step_cells[5];
        const ImVec2 o = table_origin(Mode::REPLAY);
        click(ImVec2(o.x + (r.x0 + r.x1) / 2, o.y + (r.y0 + r.y1) / 2), frames);
        check(app.settings().stops.own == before.own && app.settings().stops.opponent == before.opponent,
            at + ": a click on Replay's step bar toggles a stop");
        ++cov->step_checks;
    }
    move_mouse(ImVec2(-1000, -1000), frames);
}

void check_step_bar_clicks(
    const CheckFn &check, App &app, const std::function<void(int)> &frames, const std::string &settings_path) {
    frames(2);
    const TableModel *model = app.table().model();
    const TableLayout &L = app.table().layout();
    check(app.mode() == Mode::PLAY && model && L.step_cells.size() == STOP_STEP_COUNT,
        "Step bar: Play's Table shows no step bar at the human's decision");
    if (!model || L.step_cells.size() != STOP_STEP_COUNT) return;
    const bool own = model->active == model->bottom;
    const int step = 8;  // end of combat: not a default stop
    const StopSettings before = app.settings().stops;
    const ImVec2 o = table_origin(Mode::PLAY);
    const TRect r = L.step_cells[step];
    const ImVec2 dot(o.x + (r.x0 + r.x1) / 2, o.y + r.y1 - 3);
    click(dot, frames);
    StopSettings want = before;
    bool *flag = stop_flag(&want, own, step);
    *flag = !*flag;
    check(app.settings().stops.own == want.own && app.settings().stops.opponent == want.opponent,
        std::string("Step bar: a click on End of combat's dot does not toggle its stop in ") +
            (own ? "your turn" : "the opponent's turn") + " alone");
    frames(2);
    Settings saved;
    std::string err;
    check(load_settings(settings_path, &saved, &err) && saved.stops.own == want.own &&
              saved.stops.opponent == want.opponent,
        "Step bar: the toggled stop is not in settings.json " + err);
    click(ImVec2(o.x + (r.x0 + r.x1) / 2, o.y + r.y0 + 3), frames);
    check(app.settings().stops.own == before.own && app.settings().stops.opponent == before.opponent,
        "Step bar: a second click on the step does not turn the stop back");
    frames(2);
    check(load_settings(settings_path, &saved, &err) && saved.stops.own == before.own &&
              saved.stops.opponent == before.opponent,
        "Step bar: the stop turned back is not in settings.json " + err);
    std::printf("rmwork: Step bar: End of combat's stop in %s toggled by a click, saved, and back\n",
        own ? "your turn" : "the opponent's turn");
    move_mouse(ImVec2(-1000, -1000), frames);
}

void check_table_units(const CheckFn &check) {
    check_synthetic_layout(check, synthetic_board(0, 0), "an empty-ish board: ");
    check_synthetic_layout(check, synthetic_board(6, 7), "a busy board: ");
    check_synthetic_layout(check, synthetic_board(18, 12), "a crowded board: ");
    TableModel empty;
    empty.index();
    check_synthetic_layout(check, empty, "an empty table: ");
    check_crowding(check);
    check_step_cells(check);
    check_stop_flags(check);
    check_prompt_line(check);
    check_combat_plan(check);
    check_magnifier_rects(check);
    check_links(check);
    check_tiles(check);
}

}  // namespace rmwork
