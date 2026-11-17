#include "selftest_table.h"

#include <algorithm>
#include <map>
#include <set>

#include "imgui_internal.h"
#include "table_backdrop.h"
#include "table_layout.h"
#include "table_model.h"

namespace {

// The panel sizes the fit is checked at, and the text sizes of a 14 px UI font.
const float SIZES[][2] = {{640, 480}, {900, 700}, {1600, 1000}};
const rmwork::TableMetrics METRICS = {17, 420, 430, 19};

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
    const std::set<uint64_t> &battlefield, const std::set<uint64_t> &hand, const std::string &at);
static void check_objects(const rmwork::CheckFn &check, const agents::SeatStream &st, const rmstate::DecodedFrame &f,
    const rmwork::TableModel &model, int view, const std::string &at, rmwork::TableCoverage *cov);
static void check_stack(const rmwork::CheckFn &check, const agents::SeatStream &st, const rmstate::DecodedFrame &f,
    const rmwork::TableModel &model, const std::string &at, rmwork::TableCoverage *cov);
static rmwork::TableObject synthetic(uint64_t handle, int controller, int card, bool land);
static rmwork::TableModel synthetic_board(int extra_per_row, int hand);
static void check_synthetic_layout(const rmwork::CheckFn &check, const rmwork::TableModel &m, const std::string &at);
static void check_links(const rmwork::CheckFn &check);
static void check_tiles(const rmwork::CheckFn &check);
static ImVec2 table_origin();
static void move_mouse(ImVec2 p, const std::function<void(int)> &frames);

static int seat_of(uint64_t seat) {
    return seat == 1 ? 0 : seat == 2 ? 1 : -1;
}

static std::string where(const std::string &at, size_t decision, int view) {
    return at + ": decision " + std::to_string(decision + 1) + ", " + rmwork::view_name(view) + ": ";
}

// At each panel size: inside the panel, every battlefield and hand object placed once, each
// attachment in its host's pile, each land stack's count its pile's size.
static void check_layout(const rmwork::CheckFn &check, const rmwork::TableModel &model,
    const std::set<uint64_t> &battlefield, const std::set<uint64_t> &hand, const std::string &at) {
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
        check(stacks, here + "a land stack's count is not its pile's size");
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
    for (const auto &row : model.rows)
        for (const TablePile &p : row) {
            if (p.land_stack) ++cov->land_stacks;
            if (!p.land_stack) cov->attachments += static_cast<int>(p.objects.size()) - 1;
        }
    check_layout(check, model, battlefield, hand, at);
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

// A board: per row a land stack of three, a tapped land, a creature wearing two attachments,
// an attacking creature and `extra_per_row` more; `hand` cards in the bottom hand.
static rmwork::TableModel synthetic_board(int extra_per_row, int hand) {
    using namespace rmwork;
    TableModel m;
    uint64_t h = 1;
    for (int seat = 0; seat < 2; ++seat) {
        const bool top = seat == m.top;
        std::vector<TablePile> &lands = m.rows[top ? ROW_TOP_LANDS : ROW_BOTTOM_LANDS];
        std::vector<TablePile> &other = m.rows[top ? ROW_TOP_OTHER : ROW_BOTTOM_OTHER];
        TablePile stack;
        stack.land_stack = true;
        for (int k = 0; k < 3; ++k) {
            m.objects.push_back(synthetic(h++, seat, 1, true));
            stack.objects.push_back(static_cast<int>(m.objects.size()) - 1);
        }
        lands.push_back(stack);
        m.objects.push_back(synthetic(h++, seat, 2, true));
        m.objects.back().tapped = true;
        lands.push_back({{static_cast<int>(m.objects.size()) - 1}, false});
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
        other.push_back({{static_cast<int>(m.objects.size()) - 1}, false});
        for (int k = 0; k < extra_per_row; ++k) {
            m.objects.push_back(synthetic(h++, seat, 6 + k, k % 2 == 0));
            (k % 2 == 0 ? lands : other).push_back({{static_cast<int>(m.objects.size()) - 1}, false});
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
    check_layout(check, m, battlefield, hand, at);
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
    for (const PlacedCard &c : big.cards)
        if (c.stack_count) check(c.stack_count == 3 && c.front, at + "the land stack's count is not 3");
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

// The Table window's inner top-left corner.
static ImVec2 table_origin() {
    const ImGuiWindow *w =
        ImGui::FindWindowByName(rmwork::tool_window_name(rmwork::Mode::REPLAY, rmwork::Tool::TABLE).c_str());
    return w ? w->InnerRect.Min : ImVec2(0, 0);
}

static void move_mouse(ImVec2 p, const std::function<void(int)> &frames) {
    ImGui::GetIO().AddMousePosEvent(p.x, p.y);
    frames(2);
}

namespace rmwork {

void check_table_frames(
    const CheckFn &check, TableView &view, const Match &m, const std::string &at, TableCoverage *cov) {
    for (size_t i = 0; i <= m.decisions.size(); ++i) {
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
        const ImVec2 o = table_origin();
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
        const ImVec2 o = table_origin();
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
    move_mouse(ImVec2(-1000, -1000), frames);
}

void check_table_units(const CheckFn &check) {
    check_synthetic_layout(check, synthetic_board(0, 0), "an empty-ish board: ");
    check_synthetic_layout(check, synthetic_board(6, 7), "a busy board: ");
    check_synthetic_layout(check, synthetic_board(18, 12), "a crowded board: ");
    TableModel empty;
    empty.index();
    check_synthetic_layout(check, empty, "an empty table: ");
    check_links(check);
    check_tiles(check);
}

}  // namespace rmwork
