#include "selftest_tools.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>

#include "../../src/featurize/rmtb.h"
#include "imgui_internal.h"
#include "inspector_view.h"
#include "state_tree.h"

namespace {

// One recorded decision, as the episode file says (read again, apart from the join).
struct StepFacts {
        int seat = 0;
        std::vector<featurize::PolicyEvalRecord> policies;
};

// An expected swing.
struct Expected {
        int seat;
        size_t from, to;
        bool result;
        double dv;
};

}  // namespace

static int seat_of(uint64_t seat);
static std::string where(const std::string &at, size_t decision, int view);
static std::string field(const std::vector<rmwork::InspectorField> &fields, const std::string &name);
static std::string expected_name(const rmwork::DecisionFrame &f, uint64_t handle);
static std::set<std::string> face_names(const rmwork::DecisionFrame &f, uint64_t handle);
static void check_players(const rmwork::CheckFn &check, const rmwork::DecisionFrame &f, const rmwork::FrameModels &fm,
    const std::string &here);
static void check_zones(const rmwork::CheckFn &check, const rmwork::DecisionFrame &f, const rmwork::FrameModels &fm,
    const std::string &here, rmwork::ToolsCoverage *cov);
static void check_choices(
    const rmwork::CheckFn &check, const rmwork::ToolData &td, const std::string &here, rmwork::ToolsCoverage *cov);
static void check_inspector(
    const rmwork::CheckFn &check, const rmwork::ToolData &td, const std::string &here, rmwork::ToolsCoverage *cov);
static void check_tree_follow(
    const rmwork::CheckFn &check, const rmwork::ToolData &td, const std::string &here, rmwork::ToolsCoverage *cov);
static void check_tree(const rmwork::CheckFn &check, const rmwork::DecisionFrame &f, const rmwork::StateTreeModel &tm,
    const std::string &here, rmwork::ToolsCoverage *cov);
static void check_links(const rmwork::CheckFn &check, const rmwork::FrameModels &fm, const rmwork::StateTreeModel &tm,
    const std::string &here, rmwork::ToolsCoverage *cov);
static void key_press(ImGuiKey key, const std::function<void(int)> &frames);
static std::map<size_t, StepFacts> read_steps(const rmwork::Match &m, std::string *err);
static std::vector<Expected> expected_swings(
    const rmwork::Match &m, const std::map<size_t, StepFacts> &steps, int seat, int64_t policy, size_t n);
static std::string nats(double h);

static int seat_of(uint64_t seat) {
    return seat == 1 ? 0 : seat == 2 ? 1 : -1;
}

static std::string where(const std::string &at, size_t decision, int view) {
    return at + ": decision " + std::to_string(decision + 1) + ", " + rmwork::view_name(view) + ": ";
}

// The value of the Inspector field `name`, "<none>" when it has none.
static std::string field(const std::vector<rmwork::InspectorField> &fields, const std::string &name) {
    for (const rmwork::InspectorField &f : fields)
        if (f.name == name) return f.value;
    return "<none>";
}

// The name the frame's object table gives `handle` (its `card` column), "" when none.
static std::string expected_name(const rmwork::DecisionFrame &f, uint64_t handle) {
    const agents::Table o(*f.frame, f.stream->schema(), "object");
    for (size_t r = 0; r < o.rows(); ++r) {
        if (o.u(r, o.col("handle")) != handle) continue;
        const int card = o.card(r, o.col("card"));
        if (card < 0) return "";
        const rmwork::CardFace *face = f.book->face(card, false);
        return face ? face->name : f.stream->cards().name(card);
    }
    return "";
}

// The names an object's faces have, as the frame identifies it (its card and the card it copies,
// front and back); none when the frame does not identify it.
static std::set<std::string> face_names(const rmwork::DecisionFrame &f, uint64_t handle) {
    std::set<std::string> out;
    const agents::Table o(*f.frame, f.stream->schema(), "object");
    for (size_t r = 0; r < o.rows(); ++r) {
        if (o.u(r, o.col("handle")) != handle) continue;
        const int card = o.card(r, o.col("card")), copy = o.card(r, o.col("copy_of"));
        if (card < 0) return out;
        for (int c : {card, copy}) {
            if (c < 0) continue;
            for (bool back : {false, true}) {
                const rmwork::CardFace *face = f.book->face(c, back);
                if (face) out.insert(face->name);
            }
            out.insert(f.stream->cards().name(c));
        }
    }
    return out;
}

static void check_players(const rmwork::CheckFn &check, const rmwork::DecisionFrame &f, const rmwork::FrameModels &fm,
    const std::string &here) {
    const agents::Table p(*f.frame, f.stream->schema(), "player");
    check(p.rows() == 2, here + "the frame has " + std::to_string(p.rows()) + " player rows");
    for (size_t r = 0; r < p.rows(); ++r) {
        const int s = seat_of(p.u(r, p.col("player")));
        if (s < 0) continue;
        const rmwork::TablePlayer &pl = fm.table.players[static_cast<size_t>(s)];
        const bool same = pl.life == p.s(r, p.col("life")) && pl.hand == static_cast<int>(p.u(r, p.col("hand"))) &&
                          pl.library == static_cast<int>(p.u(r, p.col("library"))) &&
                          pl.graveyard == static_cast<int>(p.u(r, p.col("graveyard"))) &&
                          pl.exile == static_cast<int>(p.u(r, p.col("exile")));
        check(same, here + "Players: " + rmwork::seat_name(s) + "'s life or zone counts are not the frame's");
        const rmwork::ZoneModel &z = fm.zones;
        check(z.get(s, rmwork::ZoneKind::HAND).size == pl.hand &&
                  z.get(s, rmwork::ZoneKind::LIBRARY).size == pl.library &&
                  z.get(s, rmwork::ZoneKind::GRAVEYARD).size == pl.graveyard &&
                  z.get(s, rmwork::ZoneKind::EXILE).size == pl.exile,
            here + "Players: the zone icons' counts are not the player row's");
    }
}

// Each zone window lists exactly the frame's cards of the zone (and, in a seat view, the cards its
// knowledge places there), names only cards the frame gives, and counts the rest as hidden.
static void check_zones(const rmwork::CheckFn &check, const rmwork::DecisionFrame &f, const rmwork::FrameModels &fm,
    const std::string &here, rmwork::ToolsCoverage *cov) {
    using rmwork::ZoneKind;
    const agents::SeatStream &st = *f.stream;
    const agents::Table o(*f.frame, st.schema(), "object");
    const agents::Table kpos(*f.frame, st.schema(), "k_position");
    const agents::Table khand(*f.frame, st.schema(), "k_known_hand");
    const int zones[4] = {st.enum_value("zone", "HAND"), st.enum_value("zone", "LIBRARY"),
        st.enum_value("zone", "GRAVEYARD"), st.enum_value("zone", "EXILE")};
    std::set<int> given;  // every card id the frame (and its knowledge) gives
    std::map<std::pair<int, int>, std::vector<std::pair<uint64_t, uint64_t>>>
        objects;                                       // (seat, kind) -> (order, handle)
    std::map<std::pair<int, int>, int> visible_cards;  // (seat, card) -> hand objects
    for (size_t r = 0; r < o.rows(); ++r) {
        const int card = o.card(r, o.col("card")), copy = o.card(r, o.col("copy_of"));
        if (card >= 0) given.insert(card);
        if (copy >= 0) given.insert(copy);
        const int zone = static_cast<int>(o.u(r, o.col("zone")));
        const int owner = seat_of(o.u(r, o.col("owner")));
        for (int k = 0; k < 4; ++k) {
            if (zone != zones[k] || owner < 0) continue;
            const uint64_t order = k == 0 ? r : o.none(r, o.col("pos")) ? UINT64_MAX : o.u(r, o.col("pos"));
            objects[{owner, k}].emplace_back(order, o.u(r, o.col("handle")));
            if (k == 0 && card >= 0) ++visible_cards[{owner, card >= 0 && copy >= 0 ? copy : card}];
        }
    }
    for (size_t r = 0; r < kpos.rows(); ++r)
        if (kpos.card(r, kpos.col("card")) >= 0) given.insert(kpos.card(r, kpos.col("card")));
    for (size_t r = 0; r < khand.rows(); ++r) given.insert(khand.card(r, khand.col("card")));
    for (int s = 0; s < 2; ++s) {
        for (int k = 0; k < 4; ++k) {
            const rmwork::ZoneList &z = fm.zones.get(s, static_cast<ZoneKind>(k));
            const std::string w = here + "the zone window " + rmwork::seat_name(s) + " \xC2\xB7 " +
                                  rmwork::zone_title(static_cast<ZoneKind>(k)) + ": ";
            std::vector<std::pair<uint64_t, uint64_t>> exp = objects[{s, k}];
            std::stable_sort(exp.begin(), exp.end(),
                [](const std::pair<uint64_t, uint64_t> &a, const std::pair<uint64_t, uint64_t> &b) {
                    return a.first < b.first;
                });
            std::vector<uint64_t> want, got;
            for (const auto &e : exp) want.push_back(e.second);
            int known_only = 0;
            for (const rmwork::ZoneCard &c : z.cards) {
                if (c.handle) {
                    got.push_back(c.handle);
                } else {
                    known_only += c.copies;
                }
                check(c.name.empty() == (c.card < 0), w + "a card is named without its card id, or the reverse");
                if (c.card >= 0)
                    check(given.count(c.card) > 0, w + "names " + c.name + ", which the frame does not give");
                if (c.note.find("exiled with") != std::string::npos) ++cov->linked_exile;
                ++cov->zone_cards;
            }
            check(got == want, w + "its cards are not the frame's objects there, in order (" +
                                   std::to_string(got.size()) + " listed, the frame holds " +
                                   std::to_string(want.size()) + ")");
            check(z.hidden == std::max(0, z.size - z.listed()), w + "the hidden count is not the rest");
            // The omniscient viewer, a seat's own hand and the public zones: every card listed.
            const bool full = f.view == rmwork::VIEW_OMNI || k >= 2 || (k == 0 && f.view == s);
            if (full && k != 3) check(z.listed() == z.size, w + "does not list every card the viewer sees");
            if (full) check(known_only == 0, w + "lists a card its frame does not show");
            if (full) continue;
            if (k == 1) {
                // The seat's knowledge of positions: one card per k_position row not shown by the frame.
                std::set<int> shown;
                for (const rmwork::ZoneCard &c : z.cards)
                    if (c.handle) shown.insert(c.pos);
                int extra = 0;
                for (size_t r = 0; r < kpos.rows(); ++r) {
                    if (seat_of(kpos.u(r, kpos.col("owner"))) != s || kpos.card(r, kpos.col("card")) < 0) continue;
                    int pos = static_cast<int>(kpos.u(r, kpos.col("at_pos")));
                    if (st.enum_name("lib_anchor", kpos.u(r, kpos.col("at_anchor"))) == "BOTTOM")
                        pos = z.size - 1 - pos;
                    if (!shown.count(pos)) ++extra;
                }
                check(known_only == extra, w + std::to_string(known_only) +
                                               " cards known by position, its knowledge has " + std::to_string(extra));
                cov->known_library += known_only;
            } else if (k == 0) {
                // The opponent's hand: per card the more of what the frame shows and what is known.
                std::map<int, int> want_copies, got_copies;
                for (const auto &vc : visible_cards)
                    if (vc.first.first == s) want_copies[vc.first.second] = vc.second;
                for (size_t r = 0; r < khand.rows(); ++r) {
                    const int card = khand.card(r, khand.col("card"));
                    want_copies[card] = std::max(want_copies[card], static_cast<int>(khand.u(r, khand.col("copies"))));
                }
                for (const rmwork::ZoneCard &c : z.cards) got_copies[c.card] += c.copies;
                for (auto it = want_copies.begin(); it != want_copies.end();)
                    it = it->second == 0 ? want_copies.erase(it) : std::next(it);
                check(got_copies == want_copies, w + "its known cards are not the frame's and its knowledge's");
                cov->known_hand += known_only;
            } else {
                check(known_only == 0, w + "lists a card its frame does not show");
            }
        }
    }
}

static void check_choices(
    const rmwork::CheckFn &check, const rmwork::ToolData &td, const std::string &here, rmwork::ToolsCoverage *cov) {
    const rmwork::DecisionFrame &f = td.frame();
    const rmwork::MenuModel &mm = td.models().menu;
    const agents::Table menu(*f.frame, f.stream->schema(), "menu");
    const bool sees = f.view == rmwork::VIEW_OMNI || f.view == f.info->decider;
    check(mm.shown == sees, here + "Choices shows the menu in a view that does not see it, or the reverse");
    if (!sees) {
        ++cov->withheld;
        check(mm.rows.empty() && mm.chosen < 0 && menu.rows() == 0,
            here + "a non-deciding seat's view holds menu rows or a chosen row");
        check(f.visible_episode() == nullptr && rmwork::ChoicesView::shown_pi(td).empty(),
            here + "a non-deciding seat's view shows the decider's evaluations");
        return;
    }
    ++cov->menus;
    check(mm.rows.size() == menu.rows() && mm.rows.size() == static_cast<size_t>(f.info->menu_rows),
        here + "Choices has " + std::to_string(mm.rows.size()) + " rows, the menu " + std::to_string(menu.rows()));
    bool same = mm.rows.size() == td.models().table.menu.size();
    for (size_t r = 0; r < mm.rows.size() && r < menu.rows(); ++r) {
        same = same && mm.rows[r].text == menu.text(r, menu.col("description")) &&
               mm.rows[r].link.source == menu.u(r, menu.col("source")) &&
               mm.rows[r].link.target == menu.u(r, menu.col("target")) &&
               mm.rows[r].link.source == td.models().table.menu[r].source &&
               mm.rows[r].link.target == td.models().table.menu[r].target;
    }
    check(same, here + "a Choices row is not its menu row (description, source, target)");
    const int reply = f.info->reply;
    check(mm.chosen == (reply >= 0 && static_cast<size_t>(reply) < mm.rows.size() ? reply : -1),
        here + "the chosen-row marker is on row " + std::to_string(mm.chosen) + ", the log's reply " +
            std::to_string(reply));
}

static std::string nats(double h) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.3f nats", h);
    return buf;
}

static void check_inspector(
    const rmwork::CheckFn &check, const rmwork::ToolData &td, const std::string &here, rmwork::ToolsCoverage *cov) {
    const rmwork::DecisionFrame &f = td.frame();
    const std::vector<rmwork::InspectorField> fields = rmwork::inspector_fields(td);
    check(field(fields, "Decision") == std::to_string(f.index + 1) + " of " + std::to_string(f.match->decisions.size()),
        here + "the Inspector's decision is " + field(fields, "Decision"));
    check(field(fields, "Step") == f.info->phase, here + "the Inspector's step is not the decision's");
    const std::string menu = field(fields, "Menu rows");
    check(td.models().menu.shown ? menu == std::to_string(f.info->menu_rows) : menu.compare(0, 3, "\xE2\x80\x94") == 0,
        here + "the Inspector's menu size is " + menu);
    const agents::Table pending(*f.frame, f.stream->schema(), "pending");
    const agents::Table cost(*f.frame, f.stream->schema(), "pending_cost");
    check(pending.rows() ? field(fields, "Pending") != "\xE2\x80\x94" : field(fields, "Pending") == "\xE2\x80\x94",
        here + "the Inspector's pending query is not the frame's");
    check((field(fields, "Pending cost") != "<none>") == (cost.rows() > 0),
        here + "the Inspector's pending cost is not the frame's");
    if (pending.rows()) ++cov->pending;
    if (cost.rows()) ++cov->pending_costs;
    const agents::Table stack(*f.frame, f.stream->schema(), "stack");
    check(field(fields, "Stack") ==
              (stack.rows() ? std::to_string(stack.rows()) + (stack.rows() == 1 ? " object" : " objects") : "empty"),
        here + "the Inspector's stack is not the frame's");
    const rmwork::EpisodeStepInfo *e = f.visible_episode();
    bool any_v = false;
    for (const rmwork::InspectorField &fl : fields) any_v = any_v || fl.name.compare(0, 3, "V \xC2\xB7") == 0;
    if (!e) {
        check(!any_v && field(fields, "Evaluations") != "<none>",
            here + "the Inspector shows values where the viewer sees no evaluations");
        return;
    }
    for (const featurize::PolicyEvalRecord &r : e->policies) {
        const double v = td.scale_a ? rmwork::on_a_scale(e->seat, r.value) : r.value;
        const std::string got = field(fields, "V \xC2\xB7 " + rmwork::policy_label(r.id, r.version));
        check(got.compare(0, rmwork::signed_text(v, 3).size(), rmwork::signed_text(v, 3)) == 0,
            here + "the Inspector's V of policy " + std::to_string(r.id) + " is " + got);
    }
    const featurize::PolicyEvalRecord *r = rmwork::find_policy(*e, td.shown_policy());
    if (r && r->log_pi.size() == td.models().menu.rows.size()) {
        double h = 0;
        for (float l : r->log_pi) h -= std::exp(static_cast<double>(l)) * static_cast<double>(l);
        check(field(fields, "H(\xCF\x80) \xC2\xB7 p" + std::to_string(r->id)) == nats(h),
            here + "the Inspector's H(\xCF\x80) is not the policy's");
        if (e->chosen >= 0 && static_cast<size_t>(e->chosen) < r->log_pi.size()) {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%.3f", static_cast<double>(r->log_pi[static_cast<size_t>(e->chosen)]));
            check(field(fields, "log \xCF\x80(chosen)") == buf, here + "the Inspector's log \xCF\x80(chosen) is wrong");
        }
    }
    check(field(fields, "Policy version") ==
              (e->policy_version >= 0 ? std::to_string(e->policy_version) : std::string("\xE2\x80\x94")),
        here + "the Inspector's policy version is not the episode's");
    ++cov->episode_decisions;
}

// The tree lists every table of the frame with its rows, and names a handle only where the frame's
// object table identifies it.
static void check_tree(const rmwork::CheckFn &check, const rmwork::DecisionFrame &f, const rmwork::StateTreeModel &tm,
    const std::string &here, rmwork::ToolsCoverage *cov) {
    using rmwork::TreeRow;
    const TreeRow *tables = nullptr;
    for (const TreeRow &r : tm.roots())
        if (r.kind == TreeRow::Kind::TABLES) tables = &r;
    check(tables != nullptr, here + "the State Tree has no frame tables node");
    if (!tables) return;
    const std::vector<TreeRow> list = tm.children(*tables);
    check(list.size() == f.frame->tables.size(), here + "the State Tree lists " + std::to_string(list.size()) +
                                                     " tables, the frame holds " +
                                                     std::to_string(f.frame->tables.size()));
    for (size_t k = 0; k < list.size() && k < f.frame->tables.size(); ++k) {
        const rmstate::DecodedTable &t = f.frame->tables[k];
        check(list[k].name == t.def->name, here + "the State Tree's table " + list[k].name + " is not " + t.def->name);
        const size_t rows = list[k].expandable ? tm.children(list[k]).size() : 0;
        check(rows == t.rows.size(), here + "the State Tree's " + t.def->name + " has " + std::to_string(rows) +
                                         " rows, the frame " + std::to_string(t.rows.size()));
        for (size_t c = 0; c < t.def->columns.size(); ++c) {
            if (t.def->columns[c].tag != "handle") continue;
            for (size_t r = 0; r < t.rows.size(); ++r) {
                const uint64_t h = t.rows[r][c].bits;
                const std::string name = expected_name(f, h);
                const std::string want =
                    h == 0 ? "\xE2\x80\x94" : "#" + std::to_string(h) + (name.empty() ? "" : " " + name);
                const std::string got = tm.cell_text(t, r, c);
                check(got == want, here + "the State Tree shows " + t.def->name + "." + t.def->columns[c].name +
                                       " as " + got + ", the frame identifies it as " + want);
                ++cov->tree_handles;
            }
        }
    }
    // The players' zone nodes name their objects as the frame does.
    for (const TreeRow &p : tm.roots()) {
        if (p.kind != TreeRow::Kind::PLAYER) continue;
        for (const TreeRow &z : tm.children(p)) {
            if (z.kind != TreeRow::Kind::ZONE || !z.expandable) continue;
            for (const TreeRow &o : tm.children(z)) {
                const std::set<std::string> names = face_names(f, o.handle);
                bool ok = names.empty() && (o.name == "face-down card" || o.name == "unidentified card");
                for (const std::string &nm : names) ok = ok || o.name.compare(0, nm.size(), nm) == 0;
                check(ok, here + "the State Tree names object #" + std::to_string(o.handle) + " " + o.name);
            }
        }
    }
    ++cov->trees;
}

// A menu row with a source lights its objects, the object lights the row, and the State Tree's
// object row for it carries the same handle.
static void check_links(const rmwork::CheckFn &check, const rmwork::FrameModels &fm, const rmwork::StateTreeModel &tm,
    const std::string &here, rmwork::ToolsCoverage *cov) {
    using rmwork::TreeRow;
    const std::vector<rmwork::MenuLink> &menu = fm.table.menu;
    for (size_t r = 0; r < menu.size(); ++r) {
        const uint64_t src = menu[r].source;
        if (!src) continue;
        rmwork::LinkState ls;
        ls.set_menu(menu);
        ls.hover_menu_row(static_cast<int>(r));
        ls.new_frame();
        check(ls.object_linked(src) && (!menu[r].target || ls.object_linked(menu[r].target)) &&
                  ls.row_linked(static_cast<int>(r)),
            here + "hovering menu row " + std::to_string(r) + " does not light its source and target");
        ls.hover_object(src);
        ls.new_frame();
        check(ls.row_linked(static_cast<int>(r)), here + "hovering a menu row's source does not light the row");
        bool in_tree = false, in_frame = false;
        for (const TreeRow &t : tm.roots()) {
            if (t.kind != TreeRow::Kind::TABLES) continue;
            for (const TreeRow &tb : tm.children(t)) {
                if (tb.name != "object" || !tb.expandable) continue;
                for (const TreeRow &row : tm.children(tb)) {
                    in_frame = true;
                    in_tree = in_tree || row.handle == src;
                }
            }
        }
        if (in_frame && fm.table.find(src) >= 0)
            check(in_tree, here + "the State Tree has no object row for menu row " + std::to_string(r) + "'s source");
        ++cov->links;
        return;
    }
}

// A Table object or a menu row linked from elsewhere: the State Tree's scroll target is that
// object's row, shown with the nodes above it expanded; a menu row reveals its source's row.
static void check_tree_follow(
    const rmwork::CheckFn &check, const rmwork::ToolData &td, const std::string &here, rmwork::ToolsCoverage *cov) {
    using namespace rmwork;
    const TableModel &tm = td.models().table;
    StateTreeView view;
    view.refresh(td);
    const auto shown = [&](uint64_t handle, const std::string &what) {
        const std::vector<std::string> path = view.model().object_path(handle);
        if (path.empty()) return;
        check(view.scroll_target() == path.back(), here + what + " does not make its State Tree row the scroll target");
        bool found = false;
        for (const TreeRow &r : view.rows())
            if (r.key == path.back()) found = r.handle == handle;
        for (size_t k = 0; k + 1 < path.size(); ++k) found = found && view.expanded().count(path[k]) > 0;
        check(found, here + what + "'s State Tree row is not shown with its nodes expanded");
        ++cov->tree_reveals;
    };
    view.expanded().clear();
    LinkState ls;
    ls.set_menu(tm.menu);
    const size_t n = tm.objects.size();
    for (size_t k = 0; k < n; k += std::max<size_t>(1, n / 3)) {
        ls.hover_object(tm.objects[k].handle);
        ls.new_frame();
        view.follow_links(ls);
        shown(tm.objects[k].handle, "hovering a Table object");
    }
    for (size_t r = 0; r < tm.menu.size(); ++r) {
        if (!tm.menu[r].source || view.model().object_path(tm.menu[r].source).empty()) continue;
        ls.hover_menu_row(static_cast<int>(r));
        ls.new_frame();
        view.follow_links(ls);
        shown(tm.menu[r].source, "hovering menu row " + std::to_string(r));
        break;
    }
}

static void key_press(ImGuiKey key, const std::function<void(int)> &frames) {
    ImGuiIO &io = ImGui::GetIO();
    io.AddKeyEvent(key, true);
    frames(1);
    io.AddKeyEvent(key, false);
    frames(1);
}

// The match's episodes read again, by decision.
static std::map<size_t, StepFacts> read_steps(const rmwork::Match &m, std::string *err) {
    std::map<size_t, StepFacts> out;
    for (const std::string &path : m.episode_files) {
        featurize::RmtbFile file;
        featurize::Episode ep;
        if (!featurize::rmtb_read_file(path, &file, err) || !featurize::read_episode(file, &ep, err)) return {};
        for (const featurize::EpisodeStep &s : ep.steps) {
            const int at = m.find_decision(static_cast<int>(s.game), s.sideboarding, s.decision);
            if (at < 0) continue;
            StepFacts f;
            f.seat = s.seat;
            f.policies = s.policies;
            out[static_cast<size_t>(at)] = f;
        }
    }
    return out;
}

static std::vector<Expected> expected_swings(
    const rmwork::Match &m, const std::map<size_t, StepFacts> &steps, int seat, int64_t policy, size_t n) {
    std::vector<Expected> all;
    for (int s = 0; s < 2; ++s) {
        if (seat >= 0 && s != seat) continue;
        std::vector<std::pair<size_t, double>> pts;
        for (const auto &kv : steps) {
            if (kv.second.seat != s || m.decisions[kv.first].sideboarding) continue;
            for (const featurize::PolicyEvalRecord &r : kv.second.policies)
                if (r.id == policy) pts.emplace_back(kv.first, r.value);
        }
        for (size_t k = 0; k < pts.size(); ++k) {
            const int game = m.decisions[pts[k].first].game;
            if (k + 1 < pts.size() && m.decisions[pts[k + 1].first].game == game) {
                all.push_back({s, pts[k].first, pts[k + 1].first, false, pts[k + 1].second - pts[k].second});
            } else if (static_cast<size_t>(game) < m.result.game_winners.size()) {
                const int w = m.result.game_winners[static_cast<size_t>(game)];
                const double z = w < 0 ? 0 : w == s ? 1 : -1;
                all.push_back({s, pts[k].first, 0, true, z - pts[k].second});
            }
        }
    }
    std::stable_sort(all.begin(), all.end(), [](const Expected &a, const Expected &b) {
        if (std::fabs(a.dv) != std::fabs(b.dv)) return std::fabs(a.dv) > std::fabs(b.dv);
        return a.from < b.from;
    });
    if (all.size() > n) all.resize(n);
    return all;
}

namespace rmwork {

void check_tools_frames(const CheckFn &check, App &app, const std::string &at, ToolsCoverage *cov) {
    ReplaySession &s = app.replay();
    if (!s.loaded()) return;
    const size_t n = s.count();
    ToolData td;
    for (size_t i = 0; i < n; ++i) {
        for (int v = 0; v < VIEW_COUNT; ++v) {
            s.go_to(i);
            s.set_view(v);
            const std::string here = where(at, i, v);
            if (!td.refresh(s, app.table())) {
                check(false, here + "the tools cannot read the frame: " + td.error());
                continue;
            }
            const DecisionFrame &f = td.frame();
            const FrameModels &fm = td.models();
            check_players(check, f, fm, here);
            check_zones(check, f, fm, here, cov);
            check_choices(check, td, here, cov);
            check_inspector(check, td, here, cov);
            StateTreeModel tm;
            tm.build(f, fm.table);
            if (i % 4 == 0 || i + 1 == n) {
                check_tree(check, f, tm, here, cov);
                check_tree_follow(check, td, here, cov);
            }
            if (fm.menu.shown) check_links(check, fm, tm, here, cov);
            ++cov->frames;
        }
    }
    s.go_to(0);
    s.set_view(VIEW_A);
}

void check_tools_drawing(
    const CheckFn &check, App &app, const std::function<void(int)> &frames, const std::string &at, ToolsCoverage *cov) {
    ReplaySession &s = app.replay();
    if (!s.loaded()) return;
    for (int t = 0; t < TOOL_COUNT; ++t) app.set_tool_open(Mode::REPLAY, static_cast<Tool>(t), true);
    const size_t n = s.count();
    const size_t step = std::max<size_t>(1, n / 8);
    for (size_t i = 0; i < n; i += step)
        for (int v = 0; v < VIEW_COUNT; ++v) {
            s.go_to(i);
            s.set_view(v);
            frames(1);
            check(app.tools().data().ok(), where(at, i, v) + "the tools draw no frame");
            ++cov->drawn;
        }
    // A Choices row hovered lights its source on the Table; in a non-deciding seat's view Choices
    // links nothing.
    app.focus_tool(Tool::CHOICES);
    for (size_t i = 0; i < n; ++i) {
        const int decider = s.match()->decisions[i].decider;
        s.go_to(i);
        s.set_view(decider);
        frames(2);
        const TableModel *model = app.table().model();
        if (!model) continue;
        int row = -1;
        for (size_t r = 0; r < model->menu.size() && row < 0; ++r)
            if (model->menu[r].source && model->find(model->menu[r].source) >= 0) row = static_cast<int>(r);
        if (row < 0) continue;
        app.tools().choices().force_hover(row);
        frames(2);
        const uint64_t src = model->menu[static_cast<size_t>(row)].source;
        check(app.links().hovered().kind == LinkItem::Kind::MENU_ROW && app.links().hovered().row == row &&
                  app.links().object_linked(src) && app.table().model()->find(src) >= 0,
            where(at, i, decider) + "hovering a Choices row does not light its source on the Table");
        s.set_view(1 - decider);
        frames(2);
        check(app.links().hovered().kind != LinkItem::Kind::MENU_ROW,
            where(at, i, 1 - decider) + "Choices links a row in a non-deciding seat's view");
        app.tools().choices().force_hover(-1);
        frames(1);
        break;
    }
    // A card hovered on the Table, with the State Tree scrolled to its end: the tree scrolls its row
    // into view once, then leaves the scrolling alone while the hover stays.
    app.focus_tool(Tool::TREE);
    StateTreeView &tree = app.tools().tree();
    for (size_t i = n; i-- > 0;) {
        s.go_to(i);
        s.set_view(VIEW_OMNI);
        frames(2);
        const TableModel *model = app.table().model();
        if (!model) continue;
        std::string name;
        for (const TableObject &o : model->objects)
            if (!o.name.empty() && !tree.model().object_path(o.handle).empty()) name = o.name;
        if (name.empty()) continue;
        tree.expand_overview();
        tree.force_scroll_to_end();
        frames(2);
        app.table().force_card_hover(name);
        frames(3);
        const LinkItem &h = app.links().hovered();
        const std::vector<std::string> path =
            h.kind == LinkItem::Kind::OBJECT ? tree.model().object_path(h.handle) : std::vector<std::string>();
        check(!path.empty() && tree.revealed() == path.back() && tree.revealed_on_screen() &&
                  tree.scroll_target().empty(),
            where(at, i, VIEW_OMNI) + "hovering " + name +
                " on the Table does not scroll its State Tree row into view");
        tree.force_scroll_to_end();
        frames(2);
        check(!tree.revealed_on_screen() || tree.rows().size() < 40,
            where(at, i, VIEW_OMNI) + "the State Tree scrolls back to a hover that did not change");
        app.table().force_card_hover("");
        frames(1);
        ++cov->tree_scrolls;
        break;
    }

    // Zone windows at the last decision: seat A's graveyard and B's hand, in A's view.
    PlayersView &p = app.tools().players();
    s.go_to(n - 1);
    s.set_view(VIEW_A);
    p.open_zone(0, ZoneKind::GRAVEYARD);
    p.open_zone(1, ZoneKind::HAND);
    frames(2);
    const std::string gy = PlayersView::zone_window_title(0, ZoneKind::GRAVEYARD) + "###zone.0.graveyard";
    const ImGuiWindow *w = ImGui::FindWindowByName(gy.c_str());
    check(w && w->Active && p.open_zones() == 2, at + ": the zone windows do not open");
    const ZoneList &z = app.tools().data().models().zones.get(0, ZoneKind::GRAVEYARD);
    if (!z.cards.empty() && !z.cards[0].name.empty()) {
        p.force_magnify(z.cards[0].name);
        frames(2);
        check(p.magnified() != 0, at + ": the zone window's magnifier does not open");
        p.force_magnify("");
        // A right-click on the first card magnifies it; Esc closes that, then the window.
        ImVec2 a, b;
        check(p.card_rect(0, ZoneKind::GRAVEYARD, 0, &a, &b), at + ": the zone window drew no card");
        ImGuiIO &io = ImGui::GetIO();
        io.AddMousePosEvent((a.x + b.x) / 2, (a.y + b.y) / 2);
        frames(2);
        io.AddMouseButtonEvent(ImGuiMouseButton_Right, true);
        frames(1);
        io.AddMouseButtonEvent(ImGuiMouseButton_Right, false);
        frames(1);
        check(p.magnified() != 0, at + ": a right-click does not magnify a zone window's card");
        // The right-click focused the window.
        key_press(ImGuiKey_Escape, frames);
        check(p.magnified() == 0 && p.zone_open(0, ZoneKind::GRAVEYARD),
            at + ": Esc does not close the zone window's magnifier first (magnified " + std::to_string(p.magnified()) +
                ", window " + (p.zone_open(0, ZoneKind::GRAVEYARD) ? "open" : "closed") + ")");
    }
    // A click on its title bar focuses the window; Esc closes it and leaves the other open.
    if (const ImGuiWindow *gw = ImGui::FindWindowByName(gy.c_str())) {
        ImGuiIO &io = ImGui::GetIO();
        io.AddMousePosEvent(gw->Pos.x + 24, gw->Pos.y + gw->TitleBarHeight / 2);
        frames(1);
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
        frames(1);
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
        frames(1);
    }
    key_press(ImGuiKey_Escape, frames);
    check(!p.zone_open(0, ZoneKind::GRAVEYARD) && p.zone_open(1, ZoneKind::HAND),
        at + ": Esc does not close the focused zone window alone (graveyard " +
            (p.zone_open(0, ZoneKind::GRAVEYARD) ? "open" : "closed") + ", hand " +
            (p.zone_open(1, ZoneKind::HAND) ? "open" : "closed") + ")");
    p.close_zone(1, ZoneKind::HAND);
    ImGui::GetIO().AddMousePosEvent(-1000, -1000);
    frames(1);
    s.go_to(0);
    s.set_view(VIEW_A);
}

void check_tools_values(const CheckFn &check, App &app, const std::function<void(int)> &frames, const std::string &at,
    const ValuesExpect &want_values, ToolsCoverage *cov) {
    ReplaySession &s = app.replay();
    if (!s.loaded()) return;
    const Match &m = *s.match();
    std::string err;
    const std::map<size_t, StepFacts> steps = read_steps(m, &err);
    check(!steps.empty(), at + ": the episodes do not read: " + err);
    if (steps.empty()) return;
    ToolData td;
    s.go_to(0);
    s.set_view(VIEW_OMNI);
    td.refresh(s, app.table());
    const MatchValues &mv = td.values();
    check(mv.policies.size() == want_values.policies, at + ": " + std::to_string(mv.policies.size()) +
                                                          " policies evaluated, not " +
                                                          std::to_string(want_values.policies));
    if (want_values.every_decision) {
        // Every decision evaluated by every policy, each π a distribution over its menu.
        size_t missing = 0, not_distribution = 0;
        for (size_t i = 0; i < m.decisions.size(); ++i) {
            auto it = steps.find(i);
            if (it == steps.end() || it->second.policies.size() != want_values.policies) {
                ++missing;
                continue;
            }
            for (const featurize::PolicyEvalRecord &r : it->second.policies) {
                double sum = 0;
                for (float l : r.log_pi) sum += std::exp(static_cast<double>(l));
                if (static_cast<int>(r.log_pi.size()) != m.decisions[i].menu_rows || std::fabs(sum - 1) > 1e-4)
                    ++not_distribution;
            }
        }
        check(missing == 0, at + ": " + std::to_string(missing) + " decisions without every policy's evaluation");
        check(not_distribution == 0, at + ": " + std::to_string(not_distribution) +
                                         " evaluations whose \xCF\x80 is not a distribution over the menu");
    }
    // The Value Graph: every view's series are its seats', each the episodes' values on A's scale.
    for (int v = 0; v < VIEW_COUNT; ++v) {
        const std::vector<GraphSeries> series = ValueGraphView::series_for(td, v);
        std::set<int> seats;
        for (const GraphSeries &g : series) seats.insert(g.seat);
        check(v == VIEW_OMNI ? seats == std::set<int>{0, 1} : seats == std::set<int>{v},
            at + ": the Value Graph in the " + view_name(v) + " view shows another seat's values");
    }
    size_t points = 0;
    for (int seat = 0; seat < 2; ++seat)
        for (int64_t pol : mv.policies) {
            std::vector<std::pair<size_t, double>> want;
            for (const auto &kv : steps) {
                if (kv.second.seat != seat) continue;
                for (const featurize::PolicyEvalRecord &r : kv.second.policies)
                    if (r.id == pol) want.emplace_back(kv.first, seat == 1 ? -r.value : r.value);
            }
            const std::vector<std::pair<size_t, double>> got = ValueGraphView::points(td, seat, pol);
            if (want_values.every_decision) {
                size_t decided = 0;
                for (const DecisionInfo &d : m.decisions) decided += d.decider == seat ? 1 : 0;
                check(got.size() == decided, at + ": the Value Graph's " + seat_name(seat) + " series has " +
                                                 std::to_string(got.size()) + " points for " + std::to_string(decided) +
                                                 " decisions");
            }
            bool same = got.size() == want.size();
            for (size_t k = 0; same && k < got.size(); ++k)
                same = got[k].first == want[k].first && std::fabs(got[k].second - want[k].second) < 1e-6;
            check(same, at + ": the Value Graph's " + seat_name(seat) + " series of policy " + std::to_string(pol) +
                            " is not the episodes' values");
            points += got.size();
        }
    check(points > 0, at + ": the Value Graph has no points");
    // Swings, per view, recomputed.
    for (int v = 0; v < VIEW_COUNT; ++v) {
        s.set_view(v);
        td.refresh(s, app.table());
        const std::vector<Swing> got = td.swings(v, SwingsView::ROWS);
        const std::vector<Expected> want =
            expected_swings(m, steps, v == VIEW_OMNI ? -1 : v, td.shown_policy(), SwingsView::ROWS);
        bool same = got.size() == want.size() && !got.empty();
        for (size_t k = 0; same && k < got.size(); ++k)
            same = got[k].seat == want[k].seat && got[k].from == want[k].from && got[k].to_result == want[k].result &&
                   (got[k].to_result || got[k].to == want[k].to) && std::fabs(got[k].dv - want[k].dv) < 1e-6;
        check(same, at + ": the Swings in the " + std::string(view_name(v)) + " view are not the recomputed ones");
    }
    // Seat disagreement: adjacent decisions of the two seats in one game.
    {
        const std::vector<Disagreement> gaps = top_disagreements(m, mv, mv.policies.front(), 1000);
        size_t want = 0;
        for (size_t i = 0; i + 1 < m.decisions.size(); ++i) {
            auto a = steps.find(i), b = steps.find(i + 1);
            if (a == steps.end() || b == steps.end() || a->second.seat == b->second.seat) continue;
            if (m.decisions[i].game != m.decisions[i + 1].game || m.decisions[i].sideboarding ||
                m.decisions[i + 1].sideboarding)
                continue;
            ++want;
        }
        bool sorted = true;
        for (size_t k = 1; k < gaps.size(); ++k) sorted = sorted && gaps[k - 1].gap >= gaps[k].gap;
        check(gaps.size() == want && want > 0 && sorted, at + ": the seat disagreements are not every adjacent pair");
    }
    // The toolbar's next swing, from the first decision, through the omniscient view's swings.
    s.set_view(VIEW_OMNI);
    s.go_to(0);
    frames(1);
    const std::vector<Expected> all = expected_swings(m, steps, -1, td.shown_policy(), SwingsView::ROWS);
    int jumps = 0;
    for (int guard = 0; guard < 40; ++guard) {
        long want = -1;
        for (const Expected &e : all) {
            const size_t t = e.result ? e.from : e.to;
            if (t > s.cursor() && (want < 0 || t < static_cast<size_t>(want))) want = static_cast<long>(t);
        }
        const long got = app.tools().next_swing_target(s);
        check(got == want, at + ": next swing goes to " + std::to_string(got) + ", not " + std::to_string(want));
        if (got < 0 || got != want) break;
        check(
            app.next_swing() && s.cursor() == static_cast<size_t>(want), at + ": next swing does not move the cursor");
        frames(1);
        ++jumps;
    }
    check(jumps > 0, at + ": next swing never moves");
    // Per decision: the Inspector's values on both scales, Choices' π, the Model's Read, the status bar.
    for (size_t i = 0; i < m.decisions.size(); ++i) {
        auto it = steps.find(i);
        if (it == steps.end()) continue;
        s.go_to(i);
        s.set_view(VIEW_OMNI);
        td.policy = -1;
        td.scale_a = false;
        td.refresh(s, app.table());
        check_inspector(check, td, where(at, i, VIEW_OMNI), cov);
        td.scale_a = true;
        check_inspector(check, td, where(at, i, VIEW_OMNI) + "(A's scale) ", cov);
        td.scale_a = false;
        const StepFacts &f = it->second;
        std::vector<int64_t> shown = {-1};
        shown.insert(shown.end(), mv.policies.begin(), mv.policies.end());
        for (int64_t pol : shown) {
            td.policy = pol;
            const featurize::PolicyEvalRecord *want = nullptr;
            for (const featurize::PolicyEvalRecord &r : f.policies)
                if (r.id == pol) want = &r;
            if (!want && !f.policies.empty()) want = &f.policies.front();
            const std::vector<double> pi = ChoicesView::shown_pi(td);
            bool same = want && pi.size() == want->log_pi.size();
            for (size_t k = 0; same && k < pi.size(); ++k)
                same = std::fabs(pi[k] - std::exp(static_cast<double>(want->log_pi[k]))) < 1e-9;
            check(same, where(at, i, VIEW_OMNI) + "Choices' \xCF\x80 is not policy " + std::to_string(pol) + "'s");
        }
        td.policy = -1;
        size_t last = 0;
        const std::vector<double> hist = ModelReadView::history(td, &last);
        size_t want_n = 0;
        for (const auto &kv : steps)
            if (kv.first <= i && kv.second.seat == f.seat) ++want_n;
        check(hist.size() == want_n && last == i && !hist.empty() &&
                  std::fabs(hist.back() - static_cast<double>(f.policies.front().value)) < 1e-6,
            where(at, i, VIEW_OMNI) + "the Model's Read history is not the deciding seat's values so far");
        if (i % 9 == 0) {
            frames(1);
            check(app.tools().status_value() ==
                      "V " + signed_text(f.policies.front().value) + " (" + (f.seat == 0 ? "A" : "B") + ")",
                where(at, i, VIEW_OMNI) + "the status bar's V is " + app.tools().status_value());
            s.set_view(1 - f.seat);
            frames(1);
            check(app.tools().status_value() == "V \xE2\x80\x94",
                where(at, i, 1 - f.seat) + "the status bar shows the decider's V in the other seat's view");
        }
    }
    check(m.agents == want_values.agents,
        at + ": the seats' agents are not the episodes' (" + m.agents[0] + ", " + m.agents[1] + ")");
    s.go_to(0);
    s.set_view(VIEW_A);
}

void check_tools_units(const CheckFn &check) {
    // A synthetic bo3-like match: game 1 (decisions 0-5, B wins), game 2 (6-8, A wins).
    Match m;
    const int deciders[9] = {0, 1, 0, 1, 0, 1, 1, 0, 1};
    const float values[9] = {0.1f, -0.2f, 0.4f, 0.3f, -0.5f, 0.9f, 0.0f, 0.2f, -0.6f};
    const int turns[9] = {0, 0, 1, 1, 2, 2, 0, 0, 1};
    for (int i = 0; i < 9; ++i) {
        DecisionInfo d;
        d.game = i < 6 ? 0 : 1;
        d.decider = deciders[i];
        d.turn = turns[i];
        d.phase = "First main";
        d.episode_step = i;
        m.decisions.push_back(d);
        EpisodeStepInfo e;
        e.seat = deciders[i];
        featurize::PolicyEvalRecord r;
        r.id = 0;
        r.version = 7;
        r.value = values[i];
        r.log_pi = {std::log(0.25f), std::log(0.75f)};
        e.policies.push_back(r);
        m.episode_steps.push_back(e);
    }
    GameSpan g1, g2;
    g1.game = 0;
    g1.first = 0;
    g1.count = 6;
    g2.game = 1;
    g2.first = 6;
    g2.count = 3;
    m.games = {g1, g2};
    m.result.game_winners = {1, 0};
    m.logs[VIEW_OMNI] = {{"x", 1, LogLine::Kind::EVENT, -1}, {"y", 2, LogLine::Kind::EVENT, -1},
        {"z", 2, LogLine::Kind::CHOICE, 1}, {"Turn 2", 3, LogLine::Kind::HEADER, -1},
        {"w", 3, LogLine::Kind::EVENT, -1}};
    const MatchValues v = match_values(m);
    check(v.series.size() == 2 && v.policies == std::vector<int64_t>{0} && v.find(0, 0) && v.find(1, 0) &&
              v.find(0, 0)->points.size() == 4 && v.find(1, 0)->points.size() == 5,
        "evaluations: the series are not per seat and policy");
    // Seat A: 0.1 → 0.4 → -0.5 → z -1 (game 1 lost); 0.2 → z +1 (game 2).
    const std::vector<Swing> a = top_swings(m, v, 0, 0, 10);
    check(a.size() == 4 && a[0].from == 2 && a[0].to == 4 && !a[0].to_result && std::fabs(a[0].dv + 0.9) < 1e-6 &&
              a[1].from == 7 && a[1].to_result && std::fabs(a[1].dv - 0.8) < 1e-6,
        "evaluations: seat A's swings are not the largest changes in order");
    check(next_swing_target(a, 0) == 2 && next_swing_target(a, 4) == 7 && next_swing_target(a, 7) == -1,
        "evaluations: the next swing target");
    const std::vector<Disagreement> d = top_disagreements(m, v, 0, 10);
    // Adjacent pairs of different seats in one game: (0,1) (1,2) (2,3) (3,4) (4,5) (7,8) (6,7).
    check(d.size() == 7 && d[0].gap >= d.back().gap, "evaluations: the seat disagreements");
    const std::vector<TurnBand> bands = turn_bands(m);
    check(bands.size() == 5 && bands[0].label == "T1" && bands[3].first == 6, "evaluations: the turn bands");
    check(signed_text(0.4) == "+0.40" &&
              signed_text(-0.4) ==
                  "\xE2\x88\x92"
                  "0.40" &&
              signed_text(0) == "0.00" && signed_text(1, 0) == "+1",
        "evaluations: signed value text");
    check(std::fabs(entropy(policy_pi(m.episode_steps[0].policies[0])) -
                    -(0.25 * std::log(0.25) + 0.75 * std::log(0.75))) < 1e-6,
        "evaluations: H(\xCF\x80)");
    check(between_text(m, VIEW_OMNI, 0, false, 2, 5) == "x \xC2\xB7 y" &&
              between_text(m, VIEW_OMNI, 1, true, 0, 5) == "y \xC2\xB7 z \xC2\xB7 w",
        "evaluations: what happened between two decisions (" + between_text(m, VIEW_OMNI, 1, true, 0, 5) + ")");
    check(game_result(m, 0, 0) == -1 && game_result(m, 1, 0) == 1 && game_result(m, 2, 0) == 0,
        "evaluations: the games' results");
}

}  // namespace rmwork
