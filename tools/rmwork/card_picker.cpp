#include "card_picker.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <set>

namespace {

// How far back the pile's history looks.
const size_t MAX_PICK_HISTORY = 64;

// A pick made earlier in the same resolution.
struct PastPick {
        std::string category;
        int64_t ordinal = -1;  // the depth a TOP_LIBRARY row filled, -1 on top of the last
        rmwork::PileCard card;
};

// Where a card the menu names lies, as the viewer's zone model has it.
struct Spot {
        int seat = 0;
        rmwork::ZoneKind zone = rmwork::ZoneKind::LIBRARY;
        const rmwork::ZoneCard *card = nullptr;
};

}  // namespace

static bool pick_category(const std::string &c);
static bool table_category(const std::string &c);
static int object_row(const agents::DecisionView &v, uint64_t handle);
static bool on_stack(const agents::DecisionView &v, uint64_t handle);
static uint64_t pending_source(const agents::DecisionView &v);
static int menu_depth(const agents::DecisionView &v);
static bool has_category(const agents::DecisionView &v, const char *category);
static std::string lower(std::string s);
static std::string zone_word(const agents::DecisionView &v, uint64_t handle);
static rmwork::PileCard pile_card(const agents::DecisionView &v, size_t row, const rmwork::CardBook &book);
static std::vector<PastPick> past_picks(const rmwork::Match &m, size_t index, int view, uint64_t source,
    const agents::DecisionView &now, const rmwork::CardBook &book);
static std::string row_action(const agents::DecisionView &v, size_t row);
static bool on_table(const agents::DecisionView &v, uint64_t handle, int bottom);
static std::string group_title(int seat, rmwork::ZoneKind z, int you);
static int group_rank(const rmwork::CardPickGroup &g, int bottom);
static rmwork::CardPickEntry entry_of(
    int seat, rmwork::ZoneKind z, const rmwork::ZoneCard &c, const rmwork::CardBook &b);
static void build_pile(const agents::DecisionView &v, const std::vector<PastPick> &past, rmwork::PickerPile *pile);

static bool pick_category(const std::string &c) {
    return c == "SEARCH_LIBRARY" || c == "TOP_LIBRARY" || c == "BOTTOM_DECK_CARD" || c == "DIG_CHOICE" ||
           c == "CHOOSE_CARD" || c == "EXILE_FROM_YARD" || c == "SYLVAN_CHOICE" || table_category(c);
}

// A pick whose cards are mostly on the Table (a discard from the hand, a permanent sacrificed or returned,
// the legend kept): it builds no pile.
static bool table_category(const std::string &c) {
    return c == "DISCARD" || c == "SACRIFICE_PERMANENT" || c == "RETURN_PERMANENT" || c == "KEEP_LEGEND";
}

// The object table's row of `handle`, -1 when the frame lists no such object.
static int object_row(const agents::DecisionView &v, uint64_t handle) {
    const agents::Table &o = v.objects();
    const int c = o.col("handle");
    for (size_t r = 0; r < o.rows(); ++r)
        if (o.u(r, c) == handle) return static_cast<int>(r);
    return -1;
}

static bool on_stack(const agents::DecisionView &v, uint64_t handle) {
    const agents::Table &s = v.stack();
    const int c = s.col("handle");
    for (size_t r = 0; r < s.rows(); ++r)
        if (s.u(r, c) == handle) return true;
    return false;
}

static uint64_t pending_source(const agents::DecisionView &v) {
    const agents::Table p = v.table("pending");
    return p.rows() ? p.u(0, p.col("source")) : 0;
}

// The depth the menu's TOP_LIBRARY rows fill (their `ordinal`); -1 when they put a card on top of the
// last, -2 when the menu has none.
static int menu_depth(const agents::DecisionView &v) {
    const agents::Table menu = v.table("menu");
    const int c_ord = menu.col("ordinal");
    for (size_t i = 0; i < v.n(); ++i)
        if (v.category(i) == "TOP_LIBRARY") return static_cast<int>(std::max<int64_t>(-1, menu.s(i, c_ord)));
    return -2;
}

static bool has_category(const agents::DecisionView &v, const char *category) {
    for (size_t i = 0; i < v.n(); ++i)
        if (v.category(i) == category) return true;
    return false;
}

static std::string lower(std::string s) {
    for (char &ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

// The zone object `handle` is in, as a word ("library", "graveyard"); "" when the frame lists no such object.
static std::string zone_word(const agents::DecisionView &v, uint64_t handle) {
    const int row = object_row(v, handle);
    if (row < 0) return "";
    const agents::Table &o = v.objects();
    return lower(v.stream().enum_name("zone", o.u(static_cast<size_t>(row), o.col("zone"))));
}

// The card menu row `row` names, as the pile shows it.
static rmwork::PileCard pile_card(const agents::DecisionView &v, size_t row, const rmwork::CardBook &book) {
    const agents::Table menu = v.table("menu");
    rmwork::PileCard c;
    const int col = menu.col("source_card");
    c.card = menu.none(row, col) ? -1 : menu.card(row, col);
    const uint64_t h = menu.u(row, menu.col("source"));
    const int obj = object_row(v, h);
    if (obj >= 0) {
        const agents::Table &o = v.objects();
        const int c_tr = o.col("transformed");
        c.back = c_tr >= 0 && o.u(static_cast<size_t>(obj), c_tr) != 0;
    }
    const rmwork::CardFace *f = c.card >= 0 ? book.face(c.card, c.back) : nullptr;
    c.name = f ? f->name : v.description(row);
    c.from = zone_word(v, h);
    return c;
}

// The picks made at the deciding seat's decisions just before `index` for the same source, oldest first:
// each reply of a card-picker category naming a card, while the menus are of the kind `now` is (TOP_LIBRARY
// rows filling given depths, putting cards on top, or none).
static std::vector<PastPick> past_picks(const rmwork::Match &m, size_t index, int view, uint64_t source,
    const agents::DecisionView &now, const rmwork::CardBook &book) {
    std::vector<PastPick> out;
    if (index >= m.decisions.size() || source == 0) return out;
    const rmwork::DecisionInfo &d0 = m.decisions[index];
    const int depth_now = menu_depth(now);
    for (size_t back = 1; back <= std::min(index, MAX_PICK_HISTORY); ++back) {
        const size_t i = index - back;
        const rmwork::DecisionInfo &d = m.decisions[i];
        if (d.decider != d0.decider || d.game != d0.game || d.sideboarding != d0.sideboarding || d.reply < 0) break;
        std::string err;
        const auto frame = m.frame(i, view, &err);
        if (!frame) break;
        const agents::DecisionView v(m.stream(view), *frame, d.decider);
        if (pending_source(v) != source || static_cast<size_t>(d.reply) >= v.n()) break;
        const size_t r = static_cast<size_t>(d.reply);
        const std::string &category = v.category(r);
        const agents::Table menu = v.table("menu");
        if (!pick_category(category) || menu.u(r, menu.col("source")) == 0) break;
        const int depth = menu_depth(v);
        if ((depth == -2) != (depth_now == -2) || ((depth >= 0) != (depth_now >= 0))) break;
        PastPick p;
        p.category = category;
        p.ordinal = category == "TOP_LIBRARY" ? std::max<int64_t>(-1, menu.s(r, menu.col("ordinal"))) : -1;
        p.card = pile_card(v, r, book);
        out.push_back(p);
    }
    std::reverse(out.begin(), out.end());
    return out;
}

// Where row `row` puts its card, when the menu offers its card several places: "Top" (or the depth it
// fills), "Bottom", "Graveyard"; "" for a plain pick.
static std::string row_action(const agents::DecisionView &v, size_t row) {
    const std::string &c = v.category(row);
    const bool split =
        has_category(v, "TOP_LIBRARY") && (has_category(v, "BOTTOM_DECK_CARD") || has_category(v, "CHOOSE_CARD"));
    if (!split) return "";
    if (c == "TOP_LIBRARY") {
        const agents::Table menu = v.table("menu");
        const int64_t d = menu.s(row, menu.col("ordinal"));
        return d > 0 ? rmwork::depth_label(static_cast<int>(d)) + " from the top" : "Top";
    }
    if (c == "BOTTOM_DECK_CARD") return "Bottom";
    if (c == "CHOOSE_CARD") return "Graveyard";
    return "";
}

// Whether the Table shows object `handle`: on the battlefield, on the stack, in the bottom seat's hand.
static bool on_table(const agents::DecisionView &v, uint64_t handle, int bottom) {
    if (on_stack(v, handle)) return true;
    const int row = object_row(v, handle);
    if (row < 0) return false;
    const agents::Table &o = v.objects();
    const std::string zone = v.stream().enum_name("zone", o.u(static_cast<size_t>(row), o.col("zone")));
    const uint64_t owner = o.u(static_cast<size_t>(row), o.col("owner"));
    return zone == "BATTLEFIELD" || zone == "STACK" || (zone == "HAND" && owner == static_cast<uint64_t>(bottom + 1));
}

static std::string group_title(int seat, rmwork::ZoneKind z, int you) {
    const std::string zone = lower(rmwork::zone_title(z));
    if (seat == you) return "Your " + zone;
    return std::string(seat == 1 ? "Player B" : "Player A") + "'s " + zone;
}

// The groups' order: the bottom seat's library, graveyard, exile, hand, then the other seat's, then the
// cards in no zone listed.
static int group_rank(const rmwork::CardPickGroup &g, int bottom) {
    if (g.other_zone) return 100;
    static const int ZONE_RANK[] = {3, 0, 1, 2};  // HAND, LIBRARY, GRAVEYARD, EXILE
    return (g.seat == bottom ? 0 : 10) + ZONE_RANK[static_cast<int>(g.zone)];
}

static rmwork::CardPickEntry entry_of(
    int seat, rmwork::ZoneKind z, const rmwork::ZoneCard &c, const rmwork::CardBook &b) {
    rmwork::CardPickEntry e;
    e.seat = seat;
    e.zone = z;
    e.handles.push_back(c.handle);
    e.card = c.card;
    e.back = c.back;
    e.name = c.name;
    e.colors = c.colors;
    e.land = c.land;
    e.token = c.token;
    e.subtypes = c.subtypes;
    e.pos = c.pos;
    const rmwork::CardFace *f = c.card >= 0 ? b.face(c.card, c.back) : nullptr;
    if (f) e.type_line = f->type_line;
    return e;
}

// The library's top as the picks so far leave it, the place the next pick fills, and the other piles.
static void build_pile(const agents::DecisionView &v, const std::vector<PastPick> &past, rmwork::PickerPile *pile) {
    const int depth = menu_depth(v);
    if (v.n() && table_category(v.category(0))) return;
    if (depth == -2 && !has_category(v, "BOTTOM_DECK_CARD")) {
        for (const PastPick &p : past) pile->chosen.push_back(p.card);
        pile->shown = !pile->chosen.empty();
        return;
    }
    pile->shown = true;
    pile->ordered = depth >= 0;
    if (pile->ordered) {
        std::map<int, rmwork::PileCard> fill;
        int deepest = depth;
        for (const PastPick &p : past) {
            if (p.category == "TOP_LIBRARY") {
                fill[static_cast<int>(p.ordinal)] = p.card;
                deepest = std::max(deepest, static_cast<int>(p.ordinal));
            } else if (p.category == "BOTTOM_DECK_CARD") {
                pile->bottom.push_back(p.card);
            } else {
                pile->graveyard.push_back(p.card);
            }
        }
        for (int d = 0; d <= deepest; ++d) {
            rmwork::PileSlot s;
            s.label = rmwork::depth_label(d);
            s.next = d == depth;
            auto it = fill.find(d);
            if (it != fill.end()) {
                s.filled = true;
                s.card = it->second;
            }
            pile->top.push_back(s);
        }
        return;
    }
    rmwork::PileSlot next;
    next.label = rmwork::depth_label(0);
    next.next = true;
    pile->top.push_back(next);
    for (auto it = past.rbegin(); it != past.rend(); ++it) {
        if (it->category == "TOP_LIBRARY") {
            rmwork::PileSlot s;
            s.label = rmwork::depth_label(static_cast<int>(pile->top.size()));
            s.filled = true;
            s.card = it->card;
            pile->top.push_back(s);
        } else if (it->category == "BOTTOM_DECK_CARD") {
            pile->bottom.insert(pile->bottom.begin(), it->card);
        } else {
            pile->graveyard.insert(pile->graveyard.begin(), it->card);
        }
    }
}

namespace rmwork {

std::string depth_label(int depth) {
    if (depth <= 0) return "Top";
    const int n = depth + 1, t = n % 100, d = n % 10;
    const char *suffix = (t >= 11 && t <= 13) ? "th" : d == 1 ? "st" : d == 2 ? "nd" : d == 3 ? "rd" : "th";
    return std::to_string(n) + suffix;
}

bool card_pick_menu(const agents::DecisionView &v) {
    if (v.n() == 0) return false;
    const agents::Table menu = v.table("menu");
    const int c_src = menu.col("source");
    bool names = false;
    for (size_t i = 0; i < v.n(); ++i) {
        if (!pick_category(v.category(i))) return false;
        names = names || menu.u(i, c_src) != 0;
    }
    return names;
}

bool CardPicker::off_table() const {
    for (const CardPickGroup &g : groups)
        for (const CardPickEntry &e : g.entries)
            if (e.pickable()) return true;
    return false;
}

size_t CardPicker::entry_count() const {
    size_t n = 0;
    for (const CardPickGroup &g : groups) n += g.entries.size();
    return n;
}

bool CardPicker::entry_of_row(int row, size_t *group, size_t *entry) const {
    for (size_t g = 0; g < groups.size(); ++g)
        for (size_t e = 0; e < groups[g].entries.size(); ++e) {
            const std::vector<int> &rows = groups[g].entries[e].rows;
            if (std::find(rows.begin(), rows.end(), row) != rows.end()) {
                *group = g;
                *entry = e;
                return true;
            }
        }
    return false;
}

bool picker_entry_matches(const CardPickEntry &e, const std::string &filter) {
    if (filter.empty()) return true;
    const std::string f = lower(filter);
    return lower(e.name).find(f) != std::string::npos || lower(e.type_line).find(f) != std::string::npos;
}

CardPicker read_card_picker(const Match &m, size_t index, int view, const agents::DecisionView &v, const CardBook &book,
    const ZoneModel &zones, int chosen_row) {
    CardPicker p;
    if (!card_pick_menu(v)) return p;
    const int bottom = view == 1 ? 1 : 0;
    const int you = view < 2 && m.live ? view : -1;
    const agents::Table menu = v.table("menu");
    const int c_src = menu.col("source"), c_card = menu.col("source_card");
    for (size_t i = 0; i < v.n(); ++i) p.row_text.push_back(v.description(i));
    p.chosen_row = chosen_row >= 0 && static_cast<size_t>(chosen_row) < v.n() ? chosen_row : -1;
    p.source = pending_source(v);
    const agents::Table pend = v.table("pending");
    if (pend.rows() && !pend.none(0, pend.col("source_card"))) {
        const CardFace *f = book.face(pend.card(0, pend.col("source_card")), false);
        if (f) p.source_name = f->name;
    }

    std::map<uint64_t, Spot> spots;
    for (int s = 0; s < 2; ++s)
        for (int z = 0; z < ZONE_KINDS; ++z)
            for (const ZoneCard &c : zones.get(s, static_cast<ZoneKind>(z)).cards)
                if (c.handle) spots[c.handle] = Spot{s, static_cast<ZoneKind>(z), &c};
    const auto group_for = [&p](int seat, ZoneKind z, bool other) -> CardPickGroup & {
        for (CardPickGroup &g : p.groups)
            if (g.other_zone == other && (other || (g.seat == seat && g.zone == z))) return g;
        p.groups.emplace_back();
        CardPickGroup &g = p.groups.back();
        g.seat = seat;
        g.zone = z;
        g.other_zone = other;
        return g;
    };
    const auto add_row = [](CardPickEntry &e, int row, const std::string &action) {
        if (std::find(e.actions.begin(), e.actions.end(), action) != e.actions.end()) return;
        e.rows.push_back(row);
        e.actions.push_back(action);
    };
    for (size_t i = 0; i < v.n(); ++i) {
        const uint64_t h = menu.u(i, c_src);
        if (h == 0) {
            p.buttons.push_back(static_cast<int>(i));
            continue;
        }
        if (on_table(v, h, bottom)) continue;
        const std::string action = row_action(v, i);
        auto spot = spots.find(h);
        if (spot == spots.end()) {
            // A card in no zone the model lists (a sideboard): what the row says of it.
            CardPickGroup &g = group_for(-1, ZoneKind::LIBRARY, true);
            CardPickEntry e;
            e.seat = -1;
            e.other_zone = true;
            e.handles.push_back(h);
            e.card = menu.none(i, c_card) ? -1 : menu.card(i, c_card);
            const CardFace *f = e.card >= 0 ? book.face(e.card, false) : nullptr;
            e.name = f ? f->name : v.description(i);
            if (f) {
                e.type_line = f->type_line;
                e.colors = f->colors;
                e.land = f->land;
                e.subtypes = f->subtypes;
            }
            add_row(e, static_cast<int>(i), action);
            g.entries.push_back(e);
            continue;
        }
        const Spot &s = spot->second;
        CardPickGroup &g = group_for(s.seat, s.zone, false);
        CardPickEntry *e = nullptr;
        for (CardPickEntry &x : g.entries)
            if (x.card == s.card->card && x.card >= 0 && x.back == s.card->back) e = &x;
        if (!e) {
            g.entries.push_back(entry_of(s.seat, s.zone, *s.card, book));
            e = &g.entries.back();
        } else if (std::find(e->handles.begin(), e->handles.end(), h) == e->handles.end()) {
            e->handles.push_back(h);
        }
        add_row(*e, static_cast<int>(i), action);
    }
    // The other copies of each entry in its zone, and the other cards a window shows of a library or of the
    // other seat's hand (Thoughtseize's reveal).
    std::set<uint64_t> windows;
    const agents::Table rev = v.table("reveal");
    for (size_t r = 0; r < rev.rows(); ++r) windows.insert(rev.u(r, rev.col("handle")));
    for (CardPickGroup &g : p.groups) {
        if (g.other_zone) continue;
        const ZoneList &z = zones.get(g.seat, g.zone);
        g.zone_size = z.size;
        for (const ZoneCard &c : z.cards) {
            // A library card counts only while a window shows it (the omniscient viewer's frame lists the
            // whole library).
            if (!c.handle || c.card < 0 || (g.zone == ZoneKind::LIBRARY && !windows.count(c.handle))) continue;
            bool placed = false;
            for (CardPickEntry &e : g.entries) {
                if (std::find(e.handles.begin(), e.handles.end(), c.handle) != e.handles.end()) {
                    placed = true;
                    break;
                }
                if (e.card == c.card && e.back == c.back) {
                    e.handles.push_back(c.handle);
                    e.pos = e.pos < 0 ? c.pos : std::min(e.pos, c.pos);
                    placed = true;
                    break;
                }
            }
            if (!placed && (g.zone == ZoneKind::LIBRARY || (g.zone == ZoneKind::HAND && g.seat != bottom)))
                g.entries.push_back(entry_of(g.seat, g.zone, c, book));
        }
        if (g.zone == ZoneKind::LIBRARY || g.zone == ZoneKind::GRAVEYARD)
            std::stable_sort(g.entries.begin(), g.entries.end(),
                [](const CardPickEntry &a, const CardPickEntry &b) { return a.pos < b.pos; });
    }
    for (CardPickGroup &g : p.groups) g.title = g.other_zone ? "Other cards" : group_title(g.seat, g.zone, you);
    std::stable_sort(p.groups.begin(), p.groups.end(), [bottom](const CardPickGroup &a, const CardPickGroup &b) {
        return group_rank(a, bottom) < group_rank(b, bottom);
    });

    build_pile(v, past_picks(m, index, view, p.source, v, book), &p.pile);
    p.active = p.off_table() || p.pile.shown;
    if (!p.active) return p;
    int cards = 0;
    std::string where;
    for (size_t g = 0; g < p.groups.size(); ++g) {
        for (const CardPickEntry &e : p.groups[g].entries) cards += e.count();
        std::string t = p.groups[g].title;
        if (g > 0) t[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(t[0])));
        where += (g == 0 ? "" : g + 1 == p.groups.size() ? " and " : ", ") + t;
    }
    p.title = p.source_name.empty() ? std::string("Choose a card") : p.source_name;
    if (!where.empty())
        p.title += " \xC2\xB7 " + where + " \xC2\xB7 " + std::to_string(cards) + (cards == 1 ? " card" : " cards");
    return p;
}

}  // namespace rmwork
