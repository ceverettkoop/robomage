#include "state_tree.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <functional>

namespace {

// The tables whose rows belong to an object (their first handle column names it) or to a stack
// object (their `stack` column).
const char *const OBJECT_CHILD_TABLES[] = {"object_subtype", "object_keyword", "object_counter", "object_ability",
    "object_provenance", "object_link", "play_permission", "shield", "reveal"};
const char *const STACK_CHILD_TABLES[] = {"stack_mode", "stack_target"};

// The zones a player node lists even when the frame holds none of their cards (their counts are in
// the player row).
const char *const PLAYER_ZONES[] = {"BATTLEFIELD", "HAND", "LIBRARY", "GRAVEYARD", "EXILE", "SIDEBOARD"};

}  // namespace

static const char *storage_name(rmstate::Storage s);
static std::string lower(const std::string &s);
static std::string humanize(const std::string &s);
static std::string seat_letter(uint64_t seat);
static size_t first_handle_column(const rmstate::DecodedTable &t);

static const char *storage_name(rmstate::Storage s) {
    static const char *const NAMES[] = {"u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "text"};
    const size_t i = static_cast<size_t>(s);
    return i < sizeof NAMES / sizeof NAMES[0] ? NAMES[i] : "?";
}

static std::string lower(const std::string &s) {
    std::string out;
    for (char c : s) out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// FIRST_MAIN -> "First main".
static std::string humanize(const std::string &s) {
    std::string out = lower(s);
    for (char &c : out)
        if (c == '_') c = ' ';
    if (!out.empty()) out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
    return out;
}

// The `seat` enum (1 A, 2 B) as "A" / "B".
static std::string seat_letter(uint64_t seat) {
    return seat == 1 ? "A" : seat == 2 ? "B" : "";
}

// The table's first column tagged `handle` (the table's size when it has none).
static size_t first_handle_column(const rmstate::DecodedTable &t) {
    for (size_t c = 0; c < t.def->columns.size(); ++c)
        if (t.def->columns[c].tag == "handle") return c;
    return t.def->columns.size();
}

namespace rmwork {

void StateTreeModel::build(const DecisionFrame &f, const TableModel &tm) {
    frame_ = f.frame.get();
    stream_ = f.stream;
    book_ = f.book;
    viewer_ = f.view;
    roots_.clear();
    objects_.clear();
    stack_names_.clear();
    if (!frame_) return;
    if (const rmstate::DecodedTable *o = table("object")) {
        const int c = col(*o, "handle");
        for (size_t r = 0; r < o->rows.size(); ++r) objects_.emplace_back(o->rows[r][static_cast<size_t>(c)].bits, r);
        std::sort(objects_.begin(), objects_.end());
    }
    for (const TableStackItem &it : tm.stack) stack_names_.emplace_back(it.handle, it.name);

    TreeRow game;
    game.kind = TreeRow::Kind::GAME;
    game.key = "game";
    game.name = "game";
    game.type = "struct Game";
    game.expandable = true;
    std::string gv;
    if (const rmstate::DecodedTable *h = table("header"))
        if (!h->rows.empty()) {
            gv = "Game " + std::to_string(cell(*h, 0, "game") + 1);
            if (cell(*h, 0, "sideboarding")) gv += " \xC2\xB7 sideboarding";
            if (cell(*h, 0, "terminal")) gv += " \xC2\xB7 match over";
        }
    if (const rmstate::DecodedTable *t = table("turn"))
        if (!t->rows.empty()) {
            gv += " \xC2\xB7 Turn " + std::to_string(cell(*t, 0, "turn") + 1) + " \xC2\xB7 " +
                  humanize(stream_->enum_name("step", cell(*t, 0, "step")));
            const std::string active = seat_letter(cell(*t, 0, "active"));
            if (!active.empty()) gv += " \xC2\xB7 Player " + active + "'s turn";
        }
    game.value = gv;
    roots_.push_back(game);

    for (int s = 0; s < 2; ++s) {
        TreeRow p;
        p.kind = TreeRow::Kind::PLAYER;
        p.seat = s;
        p.key = player_key(s);
        p.name = std::string("player ") + (s == 0 ? "A" : "B");
        p.type = "struct Player";
        p.expandable = true;
        p.table = table_index("player");
        if (const rmstate::DecodedTable *pt = table("player"))
            for (size_t r = 0; r < pt->rows.size(); ++r)
                if (cell(*pt, r, "player") == static_cast<uint64_t>(s + 1)) {
                    p.row = static_cast<int>(r);
                    p.value = std::to_string(static_cast<int64_t>(rmstate::value_signed(
                                  pt->rows[r][static_cast<size_t>(col(*pt, "life"))], rmstate::Storage::I32))) +
                              " life \xC2\xB7 hand " + std::to_string(cell(*pt, r, "hand")) + " \xC2\xB7 library " +
                              std::to_string(cell(*pt, r, "library"));
                }
        roots_.push_back(p);
    }

    TreeRow st;
    st.kind = TreeRow::Kind::STACK;
    st.key = "stack";
    st.name = "stack";
    st.zone = "Stack";
    const rmstate::DecodedTable *stack = table("stack");
    const size_t n = stack ? stack->rows.size() : 0;
    st.type = "Object[" + std::to_string(n) + "]";
    st.value = n ? std::to_string(n) + (n == 1 ? " object" : " objects") + ", top first" : "empty";
    st.expandable = n > 0;
    st.table = table_index("stack");
    roots_.push_back(st);

    TreeRow tables;
    tables.kind = TreeRow::Kind::TABLES;
    tables.key = "tables";
    tables.name = "frame tables";
    tables.type = "STATE frame";
    tables.value = std::to_string(frame_->tables.size()) + " tables, " + std::string(view_name(viewer_)) + "'s";
    tables.expandable = true;
    roots_.push_back(tables);
}

std::vector<TreeRow> StateTreeModel::children(const TreeRow &r) const {
    std::vector<TreeRow> out;
    if (!frame_) return out;
    const int d = r.depth + 1;
    switch (r.kind) {
        case TreeRow::Kind::GAME:
            if (const rmstate::DecodedTable *h = table("header"))
                if (!h->rows.empty()) add_fields(*h, 0, r.key + "/header", d, &out);
            if (const rmstate::DecodedTable *t = table("turn"))
                if (!t->rows.empty()) add_fields(*t, 0, r.key + "/turn", d, &out);
            return out;
        case TreeRow::Kind::PLAYER:
            return player_children(r);
        case TreeRow::Kind::ZONE:
            return zone_children(r);
        case TreeRow::Kind::OBJECT: {
            const rmstate::DecodedTable &t = frame_->tables[static_cast<size_t>(r.table)];
            add_fields(t, static_cast<size_t>(r.row), r.key, d, &out);
            add_child_rows(OBJECT_CHILD_TABLES, sizeof OBJECT_CHILD_TABLES / sizeof OBJECT_CHILD_TABLES[0], r.handle,
                r.key, d, &out);
            return out;
        }
        case TreeRow::Kind::STACK: {
            const rmstate::DecodedTable *t = table("stack");
            if (!t) return out;
            std::vector<std::pair<uint64_t, size_t>> order;
            for (size_t i = 0; i < t->rows.size(); ++i) order.emplace_back(cell(*t, i, "pos"), i);
            std::sort(order.begin(), order.end());
            for (const auto &o : order) {
                const size_t i = o.second;
                TreeRow s;
                s.kind = TreeRow::Kind::STACK_ITEM;
                s.depth = d;
                s.handle = cell(*t, i, "handle");
                s.key = r.key + "/" + std::to_string(s.handle);
                std::string name = object_name(s.handle);
                for (const auto &sn : stack_names_)
                    if (sn.first == s.handle) name = sn.second;
                s.name = (o.first == 0 ? "top: " : "") + (name.empty() ? std::string("stack object") : name);
                s.zone = "Stack";
                s.type = humanize(stream_->enum_name("stack_kind", cell(*t, i, "kind"))) + " \xC2\xB7 Player " +
                         seat_letter(cell(*t, i, "controller"));
                s.card_type = true;
                auto it = std::lower_bound(objects_.begin(), objects_.end(), std::make_pair(s.handle, size_t(0)));
                if (it != objects_.end() && it->first == s.handle) {
                    const rmstate::DecodedTable *o = table("object");
                    s.swatch = true;
                    s.colors = book_->bits().wubrg(cell(*o, it->second, "colors"));
                }
                s.value = "#" + std::to_string(s.handle);
                s.expandable = true;
                s.table = table_index("stack");
                s.row = static_cast<int>(i);
                out.push_back(s);
            }
            return out;
        }
        case TreeRow::Kind::STACK_ITEM: {
            const rmstate::DecodedTable &t = frame_->tables[static_cast<size_t>(r.table)];
            add_fields(t, static_cast<size_t>(r.row), r.key, d, &out);
            add_child_rows(
                STACK_CHILD_TABLES, sizeof STACK_CHILD_TABLES / sizeof STACK_CHILD_TABLES[0], r.handle, r.key, d, &out);
            return out;
        }
        case TreeRow::Kind::TABLES:
            for (size_t i = 0; i < frame_->tables.size(); ++i) {
                const rmstate::DecodedTable &t = frame_->tables[i];
                TreeRow n;
                n.kind = TreeRow::Kind::TABLE;
                n.depth = d;
                n.name = t.def->name;
                n.key = r.key + "/" + n.name;
                n.type = "table " + std::to_string(t.id);
                n.value = std::to_string(t.rows.size()) + (t.rows.size() == 1 ? " row" : " rows");
                n.expandable = !t.rows.empty();
                n.table = static_cast<int>(i);
                out.push_back(n);
            }
            return out;
        case TreeRow::Kind::TABLE: {
            const rmstate::DecodedTable &t = frame_->tables[static_cast<size_t>(r.table)];
            const bool objects = t.def->name == "object";
            const size_t hc = first_handle_column(t);
            for (size_t i = 0; i < t.rows.size(); ++i) {
                TreeRow n;
                n.kind = TreeRow::Kind::ROW;
                n.depth = d;
                n.name = "[" + std::to_string(i) + "]";
                n.key = r.key + "/" + std::to_string(i);
                n.type = t.def->name;
                n.expandable = true;
                n.table = r.table;
                n.row = static_cast<int>(i);
                if (objects) {
                    const TreeRow o = object_row(i, r.key, d);
                    n.name += " " + o.name;
                    n.swatch = o.swatch;
                    n.colors = o.colors;
                    n.land = o.land;
                    n.zone = o.zone;
                    n.handle = o.handle;
                    n.value = o.value;
                } else {
                    if (hc == 0 && t.def->columns[0].name == "handle") n.handle = t.rows[i][0].bits;
                    for (size_t c = 0; c < t.def->columns.size() && c < 3; ++c)
                        n.value += (c ? "  " : "") + t.def->columns[c].name + "=" + cell_text(t, i, c);
                }
                out.push_back(n);
            }
            return out;
        }
        case TreeRow::Kind::ROW: {
            const rmstate::DecodedTable &t = frame_->tables[static_cast<size_t>(r.table)];
            add_fields(t, static_cast<size_t>(r.row), r.key, d, &out);
            return out;
        }
        case TreeRow::Kind::FIELD:
            return out;
    }
    return out;
}

std::vector<TreeRow> StateTreeModel::flatten(const std::set<std::string> &open) const {
    std::vector<TreeRow> out;
    const std::function<void(const TreeRow &)> walk = [&](const TreeRow &r) {
        out.push_back(r);
        if (!r.expandable || !open.count(r.key)) return;
        for (const TreeRow &c : children(r)) walk(c);
    };
    for (const TreeRow &r : roots_) walk(r);
    return out;
}

std::string StateTreeModel::object_name(uint64_t handle) const {
    if (handle == 0) return "";
    auto it = std::lower_bound(objects_.begin(), objects_.end(), std::make_pair(handle, size_t(0)));
    if (it == objects_.end() || it->first != handle) return "";
    const rmstate::DecodedTable *o = table("object");
    const rmstate::Value &v = o->rows[it->second][static_cast<size_t>(col(*o, "card"))];
    if (rmstate::value_is_all_ones(v, rmstate::Storage::U16)) return "";
    const int card = static_cast<int>(v.bits);
    const CardFace *face = book_->face(card, false);
    return face ? face->name : stream_->cards().name(card);
}

std::vector<std::string> StateTreeModel::object_path(uint64_t handle) const {
    if (!frame_ || handle == 0) return {};
    if (const rmstate::DecodedTable *s = table("stack"))
        for (size_t i = 0; i < s->rows.size(); ++i)
            if (cell(*s, i, "handle") == handle) return {"stack", "stack/" + std::to_string(handle)};
    auto it = std::lower_bound(objects_.begin(), objects_.end(), std::make_pair(handle, size_t(0)));
    if (it == objects_.end() || it->first != handle) return {};
    const rmstate::DecodedTable *o = table("object");
    const int z = static_cast<int>(cell(*o, it->second, "zone"));
    if (z == stream_->enum_value("zone", "STACK")) return {};
    const bool battlefield = z == stream_->enum_value("zone", "BATTLEFIELD");
    const uint64_t seat = cell(*o, it->second, battlefield ? "controller" : "owner");
    if (seat != 1 && seat != 2) return {};
    const std::string player = player_key(static_cast<int>(seat) - 1);
    const std::string zone = player + "/" + lower(stream_->enum_name("zone", static_cast<uint64_t>(z)));
    return {player, zone, zone + "/" + std::to_string(handle)};
}

std::string StateTreeModel::player_key(int seat) {
    return std::string("player/") + (seat == 0 ? "A" : "B");
}

std::string StateTreeModel::cell_text(const rmstate::DecodedTable &t, size_t row, size_t c) const {
    const rmstate::ColumnDef &def = t.def->columns[c];
    const rmstate::Value &v = t.rows[row][c];
    const std::string &tag = def.tag;
    if (def.storage == rmstate::Storage::TEXT) return "\"" + v.text + "\"";
    if (def.storage == rmstate::Storage::F32) {
        float f = 0;
        const uint32_t bits = static_cast<uint32_t>(v.bits);
        std::memcpy(&f, &bits, sizeof f);
        char buf[32];
        std::snprintf(buf, sizeof buf, "%g", static_cast<double>(f));
        return buf;
    }
    const bool none = rmstate::value_is_all_ones(v, def.storage);
    if (tag == "handle") {
        if (v.bits == 0) return "\xE2\x80\x94";
        const std::string name = object_name(v.bits);
        return "#" + std::to_string(v.bits) + (name.empty() ? "" : " " + name);
    }
    if (tag == "card") return none ? "\xE2\x80\x94" : stream_->cards().name(static_cast<int>(v.bits));
    if (tag == "bool") return v.bits ? "true" : "false";
    if (tag == "player-rel") return v.bits == 1 ? "self" : v.bits == 2 ? "opponent" : "none";
    if (tag.compare(0, 5, "enum:") == 0) {
        if (none && tag != "enum:seat") return "none";
        const std::string e = tag.substr(5);
        if (stream_->enum_values(e.c_str()).empty()) return std::to_string(v.bits);
        return stream_->enum_name(e.c_str(), v.bits);
    }
    if (tag.compare(0, 5, "mask:") == 0) {
        if (v.bits == 0) return "\xE2\x80\x94";
        const std::vector<std::string> &names = stream_->enum_values(tag.substr(5).c_str());
        std::string out;
        for (size_t i = 0; i < 64; ++i) {
            if (!(v.bits & (1ull << i))) continue;
            out += (out.empty() ? "" : " | ") + (i < names.size() ? names[i] : "bit " + std::to_string(i));
        }
        return out;
    }
    if (tag.compare(0, 4, "row:") == 0) return none ? "\xE2\x80\x94" : "row " + std::to_string(v.bits);
    if (tag == "index" && none) return "\xE2\x80\x94";
    if (tag == "signed" || def.storage == rmstate::Storage::I8 || def.storage == rmstate::Storage::I16 ||
        def.storage == rmstate::Storage::I32 || def.storage == rmstate::Storage::I64)
        return std::to_string(rmstate::value_signed(v, def.storage));
    return std::to_string(v.bits);
}

void StateTreeModel::add_fields(
    const rmstate::DecodedTable &t, size_t row, const std::string &key, int depth, std::vector<TreeRow> *out) const {
    for (size_t c = 0; c < t.def->columns.size(); ++c) {
        const rmstate::ColumnDef &def = t.def->columns[c];
        TreeRow f;
        f.kind = TreeRow::Kind::FIELD;
        f.depth = depth;
        f.name = def.name;
        f.key = key + "/" + def.name;
        f.table = static_cast<int>(&t - frame_->tables.data());
        f.row = static_cast<int>(row);
        f.type = std::string(storage_name(def.storage)) + (def.tag.empty() ? "" : " " + def.tag);
        f.value = cell_text(t, row, c);
        out->push_back(f);
    }
}

void StateTreeModel::add_child_rows(const char *const *tables, size_t n, uint64_t handle, const std::string &key,
    int depth, std::vector<TreeRow> *out) const {
    for (size_t k = 0; k < n; ++k) {
        const int ti = table_index(tables[k]);
        if (ti < 0) continue;
        const rmstate::DecodedTable &t = frame_->tables[static_cast<size_t>(ti)];
        const int sc = col(t, "stack");
        const size_t hc = sc >= 0 ? static_cast<size_t>(sc) : first_handle_column(t);
        if (hc >= t.def->columns.size()) continue;
        for (size_t r = 0; r < t.rows.size(); ++r) {
            if (t.rows[r][hc].bits != handle) continue;
            TreeRow n;
            n.kind = TreeRow::Kind::ROW;
            n.depth = depth;
            n.name = t.def->name + "[" + std::to_string(r) + "]";
            n.key = key + "/" + n.name;
            n.type = t.def->name;
            n.expandable = true;
            n.table = ti;
            n.row = static_cast<int>(r);
            for (size_t c = 0, shown = 0; c < t.def->columns.size() && shown < 3; ++c) {
                if (c == hc || t.def->columns[c].tag == "player-rel") continue;
                n.value += (shown ? "  " : "") + t.def->columns[c].name + "=" + cell_text(t, r, c);
                ++shown;
            }
            out->push_back(n);
        }
    }
}

std::vector<TreeRow> StateTreeModel::player_children(const TreeRow &r) const {
    std::vector<TreeRow> out;
    const int d = r.depth + 1;
    const rmstate::DecodedTable *pt = table("player");
    if (pt && r.row >= 0) {
        // The life first, then the counters and zones; every column under "player row".
        for (size_t c = 0; c < pt->def->columns.size(); ++c)
            if (pt->def->columns[c].name == "life") {
                TreeRow f;
                f.kind = TreeRow::Kind::FIELD;
                f.depth = d;
                f.name = "life";
                f.key = r.key + "/life";
                f.type = std::string(storage_name(pt->def->columns[c].storage)) + " " + pt->def->columns[c].tag;
                f.value = cell_text(*pt, static_cast<size_t>(r.row), c);
                out.push_back(f);
            }
    }
    if (const rmstate::DecodedTable *pc = table("player_counter"))
        for (size_t i = 0; i < pc->rows.size(); ++i) {
            if (cell(*pc, i, "player") != static_cast<uint64_t>(r.seat + 1)) continue;
            TreeRow c;
            c.kind = TreeRow::Kind::FIELD;
            c.depth = d;
            c.name = "counter " + pc->rows[i][static_cast<size_t>(col(*pc, "kind"))].text;
            c.key = r.key + "/" + c.name;
            c.type = "u16 count";
            c.value = std::to_string(cell(*pc, i, "count"));
            out.push_back(c);
        }
    // The zones: the usual five and the sideboard, then any other the frame puts the player's cards in.
    std::vector<int> zones;
    for (const char *z : PLAYER_ZONES) zones.push_back(stream_->enum_value("zone", z));
    const rmstate::DecodedTable *o = table("object");
    const int stack_zone = stream_->enum_value("zone", "STACK");
    if (o)
        for (size_t i = 0; i < o->rows.size(); ++i) {
            const int z = static_cast<int>(cell(*o, i, "zone"));
            if (z != stack_zone && std::find(zones.begin(), zones.end(), z) == zones.end() &&
                cell(*o, i, "owner") == static_cast<uint64_t>(r.seat + 1))
                zones.push_back(z);
        }
    const int battlefield = stream_->enum_value("zone", "BATTLEFIELD");
    for (int z : zones) {
        if (z < 0) continue;
        const std::string name = stream_->enum_name("zone", static_cast<uint64_t>(z));
        TreeRow n;
        n.kind = TreeRow::Kind::ZONE;
        n.depth = d;
        n.zone_id = z;
        n.seat = r.seat;
        n.name = lower(name);
        n.key = r.key + "/" + n.name;
        n.zone = humanize(name);
        size_t listed = 0;
        if (o)
            for (size_t i = 0; i < o->rows.size(); ++i) {
                if (static_cast<int>(cell(*o, i, "zone")) != z) continue;
                const char *who = z == battlefield ? "controller" : "owner";
                if (cell(*o, i, who) == static_cast<uint64_t>(r.seat + 1)) ++listed;
            }
        n.type = "Object[" + std::to_string(listed) + "]";
        n.expandable = listed > 0;
        const int cc = pt ? col(*pt, lower(name).c_str()) : -1;
        if (cc >= 0 && r.row >= 0) {
            const uint64_t count = pt->rows[static_cast<size_t>(r.row)][static_cast<size_t>(cc)].bits;
            n.value = std::to_string(count) + (count == 1 ? " card" : " cards");
            if (listed < count) n.value += " (" + std::to_string(listed) + " the frame shows)";
        } else {
            n.value = std::to_string(listed) + (z == battlefield ? (listed == 1 ? " permanent" : " permanents")
                                                                 : (listed == 1 ? " card" : " cards"));
        }
        out.push_back(n);
    }
    if (pt && r.row >= 0) {
        TreeRow row;
        row.kind = TreeRow::Kind::ROW;
        row.depth = d;
        row.name = "player row";
        row.key = r.key + "/row";
        row.type = "player[" + std::to_string(r.row) + "]";
        row.value = std::to_string(pt->def->columns.size()) + " columns";
        row.expandable = true;
        row.table = table_index("player");
        row.row = r.row;
        out.push_back(row);
    }
    return out;
}

std::vector<TreeRow> StateTreeModel::zone_children(const TreeRow &r) const {
    std::vector<TreeRow> out;
    const rmstate::DecodedTable *o = table("object");
    if (!o) return out;
    const bool battlefield = r.zone_id == stream_->enum_value("zone", "BATTLEFIELD");
    std::vector<std::pair<uint64_t, size_t>> order;
    const int c_pos = col(*o, "pos"), c_ts = col(*o, "timestamp");
    for (size_t i = 0; i < o->rows.size(); ++i) {
        if (static_cast<int>(cell(*o, i, "zone")) != r.zone_id) continue;
        if (cell(*o, i, battlefield ? "controller" : "owner") != static_cast<uint64_t>(r.seat + 1)) continue;
        const rmstate::Value &k = o->rows[i][static_cast<size_t>(battlefield ? c_ts : c_pos)];
        const bool none = rmstate::value_is_all_ones(k, battlefield ? rmstate::Storage::U32 : rmstate::Storage::U16);
        order.emplace_back(none ? UINT64_MAX : k.bits, i);
    }
    std::stable_sort(order.begin(), order.end(),
        [](const std::pair<uint64_t, size_t> &a, const std::pair<uint64_t, size_t> &b) { return a.first < b.first; });
    for (const auto &p : order) out.push_back(object_row(p.second, r.key, r.depth + 1));
    return out;
}

TreeRow StateTreeModel::object_row(size_t i, const std::string &parent_key, int depth) const {
    const rmstate::DecodedTable *o = table("object");
    TreeRow n;
    n.kind = TreeRow::Kind::OBJECT;
    n.depth = depth;
    n.handle = cell(*o, i, "handle");
    n.key = parent_key + "/" + std::to_string(n.handle);
    n.table = table_index("object");
    n.row = static_cast<int>(i);
    n.expandable = true;
    n.zone = humanize(stream_->enum_name("zone", cell(*o, i, "zone")));
    const rmstate::Value &cv = o->rows[i][static_cast<size_t>(col(*o, "card"))];
    const bool identified = !rmstate::value_is_all_ones(cv, rmstate::Storage::U16);
    const bool face_down = cell(*o, i, "face_down") != 0;
    const TypeBits &bits = book_->bits();
    const uint64_t types = cell(*o, i, "types");
    n.swatch = true;
    n.colors = bits.wubrg(cell(*o, i, "colors"));
    n.land = (types & bits.land) != 0 && (types & bits.creature) == 0;
    n.card_type = true;
    if (identified) {
        int card = static_cast<int>(cv.bits);
        const rmstate::Value &copy = o->rows[i][static_cast<size_t>(col(*o, "copy_of"))];
        if (!rmstate::value_is_all_ones(copy, rmstate::Storage::U16)) card = static_cast<int>(copy.bits);
        const bool back = cell(*o, i, "token") == 0 && cell(*o, i, "transformed") != 0 && book_->face(card, true);
        const CardFace *face = book_->face(card, back);
        n.name = face ? face->name : stream_->cards().name(card, back);
        if (face) n.type = face->type_line;
        if (face_down) n.name += " (face down)";
    } else {
        n.name = face_down ? "face-down card" : "unidentified card";
        n.swatch = false;
        if (face_down && types) n.type = "Creature";
    }
    n.value = object_summary(i);
    return n;
}

// "tapped · 3/3 · damage 1 · +1/+1 ×2", or "—".
std::string StateTreeModel::object_summary(size_t i) const {
    const rmstate::DecodedTable *o = table("object");
    std::vector<std::string> parts;
    if (cell(*o, i, "tapped")) parts.push_back("tapped");
    const uint64_t types = cell(*o, i, "types");
    if ((types & book_->bits().creature) &&
        static_cast<int>(cell(*o, i, "zone")) == stream_->enum_value("zone", "BATTLEFIELD")) {
        const auto sv = [&](const char *c) {
            return rmstate::value_signed(o->rows[i][static_cast<size_t>(col(*o, c))], rmstate::Storage::I16);
        };
        parts.push_back(std::to_string(sv("power")) + "/" + std::to_string(sv("toughness")));
    }
    if (const uint64_t dmg = cell(*o, i, "damage")) parts.push_back("damage " + std::to_string(dmg));
    if (cell(*o, i, "attacking")) parts.push_back("attacking");
    if (cell(*o, i, "blocking")) parts.push_back("blocking");
    if (cell(*o, i, "phased_out")) parts.push_back("phased out");
    const uint64_t handle = cell(*o, i, "handle");
    if (const rmstate::DecodedTable *c = table("object_counter"))
        for (size_t r = 0; r < c->rows.size(); ++r)
            if (cell(*c, r, "obj") == handle)
                parts.push_back(humanize(c->rows[r][static_cast<size_t>(col(*c, "kind"))].text) + " \xC3\x97" +
                                std::to_string(cell(*c, r, "count")));
    std::string out;
    for (const std::string &p : parts) out += (out.empty() ? "" : " \xC2\xB7 ") + p;
    return out.empty() ? "\xE2\x80\x94" : out;
}

int StateTreeModel::table_index(const char *name) const {
    for (size_t i = 0; i < frame_->tables.size(); ++i)
        if (frame_->tables[i].def->name == name) return static_cast<int>(i);
    return -1;
}

const rmstate::DecodedTable *StateTreeModel::table(const char *name) const {
    const int i = table_index(name);
    return i < 0 ? nullptr : &frame_->tables[static_cast<size_t>(i)];
}

int StateTreeModel::col(const rmstate::DecodedTable &t, const char *name) const {
    for (size_t c = 0; c < t.def->columns.size(); ++c)
        if (t.def->columns[c].name == name) return static_cast<int>(c);
    return -1;
}

uint64_t StateTreeModel::cell(const rmstate::DecodedTable &t, size_t row, const char *name) const {
    const int c = col(t, name);
    return c < 0 ? 0 : t.rows[row][static_cast<size_t>(c)].bits;
}

}  // namespace rmwork
