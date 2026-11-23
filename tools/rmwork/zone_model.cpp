#include "zone_model.h"

#include <algorithm>
#include <cctype>
#include <map>

namespace {

// The frame's tables and enum values the zones read.
struct Frame {
        const agents::SeatStream &s;
        const rmwork::CardBook &book;
        agents::Table obj, subtype, counter, link, player, reveal, kpos, khand;
        int z_hand, z_library, z_graveyard, z_exile;
        int anchor_bottom;
        std::map<uint64_t, size_t> rows;  // handle -> object row
        Frame(const agents::SeatStream &stream, const rmwork::CardBook &b, const rmstate::DecodedFrame &f);
};

}  // namespace

static int seat_of(uint64_t seat);
static std::string humanize(const std::string &name);
static rmwork::ZoneCard read_card(const Frame &f, size_t row);
static std::string object_name(const Frame &f, uint64_t handle);
static void read_counters(const Frame &f, std::map<uint64_t, rmwork::ZoneCard *> &by_handle);
static void read_links(const Frame &f, std::map<uint64_t, rmwork::ZoneCard *> &by_handle);
static std::string window_note(const Frame &f, uint64_t handle);
static void add_known_positions(const Frame &f, int seat, rmwork::ZoneList *lib);
static void add_known_hand(const Frame &f, rmwork::ZoneList *hand);
static void finish(rmwork::ZoneList *z);

Frame::Frame(const agents::SeatStream &stream, const rmwork::CardBook &b, const rmstate::DecodedFrame &f)
    : s(stream), book(b) {
    obj = agents::Table(f, s.schema(), "object");
    subtype = agents::Table(f, s.schema(), "object_subtype");
    counter = agents::Table(f, s.schema(), "object_counter");
    link = agents::Table(f, s.schema(), "object_link");
    player = agents::Table(f, s.schema(), "player");
    reveal = agents::Table(f, s.schema(), "reveal");
    kpos = agents::Table(f, s.schema(), "k_position");
    khand = agents::Table(f, s.schema(), "k_known_hand");
    z_hand = s.enum_value("zone", "HAND");
    z_library = s.enum_value("zone", "LIBRARY");
    z_graveyard = s.enum_value("zone", "GRAVEYARD");
    z_exile = s.enum_value("zone", "EXILE");
    anchor_bottom = s.enum_value("lib_anchor", "BOTTOM");
    const int c_handle = obj.col("handle");
    for (size_t r = 0; r < obj.rows(); ++r) rows[obj.u(r, c_handle)] = r;
}

// The `seat` enum (1 A, 2 B) as 0 A, 1 B; -1 for none.
static int seat_of(uint64_t seat) {
    return seat == 1 ? 0 : seat == 2 ? 1 : -1;
}

// TIME -> "Time", P1P1 -> "+1/+1".
static std::string humanize(const std::string &name) {
    if (name == "P1P1") return "+1/+1";
    if (name == "M1M1") return "-1/-1";
    std::string out;
    for (char c : name) out += c == '_' ? ' ' : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (!out.empty()) out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
    return out;
}

static rmwork::ZoneCard read_card(const Frame &f, size_t row) {
    const agents::Table &o = f.obj;
    rmwork::ZoneCard c;
    c.handle = o.u(row, o.col("handle"));
    c.card = o.card(row, o.col("card"));
    c.token = o.u(row, o.col("token")) != 0;
    c.face_down = o.u(row, o.col("face_down")) != 0;
    if (c.card >= 0) {
        const int copy = o.card(row, o.col("copy_of"));
        if (copy >= 0) c.card = copy;
        c.back = !c.token && o.u(row, o.col("transformed")) != 0 && f.book.face(c.card, true) != nullptr;
        const rmwork::CardFace *face = f.book.face(c.card, c.back);
        c.name = face ? face->name : f.s.cards().name(c.card, c.back);
        if (face) c.subtypes = face->subtypes;
    }
    const rmwork::TypeBits &bits = f.book.bits();
    const uint64_t types = o.u(row, o.col("types"));
    c.land = (types & bits.land) != 0 && (types & bits.creature) == 0;
    c.colors = bits.wubrg(o.u(row, o.col("colors")));
    if (!o.none(row, o.col("pos"))) c.pos = static_cast<int>(o.u(row, o.col("pos")));
    return c;
}

// The name the frame gives the object of `handle`; "" when it does not identify it.
static std::string object_name(const Frame &f, uint64_t handle) {
    auto it = f.rows.find(handle);
    if (it == f.rows.end()) return "";
    const int card = f.obj.card(it->second, f.obj.col("card"));
    if (card < 0) return "";
    const rmwork::CardFace *face = f.book.face(card, false);
    return face ? face->name : f.s.cards().name(card);
}

static void read_counters(const Frame &f, std::map<uint64_t, rmwork::ZoneCard *> &by_handle) {
    const agents::Table &c = f.counter;
    for (size_t r = 0; r < c.rows(); ++r) {
        auto it = by_handle.find(c.u(r, c.col("obj")));
        const uint64_t n = c.u(r, c.col("count"));
        if (it != by_handle.end() && n > 0)
            it->second->counters.push_back(humanize(c.text(r, c.col("kind"))) + " " + std::to_string(n));
    }
}

// An exiled card's source: "exiled with <name>" (the name the frame gives it, else "a card").
static void read_links(const Frame &f, std::map<uint64_t, rmwork::ZoneCard *> &by_handle) {
    const agents::Table &l = f.link;
    for (size_t r = 0; r < l.rows(); ++r) {
        auto it = by_handle.find(l.u(r, l.col("linked")));
        if (it == by_handle.end()) continue;
        const std::string src = object_name(f, l.u(r, l.col("obj")));
        const std::string kind = f.s.enum_name("link_kind", l.u(r, l.col("kind")));
        const std::string what = kind == "RETURNABLE" ? "returns with " : "exiled with ";
        if (!it->second->note.empty()) it->second->note += "; ";
        it->second->note += what + (src.empty() ? std::string("a card") : src);
    }
}

// What lets the viewer see a library card: "revealed", "looked at", else "in view".
static std::string window_note(const Frame &f, uint64_t handle) {
    const agents::Table &rv = f.reveal;
    for (size_t r = 0; r < rv.rows(); ++r) {
        if (rv.u(r, rv.col("handle")) != handle) continue;
        const std::string w = f.s.enum_name("window_kind", rv.u(r, rv.col("window")));
        return w.find("LOOK") != std::string::npos ? "looked at" : "revealed";
    }
    return "in view";
}

// The seat's library positions its knowledge names a card at (`k_position` with a card), where the
// frame does not list the card itself.
static void add_known_positions(const Frame &f, int seat, rmwork::ZoneList *lib) {
    const agents::Table &k = f.kpos;
    for (size_t r = 0; r < k.rows(); ++r) {
        if (seat_of(k.u(r, k.col("owner"))) != seat) continue;
        const int card = k.card(r, k.col("card"));
        if (card < 0) continue;
        int pos = static_cast<int>(k.u(r, k.col("at_pos")));
        if (static_cast<int>(k.u(r, k.col("at_anchor"))) == f.anchor_bottom) pos = lib->size - 1 - pos;
        bool listed = false;
        for (const rmwork::ZoneCard &c : lib->cards) listed = listed || c.pos == pos;
        if (listed) continue;
        rmwork::ZoneCard c;
        c.card = card;
        const rmwork::CardFace *face = f.book.face(card, false);
        c.name = face ? face->name : f.s.cards().name(card);
        if (face) {
            c.colors = face->colors;
            c.land = face->land && !face->creature;
            c.subtypes = face->subtypes;
            c.token = face->token;
        }
        c.pos = pos;
        c.note = "known";
        lib->cards.push_back(c);
    }
}

// The opponent's hand cards the viewer's knowledge holds as certainly there, beyond those its frame
// shows.
static void add_known_hand(const Frame &f, rmwork::ZoneList *hand) {
    const agents::Table &k = f.khand;
    for (size_t r = 0; r < k.rows(); ++r) {
        const int card = k.card(r, k.col("card"));
        const int copies = static_cast<int>(k.u(r, k.col("copies")));
        if (card < 0 || copies <= 0) continue;
        int shown = 0;
        for (const rmwork::ZoneCard &c : hand->cards) shown += c.card == card ? c.copies : 0;
        if (copies <= shown) continue;
        rmwork::ZoneCard c;
        c.card = card;
        const rmwork::CardFace *face = f.book.face(card, false);
        c.name = face ? face->name : f.s.cards().name(card);
        if (face) {
            c.colors = face->colors;
            c.land = face->land && !face->creature;
            c.subtypes = face->subtypes;
        }
        c.copies = copies - shown;
        c.note = "known";
        hand->cards.push_back(c);
    }
}

static void finish(rmwork::ZoneList *z) {
    z->hidden = std::max(0, z->size - z->listed());
    z->all_known = z->hidden == 0;
}

namespace rmwork {

const char *zone_title(ZoneKind z) {
    switch (z) {
        case ZoneKind::HAND:
            return "Hand";
        case ZoneKind::LIBRARY:
            return "Library";
        case ZoneKind::GRAVEYARD:
            return "Graveyard";
        case ZoneKind::EXILE:
            return "Exile";
    }
    return "";
}

const char *zone_key(ZoneKind z) {
    switch (z) {
        case ZoneKind::HAND:
            return "hand";
        case ZoneKind::LIBRARY:
            return "library";
        case ZoneKind::GRAVEYARD:
            return "graveyard";
        case ZoneKind::EXILE:
            return "exile";
    }
    return "";
}

int ZoneList::listed() const {
    int n = 0;
    for (const ZoneCard &c : cards) n += c.copies;
    return n;
}

ZoneModel build_zone_model(
    const agents::SeatStream &stream, const CardBook &book, const rmstate::DecodedFrame &frame, int viewer) {
    const Frame f(stream, book, frame);
    ZoneModel m;
    for (int s = 0; s < 2; ++s)
        for (int z = 0; z < ZONE_KINDS; ++z) {
            m.zones[static_cast<size_t>(s)][static_cast<size_t>(z)].seat = s;
            m.zones[static_cast<size_t>(s)][static_cast<size_t>(z)].kind = static_cast<ZoneKind>(z);
        }
    const agents::Table &p = f.player;
    for (size_t r = 0; r < p.rows(); ++r) {
        const int s = seat_of(p.u(r, p.col("player")));
        if (s < 0) continue;
        auto &zs = m.zones[static_cast<size_t>(s)];
        zs[static_cast<size_t>(ZoneKind::HAND)].size = static_cast<int>(p.u(r, p.col("hand")));
        zs[static_cast<size_t>(ZoneKind::LIBRARY)].size = static_cast<int>(p.u(r, p.col("library")));
        zs[static_cast<size_t>(ZoneKind::GRAVEYARD)].size = static_cast<int>(p.u(r, p.col("graveyard")));
        zs[static_cast<size_t>(ZoneKind::EXILE)].size = static_cast<int>(p.u(r, p.col("exile")));
    }
    const agents::Table &o = f.obj;
    const int c_zone = o.col("zone"), c_owner = o.col("owner");
    for (size_t r = 0; r < o.rows(); ++r) {
        const int zone = static_cast<int>(o.u(r, c_zone));
        const int owner = seat_of(o.u(r, c_owner));
        if (owner < 0) continue;
        ZoneKind kind;
        if (zone == f.z_hand) {
            kind = ZoneKind::HAND;
        } else if (zone == f.z_library) {
            kind = ZoneKind::LIBRARY;
        } else if (zone == f.z_graveyard) {
            kind = ZoneKind::GRAVEYARD;
        } else if (zone == f.z_exile) {
            kind = ZoneKind::EXILE;
        } else {
            continue;
        }
        ZoneCard c = read_card(f, r);
        // An opponent's hand card a seat frame lists is one it sees now (a reveal).
        if (kind == ZoneKind::HAND && viewer < 2 && owner != viewer) c.note = "revealed";
        if (kind == ZoneKind::LIBRARY && viewer < 2) c.note = window_note(f, c.handle);
        m.zones[static_cast<size_t>(owner)][static_cast<size_t>(kind)].cards.push_back(c);
    }
    for (auto &zs : m.zones)
        for (ZoneList &z : zs)
            if (z.kind != ZoneKind::HAND)
                std::stable_sort(
                    z.cards.begin(), z.cards.end(), [](const ZoneCard &a, const ZoneCard &b) { return a.pos < b.pos; });
    std::map<uint64_t, ZoneCard *> by_handle;
    for (auto &zs : m.zones)
        for (ZoneList &z : zs)
            for (ZoneCard &c : z.cards) by_handle[c.handle] = &c;
    read_counters(f, by_handle);
    read_links(f, by_handle);
    if (viewer < 2) {
        add_known_positions(f, 0, &m.zones[0][static_cast<size_t>(ZoneKind::LIBRARY)]);
        add_known_positions(f, 1, &m.zones[1][static_cast<size_t>(ZoneKind::LIBRARY)]);
        for (auto &zs : m.zones)
            std::stable_sort(zs[static_cast<size_t>(ZoneKind::LIBRARY)].cards.begin(),
                zs[static_cast<size_t>(ZoneKind::LIBRARY)].cards.end(),
                [](const ZoneCard &a, const ZoneCard &b) { return a.pos < b.pos; });
        add_known_hand(f, &m.zones[static_cast<size_t>(1 - viewer)][static_cast<size_t>(ZoneKind::HAND)]);
    }
    for (auto &zs : m.zones)
        for (ZoneList &z : zs) finish(&z);
    return m;
}

}  // namespace rmwork
