#include "frame_view.h"

#include <algorithm>

namespace knowledge {

namespace {

// One decoded table's columns found by name. A table the frame does not carry has no rows; a
// column the schema lacks is recorded, so the reader can refuse the frame.
class TableReader {
    public:
        TableReader(const rmstate::DecodedFrame &frame, rmstate::TableId id)
            : table_(frame.table(static_cast<uint16_t>(id))),
              def_(rmstate::find_table_def(static_cast<uint16_t>(id))) {}
        size_t rows() const { return table_ ? table_->rows.size() : 0; }
        int col(const char *name) {
            const rmstate::TableDef *def = table_ ? table_->def : def_;
            if (def)
                for (size_t i = 0; i < def->columns.size(); ++i)
                    if (def->columns[i].name == name) return static_cast<int>(i);
            missing_ += std::string(missing_.empty() ? "" : ", ") + name;
            return -1;
        }
        uint64_t u(size_t row, int col) const {
            if (!table_ || col < 0) return 0;
            return table_->rows[row][static_cast<size_t>(col)].bits;
        }
        const std::string &missing() const { return missing_; }

    private:
        const rmstate::DecodedTable *table_;
        const rmstate::TableDef *def_;
        std::string missing_;
};

// A libpos column triple.
struct LibPosCols {
        int anchor = -1, pos = -1, span = -1;
};

}  // namespace

static Zn zone_value(const Codes &codes, uint64_t v);
static int seat_value(const Codes &codes, uint64_t v);
static int card_value(uint64_t v, uint32_t width);
static LibPosCols libpos_cols(TableReader &t, const std::string &name);
static LibPos read_libpos(const TableReader &t, size_t row, const LibPosCols &c);
static bool missing_columns(const TableReader &t, const char *table, std::string *err);
static bool read_events(const rmstate::DecodedFrame &frame, const Codes &codes, Snapshot *out, std::string *err);
static bool read_snapshot_facts(
    const rmstate::DecodedFrame &frame, const Codes &codes, Snapshot *out, std::string *err);

bool is_public_zone(Zn z) {
    return z == Zn::BATTLEFIELD || z == Zn::STACK || z == Zn::GRAVEYARD || z == Zn::EXILE;
}

bool peek_frame_header(const std::string &payload, rmstate::FrameKind *kind, rmstate::Viewer *viewer) {
    if (payload.size() < rmstate::FRAME_HEADER_SIZE) return false;
    uint32_t magic = 0;
    for (int i = 0; i < 4; ++i) magic |= static_cast<uint32_t>(static_cast<unsigned char>(payload[i])) << (8 * i);
    if (magic != rmstate::FRAME_MAGIC) return false;
    *kind = static_cast<rmstate::FrameKind>(static_cast<unsigned char>(payload[6]));
    *viewer = static_cast<rmstate::Viewer>(static_cast<unsigned char>(payload[7]));
    return true;
}

bool read_codes(const rmstate::Schema &schema, Codes *out, std::string *err) {
    static const std::pair<const char *, Zn> zone_names[] = {{"LIBRARY", Zn::LIBRARY}, {"BATTLEFIELD", Zn::BATTLEFIELD},
        {"HAND", Zn::HAND}, {"STACK", Zn::STACK}, {"GRAVEYARD", Zn::GRAVEYARD}, {"EXILE", Zn::EXILE},
        {"SIDEBOARD", Zn::SIDEBOARD}};
    const rmstate::EnumTable *zone = schema.find_enum("zone");
    const rmstate::EnumTable *seat = schema.find_enum("seat");
    if (!zone || !seat) {
        *err = "the schema has no zone or seat enum";
        return false;
    }
    out->zone_of.assign(zone->values.size(), Zn::OTHER);
    for (size_t v = 0; v < zone->values.size(); ++v)
        for (const auto &zn : zone_names)
            if (zone->values[v] == zn.first) out->zone_of[v] = zn.second;
    bool found[3] = {false, false, false};
    for (size_t v = 0; v < seat->values.size(); ++v) {
        if (seat->values[v] == "PLAYER_A") out->seat_value[0] = static_cast<uint8_t>(v), found[0] = true;
        if (seat->values[v] == "PLAYER_B") out->seat_value[1] = static_cast<uint8_t>(v), found[1] = true;
        if (seat->values[v] == "UNKNOWN") out->seat_none = static_cast<uint8_t>(v), found[2] = true;
    }
    if (!found[0] || !found[1] || !found[2]) {
        *err = "the seat enum lacks PLAYER_A, PLAYER_B or UNKNOWN";
        return false;
    }
    return true;
}

static Zn zone_value(const Codes &codes, uint64_t v) {
    if (v == rmstate::NONE8) return Zn::NONE;
    return v < codes.zone_of.size() ? codes.zone_of[v] : Zn::OTHER;
}

static int seat_value(const Codes &codes, uint64_t v) {
    if (v == codes.seat_value[0]) return 0;
    if (v == codes.seat_value[1]) return 1;
    return NO_SEAT;
}

static int card_value(uint64_t v, uint32_t width) {
    const uint64_t ones = width >= 8 ? ~0ull : ((1ull << (8 * width)) - 1);
    return v == ones ? NO_CARD : static_cast<int>(v);
}

static LibPosCols libpos_cols(TableReader &t, const std::string &name) {
    LibPosCols c;
    c.anchor = t.col((name + "_anchor").c_str());
    c.pos = t.col((name + "_pos").c_str());
    c.span = t.col((name + "_span").c_str());
    return c;
}

static LibPos read_libpos(const TableReader &t, size_t row, const LibPosCols &c) {
    LibPos p;
    p.anchor = static_cast<uint8_t>(t.u(row, c.anchor));
    p.pos = static_cast<uint16_t>(t.u(row, c.pos));
    p.span = static_cast<uint16_t>(t.u(row, c.span));
    return p;
}

static bool missing_columns(const TableReader &t, const char *table, std::string *err) {
    if (t.missing().empty()) return false;
    *err = std::string("table ") + table + " lacks column(s) " + t.missing();
    return true;
}

bool read_snapshot(const rmstate::DecodedFrame &frame, const Codes &codes, Snapshot *out, std::string *err) {
    *out = Snapshot{};
    return read_snapshot_facts(frame, codes, out, err) && read_events(frame, codes, out, err);
}

static bool read_snapshot_facts(
    const rmstate::DecodedFrame &frame, const Codes &codes, Snapshot *out, std::string *err) {
    using rmstate::TableId;
    TableReader h(frame, TableId::HEADER);
    const int h_game = h.col("game"), h_decision = h.col("decision"), h_terminal = h.col("terminal");
    if (missing_columns(h, "header", err)) return false;
    if (h.rows()) {
        out->game = static_cast<uint8_t>(h.u(0, h_game));
        out->decision = static_cast<uint32_t>(h.u(0, h_decision));
        out->terminal = h.u(0, h_terminal) != 0;
    }

    TableReader m(frame, TableId::MATCH);
    const int m_sb = m.col("sideboard_phase"), m_closed = m.col("closed_decklist");
    if (missing_columns(m, "match", err)) return false;
    if (m.rows()) {
        out->sideboard_phase = m.u(0, m_sb) != 0;
        out->closed_decklist = m.u(0, m_closed) != 0;
    }

    TableReader p(frame, TableId::PLAYER);
    const int p_player = p.col("player"), p_hand = p.col("hand"), p_library = p.col("library"),
              p_exile = p.col("exile"), p_sideboard = p.col("sideboard");
    if (missing_columns(p, "player", err)) return false;
    for (size_t r = 0; r < p.rows(); ++r) {
        SnapPlayer sp;
        sp.player = seat_value(codes, p.u(r, p_player));
        sp.hand = static_cast<uint16_t>(p.u(r, p_hand));
        sp.library = static_cast<uint16_t>(p.u(r, p_library));
        sp.exile = static_cast<uint16_t>(p.u(r, p_exile));
        sp.sideboard = static_cast<uint16_t>(p.u(r, p_sideboard));
        out->players.push_back(sp);
    }

    TableReader o(frame, TableId::OBJECT);
    const int o_handle = o.col("handle"), o_zone = o.col("zone"), o_owner = o.col("owner"), o_pos = o.col("pos"),
              o_card = o.col("card"), o_token = o.col("token"), o_face_down = o.col("face_down");
    if (missing_columns(o, "object", err)) return false;
    for (size_t r = 0; r < o.rows(); ++r) {
        SnapObject so;
        so.handle = static_cast<uint32_t>(o.u(r, o_handle));
        so.zone = zone_value(codes, o.u(r, o_zone));
        so.owner = seat_value(codes, o.u(r, o_owner));
        so.pos = static_cast<uint16_t>(o.u(r, o_pos));
        so.card = card_value(o.u(r, o_card), 2);
        so.token = o.u(r, o_token) != 0;
        so.face_down = o.u(r, o_face_down) != 0;
        out->objects.push_back(so);
    }

    TableReader s(frame, TableId::STACK);
    const int s_handle = s.col("handle"), s_kind = s.col("kind"), s_copy = s.col("copy");
    if (missing_columns(s, "stack", err)) return false;
    for (size_t r = 0; r < s.rows(); ++r) {
        const bool spell = s.u(r, s_kind) == static_cast<uint64_t>(rmstate::StackKind::SPELL);
        if (!spell || s.u(r, s_copy) != 0) out->stack_non_cards.push_back(static_cast<uint32_t>(s.u(r, s_handle)));
    }
    std::sort(out->stack_non_cards.begin(), out->stack_non_cards.end());

    TableReader v(frame, TableId::REVEAL);
    const int v_handle = v.col("handle"), v_zone = v.col("zone"), v_owner = v.col("owner"), v_pos = v.col("pos"),
              v_card = v.col("card");
    if (missing_columns(v, "reveal", err)) return false;
    for (size_t r = 0; r < v.rows(); ++r) {
        SnapReveal sr;
        sr.handle = static_cast<uint32_t>(v.u(r, v_handle));
        sr.zone = zone_value(codes, v.u(r, v_zone));
        sr.owner = seat_value(codes, v.u(r, v_owner));
        sr.pos = static_cast<uint16_t>(v.u(r, v_pos));
        sr.card = card_value(v.u(r, v_card), 2);
        out->reveals.push_back(sr);
    }

    TableReader d(frame, TableId::OWN_DECK);
    const int d_player = d.col("player"), d_card = d.col("card"), d_main = d.col("main"), d_side = d.col("side");
    if (missing_columns(d, "own_deck", err)) return false;
    for (size_t r = 0; r < d.rows(); ++r) {
        DeckEntry e;
        e.player = seat_value(codes, d.u(r, d_player));
        e.card = card_value(d.u(r, d_card), 2);
        e.main = static_cast<int>(d.u(r, d_main));
        e.side = static_cast<int>(d.u(r, d_side));
        out->own_deck.push_back(e);
    }
    return true;
}

static bool read_events(const rmstate::DecodedFrame &frame, const Codes &codes, Snapshot *out, std::string *err) {
    using rmstate::TableId;
    TableReader e(frame, TableId::EVENT);
    const int seq = e.col("seq"), kind = e.col("kind"), player = e.col("player"), zone = e.col("zone"),
              to_zone = e.col("to_zone"), zone_owner = e.col("zone_owner"), to_owner = e.col("to_owner"),
              count = e.col("count"), order_known = e.col("order_known"), batch = e.col("batch"), to = e.col("to"),
              source = e.col("source"), source_card = e.col("source_card"), turn = e.col("turn"),
              opening = e.col("opening"), game = e.col("game");
    const LibPosCols range = libpos_cols(e, "range");
    if (missing_columns(e, "event", err)) return false;
    out->events.resize(e.rows());
    for (size_t r = 0; r < e.rows(); ++r) {
        Event &ev = out->events[r];
        ev.seq = static_cast<uint32_t>(e.u(r, seq));
        ev.kind = static_cast<rmstate::EventKind>(e.u(r, kind));
        ev.player = seat_value(codes, e.u(r, player));
        ev.zone = zone_value(codes, e.u(r, zone));
        ev.to_zone = zone_value(codes, e.u(r, to_zone));
        ev.zone_owner = seat_value(codes, e.u(r, zone_owner));
        ev.to_owner = seat_value(codes, e.u(r, to_owner));
        ev.count = static_cast<uint16_t>(e.u(r, count));
        ev.range = read_libpos(e, r, range);
        ev.order_known = e.u(r, order_known) != 0;
        ev.batch = static_cast<uint16_t>(e.u(r, batch));
        ev.to = seat_value(codes, e.u(r, to));
        ev.source = static_cast<uint32_t>(e.u(r, source));
        ev.source_card = card_value(e.u(r, source_card), 2);
        ev.turn = static_cast<uint16_t>(e.u(r, turn));
        ev.opening = e.u(r, opening) != 0;
        ev.game = static_cast<uint8_t>(e.u(r, game));
    }

    TableReader c(frame, TableId::EVENT_CARD);
    const int c_event = c.col("event"), hb = c.col("handle_before"), ha = c.col("handle_after"), card = c.col("card"),
              face_down = c.col("face_down"), token = c.col("token"), hand_subset = c.col("hand_subset"),
              draw_seq = c.col("draw_seq"), draw_idx = c.col("draw_idx");
    const LibPosCols from = libpos_cols(c, "from"), to_pos = libpos_cols(c, "to");
    if (missing_columns(c, "event_card", err)) return false;
    for (size_t r = 0; r < c.rows(); ++r) {
        const uint64_t ev = c.u(r, c_event);
        if (ev >= out->events.size()) {
            *err = "an event_card row names event row " + std::to_string(ev) + " past the event table";
            return false;
        }
        EventCard ec;
        ec.handle_before = static_cast<uint32_t>(c.u(r, hb));
        ec.handle_after = static_cast<uint32_t>(c.u(r, ha));
        ec.card = card_value(c.u(r, card), 2);
        ec.from = read_libpos(c, r, from);
        ec.to = read_libpos(c, r, to_pos);
        ec.face_down = c.u(r, face_down) != 0;
        ec.token = c.u(r, token) != 0;
        ec.hand_subset = static_cast<uint8_t>(c.u(r, hand_subset));
        ec.draw_seq = static_cast<uint32_t>(c.u(r, draw_seq));
        ec.draw_idx = static_cast<uint16_t>(c.u(r, draw_idx));
        out->events[ev].cards.push_back(ec);
    }

    TableReader p(frame, TableId::EVENT_PLAYER);
    const int p_event = p.col("event"), p_player = p.col("player"), p_library = p.col("library"),
              p_sideboard = p.col("sideboard"), p_wins = p.col("wins");
    if (missing_columns(p, "event_player", err)) return false;
    for (size_t r = 0; r < p.rows(); ++r) {
        const uint64_t ev = p.u(r, p_event);
        if (ev >= out->events.size()) {
            *err = "an event_player row names event row " + std::to_string(ev) + " past the event table";
            return false;
        }
        EventPlayer ep;
        ep.player = seat_value(codes, p.u(r, p_player));
        ep.library = static_cast<uint16_t>(p.u(r, p_library));
        ep.sideboard = static_cast<uint16_t>(p.u(r, p_sideboard));
        ep.wins = static_cast<uint8_t>(p.u(r, p_wins));
        out->events[ev].players.push_back(ep);
    }
    return true;
}

bool read_decklists(const rmstate::DecodedFrame &frame, const Codes &codes, Decklists *out, std::string *err) {
    using rmstate::TableId;
    *out = Decklists{};
    TableReader d(frame, TableId::DECKLIST);
    const int d_player = d.col("player");
    if (missing_columns(d, "decklist", err)) return false;
    for (size_t r = 0; r < d.rows(); ++r) {
        const int seat = seat_value(codes, d.u(r, d_player));
        if (seat != NO_SEAT) out->present[seat] = true;
    }
    TableReader c(frame, TableId::DECKLIST_CARD);
    const int c_player = c.col("player"), c_card = c.col("card"), c_main = c.col("main"), c_side = c.col("side");
    if (missing_columns(c, "decklist_card", err)) return false;
    for (size_t r = 0; r < c.rows(); ++r) {
        DeckEntry e;
        e.player = seat_value(codes, c.u(r, c_player));
        e.card = card_value(c.u(r, c_card), 2);
        e.main = static_cast<int>(c.u(r, c_main));
        e.side = static_cast<int>(c.u(r, c_side));
        out->cards.push_back(e);
    }
    return true;
}

bool read_card_names(const rmstate::DecodedFrame &frame, std::map<int, std::string> *out, std::string *err) {
    using rmstate::TableId;
    const rmstate::DecodedTable *t = frame.table(static_cast<uint16_t>(TableId::CARD));
    if (!t) return true;
    int c_card = -1, c_face = -1, c_name = -1;
    for (size_t i = 0; i < t->def->columns.size(); ++i) {
        const std::string &n = t->def->columns[i].name;
        if (n == "card") c_card = static_cast<int>(i);
        if (n == "face") c_face = static_cast<int>(i);
        if (n == "name") c_name = static_cast<int>(i);
    }
    if (c_card < 0 || c_face < 0 || c_name < 0) {
        *err = "table card lacks card, face or name";
        return false;
    }
    for (const auto &row : t->rows) {
        if (row[static_cast<size_t>(c_face)].bits != static_cast<uint64_t>(rmstate::Face::FRONT)) continue;
        (*out)[static_cast<int>(row[static_cast<size_t>(c_card)].bits)] = row[static_cast<size_t>(c_name)].text;
    }
    return true;
}

}  // namespace knowledge
