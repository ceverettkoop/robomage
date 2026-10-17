#include "k_rows.h"

namespace knowledge {

static void seat_pair(rmstate::FrameWriter::Row &r, const Codes &codes, int player, int viewer);
static void libpos(rmstate::FrameWriter::Row &r, const LibPos &p);
static uint64_t card16(int card);
static int column_of(const rmstate::DecodedTable *t, const char *name, std::string *missing);
static uint64_t cell(const rmstate::DecodedTable *t, size_t row, int col);
static int seat_of(const Codes &codes, uint64_t v);
static int card_of(uint64_t v);
static LibPos libpos_of(const rmstate::DecodedTable *t, size_t row, const int cols[3]);

static void seat_pair(rmstate::FrameWriter::Row &r, const Codes &codes, int player, int viewer) {
    if (player != 0 && player != 1) {
        r.u8(codes.seat_none).u8(0);
        return;
    }
    r.u8(codes.seat_value[player]).u8(player == viewer ? 1 : 2);
}

static void libpos(rmstate::FrameWriter::Row &r, const LibPos &p) {
    r.u8(p.anchor).u16(p.pos).u16(p.span);
}

static uint64_t card16(int card) {
    return card < 0 ? rmstate::NONE16 : static_cast<uint64_t>(card);
}

void write_k_tables(rmstate::FrameWriter &w, const KRows &rows, const Codes &codes, int viewer) {
    using rmstate::TableId;
    w.row(TableId::K_META).u8(static_cast<uint8_t>(rows.meta.level)).u8(rows.meta.exact ? 1 : 0).u32(rows.meta.steps);
    for (const KSeenRow &s : rows.seen) {
        rmstate::FrameWriter::Row r = w.row(TableId::K_SEEN);
        seat_pair(r, codes, s.player, viewer);
        r.u16(card16(s.card)).u8(s.game).u8(s.match);
    }
    for (const KPoolRow &p : rows.pools) {
        rmstate::FrameWriter::Row r = w.row(TableId::K_POOL);
        seat_pair(r, codes, p.owner, viewer);
        r.u8(static_cast<uint8_t>(p.kind)).u16(p.size);
        libpos(r, p.range);
        r.u32(p.obj);
    }
    for (const KPoolCardRow &c : rows.pool_cards)
        w.row(TableId::K_POOL_CARD).u16(c.pool).u16(card16(c.card)).u8(c.min).u8(c.max);
    for (const KPositionRow &p : rows.positions) {
        rmstate::FrameWriter::Row r = w.row(TableId::K_POSITION);
        seat_pair(r, codes, p.owner, viewer);
        libpos(r, p.at);
        r.u16(card16(p.card)).u16(p.pool);
    }
    for (const KKnownHandRow &k : rows.known_hand) w.row(TableId::K_KNOWN_HAND).u16(card16(k.card)).u8(k.copies);
}

// ── Reading the tables back ───────────────────────────────────────────────────────────────────

// Column `name` of `t` (a table the frame carries), else -1 with the name added to `missing`.
static int column_of(const rmstate::DecodedTable *t, const char *name, std::string *missing) {
    if (!t) return -1;
    for (size_t i = 0; i < t->def->columns.size(); ++i)
        if (t->def->columns[i].name == name) return static_cast<int>(i);
    *missing += std::string(missing->empty() ? "" : ", ") + name;
    return -1;
}

static uint64_t cell(const rmstate::DecodedTable *t, size_t row, int col) {
    return col < 0 ? 0 : t->rows[row][static_cast<size_t>(col)].bits;
}

static int seat_of(const Codes &codes, uint64_t v) {
    if (v == codes.seat_value[0]) return 0;
    if (v == codes.seat_value[1]) return 1;
    return NO_SEAT;
}

static int card_of(uint64_t v) {
    return v == rmstate::NONE16 ? NO_CARD : static_cast<int>(v);
}

static LibPos libpos_of(const rmstate::DecodedTable *t, size_t row, const int cols[3]) {
    LibPos p;
    p.anchor = static_cast<uint8_t>(cell(t, row, cols[0]));
    p.pos = static_cast<uint16_t>(cell(t, row, cols[1]));
    p.span = static_cast<uint16_t>(cell(t, row, cols[2]));
    return p;
}

bool read_k_tables(const rmstate::DecodedFrame &frame, const Codes &codes, KRows *out, std::string *err) {
    using rmstate::TableId;
    *out = KRows{};
    std::string missing;
    const auto table = [&frame](TableId id) { return frame.table(static_cast<uint16_t>(id)); };

    const rmstate::DecodedTable *m = table(TableId::K_META);
    if (!m || m->rows.size() != 1) {
        *err = "the frame has no single k_meta row";
        return false;
    }
    const int m_level = column_of(m, "level", &missing), m_exact = column_of(m, "exact", &missing),
              m_steps = column_of(m, "steps", &missing);
    out->meta.level = static_cast<rmstate::KnowledgeLevel>(cell(m, 0, m_level));
    out->meta.exact = cell(m, 0, m_exact) != 0;
    out->meta.steps = static_cast<uint32_t>(cell(m, 0, m_steps));

    if (const rmstate::DecodedTable *t = table(TableId::K_SEEN)) {
        const int player = column_of(t, "player", &missing), card = column_of(t, "card", &missing),
                  game = column_of(t, "game", &missing), match = column_of(t, "match", &missing);
        for (size_t r = 0; r < t->rows.size(); ++r) {
            KSeenRow row;
            row.player = seat_of(codes, cell(t, r, player));
            row.card = card_of(cell(t, r, card));
            row.game = static_cast<uint8_t>(cell(t, r, game));
            row.match = static_cast<uint8_t>(cell(t, r, match));
            out->seen.push_back(row);
        }
    }
    if (const rmstate::DecodedTable *t = table(TableId::K_POOL)) {
        const int owner = column_of(t, "owner", &missing), kind = column_of(t, "kind", &missing),
                  size = column_of(t, "size", &missing), obj = column_of(t, "obj", &missing);
        const int range[3] = {column_of(t, "range_anchor", &missing), column_of(t, "range_pos", &missing),
            column_of(t, "range_span", &missing)};
        for (size_t r = 0; r < t->rows.size(); ++r) {
            KPoolRow row;
            row.owner = seat_of(codes, cell(t, r, owner));
            row.kind = static_cast<rmstate::PoolKind>(cell(t, r, kind));
            row.size = static_cast<uint16_t>(cell(t, r, size));
            row.range = libpos_of(t, r, range);
            row.obj = static_cast<uint32_t>(cell(t, r, obj));
            out->pools.push_back(row);
        }
    }
    if (const rmstate::DecodedTable *t = table(TableId::K_POOL_CARD)) {
        const int pool = column_of(t, "pool", &missing), card = column_of(t, "card", &missing),
                  min = column_of(t, "min", &missing), max = column_of(t, "max", &missing);
        for (size_t r = 0; r < t->rows.size(); ++r) {
            KPoolCardRow row;
            row.pool = static_cast<uint16_t>(cell(t, r, pool));
            row.card = card_of(cell(t, r, card));
            row.min = static_cast<uint8_t>(cell(t, r, min));
            row.max = static_cast<uint8_t>(cell(t, r, max));
            out->pool_cards.push_back(row);
        }
    }
    if (const rmstate::DecodedTable *t = table(TableId::K_POSITION)) {
        const int owner = column_of(t, "owner", &missing), card = column_of(t, "card", &missing),
                  pool = column_of(t, "pool", &missing);
        const int at[3] = {
            column_of(t, "at_anchor", &missing), column_of(t, "at_pos", &missing), column_of(t, "at_span", &missing)};
        for (size_t r = 0; r < t->rows.size(); ++r) {
            KPositionRow row;
            row.owner = seat_of(codes, cell(t, r, owner));
            row.at = libpos_of(t, r, at);
            row.card = card_of(cell(t, r, card));
            row.pool = static_cast<uint16_t>(cell(t, r, pool));
            out->positions.push_back(row);
        }
    }
    if (const rmstate::DecodedTable *t = table(TableId::K_KNOWN_HAND)) {
        const int card = column_of(t, "card", &missing), copies = column_of(t, "copies", &missing);
        for (size_t r = 0; r < t->rows.size(); ++r) {
            KKnownHandRow row;
            row.card = card_of(cell(t, r, card));
            row.copies = static_cast<uint8_t>(cell(t, r, copies));
            out->known_hand.push_back(row);
        }
    }
    if (!missing.empty()) {
        *err = "the k_* tables lack the column(s) " + missing;
        return false;
    }
    return true;
}

}  // namespace knowledge
