#include "k_rows.h"

namespace knowledge {

static void seat_pair(rmstate::FrameWriter::Row &r, const Codes &codes, int player, int viewer);
static void libpos(rmstate::FrameWriter::Row &r, const LibPos &p);
static uint64_t card16(int card);

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

}  // namespace knowledge
