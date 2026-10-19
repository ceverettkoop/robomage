#include "decode.h"

#include <cstring>
#include <utility>

namespace rmstate {

static uint64_t read_le(const std::string &bytes, size_t at, uint32_t width);
static DecodeResult fail(DecodeStatus status, const std::string &message);
static uint32_t enum_count_frame_kind();
static uint32_t enum_count_viewer();
static uint32_t enum_count_storage();
static DecodeResult resolve_text(const std::string &pool, uint64_t offset, std::string *out);
static DecodeResult decode_rows(const std::string &payload, size_t at, uint32_t rows, const TableDef &def,
    const std::string &pool, DecodedTable *out);
static int column_index(const TableDef &def, const char *name);
static uint64_t cell(const DecodedTable &t, size_t row, int col);
static const std::string &cell_text(const DecodedTable &t, size_t row, int col);
static bool same_columns(const TableDef &a, const TableDef &b);
static DecodeResult check_table_layout(const TableDef &def);
static DecodeResult read_schema_tables(const DecodedFrame &f, Schema *out);
static DecodeResult read_schema_enums(const DecodedFrame &f, Schema *out);
static DecodeResult check_bootstrap_agrees(const Schema &schema);
static DecodeResult encodable(const DecodedFrame &frame, const DecodedTable &t);
static void write_value(FrameWriter::Row &r, Storage storage, const Value &v);
static DecodeResult scan_blocks(const std::string &payload, std::vector<std::pair<uint16_t, size_t>> *blocks);
static void write_le(std::string *bytes, size_t at, uint32_t width, uint64_t v);

// ── Small helpers ─────────────────────────────────────────────────────────────────────────────

static uint64_t read_le(const std::string &bytes, size_t at, uint32_t width) {
    const unsigned char *p = reinterpret_cast<const unsigned char *>(bytes.data()) + at;
    uint64_t v = 0;
    for (uint32_t i = 0; i < width; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
    return v;
}

static DecodeResult fail(DecodeStatus status, const std::string &message) {
    DecodeResult r;
    r.status = status;
    r.message = message;
    return r;
}

#define RMSTATE_COUNT_ONE(name) +1
static uint32_t enum_count_frame_kind() {
    return 0 RMSTATE_FRAME_KIND(RMSTATE_COUNT_ONE);
}
static uint32_t enum_count_viewer() {
    return 0 RMSTATE_VIEWER(RMSTATE_COUNT_ONE);
}
static uint32_t enum_count_storage() {
    return 0 RMSTATE_STORAGE(RMSTATE_COUNT_ONE);
}
#undef RMSTATE_COUNT_ONE

int64_t value_signed(const Value &v, Storage storage) {
    switch (storage) {
        case Storage::I8:
            return static_cast<int8_t>(static_cast<uint8_t>(v.bits));
        case Storage::I16:
            return static_cast<int16_t>(static_cast<uint16_t>(v.bits));
        case Storage::I32:
            return static_cast<int32_t>(static_cast<uint32_t>(v.bits));
        case Storage::I64:
            return static_cast<int64_t>(v.bits);
        default:
            return static_cast<int64_t>(v.bits);
    }
}

bool value_is_all_ones(const Value &v, Storage storage) {
    const uint32_t width = storage_width(storage);
    const uint64_t ones = width >= 8 ? ~0ull : ((1ull << (8 * width)) - 1);
    return v.bits == ones;
}

// The pooled string at `offset`: it must be the start of a NUL-terminated string in the pool.
static DecodeResult resolve_text(const std::string &pool, uint64_t offset, std::string *out) {
    if (offset >= pool.size()) return fail(DecodeStatus::MALFORMED, "text offset past the string pool");
    if (offset > 0 && pool[offset - 1] != '\0')
        return fail(DecodeStatus::MALFORMED, "text offset is not the start of a pooled string");
    const size_t end = pool.find('\0', offset);
    if (end == std::string::npos) return fail(DecodeStatus::MALFORMED, "unterminated pooled string");
    out->assign(pool, offset, end - offset);
    return DecodeResult();
}

// The `rows` rows of table `def` laid out from byte `at` of `payload`.
static DecodeResult decode_rows(const std::string &payload, size_t at, uint32_t rows, const TableDef &def,
    const std::string &pool, DecodedTable *out) {
    out->id = static_cast<uint16_t>(def.id);
    out->def = &def;
    std::vector<uint32_t> width(def.columns.size());
    for (size_t c = 0; c < def.columns.size(); ++c) width[c] = storage_width(def.columns[c].storage);
    out->rows.resize(rows);
    for (uint32_t r = 0; r < rows; ++r) {
        const size_t base = at + static_cast<size_t>(r) * def.row_width;
        std::vector<Value> &row = out->rows[r];
        row.resize(def.columns.size());
        for (size_t c = 0; c < def.columns.size(); ++c) {
            const ColumnDef &col = def.columns[c];
            Value &v = row[c];
            v.bits = read_le(payload, base + col.offset, width[c]);
            if (col.storage != Storage::TEXT) continue;
            DecodeResult res = resolve_text(pool, v.bits, &v.text);
            if (!res.ok()) {
                res.message += " (table " + def.name + ", row " + std::to_string(r) + ", column " + col.name + ")";
                return res;
            }
        }
    }
    return DecodeResult();
}

static int column_index(const TableDef &def, const char *name) {
    for (size_t i = 0; i < def.columns.size(); ++i)
        if (def.columns[i].name == name) return static_cast<int>(i);
    return -1;
}

static uint64_t cell(const DecodedTable &t, size_t row, int col) {
    return t.rows[row][static_cast<size_t>(col)].bits;
}

static const std::string &cell_text(const DecodedTable &t, size_t row, int col) {
    return t.rows[row][static_cast<size_t>(col)].text;
}

static bool same_columns(const TableDef &a, const TableDef &b) {
    if (a.columns.size() != b.columns.size() || a.row_width != b.row_width) return false;
    for (size_t i = 0; i < a.columns.size(); ++i) {
        const ColumnDef &x = a.columns[i];
        const ColumnDef &y = b.columns[i];
        if (x.name != y.name || x.storage != y.storage || x.offset != y.offset || x.tag != y.tag ||
            x.has_range != y.has_range || x.min != y.min || x.max != y.max)
            return false;
    }
    return true;
}

// Columns packed in order with no padding, adding up to the row width.
static DecodeResult check_table_layout(const TableDef &def) {
    uint32_t off = 0;
    for (const ColumnDef &c : def.columns) {
        if (c.offset != off)
            return fail(DecodeStatus::BAD_SCHEMA, "column " + c.name + " of table " + def.name + " is not packed");
        off += storage_width(c.storage);
    }
    if (off != def.row_width)
        return fail(DecodeStatus::BAD_SCHEMA, "columns of table " + def.name + " do not add up to its row width");
    return DecodeResult();
}

// ── Schema ────────────────────────────────────────────────────────────────────────────────────

const TableDef *Schema::find_table(uint16_t id) const {
    for (const TableDef &t : tables)
        if (static_cast<uint16_t>(t.id) == id) return &t;
    return nullptr;
}

const EnumTable *Schema::find_enum(const std::string &name) const {
    for (const EnumTable &e : enums)
        if (e.name == name) return &e;
    return nullptr;
}

const DecodedTable *DecodedFrame::table(uint16_t id) const {
    for (const DecodedTable &t : tables)
        if (t.id == id) return &t;
    return nullptr;
}

// ── Frames ────────────────────────────────────────────────────────────────────────────────────

DecodeResult decode_frame(const std::string &payload, const Schema &schema, DecodedFrame *out) {
    *out = DecodedFrame();
    if (payload.size() < FRAME_HEADER_SIZE) return fail(DecodeStatus::MALFORMED, "frame shorter than its header");
    if (read_le(payload, 0, 4) != FRAME_MAGIC) return fail(DecodeStatus::BAD_MAGIC, "not an RMSF frame");
    FrameHeader &h = out->header;
    h.version = static_cast<uint16_t>(read_le(payload, 4, 2));
    h.kind = static_cast<uint8_t>(read_le(payload, 6, 1));
    h.viewer = static_cast<uint8_t>(read_le(payload, 7, 1));
    h.schema_hash = read_le(payload, 8, 8);
    h.n_tables = static_cast<uint32_t>(read_le(payload, 16, 4));
    if (h.version != FORMAT_VERSION || h.version != schema.version)
        return fail(DecodeStatus::BAD_VERSION,
            "format version " + std::to_string(h.version) + ", expected " + std::to_string(schema.version));
    if (h.kind >= enum_count_frame_kind())
        return fail(DecodeStatus::BAD_KIND, "unknown frame kind " + std::to_string(h.kind));
    if (h.viewer >= enum_count_viewer())
        return fail(DecodeStatus::BAD_KIND, "unknown viewer " + std::to_string(h.viewer));
    if (h.schema_hash != schema.hash) return fail(DecodeStatus::SCHEMA_MISMATCH, "schema hash differs");
    if (read_le(payload, 20, 4) != 0) return fail(DecodeStatus::MALFORMED, "nonzero reserved header bytes");

    // Table blocks: ids ascending, rows padded to ROW_ALIGN, nothing after the last block.
    struct Block {
            const TableDef *def;
            uint32_t rows;
            size_t at;  // its first row's offset in the payload
    };
    std::vector<Block> blocks;
    size_t off = FRAME_HEADER_SIZE;
    int last_id = -1;
    bool have_pool = false;
    for (uint32_t i = 0; i < h.n_tables; ++i) {
        if (payload.size() - off < TABLE_HEADER_SIZE) return fail(DecodeStatus::MALFORMED, "truncated table header");
        const uint16_t id = static_cast<uint16_t>(read_le(payload, off, 2));
        const uint32_t width = static_cast<uint32_t>(read_le(payload, off + 4, 4));
        const uint32_t rows = static_cast<uint32_t>(read_le(payload, off + 8, 4));
        if (read_le(payload, off + 2, 2) != 0 || read_le(payload, off + 12, 4) != 0)
            return fail(DecodeStatus::MALFORMED, "nonzero reserved bytes in table " + std::to_string(id));
        if (static_cast<int>(id) <= last_id) return fail(DecodeStatus::MALFORMED, "table blocks out of id order");
        last_id = id;
        if (rows == 0) return fail(DecodeStatus::MALFORMED, "empty table block " + std::to_string(id));
        off += TABLE_HEADER_SIZE;
        const uint64_t len = static_cast<uint64_t>(width) * rows;
        if (len > payload.size() - off) return fail(DecodeStatus::MALFORMED, "truncated table " + std::to_string(id));
        const size_t body = off;
        off += static_cast<size_t>(len);
        while (off % ROW_ALIGN != 0) {
            if (off >= payload.size()) return fail(DecodeStatus::MALFORMED, "truncated table padding");
            if (payload[off] != '\0') return fail(DecodeStatus::MALFORMED, "nonzero table padding");
            ++off;
        }
        const TableDef *def = schema.find_table(id);
        if (!def) {
            out->skipped.push_back(id);
            continue;
        }
        if (width != def->row_width)
            return fail(DecodeStatus::SCHEMA_MISMATCH, "table " + def->name + " has row width " +
                                                           std::to_string(width) + ", schema says " +
                                                           std::to_string(def->row_width));
        if (id == static_cast<uint16_t>(TableId::STRINGS)) {
            out->strings.assign(payload, body, static_cast<size_t>(len));
            have_pool = true;
            continue;
        }
        if (static_cast<uint8_t>(def->frame_kind) != h.kind)
            return fail(DecodeStatus::SCHEMA_MISMATCH, "table " + def->name + " in a frame of another kind");
        blocks.push_back({def, rows, body});
    }
    if (off != payload.size()) return fail(DecodeStatus::MALFORMED, "bytes after the last table block");
    if (!have_pool || out->strings.empty() || out->strings[0] != '\0' || out->strings.back() != '\0')
        return fail(DecodeStatus::MALFORMED, "missing or malformed string pool");

    out->tables.reserve(blocks.size());
    for (const Block &b : blocks) {
        out->tables.emplace_back();
        DecodeResult res = decode_rows(payload, b.at, b.rows, *b.def, out->strings, &out->tables.back());
        if (!res.ok()) {
            out->tables.pop_back();
            return res;
        }
    }
    return DecodeResult();
}

// The tables and columns a SCHEMA frame lists (schema_table, schema_column).
static DecodeResult read_schema_tables(const DecodedFrame &f, Schema *out) {
    const DecodedTable *st = f.table(static_cast<uint16_t>(TableId::SCHEMA_TABLE));
    const DecodedTable *sc = f.table(static_cast<uint16_t>(TableId::SCHEMA_COLUMN));
    if (!st || !sc) return fail(DecodeStatus::BAD_SCHEMA, "SCHEMA frame without schema_table / schema_column");
    const int t_id = column_index(*st->def, "id"), t_name = column_index(*st->def, "name"),
              t_kind = column_index(*st->def, "frame_kind"), t_width = column_index(*st->def, "row_width");
    const int c_table = column_index(*sc->def, "table"), c_col = column_index(*sc->def, "col"),
              c_name = column_index(*sc->def, "name"), c_storage = column_index(*sc->def, "storage"),
              c_offset = column_index(*sc->def, "offset"), c_tag = column_index(*sc->def, "tag"),
              c_has_range = column_index(*sc->def, "has_range"), c_min = column_index(*sc->def, "min"),
              c_max = column_index(*sc->def, "max");
    for (size_t r = 0; r < st->rows.size(); ++r) {
        TableDef def;
        const uint64_t id = cell(*st, r, t_id);
        if (!out->tables.empty() && static_cast<uint16_t>(out->tables.back().id) >= id)
            return fail(DecodeStatus::BAD_SCHEMA, "schema_table rows out of id order");
        def.id = static_cast<TableId>(id);
        def.name = cell_text(*st, r, t_name);
        const uint64_t kind = cell(*st, r, t_kind);
        if (kind >= enum_count_frame_kind()) return fail(DecodeStatus::BAD_SCHEMA, "table " + def.name + ": bad kind");
        def.frame_kind = static_cast<FrameKind>(kind);
        def.row_width = static_cast<uint32_t>(cell(*st, r, t_width));
        out->tables.push_back(def);
    }
    for (size_t r = 0; r < sc->rows.size(); ++r) {
        const uint64_t table = cell(*sc, r, c_table);
        TableDef *def = nullptr;
        for (TableDef &t : out->tables)
            if (static_cast<uint16_t>(t.id) == table) def = &t;
        if (!def) return fail(DecodeStatus::BAD_SCHEMA, "schema_column names an unlisted table");
        if (cell(*sc, r, c_col) != def->columns.size())
            return fail(DecodeStatus::BAD_SCHEMA, "schema_column rows of table " + def->name + " out of order");
        ColumnDef c;
        c.name = cell_text(*sc, r, c_name);
        const uint64_t storage = cell(*sc, r, c_storage);
        if (storage >= enum_count_storage())
            return fail(DecodeStatus::BAD_SCHEMA, "column " + c.name + ": bad storage");
        c.storage = static_cast<Storage>(storage);
        c.offset = static_cast<uint32_t>(cell(*sc, r, c_offset));
        c.tag = cell_text(*sc, r, c_tag);
        c.has_range = cell(*sc, r, c_has_range) != 0;
        c.min = value_signed(sc->rows[r][static_cast<size_t>(c_min)], Storage::I64);
        c.max = value_signed(sc->rows[r][static_cast<size_t>(c_max)], Storage::I64);
        def->columns.push_back(c);
    }
    for (const TableDef &t : out->tables) {
        DecodeResult res = check_table_layout(t);
        if (!res.ok()) return res;
    }
    return DecodeResult();
}

// The enum string tables a SCHEMA frame lists (schema_enum): each enum's rows contiguous, values
// 0, 1, 2, … in order.
static DecodeResult read_schema_enums(const DecodedFrame &f, Schema *out) {
    const DecodedTable *se = f.table(static_cast<uint16_t>(TableId::SCHEMA_ENUM));
    if (!se) return DecodeResult();
    const int e_enum = column_index(*se->def, "enum"), e_value = column_index(*se->def, "value"),
              e_name = column_index(*se->def, "name");
    for (size_t r = 0; r < se->rows.size(); ++r) {
        const std::string &name = cell_text(*se, r, e_enum);
        if (out->enums.empty() || out->enums.back().name != name) {
            if (out->find_enum(name)) return fail(DecodeStatus::BAD_SCHEMA, "enum " + name + " is split");
            EnumTable e;
            e.name = name;
            out->enums.push_back(e);
        }
        EnumTable &e = out->enums.back();
        if (cell(*se, r, e_value) != e.values.size())
            return fail(DecodeStatus::BAD_SCHEMA, "enum " + name + " values out of order");
        e.values.push_back(cell_text(*se, r, e_name));
    }
    return DecodeResult();
}

// The SCHEMA frame must describe its own tables (1-4) exactly as the bootstrap read them.
static DecodeResult check_bootstrap_agrees(const Schema &schema) {
    for (TableId id : {TableId::STRINGS, TableId::SCHEMA_TABLE, TableId::SCHEMA_COLUMN, TableId::SCHEMA_ENUM}) {
        const TableDef *listed = schema.find_table(static_cast<uint16_t>(id));
        const TableDef &boot = table_def(id);
        if (!listed || listed->name != boot.name || listed->frame_kind != boot.frame_kind ||
            !same_columns(*listed, boot))
            return fail(DecodeStatus::BAD_SCHEMA, "SCHEMA frame describes table " + boot.name + " differently");
    }
    return DecodeResult();
}

DecodeResult decode_schema_frame(const std::string &payload, Schema *out) {
    *out = Schema();
    if (payload.size() < FRAME_HEADER_SIZE) return fail(DecodeStatus::MALFORMED, "frame shorter than its header");
    if (read_le(payload, 0, 4) != FRAME_MAGIC) return fail(DecodeStatus::BAD_MAGIC, "not an RMSF frame");
    if (read_le(payload, 6, 1) != static_cast<uint8_t>(FrameKind::SCHEMA))
        return fail(DecodeStatus::BAD_KIND, "not a SCHEMA frame");
    const uint64_t hash = read_le(payload, 8, 8);
    if (fnv1a64(payload.data() + FRAME_HEADER_SIZE, payload.size() - FRAME_HEADER_SIZE) != hash)
        return fail(DecodeStatus::SCHEMA_MISMATCH, "SCHEMA frame hash does not match its table blocks");

    // Bootstrap: the SCHEMA frame's own tables, with the layout schema.h defines.
    Schema boot;
    boot.version = FORMAT_VERSION;
    boot.hash = hash;
    for (TableId id : {TableId::STRINGS, TableId::SCHEMA_TABLE, TableId::SCHEMA_COLUMN, TableId::SCHEMA_ENUM})
        boot.tables.push_back(table_def(id));
    DecodedFrame f;
    DecodeResult res = decode_frame(payload, boot, &f);
    if (!res.ok()) return res;
    if (!f.skipped.empty()) return fail(DecodeStatus::BAD_SCHEMA, "SCHEMA frame with tables other than 1-4");

    Schema schema;
    schema.version = f.header.version;
    schema.hash = hash;
    res = read_schema_tables(f, &schema);
    if (!res.ok()) return res;
    res = read_schema_enums(f, &schema);
    if (!res.ok()) return res;
    res = check_bootstrap_agrees(schema);
    if (!res.ok()) return res;
    *out = std::move(schema);
    return DecodeResult();
}

// ── Re-encoding ───────────────────────────────────────────────────────────────────────────────

// `t` can be written by this build's FrameWriter: schema.h defines its id with the same column
// storages, for this frame kind.
static DecodeResult encodable(const DecodedFrame &frame, const DecodedTable &t) {
    const TableDef *mine = find_table_def(t.id);
    if (!mine) return fail(DecodeStatus::NOT_ENCODABLE, "table " + std::to_string(t.id) + " is not in schema.h");
    if (!t.def || t.def->columns.size() != mine->columns.size())
        return fail(DecodeStatus::NOT_ENCODABLE, "table " + mine->name + " has other columns in schema.h");
    for (size_t i = 0; i < mine->columns.size(); ++i)
        if (t.def->columns[i].storage != mine->columns[i].storage)
            return fail(DecodeStatus::NOT_ENCODABLE, "table " + mine->name + " has other columns in schema.h");
    if (static_cast<uint8_t>(mine->frame_kind) != frame.header.kind)
        return fail(DecodeStatus::NOT_ENCODABLE, "table " + mine->name + " belongs to another frame kind");
    return DecodeResult();
}

static void write_value(FrameWriter::Row &r, Storage storage, const Value &v) {
    switch (storage) {
        case Storage::U8:
            r.u8(v.bits);
            break;
        case Storage::U16:
            r.u16(v.bits);
            break;
        case Storage::U32:
            r.u32(v.bits);
            break;
        case Storage::U64:
            r.u64(v.bits);
            break;
        case Storage::I8:
            r.i8(value_signed(v, storage));
            break;
        case Storage::I16:
            r.i16(value_signed(v, storage));
            break;
        case Storage::I32:
            r.i32(value_signed(v, storage));
            break;
        case Storage::I64:
            r.i64(value_signed(v, storage));
            break;
        case Storage::F32: {
            const uint32_t bits = static_cast<uint32_t>(v.bits);
            float f = 0;
            std::memcpy(&f, &bits, sizeof(f));
            r.f32(f);
            break;
        }
        case Storage::TEXT:
            r.text(v.text);
            break;
    }
}

DecodeResult encode_frame(const DecodedFrame &frame, std::string *out) {
    if (!frame.skipped.empty())
        return fail(DecodeStatus::NOT_ENCODABLE, "frame had blocks the decoding schema does not know");
    if (frame.header.kind >= enum_count_frame_kind() || frame.header.viewer >= enum_count_viewer())
        return fail(DecodeStatus::BAD_KIND, "unknown frame kind or viewer");
    for (const DecodedTable &t : frame.tables) {
        DecodeResult res = encodable(frame, t);
        if (!res.ok()) return res;
    }
    FrameWriter w(
        static_cast<FrameKind>(frame.header.kind), static_cast<Viewer>(frame.header.viewer), frame.header.schema_hash);
    for (const DecodedTable &t : frame.tables) {
        for (const std::vector<Value> &row : t.rows) {
            FrameWriter::Row r = w.row(static_cast<TableId>(t.id));
            for (size_t c = 0; c < row.size(); ++c) write_value(r, t.def->columns[c].storage, row[c]);
        }
    }
    *out = w.finish();
    return DecodeResult();
}

// ── Table blocks by framing alone ─────────────────────────────────────────────────────────────

// Every block of `payload` as (table id, offset of its header), plus the end offset as a last entry
// with id 0. Checks only that the blocks add up to the payload.
static DecodeResult scan_blocks(const std::string &payload, std::vector<std::pair<uint16_t, size_t>> *blocks) {
    blocks->clear();
    if (payload.size() < FRAME_HEADER_SIZE || read_le(payload, 0, 4) != FRAME_MAGIC)
        return fail(DecodeStatus::MALFORMED, "not an RMSF frame");
    const uint32_t n = static_cast<uint32_t>(read_le(payload, 16, 4));
    size_t off = FRAME_HEADER_SIZE;
    for (uint32_t i = 0; i < n; ++i) {
        if (payload.size() - off < TABLE_HEADER_SIZE) return fail(DecodeStatus::MALFORMED, "truncated table header");
        const uint16_t id = static_cast<uint16_t>(read_le(payload, off, 2));
        const uint64_t len = read_le(payload, off + 4, 4) * read_le(payload, off + 8, 4);
        const uint64_t padded = (len + ROW_ALIGN - 1) / ROW_ALIGN * ROW_ALIGN;
        if (padded > payload.size() - off - TABLE_HEADER_SIZE)
            return fail(DecodeStatus::MALFORMED, "truncated table " + std::to_string(id));
        blocks->push_back({id, off});
        off += TABLE_HEADER_SIZE + static_cast<size_t>(padded);
    }
    if (off != payload.size()) return fail(DecodeStatus::MALFORMED, "bytes after the last table block");
    blocks->push_back({0, off});
    return DecodeResult();
}

static void write_le(std::string *bytes, size_t at, uint32_t width, uint64_t v) {
    for (uint32_t i = 0; i < width; ++i) (*bytes)[at + i] = static_cast<char>((v >> (8 * i)) & 0xFF);
}

DecodeResult split_frame_blocks(
    const std::string &payload, uint16_t first_id, std::string *head, std::string *tail, uint32_t *n_tail) {
    std::vector<std::pair<uint16_t, size_t>> blocks;
    DecodeResult res = scan_blocks(payload, &blocks);
    if (!res.ok()) return res;
    head->clear();
    tail->clear();
    *n_tail = 0;
    uint32_t n_head = 0;
    head->assign(payload, 0, FRAME_HEADER_SIZE);
    for (size_t i = 0; i + 1 < blocks.size(); ++i) {
        const size_t at = blocks[i].second;
        const size_t len = blocks[i + 1].second - at;
        if (blocks[i].first >= first_id) {
            tail->append(payload, at, len);
            ++*n_tail;
        } else {
            head->append(payload, at, len);
            ++n_head;
        }
    }
    write_le(head, 16, 4, n_head);
    return DecodeResult();
}

DecodeResult append_frame_blocks(std::string *payload, const std::string &blocks, uint32_t n) {
    std::vector<std::pair<uint16_t, size_t>> have;
    DecodeResult res = scan_blocks(*payload, &have);
    if (!res.ok()) return res;
    // The appended blocks, framed as a frame of their own, must add up too.
    std::string extra = payload->substr(0, FRAME_HEADER_SIZE) + blocks;
    write_le(&extra, 16, 4, n);
    std::vector<std::pair<uint16_t, size_t>> add;
    res = scan_blocks(extra, &add);
    if (!res.ok()) return res;
    const int last = have.size() > 1 ? have[have.size() - 2].first : -1;
    if (n > 0 && static_cast<int>(add[0].first) <= last)
        return fail(DecodeStatus::MALFORMED, "an appended table block is not above the frame's");
    const uint32_t total = static_cast<uint32_t>(read_le(*payload, 16, 4)) + n;
    payload->append(blocks);
    write_le(payload, 16, 4, total);
    return DecodeResult();
}

DecodeResult decode_appended_blocks(const std::string &blocks, uint32_t n, const Schema &schema, DecodedFrame *frame) {
    // The blocks as a frame of their own: the frame's header, a string pool holding only the empty
    // string (the appended tables hold no text) and the blocks.
    std::string extra(FRAME_HEADER_SIZE + TABLE_HEADER_SIZE + ROW_ALIGN, '\0');
    write_le(&extra, 0, 4, FRAME_MAGIC);
    write_le(&extra, 4, 2, frame->header.version);
    write_le(&extra, 6, 1, frame->header.kind);
    write_le(&extra, 7, 1, frame->header.viewer);
    write_le(&extra, 8, 8, frame->header.schema_hash);
    write_le(&extra, 16, 4, static_cast<uint64_t>(n) + 1);
    write_le(&extra, FRAME_HEADER_SIZE, 2, static_cast<uint16_t>(TableId::STRINGS));
    write_le(&extra, FRAME_HEADER_SIZE + 4, 4, 1);
    write_le(&extra, FRAME_HEADER_SIZE + 8, 4, 1);
    extra += blocks;
    DecodedFrame tail;
    DecodeResult res = decode_frame(extra, schema, &tail);
    if (!res.ok()) return res;
    if (!tail.tables.empty() && !frame->tables.empty() && tail.tables.front().id <= frame->tables.back().id)
        return fail(DecodeStatus::MALFORMED, "an appended table block is not above the frame's");
    for (DecodedTable &t : tail.tables) frame->tables.push_back(std::move(t));
    frame->skipped.insert(frame->skipped.end(), tail.skipped.begin(), tail.skipped.end());
    frame->header.n_tables += n;
    return DecodeResult();
}

// ── Stream records ────────────────────────────────────────────────────────────────────────────

bool RecordReader::next(std::string *payload, DecodeResult *result) {
    *result = DecodeResult();
    std::string head;
    int ch;
    while ((ch = std::fgetc(f_)) != EOF && ch != '\n') {
        head.push_back(static_cast<char>(ch));
        if (head.size() > 32) {
            *result = fail(DecodeStatus::MALFORMED, "record header too long");
            return false;
        }
    }
    if (ch == EOF && head.empty()) return false;
    const std::string prefix = "RMSTATE ";
    if (ch == EOF || head.compare(0, prefix.size(), prefix) != 0 || head.size() == prefix.size()) {
        *result = fail(DecodeStatus::MALFORMED, "bad record header after record " + std::to_string(count_));
        return false;
    }
    uint64_t n = 0;
    for (size_t i = prefix.size(); i < head.size(); ++i) {
        if (head[i] < '0' || head[i] > '9') {
            *result = fail(DecodeStatus::MALFORMED, "bad record length after record " + std::to_string(count_));
            return false;
        }
        n = n * 10 + static_cast<uint64_t>(head[i] - '0');
    }
    payload->assign(static_cast<size_t>(n), '\0');
    if (n > 0 && std::fread(&(*payload)[0], 1, static_cast<size_t>(n), f_) != n) {
        *result = fail(DecodeStatus::MALFORMED, "truncated record " + std::to_string(count_));
        return false;
    }
    ++count_;
    return true;
}

}  // namespace rmstate
