#ifndef STATE_FORMAT_DECODE_H
#define STATE_FORMAT_DECODE_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "schema.h"

// ── RMSTATE bytes → plain structs (docs/state_format.md §9, Appendix A) ───────────────────────
//
// Standard library and schema.h only: the knowledge sidecar and the standalone tools decode with
// this, and none of them may see an engine header.
//
// Decoding is generic. A Schema (tables, columns, enum string tables) is built from a SCHEMA frame
// (decode_schema_frame); every other frame is then decoded against it into tables of rows of typed
// values (decode_frame), with text cells resolved from the frame's string pool. Nothing here
// names a table other than the SCHEMA frame's own: the SCHEMA frame is read with the layout of
// tables 1-4 that schema.h defines (the bootstrap), and the frame's own description of those
// tables must agree with it.
//
// Errors are returned, never thrown (the engine builds with -fno-exceptions): every entry point
// returns a DecodeResult, and on failure its output is unspecified.
//
// Refused: a bad magic, a format version other than FORMAT_VERSION, a frame kind or viewer outside
// its enum, a schema hash other than the Schema's, a table the schema knows with another row width,
// table blocks out of ascending id order, nonzero reserved or padding bytes, an empty block,
// truncation or trailing bytes, a text offset that is not the start of a pooled string. Skipped: a
// table id the schema does not know (listed in DecodedFrame::skipped).

namespace rmstate {

enum class DecodeStatus : uint8_t {
    OK,
    MALFORMED,        // framing, truncation, ordering, reserved / padding bytes, string pool
    BAD_MAGIC,        // not an RMSF frame
    BAD_VERSION,      // format version other than FORMAT_VERSION
    BAD_KIND,         // frame kind or viewer outside its enum
    SCHEMA_MISMATCH,  // schema hash, or a known table's row width or frame kind, differs
    BAD_SCHEMA,       // a SCHEMA frame that does not describe a consistent schema
    NOT_ENCODABLE,    // re-encoding: a table or column this build's schema.h does not define
};

struct DecodeResult {
        DecodeStatus status = DecodeStatus::OK;
        std::string message;  // what was wrong, for an error line; empty on success
        bool ok() const { return status == DecodeStatus::OK; }
};

// A schema as a SCHEMA frame describes it. Table and column definitions reuse schema.h's structs;
// a TableDef's id may be a value TableId does not name.
struct Schema {
        uint16_t version = 0;
        uint64_t hash = 0;             // the SCHEMA frame's hash: every frame decoded against this repeats it
        std::vector<TableDef> tables;  // ascending id
        std::vector<EnumTable> enums;  // in SCHEMA frame order
        const TableDef *find_table(uint16_t id) const;
        const EnumTable *find_enum(const std::string &name) const;
};

// The 24-byte frame header (Appendix A.2).
struct FrameHeader {
        uint16_t version = 0;
        uint8_t kind = 0;    // FrameKind
        uint8_t viewer = 0;  // Viewer
        uint64_t schema_hash = 0;
        uint32_t n_tables = 0;
};

// One cell: the raw little-endian value zero-extended to 64 bits, and for a TEXT cell the pooled
// string its offset names.
struct Value {
        uint64_t bits = 0;
        std::string text;
};

// One table block, decoded against its schema definition (`def` points into the Schema the frame
// was decoded with, which must outlive it).
struct DecodedTable {
        uint16_t id = 0;
        const TableDef *def = nullptr;
        std::vector<std::vector<Value>> rows;  // rows[r][c] = column c of row r
};

struct DecodedFrame {
        FrameHeader header;
        std::string strings;               // the string pool as sent (table 1)
        std::vector<DecodedTable> tables;  // every known block except the string pool, ascending id
        std::vector<uint16_t> skipped;     // ids of blocks the schema does not know
        const DecodedTable *table(uint16_t id) const;
};

// The value of `v` stored as `storage`, sign-extended for the signed storages.
int64_t value_signed(const Value &v, Storage storage);
// Whether `v` is all ones of `storage`'s width (the none value of card, index, row: and nullable
// enum columns, Appendix A.1).
bool value_is_all_ones(const Value &v, Storage storage);

// Builds `out` from a SCHEMA frame payload: checks the frame's hash against its own table blocks,
// reads tables 1-4 with schema.h's layout, checks that the frame describes those four tables the
// same way, and that every table's column offsets and widths add up to its row width.
DecodeResult decode_schema_frame(const std::string &payload, Schema *out);

// Decodes any frame payload (a SCHEMA frame included) against `schema`.
DecodeResult decode_frame(const std::string &payload, const Schema &schema, DecodedFrame *out);

// Re-encodes a decoded frame with schema.h's FrameWriter: the same tables, rows and values (text
// re-pooled in first-use order), so a frame the engine wrote comes back byte-identical. Refused
// (NOT_ENCODABLE) when a table is unknown to this build's schema.h, is defined there with other
// column storages or another frame kind, or the frame skipped unknown blocks.
DecodeResult encode_frame(const DecodedFrame &frame, std::string *out);

// Splits a frame payload at table id `first_id` by its framing alone (header and block headers,
// Appendix A.2; nothing is decoded): `head` receives the frame without its blocks of id `first_id`
// or above (n_tables adjusted), `tail` those blocks' bytes as laid out, in order, and `n_tail`
// their count. Refused (MALFORMED) when the framing does not add up. The knowledge tables (ids 90
// and up) are appended to a STATE frame after it is built, and split off it to compare them.
DecodeResult split_frame_blocks(
    const std::string &payload, uint16_t first_id, std::string *head, std::string *tail, uint32_t *n_tail);

// Appends `n` table blocks (`blocks`, laid out as split_frame_blocks gives them) to a frame payload,
// adjusting its n_tables. Refused (MALFORMED) when the framing of either does not add up, or an
// appended block's id is not above every id the frame holds.
DecodeResult append_frame_blocks(std::string *payload, const std::string &blocks, uint32_t n);

// Decodes `n` table blocks (laid out as split_frame_blocks gives them) against `schema` and adds
// them to `frame`, decoded from the frame they were appended to (append_frame_blocks): `frame` is
// then what decoding the whole frame gives. Refused when a block does not decode, holds text, or
// is not above every table `frame` holds.
DecodeResult decode_appended_blocks(const std::string &blocks, uint32_t n, const Schema &schema, DecodedFrame *frame);

// Reads stream records ("RMSTATE <nbytes>\n" + payload, §9) from a file or pipe.
class RecordReader {
    public:
        explicit RecordReader(FILE *f) : f_(f) {}
        // The next record's payload. Returns false at a clean end of stream (`result` OK) or on a
        // framing error (`result` says what).
        bool next(std::string *payload, DecodeResult *result);
        // Records read so far.
        size_t count() const { return count_; }

    private:
        FILE *f_;
        size_t count_ = 0;
};

}  // namespace rmstate

#endif /* STATE_FORMAT_DECODE_H */
