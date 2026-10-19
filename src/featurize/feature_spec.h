#ifndef FEATURIZE_FEATURE_SPEC_H
#define FEATURIZE_FEATURE_SPEC_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../state_format/decode.h"
#include "card_tables.h"
#include "columns.h"

// ── FEATURE_SPEC: what the per-decision tensors hold (docs/featurizer.md §3, §6) ───────────────
//
// Standard library and the decoder only. Built from a SCHEMA frame (every table, column, tag and
// enum), a METAGAME frame (its tables' row counts) and the card tables (their widths, column names
// and fixed key list), so it is the same for every match. Every STATE table with id 20-48 or 90-99
// is a token type, in ascending id; each column becomes slots by its tag (columns.h, §3.2):
//
//   bool, count, signed, prob   one numeric slot (x, log1p, sign·log1p|x|, x / 10^9)
//   index                       one numeric slot (log1p, 0 at none) and its none flag
//   mask:<e>                    one numeric slot per value of e
//   enum:<e>, player-rel        one categorical slot of e's values (an absolute enum:seat column
//                               beside its <name>_rel partner is dropped: the tensors are
//                               seat-symmetric)
//   card                        one tok_card slot and a numeric none flag
//   abref (ab_card, ab_index,   one tok_ability slot, and ab_key hashed (below)
//     ab_key)
//   handle                      a relation from the row's token to the token owning the handle
//                               (the lowest-id table whose first column is a handle named `handle`:
//                               object, then stack), typed by (table, column); none when no token
//                               owns it or the row owns it itself
//   row:<t>                     a relation to row v of t's tokens; into a METAGAME table, a
//                               categorical slot of that table's rows; else dropped
//   text                        dropped
//
// A categorical slot of n values takes ids offset … offset + n of one global id space: offset + 0
// is the cell's none (all ones, or a value out of range), offset + 1 + v value v; id 0 is the pad.
// The overrides (§3.4, the only per-column rules) are listed in the spec: object.card's second card
// slot (the current face), the hashed text columns (FNV-1a 64 modulo HASH_BUCKETS, + 1; empty text
// is the none id), menu.description dropped, and the knowledge tables' METAGAME row indices
// (k_template.list, k_archetype.archetype, b_list.list, b_archetype.archetype: categorical over
// meta_list / meta_archetype rows).
//
// spec_json() writes the document with its keys in a fixed order; spec_hash is FNV-1a 64 of its
// bytes.

namespace featurize {

const int SPEC_VERSION = 1;
// Buckets of a hashed text column.
const uint32_t HASH_BUCKETS = 256;
// The library depth of the learner targets (targets.h).
const uint32_t LIBRARY_DEPTH = 10;

struct NumSlot {
        std::string name, column;
        Transform transform = Transform::BOOL;
        uint32_t bit = 0;  // MASK_BIT's bit
};

struct CatSlot {
        std::string name, column;
        std::string source;  // enum:<e>, player-rel, hashed, row:<metagame table>
        int64_t offset = 0;  // the slot's global id of none; values follow
        uint32_t size = 0;   // ids, none included
};

struct CardSlot {
        std::string name, column;
        bool current_face = false;  // the face as it is now (object.card's second slot)
};

struct RelationType {
        std::string table, column;
        std::string target;  // "handle", or the table a row: column points into
};

// One column's work on a token row, planned when the spec is built.
enum class OpKind : uint8_t {
    NUMERIC,      // num = cell_value(transform)
    INDEX,        // num = index_log1p, flag = none
    MASK,         // num … num + width - 1 = the bits
    CATEGORICAL,  // cat = offset + (v < width ? v + 1 : 0)
    HASHED,       // cat = offset + (text empty ? 0 : 1 + fnv1a64(text) % HASH_BUCKETS)
    CARD,         // card = v (-1 at none or past the card table), flag = none; face: the current face
    ABILITY,      // ability = the ability row of (ab_card, ab_index), else of ab_key, else -1
    HANDLE,       // relation to the handle's owner
    ROW,          // relation to row v of the target type
};

struct ColumnOp {
        OpKind kind = OpKind::NUMERIC;
        uint16_t col = 0;
        rmstate::Storage storage = rmstate::Storage::U8;
        Transform transform = Transform::BOOL;
        uint32_t slot = 0;       // numeric (first), categorical, card or ability slot
        int32_t flag = -1;       // INDEX, CARD: the none flag's numeric slot
        uint32_t width = 0;      // MASK: bits; CATEGORICAL: values
        int64_t offset = 0;      // CATEGORICAL, HASHED
        int32_t face_slot = -1;  // CARD: the current face's card slot (-1 none)
        uint16_t aux1 = 0;       // CARD with a face slot: the `transformed` column; ABILITY: ab_index
        uint16_t aux2 = 0;       // ABILITY: ab_key
        rmstate::Storage aux1_storage = rmstate::Storage::U8;
        int32_t relation = -1;  // HANDLE, ROW
        int32_t target = -1;    // ROW: the target token type
};

struct TokenType {
        uint16_t table = 0;
        std::string name;
        bool owns_handles = false;  // first column a handle named `handle`
        uint16_t handle_col = 0;
        std::vector<NumSlot> numeric;
        std::vector<CatSlot> categorical;
        std::vector<CardSlot> cards;
        std::vector<std::string> abilities;
        std::vector<int32_t> relations;
        std::vector<ColumnOp> ops;  // column order
};

struct Override {
        std::string table, column, treatment;
};

struct FeatureSpec {
        uint64_t schema_hash = 0;
        uint64_t vocab_hash = 0;
        uint32_t hash_buckets = HASH_BUCKETS;
        uint32_t library_depth = LIBRARY_DEPTH;
        // The rows of each METAGAME table a categorical slot counts (table name, rows), in first use.
        std::vector<std::pair<std::string, uint64_t>> metagame_rows;
        size_t F = 0, C = 0, K = 0, Q = 0;
        int64_t categorical_size = 0;  // ids 0 … categorical_size - 1
        std::vector<TokenType> types;
        std::vector<RelationType> relations;
        std::vector<Override> overrides;
        // The card tables' shapes and column names (card_tables.h).
        uint64_t card_rows = 0, text_rows = 0, text_width = 0, ability_rows = 0, part_rows = 0;
        std::vector<std::string> card_columns, ability_columns, part_columns, ab_keys;
        // Lookups: per table id its token type (-1 none); the menu's token type; the header's.
        std::vector<int32_t> type_of_table;
        int32_t menu_type = -1, header_type = -1;
        int32_t header_game = -1, header_decision = -1, header_sideboarding = -1;
        std::string json;
        uint64_t hash = 0;
};

// Builds the spec. False with `err` when the schema lacks the menu or header table.
bool build_spec(const rmstate::Schema &schema, const rmstate::DecodedFrame &metagame, const CardTables &tables,
    FeatureSpec *out, std::string *err);

// The spec document (FeatureSpec::json), keys in a fixed order.
std::string spec_json(const FeatureSpec &spec);

}  // namespace featurize

#endif /* FEATURIZE_FEATURE_SPEC_H */
