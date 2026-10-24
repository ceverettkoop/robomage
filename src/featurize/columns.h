#ifndef FEATURIZE_COLUMNS_H
#define FEATURIZE_COLUMNS_H

#include <cstdint>
#include <string>
#include <vector>

#include "../state_format/decode.h"
#include "../state_format/schema.h"

// ── Columns by semantic tag (docs/featurizer.md §3.2, §4), standard library and the decoder only ─
//
// The rules shared by the per-decision tensors (feature_spec.h) and the card tables
// (card_tables.h): a column's tag parsed, and the numeric transform of a cell. Nothing here names
// a table or a column.

namespace featurize {

// A column's semantic tag (docs/state_format.md §9, Appendix A.1).
enum class TagKind : uint8_t {
    BOOL,
    COUNT,
    SIGNED,
    INDEX,
    MASK,        // mask:<enum>
    ENUM,        // enum:<enum>
    PLAYER_REL,  // player-rel
    CARD,
    HANDLE,
    ROW,  // row:<table>
    TEXT,
    OTHER,
};

struct Tag {
        TagKind kind = TagKind::OTHER;
        std::string arg;  // the enum of mask: / enum:, the table of row:
};

Tag parse_tag(const std::string &tag);

// A numeric slot's transform (§3.2): BOOL x; LOG1P log1p(x); SIGNED_LOG1P sign(x)·log1p(|x|);
// INDEX log1p(x), 0 at the none value; NONE_FLAG 1 at the none value (all ones);
// MASK_BIT bit `arg` of x; ONE_HOT x == arg; COUNT the number of rows (card tables' child tables).
enum class Transform : uint8_t {
    BOOL,
    LOG1P,
    SIGNED_LOG1P,
    INDEX,
    NONE_FLAG,
    MASK_BIT,
    ONE_HOT,
    COUNT,
};

const char *transform_name(Transform t);

// The value of cell `v` (stored as `storage`) under `t` (`arg`: MASK_BIT's bit, ONE_HOT's value).
// COUNT is not a cell transform (0).
float cell_value(Transform t, uint32_t arg, const rmstate::Value &v, rmstate::Storage storage);

// The numeric transform of a scalar tag (BOOL, COUNT, SIGNED, INDEX); false for any other.
bool scalar_transform(TagKind kind, Transform *out);

// The string table of enum `name` in `schema` (empty when the schema has none).
const std::vector<std::string> &enum_values(const rmstate::Schema &schema, const std::string &name);

// The table named `name` in `schema`, or nullptr.
const rmstate::TableDef *table_named(const rmstate::Schema &schema, const std::string &name);

// The index of column `name` in `def`, or -1.
int column_index(const rmstate::TableDef &def, const std::string &name);

}  // namespace featurize

#endif /* FEATURIZE_COLUMNS_H */
