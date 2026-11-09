#include "card_tables.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>

#include "columns.h"

namespace {

// One feature column of a table row by the tag rules: column `col` under `transform`, written at
// position `at`.
struct StaticCol {
        int col = 0;
        featurize::Transform transform = featurize::Transform::BOOL;
        uint32_t arg = 0;
        rmstate::Storage storage = rmstate::Storage::U8;
        uint32_t at = 0;
};

// A (card, face) child table's count columns: rows with value `arg` of column `col`, at `at`.
struct CountCol {
        int col = 0;
        uint32_t arg = 0;
        uint32_t at = 0;
};

const char *const FEATURES = "features";

}  // namespace

static std::vector<StaticCol> static_layout(const rmstate::Schema &schema, const rmstate::TableDef &def,
    const std::set<std::string> &skip, const std::string &table, featurize::Registry *reg);
static void fill_static(const std::vector<rmstate::Value> &row, const std::vector<StaticCol> &layout, float *out);
static size_t table_width(
    const featurize::Registry &reg, const std::string &table, const std::vector<StaticCol> &cols, uint32_t extra_at);
static std::vector<std::string> table_names(const featurize::Registry &reg, const std::string &table, size_t width);
static bool is_face_child(const rmstate::TableDef &def);
static std::map<std::pair<uint64_t, uint64_t>, uint64_t> face_ids(
    const rmstate::TableDef &card, const std::vector<std::vector<rmstate::Value>> &rows, uint64_t card_rows);
// The face vocab id each (card, face) of the `card` rows describes: the row's face_card, the first
// row (front rows first) naming an id taking it; a face with no id of its own, or one another row
// took, describes none.
static std::map<std::pair<uint64_t, uint64_t>, uint64_t> face_ids(
    const rmstate::TableDef &card, const std::vector<std::vector<rmstate::Value>> &rows, uint64_t card_rows) {
    const int c_card = featurize::column_index(card, "card"), c_face = featurize::column_index(card, "face"),
              c_face_card = featurize::column_index(card, "face_card");
    const uint64_t none_id = card_rows - 1;
    std::vector<uint8_t> taken(card_rows, 0);
    std::map<std::pair<uint64_t, uint64_t>, uint64_t> out;
    for (int pass = 0; pass < 2; ++pass)
        for (const std::vector<rmstate::Value> &row : rows) {
            const uint64_t cid = row[static_cast<size_t>(c_card)].bits, face = row[static_cast<size_t>(c_face)].bits;
            if ((face != 0) != (pass == 1)) continue;
            const rmstate::Value &fv = row[static_cast<size_t>(c_face_card)];
            if (rmstate::value_is_all_ones(fv, card.columns[static_cast<size_t>(c_face_card)].storage) ||
                fv.bits >= none_id || taken[fv.bits])
                continue;
            taken[fv.bits] = 1;
            out[{cid, face}] = fv.bits;
        }
    return out;
}

static const std::vector<std::vector<rmstate::Value>> &rows_of(
    const rmstate::DecodedFrame &frame, const rmstate::TableDef *def);
static std::vector<size_t> sorted_rows(
    const std::vector<std::vector<rmstate::Value>> &rows, const std::vector<int> &by);
static std::set<std::string> fixed_keys(
    const rmstate::Schema &schema, const std::vector<std::vector<rmstate::Value>> &ab, int key_col);
static bool build_card_rows(const rmstate::Schema &schema, const rmstate::DecodedFrame &cards, featurize::Registry *reg,
    featurize::CardTables *out, std::string *err);
static bool build_ability_rows(const rmstate::Schema &schema, const rmstate::DecodedFrame &cards,
    featurize::Registry *reg, featurize::CardTables *out, std::string *err);
static bool build_part_rows(const rmstate::Schema &schema, const rmstate::DecodedFrame &cards, featurize::Registry *reg,
    featurize::CardTables *out, std::string *err);
static bool read_array(
    const featurize::RmtbFile &f, const char *name, size_t rank, featurize::RmtbArray const **out, std::string *err);
static bool read_features(const featurize::RmtbFile &f, std::vector<std::string> *columns, std::vector<float> *values,
    uint64_t *rows, std::string *err);
static bool read_ints(
    const featurize::RmtbFile &f, const char *name, uint64_t rows, std::vector<int64_t> *out, std::string *err);

// The feature columns of `def` by the tag rules (columns.h), at their positions in table `table` of the
// registry: a scalar one column (an index also its none flag), a mask one per bit and an enum one-hot,
// each in its enum's block at the value's registry index; cards, text, handles, row: and
// seat-relative columns, and the columns in `skip`, none.
static std::vector<StaticCol> static_layout(const rmstate::Schema &schema, const rmstate::TableDef &def,
    const std::set<std::string> &skip, const std::string &table, featurize::Registry *reg) {
    std::vector<StaticCol> out;
    for (size_t c = 0; c < def.columns.size(); ++c) {
        const rmstate::ColumnDef &col = def.columns[c];
        if (skip.count(col.name)) continue;
        const featurize::Tag tag = featurize::parse_tag(col.tag);
        StaticCol s;
        s.col = static_cast<int>(c);
        s.storage = col.storage;
        if (featurize::scalar_transform(tag.kind, &s.transform)) {
            const int32_t at = reg->column_at(table, col.name, "scalar", "");
            if (at >= 0) {
                s.at = static_cast<uint32_t>(at);
                out.push_back(s);
            }
            if (tag.kind == featurize::TagKind::INDEX) {
                const int32_t flag = reg->column_at(table, col.name + ":none", "scalar", "");
                if (flag < 0) continue;
                s.transform = featurize::Transform::NONE_FLAG;
                s.at = static_cast<uint32_t>(flag);
                out.push_back(s);
            }
        } else if (tag.kind == featurize::TagKind::MASK || tag.kind == featurize::TagKind::ENUM) {
            const std::vector<std::string> &values = featurize::enum_values(schema, tag.arg);
            const bool mask = tag.kind == featurize::TagKind::MASK;
            const std::vector<int32_t> map = reg->enum_map(tag.arg, values);
            const int32_t start = reg->column_at(table, col.name, mask ? "mask" : "onehot", tag.arg);
            if (start < 0) continue;
            s.transform = mask ? featurize::Transform::MASK_BIT : featurize::Transform::ONE_HOT;
            for (size_t v = 0; v < values.size(); ++v) {
                if (map[v] < 0) continue;
                s.arg = static_cast<uint32_t>(v);
                s.at = static_cast<uint32_t>(start + map[v]);
                out.push_back(s);
            }
        }
    }
    return out;
}

static void fill_static(const std::vector<rmstate::Value> &row, const std::vector<StaticCol> &layout, float *out) {
    for (const StaticCol &s : layout)
        out[s.at] = featurize::cell_value(s.transform, s.arg, row[static_cast<size_t>(s.col)], s.storage);
}

// A table's width: its registry capacity, or (a table an append just made) the columns it uses.
static size_t table_width(
    const featurize::Registry &reg, const std::string &table, const std::vector<StaticCol> &cols, uint32_t extra_at) {
    size_t width = reg.table_capacity(table);
    for (const StaticCol &s : cols) width = std::max<size_t>(width, s.at + 1);
    return std::max<size_t>(width, extra_at + 1);
}

// Every position's column name (Registry::column_names), `width` of them.
static std::vector<std::string> table_names(const featurize::Registry &reg, const std::string &table, size_t width) {
    std::vector<std::string> names = reg.column_names(table);
    for (size_t i = names.size(); i < width; ++i) names.push_back("#" + std::to_string(i));
    names.resize(width);
    return names;
}

// A CARDS table other than `card` whose first two columns are a card and its face.
static bool is_face_child(const rmstate::TableDef &def) {
    return def.frame_kind == rmstate::FrameKind::CARDS && def.name != "card" && def.columns.size() >= 2 &&
           def.columns[0].tag == "card" && def.columns[1].tag == "enum:face";
}

static const std::vector<std::vector<rmstate::Value>> &rows_of(
    const rmstate::DecodedFrame &frame, const rmstate::TableDef *def) {
    static const std::vector<std::vector<rmstate::Value>> none;
    const rmstate::DecodedTable *t = def ? frame.table(static_cast<uint16_t>(def->id)) : nullptr;
    return t ? t->rows : none;
}

// The row indices of `rows` ordered by the columns `by` (stable).
static std::vector<size_t> sorted_rows(
    const std::vector<std::vector<rmstate::Value>> &rows, const std::vector<int> &by) {
    std::vector<size_t> order(rows.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        for (const int c : by) {
            const uint64_t x = rows[a][static_cast<size_t>(c)].bits, y = rows[b][static_cast<size_t>(c)].bits;
            if (x != y) return x < y;
        }
        return false;
    });
    return order;
}

// The fixed key list (header comment).
static std::set<std::string> fixed_keys(
    const rmstate::Schema &schema, const std::vector<std::vector<rmstate::Value>> &ab, int key_col) {
    std::set<std::string> keys = {"LandType"};
    for (const std::vector<rmstate::Value> &row : ab)
        if (!row[static_cast<size_t>(key_col)].text.empty()) keys.insert(row[static_cast<size_t>(key_col)].text);
    for (const std::string &k : featurize::enum_values(schema, "keyword")) keys.insert("keyword:" + k);
    const size_t colors = featurize::enum_values(schema, "color").size();
    for (size_t c = 0; c < colors; ++c) keys.insert("subtype_mana:" + std::to_string(c));
    return keys;
}

static bool build_card_rows(const rmstate::Schema &schema, const rmstate::DecodedFrame &cards, featurize::Registry *reg,
    featurize::CardTables *out, std::string *err) {
    const rmstate::TableDef *card = featurize::table_named(schema, "card");
    const rmstate::TableDef *ab = featurize::table_named(schema, "card_ability");
    if (!card || !ab) {
        *err = "the schema has no card or card_ability table";
        return false;
    }
    const int c_card = featurize::column_index(*card, "card"), c_face = featurize::column_index(*card, "face"),
              c_face_card = featurize::column_index(*card, "face_card");
    const int a_card = featurize::column_index(*ab, "card"), a_face = featurize::column_index(*ab, "face");
    if (c_card < 0 || c_face < 0 || c_face_card < 0 || a_card < 0 || a_face < 0) {
        *err = "the card or card_ability table lacks card, face or face_card";
        return false;
    }
    const std::vector<StaticCol> layout = static_layout(schema, *card, {}, "card", reg);
    // The (card, face) child tables' count columns, in table order.
    std::vector<std::pair<const rmstate::TableDef *, std::vector<CountCol>>> children;
    uint32_t last = 0;
    for (const rmstate::TableDef &def : schema.tables) {
        if (!is_face_child(def)) continue;
        std::vector<CountCol> counts;
        for (size_t c = 2; c < def.columns.size(); ++c) {
            const featurize::Tag tag = featurize::parse_tag(def.columns[c].tag);
            if (tag.kind != featurize::TagKind::ENUM) continue;
            const std::vector<std::string> &values = featurize::enum_values(schema, tag.arg);
            const std::vector<int32_t> map = reg->enum_map(tag.arg, values);
            const int32_t start = reg->column_at("card", def.name + "." + def.columns[c].name, "count", tag.arg);
            if (start < 0) continue;
            for (size_t v = 0; v < values.size(); ++v)
                if (map[v] >= 0) {
                    counts.push_back(
                        {static_cast<int>(c), static_cast<uint32_t>(v), static_cast<uint32_t>(start + map[v])});
                    last = std::max(last, counts.back().at);
                }
        }
        children.emplace_back(&def, counts);
    }
    const int32_t n_abilities_col = reg->column_at("card", "n_abilities", "scalar", "");
    const int32_t present_col = reg->column_at("card", "present", "scalar", "");
    if (!reg->ok()) {
        *err = featurize::registry_error(*reg);
        return false;
    }
    const size_t width = table_width(
        *reg, "card", layout, std::max(last, static_cast<uint32_t>(std::max(n_abilities_col, present_col))));
    out->card_columns = table_names(*reg, "card", width);
    out->card.assign(out->card_rows * width, 0.0f);
    out->back_face.resize(out->card_rows);
    for (size_t c = 0; c < out->card_rows; ++c) out->back_face[c] = static_cast<int64_t>(c);
    const uint64_t none_id = out->card_rows - 1;
    const std::vector<std::vector<rmstate::Value>> &rows = rows_of(cards, card);
    const std::map<std::pair<uint64_t, uint64_t>, uint64_t> face_of = face_ids(*card, rows, out->card_rows);
    for (const std::vector<rmstate::Value> &row : rows) {
        const uint64_t cid = row[static_cast<size_t>(c_card)].bits, face = row[static_cast<size_t>(c_face)].bits;
        const rmstate::Value &fv = row[static_cast<size_t>(c_face_card)];
        if (face != 0 && cid < none_id)
            out->back_face[cid] =
                rmstate::value_is_all_ones(fv, card->columns[static_cast<size_t>(c_face_card)].storage)
                    ? -1
                    : static_cast<int64_t>(fv.bits);
        auto it = face_of.find({cid, face});
        if (it == face_of.end() || it->second != fv.bits) continue;
        float *dst = &out->card[fv.bits * width];
        fill_static(row, layout, dst);
        dst[present_col] = 1.0f;
    }
    for (const auto &child : children) {
        for (const std::vector<rmstate::Value> &row : rows_of(cards, child.first)) {
            auto it = face_of.find({row[0].bits, row[1].bits});
            if (it == face_of.end()) continue;
            float *dst = &out->card[it->second * width];
            for (const CountCol &k : child.second)
                if (row[static_cast<size_t>(k.col)].bits == k.arg) dst[k.at] += 1.0f;
        }
    }
    std::vector<uint32_t> n_ab(out->card_rows, 0);
    for (const std::vector<rmstate::Value> &row : rows_of(cards, ab)) {
        auto it = face_of.find({row[static_cast<size_t>(a_card)].bits, row[static_cast<size_t>(a_face)].bits});
        if (it != face_of.end()) ++n_ab[it->second];
    }
    for (size_t f = 0; f < out->card_rows; ++f)
        if (n_ab[f])
            out->card[f * width + static_cast<size_t>(n_abilities_col)] = std::log1p(static_cast<float>(n_ab[f]));
    return true;
}

static bool build_ability_rows(const rmstate::Schema &schema, const rmstate::DecodedFrame &cards,
    featurize::Registry *reg, featurize::CardTables *out, std::string *err) {
    const rmstate::TableDef *ab = featurize::table_named(schema, "card_ability");
    const int a_card = featurize::column_index(*ab, "card"), a_index = featurize::column_index(*ab, "index"),
              a_key = featurize::column_index(*ab, "key");
    if (a_index < 0 || a_key < 0) {
        *err = "the card_ability table lacks index or key";
        return false;
    }
    const std::vector<StaticCol> layout = static_layout(schema, *ab, {"card", "index"}, "ability", reg);
    const int32_t keyed_col = reg->column_at("ability", "keyed", "scalar", "");
    const std::vector<std::vector<rmstate::Value>> &rows = rows_of(cards, ab);
    // The fixed keys by their row in the key band.
    std::vector<std::pair<int32_t, std::string>> keys;
    for (const std::string &key : fixed_keys(schema, rows, a_key)) {
        const int32_t r = reg->key_row(key);
        if (r >= 0) keys.push_back({r, key});
    }
    if (!reg->ok()) {
        *err = featurize::registry_error(*reg);
        return false;
    }
    std::sort(keys.begin(), keys.end());
    const size_t width = table_width(*reg, "ability", layout, static_cast<uint32_t>(keyed_col));
    out->ability_columns = table_names(*reg, "ability", width);
    const uint64_t per = reg->dims.abilities_per_card;
    out->abilities_per_card = static_cast<uint32_t>(per);
    out->key_base = (out->card_rows - 1) * per;
    uint64_t key_cap = reg->dims.ab_keys;
    for (const auto &k : keys) key_cap = std::max<uint64_t>(key_cap, static_cast<uint64_t>(k.first) + 1);
    const uint64_t total = out->key_base + key_cap;
    out->ability.assign(total * width, 0.0f);
    out->ability_card.assign(total, -1);
    out->ability_index.assign(total, -1);
    std::map<std::string, uint64_t> first_with_key;
    for (const size_t r : sorted_rows(rows, {a_card, a_index})) {
        const std::vector<rmstate::Value> &row = rows[r];
        const uint64_t card = row[static_cast<size_t>(a_card)].bits, index = row[static_cast<size_t>(a_index)].bits;
        if (card >= out->card_rows - 1) continue;
        if (index >= per) {
            *err = "card " + std::to_string(card) + " has an ability of index " + std::to_string(index) +
                   ": past the registry's abilities_per_card " + std::to_string(per) + " (a layout bump)";
            return false;
        }
        const uint64_t at = card * per + index;
        if (out->ability_card[at] >= 0) {
            *err = "card " + std::to_string(card) + " has two abilities of index " + std::to_string(index);
            return false;
        }
        out->ability_card[at] = static_cast<int64_t>(card);
        out->ability_index[at] = static_cast<int64_t>(index);
        fill_static(row, layout, &out->ability[at * width]);
        first_with_key.emplace(row[static_cast<size_t>(a_key)].text, at);
    }
    for (const auto &k : keys) {
        const uint64_t r = out->key_base + static_cast<uint64_t>(k.first);
        out->ab_keys.push_back(k.second);
        out->key_rows.push_back(static_cast<int64_t>(r));
        auto it = first_with_key.find(k.second);
        if (it != first_with_key.end())
            std::copy(out->ability.begin() + static_cast<long>(it->second * width),
                out->ability.begin() + static_cast<long>((it->second + 1) * width),
                out->ability.begin() + static_cast<long>(r * width));
        out->ability[r * width + static_cast<size_t>(keyed_col)] = 1.0f;
    }
    if (!out->build_index()) {
        *err = "a card's card_ability rows are not indices 0 … n-1";
        return false;
    }
    return true;
}

static bool build_part_rows(const rmstate::Schema &schema, const rmstate::DecodedFrame &cards, featurize::Registry *reg,
    featurize::CardTables *out, std::string *err) {
    const rmstate::TableDef *part = featurize::table_named(schema, "card_ability_part");
    if (!part) {
        *err = "the schema has no card_ability_part table";
        return false;
    }
    const int p_card = featurize::column_index(*part, "card"), p_ability = featurize::column_index(*part, "ability"),
              p_part = featurize::column_index(*part, "part");
    if (p_card < 0 || p_ability < 0 || p_part < 0) {
        *err = "the card_ability_part table lacks card, ability or part";
        return false;
    }
    const std::vector<StaticCol> layout = static_layout(schema, *part, {"card", "ability", "part"}, "part", reg);
    if (!reg->ok()) {
        *err = featurize::registry_error(*reg);
        return false;
    }
    const size_t width = table_width(*reg, "part", layout, 0);
    out->part_columns = table_names(*reg, "part", width);
    const std::vector<std::vector<rmstate::Value>> &rows = rows_of(cards, part);
    for (const size_t r : sorted_rows(rows, {p_card, p_ability, p_part})) {
        const std::vector<rmstate::Value> &row = rows[r];
        const int64_t a =
            out->ability_row(row[static_cast<size_t>(p_card)].bits, row[static_cast<size_t>(p_ability)].bits);
        if (a < 0) {
            if (row[static_cast<size_t>(p_card)].bits >= out->card_rows - 1) continue;
            *err = "a card_ability_part row names no card_ability row";
            return false;
        }
        out->part_ability.push_back(a);
        out->part.resize(out->part.size() + width, 0.0f);
        fill_static(row, layout, &out->part[out->part.size() - width]);
    }
    return true;
}

static bool read_array(
    const featurize::RmtbFile &f, const char *name, size_t rank, featurize::RmtbArray const **out, std::string *err) {
    *out = f.find(name);
    if (!*out || (*out)->shape.size() != rank) {
        *err = f.kind + " file: no array " + name + " of rank " + std::to_string(rank);
        return false;
    }
    return true;
}

static bool read_features(const featurize::RmtbFile &f, std::vector<std::string> *columns, std::vector<float> *values,
    uint64_t *rows, std::string *err) {
    const featurize::RmtbArray *a = nullptr;
    if (!read_array(f, FEATURES, 2, &a, err)) return false;
    if (a->columns.size() != a->shape[1] || !featurize::rmtb_values(*a, values)) {
        *err = f.kind + " file: features is not f32 with named columns";
        return false;
    }
    *columns = a->columns;
    *rows = a->shape[0];
    return true;
}

static bool read_ints(
    const featurize::RmtbFile &f, const char *name, uint64_t rows, std::vector<int64_t> *out, std::string *err) {
    const featurize::RmtbArray *a = nullptr;
    if (!read_array(f, name, 1, &a, err)) return false;
    if (a->shape[0] != rows || !featurize::rmtb_values(*a, out)) {
        *err = f.kind + " file: " + name + " is not " + std::to_string(rows) + " integers";
        return false;
    }
    return true;
}

namespace featurize {

int64_t CardTables::ability_row(uint64_t card, uint64_t index) const {
    if (card + 1 >= card_rows || index >= abilities_per_card) return -1;
    const uint64_t r = card * abilities_per_card + index;
    return r < ability_card.size() && ability_card[r] == static_cast<int64_t>(card) ? static_cast<int64_t>(r) : -1;
}

const PrintedRows *CardTables::printed_rows(const std::string &table) const {
    for (const PrintedRows &p : printed)
        if (p.table == table) return &p;
    return nullptr;
}

bool printed_row_less(const std::vector<rmstate::Value> &a, const std::vector<rmstate::Value> &b) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i].text != b[i].text) return a[i].text < b[i].text;
        if (a[i].bits != b[i].bits) return a[i].bits < b[i].bits;
    }
    return a.size() < b.size();
}

void build_printed_rows(const rmstate::Schema &schema, const rmstate::DecodedFrame &cards, CardTables *out) {
    out->present.assign(out->card_rows, 0);
    out->printed.clear();
    const rmstate::TableDef *card = table_named(schema, "card");
    if (!card || out->card_rows < 1) return;
    const std::map<std::pair<uint64_t, uint64_t>, uint64_t> face_of =
        face_ids(*card, rows_of(cards, card), out->card_rows);
    for (const auto &f : face_of) out->present[f.second] = 1;
    out->names.assign(out->card_rows, std::string());
    const int c_card = column_index(*card, "card"), c_face = column_index(*card, "face"),
              c_name = column_index(*card, "name");
    if (c_card >= 0 && c_face >= 0 && c_name >= 0)
        for (const std::vector<rmstate::Value> &row : rows_of(cards, card)) {
            auto it = face_of.find({row[static_cast<size_t>(c_card)].bits, row[static_cast<size_t>(c_face)].bits});
            if (it != face_of.end()) out->names[it->second] = row[static_cast<size_t>(c_name)].text;
        }
    for (const rmstate::TableDef &def : schema.tables) {
        if (!is_face_child(def)) continue;
        PrintedRows p;
        p.table = def.name;
        for (size_t c = 2; c < def.columns.size(); ++c) p.columns.push_back(def.columns[c].name);
        std::vector<std::vector<std::vector<rmstate::Value>>> per_face(out->card_rows);
        for (const std::vector<rmstate::Value> &row : rows_of(cards, &def)) {
            auto it = face_of.find({row[0].bits, row[1].bits});
            if (it == face_of.end()) continue;
            per_face[it->second].emplace_back(row.begin() + 2, row.end());
            for (size_t c = 2; c < def.columns.size(); ++c)
                if (def.columns[c].storage == rmstate::Storage::TEXT) per_face[it->second].back()[c - 2].bits = 0;
        }
        p.offsets.push_back(0);
        for (std::vector<std::vector<rmstate::Value>> &rows : per_face) {
            std::sort(rows.begin(), rows.end(), printed_row_less);
            for (std::vector<rmstate::Value> &r : rows) p.rows.push_back(std::move(r));
            p.offsets.push_back(p.rows.size());
        }
        out->printed.push_back(std::move(p));
    }
}

int64_t CardTables::key_row(const std::string &key) const {
    auto it = key_index.find(key);
    return it == key_index.end() ? -1 : it->second;
}

bool CardTables::build_index() {
    key_index.clear();
    if (abilities_per_card == 0 || key_base != (card_rows - 1) * abilities_per_card || ability_card.size() < key_base ||
        ability_index.size() != ability_card.size() || key_rows.size() != ab_keys.size())
        return false;
    std::vector<int64_t> next(card_rows, 0);
    for (size_t r = 0; r < ability_card.size(); ++r) {
        const int64_t c = ability_card[r];
        if (c < 0) {
            if (ability_index[r] != -1) return false;
            continue;
        }
        const size_t ci = static_cast<size_t>(c);
        if (r >= key_base || ci + 1 >= card_rows || ability_index[r] != next[ci] ||
            r != ci * abilities_per_card + static_cast<size_t>(ability_index[r]))
            return false;
        ++next[ci];
    }
    for (size_t k = 0; k < ab_keys.size(); ++k) {
        if (key_rows[k] < static_cast<int64_t>(key_base) || key_rows[k] >= static_cast<int64_t>(ability_card.size()) ||
            !key_index.emplace(ab_keys[k], key_rows[k]).second)
            return false;
    }
    return true;
}

bool build_card_tables(const rmstate::Schema &schema, const rmstate::DecodedFrame &cards, uint64_t card_rows,
    uint64_t vocab_hash, CardTables *out, std::string *err) {
    Registry registry = builtin_registry();
    return build_card_tables(schema, cards, card_rows, vocab_hash, &registry, out, err);
}

bool build_card_tables(const rmstate::Schema &schema, const rmstate::DecodedFrame &cards, uint64_t card_rows,
    uint64_t vocab_hash, Registry *registry, CardTables *out, std::string *err) {
    *out = CardTables();
    out->vocab_hash = vocab_hash;
    out->card_rows = card_rows;
    if (card_rows < 2) {
        *err = "the card tables need at least one card row and the none row";
        return false;
    }
    if (!registry->ok()) {
        *err = registry_error(*registry);
        return false;
    }
    if (!build_card_rows(schema, cards, registry, out, err) || !build_ability_rows(schema, cards, registry, out, err) ||
        !build_part_rows(schema, cards, registry, out, err))
        return false;
    build_printed_rows(schema, cards, out);
    return true;
}

CardTableFiles card_table_files(const CardTables &t, uint64_t spec_hash) {
    CardTableFiles f;
    RmtbFile *files[] = {&f.card, &f.ability, &f.part};
    const char *kinds[] = {"card", "ability", "part"};
    for (int i = 0; i < 3; ++i) {
        files[i]->kind = kinds[i];
        files[i]->has_vocab_hash = files[i]->has_spec_hash = true;
        files[i]->vocab_hash = t.vocab_hash;
        files[i]->spec_hash = spec_hash;
    }
    f.card.arrays.push_back(rmtb_floats(FEATURES, Dtype::F32, {t.card_rows, t.card_columns.size()}, t.card));
    f.card.arrays.back().columns = t.card_columns;
    f.card.arrays.push_back(rmtb_ints("back_face", Dtype::I64, {t.card_rows}, t.back_face));
    f.ability.arrays.push_back(
        rmtb_floats(FEATURES, Dtype::F32, {t.ability_rows(), t.ability_columns.size()}, t.ability));
    f.ability.arrays.back().columns = t.ability_columns;
    f.ability.arrays.push_back(rmtb_ints("ab_card", Dtype::I64, {t.ability_rows()}, t.ability_card));
    f.ability.arrays.push_back(rmtb_ints("ab_index", Dtype::I64, {t.ability_rows()}, t.ability_index));
    f.ability.arrays.push_back(rmtb_ints("key_rows", Dtype::I64, {t.key_rows.size()}, t.key_rows));
    f.ability.arrays.back().columns = t.ab_keys;
    f.ability.set_attr("abilities_per_card", static_cast<int64_t>(t.abilities_per_card));
    f.ability.set_attr("key_base", static_cast<int64_t>(t.key_base));
    f.part.arrays.push_back(rmtb_floats(FEATURES, Dtype::F32, {t.part_rows(), t.part_columns.size()}, t.part));
    f.part.arrays.back().columns = t.part_columns;
    f.part.arrays.push_back(rmtb_ints("ability", Dtype::I64, {t.part_rows()}, t.part_ability));
    return f;
}

bool read_card_tables(const CardTableFiles &files, CardTables *out, std::string *err) {
    *out = CardTables();
    if (files.card.kind != "card" || files.ability.kind != "ability" || files.part.kind != "part") {
        *err = "the card tables' files are not of kinds card, ability and part";
        return false;
    }
    if (!files.card.has_vocab_hash || files.ability.vocab_hash != files.card.vocab_hash ||
        files.part.vocab_hash != files.card.vocab_hash) {
        *err = "the card tables' files have different vocab hashes";
        return false;
    }
    out->vocab_hash = files.card.vocab_hash;
    uint64_t ab_rows = 0, part_rows = 0;
    const RmtbArray *keys = nullptr;
    if (!read_features(files.card, &out->card_columns, &out->card, &out->card_rows, err) ||
        !read_ints(files.card, "back_face", out->card_rows, &out->back_face, err) ||
        !read_features(files.ability, &out->ability_columns, &out->ability, &ab_rows, err) ||
        !read_ints(files.ability, "ab_card", ab_rows, &out->ability_card, err) ||
        !read_ints(files.ability, "ab_index", ab_rows, &out->ability_index, err) ||
        !read_array(files.ability, "key_rows", 1, &keys, err) ||
        !read_ints(files.ability, "key_rows", keys->shape[0], &out->key_rows, err) ||
        !read_features(files.part, &out->part_columns, &out->part, &part_rows, err) ||
        !read_ints(files.part, "ability", part_rows, &out->part_ability, err))
        return false;
    out->ab_keys = keys->columns;
    const RmtbAttr *per = files.ability.attr("abilities_per_card"), *base = files.ability.attr("key_base");
    if (!per || !per->is_int || !base || !base->is_int || per->i < 0 || base->i < 0) {
        *err = "the ability file has no abilities_per_card and key_base attributes";
        return false;
    }
    out->abilities_per_card = static_cast<uint32_t>(per->i);
    out->key_base = static_cast<uint64_t>(base->i);
    if (out->ab_keys.size() != out->key_rows.size() || !out->build_index()) {
        *err = "the ability file's rows and keys do not form an index";
        return false;
    }
    return true;
}

}  // namespace featurize
