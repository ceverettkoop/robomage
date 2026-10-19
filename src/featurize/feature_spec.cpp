#include "feature_spec.h"

#include <algorithm>

#include "json.h"
#include "rmtb.h"

namespace {

// The columns treated otherwise than by their tag (docs/featurizer.md §3.4); "*" is any table.
const featurize::Override OVERRIDES[] = {
    {"object", "card", "current_face"},
    {"player_counter", "kind", "hashed"},
    {"object_keyword", "text", "hashed"},
    {"object_counter", "kind", "hashed"},
    {"player_effect", "filter", "hashed"},
    {"*", "ab_key", "hashed"},
    {"menu", "description", "dropped"},
    {"k_template", "list", "row:meta_list"},
    {"k_archetype", "archetype", "row:meta_archetype"},
    {"b_list", "list", "row:meta_list"},
    {"b_archetype", "archetype", "row:meta_archetype"},
};

}  // namespace

static bool is_token_table(const rmstate::TableDef &def);
static const featurize::Override *find_override(const std::string &table, const std::string &column);
static bool is_abref_at(const rmstate::TableDef &def, size_t c);
static bool seat_with_partner(const rmstate::TableDef &def, size_t c);
static uint64_t metagame_rows(
    const rmstate::Schema &schema, const rmstate::DecodedFrame &metagame, const std::string &table, bool *is_meta);
static void note_metagame(featurize::FeatureSpec *spec, const std::string &table, uint64_t rows);
static void add_categorical(featurize::FeatureSpec *spec, featurize::TokenType *t, const std::string &column,
    const std::string &source, uint32_t values, featurize::ColumnOp *op);
static int32_t add_relation(
    featurize::FeatureSpec *spec, featurize::TokenType *t, const std::string &column, const std::string &target);
static void add_numeric(featurize::TokenType *t, const std::string &name, const std::string &column,
    featurize::Transform transform, uint32_t bit);
static void add_card(featurize::TokenType *t, const rmstate::TableDef &def, size_t c, bool current_face);
static void plan_column(const rmstate::Schema &schema, const rmstate::DecodedFrame &metagame,
    const rmstate::TableDef &def, size_t c, featurize::FeatureSpec *spec, featurize::TokenType *t);
static void plan_type(const rmstate::Schema &schema, const rmstate::DecodedFrame &metagame,
    const rmstate::TableDef &def, featurize::FeatureSpec *spec, featurize::TokenType *t);
static std::string json_strings(const std::vector<std::string> &items);
static std::string json_type(const featurize::TokenType &t, size_t id);
static std::string json_table(uint64_t rows, size_t width, const std::vector<std::string> *columns);

// A token type: a STATE snapshot (20-48) or knowledge (90-99) table.
static bool is_token_table(const rmstate::TableDef &def) {
    const uint16_t id = static_cast<uint16_t>(def.id);
    return def.frame_kind == rmstate::FrameKind::STATE && ((id >= 20 && id <= 48) || (id >= 90 && id <= 99));
}

static const featurize::Override *find_override(const std::string &table, const std::string &column) {
    for (const featurize::Override &o : OVERRIDES)
        if ((o.table == table || o.table == "*") && o.column == column) return &o;
    return nullptr;
}

// Column c starts an ability reference: ab_card, ab_index, ab_key.
static bool is_abref_at(const rmstate::TableDef &def, size_t c) {
    return c + 2 < def.columns.size() && def.columns[c].name == "ab_card" && def.columns[c].tag == "card" &&
           def.columns[c + 1].name == "ab_index" && def.columns[c + 2].name == "ab_key";
}

// Column c is a seat pair's absolute seat: enum:seat with a player-rel column <name>_rel.
static bool seat_with_partner(const rmstate::TableDef &def, size_t c) {
    if (def.columns[c].tag != "enum:seat") return false;
    const int rel = featurize::column_index(def, def.columns[c].name + "_rel");
    return rel >= 0 && def.columns[static_cast<size_t>(rel)].tag == "player-rel";
}

// The rows of `table` in the METAGAME frame; `is_meta` whether it is a METAGAME table at all.
static uint64_t metagame_rows(
    const rmstate::Schema &schema, const rmstate::DecodedFrame &metagame, const std::string &table, bool *is_meta) {
    const rmstate::TableDef *def = featurize::table_named(schema, table);
    *is_meta = def && def->frame_kind == rmstate::FrameKind::METAGAME;
    if (!*is_meta) return 0;
    const rmstate::DecodedTable *t = metagame.table(static_cast<uint16_t>(def->id));
    return t ? t->rows.size() : 0;
}

static void note_metagame(featurize::FeatureSpec *spec, const std::string &table, uint64_t rows) {
    for (const auto &m : spec->metagame_rows)
        if (m.first == table) return;
    spec->metagame_rows.emplace_back(table, rows);
}

// A categorical slot of `values` values (and the none id) for `op`.
static void add_categorical(featurize::FeatureSpec *spec, featurize::TokenType *t, const std::string &column,
    const std::string &source, uint32_t values, featurize::ColumnOp *op) {
    featurize::CatSlot s;
    s.name = column;
    s.column = column;
    s.source = source;
    s.offset = spec->categorical_size;
    s.size = values + 1;
    op->slot = static_cast<uint32_t>(t->categorical.size());
    op->offset = s.offset;
    op->width = values;
    spec->categorical_size += s.size;
    t->categorical.push_back(s);
}

static int32_t add_relation(
    featurize::FeatureSpec *spec, featurize::TokenType *t, const std::string &column, const std::string &target) {
    const int32_t id = static_cast<int32_t>(spec->relations.size());
    spec->relations.push_back({t->name, column, target});
    t->relations.push_back(id);
    return id;
}

static void add_numeric(featurize::TokenType *t, const std::string &name, const std::string &column,
    featurize::Transform transform, uint32_t bit) {
    featurize::NumSlot s;
    s.name = name;
    s.column = column;
    s.transform = transform;
    s.bit = bit;
    t->numeric.push_back(s);
}

// A card column: its card slot, the current face's slot when asked, and the none flag.
static void add_card(featurize::TokenType *t, const rmstate::TableDef &def, size_t c, bool current_face) {
    const rmstate::ColumnDef &col = def.columns[c];
    featurize::ColumnOp op;
    op.kind = featurize::OpKind::CARD;
    op.col = static_cast<uint16_t>(c);
    op.storage = col.storage;
    op.slot = static_cast<uint32_t>(t->cards.size());
    t->cards.push_back({col.name, col.name, false});
    const int transformed = current_face ? featurize::column_index(def, "transformed") : -1;
    if (transformed >= 0) {
        op.face_slot = static_cast<int32_t>(t->cards.size());
        op.aux1 = static_cast<uint16_t>(transformed);
        op.aux1_storage = def.columns[static_cast<size_t>(transformed)].storage;
        t->cards.push_back({col.name + ":face", col.name, true});
    }
    op.flag = static_cast<int32_t>(t->numeric.size());
    add_numeric(t, col.name + ":none", col.name, featurize::Transform::NONE_FLAG, 0);
    t->ops.push_back(op);
}

// Plans column c of token type `t` (the header comment's rules).
static void plan_column(const rmstate::Schema &schema, const rmstate::DecodedFrame &metagame,
    const rmstate::TableDef &def, size_t c, featurize::FeatureSpec *spec, featurize::TokenType *t) {
    const rmstate::ColumnDef &col = def.columns[c];
    const featurize::Override *ov = find_override(def.name, col.name);
    if (ov && ov->treatment == "dropped") return;
    if (seat_with_partner(def, c)) return;
    featurize::ColumnOp op;
    op.col = static_cast<uint16_t>(c);
    op.storage = col.storage;
    if (ov && ov->treatment == "hashed") {
        op.kind = featurize::OpKind::HASHED;
        add_categorical(spec, t, col.name, "hashed", spec->hash_buckets, &op);
        t->ops.push_back(op);
        return;
    }
    const featurize::Tag tag = ov && ov->treatment.compare(0, 4, "row:") == 0 ? featurize::parse_tag(ov->treatment)
                                                                              : featurize::parse_tag(col.tag);
    switch (tag.kind) {
        case featurize::TagKind::BOOL:
        case featurize::TagKind::COUNT:
        case featurize::TagKind::SIGNED:
        case featurize::TagKind::PROB:
            op.kind = featurize::OpKind::NUMERIC;
            featurize::scalar_transform(tag.kind, &op.transform);
            op.slot = static_cast<uint32_t>(t->numeric.size());
            add_numeric(t, col.name, col.name, op.transform, 0);
            break;
        case featurize::TagKind::INDEX:
            op.kind = featurize::OpKind::INDEX;
            op.slot = static_cast<uint32_t>(t->numeric.size());
            add_numeric(t, col.name, col.name, featurize::Transform::INDEX, 0);
            op.flag = static_cast<int32_t>(t->numeric.size());
            add_numeric(t, col.name + ":none", col.name, featurize::Transform::NONE_FLAG, 0);
            break;
        case featurize::TagKind::MASK: {
            const std::vector<std::string> &values = featurize::enum_values(schema, tag.arg);
            op.kind = featurize::OpKind::MASK;
            op.slot = static_cast<uint32_t>(t->numeric.size());
            op.width = static_cast<uint32_t>(values.size());
            for (size_t v = 0; v < values.size(); ++v)
                add_numeric(
                    t, col.name + ":" + values[v], col.name, featurize::Transform::MASK_BIT, static_cast<uint32_t>(v));
            break;
        }
        case featurize::TagKind::ENUM:
            op.kind = featurize::OpKind::CATEGORICAL;
            add_categorical(
                spec, t, col.name, col.tag, static_cast<uint32_t>(featurize::enum_values(schema, tag.arg).size()), &op);
            break;
        case featurize::TagKind::PLAYER_REL:
            op.kind = featurize::OpKind::CATEGORICAL;
            add_categorical(spec, t, col.name, col.tag, col.has_range ? static_cast<uint32_t>(col.max + 1) : 3, &op);
            break;
        case featurize::TagKind::CARD:
            if (is_abref_at(def, c)) {
                op.kind = featurize::OpKind::ABILITY;
                op.slot = static_cast<uint32_t>(t->abilities.size());
                op.aux1 = static_cast<uint16_t>(c + 1);
                op.aux2 = static_cast<uint16_t>(c + 2);
                op.aux1_storage = def.columns[c + 1].storage;
                t->abilities.push_back("ability");
                break;
            }
            add_card(t, def, c, ov && ov->treatment == "current_face");
            return;
        case featurize::TagKind::HANDLE:
            op.kind = featurize::OpKind::HANDLE;
            op.relation = add_relation(spec, t, col.name, "handle");
            break;
        case featurize::TagKind::ROW: {
            const rmstate::TableDef *target = featurize::table_named(schema, tag.arg);
            const int32_t tt = target ? spec->type_of_table[static_cast<uint16_t>(target->id)] : -1;
            if (tt >= 0) {
                op.kind = featurize::OpKind::ROW;
                op.target = tt;
                op.relation = add_relation(spec, t, col.name, tag.arg);
                break;
            }
            bool is_meta = false;
            const uint64_t rows = metagame_rows(schema, metagame, tag.arg, &is_meta);
            if (!is_meta) return;
            note_metagame(spec, tag.arg, rows);
            op.kind = featurize::OpKind::CATEGORICAL;
            add_categorical(spec, t, col.name, "row:" + tag.arg, static_cast<uint32_t>(rows), &op);
            break;
        }
        default:
            return;
    }
    t->ops.push_back(op);
}

static void plan_type(const rmstate::Schema &schema, const rmstate::DecodedFrame &metagame,
    const rmstate::TableDef &def, featurize::FeatureSpec *spec, featurize::TokenType *t) {
    t->owns_handles = !def.columns.empty() && def.columns[0].name == "handle" && def.columns[0].tag == "handle";
    for (size_t c = 0; c < def.columns.size(); ++c) {
        plan_column(schema, metagame, def, c, spec, t);
        if (is_abref_at(def, c)) ++c;  // ab_index belongs to the reference; ab_key is planned next
    }
}

static std::string json_strings(const std::vector<std::string> &items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) out += (i ? "," : "") + featurize::json_quote(items[i]);
    return out + "]";
}

static std::string json_type(const featurize::TokenType &t, size_t id) {
    std::string out = "{\"id\":" + std::to_string(id) + ",\"table\":" + std::to_string(t.table) +
                      ",\"name\":" + featurize::json_quote(t.name) +
                      ",\"owns_handles\":" + (t.owns_handles ? "true" : "false") + ",\"numeric\":[";
    for (size_t i = 0; i < t.numeric.size(); ++i) {
        const featurize::NumSlot &s = t.numeric[i];
        out += std::string(i ? "," : "") + "{\"name\":" + featurize::json_quote(s.name) +
               ",\"column\":" + featurize::json_quote(s.column) + ",\"transform\":\"" +
               featurize::transform_name(s.transform) + "\"";
        if (s.transform == featurize::Transform::MASK_BIT) out += ",\"bit\":" + std::to_string(s.bit);
        out += "}";
    }
    out += "],\"categorical\":[";
    for (size_t i = 0; i < t.categorical.size(); ++i) {
        const featurize::CatSlot &s = t.categorical[i];
        out += std::string(i ? "," : "") + "{\"name\":" + featurize::json_quote(s.name) +
               ",\"column\":" + featurize::json_quote(s.column) + ",\"source\":" + featurize::json_quote(s.source) +
               ",\"offset\":" + std::to_string(s.offset) + ",\"size\":" + std::to_string(s.size) + "}";
    }
    out += "],\"cards\":[";
    for (size_t i = 0; i < t.cards.size(); ++i)
        out += std::string(i ? "," : "") + "{\"name\":" + featurize::json_quote(t.cards[i].name) +
               ",\"column\":" + featurize::json_quote(t.cards[i].column) +
               (t.cards[i].current_face ? ",\"face\":\"current\"}" : "}");
    out += "],\"abilities\":[";
    for (size_t i = 0; i < t.abilities.size(); ++i)
        out += std::string(i ? "," : "") + "{\"name\":" + featurize::json_quote(t.abilities[i]) +
               ",\"columns\":[\"ab_card\",\"ab_index\",\"ab_key\"]}";
    out += "],\"relations\":[";
    for (size_t i = 0; i < t.relations.size(); ++i) out += (i ? "," : "") + std::to_string(t.relations[i]);
    return out + "]}";
}

static std::string json_table(uint64_t rows, size_t width, const std::vector<std::string> *columns) {
    std::string out = "{\"rows\":" + std::to_string(rows) + ",\"width\":" + std::to_string(width);
    if (columns) out += ",\"columns\":" + json_strings(*columns);
    return out + "}";
}

namespace featurize {

bool build_spec(const rmstate::Schema &schema, const rmstate::DecodedFrame &metagame, const CardTables &tables,
    FeatureSpec *out, std::string *err) {
    *out = FeatureSpec();
    out->schema_hash = schema.hash;
    out->vocab_hash = tables.vocab_hash;
    out->categorical_size = 1;  // id 0: the pad
    out->type_of_table.assign(65536, -1);
    for (const rmstate::TableDef &def : schema.tables) {
        if (!is_token_table(def)) continue;
        out->type_of_table[static_cast<uint16_t>(def.id)] = static_cast<int32_t>(out->types.size());
        out->types.emplace_back();
        out->types.back().table = static_cast<uint16_t>(def.id);
        out->types.back().name = def.name;
    }
    for (TokenType &t : out->types) {
        const rmstate::TableDef &def = *schema.find_table(t.table);
        plan_type(schema, metagame, def, out, &t);
        out->F = std::max(out->F, t.numeric.size());
        out->C = std::max(out->C, t.categorical.size());
        out->K = std::max(out->K, t.cards.size());
        out->Q = std::max(out->Q, t.abilities.size());
        if (t.name == "menu") out->menu_type = static_cast<int32_t>(&t - &out->types[0]);
        if (t.name == "header") {
            out->header_type = static_cast<int32_t>(&t - &out->types[0]);
            out->header_game = column_index(def, "game");
            out->header_decision = column_index(def, "decision");
            out->header_sideboarding = column_index(def, "sideboarding");
        }
    }
    if (out->menu_type < 0 || out->header_type < 0 || out->header_game < 0 || out->header_decision < 0 ||
        out->header_sideboarding < 0) {
        *err = "the schema has no menu table or no header table with game, decision and sideboarding";
        return false;
    }
    for (const Override &o : OVERRIDES) out->overrides.push_back(o);
    out->card_rows = tables.card_rows;
    out->text_rows = tables.text_rows;
    out->text_width = tables.text_width;
    out->ability_rows = tables.ability_rows();
    out->part_rows = tables.part_rows();
    out->card_columns = tables.card_columns;
    out->ability_columns = tables.ability_columns;
    out->part_columns = tables.part_columns;
    out->ab_keys = tables.ab_keys;
    out->json = spec_json(*out);
    out->hash = rmstate::fnv1a64(out->json.data(), out->json.size());
    return true;
}

std::string spec_json(const FeatureSpec &spec) {
    std::string out = "{\"format\":\"FEATURE_SPEC\",\"version\":" + std::to_string(SPEC_VERSION) +
                      ",\"schema_hash\":\"" + hex16(spec.schema_hash) + "\",\"vocab_hash\":\"" +
                      hex16(spec.vocab_hash) + "\",\"hash_buckets\":" + std::to_string(spec.hash_buckets) +
                      ",\"library_depth\":" + std::to_string(spec.library_depth) + ",\"metagame\":{";
    for (size_t i = 0; i < spec.metagame_rows.size(); ++i)
        out += (i ? "," : "") + json_quote(spec.metagame_rows[i].first) + ":" +
               std::to_string(spec.metagame_rows[i].second);
    out += "},\"F\":" + std::to_string(spec.F) + ",\"C\":" + std::to_string(spec.C) +
           ",\"K\":" + std::to_string(spec.K) + ",\"Q\":" + std::to_string(spec.Q) +
           ",\"categorical_size\":" + std::to_string(spec.categorical_size) +
           ",\"menu\":" + json_quote(spec.types[static_cast<size_t>(spec.menu_type)].name) + ",\"token_types\":[";
    for (size_t i = 0; i < spec.types.size(); ++i) out += (i ? "," : "") + json_type(spec.types[i], i);
    out += "],\"relation_types\":[";
    for (size_t i = 0; i < spec.relations.size(); ++i)
        out += std::string(i ? "," : "") + "{\"id\":" + std::to_string(i) +
               ",\"table\":" + json_quote(spec.relations[i].table) +
               ",\"column\":" + json_quote(spec.relations[i].column) +
               ",\"target\":" + json_quote(spec.relations[i].target) + "}";
    out += "],\"overrides\":[";
    for (size_t i = 0; i < spec.overrides.size(); ++i)
        out += std::string(i ? "," : "") + "{\"table\":" + json_quote(spec.overrides[i].table) +
               ",\"column\":" + json_quote(spec.overrides[i].column) +
               ",\"treatment\":" + json_quote(spec.overrides[i].treatment) + "}";
    out += "],\"card_tables\":{\"card\":" + json_table(spec.card_rows, spec.card_columns.size(), &spec.card_columns) +
           ",\"text\":" + json_table(spec.text_rows, spec.text_width, nullptr) +
           ",\"ability\":" + json_table(spec.ability_rows, spec.ability_columns.size(), &spec.ability_columns) +
           ",\"part\":" + json_table(spec.part_rows, spec.part_columns.size(), &spec.part_columns) +
           "},\"ab_keys\":" + json_strings(spec.ab_keys) + "}";
    return out;
}

}  // namespace featurize
