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
};

// The folds (docs/featurizer.md §3.6): a LEDGER rule folds the whole table into the synthetic type
// `into`; a FILTERED or DELTA rule keeps the table's type, of that kind (`printed`:
// the CARDS table a delta compares with).
struct FoldRule {
        const char *table, *into, *rows;
        featurize::TypeKind kind;
        const char *printed;
};

const FoldRule FOLDS[] = {
    {"own_deck", "ledger", "every row", featurize::TypeKind::LEDGER, ""},
    {"k_seen", "ledger", "every row", featurize::TypeKind::LEDGER, ""},
    {"k_known_hand", "ledger", "every row", featurize::TypeKind::LEDGER, ""},
    {"k_pool_card", "ledger",
        "the rows of a HAND or SIDEBOARD pool and of a player's only LIBRARY_RANGE pool when it is a run",
        featurize::TypeKind::FILTERED, ""},
    {"object_subtype", "object_subtype", "the differences from card_subtype", featurize::TypeKind::DELTA,
        "card_subtype"},
    {"object_keyword", "object_keyword", "the differences from card_keyword", featurize::TypeKind::DELTA,
        "card_keyword"},
};

// The ledger's numeric slots (featurize::LedgerSlot order): name, source, transform.
struct LedgerSlotDef {
        const char *name, *column;
        featurize::Transform transform;
};

const LedgerSlotDef LEDGER_SLOTS[] = {
    {"deck", "own_deck", featurize::Transform::BOOL},
    {"main", "own_deck.main", featurize::Transform::LOG1P},
    {"side", "own_deck.side", featurize::Transform::LOG1P},
    {"seen_game", "k_seen.game", featurize::Transform::LOG1P},
    {"seen_match", "k_seen.match", featurize::Transform::LOG1P},
    {"known_hand", "k_known_hand.copies", featurize::Transform::LOG1P},
    {"hand_min", "k_pool_card.min", featurize::Transform::LOG1P},
    {"hand_max", "k_pool_card.max", featurize::Transform::LOG1P},
    {"library_min", "k_pool_card.min,k_position", featurize::Transform::LOG1P},
    {"library_max", "k_pool_card.max,k_position", featurize::Transform::LOG1P},
    {"sideboard_min", "k_pool_card.min", featurize::Transform::LOG1P},
    {"sideboard_max", "k_pool_card.max", featurize::Transform::LOG1P},
};

// The event tables' columns a history token leaves out: they identify the game rather than describe
// the event (docs/featurizer.md §3.7).
const char *const HISTORY_DROPPED[] = {
    "event.seq",
    "event.turn",
    "event.game",
    "event.batch",
    "event_card.draw_seq",
    "event_card.draw_idx",
};

}  // namespace

static const FoldRule *find_fold(const std::string &table);
static bool is_token_table(const rmstate::TableDef &def);
static const featurize::Override *find_override(const std::string &table, const std::string &column);
static bool is_abref_at(const rmstate::TableDef &def, size_t c);
static bool seat_with_partner(const rmstate::TableDef &def, size_t c);
static int32_t enum_map_id(
    const rmstate::Schema &schema, featurize::Registry *reg, featurize::FeatureSpec *spec, const std::string &name);
static bool add_categorical(featurize::FeatureSpec *spec, featurize::Registry *reg, featurize::TokenType *t,
    const std::string &column, const std::string &source, uint32_t values, featurize::ColumnOp *op);
static int32_t add_relation(featurize::FeatureSpec *spec, featurize::Registry *reg, featurize::TokenType *t,
    const std::string &column, const std::string &target);
static int32_t add_numeric(featurize::Registry *reg, featurize::TokenType *t, const std::string &name,
    const std::string &column, featurize::Transform transform);
static void add_card(
    featurize::Registry *reg, featurize::TokenType *t, const rmstate::TableDef &def, size_t c, bool current_face);
static bool plan_mask(const rmstate::Schema &schema, featurize::Registry *reg, featurize::FeatureSpec *spec,
    featurize::TokenType *t, const rmstate::ColumnDef &col, const std::string &enum_name, featurize::ColumnOp *op);
static void plan_column(const rmstate::Schema &schema, const rmstate::TableDef &def, size_t c,
    featurize::FeatureSpec *spec, featurize::Registry *reg, featurize::TokenType *t);
static void plan_type(const rmstate::Schema &schema, const rmstate::TableDef &def, featurize::FeatureSpec *spec,
    featurize::Registry *reg, featurize::TokenType *t);
static bool plan_delta(const rmstate::Schema &schema, const rmstate::TableDef &def, const FoldRule &rule,
    featurize::Registry *reg, featurize::TokenType *t);
static void add_rel_slot(featurize::FeatureSpec *spec, featurize::Registry *reg, featurize::TokenType *t);
static void plan_ledger(
    const rmstate::Schema &schema, featurize::FeatureSpec *spec, featurize::Registry *reg, featurize::TokenType *t);
static bool has_ledger(const rmstate::Schema &schema);
static void plan_object(const rmstate::Schema &schema, featurize::FeatureSpec *spec);
static bool history_dropped(const std::string &table, const std::string &column);
static void plan_history_type(const rmstate::Schema &schema, const rmstate::TableDef &def, featurize::FeatureSpec *spec,
    featurize::Registry *reg, featurize::TokenType *t);
static bool history_tables(const rmstate::Schema &schema, featurize::HistoryPlan *h, std::string *err);
static void plan_history(const rmstate::Schema &schema, const featurize::FeatureOptions &options,
    featurize::FeatureSpec *spec, featurize::Registry *reg);
static void place_type(featurize::Registry *reg, featurize::TokenType *t);
static std::string json_history(const featurize::FeatureSpec &spec);
static std::string json_strings(const std::vector<std::string> &items);
static std::string json_ints(const std::vector<int64_t> &items);
static std::string json_type(const featurize::TokenType &t, size_t id);
static std::string json_table(uint64_t rows, size_t width, const std::vector<std::string> *columns);

static const FoldRule *find_fold(const std::string &table) {
    for (const FoldRule &f : FOLDS)
        if (table == f.table) return &f;
    return nullptr;
}

// A token type: a STATE snapshot (20-48) or knowledge (90-95) table.
static bool is_token_table(const rmstate::TableDef &def) {
    const uint16_t id = static_cast<uint16_t>(def.id);
    return def.frame_kind == rmstate::FrameKind::STATE && ((id >= 20 && id <= 48) || (id >= 90 && id <= 95));
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

// The enum map (engine value → registry index) of enum `name`: its FeatureSpec::enum_maps index.
static int32_t enum_map_id(
    const rmstate::Schema &schema, featurize::Registry *reg, featurize::FeatureSpec *spec, const std::string &name) {
    for (size_t i = 0; i < spec->enum_names.size(); ++i)
        if (spec->enum_names[i] == name) return static_cast<int32_t>(i);
    spec->enum_names.push_back(name);
    spec->enum_maps.push_back(reg->enum_map(name, featurize::enum_values(schema, name)));
    const featurize::RegEnum *e = reg->find_enum(name);
    spec->enum_values.push_back(e ? e->values : std::vector<std::string>());
    return static_cast<int32_t>(spec->enum_maps.size() - 1);
}

// A categorical slot for `op`: of an enum (source enum:<e>) its capacity's ids and the enum map,
// else `values` values; and the none id. False (a registry problem) leaves `op` unplanned.
static bool add_categorical(featurize::FeatureSpec *spec, featurize::Registry *reg, featurize::TokenType *t,
    const std::string &column, const std::string &source, uint32_t values, featurize::ColumnOp *op) {
    uint32_t size = values + 1;
    if (source.compare(0, 5, "enum:") == 0) {
        const featurize::RegEnum *e = reg->find_enum(source.substr(5));
        if (!e) return false;
        size = e->capacity + 1;
    }
    featurize::RegCat c;
    if (!reg->categorical_slot(t->name, column, source, size, &c)) return false;
    featurize::CatSlot s;
    s.name = column;
    s.column = column;
    s.source = source;
    s.offset = c.offset;
    s.size = c.size;
    s.col = c.col;
    op->slot = c.col;
    op->offset = c.offset;
    op->width = values;
    t->categorical.push_back(s);
    return true;
}

static int32_t add_relation(featurize::FeatureSpec *spec, featurize::Registry *reg, featurize::TokenType *t,
    const std::string &column, const std::string &target) {
    const int32_t id = reg->relation_id(t->name, column, target);
    if (id < 0) return -1;
    if (spec->relations.size() <= static_cast<size_t>(id)) spec->relations.resize(static_cast<size_t>(id) + 1);
    spec->relations[static_cast<size_t>(id)] = {t->name, column, target};
    t->relations.push_back(id);
    return id;
}

// A scalar numeric slot: its column (-1: a registry problem).
static int32_t add_numeric(featurize::Registry *reg, featurize::TokenType *t, const std::string &name,
    const std::string &column, featurize::Transform transform) {
    const int32_t at = reg->numeric_at(t->name, name, "scalar", "");
    if (at < 0) return -1;
    featurize::NumSlot s;
    s.name = name;
    s.column = column;
    s.transform = transform;
    s.at = static_cast<uint32_t>(at);
    t->numeric.push_back(s);
    return at;
}

// A card column: its card slot, the current face's slot when asked, and the none flag.
static void add_card(
    featurize::Registry *reg, featurize::TokenType *t, const rmstate::TableDef &def, size_t c, bool current_face) {
    const rmstate::ColumnDef &col = def.columns[c];
    featurize::ColumnOp op;
    op.kind = featurize::OpKind::CARD;
    op.col = static_cast<uint16_t>(c);
    op.storage = col.storage;
    const int32_t at = reg->card_col(t->name, col.name);
    if (at < 0) return;
    op.slot = static_cast<uint32_t>(at);
    t->cards.push_back({col.name, col.name, false, op.slot});
    const int transformed = current_face ? featurize::column_index(def, "transformed") : -1;
    if (transformed >= 0) {
        const int32_t face = reg->card_col(t->name, col.name + ":face");
        if (face < 0) return;
        op.face_slot = face;
        op.aux1 = static_cast<uint16_t>(transformed);
        op.aux1_storage = def.columns[static_cast<size_t>(transformed)].storage;
        t->cards.push_back({col.name + ":face", col.name, true, static_cast<uint32_t>(face)});
    }
    op.flag = add_numeric(reg, t, col.name + ":none", col.name, featurize::Transform::NONE_FLAG);
    if (op.flag >= 0) t->ops.push_back(op);
}

// A mask column: its enum's block, a numeric slot per value the schema has at the block's start plus
// the value's registry index.
static bool plan_mask(const rmstate::Schema &schema, featurize::Registry *reg, featurize::FeatureSpec *spec,
    featurize::TokenType *t, const rmstate::ColumnDef &col, const std::string &enum_name, featurize::ColumnOp *op) {
    const std::vector<std::string> &values = featurize::enum_values(schema, enum_name);
    op->enum_map = enum_map_id(schema, reg, spec, enum_name);
    const int32_t start = reg->numeric_at(t->name, col.name, "mask", enum_name);
    if (start < 0) return false;
    const std::vector<int32_t> &map = spec->enum_maps[static_cast<size_t>(op->enum_map)];
    op->kind = featurize::OpKind::MASK;
    op->slot = static_cast<uint32_t>(start);
    op->width = static_cast<uint32_t>(values.size());
    for (size_t v = 0; v < values.size(); ++v) {
        if (map[v] < 0) continue;
        featurize::NumSlot s;
        s.name = col.name + ":" + values[v];
        s.column = col.name;
        s.transform = featurize::Transform::MASK_BIT;
        s.bit = static_cast<uint32_t>(v);
        s.at = static_cast<uint32_t>(start + map[v]);
        t->numeric.push_back(s);
    }
    return true;
}

// Plans column c of token type `t` (the header comment's rules).
static void plan_column(const rmstate::Schema &schema, const rmstate::TableDef &def, size_t c,
    featurize::FeatureSpec *spec, featurize::Registry *reg, featurize::TokenType *t) {
    const rmstate::ColumnDef &col = def.columns[c];
    const featurize::Override *ov = find_override(def.name, col.name);
    if (ov && ov->treatment == "dropped") return;
    if (seat_with_partner(def, c)) return;
    featurize::ColumnOp op;
    op.col = static_cast<uint16_t>(c);
    op.storage = col.storage;
    if (ov && ov->treatment == "hashed") {
        op.kind = featurize::OpKind::HASHED;
        if (add_categorical(spec, reg, t, col.name, "hashed", spec->hash_buckets, &op)) t->ops.push_back(op);
        return;
    }
    const featurize::Tag tag = featurize::parse_tag(col.tag);
    switch (tag.kind) {
        case featurize::TagKind::BOOL:
        case featurize::TagKind::COUNT:
        case featurize::TagKind::SIGNED: {
            op.kind = featurize::OpKind::NUMERIC;
            featurize::scalar_transform(tag.kind, &op.transform);
            const int32_t at = add_numeric(reg, t, col.name, col.name, op.transform);
            if (at < 0) return;
            op.slot = static_cast<uint32_t>(at);
            break;
        }
        case featurize::TagKind::INDEX: {
            op.kind = featurize::OpKind::INDEX;
            const int32_t at = add_numeric(reg, t, col.name, col.name, featurize::Transform::INDEX);
            op.flag = add_numeric(reg, t, col.name + ":none", col.name, featurize::Transform::NONE_FLAG);
            if (at < 0 || op.flag < 0) return;
            op.slot = static_cast<uint32_t>(at);
            break;
        }
        case featurize::TagKind::MASK:
            if (!plan_mask(schema, reg, spec, t, col, tag.arg, &op)) return;
            break;
        case featurize::TagKind::ENUM:
            op.kind = featurize::OpKind::CATEGORICAL;
            op.enum_map = enum_map_id(schema, reg, spec, tag.arg);
            if (!add_categorical(spec, reg, t, col.name, col.tag,
                    static_cast<uint32_t>(featurize::enum_values(schema, tag.arg).size()), &op))
                return;
            break;
        case featurize::TagKind::PLAYER_REL:
            op.kind = featurize::OpKind::CATEGORICAL;
            if (!add_categorical(
                    spec, reg, t, col.name, col.tag, col.has_range ? static_cast<uint32_t>(col.max + 1) : 3, &op))
                return;
            break;
        case featurize::TagKind::CARD:
            if (is_abref_at(def, c)) {
                const int32_t at = reg->ability_col(t->name, "ability");
                if (at < 0) return;
                op.kind = featurize::OpKind::ABILITY;
                op.slot = static_cast<uint32_t>(at);
                op.aux1 = static_cast<uint16_t>(c + 1);
                op.aux2 = static_cast<uint16_t>(c + 2);
                op.aux1_storage = def.columns[c + 1].storage;
                t->abilities.push_back({"ability", op.slot});
                break;
            }
            add_card(reg, t, def, c, ov && ov->treatment == "current_face");
            return;
        case featurize::TagKind::HANDLE:
            op.kind = featurize::OpKind::HANDLE;
            op.relation = add_relation(spec, reg, t, col.name, "handle");
            if (op.relation < 0) return;
            break;
        case featurize::TagKind::ROW: {
            const rmstate::TableDef *target = featurize::table_named(schema, tag.arg);
            const int32_t tt = target ? spec->type_of_table[static_cast<uint16_t>(target->id)] : -1;
            if (tt < 0) return;
            op.kind = featurize::OpKind::ROW;
            op.target = tt;
            op.relation = add_relation(spec, reg, t, col.name, tag.arg);
            if (op.relation < 0) return;
            break;
        }
        default:
            return;
    }
    t->ops.push_back(op);
}

static void plan_type(const rmstate::Schema &schema, const rmstate::TableDef &def, featurize::FeatureSpec *spec,
    featurize::Registry *reg, featurize::TokenType *t) {
    t->owns_handles = !def.columns.empty() && def.columns[0].name == "handle" && def.columns[0].tag == "handle";
    t->columns = static_cast<uint16_t>(def.columns.size());
    for (size_t c = 0; c < def.columns.size(); ++c) {
        plan_column(schema, def, c, spec, reg, t);
        if (is_abref_at(def, c)) ++c;  // ab_index belongs to the reference; ab_key is planned next
    }
}

// A DELTA type: every column but `obj` is a column of the printed table (after card and face), and
// every printed column is one of them. False leaves `t` a TABLE type.
static bool plan_delta(const rmstate::Schema &schema, const rmstate::TableDef &def, const FoldRule &rule,
    featurize::Registry *reg, featurize::TokenType *t) {
    const rmstate::TableDef *printed = featurize::table_named(schema, rule.printed);
    const int obj = featurize::column_index(def, "obj");
    if (!printed || obj < 0 || printed->columns.size() < 2 || def.columns.size() != printed->columns.size() - 1)
        return false;
    std::vector<uint16_t> cols;
    std::vector<uint8_t> text;
    for (size_t c = 2; c < printed->columns.size(); ++c) {
        const int at = featurize::column_index(def, printed->columns[c].name);
        if (at < 0 || at == obj || def.columns[static_cast<size_t>(at)].storage != printed->columns[c].storage)
            return false;
        cols.push_back(static_cast<uint16_t>(at));
        text.push_back(printed->columns[c].storage == rmstate::Storage::TEXT ? 1 : 0);
    }
    t->kind = featurize::TypeKind::DELTA;
    t->printed = rule.printed;
    t->obj_col = static_cast<uint16_t>(obj);
    t->value_cols = cols;
    t->value_text = text;
    t->removed_slot = add_numeric(reg, t, "removed", "", featurize::Transform::BOOL);
    return true;
}

// The player_rel categorical of a synthetic type (none, self, opponent).
static void add_rel_slot(featurize::FeatureSpec *spec, featurize::Registry *reg, featurize::TokenType *t) {
    featurize::ColumnOp op;
    if (!add_categorical(spec, reg, t, "player_rel", "player-rel", 3, &op)) return;
    t->rel_offset = op.offset;
    t->rel_values = op.width;
    t->rel_col = op.slot;
}

// The ledger's slots and the columns it reads.
static void plan_ledger(
    const rmstate::Schema &schema, featurize::FeatureSpec *spec, featurize::Registry *reg, featurize::TokenType *t) {
    featurize::LedgerPlan &l = spec->ledger;
    for (const LedgerSlotDef &d : LEDGER_SLOTS) add_numeric(reg, t, d.name, d.column, d.transform);
    add_rel_slot(spec, reg, t);
    const int32_t card = reg->card_col(t->name, "card");
    if (card >= 0) {
        t->card_col = static_cast<uint32_t>(card);
        t->cards.push_back({"card", "card", false, t->card_col});
    }
    const auto table = [&schema](const char *name) {
        const rmstate::TableDef *def = featurize::table_named(schema, name);
        return def && def->frame_kind == rmstate::FrameKind::STATE ? def : nullptr;
    };
    const auto col = [](const rmstate::TableDef *def, const char *name) {
        return def ? featurize::column_index(*def, name) : -1;
    };
    const rmstate::TableDef *deck = table("own_deck"), *seen = table("k_seen"), *known = table("k_known_hand"),
                            *pool = table("k_pool"), *pc = table("k_pool_card"), *pos = table("k_position");
    l.own_deck = deck ? static_cast<int32_t>(deck->id) : -1;
    l.k_seen = seen ? static_cast<int32_t>(seen->id) : -1;
    l.k_known_hand = known ? static_cast<int32_t>(known->id) : -1;
    l.k_pool = pool ? static_cast<int32_t>(pool->id) : -1;
    l.k_pool_card = pc ? static_cast<int32_t>(pc->id) : -1;
    l.k_position = pos ? static_cast<int32_t>(pos->id) : -1;
    l.deck_rel = col(deck, "player_rel");
    l.deck_card = col(deck, "card");
    l.deck_main = col(deck, "main");
    l.deck_side = col(deck, "side");
    l.seen_rel = col(seen, "player_rel");
    l.seen_card = col(seen, "card");
    l.seen_game = col(seen, "game");
    l.seen_match = col(seen, "match");
    l.known_card = col(known, "card");
    l.known_copies = col(known, "copies");
    l.pool_rel = col(pool, "owner_rel");
    l.pool_kind = col(pool, "kind");
    l.pool_anchor = col(pool, "range_anchor");
    l.pool_pos = col(pool, "range_pos");
    l.pool_span = col(pool, "range_span");
    l.pc_pool = col(pc, "pool");
    l.pc_card = col(pc, "card");
    l.pc_min = col(pc, "min");
    l.pc_max = col(pc, "max");
    l.pos_rel = col(pos, "owner_rel");
    l.pos_anchor = col(pos, "at_anchor");
    l.pos_pos = col(pos, "at_pos");
    l.pos_card = col(pos, "card");
    if (l.pos_card >= 0) l.pos_card_storage = pos->columns[static_cast<size_t>(l.pos_card)].storage;
    const std::vector<std::string> &kinds = featurize::enum_values(schema, "pool_kind");
    for (size_t k = 0; k < kinds.size(); ++k) {
        if (kinds[k] == "HAND") l.kind_hand = k;
        if (kinds[k] == "LIBRARY_RANGE") l.kind_library = k;
        if (kinds[k] == "SIDEBOARD") l.kind_sideboard = k;
    }
}

// Some table a LEDGER rule folds is a token table of the schema: the ledger type exists.
static bool has_ledger(const rmstate::Schema &schema) {
    for (const FoldRule &f : FOLDS) {
        if (f.kind != featurize::TypeKind::LEDGER) continue;
        const rmstate::TableDef *def = featurize::table_named(schema, f.table);
        if (def && is_token_table(*def)) return true;
    }
    return false;
}

// The object table's type and the columns a delta reads.
static void plan_object(const rmstate::Schema &schema, featurize::FeatureSpec *spec) {
    const rmstate::TableDef *def = featurize::table_named(schema, "object");
    if (!def) return;
    spec->object_type = spec->type_of_table[static_cast<uint16_t>(def->id)];
    spec->object_handle = featurize::column_index(*def, "handle");
    spec->object_card = featurize::column_index(*def, "card");
    spec->object_transformed = featurize::column_index(*def, "transformed");
    if (spec->object_card >= 0)
        spec->object_card_storage = def->columns[static_cast<size_t>(spec->object_card)].storage;
}

static bool history_dropped(const std::string &table, const std::string &column) {
    const std::string name = table + "." + column;
    for (const char *d : HISTORY_DROPPED)
        if (name == d) return true;
    return false;
}

// A HISTORY type over `def`'s columns but the dropped ones and event_card's `event` (its relation
// to the event's token).
static void plan_history_type(const rmstate::Schema &schema, const rmstate::TableDef &def, featurize::FeatureSpec *spec,
    featurize::Registry *reg, featurize::TokenType *t) {
    t->kind = featurize::TypeKind::HISTORY;
    t->table = static_cast<uint16_t>(def.id);
    t->columns = static_cast<uint16_t>(def.columns.size());
    for (size_t c = 0; c < def.columns.size(); ++c) {
        const rmstate::ColumnDef &col = def.columns[c];
        if (history_dropped(def.name, col.name)) continue;
        if (col.tag.compare(0, 4, "row:") == 0) {
            if (featurize::parse_tag(col.tag).arg != "event") continue;
            featurize::ColumnOp op;
            op.kind = featurize::OpKind::ROW;
            op.col = static_cast<uint16_t>(c);
            op.storage = col.storage;
            op.target = spec->history.event_type;
            op.relation = add_relation(spec, reg, t, col.name, "hist_event");
            if (op.relation >= 0) t->ops.push_back(op);
            continue;
        }
        plan_column(schema, def, c, spec, reg, t);
    }
}

// The event tables and the columns the event history reads; false with `err` when the schema lacks them.
static bool history_tables(const rmstate::Schema &schema, featurize::HistoryPlan *h, std::string *err) {
    const rmstate::TableDef *ev = featurize::table_named(schema, "event");
    const rmstate::TableDef *card = featurize::table_named(schema, "event_card");
    const rmstate::TableDef *turn = featurize::table_named(schema, "turn");
    if (!ev || !card || !turn) {
        *err = "history tokens: the schema has no event, event_card or turn table";
        return false;
    }
    h->event_table = static_cast<uint16_t>(ev->id);
    h->card_table = static_cast<uint16_t>(card->id);
    h->turn_table = static_cast<uint16_t>(turn->id);
    h->event_kind = featurize::column_index(*ev, "kind");
    h->event_turn = featurize::column_index(*ev, "turn");
    h->event_source = featurize::column_index(*ev, "source");
    h->card_event = featurize::column_index(*card, "event");
    h->card_before = featurize::column_index(*card, "handle_before");
    h->card_after = featurize::column_index(*card, "handle_after");
    h->turn_turn = featurize::column_index(*turn, "turn");
    const std::vector<std::string> &kinds = featurize::enum_values(schema, "event_kind");
    for (size_t k = 0; k < kinds.size(); ++k)
        if (kinds[k] == "GAME_START") h->game_start_kind = k;
    if (h->event_kind < 0 || h->event_turn < 0 || h->event_source < 0 || h->card_event < 0 || h->card_before < 0 ||
        h->card_after < 0 || h->turn_turn < 0 || h->game_start_kind == UINT64_MAX) {
        *err =
            "history tokens: the event tables lack kind, turn, source, event or the handles, or event_kind "
            "GAME_START";
        return false;
    }
    return true;
}

// The history types (FeatureOptions::history_events > 0), their ids already in the plan.
static void plan_history(const rmstate::Schema &schema, const featurize::FeatureOptions &options,
    featurize::FeatureSpec *spec, featurize::Registry *reg) {
    featurize::HistoryPlan &h = spec->history;
    h.events = options.history_events;
    h.cards_per_event = options.history_cards_per_event;
    for (const char *d : HISTORY_DROPPED) h.dropped.push_back(d);
    featurize::TokenType &e = spec->types[static_cast<size_t>(h.event_type)];
    plan_history_type(schema, *schema.find_table(h.event_table), spec, reg, &e);
    const int32_t offset = add_numeric(reg, &e, "turn_offset", "event.turn", featurize::Transform::BOOL);
    const int32_t age = add_numeric(reg, &e, "age", "", featurize::Transform::LOG1P);
    const int32_t more = add_numeric(reg, &e, "more_cards", "event_card", featurize::Transform::LOG1P);
    h.offset_slot = static_cast<uint32_t>(std::max(offset, 0));
    h.age_slot = static_cast<uint32_t>(std::max(age, 0));
    h.more_slot = static_cast<uint32_t>(std::max(more, 0));
    plan_history_type(
        schema, *schema.find_table(h.card_table), spec, reg, &spec->types[static_cast<size_t>(h.card_type)]);
}

// A planned type's block and the slot columns it uses.
static void place_type(featurize::Registry *reg, featurize::TokenType *t) {
    const featurize::RegType *r = reg->find_type(t->name);
    if (r) {
        t->num_at = r->num_at;
        t->num_capacity = r->num_capacity;
    }
    for (const featurize::CatSlot &s : t->categorical) t->cat_width = std::max(t->cat_width, s.col + 1);
    for (const featurize::CardSlot &s : t->cards) t->card_width = std::max(t->card_width, s.col + 1);
    for (const featurize::AbilitySlot &s : t->abilities) t->ability_width = std::max(t->ability_width, s.col + 1);
}

static std::string json_history(const featurize::FeatureSpec &spec) {
    const featurize::HistoryPlan &h = spec.history;
    return ",\"history\":{\"events\":" + std::to_string(h.events) +
           ",\"cards_per_event\":" + std::to_string(h.cards_per_event) +
           ",\"turns\":2,\"event_type\":" + featurize::json_quote(spec.types[static_cast<size_t>(h.event_type)].name) +
           ",\"card_type\":" + featurize::json_quote(spec.types[static_cast<size_t>(h.card_type)].name) +
           ",\"dropped\":" + json_strings(h.dropped) + "}";
}

static std::string json_strings(const std::vector<std::string> &items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) out += (i ? "," : "") + featurize::json_quote(items[i]);
    return out + "]";
}

static std::string json_ints(const std::vector<int64_t> &items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) out += (i ? "," : "") + std::to_string(items[i]);
    return out + "]";
}

static std::string json_type(const featurize::TokenType &t, size_t id) {
    std::string out = "{\"id\":" + std::to_string(id) +
                      ",\"table\":" + (t.synthetic() ? std::string("null") : std::to_string(t.table)) +
                      ",\"name\":" + featurize::json_quote(t.name) + ",\"kind\":\"" +
                      featurize::type_kind_name(t.kind) + "\"";
    if (t.kind == featurize::TypeKind::DELTA) out += ",\"printed\":" + featurize::json_quote(t.printed);
    out += std::string(",\"owns_handles\":") + (t.owns_handles ? "true" : "false") +
           ",\"num_at\":" + std::to_string(t.num_at) + ",\"num_capacity\":" + std::to_string(t.num_capacity) +
           ",\"numeric\":[";
    for (size_t i = 0; i < t.numeric.size(); ++i) {
        const featurize::NumSlot &s = t.numeric[i];
        out += std::string(i ? "," : "") + "{\"name\":" + featurize::json_quote(s.name) +
               ",\"column\":" + featurize::json_quote(s.column) + ",\"transform\":\"" +
               featurize::transform_name(s.transform) + "\"";
        if (s.transform == featurize::Transform::MASK_BIT) out += ",\"bit\":" + std::to_string(s.bit);
        out += ",\"at\":" + std::to_string(s.at) + "}";
    }
    out += "],\"categorical\":[";
    for (size_t i = 0; i < t.categorical.size(); ++i) {
        const featurize::CatSlot &s = t.categorical[i];
        out += std::string(i ? "," : "") + "{\"name\":" + featurize::json_quote(s.name) +
               ",\"column\":" + featurize::json_quote(s.column) + ",\"source\":" + featurize::json_quote(s.source) +
               ",\"col\":" + std::to_string(s.col) + ",\"offset\":" + std::to_string(s.offset) +
               ",\"size\":" + std::to_string(s.size) + "}";
    }
    out += "],\"cards\":[";
    for (size_t i = 0; i < t.cards.size(); ++i)
        out += std::string(i ? "," : "") + "{\"name\":" + featurize::json_quote(t.cards[i].name) +
               ",\"column\":" + featurize::json_quote(t.cards[i].column) +
               ",\"col\":" + std::to_string(t.cards[i].col) +
               (t.cards[i].current_face ? ",\"face\":\"current\"}" : "}");
    out += "],\"abilities\":[";
    for (size_t i = 0; i < t.abilities.size(); ++i)
        out += std::string(i ? "," : "") + "{\"name\":" + featurize::json_quote(t.abilities[i].name) +
               ",\"col\":" + std::to_string(t.abilities[i].col) + ",\"columns\":[\"ab_card\",\"ab_index\",\"ab_key\"]}";
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

bool build_spec(const rmstate::Schema &schema, const CardTables &tables, const FeatureOptions &options,
    FeatureSpec *out, std::string *err) {
    Registry registry = builtin_registry();
    return build_spec(schema, tables, options, &registry, out, err);
}

bool build_spec(const rmstate::Schema &schema, const CardTables &tables, const FeatureOptions &options, Registry *reg,
    FeatureSpec *out, std::string *err) {
    *out = FeatureSpec();
    if (!reg->ok()) {
        *err = registry_error(*reg);
        return false;
    }
    out->schema_hash = schema.hash;
    out->vocab_hash = tables.vocab_hash;
    out->hash_buckets = reg->dims.hash_buckets;
    out->type_of_table.assign(65536, -1);
    if (options.history_events > 0 && !history_tables(schema, &out->history, err)) return false;
    // The types' ids: the table types in ascending table id, the ledger, the history types.
    std::vector<std::pair<int32_t, const rmstate::TableDef *>> table_types;
    for (const rmstate::TableDef &def : schema.tables) {
        if (!is_token_table(def)) continue;
        const FoldRule *fold = find_fold(def.name);
        if (fold && fold->kind == TypeKind::LEDGER) continue;
        table_types.push_back({reg->type_id(def.name), &def});
    }
    const int32_t ledger = has_ledger(schema) ? reg->type_id("ledger") : -1;
    if (options.history_events > 0) {
        out->history.event_type = reg->type_id("hist_event");
        out->history.card_type = reg->type_id("hist_card");
    }
    if (!reg->ok()) {
        *err = registry_error(*reg);
        return false;
    }
    int32_t top = std::max({ledger, out->history.event_type, out->history.card_type});
    for (const auto &tt : table_types) top = std::max(top, tt.first);
    out->types.resize(static_cast<size_t>(top + 1));
    for (const RegType &r : reg->types)
        if (r.id < out->types.size()) out->types[r.id].name = r.name;
    for (const auto &tt : table_types) {
        TokenType &t = out->types[static_cast<size_t>(tt.first)];
        const FoldRule *fold = find_fold(tt.second->name);
        t.kind = fold && fold->kind == TypeKind::FILTERED ? TypeKind::FILTERED : TypeKind::TABLE;
        t.table = static_cast<uint16_t>(tt.second->id);
        out->type_of_table[static_cast<uint16_t>(tt.second->id)] = tt.first;
    }
    if (ledger >= 0) {
        out->types[static_cast<size_t>(ledger)].kind = TypeKind::LEDGER;
        out->ledger.type = ledger;
    }
    if (options.history_events > 0) {
        out->types[static_cast<size_t>(out->history.event_type)].kind = TypeKind::HISTORY;
        out->types[static_cast<size_t>(out->history.card_type)].kind = TypeKind::HISTORY;
    }
    // Every type planned in id order (the positions are the registry's whatever the order).
    for (size_t i = 0; i < out->types.size(); ++i) {
        TokenType &t = out->types[i];
        if (t.kind == TypeKind::TABLE || t.kind == TypeKind::FILTERED) {
            const rmstate::TableDef &def = *schema.find_table(t.table);
            plan_type(schema, def, out, reg, &t);
            const FoldRule *fold = find_fold(def.name);
            if (fold && fold->kind == TypeKind::DELTA) plan_delta(schema, def, *fold, reg, &t);
        } else if (t.kind == TypeKind::LEDGER) {
            plan_ledger(schema, out, reg, &t);
        } else if (t.kind == TypeKind::HISTORY && static_cast<int32_t>(i) == out->history.event_type) {
            plan_history(schema, options, out, reg);
        }
    }
    plan_object(schema, out);
    if (!reg->ok()) {
        *err = registry_error(*reg);
        return false;
    }
    out->F = reg->dims.F;
    out->C = reg->dims.C;
    out->K = reg->dims.K;
    out->Q = reg->dims.Q;
    out->G = reg->dims.G;
    out->categorical_size = reg->dims.categorical;
    out->type_capacity = reg->dims.types;
    out->relation_capacity = reg->dims.relations;
    out->layout_version = reg->layout_version;
    for (TokenType &t : out->types) {
        if (t.kind == TypeKind::ABSENT) continue;
        place_type(reg, &t);
        if (t.name == "menu") out->menu_type = static_cast<int32_t>(&t - &out->types[0]);
        if (t.name == "header" && !t.synthetic()) {
            const rmstate::TableDef &def = *schema.find_table(t.table);
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
    for (const FoldRule &f : FOLDS) out->folds.push_back({f.table, f.into, f.rows});
    out->card_rows = tables.card_rows;
    out->text_rows = tables.text_rows;
    out->text_width = tables.text_width;
    out->ability_rows = tables.ability_rows();
    out->part_rows = tables.part_rows();
    out->abilities_per_card = tables.abilities_per_card;
    out->key_base = tables.key_base;
    out->card_columns = tables.card_columns;
    out->ability_columns = tables.ability_columns;
    out->part_columns = tables.part_columns;
    out->ab_keys = tables.ab_keys;
    out->key_rows = tables.key_rows;
    out->layout_hash = layout_hash(*reg, tables.card_rows, out->library_depth);
    out->json = spec_json(*out);
    out->hash = rmstate::fnv1a64(out->json.data(), out->json.size());
    return true;
}

const char *type_kind_name(TypeKind k) {
    switch (k) {
        case TypeKind::ABSENT:
            return "absent";
        case TypeKind::TABLE:
            return "table";
        case TypeKind::FILTERED:
            return "filtered";
        case TypeKind::DELTA:
            return "delta";
        case TypeKind::LEDGER:
            return "ledger";
        case TypeKind::HISTORY:
            return "history";
    }
    return "?";
}

std::string spec_json(const FeatureSpec &spec) {
    std::string out = "{\"format\":\"FEATURE_SPEC\",\"version\":" + std::to_string(SPEC_VERSION) +
                      ",\"layout_version\":" + std::to_string(spec.layout_version) + ",\"layout_hash\":\"" +
                      hex16(spec.layout_hash) + "\",\"schema_hash\":\"" + hex16(spec.schema_hash) +
                      "\",\"vocab_hash\":\"" + hex16(spec.vocab_hash) +
                      "\",\"hash_buckets\":" + std::to_string(spec.hash_buckets) +
                      ",\"library_depth\":" + std::to_string(spec.library_depth);
    out += ",\"F\":" + std::to_string(spec.F) + ",\"C\":" + std::to_string(spec.C) +
           ",\"K\":" + std::to_string(spec.K) + ",\"Q\":" + std::to_string(spec.Q) +
           ",\"G\":" + std::to_string(spec.G) + ",\"categorical_size\":" + std::to_string(spec.categorical_size) +
           ",\"type_capacity\":" + std::to_string(spec.type_capacity) +
           ",\"relation_capacity\":" + std::to_string(spec.relation_capacity) +
           ",\"menu\":" + json_quote(spec.types[static_cast<size_t>(spec.menu_type)].name) + ",\"token_types\":[";
    bool first = true;
    for (size_t i = 0; i < spec.types.size(); ++i) {
        if (spec.types[i].kind == TypeKind::ABSENT) continue;
        out += (first ? "" : ",") + json_type(spec.types[i], i);
        first = false;
    }
    out += "],\"relation_types\":[";
    first = true;
    for (size_t i = 0; i < spec.relations.size(); ++i) {
        if (spec.relations[i].table.empty()) continue;
        out += std::string(first ? "" : ",") + "{\"id\":" + std::to_string(i) +
               ",\"table\":" + json_quote(spec.relations[i].table) +
               ",\"column\":" + json_quote(spec.relations[i].column) +
               ",\"target\":" + json_quote(spec.relations[i].target) + "}";
        first = false;
    }
    out += "],\"enums\":{";
    for (size_t i = 0; i < spec.enum_names.size(); ++i)
        out += (i ? "," : "") + json_quote(spec.enum_names[i]) + ":" + json_strings(spec.enum_values[i]);
    out += "},\"overrides\":[";
    for (size_t i = 0; i < spec.overrides.size(); ++i)
        out += std::string(i ? "," : "") + "{\"table\":" + json_quote(spec.overrides[i].table) +
               ",\"column\":" + json_quote(spec.overrides[i].column) +
               ",\"treatment\":" + json_quote(spec.overrides[i].treatment) + "}";
    out += "],\"folds\":[";
    for (size_t i = 0; i < spec.folds.size(); ++i)
        out += std::string(i ? "," : "") + "{\"table\":" + json_quote(spec.folds[i].table) +
               ",\"into\":" + json_quote(spec.folds[i].into) + ",\"rows\":" + json_quote(spec.folds[i].rows) + "}";
    std::string ability = json_table(spec.ability_rows, spec.ability_columns.size(), &spec.ability_columns);
    ability.pop_back();
    ability += ",\"abilities_per_card\":" + std::to_string(spec.abilities_per_card) +
               ",\"key_base\":" + std::to_string(spec.key_base) + "}";
    out += "],\"card_tables\":{\"card\":" + json_table(spec.card_rows, spec.card_columns.size(), &spec.card_columns) +
           ",\"text\":" + json_table(spec.text_rows, spec.text_width, nullptr) + ",\"ability\":" + ability +
           ",\"part\":" + json_table(spec.part_rows, spec.part_columns.size(), &spec.part_columns) +
           "},\"ab_keys\":" + json_strings(spec.ab_keys) + ",\"ab_key_rows\":" + json_ints(spec.key_rows);
    if (spec.history.on()) out += json_history(spec);
    return out + "}";
}

}  // namespace featurize
