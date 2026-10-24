#include "registry.h"

#include <algorithm>
#include <set>

#include "../state_format/schema.h"
#include "json.h"

// The committed registry (src/featurize/registry.json), compiled in: the assembler's .incbin copies
// the file's bytes, then a NUL, into read-only data (the Makefile passes the file's path as
// RM_REGISTRY_JSON and makes this object depend on it).
#ifndef RM_REGISTRY_JSON
#define RM_REGISTRY_JSON "src/featurize/registry.json"
#endif
#ifdef __APPLE__
#define RM_REGISTRY_SYM "_rm_featurize_registry_json"
#define RM_REGISTRY_PUSH ".const\n"
#define RM_REGISTRY_POP ".text\n"
#else
#define RM_REGISTRY_SYM "rm_featurize_registry_json"
#define RM_REGISTRY_PUSH ".pushsection .rodata\n"
#define RM_REGISTRY_POP ".popsection\n"
#endif
__asm__(RM_REGISTRY_PUSH ".globl " RM_REGISTRY_SYM "\n"
                         ".p2align 4\n" RM_REGISTRY_SYM ":\n"
                         ".incbin \"" RM_REGISTRY_JSON "\"\n"
                         ".byte 0\n" RM_REGISTRY_POP);
extern "C" const char rm_featurize_registry_json[];

namespace {

// Enums whose values the rules fix (a face, a seat, a step, a colour, a layer of CR 613, …) or the
// schema lists whole (every subtype, card type and supertype the parser knows): their capacity is
// their size. Every other enum is open (registry.h).
const char *const CLOSED_ENUMS[] = {
    "face", "seat", "step", "color", "layer", "day_night", "lib_anchor", "subtype", "card_type", "supertype"};

const char *const FORMAT_NAME = "FEATURE_REGISTRY";

}  // namespace

static uint32_t round_up(uint32_t x, uint32_t m);
static uint32_t open_capacity(size_t n);
static bool closed_enum(const std::string &name);
static bool is_enum_kind(const std::string &kind);
static bool get_u32(const featurize::Json &o, const char *key, uint32_t *out, std::string *err);
static bool get_str(const featurize::Json &o, const char *key, std::string *out, std::string *err);
static bool read_dims(const featurize::Json &o, featurize::RegistryDims *d, std::string *err);
static bool read_slot(const featurize::Json &o, const char *owner_key, featurize::RegSlot *s, std::string *err);
static bool read_sections(const featurize::Json &doc, featurize::Registry *r, std::string *err);
static std::string validate(const featurize::Registry &r);
static std::string check_ranges(
    std::vector<std::pair<uint32_t, uint32_t>> ranges, uint32_t limit, const std::string &what);
static std::string slot_json(const featurize::RegSlot &s, const char *owner_key);
static std::string strings_json(const std::vector<std::string> &items);
static const featurize::Registry &parse_builtin();

static uint32_t round_up(uint32_t x, uint32_t m) {
    return (x + m - 1) / m * m;
}

// An open enum's capacity: room to double, a power of two, at least 8.
static uint32_t open_capacity(size_t n) {
    uint32_t c = 8;
    while (c < 2 * n) c *= 2;
    return c;
}

static bool closed_enum(const std::string &name) {
    for (const char *c : CLOSED_ENUMS)
        if (name == c) return true;
    return false;
}

static bool is_enum_kind(const std::string &kind) {
    return kind == "mask" || kind == "onehot" || kind == "count";
}

static bool get_u32(const featurize::Json &o, const char *key, uint32_t *out, std::string *err) {
    const featurize::Json *v = o.get(key);
    if (!v || !v->is_integer() || v->as_int() < 0 || v->as_int() > 0xFFFFFFFFll) {
        *err = std::string("registry: ") + key + " is not an unsigned integer";
        return false;
    }
    *out = static_cast<uint32_t>(v->as_int());
    return true;
}

static bool get_str(const featurize::Json &o, const char *key, std::string *out, std::string *err) {
    const featurize::Json *v = o.get(key);
    if (!v || !v->is_string()) {
        *err = std::string("registry: ") + key + " is not a string";
        return false;
    }
    *out = v->as_string();
    return true;
}

static bool read_dims(const featurize::Json &o, featurize::RegistryDims *d, std::string *err) {
    return get_u32(o, "F", &d->F, err) && get_u32(o, "C", &d->C, err) && get_u32(o, "K", &d->K, err) &&
           get_u32(o, "Q", &d->Q, err) && get_u32(o, "G", &d->G, err) && get_u32(o, "types", &d->types, err) &&
           get_u32(o, "categorical", &d->categorical, err) && get_u32(o, "relations", &d->relations, err) &&
           get_u32(o, "ab_keys", &d->ab_keys, err) && get_u32(o, "abilities_per_card", &d->abilities_per_card, err) &&
           get_u32(o, "hash_buckets", &d->hash_buckets, err);
}

static bool read_slot(const featurize::Json &o, const char *owner_key, featurize::RegSlot *s, std::string *err) {
    if (!get_str(o, owner_key, &s->owner, err) || !get_str(o, "name", &s->name, err) ||
        !get_str(o, "kind", &s->kind, err) || !get_u32(o, "at", &s->at, err))
        return false;
    s->width = 1;
    if (o.has("enum") && (!get_str(o, "enum", &s->enum_name, err) || !get_u32(o, "width", &s->width, err)))
        return false;
    return true;
}

static bool read_sections(const featurize::Json &doc, featurize::Registry *r, std::string *err) {
    const auto array = [&doc, err](const char *key) -> const featurize::Json * {
        const featurize::Json *a = doc.get(key);
        if (!a || !a->is_array()) {
            *err = std::string("registry: ") + key + " is not an array";
            return nullptr;
        }
        return a;
    };
    const featurize::Json *enums = array("enums"), *types = array("token_types"), *numeric = array("numeric"),
                          *cats = array("categorical"), *cards = array("cards"), *abilities = array("abilities"),
                          *rels = array("relations"), *tables = array("tables"), *columns = array("columns"),
                          *keys = array("ab_keys");
    if (!enums || !types || !numeric || !cats || !cards || !abilities || !rels || !tables || !columns || !keys)
        return false;
    for (const featurize::Json &e : enums->items()) {
        featurize::RegEnum x;
        const featurize::Json *values = e.get("values");
        if (!get_str(e, "name", &x.name, err) || !get_u32(e, "capacity", &x.capacity, err)) return false;
        if (!values || !values->is_array()) {
            *err = "registry: enum " + x.name + " has no values array";
            return false;
        }
        x.closed = e.get("closed") && e.get("closed")->truthy();
        for (const featurize::Json &v : values->items()) x.values.push_back(v.as_string());
        r->enums.push_back(x);
    }
    for (const featurize::Json &t : types->items()) {
        featurize::RegType x;
        if (!get_u32(t, "id", &x.id, err) || !get_str(t, "name", &x.name, err) ||
            !get_u32(t, "num_at", &x.num_at, err) || !get_u32(t, "num_capacity", &x.num_capacity, err))
            return false;
        r->types.push_back(x);
    }
    for (const featurize::Json &s : numeric->items()) {
        featurize::RegSlot x;
        if (!read_slot(s, "type", &x, err)) return false;
        r->numeric.push_back(x);
    }
    for (const featurize::Json &c : cats->items()) {
        featurize::RegCat x;
        if (!get_str(c, "type", &x.type, err) || !get_str(c, "name", &x.name, err) ||
            !get_str(c, "source", &x.source, err) || !get_u32(c, "col", &x.col, err) ||
            !get_u32(c, "offset", &x.offset, err) || !get_u32(c, "size", &x.size, err))
            return false;
        r->categorical.push_back(x);
    }
    for (int pass = 0; pass < 2; ++pass)
        for (const featurize::Json &c : (pass ? abilities : cards)->items()) {
            featurize::RegCol x;
            if (!get_str(c, "type", &x.type, err) || !get_str(c, "name", &x.name, err) ||
                !get_u32(c, "col", &x.col, err))
                return false;
            (pass ? r->abilities : r->cards).push_back(x);
        }
    for (const featurize::Json &c : rels->items()) {
        featurize::RegRelation x;
        if (!get_u32(c, "id", &x.id, err) || !get_str(c, "table", &x.table, err) ||
            !get_str(c, "column", &x.column, err) || !get_str(c, "target", &x.target, err))
            return false;
        r->relations.push_back(x);
    }
    for (const featurize::Json &t : tables->items()) {
        featurize::RegTable x;
        if (!get_str(t, "name", &x.name, err) || !get_u32(t, "capacity", &x.capacity, err)) return false;
        r->tables.push_back(x);
    }
    for (const featurize::Json &s : columns->items()) {
        featurize::RegSlot x;
        if (!read_slot(s, "table", &x, err)) return false;
        r->columns.push_back(x);
    }
    for (const featurize::Json &k : keys->items()) {
        featurize::RegKey x;
        if (!get_u32(k, "row", &x.row, err) || !get_str(k, "key", &x.key, err)) return false;
        r->ab_keys.push_back(x);
    }
    return true;
}

// Ranges [first, first + size) that overlap or pass `limit`: a message, else "".
static std::string check_ranges(
    std::vector<std::pair<uint32_t, uint32_t>> ranges, uint32_t limit, const std::string &what) {
    std::sort(ranges.begin(), ranges.end());
    uint64_t end = 0;
    for (const auto &r : ranges) {
        if (r.first < end) return what + ": two entries overlap at " + std::to_string(r.first);
        end = static_cast<uint64_t>(r.first) + r.second;
        if (end > limit)
            return what + ": an entry at " + std::to_string(r.first) + " passes its capacity " + std::to_string(limit);
    }
    return "";
}

// A registry's consistency (header comment of registry.h): "" when it holds.
static std::string validate(const featurize::Registry &r) {
    const featurize::RegistryDims &d = r.dims;
    std::set<std::string> names;
    std::map<std::string, const featurize::RegEnum *> enums;
    for (const featurize::RegEnum &e : r.enums) {
        if (!enums.emplace(e.name, &e).second) return "enum " + e.name + " is listed twice";
        if (e.values.size() > e.capacity) return "enum " + e.name + " has more values than its capacity";
        std::set<std::string> vs(e.values.begin(), e.values.end());
        if (vs.size() != e.values.size()) return "enum " + e.name + " lists a value twice";
    }
    std::map<std::string, const featurize::RegType *> types;
    std::vector<std::pair<uint32_t, uint32_t>> blocks, ids;
    for (const featurize::RegType &t : r.types) {
        if (!types.emplace(t.name, &t).second) return "token type " + t.name + " is listed twice";
        if (t.num_capacity > d.F) return "token type " + t.name + ": numeric capacity past F";
        blocks.push_back({t.num_at, t.num_capacity});
        ids.push_back({t.id, 1});
    }
    std::string bad = check_ranges(ids, d.types, "token type ids");
    if (bad.empty()) bad = check_ranges(blocks, d.G, "token types' numeric blocks");
    if (!bad.empty()) return bad;
    // The slots of one owner (a type or a table) of numeric / columns.
    const auto slots = [&](const std::vector<featurize::RegSlot> &list, const char *what,
                           const std::map<std::string, uint32_t> &capacity) -> std::string {
        std::map<std::string, std::vector<std::pair<uint32_t, uint32_t>>> per;
        std::set<std::string> seen;
        for (const featurize::RegSlot &s : list) {
            if (!capacity.count(s.owner)) return std::string(what) + " " + s.name + ": no owner " + s.owner;
            if (!seen.insert(s.owner + "\x1f" + s.name).second)
                return std::string(what) + " " + s.owner + "." + s.name + " is listed twice";
            if (s.kind == "scalar") {
                if (!s.enum_name.empty() || s.width != 1)
                    return std::string(what) + " " + s.owner + "." + s.name + ": a scalar of width 1";
            } else if (!is_enum_kind(s.kind) || !enums.count(s.enum_name) || enums[s.enum_name]->capacity != s.width) {
                return std::string(what) + " " + s.owner + "." + s.name + ": an enum block of its enum's capacity";
            }
            per[s.owner].push_back({s.at, s.width});
        }
        for (auto &p : per) {
            const std::string b = check_ranges(p.second, capacity.at(p.first), std::string(what) + " of " + p.first);
            if (!b.empty()) return b;
        }
        return "";
    };
    std::map<std::string, uint32_t> type_caps, table_caps;
    for (const featurize::RegType &t : r.types) type_caps[t.name] = t.num_capacity;
    for (const featurize::RegTable &t : r.tables)
        if (!table_caps.emplace(t.name, t.capacity).second) return "table " + t.name + " is listed twice";
    bad = slots(r.numeric, "numeric slot", type_caps);
    if (bad.empty()) bad = slots(r.columns, "column", table_caps);
    if (!bad.empty()) return bad;
    std::vector<std::pair<uint32_t, uint32_t>> cat_ranges;
    std::set<std::string> cat_cols, cat_names;
    for (const featurize::RegCat &c : r.categorical) {
        if (!types.count(c.type)) return "categorical slot " + c.name + ": no token type " + c.type;
        if (!cat_names.insert(c.type + "\x1f" + c.name).second)
            return "categorical slot " + c.type + "." + c.name + " is listed twice";
        if (c.col >= d.C || !cat_cols.insert(c.type + "\x1f" + std::to_string(c.col)).second)
            return "categorical slot " + c.type + "." + c.name + ": its column is taken or past C";
        if (c.offset < 1 || c.size < 1) return "categorical slot " + c.type + "." + c.name + ": ids from 1";
        if (c.source.compare(0, 5, "enum:") == 0) {
            auto e = enums.find(c.source.substr(5));
            if (e == enums.end() || e->second->capacity + 1 != c.size)
                return "categorical slot " + c.type + "." + c.name + ": not its enum's capacity + 1 ids";
        }
        cat_ranges.push_back({c.offset, c.size});
    }
    bad = check_ranges(cat_ranges, d.categorical, "categorical ids");
    if (!bad.empty()) return bad;
    for (int pass = 0; pass < 2; ++pass) {
        std::set<std::string> cols, ns;
        for (const featurize::RegCol &c : pass ? r.abilities : r.cards) {
            const char *what = pass ? "ability slot " : "card slot ";
            if (!types.count(c.type)) return what + c.name + ": no token type " + c.type;
            if (c.col >= (pass ? d.Q : d.K) || !cols.insert(c.type + "\x1f" + std::to_string(c.col)).second ||
                !ns.insert(c.type + "\x1f" + c.name).second)
                return what + c.type + "." + c.name + ": listed twice, or its column is taken or past the dims";
        }
    }
    std::vector<std::pair<uint32_t, uint32_t>> rel_ids, key_rows;
    std::set<std::string> rel_names, key_names;
    for (const featurize::RegRelation &x : r.relations) {
        rel_ids.push_back({x.id, 1});
        if (!rel_names.insert(x.table + "\x1f" + x.column + "\x1f" + x.target).second)
            return "relation " + x.table + "." + x.column + " is listed twice";
    }
    for (const featurize::RegKey &k : r.ab_keys) {
        key_rows.push_back({k.row, 1});
        if (!key_names.insert(k.key).second) return "ability key " + k.key + " is listed twice";
    }
    bad = check_ranges(rel_ids, d.relations, "relation ids");
    if (bad.empty()) bad = check_ranges(key_rows, d.ab_keys, "ability key rows");
    return bad;
}

static std::string slot_json(const featurize::RegSlot &s, const char *owner_key) {
    std::string out = std::string("{\"") + owner_key + "\":" + featurize::json_quote(s.owner) +
                      ",\"name\":" + featurize::json_quote(s.name) + ",\"kind\":" + featurize::json_quote(s.kind);
    if (!s.enum_name.empty())
        out += ",\"enum\":" + featurize::json_quote(s.enum_name) + ",\"width\":" + std::to_string(s.width);
    return out + ",\"at\":" + std::to_string(s.at) + "}";
}

static std::string strings_json(const std::vector<std::string> &items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) out += (i ? "," : "") + featurize::json_quote(items[i]);
    return out + "]";
}

static const featurize::Registry &parse_builtin() {
    static const featurize::Registry r = [] {
        featurize::Registry out;
        std::string err;
        if (!featurize::Registry::parse(featurize::builtin_registry_text(), &out, &err)) {
            out = featurize::Registry();
            out.problems.push_back("the compiled-in registry (src/featurize/registry.json) does not parse: " + err);
        }
        return out;
    }();
    return r;
}

namespace featurize {

Registry Registry::fresh() {
    Registry r;
    r.fresh_ = true;
    r.dims.C = 12;
    r.dims.K = 6;
    r.dims.Q = 2;
    r.dims.types = 64;
    r.dims.relations = 64;
    r.dims.ab_keys = 256;
    r.dims.abilities_per_card = 16;
    r.dims.hash_buckets = 256;
    return r;
}

bool Registry::parse(const std::string &text, Registry *out, std::string *err) {
    *out = Registry();
    Json doc;
    if (!Json::parse(text, &doc, err)) {
        *err = "registry: " + *err;
        return false;
    }
    std::string format;
    uint32_t version = 0, layout = 0;
    const Json *dims = doc.get("dims");
    if (!doc.is_object() || !get_str(doc, "format", &format, err) || !get_u32(doc, "version", &version, err) ||
        !get_u32(doc, "layout_version", &layout, err))
        return false;
    if (format != FORMAT_NAME || version != static_cast<uint32_t>(REGISTRY_FORMAT)) {
        *err = "registry: not a " + std::string(FORMAT_NAME) + " of version " + std::to_string(REGISTRY_FORMAT);
        return false;
    }
    out->layout_version = static_cast<int>(layout);
    if (!dims || !read_dims(*dims, &out->dims, err) || !read_sections(doc, out, err)) return false;
    const std::string bad = validate(*out);
    if (!bad.empty()) {
        *err = "registry: " + bad;
        return false;
    }
    out->index();
    return true;
}

std::string Registry::json() const {
    const RegistryDims &d = dims;
    std::string out =
        std::string("{\"format\":\"") + FORMAT_NAME + "\",\"version\":" + std::to_string(REGISTRY_FORMAT) +
        ",\"layout_version\":" + std::to_string(layout_version) + ",\n\"dims\":{\"F\":" + std::to_string(d.F) +
        ",\"C\":" + std::to_string(d.C) + ",\"K\":" + std::to_string(d.K) + ",\"Q\":" + std::to_string(d.Q) +
        ",\"G\":" + std::to_string(d.G) + ",\"types\":" + std::to_string(d.types) +
        ",\"categorical\":" + std::to_string(d.categorical) + ",\"relations\":" + std::to_string(d.relations) +
        ",\"ab_keys\":" + std::to_string(d.ab_keys) +
        ",\"abilities_per_card\":" + std::to_string(d.abilities_per_card) +
        ",\"hash_buckets\":" + std::to_string(d.hash_buckets) + "}";
    const auto section = [&out](const char *name, const std::vector<std::string> &lines) {
        out += std::string(",\n\"") + name + "\":[";
        for (size_t i = 0; i < lines.size(); ++i) out += (i ? ",\n" : "\n") + lines[i];
        out += lines.empty() ? "]" : "\n]";
    };
    std::vector<std::string> lines;
    for (const RegEnum &e : enums)
        lines.push_back("{\"name\":" + json_quote(e.name) + ",\"capacity\":" + std::to_string(e.capacity) +
                        (e.closed ? ",\"closed\":true" : "") + ",\"values\":" + strings_json(e.values) + "}");
    section("enums", lines);
    lines.clear();
    for (const RegType &t : types)
        lines.push_back("{\"id\":" + std::to_string(t.id) + ",\"name\":" + json_quote(t.name) + ",\"num_at\":" +
                        std::to_string(t.num_at) + ",\"num_capacity\":" + std::to_string(t.num_capacity) + "}");
    section("token_types", lines);
    lines.clear();
    for (const RegSlot &s : numeric) lines.push_back(slot_json(s, "type"));
    section("numeric", lines);
    lines.clear();
    for (const RegCat &c : categorical)
        lines.push_back("{\"type\":" + json_quote(c.type) + ",\"name\":" + json_quote(c.name) +
                        ",\"source\":" + json_quote(c.source) + ",\"col\":" + std::to_string(c.col) +
                        ",\"offset\":" + std::to_string(c.offset) + ",\"size\":" + std::to_string(c.size) + "}");
    section("categorical", lines);
    for (int pass = 0; pass < 2; ++pass) {
        lines.clear();
        for (const RegCol &c : pass ? abilities : cards)
            lines.push_back("{\"type\":" + json_quote(c.type) + ",\"name\":" + json_quote(c.name) +
                            ",\"col\":" + std::to_string(c.col) + "}");
        section(pass ? "abilities" : "cards", lines);
    }
    lines.clear();
    for (const RegRelation &x : relations)
        lines.push_back("{\"id\":" + std::to_string(x.id) + ",\"table\":" + json_quote(x.table) +
                        ",\"column\":" + json_quote(x.column) + ",\"target\":" + json_quote(x.target) + "}");
    section("relations", lines);
    lines.clear();
    for (const RegTable &t : tables)
        lines.push_back("{\"name\":" + json_quote(t.name) + ",\"capacity\":" + std::to_string(t.capacity) + "}");
    section("tables", lines);
    lines.clear();
    for (const RegSlot &s : columns) lines.push_back(slot_json(s, "table"));
    section("columns", lines);
    lines.clear();
    for (const RegKey &k : ab_keys)
        lines.push_back("{\"row\":" + std::to_string(k.row) + ",\"key\":" + json_quote(k.key) + "}");
    section("ab_keys", lines);
    return out + "\n}\n";
}

std::string Registry::join(const std::string &a, const std::string &b) {
    return a + "\x1f" + b;
}

std::string Registry::join(const std::string &a, const std::string &b, const std::string &c) {
    return a + "\x1f" + b + "\x1f" + c;
}

void Registry::index() {
    enum_of_.clear();
    value_of_.clear();
    for (size_t i = 0; i < enums.size(); ++i) {
        enum_of_[enums[i].name] = i;
        value_of_.emplace_back();
        for (size_t v = 0; v < enums[i].values.size(); ++v)
            value_of_.back()[enums[i].values[v]] = static_cast<uint32_t>(v);
    }
    type_of_.clear();
    for (size_t i = 0; i < types.size(); ++i) type_of_[types[i].name] = i;
    numeric_of_.clear();
    numeric_end_.clear();
    for (size_t i = 0; i < numeric.size(); ++i) {
        numeric_of_[join(numeric[i].owner, numeric[i].name)] = i;
        uint32_t &end = numeric_end_[numeric[i].owner];
        end = std::max(end, numeric[i].at + numeric[i].width);
    }
    cat_of_.clear();
    cat_cols_.clear();
    for (size_t i = 0; i < categorical.size(); ++i) {
        cat_of_[join(categorical[i].type, categorical[i].name)] = i;
        uint32_t &cols = cat_cols_[categorical[i].type];
        cols = std::max(cols, categorical[i].col + 1);
    }
    for (int pass = 0; pass < 2; ++pass) {
        std::map<std::string, size_t> &of = pass ? ability_of_ : card_of_;
        std::map<std::string, uint32_t> &cols = pass ? ability_cols_ : card_cols_;
        const std::vector<RegCol> &list = pass ? abilities : cards;
        of.clear();
        cols.clear();
        for (size_t i = 0; i < list.size(); ++i) {
            of[join(list[i].type, list[i].name)] = i;
            cols[list[i].type] = std::max(cols[list[i].type], list[i].col + 1);
        }
    }
    relation_of_.clear();
    for (size_t i = 0; i < relations.size(); ++i)
        relation_of_[join(relations[i].table, relations[i].column, relations[i].target)] = i;
    table_of_.clear();
    for (size_t i = 0; i < tables.size(); ++i) table_of_[tables[i].name] = i;
    column_of_.clear();
    column_end_.clear();
    for (size_t i = 0; i < columns.size(); ++i) {
        column_of_[join(columns[i].owner, columns[i].name)] = i;
        uint32_t &end = column_end_[columns[i].owner];
        end = std::max(end, columns[i].at + columns[i].width);
    }
    key_of_.clear();
    for (size_t i = 0; i < ab_keys.size(); ++i) key_of_[ab_keys[i].key] = i;
}

void Registry::problem(const std::string &what) {
    if (std::find(problems.begin(), problems.end(), what) == problems.end()) problems.push_back(what);
}

const RegEnum *Registry::find_enum(const std::string &name) const {
    auto it = enum_of_.find(name);
    return it == enum_of_.end() ? nullptr : &enums[it->second];
}

uint32_t Registry::enum_width(const std::string &enum_name) const {
    const RegEnum *e = find_enum(enum_name);
    return e ? e->capacity : 0;
}

int32_t Registry::enum_index(const std::string &name, const std::string &value, size_t n) {
    auto it = enum_of_.find(name);
    if (it == enum_of_.end()) {
        if (!append) {
            problem("FEATURE_REGISTRY has no enum " + name + " (run `make registry`)");
            return -1;
        }
        RegEnum e;
        e.name = name;
        e.closed = closed_enum(name);
        e.capacity = e.closed ? static_cast<uint32_t>(n) : open_capacity(n);
        enums.push_back(e);
        value_of_.emplace_back();
        it = enum_of_.emplace(name, enums.size() - 1).first;
        added.push_back("enum " + name + " (capacity " + std::to_string(e.capacity) + ")");
    }
    RegEnum &e = enums[it->second];
    std::map<std::string, uint32_t> &values = value_of_[it->second];
    auto v = values.find(value);
    if (v != values.end()) return static_cast<int32_t>(v->second);
    if (!append) {
        problem("FEATURE_REGISTRY: enum " + name + " has no value " + value + " (run `make registry`)");
        return -1;
    }
    if (e.values.size() >= e.capacity) {
        problem("FEATURE_REGISTRY: enum " + name + " is full (capacity " + std::to_string(e.capacity) + "): " + value +
                " needs a layout bump");
        return -1;
    }
    e.values.push_back(value);
    values[value] = static_cast<uint32_t>(e.values.size() - 1);
    added.push_back("enum " + name + " value " + value);
    return static_cast<int32_t>(e.values.size() - 1);
}

std::vector<int32_t> Registry::enum_map(const std::string &name, const std::vector<std::string> &values) {
    std::vector<int32_t> out;
    for (const std::string &v : values) out.push_back(enum_index(name, v, values.size()));
    return out;
}

const RegType *Registry::find_type(const std::string &name) const {
    auto it = type_of_.find(name);
    return it == type_of_.end() ? nullptr : &types[it->second];
}

int32_t Registry::type_id(const std::string &name) {
    auto it = type_of_.find(name);
    if (it != type_of_.end()) return static_cast<int32_t>(types[it->second].id);
    if (!append) {
        problem("FEATURE_REGISTRY has no token type " + name + " (run `make registry`)");
        return -1;
    }
    uint32_t id = 0;
    for (const RegType &t : types) id = std::max(id, t.id + 1);
    if (id >= dims.types) {
        problem("FEATURE_REGISTRY: the token types are full (capacity " + std::to_string(dims.types) + "): " + name +
                " needs a layout bump");
        return -1;
    }
    RegType t;
    t.name = name;
    t.id = id;
    types.push_back(t);
    type_of_[name] = types.size() - 1;
    new_types_.push_back(name);
    added.push_back("token type " + name);
    return static_cast<int32_t>(id);
}

int32_t Registry::numeric_at(
    const std::string &type, const std::string &name, const std::string &kind, const std::string &enum_name) {
    const std::string key = join(type, name);
    auto it = numeric_of_.find(key);
    const uint32_t width = kind == "scalar" ? 1 : enum_width(enum_name);
    if (it != numeric_of_.end()) {
        const RegSlot &s = numeric[it->second];
        if (s.kind != kind || s.enum_name != enum_name) {
            problem("FEATURE_REGISTRY: numeric slot " + type + "." + name + " is registered as " + s.kind +
                    (s.enum_name.empty() ? "" : " of " + s.enum_name) + ", the schema makes it " + kind +
                    (enum_name.empty() ? "" : " of " + enum_name) + ": a layout bump");
            return -1;
        }
        return static_cast<int32_t>(s.at);
    }
    if (!append || width == 0) {
        problem("FEATURE_REGISTRY has no numeric slot " + type + "." + name + " (run `make registry`)");
        return -1;
    }
    const RegType *t = find_type(type);
    const bool placed = std::find(new_types_.begin(), new_types_.end(), type) == new_types_.end();
    uint32_t &end = numeric_end_[type];
    if (!t || (placed && end + width > t->num_capacity)) {
        problem("FEATURE_REGISTRY: token type " + type + "'s numeric slots are full (capacity " +
                std::to_string(t ? t->num_capacity : 0) + "): " + name + " needs a layout bump");
        return -1;
    }
    RegSlot s;
    s.owner = type;
    s.name = name;
    s.kind = kind;
    s.enum_name = enum_name;
    s.width = width;
    s.at = end;
    end += width;
    numeric.push_back(s);
    numeric_of_[key] = numeric.size() - 1;
    added.push_back("numeric slot " + type + "." + name);
    return static_cast<int32_t>(s.at);
}

bool Registry::categorical_slot(
    const std::string &type, const std::string &name, const std::string &source, uint32_t size, RegCat *out) {
    const std::string key = join(type, name);
    auto it = cat_of_.find(key);
    if (it != cat_of_.end()) {
        const RegCat &c = categorical[it->second];
        if (c.source != source || c.size < size) {
            problem("FEATURE_REGISTRY: categorical slot " + type + "." + name + " is registered as " + c.source +
                    " of " + std::to_string(c.size) + " ids, the schema needs " + source + " of " +
                    std::to_string(size) + ": a layout bump");
            return false;
        }
        *out = c;
        return true;
    }
    if (!append) {
        problem("FEATURE_REGISTRY has no categorical slot " + type + "." + name + " (run `make registry`)");
        return false;
    }
    uint32_t &cols = cat_cols_[type];
    uint32_t end = 1;
    for (const RegCat &c : categorical) end = std::max(end, c.offset + c.size);
    if (cols >= dims.C || (!fresh_ && end + size > dims.categorical)) {
        problem("FEATURE_REGISTRY: no room for categorical slot " + type + "." + name + " (C " +
                std::to_string(dims.C) + ", ids " + std::to_string(dims.categorical) + "): a layout bump");
        return false;
    }
    RegCat c;
    c.type = type;
    c.name = name;
    c.source = source;
    c.col = cols++;
    c.offset = end;
    c.size = size;
    categorical.push_back(c);
    cat_of_[key] = categorical.size() - 1;
    added.push_back("categorical slot " + type + "." + name);
    *out = c;
    return true;
}

int32_t Registry::card_col(const std::string &type, const std::string &name) {
    const std::string key = join(type, name);
    auto it = card_of_.find(key);
    if (it != card_of_.end()) return static_cast<int32_t>(cards[it->second].col);
    if (!append) {
        problem("FEATURE_REGISTRY has no card slot " + type + "." + name + " (run `make registry`)");
        return -1;
    }
    uint32_t &cols = card_cols_[type];
    if (cols >= dims.K) {
        problem("FEATURE_REGISTRY: no room for card slot " + type + "." + name + " (K " + std::to_string(dims.K) +
                "): a layout bump");
        return -1;
    }
    cards.push_back({type, name, cols++});
    card_of_[key] = cards.size() - 1;
    added.push_back("card slot " + type + "." + name);
    return static_cast<int32_t>(cards.back().col);
}

int32_t Registry::ability_col(const std::string &type, const std::string &name) {
    const std::string key = join(type, name);
    auto it = ability_of_.find(key);
    if (it != ability_of_.end()) return static_cast<int32_t>(abilities[it->second].col);
    if (!append) {
        problem("FEATURE_REGISTRY has no ability slot " + type + "." + name + " (run `make registry`)");
        return -1;
    }
    uint32_t &cols = ability_cols_[type];
    if (cols >= dims.Q) {
        problem("FEATURE_REGISTRY: no room for ability slot " + type + "." + name + " (Q " + std::to_string(dims.Q) +
                "): a layout bump");
        return -1;
    }
    abilities.push_back({type, name, cols++});
    ability_of_[key] = abilities.size() - 1;
    added.push_back("ability slot " + type + "." + name);
    return static_cast<int32_t>(abilities.back().col);
}

int32_t Registry::relation_id(const std::string &table, const std::string &column, const std::string &target) {
    const std::string key = join(table, column, target);
    auto it = relation_of_.find(key);
    if (it != relation_of_.end()) return static_cast<int32_t>(relations[it->second].id);
    if (!append) {
        problem("FEATURE_REGISTRY has no relation type " + table + "." + column + " → " + target +
                " (run `make registry`)");
        return -1;
    }
    uint32_t id = 0;
    for (const RegRelation &x : relations) id = std::max(id, x.id + 1);
    if (id >= dims.relations) {
        problem("FEATURE_REGISTRY: the relation types are full (capacity " + std::to_string(dims.relations) +
                "): " + table + "." + column + " needs a layout bump");
        return -1;
    }
    relations.push_back({id, table, column, target});
    relation_of_[key] = relations.size() - 1;
    added.push_back("relation type " + table + "." + column + " → " + target);
    return static_cast<int32_t>(id);
}

int32_t Registry::column_at(
    const std::string &table, const std::string &name, const std::string &kind, const std::string &enum_name) {
    const std::string key = join(table, name);
    auto it = column_of_.find(key);
    const uint32_t width = kind == "scalar" ? 1 : enum_width(enum_name);
    if (it != column_of_.end()) {
        const RegSlot &s = columns[it->second];
        if (s.kind != kind || s.enum_name != enum_name) {
            problem("FEATURE_REGISTRY: column " + table + "." + name + " is registered as " + s.kind +
                    (s.enum_name.empty() ? "" : " of " + s.enum_name) + ", the schema makes it " + kind +
                    (enum_name.empty() ? "" : " of " + enum_name) + ": a layout bump");
            return -1;
        }
        return static_cast<int32_t>(s.at);
    }
    if (!append || width == 0) {
        problem("FEATURE_REGISTRY has no column " + table + "." + name + " (run `make registry`)");
        return -1;
    }
    auto t = table_of_.find(table);
    if (t == table_of_.end()) {
        tables.push_back({table, 0});
        t = table_of_.emplace(table, tables.size() - 1).first;
        new_tables_.push_back(table);
        added.push_back("table " + table);
    }
    const bool placed = std::find(new_tables_.begin(), new_tables_.end(), table) == new_tables_.end();
    uint32_t &end = column_end_[table];
    if (placed && end + width > tables[t->second].capacity) {
        problem("FEATURE_REGISTRY: table " + table + "'s columns are full (capacity " +
                std::to_string(tables[t->second].capacity) + "): " + name + " needs a layout bump");
        return -1;
    }
    RegSlot s;
    s.owner = table;
    s.name = name;
    s.kind = kind;
    s.enum_name = enum_name;
    s.width = width;
    s.at = end;
    end += width;
    columns.push_back(s);
    column_of_[key] = columns.size() - 1;
    added.push_back("column " + table + "." + name);
    return static_cast<int32_t>(s.at);
}

uint32_t Registry::table_capacity(const std::string &table) const {
    auto it = table_of_.find(table);
    return it == table_of_.end() ? 0 : tables[it->second].capacity;
}

std::vector<std::string> Registry::column_names(const std::string &table) const {
    const uint32_t cap = table_capacity(table);
    std::vector<std::string> out(cap);
    for (uint32_t i = 0; i < cap; ++i) out[i] = "#" + std::to_string(i);
    for (const RegSlot &s : columns) {
        if (s.owner != table) continue;
        if (s.kind == "scalar") {
            if (s.at < cap) out[s.at] = s.name;
            continue;
        }
        const RegEnum *e = find_enum(s.enum_name);
        const std::string sep = s.kind == "mask" ? ":" : "=";
        for (uint32_t i = 0; i < s.width && s.at + i < cap; ++i)
            out[s.at + i] = e && i < e->values.size() ? s.name + sep + e->values[i] : s.name + "#" + std::to_string(i);
    }
    return out;
}

int32_t Registry::key_row(const std::string &key) {
    auto it = key_of_.find(key);
    if (it != key_of_.end()) return static_cast<int32_t>(ab_keys[it->second].row);
    if (!append) {
        problem("FEATURE_REGISTRY has no ability key " + key + " (run `make registry`)");
        return -1;
    }
    uint32_t row = 0;
    for (const RegKey &k : ab_keys) row = std::max(row, k.row + 1);
    if (row >= dims.ab_keys) {
        problem("FEATURE_REGISTRY: the ability key band is full (capacity " + std::to_string(dims.ab_keys) +
                "): " + key + " needs a layout bump");
        return -1;
    }
    ab_keys.push_back({row, key});
    key_of_[key] = ab_keys.size() - 1;
    added.push_back("ability key " + key);
    return static_cast<int32_t>(row);
}

void Registry::finish() {
    uint32_t end = 0, max_capacity = 0;
    for (const RegType &t : types)
        if (std::find(new_types_.begin(), new_types_.end(), t.name) == new_types_.end())
            end = std::max(end, t.num_at + t.num_capacity);
    for (const std::string &name : new_types_) {
        RegType &t = types[type_of_[name]];
        const uint32_t used = numeric_end_[name];
        uint32_t cap = round_up(used + std::max<uint32_t>(8, (used + 3) / 4), 8);
        if (!fresh_ && cap > dims.F) cap = dims.F;
        if (!fresh_ && used > cap)
            problem("FEATURE_REGISTRY: token type " + name + " needs " + std::to_string(used) +
                    " numeric slots, past F " + std::to_string(dims.F) + ": a layout bump");
        t.num_at = end;
        t.num_capacity = cap;
        end += cap;
        max_capacity = std::max(max_capacity, cap);
    }
    for (const std::string &name : new_tables_) {
        RegTable &t = tables[table_of_[name]];
        const uint32_t used = column_end_[name];
        t.capacity = round_up(used + std::max<uint32_t>(32, used / 4), 64);
    }
    uint32_t cat_end = 1;
    for (const RegCat &c : categorical) cat_end = std::max(cat_end, c.offset + c.size);
    if (fresh_) {
        dims.F = round_up(max_capacity, 8);
        dims.G = round_up(end + end / 2, 128);
        dims.categorical = round_up(cat_end + cat_end / 2, 512);
    }
    if (end > dims.G)
        problem(
            "FEATURE_REGISTRY: the token types' numeric blocks pass G " + std::to_string(dims.G) + ": a layout bump");
    if (cat_end > dims.categorical)
        problem("FEATURE_REGISTRY: the categorical ids pass their capacity " + std::to_string(dims.categorical) +
                ": a layout bump");
    new_types_.clear();
    new_tables_.clear();
    fresh_ = false;
}

const Registry &builtin_registry() {
    return parse_builtin();
}

std::string registry_error(const Registry &registry) {
    std::string out;
    const size_t shown = std::min<size_t>(registry.problems.size(), 8);
    for (size_t i = 0; i < shown; ++i) out += (i ? "; " : "") + registry.problems[i];
    if (registry.problems.size() > shown) out += "; and " + std::to_string(registry.problems.size() - shown) + " more";
    return out;
}

const std::string &builtin_registry_text() {
    static const std::string text(rm_featurize_registry_json);
    return text;
}

uint64_t layout_hash(const Registry &r, uint64_t card_rows, uint32_t library_depth) {
    const RegistryDims &d = r.dims;
    std::string s = "layout_version=" + std::to_string(r.layout_version) + ";F=" + std::to_string(d.F) +
                    ";C=" + std::to_string(d.C) + ";K=" + std::to_string(d.K) + ";Q=" + std::to_string(d.Q) +
                    ";G=" + std::to_string(d.G) + ";types=" + std::to_string(d.types) +
                    ";categorical=" + std::to_string(d.categorical) + ";relations=" + std::to_string(d.relations) +
                    ";ab_keys=" + std::to_string(d.ab_keys) +
                    ";abilities_per_card=" + std::to_string(d.abilities_per_card) +
                    ";hash_buckets=" + std::to_string(d.hash_buckets) + ";card_rows=" + std::to_string(card_rows) +
                    ";library_depth=" + std::to_string(library_depth);
    for (const RegTable &t : r.tables) s += ";table:" + t.name + "=" + std::to_string(t.capacity);
    return rmstate::fnv1a64(s.data(), s.size());
}

std::vector<std::string> registry_moves(const Registry &base, const Registry &now) {
    std::vector<std::string> out;
    const RegistryDims &a = base.dims, &b = now.dims;
    if (base.layout_version != now.layout_version) out.push_back("layout_version changed");
    if (a.F != b.F || a.C != b.C || a.K != b.K || a.Q != b.Q || a.G != b.G || a.types != b.types ||
        a.categorical != b.categorical || a.relations != b.relations || a.ab_keys != b.ab_keys ||
        a.abilities_per_card != b.abilities_per_card || a.hash_buckets != b.hash_buckets)
        out.push_back("the dims changed");
    for (const RegEnum &e : base.enums) {
        const RegEnum *n = now.find_enum(e.name);
        if (!n || n->capacity != e.capacity || n->closed != e.closed || n->values.size() < e.values.size() ||
            !std::equal(e.values.begin(), e.values.end(), n->values.begin()))
            out.push_back("enum " + e.name + ": removed, its capacity changed or a value moved");
    }
    for (const RegType &t : base.types) {
        const RegType *n = now.find_type(t.name);
        if (!n || n->id != t.id || n->num_at != t.num_at || n->num_capacity != t.num_capacity)
            out.push_back("token type " + t.name + " removed or moved");
    }
    const auto same_slots = [&out](const std::vector<RegSlot> &x, const std::vector<RegSlot> &y, const char *what) {
        std::map<std::string, const RegSlot *> by;
        for (const RegSlot &s : y) by[s.owner + "\x1f" + s.name] = &s;
        for (const RegSlot &s : x) {
            auto it = by.find(s.owner + "\x1f" + s.name);
            if (it == by.end() || it->second->at != s.at || it->second->width != s.width ||
                it->second->kind != s.kind || it->second->enum_name != s.enum_name)
                out.push_back(std::string(what) + " " + s.owner + "." + s.name + " removed or moved");
        }
    };
    same_slots(base.numeric, now.numeric, "numeric slot");
    same_slots(base.columns, now.columns, "column");
    std::map<std::string, const RegCat *> cats;
    for (const RegCat &c : now.categorical) cats[c.type + "\x1f" + c.name] = &c;
    for (const RegCat &c : base.categorical) {
        auto it = cats.find(c.type + "\x1f" + c.name);
        if (it == cats.end() || it->second->col != c.col || it->second->offset != c.offset ||
            it->second->size != c.size || it->second->source != c.source)
            out.push_back("categorical slot " + c.type + "." + c.name + " removed or moved");
    }
    for (int pass = 0; pass < 2; ++pass) {
        std::map<std::string, uint32_t> cols;
        for (const RegCol &c : pass ? now.abilities : now.cards) cols[c.type + "\x1f" + c.name] = c.col;
        for (const RegCol &c : pass ? base.abilities : base.cards) {
            auto it = cols.find(c.type + "\x1f" + c.name);
            if (it == cols.end() || it->second != c.col)
                out.push_back(
                    std::string(pass ? "ability" : "card") + " slot " + c.type + "." + c.name + " removed or moved");
        }
    }
    std::map<std::string, uint32_t> rels, keys, tabs;
    for (const RegRelation &x : now.relations) rels[x.table + "\x1f" + x.column + "\x1f" + x.target] = x.id;
    for (const RegRelation &x : base.relations) {
        auto it = rels.find(x.table + "\x1f" + x.column + "\x1f" + x.target);
        if (it == rels.end() || it->second != x.id)
            out.push_back("relation type " + x.table + "." + x.column + " removed or moved");
    }
    for (const RegTable &t : now.tables) tabs[t.name] = t.capacity;
    for (const RegTable &t : base.tables)
        if (!tabs.count(t.name) || tabs[t.name] != t.capacity)
            out.push_back("table " + t.name + " removed or its capacity changed");
    for (const RegKey &k : now.ab_keys) keys[k.key] = k.row;
    for (const RegKey &k : base.ab_keys)
        if (!keys.count(k.key) || keys[k.key] != k.row) out.push_back("ability key " + k.key + " removed or moved");
    return out;
}

}  // namespace featurize
