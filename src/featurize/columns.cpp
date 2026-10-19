#include "columns.h"

#include <cmath>

static bool has_prefix(const std::string &s, const char *prefix, std::string *rest);
static float log1p_of(uint64_t x);

static bool has_prefix(const std::string &s, const char *prefix, std::string *rest) {
    const std::string p(prefix);
    if (s.compare(0, p.size(), p) != 0) return false;
    *rest = s.substr(p.size());
    return true;
}

// log1p(x), read from a table for the small values most cells hold.
static float log1p_of(uint64_t x) {
    static const std::vector<float> small = [] {
        std::vector<float> t(1024);
        for (size_t i = 0; i < t.size(); ++i) t[i] = std::log1p(static_cast<float>(i));
        return t;
    }();
    return x < small.size() ? small[x] : std::log1p(static_cast<float>(x));
}

namespace featurize {

Tag parse_tag(const std::string &tag) {
    Tag t;
    if (has_prefix(tag, "mask:", &t.arg)) {
        t.kind = TagKind::MASK;
    } else if (has_prefix(tag, "enum:", &t.arg)) {
        t.kind = TagKind::ENUM;
    } else if (has_prefix(tag, "row:", &t.arg)) {
        t.kind = TagKind::ROW;
    } else if (tag == "bool") {
        t.kind = TagKind::BOOL;
    } else if (tag == "count") {
        t.kind = TagKind::COUNT;
    } else if (tag == "signed") {
        t.kind = TagKind::SIGNED;
    } else if (tag == "prob") {
        t.kind = TagKind::PROB;
    } else if (tag == "index") {
        t.kind = TagKind::INDEX;
    } else if (tag == "player-rel") {
        t.kind = TagKind::PLAYER_REL;
    } else if (tag == "card") {
        t.kind = TagKind::CARD;
    } else if (tag == "handle") {
        t.kind = TagKind::HANDLE;
    } else if (tag == "text") {
        t.kind = TagKind::TEXT;
    }
    return t;
}

const char *transform_name(Transform t) {
    switch (t) {
        case Transform::BOOL:
            return "bool";
        case Transform::LOG1P:
            return "log1p";
        case Transform::SIGNED_LOG1P:
            return "signed_log1p";
        case Transform::PROB:
            return "prob";
        case Transform::INDEX:
            return "index_log1p";
        case Transform::NONE_FLAG:
            return "none_flag";
        case Transform::MASK_BIT:
            return "mask_bit";
        case Transform::ONE_HOT:
            return "one_hot";
        case Transform::COUNT:
            return "count";
    }
    return "?";
}

float cell_value(Transform t, uint32_t arg, const rmstate::Value &v, rmstate::Storage storage) {
    switch (t) {
        case Transform::BOOL:
            return v.bits != 0 ? 1.0f : 0.0f;
        case Transform::LOG1P:
            return log1p_of(v.bits);
        case Transform::SIGNED_LOG1P: {
            const int64_t x = rmstate::value_signed(v, storage);
            const float m = log1p_of(static_cast<uint64_t>(x < 0 ? -x : x));
            return x < 0 ? -m : m;
        }
        case Transform::PROB:
            return static_cast<float>(static_cast<double>(v.bits) / 1e9);
        case Transform::INDEX:
            return rmstate::value_is_all_ones(v, storage) ? 0.0f : log1p_of(v.bits);
        case Transform::NONE_FLAG:
            return rmstate::value_is_all_ones(v, storage) ? 1.0f : 0.0f;
        case Transform::MASK_BIT:
            return arg < 64 && ((v.bits >> arg) & 1u) ? 1.0f : 0.0f;
        case Transform::ONE_HOT:
            return v.bits == arg ? 1.0f : 0.0f;
        case Transform::COUNT:
            return 0.0f;
    }
    return 0.0f;
}

bool scalar_transform(TagKind kind, Transform *out) {
    switch (kind) {
        case TagKind::BOOL:
            *out = Transform::BOOL;
            return true;
        case TagKind::COUNT:
            *out = Transform::LOG1P;
            return true;
        case TagKind::SIGNED:
            *out = Transform::SIGNED_LOG1P;
            return true;
        case TagKind::PROB:
            *out = Transform::PROB;
            return true;
        case TagKind::INDEX:
            *out = Transform::INDEX;
            return true;
        default:
            return false;
    }
}

const std::vector<std::string> &enum_values(const rmstate::Schema &schema, const std::string &name) {
    static const std::vector<std::string> none;
    const rmstate::EnumTable *e = schema.find_enum(name);
    return e ? e->values : none;
}

const rmstate::TableDef *table_named(const rmstate::Schema &schema, const std::string &name) {
    for (const rmstate::TableDef &def : schema.tables)
        if (def.name == name) return &def;
    return nullptr;
}

int column_index(const rmstate::TableDef &def, const std::string &name) {
    for (size_t i = 0; i < def.columns.size(); ++i)
        if (def.columns[i].name == name) return static_cast<int>(i);
    return -1;
}

}  // namespace featurize
