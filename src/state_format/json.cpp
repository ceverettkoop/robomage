#include "json.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace rmstate {

static void put_string(std::string &out, const std::string &s);
static void put_uint(std::string &out, uint64_t v);
static void put_int(std::string &out, int64_t v);
static bool has_prefix(const std::string &s, const char *prefix);
static const std::string *enum_name(const Schema &schema, const std::string &enum_name, uint64_t value);
static void put_value(std::string &out, const Schema &schema, const ColumnDef &col, const Value &v);

static void put_string(std::string &out, const std::string &s) {
    out.push_back('"');
    for (char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out.push_back(ch);
                }
        }
    }
    out.push_back('"');
}

static void put_uint(std::string &out, uint64_t v) {
    out += std::to_string(v);
}

static void put_int(std::string &out, int64_t v) {
    out += std::to_string(v);
}

static bool has_prefix(const std::string &s, const char *prefix) {
    return s.compare(0, std::strlen(prefix), prefix) == 0;
}

// The name of `value` in the string table `enum_name`, or nullptr.
static const std::string *enum_name(const Schema &schema, const std::string &enum_name, uint64_t value) {
    const EnumTable *e = schema.find_enum(enum_name);
    if (!e || value >= e->values.size()) return nullptr;
    return &e->values[static_cast<size_t>(value)];
}

static void put_value(std::string &out, const Schema &schema, const ColumnDef &col, const Value &v) {
    if (col.storage == Storage::TEXT) {
        put_string(out, v.text);
        return;
    }
    if (has_prefix(col.tag, "enum:")) {
        if (value_is_all_ones(v, col.storage)) {
            out += "null";
            return;
        }
        const std::string *name = enum_name(schema, col.tag.substr(5), v.bits);
        if (name)
            put_string(out, *name);
        else
            put_uint(out, v.bits);
        return;
    }
    if (has_prefix(col.tag, "mask:")) {
        out.push_back('[');
        bool first = true;
        for (uint32_t bit = 0; bit < 64; ++bit) {
            if (!((v.bits >> bit) & 1u)) continue;
            if (!first) out.push_back(',');
            first = false;
            const std::string *name = enum_name(schema, col.tag.substr(5), bit);
            if (name)
                put_string(out, *name);
            else
                put_uint(out, bit);
        }
        out.push_back(']');
        return;
    }
    if (col.tag == "bool" && v.bits <= 1) {
        out += v.bits ? "true" : "false";
        return;
    }
    if ((col.tag == "card" || col.tag == "index" || has_prefix(col.tag, "row:")) && value_is_all_ones(v, col.storage)) {
        out += "null";
        return;
    }
    switch (col.storage) {
        case Storage::I8:
        case Storage::I16:
        case Storage::I32:
        case Storage::I64:
            put_int(out, value_signed(v, col.storage));
            return;
        case Storage::F32: {
            const uint32_t bits = static_cast<uint32_t>(v.bits);
            float f = 0;
            std::memcpy(&f, &bits, sizeof(f));
            if (!std::isfinite(f)) {
                out += "null";
                return;
            }
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(f));
            out += buf;
            return;
        }
        default:
            put_uint(out, v.bits);
    }
}

std::string frame_json(const DecodedFrame &frame, const Schema &schema) {
    std::string out = "{\"frame\":";
    const std::string *kind = enum_name(schema, "frame_kind", frame.header.kind);
    if (kind)
        put_string(out, *kind);
    else
        put_uint(out, frame.header.kind);
    out += ",\"viewer\":";
    const std::string *viewer = enum_name(schema, "viewer", frame.header.viewer);
    if (viewer)
        put_string(out, *viewer);
    else
        put_uint(out, frame.header.viewer);
    out += ",\"version\":";
    put_uint(out, frame.header.version);
    char hash[24];
    std::snprintf(hash, sizeof(hash), "%016llx", static_cast<unsigned long long>(frame.header.schema_hash));
    out += ",\"schema_hash\":\"";
    out += hash;
    out += "\",\"tables\":{";
    for (size_t t = 0; t < frame.tables.size(); ++t) {
        const DecodedTable &table = frame.tables[t];
        if (t) out.push_back(',');
        put_string(out, table.def->name);
        out += ":[";
        for (size_t r = 0; r < table.rows.size(); ++r) {
            if (r) out.push_back(',');
            out.push_back('{');
            for (size_t c = 0; c < table.def->columns.size(); ++c) {
                const ColumnDef &col = table.def->columns[c];
                if (c) out.push_back(',');
                put_string(out, col.name);
                out.push_back(':');
                put_value(out, schema, col, table.rows[r][c]);
            }
            out.push_back('}');
        }
        out.push_back(']');
    }
    out.push_back('}');
    if (!frame.skipped.empty()) {
        out += ",\"skipped\":[";
        for (size_t i = 0; i < frame.skipped.size(); ++i) {
            if (i) out.push_back(',');
            put_uint(out, frame.skipped[i]);
        }
        out.push_back(']');
    }
    out.push_back('}');
    return out;
}

}  // namespace rmstate
