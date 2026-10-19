#include "rmtb.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

#include "json.h"

namespace {

const char MAGIC[4] = {'R', 'M', 'T', 'B'};
const uint32_t VERSION = 1;
const size_t HEADER_SIZE = 12;
const featurize::Dtype ALL_DTYPES[] = {featurize::Dtype::F32, featurize::Dtype::F16, featurize::Dtype::I64,
    featurize::Dtype::I32, featurize::Dtype::U16, featurize::Dtype::U8};

}  // namespace

static void put_le(std::string *out, uint64_t v, size_t width);
static uint64_t get_le(const std::string &s, size_t at, size_t width);
static uint64_t align8(uint64_t n);
static bool is_float(featurize::Dtype d);
static bool dtype_of(const std::string &name, featurize::Dtype *out);
static bool hash_member(const featurize::Json &meta, const char *key, bool *has, uint64_t *out, std::string *err);
static bool count_member(const featurize::Json &meta, const char *key, uint64_t *out, std::string *err);
static bool decode_rows_meta(const featurize::Json &rm, featurize::RmtbFile *out, std::string *err);
static bool decode_array_entry(const featurize::Json &a, featurize::RmtbArray *out, uint64_t *offset, std::string *err);
static bool decode_meta(
    const std::string &json, featurize::RmtbFile *out, std::vector<uint64_t> *offsets, std::string *err);
static bool well_formed(const featurize::RmtbFile &file);

static void put_le(std::string *out, uint64_t v, size_t width) {
    for (size_t i = 0; i < width; ++i) out->push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}

static uint64_t get_le(const std::string &s, size_t at, size_t width) {
    uint64_t v = 0;
    for (size_t i = width; i-- > 0;) v = (v << 8) | static_cast<unsigned char>(s[at + i]);
    return v;
}

static uint64_t align8(uint64_t n) {
    return (n + 7) & ~static_cast<uint64_t>(7);
}

static bool is_float(featurize::Dtype d) {
    return d == featurize::Dtype::F32 || d == featurize::Dtype::F16;
}

static bool dtype_of(const std::string &name, featurize::Dtype *out) {
    for (const featurize::Dtype d : ALL_DTYPES)
        if (name == featurize::dtype_name(d)) {
            *out = d;
            return true;
        }
    return false;
}

// An optional hex hash member.
static bool hash_member(const featurize::Json &meta, const char *key, bool *has, uint64_t *out, std::string *err) {
    const featurize::Json *v = meta.get(key);
    *has = v != nullptr;
    *out = 0;
    if (!v) return true;
    if (!v->is_string() || !featurize::parse_hex16(v->as_string(), out)) {
        *err = std::string("meta ") + key + " is not 16 hex digits";
        return false;
    }
    return true;
}

// A required non-negative integer member.
static bool count_member(const featurize::Json &meta, const char *key, uint64_t *out, std::string *err) {
    const featurize::Json *v = meta.get(key);
    if (!v || !v->is_number() || !v->is_integer() || v->as_number() < 0) {
        *err = std::string("meta ") + key + " is not a non-negative integer";
        return false;
    }
    *out = static_cast<uint64_t>(v->as_int());
    return true;
}

static bool decode_rows_meta(const featurize::Json &rm, featurize::RmtbFile *out, std::string *err) {
    out->has_rows_meta = true;
    if (!rm.is_array()) {
        *err = "meta rows_meta is not a list";
        return false;
    }
    for (const featurize::Json &r : rm.items()) {
        featurize::RmtbRowMeta row;
        uint64_t id = 0;
        bool has = false;
        if (!r.is_object() || !count_member(r, "id", &id, err) || !hash_member(r, "hash", &has, &row.hash, err) ||
            !has) {
            *err = "meta rows_meta holds an entry that is not {\"id\":<n>,\"hash\":\"<16 hex>\"}";
            return false;
        }
        row.id = static_cast<int64_t>(id);
        out->rows_meta.push_back(row);
    }
    return true;
}

static bool decode_array_entry(
    const featurize::Json &a, featurize::RmtbArray *out, uint64_t *offset, std::string *err) {
    const featurize::Json *name = a.get("name"), *dtype = a.get("dtype"), *shape = a.get("shape"),
                          *columns = a.get("columns");
    if (!a.is_object() || !name || !name->is_string() || name->as_string().empty() || !dtype || !dtype->is_string() ||
        !dtype_of(dtype->as_string(), &out->dtype) || !shape || !shape->is_array() ||
        !count_member(a, "offset", offset, err)) {
        *err = "meta arrays holds an entry that is not {name, dtype (f32 f16 i64 i32 u16 u8), shape, offset}";
        return false;
    }
    out->name = name->as_string();
    for (const featurize::Json &d : shape->items()) {
        if (!d.is_integer() || d.as_int() < 0) {
            *err = "array " + out->name + ": shape holds a non-count";
            return false;
        }
        out->shape.push_back(static_cast<uint64_t>(d.as_int()));
    }
    if (columns) {
        if (!columns->is_array()) {
            *err = "array " + out->name + ": columns is not a list";
            return false;
        }
        for (const featurize::Json &c : columns->items()) {
            if (!c.is_string()) {
                *err = "array " + out->name + ": columns holds a non-string";
                return false;
            }
            out->columns.push_back(c.as_string());
        }
    }
    return true;
}

static bool decode_meta(
    const std::string &json, featurize::RmtbFile *out, std::vector<uint64_t> *offsets, std::string *err) {
    featurize::Json doc;
    if (!featurize::Json::parse(json, &doc, err)) {
        *err = "meta: " + *err;
        return false;
    }
    if (!doc.is_object()) {
        *err = "meta is not an object";
        return false;
    }
    *out = featurize::RmtbFile();
    const featurize::Json *kind = doc.get("kind"), *arrays = doc.get("arrays");
    if (!kind || !kind->is_string() || kind->as_string().empty()) {
        *err = "meta kind is not a string";
        return false;
    }
    out->kind = kind->as_string();
    if (!hash_member(doc, "vocab_hash", &out->has_vocab_hash, &out->vocab_hash, err) ||
        !hash_member(doc, "spec_hash", &out->has_spec_hash, &out->spec_hash, err))
        return false;
    for (const std::pair<const char *, std::string *> &s :
        {std::make_pair("model", &out->model), std::make_pair("pooling", &out->pooling)}) {
        const featurize::Json *v = doc.get(s.first);
        if (!v) continue;
        if (!v->is_string() || v->as_string().empty()) {
            *err = std::string("meta ") + s.first + " is not a non-empty string";
            return false;
        }
        *s.second = v->as_string();
    }
    if (doc.has("norm_version")) {
        uint64_t n = 0;
        if (!count_member(doc, "norm_version", &n, err)) return false;
        out->norm_version = static_cast<int64_t>(n);
    }
    if (const featurize::Json *rm = doc.get("rows_meta"))
        if (!decode_rows_meta(*rm, out, err)) return false;
    if (!arrays || !arrays->is_array()) {
        *err = "meta arrays is not a list";
        return false;
    }
    for (const featurize::Json &a : arrays->items()) {
        out->arrays.emplace_back();
        offsets->push_back(0);
        if (!decode_array_entry(a, &out->arrays.back(), &offsets->back(), err)) return false;
    }
    return true;
}

// Every array's data holds its shape's count of values, columns fit the last dimension, and the
// names are distinct.
static bool well_formed(const featurize::RmtbFile &file) {
    std::set<std::string> names;
    for (const featurize::RmtbArray &a : file.arrays) {
        if (a.name.empty() || !names.insert(a.name).second) return false;
        if (a.data.size() != a.count() * featurize::dtype_width(a.dtype)) return false;
        if (!a.columns.empty() && (a.shape.empty() || a.columns.size() != a.shape.back())) return false;
    }
    return true;
}

namespace featurize {

uint64_t RmtbArray::count() const {
    uint64_t n = 1;
    for (const uint64_t d : shape) n *= d;
    return n;
}

const RmtbArray *RmtbFile::find(const std::string &name) const {
    for (const RmtbArray &a : arrays)
        if (a.name == name) return &a;
    return nullptr;
}

size_t dtype_width(Dtype d) {
    switch (d) {
        case Dtype::F32:
        case Dtype::I32:
            return 4;
        case Dtype::F16:
        case Dtype::U16:
            return 2;
        case Dtype::I64:
            return 8;
        case Dtype::U8:
            return 1;
    }
    return 0;
}

const char *dtype_name(Dtype d) {
    switch (d) {
        case Dtype::F32:
            return "f32";
        case Dtype::F16:
            return "f16";
        case Dtype::I64:
            return "i64";
        case Dtype::I32:
            return "i32";
        case Dtype::U16:
            return "u16";
        case Dtype::U8:
            return "u8";
    }
    return "";
}

uint16_t f32_to_f16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof bits);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu);
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp == 0xFF) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u | (mant >> 13) : 0u));
    const int32_t e = exp - 127 + 15;
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    if (e <= 0) {
        // A subnormal half (or zero): the 24-bit significand shifted into the 10 mantissa bits.
        const int32_t shift = 14 - e;
        if (shift > 24) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1), halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t half = (static_cast<uint32_t>(e) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;  // a carry into the exponent is right
    return static_cast<uint16_t>(sign | half);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        const float mag = std::ldexp(static_cast<float>(mant), -24);
        return sign ? -mag : mag;
    }
    if (exp == 31)
        bits = sign | 0x7F800000u | (mant << 13);
    else
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float out;
    std::memcpy(&out, &bits, sizeof out);
    return out;
}

RmtbArray rmtb_floats(
    const std::string &name, Dtype dtype, const std::vector<uint64_t> &shape, const std::vector<float> &values) {
    RmtbArray a;
    a.name = name;
    a.dtype = dtype;
    a.shape = shape;
    if (!is_float(dtype) || values.size() != a.count()) return a;
    a.data.reserve(values.size() * dtype_width(dtype));
    for (const float v : values) {
        if (dtype == Dtype::F16) {
            put_le(&a.data, f32_to_f16(v), 2);
        } else {
            uint32_t bits;
            std::memcpy(&bits, &v, sizeof bits);
            put_le(&a.data, bits, 4);
        }
    }
    return a;
}

RmtbArray rmtb_ints(
    const std::string &name, Dtype dtype, const std::vector<uint64_t> &shape, const std::vector<int64_t> &values) {
    RmtbArray a;
    a.name = name;
    a.dtype = dtype;
    a.shape = shape;
    if (is_float(dtype) || values.size() != a.count()) return a;
    a.data.reserve(values.size() * dtype_width(dtype));
    for (const int64_t v : values) put_le(&a.data, static_cast<uint64_t>(v), dtype_width(dtype));
    return a;
}

bool rmtb_values(const RmtbArray &a, std::vector<float> *out) {
    out->clear();
    if (!is_float(a.dtype) || a.data.size() != a.count() * dtype_width(a.dtype)) return false;
    const size_t w = dtype_width(a.dtype);
    out->reserve(a.data.size() / w);
    for (size_t at = 0; at < a.data.size(); at += w) {
        if (a.dtype == Dtype::F16) {
            out->push_back(f16_to_f32(static_cast<uint16_t>(get_le(a.data, at, 2))));
        } else {
            const uint32_t bits = static_cast<uint32_t>(get_le(a.data, at, 4));
            float v;
            std::memcpy(&v, &bits, sizeof v);
            out->push_back(v);
        }
    }
    return true;
}

bool rmtb_values(const RmtbArray &a, std::vector<int64_t> *out) {
    out->clear();
    if (is_float(a.dtype) || a.data.size() != a.count() * dtype_width(a.dtype)) return false;
    const size_t w = dtype_width(a.dtype);
    out->reserve(a.data.size() / w);
    for (size_t at = 0; at < a.data.size(); at += w) {
        const uint64_t u = get_le(a.data, at, w);
        if (a.dtype == Dtype::I32)
            out->push_back(static_cast<int32_t>(static_cast<uint32_t>(u)));
        else
            out->push_back(static_cast<int64_t>(u));
    }
    return true;
}

std::string hex16(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

bool parse_hex16(const std::string &s, uint64_t *out) {
    if (s.size() != 16) return false;
    uint64_t v = 0;
    for (const char c : s) {
        v <<= 4;
        if (c >= '0' && c <= '9')
            v |= static_cast<uint64_t>(c - '0');
        else if (c >= 'a' && c <= 'f')
            v |= static_cast<uint64_t>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            v |= static_cast<uint64_t>(c - 'A' + 10);
        else
            return false;
    }
    *out = v;
    return true;
}

std::string rmtb_meta_json(const RmtbFile &file) {
    std::string j = "{\"kind\":" + json_quote(file.kind);
    if (file.has_vocab_hash) j += ",\"vocab_hash\":\"" + hex16(file.vocab_hash) + "\"";
    if (file.has_spec_hash) j += ",\"spec_hash\":\"" + hex16(file.spec_hash) + "\"";
    if (!file.model.empty()) j += ",\"model\":" + json_quote(file.model);
    if (!file.pooling.empty()) j += ",\"pooling\":" + json_quote(file.pooling);
    if (file.norm_version >= 0) j += ",\"norm_version\":" + std::to_string(file.norm_version);
    if (file.has_rows_meta) {
        j += ",\"rows_meta\":[";
        for (size_t i = 0; i < file.rows_meta.size(); ++i)
            j += std::string(i ? "," : "") + "{\"id\":" + std::to_string(file.rows_meta[i].id) + ",\"hash\":\"" +
                 hex16(file.rows_meta[i].hash) + "\"}";
        j += "]";
    }
    j += ",\"arrays\":[";
    uint64_t offset = 0;
    for (size_t i = 0; i < file.arrays.size(); ++i) {
        const RmtbArray &a = file.arrays[i];
        offset = align8(offset);
        j += std::string(i ? "," : "") + "{\"name\":" + json_quote(a.name) + ",\"dtype\":\"" + dtype_name(a.dtype) +
             "\",\"shape\":[";
        for (size_t d = 0; d < a.shape.size(); ++d) j += (d ? "," : "") + std::to_string(a.shape[d]);
        j += "],\"offset\":" + std::to_string(offset);
        if (!a.columns.empty()) {
            j += ",\"columns\":[";
            for (size_t c = 0; c < a.columns.size(); ++c) j += (c ? "," : "") + json_quote(a.columns[c]);
            j += "]";
        }
        j += "}";
        offset += a.data.size();
    }
    return j + "]}";
}

std::string rmtb_encode(const RmtbFile &file) {
    if (!well_formed(file)) return std::string();
    std::string meta = rmtb_meta_json(file);
    while (meta.size() % 8) meta += ' ';
    std::string out(MAGIC, sizeof MAGIC);
    put_le(&out, VERSION, 4);
    put_le(&out, meta.size(), 4);
    out += meta;
    const size_t data_start = out.size();
    for (const RmtbArray &a : file.arrays) {
        out.resize(data_start + align8(out.size() - data_start), '\0');
        out += a.data;
    }
    return out;
}

bool rmtb_decode(const std::string &bytes, RmtbFile *out, std::string *err) {
    *out = RmtbFile();
    if (bytes.size() < HEADER_SIZE || std::memcmp(bytes.data(), MAGIC, sizeof MAGIC) != 0) {
        *err = "not an RMTB file (no RMTB magic)";
        return false;
    }
    const uint64_t version = get_le(bytes, 4, 4);
    if (version != VERSION) {
        *err = "RMTB version " + std::to_string(version) + ", expected " + std::to_string(VERSION);
        return false;
    }
    const uint64_t meta_len = get_le(bytes, 8, 4);
    if (meta_len % 8 || HEADER_SIZE + meta_len > bytes.size()) {
        *err = "meta length " + std::to_string(meta_len) + " is not a multiple of 8 within the file";
        return false;
    }
    std::vector<uint64_t> offsets;
    if (!decode_meta(bytes.substr(HEADER_SIZE, meta_len), out, &offsets, err)) return false;
    const size_t data_start = HEADER_SIZE + meta_len;
    const uint64_t data_size = bytes.size() - data_start;
    uint64_t end = 0;
    std::set<std::string> names;
    for (size_t i = 0; i < out->arrays.size(); ++i) {
        RmtbArray &a = out->arrays[i];
        const uint64_t size = a.count() * dtype_width(a.dtype);
        if (!names.insert(a.name).second) {
            *err = "two arrays are named " + a.name;
            return false;
        }
        if (offsets[i] != align8(end) || offsets[i] + size > data_size) {
            *err = "array " + a.name + ": offset " + std::to_string(offsets[i]) + " (" + std::to_string(size) +
                   " bytes) is not the 8-aligned end of the previous array within the data";
            return false;
        }
        if (!a.columns.empty() && (a.shape.empty() || a.columns.size() != a.shape.back())) {
            *err = "array " + a.name + ": columns does not name the last dimension";
            return false;
        }
        a.data = bytes.substr(data_start + offsets[i], size);
        end = offsets[i] + size;
    }
    if (end != data_size) {
        *err = "the data section is " + std::to_string(data_size) + " bytes, the arrays end at " + std::to_string(end);
        return false;
    }
    return true;
}

bool rmtb_read_file(const std::string &path, RmtbFile *out, std::string *err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        *err = "cannot open " + path;
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    if (!rmtb_decode(ss.str(), out, err)) {
        *err = path + ": " + *err;
        return false;
    }
    return true;
}

bool rmtb_write_file(const std::string &path, const RmtbFile &file, std::string *err) {
    const std::string bytes = rmtb_encode(file);
    if (bytes.empty()) {
        *err = "an array's data does not match its shape and columns, or two arrays share a name";
        return false;
    }
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) {
            *err = "cannot write " + tmp;
            return false;
        }
        f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!f.good()) {
            *err = "write failed: " + tmp;
            return false;
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        *err = "cannot rename " + tmp + " to " + path;
        return false;
    }
    return true;
}

}  // namespace featurize
