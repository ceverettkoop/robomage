#include "safetensors.h"

#include <fstream>
#include <sstream>
#include <vector>

#include "../../src/featurize/json.h"

namespace {

// A safetensors dtype: its torch type, its width and what it becomes.
struct DtypeInfo {
        const char *name;
        torch::ScalarType stored;
        size_t width;
        torch::ScalarType loaded;
};

const DtypeInfo DTYPES[] = {
    {"F32", torch::kFloat32, 4, torch::kFloat32},
    {"F16", torch::kFloat16, 2, torch::kFloat32},
    {"BF16", torch::kBFloat16, 2, torch::kFloat32},
    {"F64", torch::kFloat64, 8, torch::kFloat32},
    {"I64", torch::kInt64, 8, torch::kInt64},
    {"I32", torch::kInt32, 4, torch::kInt64},
};

}  // namespace

static const DtypeInfo *dtype_info(const std::string &name);
static bool tensor_of(const std::string &data, const std::string &name, const featurize::Json &spec, torch::Tensor *out,
    std::string *err);

static const DtypeInfo *dtype_info(const std::string &name) {
    for (const DtypeInfo &d : DTYPES)
        if (name == d.name) return &d;
    return nullptr;
}

// The tensor `name` of the header entry `spec` over the data block `data`.
static bool tensor_of(const std::string &data, const std::string &name, const featurize::Json &spec, torch::Tensor *out,
    std::string *err) {
    const featurize::Json *dtype = spec.get("dtype"), *shape = spec.get("shape"), *offsets = spec.get("data_offsets");
    if (!dtype || !dtype->is_string() || !shape || !shape->is_array() || !offsets || !offsets->is_array() ||
        offsets->items().size() != 2) {
        *err = name + ": header entry is not {dtype, shape, data_offsets}";
        return false;
    }
    const DtypeInfo *info = dtype_info(dtype->as_string());
    if (!info) {
        *err = name + ": dtype " + dtype->as_string() + " is not supported";
        return false;
    }
    std::vector<int64_t> sizes;
    uint64_t count = 1;
    for (const featurize::Json &d : shape->items()) {
        if (!d.is_integer() || d.as_int() < 0) {
            *err = name + ": bad shape";
            return false;
        }
        sizes.push_back(d.as_int());
        count *= static_cast<uint64_t>(d.as_int());
    }
    const featurize::Json &b = offsets->items()[0], &e = offsets->items()[1];
    if (!b.is_integer() || !e.is_integer() || b.as_int() < 0 || e.as_int() < b.as_int() ||
        static_cast<uint64_t>(e.as_int()) > data.size()) {
        *err = name + ": data_offsets outside the file";
        return false;
    }
    const uint64_t begin = static_cast<uint64_t>(b.as_int()), end = static_cast<uint64_t>(e.as_int());
    if (end - begin != count * info->width) {
        *err = name + ": " + std::to_string(end - begin) + " bytes for " + std::to_string(count) + " " + info->name +
               " values";
        return false;
    }
    std::vector<char> buf(data.begin() + static_cast<long>(begin), data.begin() + static_cast<long>(end));
    torch::Tensor t = torch::from_blob(buf.data(), sizes, torch::TensorOptions().dtype(info->stored));
    *out = t.to(info->loaded).clone();
    return true;
}

bool load_safetensors(const std::string &path, std::map<std::string, torch::Tensor> *out, std::string *err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        *err = "cannot open " + path;
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string bytes = ss.str();
    if (bytes.size() < 8) {
        *err = path + ": shorter than a safetensors header";
        return false;
    }
    uint64_t header_len = 0;
    for (int i = 7; i >= 0; --i)
        header_len = (header_len << 8) | static_cast<unsigned char>(bytes[static_cast<size_t>(i)]);
    if (8 + header_len > bytes.size()) {
        *err = path + ": header length past the end of the file";
        return false;
    }
    featurize::Json header;
    std::string perr;
    if (!featurize::Json::parse(bytes.substr(8, header_len), &header, &perr) || !header.is_object()) {
        *err = path + ": header is not a JSON object: " + perr;
        return false;
    }
    const std::string data = bytes.substr(8 + header_len);
    out->clear();
    for (const auto &m : header.members()) {
        if (m.first == "__metadata__") continue;
        torch::Tensor t;
        if (!tensor_of(data, m.first, m.second, &t, err)) {
            *err = path + ": " + *err;
            return false;
        }
        (*out)[m.first] = t;
    }
    return true;
}
