#include "embed.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <vector>

#include "../../src/featurize/json.h"
#include "../../src/featurize/oracle_text.h"
#include "../../src/featurize/rmtb.h"
#include "../../src/featurize/wordpiece.h"
#include "proc.h"
#include "util.h"

static std::string ids_text(const std::vector<int64_t> &ids);
static std::vector<std::pair<float, uint16_t>> f16_cases();
static featurize::RmtbFile sample_arrays();
static featurize::RmtbFile sample_text_table();
static std::string file_json(
    const std::string &path, const featurize::RmtbFile &written, const featurize::RmtbFile &read);
static void check_current_entries(const rmcheck::Options &opt, rmcheck::Report &rep, const featurize::WordPiece &wp,
    const std::set<std::string> &golden_texts);

static std::string ids_text(const std::vector<int64_t> &ids) {
    std::string out;
    for (size_t i = 0; i < ids.size(); ++i) out += (i ? " " : "") + std::to_string(ids[i]);
    return out;
}

// An f16 value's rounding (round to nearest, ties to even): each pair is a float and its binary16
// bits.
static std::vector<std::pair<float, uint16_t>> f16_cases() {
    return {{1.0f, 0x3C00}, {-0.0f, 0x8000}, {65504.0f, 0x7BFF}, {65520.0f, 0x7C00}, {1e9f, 0x7C00},
        {std::ldexp(1.0f, -24), 0x0001}, {std::ldexp(1.0f, -25), 0x0000}, {std::ldexp(3.0f, -25), 0x0002},
        {1.0f + std::ldexp(1.0f, -11), 0x3C00}, {1.0f + std::ldexp(3.0f, -11), 0x3C02}, {std::ldexp(1.0f, -14), 0x0400},
        {std::ldexp(1023.0f, -24) + std::ldexp(1.0f, -25), 0x0400}, {-2.5f, 0xC100},
        {std::numeric_limits<float>::infinity(), 0x7C00}};
}

// Every dtype, a rank-0, rank-1 and rank-3 shape, a ragged values + offsets pair, column names and
// sizes that need padding between arrays; f32 values a decimal rendering must keep exactly.
static featurize::RmtbFile sample_arrays() {
    featurize::RmtbFile f;
    f.kind = "tensors";
    f.has_spec_hash = true;
    f.spec_hash = 0x8000000000000001ULL;
    featurize::RmtbArray m = featurize::rmtb_floats("matrix", featurize::Dtype::F32, {3, 4},
        {1.0f / 3.0f, -2.5f, -0.0f, 1e-38f, 0, 0, 0, 0, 3.4e38f, -1.17549435e-38f, 0.1f, 7.0f});
    m.columns = {"a", "b \"quoted\"", "caf\xc3\xa9", "d"};
    f.arrays.push_back(m);
    std::vector<float> halves;
    for (const auto &c : f16_cases()) halves.push_back(c.first);
    halves.push_back(0.1f);
    halves.push_back(-1234.567f);
    f.arrays.push_back(featurize::rmtb_floats("halves", featurize::Dtype::F16, {2, 8}, halves));
    f.arrays.push_back(
        featurize::rmtb_ints("ragged_values", featurize::Dtype::I32, {5}, {-7, 2147483647, -2147483647 - 1, 0, 5}));
    f.arrays.push_back(featurize::rmtb_ints("ragged_offsets", featurize::Dtype::I64, {4}, {0, 2, 2, 5}));
    f.arrays.push_back(featurize::rmtb_ints("ids", featurize::Dtype::U16, {1, 3, 1}, {0, 65535, 1024}));
    f.arrays.push_back(featurize::rmtb_ints("bytes", featurize::Dtype::U8, {3}, {0, 255, 17}));
    f.arrays.push_back(featurize::rmtb_ints("big", featurize::Dtype::I64, {2, 2},
        {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min(), 1099511627776LL, -1}));
    f.arrays.push_back(featurize::rmtb_ints("scalar", featurize::Dtype::I64, {}, {42}));
    return f;
}

// A small text table: every file-level key, the vectors array.
static featurize::RmtbFile sample_text_table() {
    featurize::RmtbFile f;
    f.kind = "text";
    f.has_vocab_hash = true;
    f.vocab_hash = 0x0123456789abcdefULL;
    f.model = "example/model@0000000000000000000000000000000000000000";
    f.pooling = "mean";
    f.norm_version = 1;
    f.has_rows_meta = true;
    f.rows_meta = {{0, 0xfedcba9876543210ULL}, {2, 0x00000000000000ffULL}};
    f.arrays.push_back(
        featurize::rmtb_floats(featurize::TEXT_VECTORS, featurize::Dtype::F32, {3, 2}, {0.6f, 0.8f, 0, 0, -1.0f, 0}));
    return f;
}

// {"path":…,"meta":<the meta JSON>,"arrays":{"<name>":{"values":[…][,"source":[…]]},…}}: every
// array's values as the C++ reader decodes them (floats as "%.9g" with a ".0" on an integral one so
// -0 stays a float, exact for a float or a half; infinities as "inf" / "-inf"), and
// for an f16 array the floats it was written from.
static std::string file_json(
    const std::string &path, const featurize::RmtbFile &written, const featurize::RmtbFile &read) {
    auto floats = [](const std::vector<float> &v) {
        std::string out = "[";
        for (size_t i = 0; i < v.size(); ++i) {
            char buf[40];
            std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(v[i]));
            std::string num = buf;
            if (num.find_first_of(".e") == std::string::npos) num += ".0";  // JSON's -0 is an integer
            out += std::string(i ? "," : "") + (std::isinf(v[i]) ? (v[i] > 0 ? "\"inf\"" : "\"-inf\"") : num);
        }
        return out + "]";
    };
    std::string out = "{\"path\":" + featurize::json_quote(path) + ",\"meta\":" + featurize::rmtb_meta_json(written) +
                      ",\"arrays\":{";
    for (size_t i = 0; i < read.arrays.size(); ++i) {
        const featurize::RmtbArray &a = read.arrays[i];
        out += std::string(i ? "," : "") + featurize::json_quote(a.name) + ":{\"values\":";
        std::vector<float> f;
        std::vector<int64_t> n;
        if (featurize::rmtb_values(a, &f)) {
            out += floats(f);
        } else {
            featurize::rmtb_values(a, &n);
            out += "[";
            for (size_t k = 0; k < n.size(); ++k) out += (k ? "," : "") + std::to_string(n[k]);
            out += "]";
        }
        if (a.dtype == featurize::Dtype::F16) {
            std::vector<float> src;
            for (const auto &c : f16_cases()) src.push_back(c.first);
            src.push_back(0.1f);
            src.push_back(-1234.567f);
            out += ",\"source\":" + floats(src);
        }
        out += "}";
    }
    return out + "}}";
}

// Every current text entry tokenises (a card text with a character outside the tokenizer's tables
// is an error); the ones the golden does not hold yet are counted.
static void check_current_entries(const rmcheck::Options &opt, rmcheck::Report &rep, const featurize::WordPiece &wp,
    const std::set<std::string> &golden_texts) {
    const std::string dir = rmcheck::make_temp_dir(opt.out(""), ".embed_");
    const std::string inputs = rmcheck::path_join(dir, "inputs.jsonl");
    rmcheck::ProcSpec spec;
    spec.argv = {opt.binary, "--text-table-inputs", inputs, "--model-pin", opt.repo + "/tools/embed/MODEL_PIN"};
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 300;
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    featurize::TextTableSpec ts;
    std::vector<std::pair<int64_t, std::string>> present;
    std::string err;
    if (!r.ok() || !featurize::read_text_inputs(inputs, &ts, &present, &err)) {
        rep.error("vocab", "tokenizer: robomage --text-table-inputs failed (" + r.status_text() + "): " + err + r.err);
        rmcheck::remove_tree(dir);
        return;
    }
    rmcheck::remove_tree(dir);
    size_t fresh = 0;
    for (const featurize::TextEntry &e : ts.entries) {
        std::vector<int64_t> ids;
        if (!wp.encode(e.text, &ids, &err))
            rep.error("vocab", "tokenizer: " + std::to_string(e.id) + " " + e.name + ": " + err);
        if (!golden_texts.count(e.text)) ++fresh;
    }
    if (fresh)
        rep.info("tokenizer: " + std::to_string(fresh) + " of " + std::to_string(ts.entries.size()) +
                 " current text entries are not in the golden yet (re-record: tools/embed/make_goldens.py tokens)");
}

namespace rmcheck {

void embed_tokenizer_check(const Options &opt, Report &rep) {
    featurize::ModelPin pin;
    std::string err;
    if (!featurize::read_model_pin(opt.repo + "/tools/embed/MODEL_PIN", &pin, &err)) {
        rep.error("vocab", "tokenizer: " + err);
        return;
    }
    const std::string model = featurize::model_cache_dir(opt.repo + "/.cache/embed", pin);
    if (!file_exists(model + "/vocab.txt")) {
        rep.info("tokenizer: skipped: no model vocab (" + model + "/vocab.txt; python3 tools/embed/fetch_model.py)");
        return;
    }
    featurize::WordPiece wp;
    if (!wp.load(model, &err)) {
        rep.error("vocab", "tokenizer: " + err);
        return;
    }
    const std::string golden = opt.repo + "/tests/embed/tokens_golden.jsonl";
    std::string text;
    if (!read_file(golden, &text)) {
        rep.error("vocab", "tokenizer: cannot read " + golden);
        return;
    }
    const std::vector<std::string> lines = split_lines(text);
    featurize::Json head;
    if (lines.empty() || !featurize::Json::parse(lines[0], &head, &err) || !head.get("model") ||
        head.get("model")->text() != pin.model()) {
        rep.warn("vocab", "tokenizer: " + golden + " was not recorded for " + pin.model() +
                              ": re-record it (tools/embed/make_goldens.py tokens)");
        return;
    }
    size_t checked = 0, bad = 0;
    std::set<std::string> texts;
    for (size_t n = 1; n < lines.size(); ++n) {
        featurize::Json row;
        if (trim(lines[n]).empty()) continue;
        const featurize::Json *name = nullptr, *t = nullptr, *ids = nullptr;
        if (!featurize::Json::parse(lines[n], &row, &err) || !(name = row.get("name")) || !(t = row.get("text")) ||
            !(ids = row.get("ids")) || !ids->is_array()) {
            rep.error("vocab", "tokenizer: " + golden + ":" + std::to_string(n + 1) + ": not {name, text, ids}");
            ++bad;
            continue;
        }
        std::vector<int64_t> want, got;
        for (const featurize::Json &i : ids->items()) want.push_back(i.as_int());
        texts.insert(t->as_string());
        ++checked;
        if (!wp.encode(t->as_string(), &got, &err)) {
            if (++bad <= 5) rep.error("vocab", "tokenizer: " + name->as_string() + ": " + err);
        } else if (got != want) {
            if (++bad <= 5)
                rep.error("vocab",
                    "tokenizer: " + name->as_string() + ": ids " + ids_text(got) + ", golden " + ids_text(want));
        }
    }
    if (bad > 5) rep.error("vocab", "tokenizer: " + std::to_string(bad) + " golden texts differ in all");
    check_current_entries(opt, rep, wp, texts);
    if (!bad) rep.info("OK   tokenizer: " + std::to_string(checked) + " golden texts tokenise to the golden ids");
}

std::string write_rmtb_samples(const Options &opt, Report &rep) {
    for (const auto &c : f16_cases()) {
        const uint16_t h = featurize::f32_to_f16(c.first);
        if (h != c.second) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "rmtb: f16 of %.9g is 0x%04x, expected 0x%04x", static_cast<double>(c.first),
                static_cast<unsigned>(h), static_cast<unsigned>(c.second));
            rep.error("smoke", buf);
        }
        const float back = featurize::f16_to_f32(h);
        if (featurize::f32_to_f16(back) != h)
            rep.error("smoke", "rmtb: f16 0x" + featurize::hex16(h) + " does not round-trip");
    }
    const std::string dir = opt.out("");
    std::string json = "{\"files\":[";
    const featurize::RmtbFile files[] = {sample_arrays(), sample_text_table()};
    const char *names[] = {"smoke_rmtb_arrays.rmtb", "smoke_rmtb_text.rmtb"};
    for (size_t i = 0; i < 2; ++i) {
        const std::string path = path_join(dir, names[i]);
        std::string err;
        featurize::RmtbFile back;
        if (!featurize::rmtb_write_file(path, files[i], &err) || !featurize::rmtb_read_file(path, &back, &err)) {
            rep.error("smoke", "rmtb: " + err);
            return "";
        }
        if (featurize::rmtb_encode(back) != featurize::rmtb_encode(files[i])) {
            rep.error("smoke", "rmtb: " + path + " does not decode to the file written");
            return "";
        }
        json += std::string(i ? "," : "") + file_json(path, files[i], back);
    }
    const std::string path = path_join(dir, "smoke_rmtb.json");
    if (!write_file(path, json + "]}\n")) {
        rep.error("smoke", "rmtb: cannot write " + path);
        return "";
    }
    return path;
}

}  // namespace rmcheck
