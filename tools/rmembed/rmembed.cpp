// rmembed: the oracle-text table's generator (plan step 5c, docs/featurizer.md §5). libtorch; links
// no engine translation unit (only src/featurize/ and the std-only schema for FNV-1a).
//
//   rmembed --model DIR --pin PIN --inputs ENTRIES.jsonl --table TABLE
//   rmembed --tokenize --model DIR < texts > ids
//   rmembed --embed-test --model DIR --texts FILE --out FILE [--pooling cls|mean] [--batch N]
//
// Table mode reads the inputs file the engine wrote (robomage --text-table-inputs, oracle_text.h:
// the row count, the vocab hash, the model, pooling and normalisation it was computed for, every
// present entry, and the entries to encode), checks it against PIN, tokenises each entry with the
// model's WordPiece tokenizer (a text longer than the model's position table is fatal), encodes,
// pools and normalises it (bert.h), and rewrites TABLE: an RMTB file of kind `text` whose array
// `vectors` is f32 [rows, hidden], a row per vocab id, the encoded rows replaced, every other present row's bytes kept,
// the rows of ids no longer present and the none row zero; rows_meta and the meta updated. A TABLE that is missing or
// was made for another model, pooling, normalisation, vocab or size is rebuilt from the inputs alone, which must then
// hold every present entry. Each text is encoded alone (batch 1) on one thread, so a row's value depends on its text
// only. Then a nearest-neighbour line per encoded row (its three most similar rows) for review.
//
// --tokenize prints, per stdin line, the line's token ids space-separated. --embed-test writes the
// pooled, normalised vectors of FILE's lines (one text per line) as an RMTB file of kind
// `tensors` (array `vectors`, f32 [texts, hidden]), in
// batches of N padded texts (default 1), with --pooling (default cls). Exit status 0 on success, 1
// on any error (printed on stderr as "rmembed: …").

#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../../src/featurize/oracle_text.h"
#include "../../src/featurize/rmtb.h"
#include "../../src/featurize/wordpiece.h"
#include "bert.h"

namespace {

struct Options {
        std::string model, pin, inputs, table, texts, out, pooling;
        bool tokenize = false;
        bool embed_test = false;
        int batch = 1;
};

}  // namespace

static bool parse_args(int argc, char **argv, Options *opt);
static int fail(const std::string &msg);
static bool pooling_of(const std::string &name, BertEncoder::Pooling *out);
static bool read_lines(const std::string &path, std::vector<std::string> *out);
static bool tokenize_all(const featurize::WordPiece &wp, const BertEncoder &enc, const std::vector<std::string> &texts,
    std::vector<std::vector<int64_t>> *ids, std::string *err);
static bool keeps_rows(const featurize::RmtbFile &old, const featurize::TextTableSpec &spec, int64_t hidden);
static void print_neighbours(const std::vector<float> &vectors, int64_t cols,
    const std::vector<featurize::RmtbRowMeta> &rows, const std::map<int64_t, std::string> &names,
    const std::vector<int64_t> &encoded);
static int run_tokenize(const Options &opt);
static int run_embed_test(const Options &opt);
static int run_table(const Options &opt);

static bool parse_args(int argc, char **argv, Options *opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](std::string *dst) {
            if (i + 1 >= argc) return false;
            *dst = argv[++i];
            return true;
        };
        std::string n;
        if (a == "--tokenize") {
            opt->tokenize = true;
        } else if (a == "--embed-test") {
            opt->embed_test = true;
        } else if (a == "--model") {
            if (!value(&opt->model)) return false;
        } else if (a == "--pin") {
            if (!value(&opt->pin)) return false;
        } else if (a == "--inputs") {
            if (!value(&opt->inputs)) return false;
        } else if (a == "--table") {
            if (!value(&opt->table)) return false;
        } else if (a == "--texts") {
            if (!value(&opt->texts)) return false;
        } else if (a == "--out") {
            if (!value(&opt->out)) return false;
        } else if (a == "--pooling") {
            if (!value(&opt->pooling)) return false;
        } else if (a == "--batch") {
            if (!value(&n)) return false;
            opt->batch = std::max(1, std::atoi(n.c_str()));
        } else {
            std::fprintf(stderr, "rmembed: unknown argument %s\n", a.c_str());
            return false;
        }
    }
    return !opt->model.empty();
}

static int fail(const std::string &msg) {
    std::fprintf(stderr, "rmembed: %s\n", msg.c_str());
    return 1;
}

static bool pooling_of(const std::string &name, BertEncoder::Pooling *out) {
    if (name == "cls" || name.empty()) {
        *out = BertEncoder::Pooling::CLS;
        return true;
    }
    if (name == "mean") {
        *out = BertEncoder::Pooling::MEAN;
        return true;
    }
    return false;
}

static bool read_lines(const std::string &path, std::vector<std::string> *out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out->push_back(line);
    }
    return true;
}

// The token ids of every text; false naming the first one the tokenizer refuses or the model's
// position table cannot hold (the encoder never truncates).
static bool tokenize_all(const featurize::WordPiece &wp, const BertEncoder &enc, const std::vector<std::string> &texts,
    std::vector<std::vector<int64_t>> *ids, std::string *err) {
    ids->assign(texts.size(), {});
    for (size_t i = 0; i < texts.size(); ++i) {
        std::string why;
        if (!wp.encode(texts[i], &(*ids)[i], &why)) {
            *err = "text " + std::to_string(i) + ": " + why;
            return false;
        }
        if (static_cast<int64_t>((*ids)[i].size()) > enc.max_positions()) {
            *err = "text " + std::to_string(i) + " is " + std::to_string((*ids)[i].size()) +
                   " tokens, past the model's " + std::to_string(enc.max_positions()) + " positions: \"" + texts[i] +
                   "\"";
            return false;
        }
    }
    return true;
}

// True when `old`'s rows were made for `spec` (model, pooling, normalisation, vocab, size).
static bool keeps_rows(const featurize::RmtbFile &old, const featurize::TextTableSpec &spec, int64_t hidden) {
    const featurize::RmtbArray *v = featurize::text_vectors(old);
    return old.kind == "text" && v && old.has_rows_meta && v->shape[0] == spec.rows &&
           v->shape[1] == static_cast<uint64_t>(hidden) && old.model == spec.model && old.pooling == spec.pooling &&
           old.norm_version == spec.norm_version && old.has_vocab_hash && old.vocab_hash == spec.vocab_hash;
}

static void print_neighbours(const std::vector<float> &vectors, int64_t cols,
    const std::vector<featurize::RmtbRowMeta> &rows, const std::map<int64_t, std::string> &names,
    const std::vector<int64_t> &encoded) {
    for (const int64_t id : encoded) {
        std::vector<std::pair<double, int64_t>> sims;
        for (const featurize::RmtbRowMeta &r : rows) {
            if (r.id == id) continue;
            double dot = 0;
            for (int64_t c = 0; c < cols; ++c)
                dot += static_cast<double>(vectors[static_cast<size_t>(id * cols + c)]) *
                       static_cast<double>(vectors[static_cast<size_t>(r.id * cols + c)]);
            sims.emplace_back(-dot, r.id);
        }
        std::sort(sims.begin(), sims.end());
        std::string line = "rmembed: near " + names.at(id) + ":";
        for (size_t k = 0; k < sims.size() && k < 3; ++k) {
            char buf[32];
            std::snprintf(buf, sizeof buf, " %.3f", -sims[k].first);
            line += std::string(k ? "," : "") + " " + names.at(sims[k].second) + buf;
        }
        std::printf("%s\n", line.c_str());
    }
}

static int run_tokenize(const Options &opt) {
    featurize::WordPiece wp;
    std::string err;
    if (!wp.load(opt.model, &err)) return fail(err);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<int64_t> ids;
        if (!wp.encode(line, &ids, &err)) return fail(err);
        std::string out;
        for (size_t i = 0; i < ids.size(); ++i) out += (i ? " " : "") + std::to_string(ids[i]);
        std::printf("%s\n", out.c_str());
    }
    return 0;
}

static int run_embed_test(const Options &opt) {
    BertEncoder::Pooling pooling;
    if (!pooling_of(opt.pooling, &pooling)) return fail("--pooling must be cls or mean");
    if (opt.texts.empty() || opt.out.empty()) return fail("--embed-test needs --texts FILE and --out FILE");
    featurize::WordPiece wp;
    BertEncoder enc;
    std::string err;
    if (!wp.load(opt.model, &err) || !enc.load(opt.model, &err)) return fail(err);
    std::vector<std::string> texts;
    if (!read_lines(opt.texts, &texts)) return fail("cannot read " + opt.texts);
    std::vector<std::vector<int64_t>> ids;
    if (!tokenize_all(wp, enc, texts, &ids, &err)) return fail(err);
    std::vector<float> values;
    for (size_t at = 0; at < ids.size(); at += static_cast<size_t>(opt.batch)) {
        const size_t end = std::min(ids.size(), at + static_cast<size_t>(opt.batch));
        const std::vector<std::vector<int64_t>> batch(
            ids.begin() + static_cast<long>(at), ids.begin() + static_cast<long>(end));
        const torch::Tensor v = enc.embed(batch, pooling).contiguous();
        const float *p = v.data_ptr<float>();
        values.insert(values.end(), p, p + v.numel());
    }
    featurize::RmtbFile t;
    t.kind = "tensors";
    t.arrays.push_back(featurize::rmtb_floats(
        featurize::TEXT_VECTORS, featurize::Dtype::F32, {texts.size(), static_cast<uint64_t>(enc.hidden())}, values));
    if (!featurize::rmtb_write_file(opt.out, t, &err)) return fail(err);
    return 0;
}

static int run_table(const Options &opt) {
    if (opt.pin.empty() || opt.inputs.empty() || opt.table.empty())
        return fail("table mode needs --pin PIN --inputs ENTRIES.jsonl --table TABLE");
    featurize::ModelPin pin;
    featurize::TextTableSpec spec;
    std::vector<std::pair<int64_t, std::string>> present;
    std::string err;
    if (!featurize::read_model_pin(opt.pin, &pin, &err)) return fail(err);
    if (!featurize::read_text_inputs(opt.inputs, &spec, &present, &err)) return fail(err);
    if (spec.model != pin.model() || spec.pooling != pin.pooling || spec.norm_version != featurize::NORM_VERSION)
        return fail(opt.inputs + " was written for " + spec.model + " pooling " + spec.pooling + " norm_version " +
                    std::to_string(spec.norm_version) + ", not the pin's " + pin.model() + " pooling " + pin.pooling +
                    " norm_version " + std::to_string(featurize::NORM_VERSION) + ": re-run --text-table-inputs");
    BertEncoder::Pooling pooling;
    if (!pooling_of(pin.pooling, &pooling)) return fail("pin pooling must be cls or mean");
    featurize::WordPiece wp;
    BertEncoder enc;
    if (!wp.load(opt.model, &err) || !enc.load(opt.model, &err)) return fail(err);
    if (static_cast<int64_t>(wp.vocab_size()) != enc.vocab_size())
        return fail("vocab.txt has " + std::to_string(wp.vocab_size()) + " tokens, the model " +
                    std::to_string(enc.vocab_size()));
    const int64_t hidden = enc.hidden();
    std::map<int64_t, std::string> names;
    for (const auto &p : present) {
        if (p.first < 0 || static_cast<uint64_t>(p.first) >= spec.rows)
            return fail("present id " + std::to_string(p.first) + " is outside the table's rows");
        names[p.first] = p.second;
    }
    featurize::RmtbFile old;
    std::ifstream probe(opt.table, std::ios::binary);
    const bool have_old = probe.is_open() && featurize::rmtb_read_file(opt.table, &old, &err);
    const bool keep = have_old && keeps_rows(old, spec, hidden);
    std::map<int64_t, uint64_t> old_hash;
    std::vector<float> old_vectors;
    if (keep) {
        featurize::rmtb_values(*featurize::text_vectors(old), &old_vectors);
        for (const featurize::RmtbRowMeta &r : old.rows_meta) old_hash[r.id] = r.hash;
    }
    std::map<int64_t, size_t> encode;  // id → index in spec.entries
    for (size_t i = 0; i < spec.entries.size(); ++i) {
        if (!names.count(spec.entries[i].id))
            return fail("entry " + std::to_string(spec.entries[i].id) + " is not in the inputs' present list");
        encode[spec.entries[i].id] = i;
    }
    for (const auto &n : names)
        if (!encode.count(n.first) && !old_hash.count(n.first))
            return fail("present entry " + std::to_string(n.first) + " " + n.second + " is neither in " + opt.table +
                        " nor in the inputs: re-run --text-table-inputs against this table");
    std::vector<std::string> texts;
    for (const featurize::TextEntry &e : spec.entries) texts.push_back(e.text);
    std::vector<std::vector<int64_t>> ids;
    if (!tokenize_all(wp, enc, texts, &ids, &err)) return fail(err);

    featurize::RmtbFile m;
    m.kind = "text";
    m.has_vocab_hash = true;
    m.vocab_hash = spec.vocab_hash;
    m.model = spec.model;
    m.pooling = spec.pooling;
    m.norm_version = spec.norm_version;
    m.has_rows_meta = true;
    const size_t cols = static_cast<size_t>(hidden);
    std::vector<float> vectors(static_cast<size_t>(spec.rows) * cols, 0.0f);
    std::vector<int64_t> encoded;
    size_t kept = 0;
    for (const auto &n : names) {
        const size_t row = static_cast<size_t>(n.first);
        auto e = encode.find(n.first);
        featurize::RmtbRowMeta rm;
        rm.id = n.first;
        if (e != encode.end()) {
            const torch::Tensor v = enc.embed({ids[e->second]}, pooling).contiguous();
            std::memcpy(&vectors[row * cols], v.data_ptr<float>(), cols * sizeof(float));
            rm.hash = spec.entries[e->second].hash;
            encoded.push_back(n.first);
        } else {
            std::memcpy(&vectors[row * cols], &old_vectors[row * cols], cols * sizeof(float));
            rm.hash = old_hash.at(n.first);
            ++kept;
        }
        m.rows_meta.push_back(rm);
    }
    m.arrays.push_back(featurize::rmtb_floats(
        featurize::TEXT_VECTORS, featurize::Dtype::F32, {spec.rows, static_cast<uint64_t>(hidden)}, vectors));
    if (!featurize::rmtb_write_file(opt.table, m, &err)) return fail(err);
    size_t dropped = 0;
    for (const auto &h : old_hash) dropped += names.count(h.first) ? 0 : 1;
    std::printf("rmembed: %s: %s, encoded %zu row(s), kept %zu, dropped %zu (%s, pooling %s, %lld x %lld)\n",
        opt.table.c_str(), keep ? "updated" : "rebuilt", encoded.size(), kept, dropped, m.model.c_str(),
        m.pooling.c_str(), static_cast<long long>(spec.rows), static_cast<long long>(hidden));
    print_neighbours(vectors, hidden, m.rows_meta, names, encoded);
    return 0;
}

int main(int argc, char **argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        std::fprintf(stderr,
            "usage: rmembed --model DIR --pin PIN --inputs ENTRIES.jsonl --table TABLE\n"
            "       rmembed --tokenize --model DIR < texts\n"
            "       rmembed --embed-test --model DIR --texts FILE --out FILE [--pooling cls|mean] [--batch N]\n");
        return 2;
    }
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    if (opt.tokenize) return run_tokenize(opt);
    if (opt.embed_test) return run_embed_test(opt);
    return run_table(opt);
}
