#include "oracle_text.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#include "../state_format/schema.h"
#include "json.h"

static bool is_word_byte(char c);
static bool word_at(const std::string &s, size_t at, const std::string &word);
static std::vector<std::string> replaced_names(const std::vector<std::string> &names);
static std::string trim_ws(const std::string &s);
static bool hash_text(const featurize::Json &v, uint64_t *out);
static bool parse_header(const featurize::Json &h, featurize::TextTableSpec *spec,
    std::vector<std::pair<int64_t, std::string>> *present, std::string *err);
static bool parse_entry(const featurize::Json &e, featurize::TextEntry *out);

static bool is_word_byte(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

// True when `word` occurs at `at` of `s` with no ASCII letter or digit on either side.
static bool word_at(const std::string &s, size_t at, const std::string &word) {
    if (s.compare(at, word.size(), word) != 0) return false;
    if (at > 0 && is_word_byte(s[at - 1])) return false;
    const size_t end = at + word.size();
    return end >= s.size() || !is_word_byte(s[end]);
}

// The names normalise_card_text replaces: each name and its short name, longest first, distinct.
static std::vector<std::string> replaced_names(const std::vector<std::string> &names) {
    std::set<std::string> all;
    for (const std::string &n : names) {
        if (n.empty()) continue;
        all.insert(n);
        const size_t comma = n.find(", ");
        if (comma != std::string::npos && comma > 0) all.insert(n.substr(0, comma));
    }
    std::vector<std::string> out(all.begin(), all.end());
    std::stable_sort(
        out.begin(), out.end(), [](const std::string &a, const std::string &b) { return a.size() > b.size(); });
    return out;
}

static std::string trim_ws(const std::string &s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return std::string();
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static bool hash_text(const featurize::Json &v, uint64_t *out) {
    return v.is_string() && featurize::parse_hex16(v.as_string(), out);
}

static bool parse_header(const featurize::Json &h, featurize::TextTableSpec *spec,
    std::vector<std::pair<int64_t, std::string>> *present, std::string *err) {
    const featurize::Json *rows = h.get("rows"), *vh = h.get("vocab_hash"), *model = h.get("model"),
                          *pooling = h.get("pooling"), *norm = h.get("norm_version"), *pres = h.get("present");
    if (!h.is_object() || !rows || !rows->is_integer() || !vh || !hash_text(*vh, &spec->vocab_hash) || !model ||
        !model->is_string() || !pooling || !pooling->is_string() || !norm || !norm->is_integer() || !pres ||
        !pres->is_array()) {
        *err = "the header is not {rows, vocab_hash, model, pooling, norm_version, present}";
        return false;
    }
    spec->rows = static_cast<uint64_t>(rows->as_int());
    spec->model = model->as_string();
    spec->pooling = pooling->as_string();
    spec->norm_version = norm->as_int();
    for (const featurize::Json &p : pres->items()) {
        const featurize::Json *id = p.get("id"), *name = p.get("name");
        if (!id || !id->is_integer() || !name || !name->is_string()) {
            *err = "a present entry is not {id, name}";
            return false;
        }
        present->emplace_back(id->as_int(), name->as_string());
    }
    return true;
}

static bool parse_entry(const featurize::Json &e, featurize::TextEntry *out) {
    const featurize::Json *id = e.get("id"), *name = e.get("name"), *text = e.get("text"), *hash = e.get("hash");
    if (!e.is_object() || !id || !id->is_integer() || !name || !name->is_string() || !text || !text->is_string() ||
        !hash || !hash_text(*hash, &out->hash))
        return false;
    out->id = id->as_int();
    out->name = name->as_string();
    out->text = text->as_string();
    return true;
}

namespace featurize {

bool read_model_pin(const std::string &path, ModelPin *out, std::string *err) {
    std::ifstream f(path);
    if (!f.is_open()) {
        *err = "cannot read model pin " + path;
        return false;
    }
    *out = ModelPin();
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(f, line)) {
        const std::string t = trim_ws(line);
        if (t.empty() || t[0] == '#') continue;
        const size_t sp = t.find(' ');
        kv[t.substr(0, sp)] = sp == std::string::npos ? std::string() : trim_ws(t.substr(sp + 1));
    }
    out->repo = kv["repo"];
    out->commit = kv["commit"];
    out->pooling = kv["pooling"];
    for (const char *key : {"repo", "commit", "pooling"})
        if (kv[key].empty()) {
            *err = path + ": missing '" + key + "' line";
            return false;
        }
    if (out->pooling != "cls" && out->pooling != "mean") {
        *err = path + ": pooling '" + out->pooling + "' is not cls or mean";
        return false;
    }
    auto it = kv.find("prefix");
    if (it != kv.end()) {
        Json v;
        std::string perr;
        if (!Json::parse(it->second, &v, &perr) || !v.is_string()) {
            *err = path + ": prefix is not a JSON string";
            return false;
        }
        out->prefix = v.as_string();
    }
    return true;
}

std::string model_cache_dir(const std::string &cache_root, const ModelPin &pin) {
    std::string repo = pin.repo;
    for (size_t at = repo.find('/'); at != std::string::npos; at = repo.find('/', at + 2)) repo.replace(at, 1, "__");
    return cache_root + "/" + repo + "/" + pin.commit;
}

uint64_t text_hash(const std::string &normalised) {
    return rmstate::fnv1a64(normalised.data(), normalised.size());
}

uint64_t vocab_hash(const std::vector<std::pair<int, std::string>> &entries) {
    std::string bytes;
    for (const auto &e : entries) {
        const uint32_t id = static_cast<uint32_t>(e.first);
        for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<char>((id >> (8 * i)) & 0xFF));
        bytes += e.second;
        bytes.push_back('\0');
    }
    return rmstate::fnv1a64(bytes.data(), bytes.size());
}

std::string normalise_card_text(
    const std::vector<FaceText> &faces, const std::vector<std::string> &names, const std::string &prefix) {
    const std::vector<std::string> words = replaced_names(names);
    std::string raw;
    for (const FaceText &f : faces) {
        std::string body;
        for (size_t i = 0; i < f.oracle.size();) {
            bool hit = false;
            for (const std::string &w : words)
                if (word_at(f.oracle, i, w)) {
                    body += "CARDNAME";
                    i += w.size();
                    hit = true;
                    break;
                }
            if (!hit) body += f.oracle[i++];
        }
        raw += (raw.empty() ? "" : " // ") + f.type_line + ". " + body;
    }
    std::string out;
    bool space = false;
    for (char c : raw) {
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
        if (c == ' ') {
            space = true;
            continue;
        }
        if (space && !out.empty()) out += ' ';
        space = false;
        out += c;
    }
    return prefix + out;
}

const RmtbArray *text_vectors(const RmtbFile &file) {
    const RmtbArray *v = file.find(TEXT_VECTORS);
    return v && v->dtype == Dtype::F32 && v->shape.size() == 2 ? v : nullptr;
}

TextTableStatus text_table_status(const RmtbFile *table, const TextTableSpec &spec) {
    TextTableStatus st;
    if (!table) {
        st.mismatches.push_back("no table");
    } else {
        const RmtbFile &m = *table;
        const RmtbArray *v = text_vectors(m);
        if (m.kind != "text" || !v || !m.has_rows_meta)
            st.mismatches.push_back("not a text table with rows_meta and an f32 [rows, hidden] vectors array");
        else if (v->shape[0] != spec.rows)
            st.mismatches.push_back("rows " + std::to_string(v->shape[0]) + ", expected " + std::to_string(spec.rows));
        if (m.model != spec.model) st.mismatches.push_back("model " + m.model + ", pin " + spec.model);
        if (m.pooling != spec.pooling) st.mismatches.push_back("pooling " + m.pooling + ", pin " + spec.pooling);
        if (m.norm_version != spec.norm_version)
            st.mismatches.push_back(
                "norm_version " + std::to_string(m.norm_version) + ", expected " + std::to_string(spec.norm_version));
        if (!m.has_vocab_hash || m.vocab_hash != spec.vocab_hash)
            st.mismatches.push_back("vocab_hash " + (m.has_vocab_hash ? hex16(m.vocab_hash) : std::string("absent")) +
                                    ", vocab " + hex16(spec.vocab_hash));
    }
    if (!st.mismatches.empty()) {
        for (size_t i = 0; i < spec.entries.size(); ++i) st.stale.push_back(i);
        return st;
    }
    std::map<int64_t, uint64_t> rows;
    for (const RmtbRowMeta &r : table->rows_meta) rows[r.id] = r.hash;
    std::set<int64_t> ids;
    for (size_t i = 0; i < spec.entries.size(); ++i) {
        const TextEntry &e = spec.entries[i];
        ids.insert(e.id);
        auto it = rows.find(e.id);
        if (it == rows.end() || it->second != e.hash) st.stale.push_back(i);
    }
    for (const auto &r : rows)
        if (!ids.count(r.first)) st.extra.push_back(r.first);
    return st;
}

std::string text_inputs_jsonl(const TextTableSpec &spec, const std::vector<size_t> &which) {
    std::string out = "{\"rows\":" + std::to_string(spec.rows) + ",\"vocab_hash\":\"" + hex16(spec.vocab_hash) +
                      "\",\"model\":" + json_quote(spec.model) + ",\"pooling\":" + json_quote(spec.pooling) +
                      ",\"norm_version\":" + std::to_string(spec.norm_version) + ",\"present\":[";
    for (size_t i = 0; i < spec.entries.size(); ++i)
        out += std::string(i ? "," : "") + "{\"id\":" + std::to_string(spec.entries[i].id) +
               ",\"name\":" + json_quote(spec.entries[i].name) + "}";
    out += "]}\n";
    for (const size_t i : which) {
        const TextEntry &e = spec.entries[i];
        out += "{\"id\":" + std::to_string(e.id) + ",\"name\":" + json_quote(e.name) +
               ",\"text\":" + json_quote(e.text) + ",\"hash\":\"" + hex16(e.hash) + "\"}\n";
    }
    return out;
}

bool read_text_inputs(const std::string &path, TextTableSpec *spec,
    std::vector<std::pair<int64_t, std::string>> *present, std::string *err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        *err = "cannot read " + path;
        return false;
    }
    *spec = TextTableSpec();
    present->clear();
    std::string line;
    size_t n = 0;
    while (std::getline(f, line)) {
        ++n;
        if (trim_ws(line).empty()) continue;
        Json v;
        std::string perr;
        if (!Json::parse(line, &v, &perr)) {
            *err = path + ":" + std::to_string(n) + ": " + perr;
            return false;
        }
        if (n == 1) {
            if (!parse_header(v, spec, present, &perr)) {
                *err = path + ":1: " + perr;
                return false;
            }
            continue;
        }
        TextEntry e;
        if (!parse_entry(v, &e)) {
            *err = path + ":" + std::to_string(n) + ": not {id, name, text, hash}";
            return false;
        }
        spec->entries.push_back(e);
    }
    if (n == 0) {
        *err = path + ": empty";
        return false;
    }
    return true;
}

}  // namespace featurize
