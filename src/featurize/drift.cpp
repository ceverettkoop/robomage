#include "drift.h"

#include <algorithm>
#include <cstdio>
#include <map>

#include "../state_format/schema.h"
#include "json.h"
#include "rmtb.h"

namespace {

const char *const FORMAT_NAME = "CARD_DIGEST";
const int FORMAT_VERSION = 1;
// The most changed columns a table_drift line names.
const size_t SHOWN_COLUMNS = 6;

// One row's nonzero values by column name.
using RowValues = std::map<std::string, float>;

// A card's rows: its card row, per ability index its ability row and that row's parts (in row order).
struct CardRows {
        RowValues card;
        std::map<uint64_t, RowValues> abilities;
        std::map<uint64_t, std::vector<RowValues>> parts;
};

}  // namespace

static RowValues row_values(const std::vector<float> &values, const std::vector<std::string> &columns, size_t r);
static std::vector<uint8_t> described(const featurize::CardTables &t);
static std::vector<std::vector<uint64_t>> parts_by_ability(const featurize::CardTables &t);
static CardRows card_rows(const featurize::CardTables &t, const std::vector<std::vector<uint64_t>> &parts, uint64_t id);
static std::string printed(float v);
static void add_text(const RowValues &row, std::string *out);
static std::string digest_of(const CardRows &rows);
static void diff_row(const std::string &where, const RowValues &a, const RowValues &b, std::vector<std::string> *out);
static std::string card_label(uint64_t id, const std::string &name);
static std::string name_of(const std::vector<std::string> &names, uint64_t id);

static RowValues row_values(const std::vector<float> &values, const std::vector<std::string> &columns, size_t r) {
    RowValues out;
    const size_t w = columns.size();
    for (size_t c = 0; c < w; ++c)
        if (values[r * w + c] != 0.0f) out[columns[c]] = values[r * w + c];
    return out;
}

// Per id whether a card row describes it (its `present` column).
static std::vector<uint8_t> described(const featurize::CardTables &t) {
    std::vector<uint8_t> out(t.card_rows, 0);
    const size_t w = t.card_columns.size();
    for (size_t c = 0; c < w; ++c)
        if (t.card_columns[c] == "present")
            for (uint64_t id = 0; id < t.card_rows; ++id) out[id] = t.card[id * w + c] != 0.0f;
    return out;
}

// Per ability row the part rows that belong to it, in row order.
static std::vector<std::vector<uint64_t>> parts_by_ability(const featurize::CardTables &t) {
    std::vector<std::vector<uint64_t>> out(t.ability_rows());
    for (size_t p = 0; p < t.part_rows(); ++p) {
        const int64_t a = t.part_ability[p];
        if (a >= 0 && static_cast<size_t>(a) < out.size()) out[static_cast<size_t>(a)].push_back(p);
    }
    return out;
}

static CardRows card_rows(
    const featurize::CardTables &t, const std::vector<std::vector<uint64_t>> &parts, uint64_t id) {
    CardRows out;
    out.card = row_values(t.card, t.card_columns, id);
    for (uint64_t i = 0; i < t.abilities_per_card; ++i) {
        const uint64_t row = id * t.abilities_per_card + i;
        if (row >= t.ability_rows() || t.ability_card[row] < 0) continue;
        out.abilities[i] = row_values(t.ability, t.ability_columns, row);
        for (uint64_t p : parts[row]) out.parts[i].push_back(row_values(t.part, t.part_columns, p));
    }
    return out;
}

// 6 significant digits: a last-bit difference of another libm prints the same.
static std::string printed(float v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.6g", static_cast<double>(v));
    return buf;
}

static void add_text(const RowValues &row, std::string *out) {
    for (const auto &kv : row) *out += kv.first + "=" + printed(kv.second) + ";";
    *out += "|";
}

static std::string digest_of(const CardRows &rows) {
    std::string text;
    add_text(rows.card, &text);
    for (const auto &a : rows.abilities) {
        text += "a" + std::to_string(a.first) + ":";
        add_text(a.second, &text);
        auto it = rows.parts.find(a.first);
        if (it == rows.parts.end()) continue;
        for (const RowValues &p : it->second) {
            text += "p:";
            add_text(p, &text);
        }
    }
    return featurize::hex16(rmstate::fnv1a64(text.data(), text.size()));
}

// `where [col a -> b, …]` per column whose value differs, appended to `out`.
static void diff_row(const std::string &where, const RowValues &a, const RowValues &b, std::vector<std::string> *out) {
    std::map<std::string, std::pair<float, float>> diffs;
    for (const auto &kv : a) diffs[kv.first].first = kv.second;
    for (const auto &kv : b) diffs[kv.first].second = kv.second;
    std::string cols;
    size_t n = 0;
    for (const auto &d : diffs) {
        if (d.second.first == d.second.second) continue;
        if (n++ < SHOWN_COLUMNS)
            cols += std::string(cols.empty() ? "" : ", ") + d.first + " " + printed(d.second.first) + " -> " +
                    printed(d.second.second);
    }
    if (n > SHOWN_COLUMNS) cols += " and " + std::to_string(n - SHOWN_COLUMNS) + " more";
    if (n) out->push_back(where + " [" + cols + "]");
}

static std::string card_label(uint64_t id, const std::string &name) {
    return "card " + std::to_string(id) + (name.empty() ? std::string() : " '" + name + "'");
}

static std::string name_of(const std::vector<std::string> &names, uint64_t id) {
    return id < names.size() ? names[id] : std::string();
}

namespace featurize {

std::vector<CardDigest> card_digests(const CardTables &t, const std::vector<std::string> &names) {
    std::vector<CardDigest> out;
    const std::vector<std::vector<uint64_t>> parts = parts_by_ability(t);
    const std::vector<uint8_t> present = described(t);
    for (uint64_t id = 0; id < t.card_rows; ++id) {
        if (!present[id]) continue;
        CardDigest d;
        d.id = id;
        d.name = name_of(names, id);
        d.digest = digest_of(card_rows(t, parts, id));
        out.push_back(d);
    }
    return out;
}

std::string digest_json(const std::vector<CardDigest> &digests) {
    std::string out = std::string("{\"format\":\"") + FORMAT_NAME + "\",\"version\":" + std::to_string(FORMAT_VERSION) +
                      ",\n\"cards\":[";
    for (size_t i = 0; i < digests.size(); ++i)
        out += std::string(i ? ",\n" : "\n") + "{\"id\":" + std::to_string(digests[i].id) +
               ",\"name\":" + json_quote(digests[i].name) + ",\"digest\":\"" + digests[i].digest + "\"}";
    return out + "\n]}\n";
}

bool parse_digest_json(const std::string &text, std::vector<CardDigest> *out, std::string *err) {
    Json j;
    if (!Json::parse(text, &j, err)) return false;
    const Json *format = j.get("format"), *version = j.get("version"), *cards = j.get("cards");
    if (!j.is_object() || !format || format->text() != FORMAT_NAME || !version || version->as_int() != FORMAT_VERSION ||
        !cards || !cards->is_array()) {
        *err = std::string("not a ") + FORMAT_NAME + " version " + std::to_string(FORMAT_VERSION) + " file";
        return false;
    }
    out->clear();
    for (const Json &c : cards->items()) {
        const Json *id = c.get("id"), *name = c.get("name"), *digest = c.get("digest");
        if (!id || !id->is_integer() || !name || !name->is_string() || !digest || !digest->is_string()) {
            *err = "a card digest entry is not {id, name, digest}";
            return false;
        }
        out->push_back({static_cast<uint64_t>(id->as_int()), name->as_string(), digest->as_string()});
    }
    return true;
}

Drift digest_drift(const std::vector<CardDigest> &base, const std::vector<CardDigest> &now) {
    Drift d;
    std::map<uint64_t, const CardDigest *> cur;
    for (const CardDigest &c : now) cur[c.id] = &c;
    for (const CardDigest &b : base) {
        auto it = cur.find(b.id);
        if (it == cur.end()) {
            d.removed.push_back(card_label(b.id, b.name));
            continue;
        }
        if (it->second->name != b.name)
            d.changed.push_back(card_label(b.id, it->second->name) + ": named '" + b.name + "' before");
        else if (it->second->digest != b.digest)
            d.changed.push_back(card_label(b.id, b.name) + ": its card, ability or part rows changed");
        cur.erase(it);
    }
    for (const auto &kv : cur) d.added.push_back(card_label(kv.first, kv.second->name));
    return d;
}

Drift table_drift(const CardTables &before, const std::vector<std::string> &before_names, const CardTables &after,
    const std::vector<std::string> &after_names) {
    Drift d;
    const std::vector<std::vector<uint64_t>> parts_a = parts_by_ability(before), parts_b = parts_by_ability(after);
    const std::vector<uint8_t> present_a = described(before), present_b = described(after);
    const uint64_t rows = std::max(before.card_rows, after.card_rows);
    for (uint64_t id = 0; id < rows; ++id) {
        const bool in_a = id < before.card_rows && present_a[id], in_b = id < after.card_rows && present_b[id];
        if (!in_a && !in_b) continue;
        if (!in_b) {
            d.removed.push_back(card_label(id, name_of(before_names, id)));
            continue;
        }
        if (!in_a) {
            d.added.push_back(card_label(id, name_of(after_names, id)));
            continue;
        }
        const CardRows a = card_rows(before, parts_a, id), b = card_rows(after, parts_b, id);
        std::vector<std::string> what;
        const std::string was = name_of(before_names, id), is = name_of(after_names, id);
        if (!was.empty() && !is.empty() && was != is)
            what.push_back("named '" + name_of(before_names, id) + "' before");
        diff_row("card", a.card, b.card, &what);
        std::map<uint64_t, std::pair<const RowValues *, const RowValues *>> abilities;
        for (const auto &x : a.abilities) abilities[x.first].first = &x.second;
        for (const auto &x : b.abilities) abilities[x.first].second = &x.second;
        const RowValues none;
        for (const auto &x : abilities) {
            const std::string where = "ability " + std::to_string(x.first);
            diff_row(where, x.second.first ? *x.second.first : none, x.second.second ? *x.second.second : none, &what);
            auto pa = a.parts.find(x.first), pb = b.parts.find(x.first);
            const size_t na = pa == a.parts.end() ? 0 : pa->second.size(),
                         nb = pb == b.parts.end() ? 0 : pb->second.size();
            if (na != nb) {
                what.push_back(where + " [" + std::to_string(na) + " parts -> " + std::to_string(nb) + "]");
                continue;
            }
            for (size_t p = 0; p < na; ++p)
                diff_row(where + " part " + std::to_string(p), pa->second[p], pb->second[p], &what);
        }
        if (what.empty()) continue;
        std::string line = card_label(id, name_of(after_names, id)) + ":";
        for (size_t i = 0; i < what.size(); ++i) line += std::string(i ? ";" : "") + " " + what[i];
        d.changed.push_back(line);
    }
    return d;
}

}  // namespace featurize
