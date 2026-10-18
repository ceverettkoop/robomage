#include "metagame.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

#include "../classes/deck.h"
#include "../classes/deck_state.h"
#include "../game_driver.h"
#include "../queries/visibility.h"

namespace {

// The fallback label of an untagged deck, last in the list.
const char *const UNKNOWN_LABEL = "unknown";

// A JSON value of the subset archetypes.json uses: a string, a list of values, an object (its
// members in file order), or anything else (numbers, literals), whose text is not kept.
struct JsonValue {
        enum Kind {
            STRING,
            LIST,
            OBJECT,
            OTHER
        } kind = OTHER;
        std::string text;
        std::vector<JsonValue> items;
        std::vector<std::pair<std::string, JsonValue>> members;
};

// A recursive-descent reader over one document; `ok` false once anything does not parse.
struct JsonReader {
        const std::string &s;
        size_t at = 0;
        bool ok = true;
        explicit JsonReader(const std::string &src) : s(src) {}
};

}  // namespace

static void skip_space(JsonReader &r);
static bool read_string(JsonReader &r, std::string *out);
static bool read_value(JsonReader &r, JsonValue *out, int depth);
static bool parse_json(const std::string &text, JsonValue *out);
static ReferenceMetagame load_metagame();
static bool load_list(const std::string &stem, int archetype, ReferenceList *out, std::string *why);

static void skip_space(JsonReader &r) {
    while (r.at < r.s.size() && (r.s[r.at] == ' ' || r.s[r.at] == '\n' || r.s[r.at] == '\r' || r.s[r.at] == '\t'))
        ++r.at;
}

// A string at the reader's position (escapes decoded; a \u escape outside ASCII becomes '?').
static bool read_string(JsonReader &r, std::string *out) {
    out->clear();
    if (r.at >= r.s.size() || r.s[r.at] != '"') return false;
    ++r.at;
    while (r.at < r.s.size()) {
        const char c = r.s[r.at++];
        if (c == '"') return true;
        if (c != '\\') {
            out->push_back(c);
            continue;
        }
        if (r.at >= r.s.size()) return false;
        const char e = r.s[r.at++];
        switch (e) {
            case 'n':
                out->push_back('\n');
                break;
            case 't':
                out->push_back('\t');
                break;
            case 'r':
                out->push_back('\r');
                break;
            case 'b':
                out->push_back('\b');
                break;
            case 'f':
                out->push_back('\f');
                break;
            case 'u': {
                if (r.at + 4 > r.s.size()) return false;
                const unsigned long code = std::strtoul(r.s.substr(r.at, 4).c_str(), nullptr, 16);
                r.at += 4;
                out->push_back(code < 0x80 ? static_cast<char>(code) : '?');
                break;
            }
            default:
                out->push_back(e);
                break;
        }
    }
    return false;
}

static bool read_value(JsonReader &r, JsonValue *out, int depth) {
    *out = JsonValue{};
    if (depth > 32) return false;
    skip_space(r);
    if (r.at >= r.s.size()) return false;
    const char c = r.s[r.at];
    if (c == '"') {
        out->kind = JsonValue::STRING;
        return read_string(r, &out->text);
    }
    if (c == '[' || c == '{') {
        const bool object = c == '{';
        out->kind = object ? JsonValue::OBJECT : JsonValue::LIST;
        ++r.at;
        skip_space(r);
        if (r.at < r.s.size() && r.s[r.at] == (object ? '}' : ']')) {
            ++r.at;
            return true;
        }
        while (true) {
            JsonValue v;
            if (object) {
                std::string key;
                skip_space(r);
                if (!read_string(r, &key)) return false;
                skip_space(r);
                if (r.at >= r.s.size() || r.s[r.at] != ':') return false;
                ++r.at;
                if (!read_value(r, &v, depth + 1)) return false;
                out->members.emplace_back(key, v);
            } else {
                if (!read_value(r, &v, depth + 1)) return false;
                out->items.push_back(v);
            }
            skip_space(r);
            if (r.at >= r.s.size()) return false;
            const char sep = r.s[r.at++];
            if (sep == ',') continue;
            return sep == (object ? '}' : ']');
        }
    }
    // A number or a literal: its characters up to the next separator.
    const size_t start = r.at;
    while (r.at < r.s.size() && r.s[r.at] != ',' && r.s[r.at] != ']' && r.s[r.at] != '}' && r.s[r.at] != ' ' &&
           r.s[r.at] != '\n' && r.s[r.at] != '\r' && r.s[r.at] != '\t')
        ++r.at;
    return r.at > start;
}

static bool parse_json(const std::string &text, JsonValue *out) {
    JsonReader r(text);
    if (!read_value(r, out, 0)) return false;
    skip_space(r);
    return r.at == text.size();
}

// The deck `stem` as reference list: false (with `why`) when its file is missing or a card has no
// vocab entry.
static bool load_list(const std::string &stem, int archetype, ReferenceList *out, std::string *why) {
    std::ifstream f(RESOURCE_DIR + "/decks/" + stem + ".dk");
    if (!f.is_open()) {
        *why = "no deck file";
        return false;
    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    const Deck deck = Deck::from_string(buffer.str());
    *out = ReferenceList{};
    out->name = stem;
    out->archetype = archetype;
    for (int side = 0; side < 2; ++side)
        for (const auto &entry : side ? deck.sideboard : deck.main_deck) {
            const int vocab = deck_card_vocab(entry.second);
            if (vocab < 0) {
                *why = "card \"" + entry.second + "\" has no vocab entry";
                return false;
            }
            std::pair<int, int> &c = out->cards[physical_card_of(vocab)];
            (side ? c.second : c.first) += static_cast<int>(entry.first);
            (side ? out->side : out->main) += static_cast<int>(entry.first);
        }
    return true;
}

static ReferenceMetagame load_metagame() {
    ReferenceMetagame m;
    const std::string path = RESOURCE_DIR + "/decks/archetypes.json";
    std::ifstream f(path);
    std::stringstream buffer;
    if (f.is_open()) buffer << f.rdbuf();
    JsonValue doc;
    if (!f.is_open() || !parse_json(buffer.str(), &doc) || doc.kind != JsonValue::OBJECT) {
        m.skipped.push_back(path + ": not readable as a JSON object");
        return m;
    }
    for (const auto &kv : doc.members) {
        if (kv.first != "_archetypes") continue;
        for (const JsonValue &v : kv.second.items)
            if (v.kind == JsonValue::STRING) m.archetypes.push_back(v.text);
    }
    if (m.archetypes.empty()) {
        m.skipped.push_back(path + ": no _archetypes list");
        return m;
    }
    m.archetypes.push_back(UNKNOWN_LABEL);
    std::map<std::string, int> tagged;  // ascending stem
    for (const auto &kv : doc.members) {
        if (kv.first.empty() || kv.first[0] == '_') continue;
        int archetype = -1;
        for (size_t i = 0; i + 1 < m.archetypes.size(); ++i)
            if (kv.second.kind == JsonValue::STRING && m.archetypes[i] == kv.second.text)
                archetype = static_cast<int>(i);
        if (archetype < 0) {
            m.skipped.push_back(kv.first + ": its archetype is not in _archetypes");
            continue;
        }
        tagged[kv.first] = archetype;
    }
    for (const auto &t : tagged) {
        ReferenceList list;
        std::string why;
        if (load_list(t.first, t.second, &list, &why))
            m.lists.push_back(list);
        else
            m.skipped.push_back(t.first + ": " + why);
    }
    return m;
}

const ReferenceMetagame &reference_metagame() {
    static const ReferenceMetagame m = load_metagame();
    return m;
}
