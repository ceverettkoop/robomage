#include "text_table.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <set>
#include <vector>

#include "card_db.h"
#include "card_vocab.h"
#include "components/carddata.h"
#include "components/token.h"
#include "parse.h"

std::string text_table_check_path;
std::string text_table_inputs_path;
std::string text_table_path;
std::string model_pin_path;

static std::string type_line(const std::string &types_text, const std::set<Type> &types, bool token);
static std::string join_words(const std::vector<std::string> &words);
static bool has_text(const std::string &s);
static void add_card_entry(
    int id, const std::string &name, const std::string &prefix, std::vector<featurize::TextEntry> *out);
static void add_token_entry(
    const TokenVocabEntry &tv, const std::string &prefix, std::vector<featurize::TextEntry> *out);
static uint64_t engine_vocab_hash();
static bool load_table(const std::string &path, featurize::RmtbFile *table);
static int check_table(const featurize::TextTableSpec &spec);
static int write_inputs(const featurize::TextTableSpec &spec);

// "<supertypes> [Token] <card types> — <subtypes>" in the order the Types: line writes them, each
// word placed by the kind the parser gave it.
static std::string type_line(const std::string &types_text, const std::set<Type> &types, bool token) {
    std::vector<std::string> supers, cards, subs;
    std::vector<std::string> words;
    size_t at = 0;
    while (at < types_text.size()) {
        const size_t sp = types_text.find(' ', at);
        const size_t end = sp == std::string::npos ? types_text.size() : sp;
        if (end > at) words.push_back(types_text.substr(at, end - at));
        at = end + 1;
    }
    for (const std::string &w : words) {
        TypeCategory kind = SUBTYPE;
        for (const Type &t : types)
            if (t.name == w) {
                kind = t.kind;
                break;
            }
        (kind == SUPERTYPE ? supers : kind == TYPE ? cards : subs).push_back(w);
    }
    if (token) supers.push_back("Token");
    supers.insert(supers.end(), cards.begin(), cards.end());
    std::string line = join_words(supers);
    if (!subs.empty()) line += " \xE2\x80\x94 " + join_words(subs);
    return line;
}

static std::string join_words(const std::vector<std::string> &words) {
    std::string out;
    for (const std::string &w : words) out += (out.empty() ? "" : " ") + w;
    return out;
}

static bool has_text(const std::string &s) {
    return s.find_first_not_of(" \t\r\n") != std::string::npos;
}

// The entry of card vocab entry `id` (`name`): the face that has this id, when it has oracle text,
// and the other face too when it has the same id (a split card).
static void add_card_entry(
    int id, const std::string &name, const std::string &prefix, std::vector<featurize::TextEntry> *out) {
    const CardData *cd = card_data_by_name(name);
    if (!cd) return;
    const CardData *face = nullptr, *other = nullptr;
    if (card_name_to_index(cd->name) == id) {
        face = cd;
        other = cd->backside.get();
    } else if (cd->backside && card_name_to_index(cd->backside->name) == id) {
        face = cd->backside.get();
        other = cd;
    }
    if (!face || !has_text(face->oracle_text)) return;
    std::vector<std::string> names = {face->name};
    std::vector<featurize::FaceText> faces = {{type_line(face->types_text, face->types, false), face->oracle_text}};
    if (other) {
        names.push_back(other->name);
        if (card_name_to_index(other->name) == id && has_text(other->oracle_text))
            faces.push_back({type_line(other->types_text, other->types, false), other->oracle_text});
    }
    featurize::TextEntry e;
    e.id = id;
    e.name = name;
    e.text = featurize::normalise_card_text(faces, names, prefix);
    e.hash = featurize::text_hash(e.text);
    out->push_back(e);
}

// The entry of a registered token, when its script exists and has oracle text.
static void add_token_entry(
    const TokenVocabEntry &tv, const std::string &prefix, std::vector<featurize::TextEntry> *out) {
    if (token_script_text(tv.script).empty()) return;
    const Token tok = parse_token_script(tv.script);
    if (tok.name.empty() || !has_text(tok.oracle_text)) return;
    featurize::TextEntry e;
    e.id = tv.index;
    e.name = tv.name;
    e.text = featurize::normalise_card_text(
        {{type_line(tok.types_text, tok.types, true), tok.oracle_text}}, {tok.name}, prefix);
    e.hash = featurize::text_hash(e.text);
    out->push_back(e);
}

// featurize::vocab_hash over the card entries and the token band.
static uint64_t engine_vocab_hash() {
    std::vector<std::pair<int, std::string>> pairs;
    for (int i = 0; i < CARD_VOCAB_SIZE; ++i)
        pairs.emplace_back(card_vocab_entries[i].index, card_vocab_entries[i].name);
    for (int i = 0; i < TOKEN_VOCAB_SIZE; ++i)
        pairs.emplace_back(token_vocab_entries[i].index, token_vocab_entries[i].name);
    std::sort(pairs.begin(), pairs.end());
    return featurize::vocab_hash(pairs);
}

// The table at `path`; false (with a TEXT-TABLE line saying why) when it is missing or unreadable.
static bool load_table(const std::string &path, featurize::RmtbFile *table) {
    std::ifstream probe(path, std::ios::binary);
    if (!probe.is_open()) {
        std::printf("TEXT-TABLE: no table at %s\n", path.c_str());
        return false;
    }
    std::string err;
    if (!featurize::rmtb_read_file(path, table, &err)) {
        std::printf("TEXT-TABLE: unreadable table: %s\n", err.c_str());
        return false;
    }
    return true;
}

static int check_table(const featurize::TextTableSpec &spec) {
    featurize::RmtbFile table;
    const bool loaded = load_table(text_table_check_path, &table);
    const featurize::TextTableStatus st = featurize::text_table_status(loaded ? &table : nullptr, spec);
    if (st.ok()) {
        std::printf("TEXT-TABLE: ok, %zu entries (%s, pooling %s, norm_version %lld)\n", spec.entries.size(),
            spec.model.c_str(), spec.pooling.c_str(), static_cast<long long>(spec.norm_version));
        return 0;
    }
    for (const std::string &m : st.mismatches)
        if (loaded) std::printf("TEXT-TABLE: header mismatch: %s\n", m.c_str());
    for (const size_t i : st.stale)
        std::printf(
            "TEXT-TABLE: stale %lld %s\n", static_cast<long long>(spec.entries[i].id), spec.entries[i].name.c_str());
    for (const int64_t id : st.extra)
        std::printf(
            "TEXT-TABLE: extra %lld %s\n", static_cast<long long>(id), card_index_to_name(static_cast<int>(id)));
    std::printf("TEXT-TABLE: %zu of %zu entries stale, %zu extra row(s); regenerate with `make text-table`\n",
        st.stale.size(), spec.entries.size(), st.extra.size());
    return 1;
}

static int write_inputs(const featurize::TextTableSpec &spec) {
    std::vector<size_t> which;
    featurize::RmtbFile table;
    if (!text_table_path.empty() && load_table(text_table_path, &table)) {
        which = featurize::text_table_status(&table, spec).stale;
    } else {
        for (size_t i = 0; i < spec.entries.size(); ++i) which.push_back(i);
    }
    std::ofstream f(text_table_inputs_path, std::ios::binary | std::ios::trunc);
    const std::string text = featurize::text_inputs_jsonl(spec, which);
    f.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!f.good()) {
        std::fprintf(stderr, "TEXT-TABLE: cannot write %s\n", text_table_inputs_path.c_str());
        return 2;
    }
    std::printf("TEXT-TABLE: wrote %zu of %zu entries to %s\n", which.size(), spec.entries.size(),
        text_table_inputs_path.c_str());
    return 0;
}

bool text_table_requested() {
    return !text_table_check_path.empty() || !text_table_inputs_path.empty();
}

featurize::TextTableSpec text_table_spec(const featurize::ModelPin &pin) {
    featurize::TextTableSpec spec;
    spec.rows = static_cast<uint64_t>(N_CARD_TYPES) + 1;
    spec.vocab_hash = engine_vocab_hash();
    spec.model = pin.model();
    spec.pooling = pin.pooling;
    spec.norm_version = featurize::NORM_VERSION;
    std::set<int> seen;  // an id several vocab names alias (a split card) has one entry
    for (int i = 0; i < CARD_VOCAB_SIZE; ++i)
        if (seen.insert(card_vocab_entries[i].index).second)
            add_card_entry(card_vocab_entries[i].index, card_vocab_entries[i].name, pin.prefix, &spec.entries);
    for (int i = 0; i < TOKEN_VOCAB_SIZE; ++i) add_token_entry(token_vocab_entries[i], pin.prefix, &spec.entries);
    std::sort(spec.entries.begin(), spec.entries.end(),
        [](const featurize::TextEntry &a, const featurize::TextEntry &b) { return a.id < b.id; });
    return spec;
}

int run_text_table() {
    if (!text_table_check_path.empty() && !text_table_inputs_path.empty()) {
        std::fprintf(stderr, "TEXT-TABLE: give --text-table-check or --text-table-inputs, not both\n");
        return 2;
    }
    featurize::ModelPin pin;
    std::string err;
    if (model_pin_path.empty()) {
        std::fprintf(stderr, "TEXT-TABLE: --model-pin PIN is required\n");
        return 2;
    }
    if (!featurize::read_model_pin(model_pin_path, &pin, &err)) {
        std::fprintf(stderr, "TEXT-TABLE: %s\n", err.c_str());
        return 2;
    }
    const featurize::TextTableSpec spec = text_table_spec(pin);
    if (spec.entries.empty()) {
        std::fprintf(stderr,
            "TEXT-TABLE: no vocab card has oracle text: are the card scripts provisioned "
            "(python3 tools/forge_fetch/provision_decks.py)?\n");
        return 2;
    }
    return text_table_inputs_path.empty() ? check_table(spec) : write_inputs(spec);
}
