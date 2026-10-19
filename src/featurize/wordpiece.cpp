#include "wordpiece.h"

#include <cstdio>
#include <fstream>
#include <sstream>

#include "json.h"

namespace {

// A character's role in the basic tokenization.
enum class CharClass {
    LETTER,  // part of a word
    PUNCT,   // a word of its own
    SPACE,   // separates words
    DROP,    // removed by cleaning (control and format characters)
    MARK,    // a nonspacing mark: removed when stripping accents, else part of a word
    UNKNOWN  // outside the tables
};

// The longest word WordPiece splits; a longer one is [UNK].
const size_t MAX_WORD_CHARS = 100;

// U+00A0-U+017F (Latin-1 Supplement, Latin Extended-A): each character's class (W space, P
// punctuation, X dropped: control or format, L letter) and its uncased form (NFD with nonspacing marks dropped, then
// lowercased; one character each). From Python's unicodedata (Unicode 15.1); the golden's coverage
// string checks both against the reference tokenizer.
const uint32_t LATIN_FIRST = 0x00A0;
const char LATIN_CLASS[] =
    "WPLLLLLPLLLPLXLLLLLLLLPPLLLPLLLPLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL"
    "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL";
// clang-format off
const uint16_t LATIN_UNCASED[] = {
    0x00A0, 0x00A1, 0x00A2, 0x00A3, 0x00A4, 0x00A5, 0x00A6, 0x00A7, 0x00A8, 0x00A9, 0x00AA, 0x00AB,
    0x00AC, 0x00AD, 0x00AE, 0x00AF, 0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7,
    0x00B8, 0x00B9, 0x00BA, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x00BF, 0x0061, 0x0061, 0x0061, 0x0061,
    0x0061, 0x0061, 0x00E6, 0x0063, 0x0065, 0x0065, 0x0065, 0x0065, 0x0069, 0x0069, 0x0069, 0x0069,
    0x00F0, 0x006E, 0x006F, 0x006F, 0x006F, 0x006F, 0x006F, 0x00D7, 0x00F8, 0x0075, 0x0075, 0x0075,
    0x0075, 0x0079, 0x00FE, 0x00DF, 0x0061, 0x0061, 0x0061, 0x0061, 0x0061, 0x0061, 0x00E6, 0x0063,
    0x0065, 0x0065, 0x0065, 0x0065, 0x0069, 0x0069, 0x0069, 0x0069, 0x00F0, 0x006E, 0x006F, 0x006F,
    0x006F, 0x006F, 0x006F, 0x00F7, 0x00F8, 0x0075, 0x0075, 0x0075, 0x0075, 0x0079, 0x00FE, 0x0079,
    0x0061, 0x0061, 0x0061, 0x0061, 0x0061, 0x0061, 0x0063, 0x0063, 0x0063, 0x0063, 0x0063, 0x0063,
    0x0063, 0x0063, 0x0064, 0x0064, 0x0111, 0x0111, 0x0065, 0x0065, 0x0065, 0x0065, 0x0065, 0x0065,
    0x0065, 0x0065, 0x0065, 0x0065, 0x0067, 0x0067, 0x0067, 0x0067, 0x0067, 0x0067, 0x0067, 0x0067,
    0x0068, 0x0068, 0x0127, 0x0127, 0x0069, 0x0069, 0x0069, 0x0069, 0x0069, 0x0069, 0x0069, 0x0069,
    0x0069, 0x0131, 0x0133, 0x0133, 0x006A, 0x006A, 0x006B, 0x006B, 0x0138, 0x006C, 0x006C, 0x006C,
    0x006C, 0x006C, 0x006C, 0x0140, 0x0140, 0x0142, 0x0142, 0x006E, 0x006E, 0x006E, 0x006E, 0x006E,
    0x006E, 0x0149, 0x014B, 0x014B, 0x006F, 0x006F, 0x006F, 0x006F, 0x006F, 0x006F, 0x0153, 0x0153,
    0x0072, 0x0072, 0x0072, 0x0072, 0x0072, 0x0072, 0x0073, 0x0073, 0x0073, 0x0073, 0x0073, 0x0073,
    0x0073, 0x0073, 0x0074, 0x0074, 0x0074, 0x0074, 0x0167, 0x0167, 0x0075, 0x0075, 0x0075, 0x0075,
    0x0075, 0x0075, 0x0075, 0x0075, 0x0075, 0x0075, 0x0075, 0x0075, 0x0077, 0x0077, 0x0079, 0x0079,
    0x0079, 0x007A, 0x007A, 0x007A, 0x007A, 0x007A, 0x007A, 0x017F,
};
// clang-format on
const uint32_t LATIN_LAST = LATIN_FIRST + sizeof(LATIN_UNCASED) / sizeof(LATIN_UNCASED[0]) - 1;
static_assert(sizeof(LATIN_CLASS) - 1 == sizeof(LATIN_UNCASED) / sizeof(LATIN_UNCASED[0]), "one class per character");
static_assert(LATIN_LAST == 0x017F, "the Latin table ends with Latin Extended-A");

// U+2000-U+206F (General Punctuation): each character's class. None has an uncased form of its
// own (the two canonical decompositions here are spaces). The unassigned U+2065 is a letter: the
// reference drops format and control characters, not unassigned ones.
const uint32_t PUNCT_FIRST = 0x2000;
const char PUNCT_CLASS[] =
    "WWWWWWWWWWWXXXXXPPPPPPPPPPPPPPPPPPPPPPPPWWXXXXXWPPPPPPPPPPPPPPPPPPPPLPPPPPPPPPPPPPLPPPPPPPPPPPPWXXXXXLXXXXXXXXXX";
const uint32_t PUNCT_LAST = PUNCT_FIRST + sizeof(PUNCT_CLASS) - 2;
static_assert(PUNCT_LAST == 0x206F, "the punctuation table is the General Punctuation block");

// U+2200-U+22FF (Mathematical Operators): every character is a letter (category Sm); the negated
// operators' uncased forms (their NFD is the operator and U+0338, a nonspacing mark), ascending.
const uint32_t MATH_FIRST = 0x2200;
const uint32_t MATH_LAST = 0x22FF;
// clang-format off
const uint32_t MATH_UNCASED[][2] = {
    {0x2204, 0x2203}, {0x2209, 0x2208}, {0x220C, 0x220B}, {0x2224, 0x2223}, {0x2226, 0x2225}, {0x2241, 0x223C},
    {0x2244, 0x2243}, {0x2247, 0x2245}, {0x2249, 0x2248}, {0x2260, 0x003D}, {0x2262, 0x2261}, {0x226D, 0x224D},
    {0x226E, 0x003C}, {0x226F, 0x003E}, {0x2270, 0x2264}, {0x2271, 0x2265}, {0x2274, 0x2272}, {0x2275, 0x2273},
    {0x2278, 0x2276}, {0x2279, 0x2277}, {0x2280, 0x227A}, {0x2281, 0x227B}, {0x2284, 0x2282}, {0x2285, 0x2283},
    {0x2288, 0x2286}, {0x2289, 0x2287}, {0x22AC, 0x22A2}, {0x22AD, 0x22A8}, {0x22AE, 0x22A9}, {0x22AF, 0x22AB},
    {0x22E0, 0x227C}, {0x22E1, 0x227D}, {0x22E2, 0x2291}, {0x22E3, 0x2292}, {0x22EA, 0x22B2}, {0x22EB, 0x22B3},
    {0x22EC, 0x22B4}, {0x22ED, 0x22B5},
};
// clang-format on

}  // namespace

static bool read_text(const std::string &path, std::string *out);
static bool decode_utf8(const std::string &s, std::vector<uint32_t> *out, std::string *err);
static void append_utf8(uint32_t cp, std::string *out);
static bool is_cjk(uint32_t cp);
static CharClass char_class(uint32_t cp);
static CharClass table_class(char c);
static uint32_t uncased(uint32_t cp);
static std::string utf8_of(const std::vector<uint32_t> &cps, size_t from, size_t to);
static bool config_flag(const featurize::Json &cfg, const char *key, bool fallback, std::string *err, bool *out);

static bool read_text(const std::string &path, std::string *out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    *out = ss.str();
    return true;
}

static bool decode_utf8(const std::string &s, std::vector<uint32_t> *out, std::string *err) {
    out->clear();
    for (size_t i = 0; i < s.size();) {
        const unsigned char b = static_cast<unsigned char>(s[i]);
        size_t len = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 0;
        if (len == 0 || i + len > s.size()) {
            *err = "malformed UTF-8 at byte " + std::to_string(i);
            return false;
        }
        uint32_t cp = len == 1 ? b : len == 2 ? (b & 0x1Fu) : len == 3 ? (b & 0x0Fu) : (b & 0x07u);
        for (size_t k = 1; k < len; ++k) {
            const unsigned char c = static_cast<unsigned char>(s[i + k]);
            if ((c >> 6) != 2) {
                *err = "malformed UTF-8 at byte " + std::to_string(i + k);
                return false;
            }
            cp = (cp << 6) | (c & 0x3Fu);
        }
        out->push_back(cp);
        i += len;
    }
    return true;
}

static void append_utf8(uint32_t cp, std::string *out) {
    if (cp < 0x80) {
        *out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        *out += static_cast<char>(0xC0 | (cp >> 6));
        *out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        *out += static_cast<char>(0xE0 | (cp >> 12));
        *out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        *out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        *out += static_cast<char>(0xF0 | (cp >> 18));
        *out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        *out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        *out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// The unified CJK ideograph blocks BERT splits off (its _is_chinese_char, less the compatibility
// ideographs, whose canonical decompositions the tables do not carry).
static bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF);
}

static CharClass table_class(char c) {
    switch (c) {
        case 'W':
            return CharClass::SPACE;
        case 'P':
            return CharClass::PUNCT;
        case 'X':
            return CharClass::DROP;
        default:
            return CharClass::LETTER;
    }
}

static CharClass char_class(uint32_t cp) {
    if (cp < 0x80) {
        if (cp == '\t' || cp == '\n' || cp == '\r' || cp == ' ') return CharClass::SPACE;
        if (cp < 0x20 || cp == 0x7F) return CharClass::DROP;
        const bool punct =
            (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126);
        return punct ? CharClass::PUNCT : CharClass::LETTER;
    }
    if (cp < LATIN_FIRST) return CharClass::DROP;  // the C1 controls
    if (cp <= LATIN_LAST) return table_class(LATIN_CLASS[cp - LATIN_FIRST]);
    if (cp >= 0x0300 && cp <= 0x036F) return CharClass::MARK;
    if (cp >= PUNCT_FIRST && cp <= PUNCT_LAST) return table_class(PUNCT_CLASS[cp - PUNCT_FIRST]);
    if (cp >= MATH_FIRST && cp <= MATH_LAST) return CharClass::LETTER;
    if (cp == 0x3000) return CharClass::SPACE;
    if (cp == 0xFFFD) return CharClass::DROP;
    if (is_cjk(cp)) return CharClass::LETTER;
    return CharClass::UNKNOWN;
}

// The uncased form of a letter or punctuation character (char_class known, not a mark).
static uint32_t uncased(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    if (cp >= LATIN_FIRST && cp <= LATIN_LAST) return LATIN_UNCASED[cp - LATIN_FIRST];
    if (cp >= MATH_FIRST && cp <= MATH_LAST)
        for (const auto &m : MATH_UNCASED)
            if (m[0] == cp) return m[1];
    return cp;
}

static std::string utf8_of(const std::vector<uint32_t> &cps, size_t from, size_t to) {
    std::string out;
    for (size_t i = from; i < to; ++i) append_utf8(cps[i], &out);
    return out;
}

// `key` of tokenizer_config.json as a bool (`fallback` when absent or null).
static bool config_flag(const featurize::Json &cfg, const char *key, bool fallback, std::string *err, bool *out) {
    const featurize::Json *v = cfg.get(key);
    *out = fallback;
    if (!v || v->is_null()) return true;
    if (!v->is_bool()) {
        *err = std::string("tokenizer_config.json: ") + key + " is not a bool";
        return false;
    }
    *out = v->as_bool();
    return true;
}

namespace featurize {

bool WordPiece::load(const std::string &model_dir, std::string *err) {
    std::string vocab, cfg_text;
    if (!read_text(model_dir + "/vocab.txt", &vocab)) {
        *err = "cannot read " + model_dir + "/vocab.txt";
        return false;
    }
    if (!read_text(model_dir + "/tokenizer_config.json", &cfg_text)) {
        *err = "cannot read " + model_dir + "/tokenizer_config.json";
        return false;
    }
    tokens_.clear();
    ids_.clear();
    size_t start = 0;
    while (start < vocab.size()) {
        size_t nl = vocab.find('\n', start);
        if (nl == std::string::npos) nl = vocab.size();
        std::string tok = vocab.substr(start, nl - start);
        if (!tok.empty() && tok.back() == '\r') tok.pop_back();
        ids_[tok] = static_cast<int64_t>(tokens_.size());
        tokens_.push_back(tok);
        start = nl + 1;
    }
    Json cfg;
    if (!Json::parse(cfg_text, &cfg, err) || !cfg.is_object()) {
        *err = "tokenizer_config.json: " + (err->empty() ? std::string("not an object") : *err);
        return false;
    }
    bool lower = true, basic = true;
    if (!config_flag(cfg, "do_lower_case", true, err, &lower) ||
        !config_flag(cfg, "tokenize_chinese_chars", true, err, &chinese_) ||
        !config_flag(cfg, "do_basic_tokenize", true, err, &basic))
        return false;
    const Json *strip = cfg.get("strip_accents");
    const bool strip_null = !strip || strip->is_null();
    if (!strip_null && !strip->is_bool()) {
        *err = "tokenizer_config.json: strip_accents is not a bool";
        return false;
    }
    const bool strip_on = strip_null ? lower : strip->as_bool();
    if (!basic || lower != strip_on) {
        *err =
            "tokenizer_config.json: only basic tokenization with lowercasing and accent stripping both on or both "
            "off is implemented";
        return false;
    }
    if (const Json *ns = cfg.get("never_split"))
        if (!ns->is_null() && !(ns->is_array() && ns->items().empty())) {
            *err = "tokenizer_config.json: never_split is not implemented";
            return false;
        }
    uncased_ = lower;
    int64_t mask = -1;
    const std::pair<const char *, int64_t *> specials[] = {
        {"cls_token", &cls_}, {"sep_token", &sep_}, {"unk_token", &unk_}, {"pad_token", &pad_}, {"mask_token", &mask}};
    const char *defaults[] = {"[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"};
    specials_.clear();
    for (size_t i = 0; i < 5; ++i) {
        const Json *v = cfg.get(specials[i].first);
        const std::string name = v && v->is_string() ? v->as_string() : defaults[i];
        auto it = ids_.find(name);
        if (it == ids_.end()) {
            *err = "vocab.txt has no " + name;
            return false;
        }
        *specials[i].second = it->second;
        specials_.emplace_back(name, it->second);
    }
    return true;
}

bool WordPiece::normalise(const std::vector<uint32_t> &in, std::vector<uint32_t> *out, std::string *err) const {
    out->clear();
    for (const uint32_t cp : in) {
        const CharClass c = char_class(cp);
        if (c == CharClass::UNKNOWN) {
            char buf[16];
            std::snprintf(buf, sizeof buf, "U+%04X", static_cast<unsigned>(cp));
            *err = std::string("character ") + buf +
                   " is outside the tokenizer's Unicode tables (src/featurize/wordpiece.cpp)";
            return false;
        }
        if (c == CharClass::DROP || (c == CharClass::MARK && uncased_)) continue;
        if (c == CharClass::SPACE) {
            out->push_back(' ');
        } else if (chinese_ && is_cjk(cp)) {
            out->push_back(' ');
            out->push_back(cp);
            out->push_back(' ');
        } else {
            out->push_back(uncased_ ? uncased(cp) : cp);
        }
    }
    return true;
}

bool WordPiece::word_ids(const std::vector<uint32_t> &word, std::vector<int64_t> *ids) const {
    if (word.size() > MAX_WORD_CHARS) return false;
    std::vector<int64_t> pieces;
    size_t start = 0;
    while (start < word.size()) {
        bool found = false;
        for (size_t end = word.size(); end > start; --end) {
            const std::string piece = (start > 0 ? "##" : "") + utf8_of(word, start, end);
            auto it = ids_.find(piece);
            if (it == ids_.end()) continue;
            pieces.push_back(it->second);
            start = end;
            found = true;
            break;
        }
        if (!found) return false;
    }
    ids->insert(ids->end(), pieces.begin(), pieces.end());
    return true;
}

bool WordPiece::encode(const std::string &text, std::vector<int64_t> *ids, std::string *err) const {
    ids->clear();
    ids->push_back(cls_);
    size_t at = 0;
    while (at <= text.size()) {
        size_t hit = std::string::npos, hit_len = 0;
        int64_t hit_id = -1;
        for (const auto &sp : specials_) {
            const size_t f = text.find(sp.first, at);
            if (f != std::string::npos && (f < hit || (f == hit && sp.first.size() > hit_len))) {
                hit = f;
                hit_len = sp.first.size();
                hit_id = sp.second;
            }
        }
        const size_t end = hit == std::string::npos ? text.size() : hit;
        if (!segment_ids(text.substr(at, end - at), ids, err)) return false;
        if (hit == std::string::npos) break;
        ids->push_back(hit_id);
        at = hit + hit_len;
    }
    ids->push_back(sep_);
    return true;
}

bool WordPiece::segment_ids(const std::string &text, std::vector<int64_t> *ids, std::string *err) const {
    std::vector<uint32_t> cps, norm;
    if (!decode_utf8(text, &cps, err) || !normalise(cps, &norm, err)) return false;
    std::vector<uint32_t> word;
    auto flush = [&]() {
        if (word.empty()) return;
        if (!word_ids(word, ids)) ids->push_back(unk_);
        word.clear();
    };
    for (const uint32_t cp : norm) {
        const CharClass c = char_class(cp);
        if (c == CharClass::SPACE) {
            flush();
        } else if (c == CharClass::PUNCT) {
            flush();
            word.push_back(cp);
            flush();
        } else {
            word.push_back(cp);
        }
    }
    flush();
    return true;
}

}  // namespace featurize
