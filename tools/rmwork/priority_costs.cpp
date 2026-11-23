#include "priority_costs.h"

#include <cctype>

namespace {

// UTF-8 em dash, as oracle text writes a keyword's non-mana cost ("Flashback—Sacrifice a creature.").
const char *const EM_DASH = "\xE2\x80\x94";
// The keywords whose cost turns a face-down permanent face up (CR 702.37e, 702.168e).
const char *const FACE_UP_KEYWORDS[] = {"Megamorph", "Morph", "Disguise"};
// The way a spell is cast (the menu's `variant` bits) and the keyword whose line holds its cost.
struct VariantKeyword {
        const char *variant;
        const char *keyword;
};
const VariantKeyword VARIANT_KEYWORDS[] = {{"FLASHBACK", "Flashback"}, {"ESCAPE", "Escape"}, {"EVOKE", "Evoke"},
    {"WARP", "Warp"}, {"IMPENDING", "Impending"}, {"OVERLOAD", "Overload"}, {"BESTOW", "Bestow"},
    {"SUSPEND", "Suspend"}};

}  // namespace

static bool has_variant(const agents::DecisionView &v, size_t row, const char *name);
static const rmwork::CardFace *face_of(const rmwork::CardBook &book, int card, bool back);
static std::string oracle_of(const rmwork::CardBook &book, int card);
static std::string card_ability_cost(const agents::DecisionView &v, size_t row);
static std::vector<std::string> cost_tokens(const std::string &cost);
static bool mana_token(const std::string &t);
static std::string cost_part(const std::string &token);
static std::string object_phrase(const std::vector<std::string> &args, const char *adjective);
static std::string count_word(const std::string &n);
static bool card_type_word(const std::string &w);
static std::string lower_first(std::string s);
static std::string trim(const std::string &s);
static void add_text(rmwork::WayCost *c, const std::string &part);
static std::vector<std::string> split(const std::string &s, char sep);

static bool has_variant(const agents::DecisionView &v, size_t row, const char *name) {
    const agents::Table menu = v.table("menu");
    const int bit = v.stream().enum_value("cast_variant", name);
    return bit >= 0 && bit < 32 && (menu.u(row, menu.col("variant")) & (1ull << bit)) != 0;
}

// Face `back` of card `card`, else its other face.
static const rmwork::CardFace *face_of(const rmwork::CardBook &book, int card, bool back) {
    if (card < 0) return nullptr;
    const rmwork::CardFace *f = book.face(card, back);
    return f ? f : book.face(card, !back);
}

static std::string oracle_of(const rmwork::CardBook &book, int card) {
    const rmwork::CardFace *f = face_of(book, card, false);
    return f ? f->oracle : std::string();
}

// The `card_ability.cost` of the row's ability (`ab_card`, `ab_index`), "" when the CARDS frame has none.
static std::string card_ability_cost(const agents::DecisionView &v, size_t row) {
    const agents::Table menu = v.table("menu");
    const int card = menu.card(row, menu.col("ab_card"));
    const int c_index = menu.col("ab_index");
    if (card < 0 || menu.none(row, c_index)) return "";
    const uint64_t index = menu.u(row, c_index);
    const agents::Table a(v.stream().cards_frame(), v.stream().schema(), "card_ability");
    const int c_card = a.col("card"), c_idx = a.col("index"), c_cost = a.col("cost");
    for (size_t r = 0; r < a.rows(); ++r)
        if (a.card(r, c_card) == card && a.u(r, c_idx) == index) return a.text(r, c_cost);
    return "";
}

// The cost's space-separated parts; a part's <…> may hold spaces ("Sac<1/CARDNAME/this token>").
static std::vector<std::string> cost_tokens(const std::string &cost) {
    std::vector<std::string> out;
    std::string cur;
    int depth = 0;
    for (char ch : cost) {
        if (ch == '<') ++depth;
        if (ch == '>' && depth > 0) --depth;
        if (ch == ' ' && depth == 0) {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
            continue;
        }
        cur += ch;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// A mana symbol: a number, W U B R G C S X, a hybrid ("W/U", "2/W") or a Phyrexian pip ("U/P").
static bool mana_token(const std::string &t) {
    if (t.empty() || t.size() > 3) return false;
    for (char ch : t)
        if (!std::isdigit(static_cast<unsigned char>(ch)) && std::string("WUBRGCSXP/").find(ch) == std::string::npos)
            return false;
    return t != "/" && t != "P";
}

// "two Mountains", "a creature", "it" (CARDNAME), "this token" (the part's own words), with `adjective`
// before the type ("untapped").
static std::string object_phrase(const std::vector<std::string> &args, const char *adjective) {
    const std::string n = args.empty() ? "1" : args[0];
    const std::string type = args.size() > 1 ? args[1] : "Card";
    if (args.size() > 2 && !args[2].empty()) return n == "1" ? args[2] : n + " " + args[2];
    if (type.rfind("CARDNAME", 0) == 0) return "it";
    std::string base = type.substr(0, type.find_first_of(".+,;"));
    if (card_type_word(base)) base = lower_first(base);
    if (adjective) base = std::string(adjective) + " " + base;
    if (n == "1") {
        const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(base.empty() ? 'x' : base[0])));
        return std::string(std::string("aeiou").find(c) != std::string::npos ? "an " : "a ") + base;
    }
    return count_word(n) + " " + base + "s";
}

static std::string count_word(const std::string &n) {
    static const char *const WORDS[] = {"zero", "one", "two", "three", "four", "five", "six", "seven"};
    if (n.size() == 1 && n[0] >= '0' && n[0] <= '7') return WORDS[n[0] - '0'];
    return n;
}

static bool card_type_word(const std::string &w) {
    static const char *const TYPES[] = {"Card", "Land", "Creature", "Artifact", "Enchantment", "Planeswalker",
        "Instant", "Sorcery", "Permanent", "Battle", "Spell"};
    for (const char *t : TYPES)
        if (w == t) return true;
    return false;
}

static std::string lower_first(std::string s) {
    if (!s.empty()) s[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[0])));
    return s;
}

static std::string trim(const std::string &s) {
    const size_t a = s.find_first_not_of(" \t");
    if (a == std::string::npos) return "";
    const size_t b = s.find_last_not_of(" \t.");
    return b == std::string::npos || b < a ? "" : s.substr(a, b - a + 1);
}

static void add_text(rmwork::WayCost *c, const std::string &part) {
    if (part.empty()) return;
    c->text += (c->text.empty() ? "" : ", ") + part;
}

static std::vector<std::string> split(const std::string &s, char sep) {
    std::vector<std::string> out;
    size_t from = 0;
    while (true) {
        const size_t at = s.find(sep, from);
        out.push_back(s.substr(from, at == std::string::npos ? std::string::npos : at - from));
        if (at == std::string::npos) break;
        from = at + 1;
    }
    return out;
}

// One non-mana part of an ability's cost in words ("PayLife<1>": "pay 1 life").
static std::string cost_part(const std::string &token) {
    const size_t lt = token.find('<');
    if (lt == std::string::npos || token.back() != '>') return token;
    const std::string name = token.substr(0, lt);
    const std::vector<std::string> args = split(token.substr(lt + 1, token.size() - lt - 2), '/');
    const std::string n = args.empty() ? "1" : args[0];
    if (name == "PayLife") return "pay " + n + " life";
    if (name == "PayEnergy") return "pay " + n + " energy";
    if (name == "Sac") return "sacrifice " + object_phrase(args, nullptr);
    if (name == "Discard") {
        if (args.size() > 1 && args[1] == "Hand") return "discard your hand";
        return "discard " + object_phrase(args, nullptr);
    }
    if (name == "Exile") return "exile " + object_phrase(args, nullptr);
    if (name == "ExileFromGrave") return "exile " + object_phrase(args, nullptr) + " from your graveyard";
    if (name == "ExileFromHand") return "exile " + object_phrase(args, nullptr) + " from your hand";
    if (name == "Return") return "return " + object_phrase(args, nullptr) + " to its owner's hand";
    if (name == "tapXType") return "tap " + object_phrase(args, "untapped");
    if (name == "Mill") return "mill " + n;
    if (name == "AddCounter" || name == "SubCounter") {
        const std::string kind = args.size() > 1 ? args[1] : "";
        const bool add = name == "AddCounter";
        if (kind == "LOYALTY") return (add ? "+" : "\xE2\x88\x92") + n + " loyalty";
        const std::string label = kind == "P1P1" ? "+1/+1" : kind == "M1M1" ? "-1/-1" : lower_first(kind);
        return (add ? "put " : "remove ") + n + " " + label + (n == "1" ? " counter" : " counters");
    }
    std::string out = name;
    for (const std::string &a : args) out += " " + a;
    return out;
}

namespace rmwork {

WayCost ability_cost(const std::string &script_cost) {
    WayCost c;
    for (const std::string &t : cost_tokens(script_cost)) {
        if (t == "T" || t == "Q" || mana_token(t)) {
            c.pips.push_back(t);
        } else {
            add_text(&c, cost_part(t));
        }
    }
    return c;
}

WayCost keyword_cost(const std::string &oracle, const std::string &keyword) {
    WayCost c;
    for (const std::string &line : split(oracle, '\n')) {
        if (line.rfind(keyword, 0) != 0) continue;
        std::string rest = line.substr(keyword.size());
        // Not a longer word ("Escaped"), nor a keyword named in the middle of a sentence.
        if (!rest.empty() && std::isalpha(static_cast<unsigned char>(rest[0]))) continue;
        const size_t reminder = rest.find(" (");
        if (reminder != std::string::npos) rest = rest.substr(0, reminder);
        rest = trim(rest);
        // Suspend's count ("Suspend 4—{1}{U}").
        std::string count;
        while (!rest.empty() && std::isdigit(static_cast<unsigned char>(rest[0]))) {
            count += rest[0];
            rest = rest.substr(1);
        }
        if (rest.rfind(EM_DASH, 0) == 0) rest = rest.substr(3);
        rest = trim(rest);
        while (rest.size() > 2 && rest[0] == '{') {
            const size_t close = rest.find('}');
            if (close == std::string::npos) break;
            c.pips.push_back(rest.substr(1, close - 1));
            rest = rest.substr(close + 1);
        }
        if (rest.rfind(",", 0) == 0) rest = rest.substr(1);
        rest = trim(rest);
        if (!count.empty()) add_text(&c, "with " + count + " time counters");
        add_text(&c, lower_first(rest));
        return c;
    }
    return c;
}

std::string alternative_cost(const std::string &oracle) {
    for (const std::string &line : split(oracle, '\n')) {
        const size_t rather = line.find(" rather than pay");
        if (rather == std::string::npos) continue;
        size_t may = line.rfind("may ", rather);
        if (may == std::string::npos) continue;
        return trim(line.substr(may + 4, rather - may - 4));
    }
    return "";
}

std::string way_cost_text(const WayCost &c) {
    std::string out;
    for (const std::string &p : c.pips) out += "{" + p + "}";
    if (!c.text.empty()) out += (out.empty() ? "" : ", ") + c.text;
    return out;
}

std::string way_label(const std::string &description, const WayCost &c) {
    if (c.empty()) return description;
    return description + (c.text.empty() ? " " : " \xE2\x80\x94 ") + way_cost_text(c);
}

WayCost way_cost(const agents::DecisionView &v, const CardBook &book, size_t row) {
    WayCost c;
    if (row >= v.n()) return c;
    const std::string &category = v.category(row);
    const agents::Table menu = v.table("menu");
    const int card = menu.card(row, menu.col("source_card"));
    if (category == "ACTIVATE_ABILITY") return ability_cost(card_ability_cost(v, row));
    if (category == "COMPANION") {
        c.pips.push_back("3");
        c.text = "put it into your hand";
        return c;
    }
    if (category == "TURN_FACE_UP") {
        const std::string oracle = oracle_of(book, card);
        for (const char *k : FACE_UP_KEYWORDS) {
            c = keyword_cost(oracle, k);
            if (!c.empty()) return c;
        }
        const CardFace *f = face_of(book, card, false);
        if (f) c.pips = mana_symbols(f->mana_cost);
        return c;
    }
    if (category != "CAST_SPELL") return c;
    if (has_variant(v, row, "COMPANION")) {
        c.pips.push_back("3");
        c.text = "put it into your hand";
        return c;
    }
    if (has_variant(v, row, "WITHOUT_MANA_COST")) {
        c.text = "without paying its mana cost";
        return c;
    }
    if (has_variant(v, row, "FACE_DOWN")) {
        c.pips.push_back("3");
        c.text = "face down";
        return c;
    }
    const std::string oracle = oracle_of(book, card);
    for (const VariantKeyword &k : VARIANT_KEYWORDS)
        if (has_variant(v, row, k.variant)) {
            c = keyword_cost(oracle, k.keyword);
            if (c.empty()) c.text = lower_first(k.keyword);
            return c;
        }
    if (has_variant(v, row, "ALT_COST")) {
        c.text = alternative_cost(oracle);
        if (c.text.empty()) c.text = "its alternative cost";
        return c;
    }
    const bool back = has_variant(v, row, "BACK_FACE") || has_variant(v, row, "ADVENTURE");
    const CardFace *f = back ? book.face(card, true) : nullptr;
    if (!f) f = face_of(book, card, false);
    if (f) c.pips = mana_symbols(f->mana_cost);
    return c;
}

}  // namespace rmwork
