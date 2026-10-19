#include "play_spec.h"

#include <cctype>
#include <map>
#include <set>
#include <tuple>

#include "../str_util.h"

namespace agents {

namespace {

// A parsed spec: what the author wants, independent of any menu.
struct Intent {
        std::string verb;
        std::set<std::string> cats;       // candidate categories (empty: any)
        std::optional<std::string> card;  // normalized card / description text
        std::string side;                 // "own" | "opp" | "" (the action's controller)
        char seat = 0;                    // 'A' | 'B' | 0
        std::string color;                // a MANA_* category, or ""
        std::string keyword;              // "done" | "fail" | "keep" | "mulligan" | ""
        int literal = -1;                 // a literal menu index, or -1
        bool has_sentinel = false;
        int sentinel = 0;  // a concession sentinel
};

}  // namespace

static std::string norm(const std::string &s);
static std::string trim(const std::string &s);
static std::string lower(const std::string &s);
static std::string quoted(const std::string &s);
static bool split_seat_key(const std::string &raw, char *seat, std::string *rest);
static bool parse_literal(const std::string &raw, int *out);
static const std::map<std::string, std::set<std::string>> &verb_cats();
static bool parse_spec(const std::string &token, Intent *out, std::string *err);
static bool card_matches(const std::string &intent_card, const MenuAction &a, bool exact);
static bool matches(const Intent &intent, const MenuAction &a, bool exact);
static const char *controller_word(int controller);

static constexpr int CONCEDE_GAME = -2;
static constexpr int CONCEDE_MATCH = -3;

// Lowercase, apostrophes dropped, every other character outside [a-z0-9 ] a space, whitespace
// collapsed and trimmed (action_spec._norm; a non-ASCII character becomes one space there, its
// bytes several here, collapsed alike).
static std::string norm(const std::string &s) {
    std::string spaced;
    spaced.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == '\'') continue;
        if (c == 0xE2 && i + 2 < s.size() && static_cast<unsigned char>(s[i + 1]) == 0x80 &&
            static_cast<unsigned char>(s[i + 2]) == 0x99) {
            i += 2;  // U+2019, the typographic apostrophe
            continue;
        }
        if (c < 0x80 && std::isalnum(c))
            spaced.push_back(static_cast<char>(std::tolower(c)));
        else
            spaced.push_back(' ');
    }
    std::string out;
    out.reserve(spaced.size());
    for (char c : spaced) {
        if (c == ' ' && (out.empty() || out.back() == ' ')) continue;
        out.push_back(c);
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

static std::string trim(const std::string &s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

static std::string lower(const std::string &s) {
    return ascii_lower(s);
}

// Python's repr of a plain string, as the resolver's messages quote specs.
static std::string quoted(const std::string &s) {
    const bool has_single = s.find('\'') != std::string::npos, has_double = s.find('"') != std::string::npos;
    const char q = has_single && !has_double ? '"' : '\'';
    std::string out(1, q);
    for (char c : s) {
        if (c == q || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    out.push_back(q);
    return out;
}

// A leading seat key: optional whitespace, A or B (either case), optional whitespace, ':'.
static bool split_seat_key(const std::string &raw, char *seat, std::string *rest) {
    size_t i = 0;
    while (i < raw.size() && std::isspace(static_cast<unsigned char>(raw[i]))) ++i;
    if (i >= raw.size()) return false;
    const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(raw[i])));
    if (c != 'A' && c != 'B') return false;
    size_t j = i + 1;
    while (j < raw.size() && std::isspace(static_cast<unsigned char>(raw[j]))) ++j;
    if (j >= raw.size() || raw[j] != ':') return false;
    ++j;
    while (j < raw.size() && std::isspace(static_cast<unsigned char>(raw[j]))) ++j;
    *seat = c;
    *rest = raw.substr(j);
    return true;
}

// '#<digits>' or '<digits>'.
static bool parse_literal(const std::string &raw, int *out) {
    size_t i = !raw.empty() && raw[0] == '#' ? 1 : 0;
    if (i >= raw.size()) return false;
    long v = 0;
    for (size_t k = i; k < raw.size(); ++k) {
        if (!std::isdigit(static_cast<unsigned char>(raw[k]))) return false;
        v = v * 10 + (raw[k] - '0');
        if (v > 1000000) return false;
    }
    *out = static_cast<int>(v);
    return true;
}

// Verb -> the action categories it may resolve to (action_spec._VERB_CATS).
static const std::map<std::string, std::set<std::string>> &verb_cats() {
    static const std::set<std::string> other = {"OTHER_CHOICE", "SACRIFICE_PERMANENT", "RETURN_PERMANENT", "CHOOSE_X",
        "DISCARD", "CHOOSE_MODE", "CHOOSE_MANA_COLOR", "PAY_UNLESS", "NAME_CARD", "CHOOSE_TYPE", "KEEP_LEGEND",
        "ORDER_TRIGGERS", "CHOOSE_REPLACEMENT", "ATTACK_TARGET", "BLOCK_TARGET", "OPTIONAL_YESNO", "SYLVAN_CHOICE",
        "CHOOSE_CARD", "DONT_SHUFFLE", "KEEP_HAND", "EXILE_FROM_YARD"};
    static const std::set<std::string> mana = {"MANA_W", "MANA_U", "MANA_B", "MANA_R", "MANA_G", "MANA_C"};
    // tap:<card> also picks a source in the mana payment loop (--no-auto-mana), whose rows are
    // PAYING_COSTS.
    static const std::set<std::string> tap = {
        "MANA_W", "MANA_U", "MANA_B", "MANA_R", "MANA_G", "MANA_C", "PAYING_COSTS"};
    static const std::map<std::string, std::set<std::string>> table = {
        {"pass", {"PASS_PRIORITY"}},
        {"cast", {"CAST_SPELL"}},
        {"play", {"PLAY_LAND"}},
        {"land", {"PLAY_LAND"}},
        {"activate", {"ACTIVATE_ABILITY"}},
        {"target", {"SELECT_TARGET"}},
        {"attack", {"SELECT_ATTACKER", "CONFIRM_ATTACKERS"}},
        {"block", {"SELECT_BLOCKER", "CONFIRM_BLOCKERS"}},
        {"mana", mana},
        {"tap", tap},
        {"search", {"SEARCH_LIBRARY", "CHOOSE_CARD"}},
        {"top", {"TOP_LIBRARY"}},
        {"bottom", {"BOTTOM_DECK_CARD"}},
        {"dig", {"DIG_CHOICE"}},
        {"mulligan", {"MULLIGAN"}},
        {"keep", {"KEEP_HAND"}},
        {"pay", {"PAYING_COSTS"}},
        {"choice", other},
        {"shuffle", {"SHUFFLE"}},
        {"noshuffle", {"DONT_SHUFFLE"}},
        {"sb-in", {"SIDEBOARD_IN"}},
        {"sb-out", {"SIDEBOARD_OUT"}},
        {"sb-done", {"SIDEBOARD_DONE"}},
        {"companion", {"COMPANION"}},
        {"sacrifice", {"SACRIFICE_PERMANENT"}},
        {"return", {"RETURN_PERMANENT"}},
        {"x", {"CHOOSE_X"}},
        {"discard", {"DISCARD"}},
        {"mode", {"CHOOSE_MODE"}},
        {"color", {"CHOOSE_MANA_COLOR"}},
        {"name", {"NAME_CARD"}},
        {"exile", {"EXILE_FROM_YARD"}},
        {"desc", {}},
        {"concede", {}},
    };
    return table;
}

static bool parse_spec(const std::string &token, Intent *out, std::string *err) {
    std::string raw = trim(token);
    char seat = 0;
    std::string rest;
    if (split_seat_key(raw, &seat, &rest)) {
        out->seat = seat;
        raw = trim(rest);
        if (raw.empty()) {
            *err = "seat key with no action in spec " + quoted(token);
            return false;
        }
    }
    int literal = 0;
    if (parse_literal(raw, &literal)) {
        out->literal = literal;
        return true;
    }
    const size_t colon = raw.find(':');
    const std::string verb = lower(trim(raw.substr(0, colon)));
    std::string arg = colon == std::string::npos ? std::string() : trim(raw.substr(colon + 1));
    auto it = verb_cats().find(verb);
    if (it == verb_cats().end()) {
        *err = "unknown action verb " + quoted(verb) + " in spec " + quoted(raw);
        return false;
    }
    out->verb = verb;
    out->cats = it->second;
    const size_t at = arg.rfind('@');
    if (at != std::string::npos) {
        static const std::map<std::string, std::string> sides = {{"own", "own"}, {"me", "own"}, {"my", "own"},
            {"self", "own"}, {"you", "own"}, {"mine", "own"}, {"opp", "opp"}, {"them", "opp"}, {"their", "opp"},
            {"enemy", "opp"}, {"opponent", "opp"}};
        const std::string side = arg.substr(at + 1);
        arg = trim(arg.substr(0, at));
        auto s = sides.find(lower(trim(side)));
        if (s == sides.end()) {
            *err = "unknown @side " + quoted(side) + " in spec " + quoted(raw);
            return false;
        }
        out->side = s->second;
    }
    const std::string low = lower(arg);
    static const std::map<std::string, std::string> colors = {
        {"w", "MANA_W"}, {"u", "MANA_U"}, {"b", "MANA_B"}, {"r", "MANA_R"}, {"g", "MANA_G"}, {"c", "MANA_C"}};
    if (verb == "concede") {
        static const std::map<std::string, int> scopes = {{"", CONCEDE_GAME}, {"game", CONCEDE_GAME},
            {"g", CONCEDE_GAME}, {"match", CONCEDE_MATCH}, {"m", CONCEDE_MATCH}};
        auto s = scopes.find(low);
        if (s == scopes.end()) {
            *err = "unknown concede scope " + quoted(arg) + " in spec " + quoted(raw) +
                   " (use 'concede' / 'concede:game' / 'concede:match')";
            return false;
        }
        out->has_sentinel = true;
        out->sentinel = s->second;
    } else if (verb == "mulligan" || verb == "keep") {
        out->keyword = verb;
    } else if ((verb == "attack" || verb == "block") && (low == "done" || low == "confirm" || low.empty())) {
        out->keyword = "done";
    } else if (verb == "search" && (low == "fail" || low == "none" || low.empty())) {
        out->keyword = "fail";
    } else if ((verb == "mana" || verb == "tap") && colors.count(low)) {
        out->color = colors.at(low);
    } else if (!arg.empty()) {
        out->card = norm(arg);
    }
    return true;
}

// The intent's card / description text against one action: a card-bearing action by its name
// (equal on the exact pass, a substring on the loose one), a card-less one by its description.
static bool card_matches(const std::string &intent_card, const MenuAction &a, bool exact) {
    const std::string name = a.card && !a.card->empty() ? norm(*a.card) : std::string();
    if (!name.empty()) return exact ? name == intent_card : name.find(intent_card) != std::string::npos;
    return norm(a.description).find(intent_card) != std::string::npos;
}

static const char *controller_word(int controller) {
    return controller == 1 ? "own" : controller == 0 ? "opp" : "";
}

static bool matches(const Intent &intent, const MenuAction &a, bool exact) {
    if (!intent.cats.empty() && !intent.cats.count(a.category)) return false;
    if (!intent.side.empty() && intent.side != controller_word(a.controller)) return false;
    if (!intent.color.empty() && a.category != intent.color) return false;
    if (intent.keyword == "done") return a.category == "CONFIRM_ATTACKERS" || a.category == "CONFIRM_BLOCKERS";
    if (intent.keyword == "fail") return !a.card.has_value();
    if (intent.keyword == "keep" || intent.keyword == "mulligan")
        return norm(a.description).find(intent.keyword) != std::string::npos;
    if (intent.verb == "desc" && intent.card && !intent.card->empty())
        return norm(a.description).find(*intent.card) != std::string::npos;
    if (intent.card && !intent.card->empty()) return card_matches(*intent.card, a, exact);
    return true;
}

std::vector<MenuAction> resolver_menu(const DecisionView &v) {
    std::vector<MenuAction> out(v.n());
    for (size_t i = 0; i < v.n(); ++i) {
        MenuAction &a = out[i];
        a.index = i;
        a.category = v.category(i);
        const int cid = v.menu_card_id(i);
        if (cid >= 0) a.card = ascii_fold_card_name(v.cards().face_name(cid));
        a.controller = v.menu_controller(i);
        a.description = v.description(i);
    }
    return out;
}

ResolveResult resolve_spec(const std::string &token, const std::vector<MenuAction> &actions) {
    ResolveResult res;
    Intent intent;
    std::string err;
    if (!parse_spec(token, &intent, &err)) {
        res.kind = ResolveKind::PARSE_ERROR;
        res.reason = err;
        return res;
    }
    if (intent.has_sentinel) {
        res.kind = ResolveKind::OK;
        res.index = intent.sentinel;
        return res;
    }
    if (intent.literal >= 0) {
        if (static_cast<size_t>(intent.literal) < actions.size()) {
            res.kind = ResolveKind::OK;
            res.index = intent.literal;
        } else {
            res.kind = ResolveKind::BAD_INDEX;
            res.reason = "literal index " + std::to_string(intent.literal) + " out of range (0.." +
                         std::to_string(static_cast<long>(actions.size()) - 1) + ")";
        }
        return res;
    }
    for (bool exact : {true, false}) {
        std::vector<const MenuAction *> hits;
        for (const MenuAction &a : actions)
            if (matches(intent, a, exact)) hits.push_back(&a);
        if (hits.empty()) continue;
        // Interchangeable matches (same category, card and controller) collapse to the first.
        std::set<std::tuple<std::string, std::string, int>> keys;
        for (const MenuAction *a : hits) keys.emplace(a->category, norm(a->card ? *a->card : ""), a->controller);
        if (keys.size() == 1) {
            res.kind = ResolveKind::OK;
            res.index = static_cast<int>(hits[0]->index);
        } else {
            res.kind = ResolveKind::AMBIGUOUS;
            res.reason = quoted(token) + " is ambiguous (" + std::to_string(hits.size()) + " matches)";
        }
        return res;
    }
    res.kind = ResolveKind::NO_MATCH;
    res.reason = "no legal action matches " + quoted(token);
    return res;
}

char spec_seat(const std::string &token) {
    char seat = 0;
    std::string rest;
    return split_seat_key(token, &seat, &rest) ? seat : 0;
}

bool spec_is_pass(const std::string &token) {
    Intent intent;
    std::string err;
    return parse_spec(token, &intent, &err) && intent.verb == "pass";
}

bool spec_names_row(const std::string &token) {
    Intent intent;
    std::string err;
    return parse_spec(token, &intent, &err) && intent.verb != "pass" && intent.literal < 0 && !intent.has_sentinel;
}

std::vector<std::string> parse_spec_list(const std::string &csv) {
    std::vector<std::string> out;
    for (const std::string &piece : split(csv, ',')) {
        std::string t = trim(piece);
        if (!t.empty()) out.push_back(t);
    }
    return out;
}

std::string format_menu(const std::vector<MenuAction> &actions) {
    std::string out;
    for (const MenuAction &a : actions) {
        if (!out.empty()) out += "  ";
        out += "[" + std::to_string(a.index) + "] " + a.description;
    }
    return out;
}

PlayScript::PlayScript(std::vector<std::string> specs, Agent *player_a, Agent *player_b)
    : specs_(std::move(specs)), players_{player_a, player_b} {}

std::vector<std::string> PlayScript::unused_specs() const {
    return std::vector<std::string>(specs_.begin() + static_cast<long>(next_), specs_.end());
}

void PlayScript::auto_taken(char seat, const MenuAction &row) {
    if (next_ >= specs_.size()) return;
    const std::string &spec = specs_[next_];
    const char key = spec_seat(spec);
    if ((key && key != seat) || !spec_names_row(spec)) return;
    if (!resolve_spec(spec, {row}).ok()) return;
    ++next_;
    wait_ = 0;
}

int PlayScript::player_choice(const Decision &d) {
    Agent *player = players_[d.view.seat()];
    delegate_ = player;
    const int idx = player->decide(d);
    if (idx == AGENT_FAILED) {
        failure_ = player->failure();
        return idx;
    }
    resolved_.push_back(idx);
    return idx;
}

int PlayScript::decide(const Decision &d) {
    delegate_ = nullptr;
    if (next_ >= specs_.size()) return player_choice(d);
    const std::string &spec = specs_[next_];
    const char seat = spec_seat(spec);
    const char deciding = d.view.seat() == 0 ? 'A' : 'B';
    const std::vector<MenuAction> menu = resolver_menu(d.view);
    if (seat && seat != deciding) {
        const ResolveResult pass = resolve_spec("pass", menu);
        if (!pass.ok()) return player_choice(d);
        resolved_.push_back(pass.index);
        return pass.index;
    }
    ResolveResult r = resolve_spec(spec, menu);
    if (r.ok()) {
        ++next_;
        wait_ = 0;
        resolved_.push_back(r.index);
        return r.index;
    }
    if (seat && r.kind == ResolveKind::NO_MATCH && wait_ < MAX_WAIT) {
        const ResolveResult pass = resolve_spec("pass", menu);
        if (pass.ok()) {
            ++wait_;
            resolved_.push_back(pass.index);
            return pass.index;
        }
    }
    if (r.kind == ResolveKind::NO_MATCH && spec_is_pass(spec))
        r.reason +=
            " — this is a mandatory choice, not a priority window. The engine never asks a seat whose only "
            "legal action is a pass, so a pass written for such a window is not needed (drop it)";
    failure_ = "--play: " + r.reason + "\n    legal actions: " + format_menu(menu);
    return AGENT_FAILED;
}

}  // namespace agents
