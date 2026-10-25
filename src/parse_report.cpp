#include "parse_report.h"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

#include "card_db.h"
#include "card_vocab.h"
#include "classes/deck.h"
#include "components/carddata.h"
#include "components/token.h"
#include "featurize/json.h"
#include "parse.h"
#include "queries/filters.h"
#include "queries/tokens.h"
#include "str_util.h"
#include "svar_eval.h"

std::vector<std::string> parse_report_args;

namespace {

// One card the report names: the name it was asked by and the script that name resolves to.
struct ReportCard {
        std::string query;
        std::string script;  // "" when the name has no script
};

}  // namespace

static bool expand_report_arg(const std::string &arg, std::vector<std::string> &names);
static std::vector<std::string> file_card_names(const std::string &path, bool &ok);
static std::vector<ReportCard> report_cards(const std::vector<std::string> &names);
static std::set<std::string> card_tokens(const CardData &card, const std::string &script_text);
static void add_ability_tokens(const std::vector<const AbilityDef *> &defs, std::set<std::string> &out);
static void scan_token_scripts(const std::string &script_text, std::set<std::string> &out);
static void filter_findings(const std::string &script_text, std::vector<ParseFinding> &findings);
static void check_filter(
    const std::string &spec, const std::string &where, const std::string &line, std::vector<ParseFinding> &findings);
static bool filter_token_read_elsewhere(const std::string &token);
static void svar_expr_findings(const std::string &script_text, std::vector<ParseFinding> &findings);
static bool svar_expr_read_at_parse(const std::string &expr);
static std::string card_layout(const CardData &card);
static bool token_script_present(const std::string &stem);
static std::string file_text(const std::string &path);
static std::string relative_script_path(const std::string &path);
static std::string finding_json(const ParseFinding &f);
static std::string missing_card_json(const std::string &query);

bool parse_report_requested() {
    return !parse_report_args.empty();
}

int run_parse_report() {
    std::vector<std::string> names;
    for (const std::string &arg : parse_report_args)
        if (!expand_report_arg(arg, names)) return 2;
    size_t with_findings = 0, without_script = 0;
    const std::vector<ReportCard> cards = report_cards(names);
    for (const ReportCard &rc : cards) {
        if (rc.script.empty()) {
            ++without_script;
            std::printf("%s\n", missing_card_json(rc.query).c_str());
            continue;
        }
        CardData card;
        std::vector<ParseFinding> findings;
        if (!parse_card_findings(rc.script, card, findings)) {
            ++without_script;
            std::printf("%s\n", missing_card_json(rc.query).c_str());
            continue;
        }
        const std::string script_text = file_text(rc.script);
        filter_findings(script_text, findings);
        svar_expr_findings(script_text, findings);
        std::string tokens_json;
        for (const std::string &stem : card_tokens(card, script_text)) {
            const bool present = token_script_present(stem);
            if (!tokens_json.empty()) tokens_json += ",";
            tokens_json +=
                "{\"script\":" + featurize::json_quote(stem) + ",\"present\":" + (present ? "true" : "false") + "}";
            if (!present) {
                findings.push_back({"token_script", stem, "the token script is not on disk", ""});
                continue;
            }
            Token tok;
            std::vector<ParseFinding> token_findings;
            parse_token_findings(stem, tok, token_findings);
            for (ParseFinding &f : token_findings) {
                f.line = "token:" + stem + " " + f.line;
                findings.push_back(f);
            }
        }
        std::string faces_json = featurize::json_quote(card.name);
        std::string vocab_json = std::to_string(card_name_to_index(card.name));
        if (card.backside) {
            faces_json += "," + featurize::json_quote(card.backside->name);
            vocab_json += "," + std::to_string(card_name_to_index(card.backside->name));
        }
        std::string findings_json;
        for (const ParseFinding &f : findings) findings_json += (findings_json.empty() ? "" : ",") + finding_json(f);
        if (!findings.empty()) ++with_findings;
        std::printf(
            "{\"query\":%s,\"card\":%s,\"script\":%s,\"faces\":[%s],\"layout\":%s,\"vocab\":[%s],"
            "\"tokens\":[%s],\"clean\":%s,\"findings\":[%s]}\n",
            featurize::json_quote(rc.query).c_str(), featurize::json_quote(card.name).c_str(),
            featurize::json_quote(relative_script_path(rc.script)).c_str(), faces_json.c_str(),
            featurize::json_quote(card_layout(card)).c_str(), vocab_json.c_str(), tokens_json.c_str(),
            findings.empty() ? "true" : "false", findings_json.c_str());
    }
    std::fprintf(stderr, "parse-report: %zu card(s), %zu with findings, %zu without a script\n", cards.size(),
        with_findings, without_script);
    return with_findings == 0 && without_script == 0 ? 0 : 1;
}

// Appends the card names `arg` stands for; false (after an ERROR line) for an unreadable @FILE.
static bool expand_report_arg(const std::string &arg, std::vector<std::string> &names) {
    if (arg == "@vocab") {
        for (int i = 0; i < CARD_VOCAB_SIZE; i++) names.push_back(card_vocab_entries[i].name);
        return true;
    }
    if (arg.size() > 1 && arg[0] == '@') {
        bool ok = false;
        const std::vector<std::string> from_file = file_card_names(arg.substr(1), ok);
        if (!ok) {
            std::fprintf(stderr, "parse-report: ERROR cannot read %s\n", arg.substr(1).c_str());
            return false;
        }
        names.insert(names.end(), from_file.begin(), from_file.end());
        return true;
    }
    names.push_back(arg);
    return true;
}

// The card names of the file at `path`: a .dk deck's main deck and sideboard, else one name per
// non-empty line.
static std::vector<std::string> file_card_names(const std::string &path, bool &ok) {
    std::vector<std::string> names;
    std::ifstream f(path);
    ok = f.is_open();
    if (!ok) return names;
    if (path.size() > 3 && path.compare(path.size() - 3, 3, ".dk") == 0) {
        const Deck deck(path);
        for (const auto *section : {&deck.main_deck, &deck.sideboard})
            for (const auto &entry : *section) names.push_back(entry.second);
        return names;
    }
    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (!line.empty()) names.push_back(line);
    }
    return names;
}

// The names in order, each script once (a card asked by both face names, or twice, is reported
// under its first name).
static std::vector<ReportCard> report_cards(const std::vector<std::string> &names) {
    std::vector<ReportCard> out;
    std::set<std::string> scripts;
    for (const std::string &name : names) {
        const std::string script = card_script_path(name);
        if (!script.empty() && !scripts.insert(script).second) continue;
        out.push_back({name, script});
    }
    return out;
}

// The token scripts `card` can make: its parsed abilities' (queries/tokens.h) and every
// TokenScript$ its script text names, so a token of an ability the parser dropped is listed too.
static std::set<std::string> card_tokens(const CardData &card, const std::string &script_text) {
    std::set<std::string> out;
    const CardData *faces[] = {&card, card.backside.get()};
    for (const CardData *face : faces) {
        if (!face) continue;
        add_ability_tokens(face->abilities, out);
        add_ability_tokens(face->gift_abilities, out);
        add_ability_tokens(face->saga_chapters, out);
        add_ability_tokens(face->opening_hand_abilities, out);
    }
    keyword_token_scripts(card, out);
    scan_token_scripts(script_text, out);
    return out;
}

static void add_ability_tokens(const std::vector<const AbilityDef *> &defs, std::set<std::string> &out) {
    for (const AbilityDef *def : defs)
        if (def) ability_token_scripts(*def, out);
}

// Every stem a "TokenScript$ <stem>[,<stem>…]" param of the script text names.
static void scan_token_scripts(const std::string &script_text, std::set<std::string> &out) {
    static const std::string key = "TokenScript$";
    for (size_t at = script_text.find(key); at != std::string::npos; at = script_text.find(key, at + key.size())) {
        if (at > 0 && std::isalnum(static_cast<unsigned char>(script_text[at - 1]))) continue;  // LockTokenScript$
        const size_t start = at + key.size();
        const size_t end = script_text.find_first_of("|\n", start);
        for (const std::string &stem : split(script_text.substr(start, end - start), ','))
            if (!trim(stem).empty()) out.insert(trim(stem));
    }
}

// The filter specs of a script's A:, T:, S:, R:, SVar: and K:Enchant lines — the params that name
// objects (ValidTgts$, ValidCard$, ChangeType$, Affected$, ...), the cost tokens' "<N/filter>"
// arguments and the Count$Valid… expressions — checked against the filter matcher
// (filter_unknown_qualifiers): every token it has no reading for is a "filter" finding.
static void filter_findings(const std::string &script_text, std::vector<ParseFinding> &findings) {
    static const char *const kFilterKeys[] = {"ValidTgts", "ValidCard", "ValidCards", "ChangeType", "Affected",
        "SacValid", "DiscardValid", "ValidSource", "ValidTarget", "IsPresent", "ConditionPresent", "ValidBlocker",
        "ValidAttacker", "ValidCreature", "RestrictValid", "ValidLKI", "ValidDefined", "TgtValid", "ChangeValid",
        "Valid"};
    static const char *const kCountPrefixes[] = {"Count$ValidGraveyard ", "Count$ValidLibrary ", "Count$ValidHand ",
        "Count$ValidExile ", "Count$ValidStack ", "Count$Valid ", "Remembered$Valid "};
    static const char *const kCostTokens[] = {
        "Sac<", "Discard<", "ExileFromGrave<", "ExileFromHand<", "Return<", "Exile<", "tapXType<"};
    for (const std::string &raw : split(script_text, '\n')) {
        const std::string line = trim(raw);
        std::string body;
        if (line.rfind("K:Enchant:", 0) == 0) {  // K:Enchant:<filter>[:<description>]
            const std::string arg = line.substr(10);
            check_filter(arg.substr(0, arg.find(':')), "Enchant", line, findings);
            continue;
        }
        if (line.rfind("A:", 0) == 0 || line.rfind("T:", 0) == 0 || line.rfind("S:", 0) == 0 ||
            line.rfind("R:", 0) == 0)
            body = line.substr(2);
        else if (line.rfind("SVar:", 0) == 0 && line.find(':', 5) != std::string::npos)
            body = line.substr(line.find(':', 5) + 1);
        else
            continue;
        for (const std::string &param : split(body, '|')) {
            const size_t dollar = param.find('$');
            if (dollar == std::string::npos) continue;
            const std::string key = trim(param.substr(0, dollar));
            const std::string value = trim(param.substr(dollar + 1));
            for (const char *k : kFilterKeys)
                if (key == k) check_filter(value, key + "$", line, findings);
            // A ChooseCard's Choices$ is a card filter (a Charm's names its modes' SVars).
            if (key == "Choices" && body.find("$ ChooseCard") != std::string::npos)
                check_filter(value, key + "$", line, findings);
            for (const char *prefix : kCountPrefixes) {
                const size_t at = param.find(prefix);
                if (at == std::string::npos) continue;
                const std::string rest = param.substr(at + std::strlen(prefix));
                check_filter(trim(rest.substr(0, rest.find_first_of("$/ "))), trim(prefix), line, findings);
                break;
            }
            for (const char *tok : kCostTokens)
                for (size_t at = value.find(tok); at != std::string::npos; at = value.find(tok, at + 1)) {
                    const size_t slash = value.find('/', at);
                    const size_t close = value.find('>', at);
                    if (slash == std::string::npos || close == std::string::npos || slash > close) continue;
                    const std::string arg = value.substr(slash + 1, close - slash - 1);
                    const std::string spec = arg.substr(0, arg.find('/'));
                    if (spec != "CARDNAME" && spec != "Hand" && spec != "Card") check_filter(spec, tok, line, findings);
                }
        }
    }
}

// Records each token of `spec` the filter matcher has no reading for; a targeting spec's player
// alternatives (Player, Opponent, You, Any) are not object filters and are skipped.
static void check_filter(
    const std::string &spec, const std::string &where, const std::string &line, std::vector<ParseFinding> &findings) {
    std::string objects;
    for (const std::string &alt : split(spec, ',', /*skip_empty=*/true)) {
        const std::string head = alt.substr(0, alt.find_first_of(".+"));
        if (head == "Player" || head == "Opponent" || head == "You" || head == "Any") continue;
        objects += (objects.empty() ? "" : ",") + alt;
    }
    if (objects.empty()) return;
    for (const std::string &token : filter_unknown_qualifiers(objects))
        if (!filter_token_read_elsewhere(token)) findings.push_back({"filter", token, where + " " + spec, line});
}

// Filter tokens the shared matcher has no reading for that the engine reads where the filter is
// used, in the forms its cards use them: the attachment statics' Affected$ (EquippedBy, EnchantedBy,
// state_manager_statics.cpp), targeting (Emblem, YouDontCtrl, targeting.cpp), the cards exiled with
// a source (ExiledWithSource), the sameName / Remembered / TargetedCard / NotDefinedTargeted /
// sharesNameWith movers (effect_change_zone*.cpp), chosen cards and types (effect_choose_card.cpp),
// the player-relative counts (RememberedPlayerCtrl, TargetedPlayerCtrl, svar_eval.cpp), cast history
// (!wasCast, wasCastFromYourHandByYou), kicker-linked triggers ("kicked N"), delayed-trigger memory
// (IsTriggerRemembered), Animate Dead's StrictlySelf, Eldrazi Temple's Activated mana restriction and
// Sylvan Library's DrawnThisTurn (its own SylvanLibrary resolution, parse_one_trigger).
// The same token in another param is a gap this list hides; the parse report does not prove a card.
static bool filter_token_read_elsewhere(const std::string &token) {
    static const char *const kTokens[] = {"EquippedBy", "EnchantedBy", "Emblem", "YouDontCtrl", "ExiledWithSource",
        "sameName", "Remembered", "TargetedCard", "NotDefinedTargeted", "sharesNameWith Targeted", "ChosenCard",
        "ChosenType", "RememberedPlayerCtrl", "TargetedPlayerCtrl", "!wasCast", "wasCastFromYourHandByYou",
        "IsTriggerRemembered", "StrictlySelf", "Activated", "DrawnThisTurn"};
    for (const char *t : kTokens)
        if (token == t) return true;
    return token.rfind("kicked", 0) == 0;
}

// The count expressions of a script's SVar: lines — a body that is no ability, trigger, static or
// replacement and holds a '$' (an AI hint SVar, AI… / NeedsToPlay…, is skipped) — checked against
// the evaluator (svar_expr_recognized) and the forms the parser resolves itself: each one neither
// reads is an "svar_expr" finding (it would evaluate to 0, an error, at runtime).
static void svar_expr_findings(const std::string &script_text, std::vector<ParseFinding> &findings) {
    for (const std::string &raw : split(script_text, '\n')) {
        const std::string line = trim(raw);
        if (line.rfind("SVar:", 0) != 0) continue;
        const size_t colon = line.find(':', 5);
        if (colon == std::string::npos) continue;
        const std::string name = line.substr(5, colon - 5);
        const std::string body = trim(line.substr(colon + 1));
        if (name.rfind("AI", 0) == 0 || name.rfind("NeedsToPlay", 0) == 0) continue;
        if (body.find('$') == std::string::npos || body.rfind("DB$", 0) == 0 || body.rfind("AB$", 0) == 0 ||
            body.rfind("SP$", 0) == 0 || body.find("Mode$") != std::string::npos ||
            body.find("Event$") != std::string::npos)
            continue;
        if (svar_expr_recognized(body) || svar_expr_read_at_parse(body)) continue;
        // The etbCounter keyword counts its delve-exiled cards itself (kw_etb_counter: Murktide Regent).
        if (body.find("ExiledWithSource") != std::string::npos &&
            script_text.find("K:etbCounter:") != std::string::npos &&
            script_text.find(":" + name, script_text.find("K:etbCounter:")) != std::string::npos)
            continue;
        findings.push_back({"svar_expr", body.substr(0, body.find(' ')), name + " = " + body, line});
    }
}

// The SVar forms the parser resolves into its own fields rather than handing to evaluate_svar:
// SVar$<name> references and their sums (resolve_additive_svar), Count$Compare and Count$Delirium
// (resolve_conditional_amount / delirium_scale_pos), the combat-damage amount
// (TriggerCount$DamageAmount), Surgical Extraction's TargetedController$CardsIn<zone>, Mox
// Diamond's Remembered$Amount and Green Sun's Zenith's $GreatestCardManaCost search bound.
static bool svar_expr_read_at_parse(const std::string &expr) {
    for (const char *p : {"SVar$", "Count$Compare ", "Count$Delirium.", "TargetedController$CardsIn"})
        if (expr.rfind(p, 0) == 0) return true;
    if (expr == "TriggerCount$DamageAmount" || expr == "Remembered$Amount") return true;
    const std::string greatest = "$GreatestCardManaCost";
    return expr.rfind("Count$ValidLibrary ", 0) == 0 && expr.size() > greatest.size() &&
           expr.compare(expr.size() - greatest.size(), greatest.size(), greatest) == 0;
}

static std::string card_layout(const CardData &card) {
    if (card.is_split) return "split";
    if (card.is_modal_dfc) return "modal";
    if (card.backside) return "transform";
    return "normal";
}

static bool token_script_present(const std::string &stem) {
    std::ifstream f(RESOURCE_DIR + "/tokenscripts/" + stem + ".txt");
    return f.is_open();
}

static std::string file_text(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream text;
    if (f.is_open()) text << f.rdbuf();
    return text.str();
}

// A script path relative to resources/ ("cardsfolder/b/brainstorm.txt").
static std::string relative_script_path(const std::string &path) {
    const std::string prefix = RESOURCE_DIR + "/";
    return path.rfind(prefix, 0) == 0 ? path.substr(prefix.size()) : path;
}

static std::string finding_json(const ParseFinding &f) {
    return "{\"kind\":" + featurize::json_quote(f.kind) + ",\"name\":" + featurize::json_quote(f.name) +
           ",\"detail\":" + featurize::json_quote(f.detail) + ",\"line\":" + featurize::json_quote(f.line) + "}";
}

static std::string missing_card_json(const std::string &query) {
    return "{\"query\":" + featurize::json_quote(query) + ",\"card\":null,\"error\":\"no script\",\"clean\":false}";
}
