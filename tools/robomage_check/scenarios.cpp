#include "scenarios.h"

#include <fnmatch.h>

#include <cctype>
#include <map>
#include <set>

#include "frames.h"
#include "knowledge_lines.h"
#include "proc.h"
#include "scan.h"
#include "util.h"

using featurize::Json;

static const int DEFAULT_MAX_DECISIONS = 60;

// Scenario file → its malformed regex (prevalidate_scenarios; read-only once the runs start).
static std::map<std::string, std::string> bad_regex;
static const double TIMEOUT_S = 120;

static const std::set<std::string> &harness_keys();
static const std::set<std::string> &dump_keys();
static const std::set<std::string> &assert_keys();
static const std::set<std::string> &known_keys();
static std::vector<std::string> string_list(const Json *v);
static std::vector<std::string> regex_list(const Json &scn, const char *key);
static std::vector<std::string> all_regexes(const Json &scn);
static std::string flag_list(const Json &cards, bool strip_apostrophes);
static std::vector<std::string> engine_args(
    const Json &scn, const std::string &dump_path, const std::vector<std::string> &extra);
static bool is_scenario_error_line(const std::string &line);
static bool turn_banner(const std::string &line);
static bool game_result(const std::string &text, std::string *got);
static std::vector<std::string> check_output(const Json &scn, const std::string &text);
static std::vector<std::string> check_dump(
    const Json &scn, const std::vector<std::pair<char, std::vector<std::string>>> &frames);
static std::vector<std::string> same_lines(const std::vector<std::string> &patterns, const std::string &text);
static std::string play_engine(const rmcheck::Options &opt, const std::vector<std::string> &args, bool *timed_out);
static Json without(const Json &scn, const std::vector<std::string> &keys);
static std::vector<std::string> check_variants(const rmcheck::Options &opt, const Json &scn, const std::string &text);
static std::string command_text(const rmcheck::Options &opt, const std::vector<std::string> &args);

static const std::set<std::string> &harness_keys() {
    static const std::set<std::string> keys = {"name", "hand_a", "hand_b", "library_a", "library_b", "battlefield_a",
        "battlefield_b", "graveyard_a", "graveyard_b", "exile_a", "exile_b", "sideboard_a", "sideboard_b", "life_a",
        "life_b", "play", "actions", "seed", "max_decisions"};
    return keys;
}

static const std::set<std::string> &dump_keys() {
    static const std::set<std::string> keys = {
        "expect_dump", "expect_dump_not", "expect_dump_final", "expect_dump_final_not"};
    return keys;
}

// (Function-local statics with an initializer: built once, thread-safely, by the first caller.)
static const std::set<std::string> &assert_keys() {
    static const std::set<std::string> keys = [] {
        std::set<std::string> k = {"expect", "expect_not", "expect_order", "expect_count", "expect_menu",
            "expect_menu_not", "expect_result", "expect_unique", "expect_same"};
        k.insert(dump_keys().begin(), dump_keys().end());
        return k;
    }();
    return keys;
}

static const std::set<std::string> &known_keys() {
    static const std::set<std::string> keys = [] {
        std::set<std::string> k = harness_keys();
        k.insert(assert_keys().begin(), assert_keys().end());
        for (const char *key :
            {"why", "format", "player_a", "player_b", "deck_a", "deck_b", "offer_cancel", "engine_flags", "variants"})
            k.insert(key);
        return k;
    }();
    return keys;
}

// A JSON list's items as text (Python's str() of each); empty for a missing key.
static std::vector<std::string> string_list(const Json *v) {
    std::vector<std::string> out;
    if (!v) return out;
    if (v->is_array()) {
        for (const Json &item : v->items()) out.push_back(item.text());
    } else if (v->is_object()) {
        for (const auto &m : v->members()) out.push_back(m.first);
    }
    return out;
}

static std::vector<std::string> regex_list(const Json &scn, const char *key) {
    return string_list(scn.get(key));
}

// Every regex a scenario holds: its assertion keys' and its variants' expect_same.
static std::vector<std::string> all_regexes(const Json &scn) {
    std::vector<std::string> out;
    for (const std::string &key : assert_keys()) {
        if (key == "expect_result") continue;
        for (const std::string &rx : regex_list(scn, key.c_str())) out.push_back(rx);
    }
    if (const Json *variants = scn.get("variants"))
        for (const Json &v : variants->items())
            for (const std::string &rx : string_list(v.get("expect_same"))) out.push_back(rx);
    return out;
}

static std::string flag_list(const Json &cards, bool strip_apostrophes) {
    std::vector<std::string> parts;
    for (const Json &c : cards.items()) {
        std::string s = c.text();
        if (strip_apostrophes) {
            std::string t;
            for (size_t i = 0; i < s.size(); ++i) {
                if (s[i] == '\'') continue;
                if (s.compare(i, 3, "\xe2\x80\x99") == 0) {
                    i += 2;
                    continue;
                }
                t += s[i];
            }
            s = t;
        }
        parts.push_back(s);
    }
    return rmcheck::join(parts, ",");
}

// The engine arguments (after the binary) that play `scn` as the harness does.
static std::vector<std::string> engine_args(
    const Json &scn, const std::string &dump_path, const std::vector<std::string> &extra) {
    std::vector<std::string> a = {
        "--game-seed", scn.has("seed") ? scn.get("seed")->text() : "1", "--narrative", "--trace-decisions"};
    if (scn.has("format") && scn.get("format")->text() == "bo3") a.push_back("--bo3");
    a.push_back("--max-decisions");
    a.push_back(scn.has("max_decisions") ? scn.get("max_decisions")->text() : std::to_string(DEFAULT_MAX_DECISIONS));
    for (const char *seat : {"a", "b"}) {
        const std::string s = seat;
        const Json *hand = scn.get("hand_" + s);
        if (hand && hand->truthy()) {
            a.push_back("--hand-" + s);
            a.push_back(flag_list(*hand, false));
            const Json *lib = scn.get("library_" + s);
            if (lib && lib->truthy()) {
                a.push_back("--library-" + s);
                a.push_back(flag_list(*lib, false));
            }
        } else {
            a.push_back("--deck-" + s);
            a.push_back(scn.has("deck_" + s) ? scn.get("deck_" + s)->text() : "delver");
        }
        for (const char *zone : {"battlefield", "graveyard", "exile", "sideboard"}) {
            const Json *cards = scn.get(std::string(zone) + "_" + s);
            if (cards && cards->truthy()) {
                a.push_back("--" + std::string(zone) + "-" + s);
                a.push_back(flag_list(*cards, true));
            }
        }
        const Json *life = scn.get("life_" + s);
        if (life && !life->is_null()) {
            a.push_back("--life-" + s);
            a.push_back(life->text());
        }
    }
    const Json *play = scn.get("play");
    const Json *actions = scn.get("actions");
    if (play && play->truthy()) {
        std::vector<std::string> specs;
        for (const Json &p : play->items()) {
            const std::string t = rmcheck::trim(p.text());
            if (!t.empty()) specs.push_back(t);
        }
        a.push_back("--play");
        a.push_back(rmcheck::join(specs, ","));
    } else if (actions && actions->truthy()) {
        std::vector<std::string> ints;
        for (const Json &i : actions->items()) ints.push_back(i.text());
        a.push_back("--actions");
        a.push_back(rmcheck::join(ints, ","));
    } else {
        for (const char *s : {"--agent-a", "first", "--agent-b", "first"}) a.push_back(s);
    }
    const Json *cancel = scn.get("offer_cancel");
    if (cancel && cancel->truthy()) a.push_back("--offer-cancel");
    for (const std::string &f : string_list(scn.get("engine_flags"))) a.push_back(f);
    for (const std::string &f : extra) a.push_back(f);
    if (!dump_path.empty()) {
        a.push_back("--rmstate-dump");
        a.push_back(dump_path);
    }
    return a;
}

// Whether a line of a scenario run fails it: ERROR: / FATAL:, a crash (scan.h), a --play
// resolution error or SCRIPT INCOMPLETE.
static bool is_scenario_error_line(const std::string &line) {
    const std::string t = rmcheck::trim(line);
    if (rmcheck::starts_with(t, "ERROR:") || rmcheck::starts_with(t, "FATAL:")) return true;
    return rmcheck::is_error_line(line) || rmcheck::contains(line, "PlayResolveError") ||
           rmcheck::contains(line, "SCRIPT INCOMPLETE");
}

// ^-+ TURN \d+ (searched)
static bool turn_banner(const std::string &line) {
    if (line.empty() || line[0] != '-') return false;
    size_t i = 0;
    while (i < line.size() && line[i] == '-') ++i;
    if (line.compare(i, 6, " TURN ") != 0) return false;
    i += 6;
    const size_t d = i;
    while (i < line.size() && line[i] >= '0' && line[i] <= '9') ++i;
    return i > d && i < line.size() && line[i] == ' ';
}

// The first GAME_RESULT: \d+ (Player (A|B) wins|draw) in `text`: "A", "B" or "draw".
static bool game_result(const std::string &text, std::string *got) {
    for (size_t at = text.find("GAME_RESULT: "); at != std::string::npos; at = text.find("GAME_RESULT: ", at + 1)) {
        size_t i = at + 13;
        const size_t d = i;
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') ++i;
        if (i == d || i >= text.size() || text[i] != ' ') continue;
        ++i;
        if (text.compare(i, 13, "Player A wins") == 0) {
            *got = "A";
            return true;
        }
        if (text.compare(i, 13, "Player B wins") == 0) {
            *got = "B";
            return true;
        }
        if (text.compare(i, 4, "draw") == 0) {
            *got = "draw";
            return true;
        }
    }
    return false;
}

// The failures of one engine run's output against the scenario's assertions: an error line, then
// the narrative, menu and result keys.
static std::vector<std::string> check_output(const Json &scn, const std::string &text) {
    std::vector<std::string> fails;
    for (const std::string &line : rmcheck::split_lines(text))
        if (is_scenario_error_line(line)) {
            fails.push_back("error line: " + rmcheck::trim(line));
            break;
        }
    std::vector<std::string> narrative, actions;
    rmcheck::split_engine_output(text, &narrative, &actions);
    for (const std::string &rx : regex_list(scn, "expect")) {
        const std::regex re(rx);
        bool hit = false;
        for (const std::string &ln : narrative) hit = hit || std::regex_search(ln, re);
        if (!hit) fails.push_back("missing narrative line /" + rx + "/");
    }
    for (const std::string &rx : regex_list(scn, "expect_not")) {
        const std::regex re(rx);
        for (const std::string &ln : narrative)
            if (std::regex_search(ln, re)) {
                fails.push_back("forbidden narrative line /" + rx + "/: " + rmcheck::quoted(ln));
                break;
            }
    }
    size_t pos = 0;
    for (const std::string &rx : regex_list(scn, "expect_order")) {
        const std::regex re(rx);
        size_t k = pos;
        while (k < narrative.size() && !std::regex_search(narrative[k], re)) ++k;
        if (k >= narrative.size()) {
            fails.push_back(
                "out of order or missing narrative line /" + rx + "/ (searched from line " + std::to_string(pos) + ")");
            break;
        }
        pos = k + 1;
    }
    if (const Json *counts = scn.get("expect_count")) {
        for (const auto &m : counts->members()) {
            const std::regex re(m.first);
            int64_t got = 0;
            for (const std::string &ln : narrative) got += std::regex_search(ln, re) ? 1 : 0;
            if (got != m.second.as_int())
                fails.push_back("narrative line /" + m.first + "/ matched " + std::to_string(got) +
                                " time(s), expected " + m.second.text());
        }
    }
    for (const std::string &rx : regex_list(scn, "expect_menu")) {
        const std::regex re(rx);
        bool hit = false;
        for (const std::string &a : actions) hit = hit || std::regex_search(a, re);
        if (!hit) fails.push_back("no offered action matches /" + rx + "/");
    }
    for (const std::string &rx : regex_list(scn, "expect_menu_not")) {
        const std::regex re(rx);
        for (const std::string &a : actions)
            if (std::regex_search(a, re)) {
                fails.push_back("forbidden action offered /" + rx + "/: " + rmcheck::quoted(a));
                break;
            }
    }
    for (const std::string &rx : regex_list(scn, "expect_unique")) {
        const std::regex re(rx);
        std::map<std::string, std::string> seen;
        for (const std::string &ln : narrative) {
            std::smatch m;
            if (!std::regex_search(ln, m, re)) continue;
            std::vector<std::string> groups;
            for (size_t g = 1; g < m.size(); ++g)
                if (m[g].matched) groups.push_back(m[g].str());
            const std::string key = rmcheck::join(groups, " ");
            auto it = seen.find(key);
            if (it != seen.end()) {
                fails.push_back("/" + rx + "/ captured " + rmcheck::quoted(key) +
                                " twice: " + rmcheck::quoted(it->second) + " and " + rmcheck::quoted(ln));
                break;
            }
            seen.emplace(key, ln);
        }
    }
    if (const Json *want = scn.get("expect_result")) {
        if (!want->is_null()) {
            std::string got;
            const bool have = game_result(text, &got);
            if (!have || got != want->text())
                fails.push_back("expected result " + rmcheck::quoted(want->text()) + ", got " +
                                (have ? rmcheck::quoted(got) : std::string("None")));
        }
    }
    return fails;
}

// The failures of the seat frames' knowledge lines against the scenario's expect_dump* keys.
static std::vector<std::string> check_dump(
    const Json &scn, const std::vector<std::pair<char, std::vector<std::string>>> &frames) {
    if (frames.empty()) return {"no seat STATE frame in the --state-dump json output"};
    std::vector<std::string> fails, lines, final_lines;
    for (const auto &f : frames) lines.insert(lines.end(), f.second.begin(), f.second.end());
    for (char seat : {'A', 'B'}) {
        const std::vector<std::string> *last = nullptr;
        for (const auto &f : frames)
            if (f.first == seat) last = &f.second;
        if (last) final_lines.insert(final_lines.end(), last->begin(), last->end());
    }
    for (const std::string &rx : regex_list(scn, "expect_dump")) {
        const std::regex re(rx);
        bool hit = false;
        for (const std::string &ln : lines) hit = hit || std::regex_search(ln, re);
        if (!hit) fails.push_back("missing knowledge line /" + rx + "/");
    }
    for (const std::string &rx : regex_list(scn, "expect_dump_not")) {
        const std::regex re(rx);
        for (const std::string &ln : lines)
            if (std::regex_search(ln, re)) {
                fails.push_back("forbidden knowledge line /" + rx + "/: " + rmcheck::quoted(ln));
                break;
            }
    }
    for (const std::string &rx : regex_list(scn, "expect_dump_final")) {
        const std::regex re(rx);
        bool hit = false;
        for (const std::string &ln : final_lines) hit = hit || std::regex_search(ln, re);
        if (!hit) fails.push_back("missing knowledge line in a seat's last frame /" + rx + "/");
    }
    for (const std::string &rx : regex_list(scn, "expect_dump_final_not")) {
        const std::regex re(rx);
        for (const std::string &ln : final_lines)
            if (std::regex_search(ln, re)) {
                fails.push_back("forbidden knowledge line in a seat's last frame /" + rx + "/: " + rmcheck::quoted(ln));
                break;
            }
    }
    return fails;
}

// The narrative lines of one run that match any of the expect_same `patterns`.
static std::vector<std::string> same_lines(const std::vector<std::string> &patterns, const std::string &text) {
    std::vector<std::regex> rxs;
    for (const std::string &p : patterns) rxs.emplace_back(p);
    std::vector<std::string> narrative, actions, out;
    rmcheck::split_engine_output(text, &narrative, &actions);
    for (const std::string &ln : narrative)
        for (const std::regex &re : rxs)
            if (std::regex_search(ln, re)) {
                out.push_back(ln);
                break;
            }
    return out;
}

// The engine's stdout + stderr for `args` (an error line appended for a nonzero exit).
static std::string play_engine(const rmcheck::Options &opt, const std::vector<std::string> &args, bool *timed_out) {
    rmcheck::ProcSpec spec;
    spec.argv = {opt.binary};
    spec.argv.insert(spec.argv.end(), args.begin(), args.end());
    spec.cwd = opt.bin_dir;
    spec.timeout_s = TIMEOUT_S;
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    *timed_out = r.timed_out;
    std::string out = r.out + r.err;
    if (!r.ok() && !r.timed_out)
        out += "\nERROR: engine exited with code " + std::to_string(r.signal ? -r.signal : r.exit_code) + "\n";
    if (!r.started) out += "\nERROR: " + r.err + "\n";
    return out;
}

static Json without(const Json &scn, const std::vector<std::string> &keys) {
    Json out = scn;
    for (const std::string &k : keys) out.erase(k);
    return out;
}

// The failures of the scenario's variants: each variant run against the scenario's checks and
// its expect_same lines against the scenario's own run.
static std::vector<std::string> check_variants(const rmcheck::Options &opt, const Json &scn, const std::string &text) {
    std::vector<std::string> fails;
    const Json *variants = scn.get("variants");
    const Json base_checks = without(scn, {"expect_same", "variants"});
    for (size_t i = 0; i < variants->items().size(); ++i) {
        const Json &variant = variants->items()[i];
        const std::string tag = "variant " + std::to_string(i) + ": ";
        const std::vector<std::string> patterns =
            variant.has("expect_same") ? string_list(variant.get("expect_same")) : string_list(scn.get("expect_same"));
        const std::vector<std::string> base = same_lines(patterns, text);
        if (base.empty()) {
            fails.push_back(tag + "expect_same matched no narrative line");
            continue;
        }
        Json merged = without(scn, {"variants"});
        for (const auto &m : variant.members())
            if (m.first != "expect_same") merged.set(m.first, m.second);
        bool timed_out = false;
        const std::string out = play_engine(opt, engine_args(merged, "", {}), &timed_out);
        if (timed_out) {
            fails.push_back("variant " + std::to_string(i) + " timed out after 120s");
            continue;
        }
        for (const std::string &f : check_output(base_checks, out)) fails.push_back(tag + f);
        const std::vector<std::string> got = same_lines(patterns, out);
        if (got != base) {
            size_t k = 0;
            while (k < base.size() && k < got.size() && base[k] == got[k]) ++k;
            fails.push_back(tag + "expect_same line " + std::to_string(k) +
                            " differs: " + (k < base.size() ? rmcheck::quoted(base[k]) : "None") + " vs " +
                            (k < got.size() ? rmcheck::quoted(got[k]) : "None"));
        }
    }
    return fails;
}

static std::string command_text(const rmcheck::Options &opt, const std::vector<std::string> &args) {
    std::string out = "(cd " + opt.bin_dir + " && " + opt.binary;
    for (const std::string &a : args) {
        bool plain = !a.empty();
        for (char c : a)
            plain = plain && (std::isalnum(static_cast<unsigned char>(c)) ||
                                 std::string("-_./:=,@").find(c) != std::string::npos);
        out += " " + (plain ? a : rmcheck::quoted(a));
    }
    return out + ")";
}

namespace rmcheck {

std::vector<std::string> scenario_paths(const Options &opt) {
    std::vector<std::string> out;
    for (const std::string &name : list_dir(opt.scenario_dir(), ".json")) {
        const std::string stem = name.substr(0, name.size() - 5);
        bool keep = opt.scenario_patterns.empty();
        for (const std::string &p : opt.scenario_patterns)
            keep = keep || fnmatch(p.c_str(), stem.c_str(), 0) == 0 || fnmatch(p.c_str(), name.c_str(), 0) == 0;
        if (keep) out.push_back(path_join(opt.scenario_dir(), name));
    }
    return out;
}

void prevalidate_scenarios(const std::vector<std::string> &paths) {
    for (const std::string &path : paths) {
        std::string text, err;
        Json scn;
        if (!read_file(path, &text) || !Json::parse(text, &scn, &err) || !scn.is_object()) continue;
        const std::vector<std::string> rxs = all_regexes(scn);
        const size_t bad = first_bad_regex(rxs);
        if (bad != std::string::npos) bad_regex[path] = rxs[bad];
    }
}

bool load_scenario(const std::string &path, Scenario *out, std::string *err) {
    out->path = path;
    out->name = strip_ext(basename_of(path));
    std::string text;
    if (!read_file(path, &text)) {
        *err = "cannot read " + path;
        return false;
    }
    if (!Json::parse(text, &out->doc, err)) return false;
    const Json &scn = out->doc;
    if (!scn.is_object()) {
        *err = "not a JSON object";
        return false;
    }
    std::vector<std::string> unknown;
    bool any_assert = false;
    for (const auto &m : scn.members()) {
        if (!known_keys().count(m.first)) unknown.push_back(m.first);
        if (assert_keys().count(m.first)) any_assert = true;
    }
    if (!unknown.empty()) {
        *err = "unknown key(s) " + join(unknown, ", ");
        return false;
    }
    if (!scn.has("why") || !scn.get("why")->truthy()) {
        *err = "missing 'why'";
        return false;
    }
    const std::string format = scn.has("format") ? scn.get("format")->text() : "bo1";
    if (format != "bo1" && format != "bo3") {
        *err = "format must be bo1 or bo3, got " + quoted(format);
        return false;
    }
    auto bad = bad_regex.find(path);
    if (bad != bad_regex.end()) {
        *err = "invalid regex /" + bad->second + "/";
        return false;
    }
    if (!any_assert) {
        *err =
            "no assertion (expect / expect_not / expect_order / expect_count / expect_menu / expect_menu_not / "
            "expect_result)";
        return false;
    }
    for (const std::string &rx : regex_list(scn, "expect_unique"))
        if (std::regex(rx).mark_count() < 1) {
            *err = "expect_unique /" + rx + "/ needs a capture group";
            return false;
        }
    const Json *variants = scn.get("variants");
    if (scn.has("expect_same") && scn.get("expect_same")->truthy() && !(variants && variants->truthy())) {
        *err = "expect_same needs variants";
        return false;
    }
    if (variants) {
        for (const Json &v : variants->items()) {
            for (const auto &m : v.members())
                if (m.first != "expect_same" && (m.first == "name" || !harness_keys().count(m.first))) {
                    *err = "variant overrides non-harness key(s) " + m.first;
                    return false;
                }
            const Json *own = v.get("expect_same");
            const bool has_same = own ? own->truthy() : (scn.has("expect_same") && scn.get("expect_same")->truthy());
            if (!has_same) {
                *err = "a variant needs an expect_same (its own or the scenario's)";
                return false;
            }
        }
    }
    return true;
}

bool wants_dump(const Json &scn) {
    for (const std::string &k : dump_keys())
        if (scn.has(k)) return true;
    return false;
}

void split_engine_output(
    const std::string &text, std::vector<std::string> *narrative, std::vector<std::string> *actions) {
    std::vector<std::string> all;
    for (const std::string &line : split_lines(text)) {
        if (starts_with(line, "| ")) {
            // ^\|   \d+: (.*)$
            if (starts_with(line, "|   ")) {
                size_t i = 4;
                const size_t d = i;
                while (i < line.size() && line[i] >= '0' && line[i] <= '9') ++i;
                if (i > d && line.compare(i, 2, ": ") == 0) actions->push_back(line.substr(i + 2));
            }
        } else if (!trim(line).empty()) {
            all.push_back(line);
        }
    }
    size_t first = 0;
    while (first < all.size() && !turn_banner(all[first])) ++first;
    if (first == all.size()) first = 0;
    narrative->assign(all.begin() + static_cast<long>(first), all.end());
}

ScenarioRun run_scenario(const Options &opt, const std::string &path, const std::vector<std::string> &extra_flags) {
    ScenarioRun run;
    Scenario scn;
    std::string err;
    run.name = strip_ext(basename_of(path));
    if (!load_scenario(path, &scn, &err)) {
        run.fails.push_back("bad scenario file: " + err);
        return run;
    }
    std::string dump_dir, dump_path;
    if (wants_dump(scn.doc)) {
        dump_dir = make_temp_dir(opt.out("scenarios"), "dump_" + run.name + "_");
        dump_path = path_join(dump_dir, "frames.rms");
    }
    const std::vector<std::string> args = engine_args(scn.doc, dump_path, extra_flags);
    run.reproduce = command_text(opt, args);
    bool timed_out = false;
    run.output = play_engine(opt, args, &timed_out);
    if (timed_out) {
        run.fails.push_back("timed out after 120s");
        if (!dump_dir.empty()) remove_tree(dump_dir);
        return run;
    }
    run.fails = check_output(scn.doc, run.output);
    if (!dump_path.empty()) {
        std::vector<std::string> payloads;
        if (file_exists(dump_path)) {
            if (!read_records(dump_path, &payloads, &err) || !knowledge_frames(payloads, &run.frames, &err))
                run.fails.push_back("the --rmstate-dump does not decode: " + err);
        }
        for (const std::string &f : check_dump(scn.doc, run.frames)) run.fails.push_back(f);
        remove_tree(dump_dir);
    }
    if (const Json *variants = scn.doc.get("variants"))
        if (variants->truthy())
            for (const std::string &f : check_variants(opt, scn.doc, run.output)) run.fails.push_back(f);
    return run;
}

}  // namespace rmcheck
