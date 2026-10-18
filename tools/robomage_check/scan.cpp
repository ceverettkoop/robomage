#include "scan.h"

#include "json.h"
#include "util.h"

static bool is_word_char(char c);
static bool has_word(const std::string &line, const char *word);
static bool summary_line(const std::string &line, std::string *kind, bool *clean);

static bool is_word_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

// `word` in `line` with a word boundary on both sides (the regex \bword\b).
static bool has_word(const std::string &line, const char *word) {
    const size_t n = std::string(word).size();
    for (size_t at = line.find(word); at != std::string::npos; at = line.find(word, at + 1)) {
        const bool left = at == 0 || !is_word_char(line[at - 1]);
        const bool right = at + n >= line.size() || !is_word_char(line[at + n]);
        if (left && right) return true;
    }
    return false;
}

// A "SELFCHECK-<kind>: decisions=<n> … errors=<n>" line: its kind and whether errors=0.
static bool summary_line(const std::string &line, std::string *kind, bool *clean) {
    if (!rmcheck::starts_with(line, "SELFCHECK-")) return false;
    const size_t colon = line.find(": decisions=");
    if (colon == std::string::npos) return false;
    *kind = line.substr(10, colon - 10);
    if (*kind != "STATE" && *kind != "KNOWLEDGE") return false;
    const std::string t = rmcheck::trim(line);
    const size_t e = t.rfind(" errors=");
    if (e == std::string::npos) return false;
    const std::string n = t.substr(e + 8);
    if (n.empty() || n.find_first_not_of("0123456789") != std::string::npos) return false;
    *clean = n == "0";
    return true;
}

namespace rmcheck {

bool is_error_line(const std::string &line) {
    if (starts_with(line, "ERROR:") || starts_with(line, "FATAL:")) return true;
    if (contains(line, "Traceback (most recent call last)") || contains(line, "Segmentation fault") ||
        contains(line, "core dumped"))
        return true;
    const size_t a = line.find("Assertion ");
    if (a != std::string::npos && line.find(" failed", a + 9) != std::string::npos) return true;
    return has_word(line, "Aborted");
}

bool is_warning_line(const std::string &line) {
    return starts_with(line, "WARNING:") && !contains(line, "Unrecognized ability param");
}

void scan_transcript(const std::string &text, std::vector<ScanLine> *errors, std::vector<ScanLine> *warnings) {
    const std::vector<std::string> lines = split_lines(text);
    for (size_t i = 0; i < lines.size(); ++i) {
        if (is_error_line(lines[i]))
            errors->push_back({i + 1, lines[i]});
        else if (warnings && is_warning_line(lines[i]))
            warnings->push_back({i + 1, lines[i]});
    }
}

std::vector<std::string> selfcheck_failures(const std::string &text, const std::string &where, const std::string &label,
    int summaries, const std::vector<std::string> &kinds) {
    std::vector<std::string> fails;
    std::map<std::string, int> found;
    for (const std::string &k : kinds) found[k] = 0;
    const std::vector<std::string> lines = split_lines(text);
    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string &line = lines[i];
        const std::string at = label + ": " + where + ":" + std::to_string(i + 1) + ": " + line;
        if (is_error_line(line)) fails.push_back(at);
        std::string kind;
        bool clean = false;
        if (summary_line(line, &kind, &clean)) {
            if (found.count(kind)) ++found[kind];
            if (!clean) fails.push_back(at);
        }
    }
    if (summaries >= 0)
        for (const auto &f : found)
            if (f.second != summaries)
                fails.push_back(label + ": " + std::to_string(f.second) + " SELFCHECK-" + f.first +
                                " summary line(s) in " + where + ", expected " + std::to_string(summaries));
    if (fails.size() > 20) fails.resize(20);
    return fails;
}

std::string knowledge_summary(const std::string &text) {
    std::string last;
    for (const std::string &line : split_lines(text))
        if (starts_with(line, "SELFCHECK-KNOWLEDGE:")) last = trim(line);
    return last.empty() ? "no SELFCHECK-KNOWLEDGE summary" : last;
}

bool ShadowSignatures::load(const std::string &path, const std::string &scenario_dir, std::string *err) {
    sigs_.clear();
    std::string text;
    if (!read_file(path, &text)) {
        *err = path + ": cannot read";
        return false;
    }
    Json doc;
    if (!Json::parse(text, &doc, err)) {
        *err = path + ": " + *err;
        return false;
    }
    if (!doc.is_array() || doc.items().empty()) {
        *err = path + ": expected a non-empty list of signatures";
        return false;
    }
    static const std::set<std::string> keys = {
        "name", "kind", "direction", "owner", "pattern", "classification", "why", "scenario"};
    std::set<std::string> names;
    for (size_t i = 0; i < doc.items().size(); ++i) {
        const Json &sig = doc.items()[i];
        const Json *name = sig.get("name");
        const std::string where = path + ": signature " + std::to_string(i) + " (" +
                                  (name && name->is_string() ? name->as_string() : "?") + ")";
        if (!sig.is_object()) {
            *err = where + ": not an object";
            return false;
        }
        for (const auto &m : sig.members())
            if (!keys.count(m.first)) {
                *err = where + ": unknown key " + m.first;
                return false;
            }
        for (const std::string &k : keys)
            if (k != "owner" && !sig.has(k)) {
                *err = where + ": missing " + k;
                return false;
            }
        Signature s;
        s.name = sig.get("name")->as_string();
        if (!names.insert(s.name).second) {
            *err = where + ": duplicate name";
            return false;
        }
        const Json &kind = *sig.get("kind");
        if (kind.is_array()) {
            for (const Json &k : kind.items()) s.kinds.insert(k.as_string());
        } else {
            s.kinds.insert(kind.as_string());
        }
        for (const std::string &k : s.kinds)
            if (k != "top" && k != "hand" && k != "seen") s.kinds.clear();
        if (s.kinds.empty()) {
            *err = where + ": kind must be top, hand, seen or a list of them";
            return false;
        }
        s.direction = sig.get("direction")->as_string();
        if (s.direction != "old_more" && s.direction != "new_more" && s.direction != "conflict") {
            *err = where + ": direction must be old_more, new_more or conflict";
            return false;
        }
        if (const Json *owner = sig.get("owner")) {
            s.owner = owner->as_string();
            if (s.owner != "self" && s.owner != "opp") {
                *err = where + ": owner must be self or opp";
                return false;
            }
        }
        const std::string cls = sig.get("classification")->as_string();
        if (cls != "old-vector leak" && cls != "old-vector quirk") {
            *err = where + ": classification must be old-vector leak or old-vector quirk";
            return false;
        }
        s.scenario = sig.get("scenario")->as_string();
        if (!file_exists(scenario_dir + "/" + s.scenario + ".json")) {
            *err = where + ": no scenario " + s.scenario;
            return false;
        }
        const std::string pattern = sig.get("pattern")->as_string();
        if (first_bad_regex({pattern}) != std::string::npos) {
            *err = where + ": invalid pattern /" + pattern + "/";
            return false;
        }
        s.regex = std::regex(pattern);
        sigs_.push_back(std::move(s));
    }
    return true;
}

std::string ShadowSignatures::classify(const std::string &line) const {
    // ^SHADOW: (top|hand|seen) (old_more|new_more|conflict) viewer=[AB] owner=(self|opp)
    if (!starts_with(line, "SHADOW: ")) return "";
    const std::vector<std::string> f = split(line.substr(8), ' ');
    if (f.size() < 5) return "";
    const std::string &kind = f[0], &direction = f[1];
    if (kind != "top" && kind != "hand" && kind != "seen") return "";
    if (direction != "old_more" && direction != "new_more" && direction != "conflict") return "";
    if (f[2] != "viewer=A" && f[2] != "viewer=B") return "";
    if (f[3] != "owner=self" && f[3] != "owner=opp") return "";
    const std::string owner = f[3].substr(6);
    for (const Signature &s : sigs_)
        if (s.kinds.count(kind) && s.direction == direction && (s.owner.empty() || s.owner == owner) &&
            std::regex_search(line, s.regex))
            return s.name;
    return "";
}

std::vector<std::string> ShadowSignatures::log_failures(const std::string &text, const std::string &where,
    const std::string &label, int summaries, std::map<std::string, int> *counts) const {
    std::vector<std::string> fails;
    int unclassified = 0, found = 0;
    const std::vector<std::string> lines = split_lines(text);
    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string &line = lines[i];
        if (starts_with(line, "SHADOW-KNOWLEDGE:")) ++found;
        if (!starts_with(line, "SHADOW:")) continue;
        const std::string name = classify(line);
        if (name.empty()) {
            if (++unclassified <= 10)
                fails.push_back(
                    label + ": unclassified shadow mismatch " + where + ":" + std::to_string(i + 1) + ": " + line);
        } else if (counts) {
            ++(*counts)[name];
        }
    }
    if (unclassified > 10)
        fails.push_back(
            label + ": " + std::to_string(unclassified - 10) + " more unclassified shadow mismatch(es) in " + where);
    if (summaries >= 0 && found != summaries)
        fails.push_back(label + ": " + std::to_string(found) + " SHADOW-KNOWLEDGE: summary line(s) in " + where +
                        ", expected " + std::to_string(summaries));
    return fails;
}

}  // namespace rmcheck
