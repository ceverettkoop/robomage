#include "scan.h"

#include <map>

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

}  // namespace rmcheck
