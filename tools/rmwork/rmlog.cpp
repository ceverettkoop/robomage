#include "rmlog.h"

#include <cstdlib>

static std::vector<std::string> split_lines(const std::string &text);
static bool header_field(
    const std::vector<std::string> &lines, size_t *at, const char *key, std::string *value, std::string *err);
static bool parse_int(const std::string &s, long long *out);

static std::vector<std::string> split_lines(const std::string &text) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start < text.size()) {
        size_t nl = text.find('\n', start);
        if (nl == std::string::npos) nl = text.size();
        std::string line = text.substr(start, nl - start);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out.push_back(line);
        start = nl + 1;
    }
    return out;
}

// Line `*at` is "<key>" or "<key> <value>"; advances past it.
static bool header_field(
    const std::vector<std::string> &lines, size_t *at, const char *key, std::string *value, std::string *err) {
    const std::string k = key;
    if (*at >= lines.size()) {
        *err = "the header ends before " + k;
        return false;
    }
    const std::string &line = lines[*at];
    if (line == k) {
        value->clear();
    } else if (line.compare(0, k.size() + 1, k + " ") == 0) {
        *value = line.substr(k.size() + 1);
    } else {
        *err = "line " + std::to_string(*at + 1) + ": expected " + k + ", found \"" + line.substr(0, 60) + "\"";
        return false;
    }
    ++*at;
    return true;
}

static bool parse_int(const std::string &s, long long *out) {
    if (s.empty()) return false;
    char *end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (!end || *end != '\0') return false;
    *out = v;
    return true;
}

namespace rmwork {

bool RmlogInfo::has_flag(const std::string &token) const {
    size_t start = 0;
    while (start <= flags.size()) {
        size_t sp = flags.find(' ', start);
        if (sp == std::string::npos) sp = flags.size();
        const std::string t = flags.substr(start, sp - start);
        if (t == token || t.compare(0, token.size() + 1, token + "=") == 0) return true;
        start = sp + 1;
    }
    return false;
}

bool parse_rmlog(const std::string &text, RmlogInfo *out, std::string *err) {
    *out = RmlogInfo();
    const std::vector<std::string> lines = split_lines(text);
    if (lines.empty() || lines[0] != "RMLOG v3") {
        *err = lines.empty() ? "the file is empty"
                             : "not an RMLOG v3 decision log (first line \"" + lines[0].substr(0, 40) + "\")";
        return false;
    }
    size_t at = 1;
    std::string seed;
    if (!header_field(lines, &at, "SEED", &seed, err)) return false;
    long long s = 0;
    if (!parse_int(seed, &s) || s < 0) {
        *err = "SEED is not a number: \"" + seed + "\"";
        return false;
    }
    out->seed = static_cast<uint64_t>(s);
    if (!header_field(lines, &at, "FLAGS", &out->flags, err)) return false;
    if (out->flags == "-") out->flags.clear();
    for (std::string *deck : {&out->deck_a, &out->deck_b}) {
        if (!header_field(lines, &at, deck == &out->deck_a ? "DECK_A" : "DECK_B", deck, err)) return false;
        while (at < lines.size() && lines[at] != "END_DECK") ++at;
        if (at >= lines.size()) {
            *err = "a decklist has no END_DECK";
            return false;
        }
        ++at;
    }
    std::string none;
    if (!header_field(lines, &at, "DECISIONS", &none, err)) return false;
    for (; at < lines.size(); ++at) {
        const std::string &line = lines[at];
        if (line.empty()) continue;
        if (line == "QUIT") {
            out->quit = true;
            break;
        }
        long long v = 0;
        if (!parse_int(line, &v) || v < -3 || v > 1000000) {
            *err = "line " + std::to_string(at + 1) + ": \"" + line.substr(0, 40) + "\" is not a reply";
            return false;
        }
        out->replies.push_back(static_cast<int>(v));
    }
    return true;
}

}  // namespace rmwork
