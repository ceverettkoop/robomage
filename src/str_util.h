#ifndef STR_UTIL_H
#define STR_UTIL_H

#include <cctype>
#include <string>
#include <vector>

// Shared string utilities. Header-only, C-style global helpers (see CLAUDE.md).

// Splits `s` into tokens on every occurrence of `delim`.
//
// Default (skip_empty == false) keeps empty tokens and always yields at least
// one token (an empty string for empty input), matching the classic
// substr-on-each-delimiter loop. With skip_empty == true, empty tokens are
// dropped (so "" yields no tokens, "a,,b" yields {"a","b"}).
//
// No whitespace trimming is performed; callers that need it should trim each
// returned token themselves (split first, then transform).
inline std::vector<std::string> split(const std::string &s, char delim, bool skip_empty = false) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        size_t pos = s.find(delim, start);
        std::string tok =
            s.substr(start, pos == std::string::npos ? std::string::npos : pos - start);
        if (!skip_empty || !tok.empty()) out.push_back(tok);
        if (pos == std::string::npos) break;
        start = pos + 1;
    }
    return out;
}

// Returns `s` lowercased byte-wise (ASCII only). For case-insensitive matching of
// script tokens whose casing varies across Forge card scripts (e.g. Cityscape
// Leveler's "nonland" vs Abrupt Decay's "nonLand").
inline std::string ascii_lower(const std::string &s) {
    std::string out = s;
    for (char &c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

#endif  // STR_UTIL_H
