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

// Returns `s` without its leading and trailing whitespace.
inline std::string trim(const std::string &s) {
    size_t start = 0, end = s.size();
    while (start < end && std::isspace(static_cast<unsigned char>(s[start]))) start++;
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) end--;
    return s.substr(start, end - start);
}

// Returns `s` lowercased byte-wise (ASCII only). For case-insensitive matching of
// script tokens whose casing varies across Forge card scripts (e.g. Cityscape
// Leveler's "nonland" vs Abrupt Decay's "nonLand").
inline std::string ascii_lower(const std::string &s) {
    std::string out = s;
    for (char &c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// Fold a UTF-8 card name to ASCII so an accented Forge `Name:` (the .txt is
// gitignored and re-fetched accented, e.g. Lorien Revealed's script carries an
// accented "o") still matches an ASCII vocab entry. Transliterates the Latin-1
// Supplement accented letters (UTF-8 lead byte 0xC3) to their base ASCII letter,
// passes ASCII through unchanged, and drops any other non-ASCII byte. This is the
// ONE place name matching crosses the ASCII boundary; name_to_uid (filenames) and
// deck files stay ASCII as before.
inline std::string ascii_fold_card_name(const std::string &s) {
    // index = second UTF-8 byte - 0x80, i.e. codepoint U+00C0..U+00FF; 0 = drop.
    static const char latin1_base[64] = {
        'A', 'A', 'A', 'A', 'A', 'A', 0,   'C', 'E', 'E', 'E', 'E', 'I', 'I', 'I', 'I',  // C0..CF
        'D', 'N', 'O', 'O', 'O', 'O', 'O', 0,   'O', 'U', 'U', 'U', 'U', 'Y', 0,   0,    // D0..DF
        'a', 'a', 'a', 'a', 'a', 'a', 0,   'c', 'e', 'e', 'e', 'e', 'i', 'i', 'i', 'i',  // E0..EF
        'd', 'n', 'o', 'o', 'o', 'o', 'o', 0,   'o', 'u', 'u', 'u', 'u', 'y', 0,   'y',  // F0..FF
    };
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
        } else if (c == 0xC3 && i + 1 < s.size()) {
            unsigned char c2 = static_cast<unsigned char>(s[i + 1]);
            if (c2 >= 0x80 && c2 <= 0xBF) {
                char base = latin1_base[c2 - 0x80];
                if (base) out.push_back(base);
                ++i;  // consumed the continuation byte
            }
        }
        // any other non-ASCII byte is dropped
    }
    return out;
}

#endif  // STR_UTIL_H
