#ifndef COLORS_H
#define COLORS_H

#include <set>
#include <string>

enum Colors {
    WHITE,
    BLUE,
    BLACK,
    RED,
    GREEN,
    COLORLESS,
    GENERIC,
    NO_COLOR
};

using ManaValue = std::multiset<Colors>;

std::string mana_symbol(Colors color);

// Same symbol as mana_symbol() but as a string literal (static lifetime), for the
// many printf/game_log call sites that want a `const char*` without constructing a
// std::string. Single source shared across the systems.
const char *mana_symbol_str(Colors color);

#endif /* COLORS_H */
