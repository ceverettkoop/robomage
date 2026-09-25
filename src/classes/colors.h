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

// A mana value written as a printed cost: the generic amount first, then each colored/colorless
// pip in WUBRGC order ("{2}{R}", "{R}{R}", "{1}"). An empty value reads "{0}".
std::string mana_value_text(const ManaValue &mana);

#endif /* COLORS_H */
