#ifndef COLORS_H
#define COLORS_H

#include <set>
#include <string>

enum Colors{
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
using ColorIdentity = std::set<Colors>;

std::string mana_symbol(Colors color);

#endif /* COLORS_H */
