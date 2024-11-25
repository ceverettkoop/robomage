#ifndef COLORS_H
#define COLORS_H

#include <set>

enum Colors{
    WHITE,
    BLUE,
    BLACK,
    RED,
    GREEN,
    COLORLESS,
    GENERIC
};

using ManaValue = std::multiset<Colors>;
using ColorIdentity = std::set<Colors>;

#endif /* COLORS_H */
