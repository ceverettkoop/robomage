#include "mana_unit.h"

#include <algorithm>
#include <set>

const ManaRestriction *intern_mana_restriction(const ManaRestriction &r) {
    static std::set<ManaRestriction> store;
    return &*store.insert(r).first;
}

size_t mana_of_color(const ManaPaid &paid, Colors color) {
    return static_cast<size_t>(
        std::count_if(paid.begin(), paid.end(), [color](const ManaUnit &unit) { return unit.color == color; }));
}

std::set<Colors> colors_of(const ManaPaid &paid) {
    std::set<Colors> colors;
    for (const ManaUnit &unit : paid)
        if (unit.color >= WHITE && unit.color <= GREEN) colors.insert(unit.color);
    return colors;
}

size_t mana_from(const ManaPaid &paid, Entity source) {
    return static_cast<size_t>(std::count_if(
        paid.begin(), paid.end(), [source](const ManaUnit &unit) { return unit.source.lki_entity() == source; }));
}

size_t ManaPool::count(Colors color) const {
    return static_cast<size_t>(std::count_if(
        units.begin(), units.end(), [color](const ManaUnit &unit) { return unit.color == color; }));
}

ManaPool::const_iterator ManaPool::find(Colors color) const {
    return std::find_if(units.begin(), units.end(), [color](const ManaUnit &unit) { return unit.color == color; });
}

void ManaPool::insert(const ManaUnit &unit) {
    units.insert(std::upper_bound(units.begin(), units.end(), unit,
                     [](const ManaUnit &a, const ManaUnit &b) { return a.color < b.color; }),
        unit);
}
