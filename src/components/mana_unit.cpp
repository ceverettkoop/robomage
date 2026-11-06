#include "mana_unit.h"

#include <algorithm>
#include <set>

const ManaRestriction *intern_mana_restriction(const ManaRestriction &r) {
    static std::set<ManaRestriction> store;
    return &*store.insert(r).first;
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
