#include "types.h"


const std::vector<std::string> &nonbasic_land_types() {
    static const std::vector<std::string> kTypes = {
        "Cave", "Desert", "Gate", "Lair", "Locus", "Mine", "Planet",
        "Power-Plant", "Sphere", "Tower", "Town", "Urza's"};
    return kTypes;
}

bool is_land_subtype(const std::string &name) {
    if (is_basic_land_subtype(name)) return true;
    for (const auto &t : nonbasic_land_types())
        if (t == name) return true;
    return false;
}
