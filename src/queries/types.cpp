#include "types.h"

#include "../components/static_ability.h"
#include "../type_constants.h"


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

std::vector<Type> parse_added_types(const std::string &spec) {
    std::vector<Type> out;
    size_t p = 0;
    while (p < spec.size()) {
        size_t sep = spec.find(" & ", p);
        if (sep == std::string::npos) sep = spec.size();
        std::string name = spec.substr(p, sep - p);
        if (!name.empty()) {
            Type t;
            t.name = name;
            if (const std::string *entry = subtype_entry(name)) {
                t.name = *entry;
                t.kind = SUBTYPE;
            } else if (all_types.find(name) != all_types.end()) {
                t.kind = TYPE;
            } else if (all_supertypes.find(name) != all_supertypes.end()) {
                t.kind = SUPERTYPE;
            } else {
                t.kind = SUBTYPE;
            }
            out.push_back(t);
        }
        p = (sep < spec.size()) ? sep + 3 : sep;
    }
    return out;
}

std::vector<Type> types_added_by_static(const StaticAbility &sa) {
    if (sa.add_type != "AllNonBasicLandType") return parse_added_types(sa.add_type);
    std::vector<Type> out;
    for (const std::string &t : nonbasic_land_types()) out.push_back({SUBTYPE, t});
    return out;
}
