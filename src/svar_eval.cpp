#include "svar_eval.h"

#include <set>
#include <string>

#include "components/carddata.h"
#include "components/types.h"
#include "components/zone.h"
#include "ecs/coordinator.h"

extern Coordinator global_coordinator;

// Simple SVar comparison logic (shared between statics and alt costs)
bool compare_svar(int value, const std::string &compare) {
    if (compare.rfind("EQ", 0) == 0)  return value == std::stoi(compare.substr(2));
    if (compare.rfind("NE", 0) == 0)  return value != std::stoi(compare.substr(2));
    if (compare.rfind("GE", 0) == 0)  return value >= std::stoi(compare.substr(2));
    if (compare.rfind("LE", 0) == 0)  return value <= std::stoi(compare.substr(2));
    if (compare.rfind("GT", 0) == 0)  return value >  std::stoi(compare.substr(2));
    if (compare.rfind("LT", 0) == 0)  return value <  std::stoi(compare.substr(2));
    return false;
}

// Evaluate a StaticAbility SVar expression such as "Count$TypeInYourYard.Land".
// Returns the computed integer value.
int evaluate_sa_svar(const std::string &expr, Zone::Ownership controller) {
    // Handle /Plus.N suffix: strip it, evaluate the base, then add N
    size_t plus_pos = expr.find("/Plus.");
    if (plus_pos != std::string::npos) {
        std::string base = expr.substr(0, plus_pos);
        int offset = std::stoi(expr.substr(plus_pos + 6));
        return evaluate_sa_svar(base, controller) + offset;
    }

    // Count$TypeInYourYard.<TypeName> — count cards of that type in controller's graveyard
    if (expr.rfind("Count$TypeInYourYard.", 0) == 0) {
        std::string type_name = expr.substr(21);  // after "Count$TypeInYourYard."
        int count = 0;
        Entity max_e = global_coordinator.GetMaxIssuedEntity();
        for (Entity e = 0; e < max_e; ++e) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location != Zone::GRAVEYARD || z.owner != controller) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            for (auto &t : cd.types) {
                if (t.name == type_name) { count++; break; }
            }
        }
        return count;
    }

    // Count$ValidGraveyard Card$CardTypes — count distinct card types (Creature, Instant, etc.)
    // across both players' graveyards (Barrowgoyf)
    if (expr == "Count$CardTypesInAllGraveyards" ||
        expr == "Count$ValidGraveyard Card$CardTypes") {
        std::set<std::string> type_names;
        Entity max_e = global_coordinator.GetMaxIssuedEntity();
        for (Entity e = 0; e < max_e; ++e) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location != Zone::GRAVEYARD) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            for (auto &t : cd.types) {
                if (t.kind == TYPE) type_names.insert(t.name);
            }
        }
        return static_cast<int>(type_names.size());
    }

    // Count$ValidGraveyard <Type>[.<Restriction>...] — count cards of the given
    // type in graveyards, optionally restricted to the controller's own cards
    // (YouOwn/YouCtrl). E.g. "Land.YouOwn" for Knight of the Reliquary.
    if (expr.rfind("Count$ValidGraveyard ", 0) == 0) {
        std::string spec = expr.substr(21);  // after "Count$ValidGraveyard "
        std::string type_name = spec;
        bool you_own = false;
        size_t dot = spec.find('.');
        if (dot != std::string::npos) {
            type_name = spec.substr(0, dot);
            std::string rest = spec.substr(dot + 1);
            you_own = rest.find("YouOwn") != std::string::npos ||
                      rest.find("YouCtrl") != std::string::npos;
        }
        int count = 0;
        Entity max_e = global_coordinator.GetMaxIssuedEntity();
        for (Entity e = 0; e < max_e; ++e) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location != Zone::GRAVEYARD) continue;
            if (you_own && z.owner != controller) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            for (auto &t : cd.types) {
                if (t.kind == TYPE && t.name == type_name) { count++; break; }
            }
        }
        return count;
    }

    return 0;
}
