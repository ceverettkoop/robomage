#include "svar_eval.h"

#include <cctype>
#include <set>
#include <string>

#include "components/carddata.h"
#include "components/permanent.h"
#include "components/types.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "game_queries.h"

extern Coordinator global_coordinator;

// The bare operator table — one home for the EQ/NE/GE/LE/GT/LT switch.
bool apply_svar_op(int lhs, const std::string &op2, int rhs) {
    if (op2 == "EQ") return lhs == rhs;
    if (op2 == "NE") return lhs != rhs;
    if (op2 == "GE") return lhs >= rhs;
    if (op2 == "LE") return lhs <= rhs;
    if (op2 == "GT") return lhs >  rhs;
    if (op2 == "LT") return lhs <  rhs;
    return false;
}

// Simple SVar comparison logic (shared between statics and alt costs).
// A leading two-letter operator followed by an integer; anything else is false.
bool compare_svar(int value, const std::string &compare) {
    if (compare.size() < 2) return false;
    std::string op = compare.substr(0, 2);
    if (op != "EQ" && op != "NE" && op != "GE" &&
        op != "LE" && op != "GT" && op != "LT") return false;
    return apply_svar_op(value, op, std::stoi(compare.substr(2)));
}

// Evaluate a StaticAbility SVar expression such as "Count$TypeInYourYard.Land".
// Returns the computed integer value.
int evaluate_sa_svar(const std::string &expr, Zone::Ownership controller, Entity source) {
    // A plain integer literal (e.g. Humility's SetPower$ 1 / SetToughness$ 1) evaluates
    // to itself. Without this, a constant SetPower/SetToughness would fall through to the
    // Count$ handlers and return 0 (making the creature 0/0).
    if (!expr.empty()) {
        size_t i = (expr[0] == '-') ? 1 : 0;
        bool all_digits = i < expr.size();
        for (; i < expr.size(); ++i)
            if (!std::isdigit(static_cast<unsigned char>(expr[i]))) { all_digits = false; break; }
        if (all_digits) return std::stoi(expr);
    }

    // Handle /Plus.N suffix: strip it, evaluate the base, then add N
    size_t plus_pos = expr.find("/Plus.");
    if (plus_pos != std::string::npos) {
        std::string base = expr.substr(0, plus_pos);
        int offset = std::stoi(expr.substr(plus_pos + 6));
        return evaluate_sa_svar(base, controller, source) + offset;
    }

    // Handle /LimitMax.N suffix: strip it, evaluate the base, then cap at N
    // (e.g. "Count$ValidGraveyard Instant.YouOwn/LimitMax.1" → 0 or 1).
    size_t limit_pos = expr.find("/LimitMax.");
    if (limit_pos != std::string::npos) {
        std::string base = expr.substr(0, limit_pos);
        int cap = std::stoi(expr.substr(limit_pos + 10));
        int val = evaluate_sa_svar(base, controller, source);
        return val < cap ? val : cap;
    }

    // Count$ValidExile ... CardTypes — distinct card types among the cards exiled
    // with `source` (e.g. Keen-Eyed Curator's exiled-with pile). Scoped to the
    // source permanent, hence the `source` parameter.
    if (expr.find("Count$ValidExile") != std::string::npos &&
        expr.find("CardTypes") != std::string::npos) {
        if (source == 0 || !global_coordinator.entity_has_component<Permanent>(source))
            return 0;
        auto &eperm = global_coordinator.GetComponent<Permanent>(source);
        std::set<std::string> type_names;
        for (auto ex_e : eperm.exiled_with) {
            if (!global_coordinator.entity_has_component<CardData>(ex_e)) continue;
            for (auto &t : global_coordinator.GetComponent<CardData>(ex_e).types)
                if (t.kind == TYPE) type_names.insert(t.name);
        }
        return static_cast<int>(type_names.size());
    }

    // Count$CardCounters.<CounterType> — number of counters of that kind on the SVar's
    // own source permanent (Aether Vial: "Count$CardCounters.CHARGE" for the charge-counter
    // count, used as the mana-value bound on the creature it can put onto the battlefield).
    // Scoped to `source`, hence the parameter (CR 122.1).
    if (expr.rfind("Count$CardCounters.", 0) == 0) {
        std::string counter_type = expr.substr(19);  // after "Count$CardCounters."
        if (source == 0) return 0;
        return get_counters(source, counter_type);
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
