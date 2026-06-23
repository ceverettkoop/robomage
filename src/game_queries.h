#ifndef GAME_QUERIES_H
#define GAME_QUERIES_H

#include <set>
#include <string>
#include "ecs/entity.h"
#include "components/carddata.h"
#include "components/creature.h"
#include "components/types.h"
#include "components/zone.h"
#include "ecs/coordinator.h"

extern Coordinator global_coordinator;

// ── Shared entity/card predicates ───────────────────────────────────────────
// Small, hot, behaviour-preserving queries factored out of the systems so the
// same type/zone/keyword checks are written once. All are header-inline because
// they sit on the SBA fixpoint loop and per-action legality scans.

// True if the card has the given top-level type (kind == TYPE), e.g. "Creature".
inline bool card_has_type(const CardData &cd, const std::string &type_name) {
    for (const auto &t : cd.types)
        if (t.kind == TYPE && t.name == type_name) return true;
    return false;
}

inline bool is_creature_card(const CardData &cd) { return card_has_type(cd, "Creature"); }
inline bool is_land_card(const CardData &cd)     { return card_has_type(cd, "Land"); }

// True if the entity currently sits on the battlefield.
inline bool on_battlefield(Entity e) {
    return global_coordinator.entity_has_component<Zone>(e) &&
           global_coordinator.GetComponent<Zone>(e).location == Zone::BATTLEFIELD;
}

// True if the creature carries the given keyword string (exact match).
inline bool creature_has_keyword(const Creature &cr, const char *kw) {
    for (const auto &k : cr.keywords)
        if (k == kw) return true;
    return false;
}

// True for the six basic land subtype names that carry an innate mana ability.
inline bool is_basic_land_subtype(const std::string &name) {
    return name == "Mountain" || name == "Forest" || name == "Plains" ||
           name == "Island" || name == "Swamp" || name == "Wastes";
}

// Returns true when the given player has 4+ card types among cards in their graveyard.
inline bool check_delirium(Zone::Ownership owner, const std::set<Entity> &entities) {
    std::set<std::string> type_names;
    for (auto entity : entities) {
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(entity);
        if (z.location != Zone::GRAVEYARD || z.owner != owner) continue;
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;
        for (auto &t : global_coordinator.GetComponent<CardData>(entity).types)
            if (t.kind == TYPE) type_names.insert(t.name);
    }
    return type_names.size() >= 4;
}

#endif /* GAME_QUERIES_H */
