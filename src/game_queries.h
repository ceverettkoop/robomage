#ifndef GAME_QUERIES_H
#define GAME_QUERIES_H

#include <set>
#include <string>
#include <vector>
#include "ecs/entity.h"
#include "components/carddata.h"
#include "components/creature.h"
#include "components/permanent.h"
#include "components/spell.h"
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
inline bool is_planeswalker_card(const CardData &cd) { return card_has_type(cd, "Planeswalker"); }

// True if the type list (e.g. Permanent::types) carries the given top-level type.
inline bool type_set_has(const std::set<Type> &types, const std::string &type_name) {
    for (const auto &t : types)
        if (t.kind == TYPE && t.name == type_name) return true;
    return false;
}
inline bool is_planeswalker(const std::set<Type> &types) { return type_set_has(types, "Planeswalker"); }

// True if `e` is a planeswalker permanent on the battlefield (the form damage/combat care about).
inline bool is_planeswalker_permanent(Entity e) {
    return global_coordinator.entity_has_component<Permanent>(e) &&
           is_planeswalker(global_coordinator.GetComponent<Permanent>(e).types);
}

// Damage to a planeswalker removes that many loyalty counters (306.8). The loyalty-0 SBA
// (704.5i) then moves it to the graveyard. Single source for both combat and noncombat damage.
inline void damage_planeswalker(Entity pw, size_t amount) {
    auto &perm = global_coordinator.GetComponent<Permanent>(pw);
    perm.loyalty -= static_cast<int>(amount);
}

// True if the type list carries the "Legendary" supertype (drives the legend rule, 704.5j).
inline bool has_legendary_supertype(const std::set<Type> &types) {
    for (const auto &t : types)
        if (t.kind == SUPERTYPE && t.name == "Legendary") return true;
    return false;
}

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

// True if the creature deals damage during the first-strike combat damage step
// (it has First Strike or Double Strike). Single source for "does a first-strike
// damage step matter": the step-skip scan and the per-creature damage gate both
// consume this so they cannot drift on the keyword literals.
inline bool creature_deals_first_strike_damage(const Creature &cr) {
    return creature_has_keyword(cr, "First Strike") ||
           creature_has_keyword(cr, "Double Strike");
}

// True if `e` is a spell that was cast via flashback. Such a spell is exiled
// (rather than sent to the graveyard) when it leaves the stack — whether it
// resolves or is countered. Single source for that "leaves-stack → exile" rule.
inline bool spell_cast_with_flashback(Entity e) {
    return global_coordinator.entity_has_component<Spell>(e) &&
           global_coordinator.GetComponent<Spell>(e).cast_with_flashback;
}

// True if the type list carries the "Basic" supertype (i.e. a basic land). This is
// the supertype check used for nonBasic-land target/search filters — distinct from
// is_basic_land_subtype(), which matches the six basic-land *subtype* names.
inline bool has_basic_supertype(const std::set<Type> &types) {
    for (const auto &t : types)
        if (t.kind == SUPERTYPE && t.name == "Basic") return true;
    return false;
}

// True for the six basic land subtype names that carry an innate mana ability.
inline bool is_basic_land_subtype(const std::string &name) {
    return name == "Mountain" || name == "Forest" || name == "Plains" ||
           name == "Island" || name == "Swamp" || name == "Wastes";
}

// True if `perm` has a subtype/type named in the ';'-delimited `spec`
// (e.g. a Sacrifice cost "Forest;Plains", or a single Return-cost subtype).
inline bool permanent_matches_subtype_spec(const Permanent &perm, const std::string &spec) {
    size_t pp = 0;
    while (pp <= spec.size()) {
        size_t sc = spec.find(';', pp);
        if (sc == std::string::npos) sc = spec.size();
        std::string sub = spec.substr(pp, sc - pp);
        for (const auto &t : perm.types)
            if (t.name == sub) return true;
        pp = sc + 1;
    }
    return false;
}

// Battlefield permanents controlled by `player` matching the ';'-delimited subtype
// `spec`. Drives both Sacrifice-a-<type> and Return-a-<type> activation costs:
// non-empty == the cost is payable (legality), and the list itself is the player's
// choice menu (payment).
inline std::vector<Entity> controlled_permanents_matching(
    Zone::Ownership player, const std::string &spec, const std::set<Entity> &entities) {
    std::vector<Entity> out;
    for (auto e : entities) {
        if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location != Zone::BATTLEFIELD) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.controller != player) continue;
        if (permanent_matches_subtype_spec(perm, spec)) out.push_back(e);
    }
    return out;
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
