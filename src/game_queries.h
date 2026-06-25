#ifndef GAME_QUERIES_H
#define GAME_QUERIES_H

#include <set>
#include <string>
#include <vector>
#include "ecs/entity.h"
#include "components/carddata.h"
#include "components/creature.h"
#include "components/damage.h"
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

// ── Counters (122.1) ────────────────────────────────────────────────────────
// Every counter kind (+1/+1, -1/-1, loyalty, keyword) lives in Permanent::counters
// keyed by type; these helpers are the single add/remove/query path (T2.4).

// Number of counters of `type` on `e` (0 if none, or `e` is not a permanent).
inline int get_counters(Entity e, const std::string &type) {
    if (!global_coordinator.entity_has_component<Permanent>(e)) return 0;
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    auto it = perm.counters.find(type);
    return it == perm.counters.end() ? 0 : it->second;
}

// Recompute a creature's cached +1/+1 − -1/-1 P/T contribution from its counters
// (layer 7c, 613.4c) and refresh effective P/T. No-op if `e` is not a creature.
inline void refresh_counter_pt(Entity e) {
    if (!global_coordinator.entity_has_component<Creature>(e)) return;
    auto &cr = global_coordinator.GetComponent<Creature>(e);
    cr.counter_pt_bonus = get_counters(e, "P1P1") - get_counters(e, "M1M1");
    recompute_pt(cr);
}

// Add `delta` counters of `type` to `e` (delta may be negative). An entry reaching
// exactly 0 is erased so the map only holds live counters. +1/+1 and -1/-1 changes
// resync the creature's P/T. Returns the new total; no-op (returns current) if not a permanent.
inline int add_counters(Entity e, const std::string &type, int delta) {
    if (delta == 0 || !global_coordinator.entity_has_component<Permanent>(e))
        return get_counters(e, type);
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    int total = perm.counters[type] + delta;
    if (total == 0) perm.counters.erase(type);
    else perm.counters[type] = total;
    if (type == "P1P1" || type == "M1M1") refresh_counter_pt(e);
    return total;
}

// Damage to a planeswalker removes that many loyalty counters (306.8). The loyalty-0 SBA
// (704.5i) then moves it to the graveyard. Single source for both combat and noncombat damage.
inline void damage_planeswalker(Entity pw, size_t amount) {
    add_counters(pw, "LOYALTY", -static_cast<int>(amount));
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

// True if `e` is a *live* battlefield permanent: it carries a Permanent component,
// its Zone is BATTLEFIELD, and it is not phased out (702.26e — a phased-out permanent
// is treated as though it doesn't exist), optionally controlled by `ctrl` (UNKNOWN =
// any controller). This is the single source of "is this on the battlefield": prefer
// it (or battlefield_permanents() below) over open-coding the
// Permanent+Zone+BATTLEFIELD(+phased)(+controller) check, so the phasing rule lives in
// exactly one place. The only code that should read Permanent::is_phased_out directly
// is the phasing subsystem itself (the untap-step phase-in/skip in game.cpp) and the
// rare loop that must still process phased-out permanents (e.g. resetting their cached
// P/T before skipping them when gathering static abilities).
inline bool is_battlefield_permanent(Entity e, Zone::Ownership ctrl = Zone::UNKNOWN) {
    if (!global_coordinator.entity_has_component<Permanent>(e)) return false;
    if (!global_coordinator.entity_has_component<Zone>(e)) return false;
    if (global_coordinator.GetComponent<Zone>(e).location != Zone::BATTLEFIELD) return false;
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    if (perm.is_phased_out) return false;
    if (ctrl != Zone::UNKNOWN && perm.controller != ctrl) return false;
    return true;
}

// All live battlefield permanents (phased-out excluded), optionally only those
// controlled by `ctrl`. Pass the iterating system's mEntities (or orderer->mEntities).
// Prefer this over re-scanning entities inline when you need the whole set.
inline std::vector<Entity> battlefield_permanents(
    const std::set<Entity> &entities, Zone::Ownership ctrl = Zone::UNKNOWN) {
    std::vector<Entity> out;
    for (auto e : entities)
        if (is_battlefield_permanent(e, ctrl)) out.push_back(e);
    return out;
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

// Combat damage already marked on an entity this turn (0 if it has no Damage component).
inline uint32_t marked_damage_on(Entity e) {
    if (global_coordinator.entity_has_component<Damage>(e))
        return static_cast<uint32_t>(global_coordinator.GetComponent<Damage>(e).damage_counters);
    return 0u;
}

// Damage `attacker` must assign to `blocker` for that blocker to count as receiving
// lethal damage (used both for the auto-assign path and the "can it kill everything?"
// threshold). Deathtouch makes any nonzero amount lethal (702.2c); otherwise lethal is
// the blocker's remaining toughness after damage already marked on it (702.19b / T3.11).
inline uint32_t lethal_needed_for_blocker(Entity attacker, Entity blocker) {
    const Creature &acr = global_coordinator.GetComponent<Creature>(attacker);
    const Creature &bcr = global_coordinator.GetComponent<Creature>(blocker);
    if (creature_has_keyword(acr, "Deathtouch")) return bcr.toughness > 0 ? 1u : 0u;
    uint32_t marked = marked_damage_on(blocker);
    return (bcr.toughness > marked) ? bcr.toughness - marked : 0u;
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
