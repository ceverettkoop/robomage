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
#include "components/player.h"
#include "components/spell.h"
#include "components/token.h"
#include "components/types.h"
#include "components/zone.h"
#include "classes/colors.h"
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

// True if the permanent carries a type/subtype whose name matches `type_name`
// (any kind — top-level type, supertype, or subtype). Used by effects that scan a
// permanent's type line (Amass's Army check, sacrifice's SacValid$ filter).
inline bool permanent_has_type(const Permanent &perm, const std::string &type_name) {
    for (const auto &t : perm.types)
        if (t.name == type_name) return true;
    return false;
}

inline bool is_creature_card(const CardData &cd) { return card_has_type(cd, "Creature"); }
inline bool is_land_card(const CardData &cd)     { return card_has_type(cd, "Land"); }
inline bool is_planeswalker_card(const CardData &cd) { return card_has_type(cd, "Planeswalker"); }

// Printed colors of a card: an explicit Colors$ override if present (e.g. Devoid's COLORLESS),
// otherwise the colors of its mana cost (CR 105.2 / 202.2). Single source for the color of a
// card object, shared by the targeting color checks and the last-known-info snapshot.
inline std::set<Colors> card_colors(const CardData &cd) {
    if (!cd.explicit_colors.empty()) return cd.explicit_colors;
    std::set<Colors> result;
    for (Colors c : {WHITE, BLUE, BLACK, RED, GREEN})
        if (cd.mana_cost.count(c)) result.insert(c);
    return result;
}

// Enforce a positive color target restriction (e.g. ValidTgts$ Permanent.Blue on Red Elemental
// Blast: "target blue permanent", CR 115.1) against an already-resolved color set. Sharing the
// color set (rather than re-reading printed colors) is what lets battlefield/last-known callers
// honour effective color uniformly.
inline bool color_set_passes(const std::string &vt, const std::set<Colors> &colors) {
    static const struct { const char *tok; Colors col; } table[] = {
        {".White", WHITE}, {".Blue", BLUE}, {".Black", BLACK}, {".Red", RED}, {".Green", GREEN}};
    for (auto &e : table)
        if (vt.find(e.tok) != std::string::npos && !colors.count(e.col)) return false;
    return true;
}

// Enforce a "non<Color>" target restriction (e.g. ValidTgts$ Creature.nonBlack on Snuff Out)
// against an already-resolved color set: reject when the candidate is one of the excluded
// colors. Counterpart to color_set_passes; routed through effective_colors for the same reason.
inline bool color_set_passes_noncolor(const std::string &vt, const std::set<Colors> &colors) {
    static const struct { const char *tok; Colors col; } table[] = {
        {"nonWhite", WHITE}, {"nonBlue", BLUE}, {"nonBlack", BLACK}, {"nonRed", RED}, {"nonGreen", GREEN}};
    for (auto &e : table)
        if (vt.find(e.tok) != std::string::npos && colors.count(e.col)) return false;
    return true;
}

// Unified "characteristic at the time it is read" accessors (CR 608.2h). Each returns the
// object's effective value: read live from its battlefield components while it is in play
// (so all applied continuous effects/counters are reflected — and, because every effective-P/T
// mutator resyncs the cached Creature P/T synchronously, this is correct even mid-resolution),
// and from the last-known-info snapshot captured at battlefield-leave once it has gone. Falls
// back to the card's printed characteristics for objects that were never on the battlefield
// (e.g. a spell on the stack). Defined in game_queries.cpp (needs cur_game). All
// "read a target's power/toughness/color at resolution" sites route through these.
int effective_power(Entity e);
int effective_toughness(Entity e);
std::set<Colors> effective_colors(Entity e);

// True if the card has a permanent card type (CR 110.4a: artifact, battle, creature,
// enchantment, land, planeswalker). Used by ValidCard$ Permanent zone-change filters
// (Moonshadow: "permanent cards put into your graveyard" excludes instants/sorceries).
inline bool is_permanent_card(const CardData &cd) {
    return card_has_type(cd, "Artifact") || card_has_type(cd, "Battle") ||
           card_has_type(cd, "Creature") || card_has_type(cd, "Enchantment") ||
           card_has_type(cd, "Land") || card_has_type(cd, "Planeswalker");
}

// True if the card is colorless (CR 105.2c): no colored mana symbol in its mana cost and no
// Colors: override granting it a color. A `Colors:`/Devoid override (explicit_colors) takes
// precedence over the cost; otherwise the printed mana cost's colored symbols decide. Mirrors
// the colorless test in mana_system.cpp so spell/permanent colorlessness is computed once.
inline bool is_colorless_card(const CardData &cd) {
    for (Colors c : {WHITE, BLUE, BLACK, RED, GREEN}) {
        if (!cd.explicit_colors.empty()) {
            if (cd.explicit_colors.count(c)) return false;
        } else if (cd.mana_cost.count(c)) {
            return false;
        }
    }
    return true;
}

// True if the entity is colorless (CR 105.2c), handling both real cards (CardData) and
// tokens (Token, which have no mana cost — their color is the token's color indicator).
inline bool is_colorless_entity(Entity e) {
    if (global_coordinator.entity_has_component<CardData>(e))
        return is_colorless_card(global_coordinator.GetComponent<CardData>(e));
    if (global_coordinator.entity_has_component<Token>(e)) {
        const auto &cols = global_coordinator.GetComponent<Token>(e).explicit_colors;
        for (Colors c : {WHITE, BLUE, BLACK, RED, GREEN})
            if (cols.count(c)) return false;
        return true;  // no colored indicator ⇒ colorless
    }
    return false;
}

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

// True if `perm` (the entity `perm_entity`) matches the ';'-delimited `spec` used by
// Sacrifice-a-<type> / Return-a-<type> activation costs (e.g. "Forest;Plains",
// "Creature", "Creature.Other"). Each ';'-delimited alternative is a type/subtype name
// (a top-level card type like "Creature"/"Land" matches alongside subtype names, since
// Permanent::types stores both) optionally followed by a '.'-joined qualifier:
//   .Other  — self-exclusion (CR 700.5 / 109.3): the permanent must NOT be the cost's
//             source (`exclude_entity`), i.e. "another <type>". An empty/0 exclude makes
//             .Other a no-op match on type alone.
//   .<Color> / .non<Color>  — a color restriction (e.g. "Creature.Green" on Natural Order's
//             "sacrifice a green creature"): the permanent's effective color must satisfy the
//             qualifier, evaluated via color_set_passes / color_set_passes_noncolor against the
//             permanent's effective_colors (read live, so continuous color-changing effects
//             count). Honoured only when `perm_entity` is supplied.
inline bool permanent_matches_subtype_spec(const Permanent &perm, const std::string &spec,
                                           Entity perm_entity = 0, Entity exclude_entity = 0) {
    size_t pp = 0;
    while (pp <= spec.size()) {
        size_t sc = spec.find(';', pp);
        if (sc == std::string::npos) sc = spec.size();
        std::string alt = spec.substr(pp, sc - pp);
        pp = sc + 1;

        // Split off a '.'-joined qualifier (e.g. "Creature.Other" → "Creature" + "Other").
        std::string type_name = alt;
        std::string qual;
        bool require_other = false;
        size_t dot = alt.find('.');
        if (dot != std::string::npos) {
            type_name = alt.substr(0, dot);
            qual = alt.substr(dot + 1);
            if (qual == "Other") require_other = true;
        }

        // .Other: the matching permanent must not be the cost's source (self-exclusion).
        if (require_other && perm_entity != 0 && exclude_entity != 0 && perm_entity == exclude_entity)
            continue;

        bool type_match = false;
        for (const auto &t : perm.types)
            if (t.name == type_name) { type_match = true; break; }
        if (!type_match) continue;

        // Color qualifier (e.g. ".Green" / ".nonGreen"): apply against the permanent's
        // effective colors. The matchers key on the raw qualifier text (".Green",
        // "nonGreen"), so feed them `alt` directly. A non-color qualifier (e.g. "Other")
        // is ignored here — color_set_passes only reacts to color tokens.
        if (!qual.empty() && qual != "Other" && perm_entity != 0) {
            std::set<Colors> cols = effective_colors(perm_entity);
            if (!color_set_passes(alt, cols)) continue;
            if (!color_set_passes_noncolor(alt, cols)) continue;
        }
        return true;
    }
    return false;
}

// Battlefield permanents controlled by `player` matching the ';'-delimited subtype
// `spec`. Drives both Sacrifice-a-<type> and Return-a-<type> activation costs:
// non-empty == the cost is payable (legality), and the list itself is the player's
// choice menu (payment). `exclude_entity` is the cost's source, honoured by a `.Other`
// qualifier in the spec (e.g. Wight of the Reliquary's "Sacrifice another creature").
inline std::vector<Entity> controlled_permanents_matching(
    Zone::Ownership player, const std::string &spec, const std::set<Entity> &entities,
    Entity exclude_entity = 0) {
    std::vector<Entity> out;
    for (auto e : entities) {
        if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location != Zone::BATTLEFIELD) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.controller != player) continue;
        if (permanent_matches_subtype_spec(perm, spec, e, exclude_entity)) out.push_back(e);
    }
    return out;
}

// The additional Sacrifice-a-<type> cost a spell pays as it is cast, taken from the
// card's SPELL ability `Cost$` (e.g. Natural Order's "Sac<1/Creature.Green>" →
// "Creature.Green"). Empty when the spell has no additional sacrifice cost. Mirrors how
// an activated ability stores its Sac cost on the Ability; reading it from the SPELL
// ability keeps the parser's real Cost$ tag authoritative (no retag). The cast path pays
// it with the same SACRIFICE_PERMANENT machinery activated abilities use, and cast
// legality requires a matching permanent (CR 601.2f).
inline std::string spell_additional_sac_spec(const CardData &cd) {
    for (const auto &ab : cd.abilities)
        if (ab.ability_type == Ability::SPELL && !ab.sac_cost_spec.empty())
            return ab.sac_cost_spec;
    return "";
}

// Single source for "a player gains life": raises their life total and accumulates
// life_gained_this_turn (118.9 / the "if you gained life this turn" check on cards like
// Ocelot Pride). Every life-gain site (lifelink, GainLife effects, combat lifelink) routes
// through this so the per-turn counter cannot drift from life_total. Pass the player's
// Player-component entity. No-op for amount <= 0.
inline void player_gain_life(Entity player_entity, int32_t amount) {
    if (amount <= 0 || !global_coordinator.entity_has_component<Player>(player_entity)) return;
    auto &pl = global_coordinator.GetComponent<Player>(player_entity);
    pl.life_total += amount;
    pl.life_gained_this_turn += amount;
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
