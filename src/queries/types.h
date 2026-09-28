#ifndef QUERIES_TYPES_H
#define QUERIES_TYPES_H

#include <set>
#include <string>
#include <vector>
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/types.h"

// ── Type-line and printed-keyword predicates (CR 205) ─────────────────────────
// Pure reads of a card's printed type line or a permanent's live Type set. Header-inline
// because they sit on the SBA fixpoint loop and per-action legality scans.

// True if the card has the given top-level type (kind == TYPE), e.g. "Creature".
inline bool card_has_type(const CardData &cd, const std::string &type_name) {
    for (const auto &t : cd.types)
        if (t.kind == TYPE && t.name == type_name) return true;
    return false;
}

// True if the card's printed keywords include `keyword`, e.g. "Flash".
inline bool card_has_keyword(const CardData &cd, const std::string &keyword) {
    for (const auto &kw : cd.keywords)
        if (kw == keyword) return true;
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

// True if this card in HAND consumes a land drop when played (CR 305.2): a land card, or
// a modal DFC whose BACK face is a land (Witch Enchanter // Witch-Blessed Meadow — playing
// that face is a land play, not a cast). Single source shared by the PLAY_LAND legal-action
// enumeration and the ML observation's lands-in-hand count, so the two can't disagree about
// which hand cards are land drops.
inline bool card_playable_as_land(const CardData &cd) {
    return is_land_card(cd) ||
           (cd.is_modal_dfc && cd.backside && is_land_card(*cd.backside));
}

// True if the card has a permanent card type (CR 110.4a: artifact, battle, creature,
// enchantment, land, planeswalker). Used by ValidCard$ Permanent zone-change filters
// (Moonshadow: "permanent cards put into your graveyard" excludes instants/sorceries).
inline bool is_permanent_card(const CardData &cd) {
    return card_has_type(cd, "Artifact") || card_has_type(cd, "Battle") ||
           card_has_type(cd, "Creature") || card_has_type(cd, "Enchantment") ||
           card_has_type(cd, "Land") || card_has_type(cd, "Planeswalker");
}

// True if the type list (e.g. Permanent::types) carries the given top-level type.
inline bool type_set_has(const std::set<Type> &types, const std::string &type_name) {
    for (const auto &t : types)
        if (t.kind == TYPE && t.name == type_name) return true;
    return false;
}
inline bool is_planeswalker(const std::set<Type> &types) { return type_set_has(types, "Planeswalker"); }

// True if the type list carries the "Legendary" supertype (drives the legend rule, 704.5j).
inline bool has_legendary_supertype(const std::set<Type> &types) {
    for (const auto &t : types)
        if (t.kind == SUPERTYPE && t.name == "Legendary") return true;
    return false;
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

// The nonbasic land types: every land type of CR 205.3i other than the five basic ones.
const std::vector<std::string> &nonbasic_land_types();

// True for a land type (CR 205.3i), basic or nonbasic, including the engine's Wastes marker.
bool is_land_subtype(const std::string &name);

// True when the type list carries at least one of the FIVE basic land types —
// Plains/Island/Swamp/Mountain/Forest (CR 205.3i / 305.6). Wastes is deliberately
// excluded here: it is a basic land with NO basic land type, even though
// is_basic_land_subtype() lists it for the innate-mana-ability rule. A basic
// Mountain qualifies (subtype Mountain), a dual like Scrubland qualifies
// (Plains Swamp), a subtype-less utility land (Wasteland) does not. Drives the
// `hasABasicLandType` filter qualifier (Boseiju, Who Endures' compensation search).
inline bool has_a_basic_land_type(const std::set<Type> &types) {
    for (const auto &t : types)
        if (t.kind == SUBTYPE && t.name != "Wastes" && is_basic_land_subtype(t.name))
            return true;
    return false;
}

#endif /* QUERIES_TYPES_H */
