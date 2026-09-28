#ifndef QUERIES_CHARACTERISTICS_H
#define QUERIES_CHARACTERISTICS_H

#include <set>
#include <string>
#include <vector>
#include "../classes/colors.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/spell.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"
#include "types.h"

// ── Characteristics (CR 109.3) ─────────────────────────────────────────────────
// An object's characteristics as the rules read them: the face that's up, its colors, mana
// value, effective power/toughness, and its display name.

// The CardData face a permanent entity is currently showing: the back face while the
// permanent is transformed (CR 712.8e — a face-up back face has its own characteristics),
// else the front. A spell cast as its modal back face or as a split card's second half has only
// that face's characteristics on the stack (CR 712.8f, 709.3). Falls back to the front face for
// single-faced cards and any other object.
// Use this instead of reading GetComponent<CardData>(e) directly wherever a transformed
// permanent's PRINTED characteristics (colors, types, P/T) matter; note that MANA VALUE is
// the one characteristic that does NOT follow the active face for a nonmodal DFC (712.8e
// computes it from the front face's cost; only a MODAL back face carries its own, 712.8d).
inline const CardData &active_face(Entity e, const CardData &cd) {
    if (!cd.backside) return cd;
    if (global_coordinator.entity_has_component<Permanent>(e) &&
        global_coordinator.GetComponent<Permanent>(e).transformed)
        return *cd.backside;
    if (global_coordinator.entity_has_component<Spell>(e) &&
        global_coordinator.GetComponent<Spell>(e).cast_back_face)
        return *cd.backside;
    return cd;
}

// The replacement effects permanent `e` generates (CR 614): those of a card's face that's up
// (CR 712.8e), or a token's (a token copy has the copied card's, CR 707.2). Empty otherwise.
const std::vector<Effect::Replacement> &permanent_replacement_effects(Entity e);

// Printed colors of a card: an explicit Colors$ override if present (a color indicator, or
// Devoid's COLORLESS), otherwise the colors of its mana cost (CR 105.2 / 202.2). Only the five
// colors are ever returned — the COLORLESS marker means "no color" (CR 105.2c) — so the set is
// empty exactly when the card is colorless. Single source for the color of a card object, shared
// by the targeting color checks and the last-known-info snapshot.
std::set<Colors> card_colors(const CardData &cd);

// Mana value (converted mana cost, CR 202.3) of a card: one per non-hybrid pip in mana_cost
// plus each hybrid pip's contribution (1 for a color hybrid, N for an {N/color} twobrid —
// CR 202.3f: a hybrid symbol's MV is the greatest of its component symbols' MVs). X counts 0
// outside the stack/cast (CR 202.3b). Each Phyrexian pip ({B/P}) counts 1 toward mana value
// (CR 202.3f: a Phyrexian symbol is worth its color component, MV 1) regardless of whether the
// cost is actually paid with mana or life; phyrexian_mana is kept out of mana_cost (mirroring
// hybrid_mana), so fold it in here. Single source for "card CMC" of one face — a split card's
// half, not the whole card (see object_mana_value for a card in a zone).
inline int card_mana_value(const CardData &cd) {
    int mv = static_cast<int>(cd.mana_cost.size());
    for (const auto &pip : cd.hybrid_mana) mv += pip.mana_value;
    mv += static_cast<int>(cd.phyrexian_mana.size());
    return mv;
}

// Mana value of object `e` (printed characteristics `cd`). A spell on the stack has the face
// that was cast (a split card's half or a modal back face, CR 709.3, 712.8f; active_face), and
// each {X} in its mana cost counts as the value announced for X (CR 202.3e). In every other zone
// X is 0 (CR 202.3b) and a split card's mana value is that of its two halves combined (CR
// 709.4b). With no entity (e == 0) the face `cd` is read on its own.
int object_mana_value(Entity e, const CardData &cd);

// The face whose mana cost gives a permanent its mana value (CR 112.7): a transformed NONMODAL
// permanent keeps the front face's (CR 712.8e — Insectile Aberration is MV 1 from Delver's cost),
// but a face-up MODAL back has entirely its own characteristics (CR 712.8d), so Witch-Blessed
// Meadow in play is MV 0, not the front spell's 4.
inline const CardData &mana_value_face(const CardData &cd, bool transformed) {
    return (cd.is_modal_dfc && transformed && cd.backside) ? *cd.backside : cd;
}

// Unified "characteristic at the time it is read" accessors (CR 608.2h). Each returns the
// object's effective value: read live from its battlefield components while it is in play
// (so all applied continuous effects/counters are reflected — and, because every effective-P/T
// mutator resyncs the cached Creature P/T synchronously, this is correct even mid-resolution),
// and from the last-known-info snapshot captured at battlefield-leave once it has gone. Falls
// back to the card's printed characteristics for objects that were never on the battlefield
// (e.g. a spell on the stack). Defined in characteristics.cpp (needs cur_game). All
// "read a target's power/toughness/color at resolution" sites route through these.
int effective_power(Entity e);
int effective_toughness(Entity e);
std::set<Colors> effective_colors(Entity e);

// Layer-5 (CR 613.1e / 612) global color-changing override. If an active SetColor$ continuous
// static (Mycosynth Lattice) designates `e` — via its Affected$ filter and AffectedZone$ — write
// the override color set into `out` and return true; otherwise return false (no override).
// "Colorless" yields an empty set (CR 105.2c); an explicit color list yields those colors.
// Defined in state_manager_statics.cpp (where g_active_statics lives). Consulted by
// effective_colors and the colorless queries so every color-dependent check (protection-from-
// color targeting, is_colorless) sees the affected object's effective color. Matches against the
// object's PRINTED characteristics (card_matches_filter) to avoid recursing back into
// effective_colors; tokens (no CardData) are not matched here.
bool setcolor_override_for(Entity e, std::set<Colors> &out);

// True if the object is colorless (CR 105.2c): it has none of the five colors, read from its
// effective colors (so a Devoid card, a colorless token, or anything under a Mycosynth Lattice
// SetColor override is colorless, and a transformed face reads its own colors).
inline bool is_colorless(Entity e) { return effective_colors(e).empty(); }

// True if `e` is a planeswalker permanent on the battlefield (the form damage/combat care about).
inline bool is_planeswalker_permanent(Entity e) {
    return global_coordinator.entity_has_component<Permanent>(e) &&
           is_planeswalker(global_coordinator.GetComponent<Permanent>(e).types);
}

// Human-readable name for an entity, used in game_log output and action labels. The single
// shared resolver (do not open-code a CardData/Permanent name ternary at a log site):
// a battlefield object's Permanent name (a token is tagged " token"), then the card's
// printed CardData name, then a lingering Token component (a token that already left the
// battlefield keeps only Token), then a standalone ability entity described via its source
// card ("Sylvan Library's ability") or its effect category, then the last-known name of a
// permanent that left play and ceased to exist (a token, CR 111.7). "<unknown>" only when the
// entity carries no name-bearing component and no last-known information. Defined in
// characteristics.cpp.
std::string entity_name(Entity e);

#endif /* QUERIES_CHARACTERISTICS_H */
