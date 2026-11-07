#ifndef QUERIES_CHARACTERISTICS_H
#define QUERIES_CHARACTERISTICS_H

#include <set>
#include <string>
#include <vector>
#include "../classes/colors.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/spell.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"
#include "types.h"

// ── Characteristics (CR 109.3) ─────────────────────────────────────────────────
// An object's characteristics as the rules read them: the face that's up, its colors, mana
// value, effective power/toughness, and its display name.

// The characteristics `cd` has when `overlay` is applied to it; null when it has none of its own.
inline const CardData *overlay_face(const CardData &cd, SpellOverlay overlay) {
    for (const OverlayFace &o : cd.overlay_faces)
        if (o.kind == overlay) return o.face.get();
    return nullptr;
}

// The characteristics a spell of `cd` cast by `how` has on the stack: those of its overlay, else of
// the face it is cast as (CR 712.8f, 709.3).
inline const CardData &cast_face(const CardData &cd, const CastOption &how) {
    if (const CardData *overlaid = overlay_face(cd, how.overlay)) return *overlaid;
    if (how.face == CastFace::BOTH && cd.combined) return *cd.combined;
    return how.face == CastFace::BACK && cd.backside ? *cd.backside : cd;
}

// The CardData face a permanent entity is currently showing: the back face while the
// permanent is transformed (CR 712.8e — a face-up back face has its own characteristics),
// else the front, or what the overlay it entered with gives it (bestowed). A spell has the
// characteristics it was cast with (cast_face). Falls back to the
// front face for single-faced cards and any other object.
// Use this instead of reading GetComponent<CardData>(e) directly wherever a transformed
// permanent's PRINTED characteristics (colors, types, P/T) matter; note that MANA VALUE is
// the one characteristic that does NOT follow the active face for a nonmodal DFC (712.8e
// computes it from the front face's cost; only a MODAL back face carries its own, 712.8d).
inline const CardData &active_face(Entity e, const CardData &cd) {
    if (!cd.backside && cd.overlay_faces.empty()) return cd;
    if (global_coordinator.entity_has_component<Permanent>(e)) {
        const Permanent &perm = global_coordinator.GetComponent<Permanent>(e);
        if (cd.backside && perm.transformed) return *cd.backside;
        if (const CardData *overlaid = overlay_face(cd, perm.overlay)) return *overlaid;
    }
    if (global_coordinator.entity_has_component<Spell>(e))
        return cast_face(cd, global_coordinator.GetComponent<Spell>(e).chosen.option);
    return cd;
}

// The card `cd` as it is anywhere but the battlefield: its printed face, under what its
// characteristic-defining abilities make it there (CR 604.3). For a card with no entity to ask, one
// about to be cast or named.
inline const CardData &off_battlefield_face(const CardData &cd) {
    return cd.off_battlefield ? *cd.off_battlefield : cd;
}

// The characteristics of the card `e` while it isn't a permanent: on the stack those it was cast
// with, in any other zone a split card's two halves combined (CR 709.3b, 709.4), and what a
// characteristic-defining ability makes it in either (CR 604.3). A card on the battlefield reads as
// printed: a Grist there is the planeswalker it says it is.
inline const CardData &card_characteristics(Entity e, const CardData &cd) {
    const CardData &face = global_coordinator.entity_has_component<Spell>(e) ? active_face(e, cd)
                           : cd.combined                                     ? *cd.combined
                                                                             : cd;
    if (!face.off_battlefield) return face;
    const bool in_play = global_coordinator.entity_has_component<Zone>(e) &&
                         global_coordinator.GetComponent<Zone>(e).location == Zone::BATTLEFIELD;
    return in_play ? face : *face.off_battlefield;
}

// ── One object in any zone (CR 109.3) ─────────────────────────────────────────
// What an object is where it is now: the reader behind filters and target checks. The layers own
// the battlefield (CR 613), so a permanent's values are what they left on its components. Anywhere
// else a card has its printed values, read through card_characteristics, and only the color a
// layer-5 static sets there changes them. None of these read last-known information: a card in a
// new zone is a new object with its own characteristics (CR 400.7), and a look back at the object
// that left is lki_for's, through effective_power and its neighbours below.
struct ObjectCharacteristics {
    const std::set<Type> *types = nullptr;
    std::set<Colors> colors;
    bool has_pt = false;  // a card always has its printed P/T; a head type guard scopes P/T filters
    int power = 0;
    int toughness = 0;
    // A card's characteristic-defining ability counts its P/T, so power and toughness are read
    // through card_power_toughness when asked. A count filters cards, this one among them, so it
    // can't be taken as the characteristics are read.
    bool counted_pt = false;
};

struct PowerToughness {
    int power = 0;
    int toughness = 0;
};

// The P/T of the card `e` where it is, as card_characteristics reads it, with what a
// characteristic-defining ability counts there (CR 604.3: Nethergoyf in a graveyard or a hand).
PowerToughness card_power_toughness(Entity e, const CardData &cd);

// `cd` as printed, on the face `e` shows; with e == 0, the face `cd` on its own.
ObjectCharacteristics printed_characteristics(Entity e, const CardData &cd);

// The card `e` (`cd` its CardData) as a card, wherever it is: printed, but for the color a layer-5
// static sets or adds in any zone (apply_color_statics).
ObjectCharacteristics card_object_characteristics(Entity e, const CardData &cd);

// The battlefield permanent `e`, as the layers left it.
ObjectCharacteristics permanent_characteristics(Entity e, const Permanent &perm);

// The type line of `e` where it is now: Permanent::types on the battlefield (layer 4 writes there,
// CR 613.1d), the face card_characteristics reads anywhere else, a token's own. Empty for an object
// with none of these (a player, an ability).
const std::set<Type> &effective_types(Entity e);

// The abilities `e` has where it is now. A permanent has those of the face it shows, unless an
// effect removed them, and every ability on Permanent::abilities (CR 613.1f); a card anywhere else
// those of card_characteristics, a token its Token's. Keywords are read through
// permanent_has_keyword (queries/keywords.h), in any zone too.
std::vector<const AbilityDef *> object_abilities(Entity e);

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

// The name of the card `cd` as a whole, as a decklist or a sideboard names it: a split card's two
// names joined by " // " (CR 709.4: outside the stack it has both halves' characteristics), any
// other card's name (a double-faced card's is its front face's, CR 712.8a).
std::string whole_card_name(const CardData &cd);

// The names a player may choose when an effect has them choose a card name and they want this card:
// a split card's each half's (CR 709.4a), an adventurer or preparation card's alternative name too
// (CR 715.5, 722.5).
std::vector<std::string> card_names(const CardData &cd);

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

// Mana value of permanent `e` now, from the face mana_value_face picks. A token (no card) has the
// mana value it copied (Token::mana_value, CR 707.2), 0 for a scripted token.
int permanent_mana_value(Entity e);

// Unified "characteristic at the time it is read" accessors (CR 608.2h). Each returns the
// object's effective value: read live from its battlefield components while it is in play
// (so all applied continuous effects/counters are reflected — and, because every effective-P/T
// mutator resyncs the cached Creature P/T synchronously, this is correct even mid-resolution),
// and from the last-known-info snapshot captured at battlefield-leave once it has gone. Falls
// back to the card's printed characteristics for objects that were never on the battlefield
// (e.g. a spell on the stack). Defined in characteristics.cpp (needs cur_game). All
// "read a target's power/toughness/color at resolution" sites route through these.
int effective_power(Entity e);
// The power of the object `ref` names: as it is while it is on the battlefield as that object, else
// as it last existed there (CR 608.2h), however long ago it left.
int object_power(const ObjectRef &ref);
int effective_toughness(Entity e);
std::set<Colors> effective_colors(Entity e);

// Layer 5 (CR 613.1e / 612): apply to `colors` every active color static that reaches `e`, by its
// Affected$ filter and AffectedZone$, a SetColor$ (Mycosynth Lattice) replacing them and an AddColor$
// (Painter's Servant) adding to them, in timestamp order. False if none reaches it. "Colorless" sets
// an empty set (CR 105.2c). Defined in state_manager_statics.cpp (where g_active_statics lives).
// Consulted by effective_colors, the colorless queries and the filter matcher's card view so every
// color-dependent check (protection-from-color targeting, is_colorless, a color qualifier on a card
// in any zone) sees the affected object's effective color. Matches against the object's PRINTED
// characteristics (card_matches_printed) to avoid recursing back into effective_colors; tokens (no
// CardData) are not matched here.
bool apply_color_statics(Entity e, std::set<Colors> &colors);

// True if the object is colorless (CR 105.2c): it has none of the five colors, read from its
// effective colors (so a Devoid card, a colorless token, or anything under a Mycosynth Lattice
// SetColor$ is colorless, and a transformed face reads its own colors).
inline bool is_colorless(Entity e) { return effective_colors(e).empty(); }

// True if `e` is a planeswalker permanent on the battlefield (the form damage/combat care about).
inline bool is_planeswalker_permanent(Entity e) {
    return global_coordinator.entity_has_component<Permanent>(e) &&
           is_planeswalker(global_coordinator.GetComponent<Permanent>(e).types);
}

// Human-readable name for an entity, used in game_log output and action labels. The single
// shared resolver (do not open-code a CardData/Permanent name ternary at a log site):
// a face-down permanent or spell's words for what it is (it has no name, CR 708.2a), a battlefield object's Permanent name (a token is tagged " token"), then the name of the
// card's active face (on the stack, the half or face cast, CR 709.3b), then a lingering Token component (a token that already left the
// battlefield keeps only Token), then a standalone ability entity described via its source
// card ("Sylvan Library's ability") or its effect category, then the last-known name of a
// permanent that left play and ceased to exist (a token, CR 111.7). "<unknown>" only when the
// entity carries no name-bearing component and no last-known information. Defined in
// characteristics.cpp.
std::string entity_name(Entity e);

// Card vocab index of an object or an ability's source (card_vocab.h), by its name as it is now:
// a permanent's current name, a token's token-band index (else TOKEN_SENTINEL), a card's CardData
// name; an ability entity resolves to its source. A source that ceased to exist (a token, CR 111.7)
// is identified by its last-known information. -1 for entity 0 or an object with neither.
int source_vocab_idx(Entity e);

#endif /* QUERIES_CHARACTERISTICS_H */
