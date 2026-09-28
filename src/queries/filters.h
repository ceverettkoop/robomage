#ifndef QUERIES_FILTERS_H
#define QUERIES_FILTERS_H

#include <climits>
#include <set>
#include <string>
#include "../classes/colors.h"
#include "../components/types.h"
#include "../components/zone.h"
#include "../ecs/entity.h"

struct CardData;

// ── Unified filter matcher (CR 109/110/115 characteristic matching) ─────────
// One grammar, one evaluator, two entry points. A Forge filter spec is a ';'-delimited
// list of OR alternatives; each alternative is a head type/subtype name (or a "Card" /
// "Permanent" / "Spell" wildcard) followed by '.'/'+'-joined qualifiers that are ANDed:
// colors (White/nonGreen/Colorless), supertypes (Basic/nonBasic), main-type negations
// (nonLand/nonCreature…), subtypes (Goblin, Plains), power/toughness comparators
// (toughnessLE2), mana-value bounds (cmcLEX / the dynamic ctx bound), control (YouCtrl/
// OppCtrl), token state (token/nonToken), combat/tap state (attacking/blocking/tapped/
// untapped), timing (ThisTurnEntered) and identity (IsRemembered/Other/nonChosenCard).
// Unrecognized qualifiers fail closed and warn once.
//
// `card_matches_filter` reads PRINTED characteristics (a card in hand/library/graveyard
// or a spell on the stack). `permanent_matches_filter` reads a battlefield permanent's
// LIVE characteristics — types and P/T from its components, but color and mana value from
// the card (CR 105/112.7: the engine has no live color/MV layer; effective_colors is the
// single seam). Both share one evaluator, so qualifier coverage can never drift again.
struct MatchCtx {
    Zone::Ownership controller = Zone::UNKNOWN;  // the "you" reference for YouCtrl/OppCtrl
    Entity source = 0;                           // self-exclusion source for .Other
    int cmc_bound = -1;                          // dynamic mana-value bound (Aether Vial); <0 = none
    std::string cmc_op = "";                     // comparator for cmc_bound (EQ/LE/GE/…)
    // Dynamic value of SVar X for a power/toughness-vs-X qualifier (Ensnaring Bridge's
    // Creature.powerGTX): the caller resolves X (e.g. the controller's hand size) and supplies
    // it here. INT_MIN = "no X provided", in which case a powerGTX/-style qualifier fails closed.
    int x_bound = INT_MIN;
    // The CARD object targeted by the resolving ability chain, for the `targetedBy` qualifier
    // (Cloak and Dagger, Entwined's ChangeType$ Card.targetedBy — "the chosen creature" is
    // exactly the creature the DBPump sub targeted). 0 = no card target in context, in which
    // case the qualifier fails closed.
    Entity chain_target = 0;
    // The player the ability's chain targeted (Ability::targeted_player's seat), for the
    // `ControlledBy ParentTarget` qualifier (Cloak and Dagger, Entwined's DBPump: "target
    // creature they control", where "they" is the opponent TrigRevealHand targeted). UNKNOWN =
    // no player targeted yet, in which case the qualifier fails closed for a permanent.
    Zone::Ownership targeted_player = Zone::UNKNOWN;
};

bool card_matches_filter(Entity e, const std::string &spec, const MatchCtx &ctx = MatchCtx{});
bool card_matches_filter(const CardData &cd, const std::string &spec, const MatchCtx &ctx = MatchCtx{});
bool permanent_matches_filter(Entity e, const std::string &spec, const MatchCtx &ctx = MatchCtx{});
// `e` in whatever zone it is in: a battlefield permanent by its live characteristics
// (permanent_matches_filter), any other object by its card's (card_matches_filter).
bool object_matches_filter(Entity e, const std::string &spec, const MatchCtx &ctx = MatchCtx{});

// The object a zone-change event names, matched as CR 603.6a / 603.10a direct: an object that
// left the battlefield (origin == BATTLEFIELD) is matched as it last existed there (its
// last-known information: live types, controller, colors and P/T as it left, CR 608.2h) — this
// also covers a token that has since ceased to exist; an object that entered the battlefield is
// matched by its live characteristics as a permanent (so an animated land counts as a creature,
// a Clue token doesn't, and an opponent's creature you reanimated is YouCtrl), or by its
// last-known information if it has already left again; any other move matches the card's
// characteristics in its new zone. The one matcher behind trigger ValidCard$ filters.
bool zone_change_object_matches(Entity e, Zone::ZoneValue origin, Zone::ZoneValue destination,
                                const std::string &spec, const MatchCtx &ctx = MatchCtx{});

// Filter-spec structure (read at parse time, not for matching an object): true when `token`
// is one whole '.'/'+'/','/';'-delimited token of the spec — "nonCreature" is a token of
// "Card.nonCreature" but "Creature" is not — or, for filter_has_head, the head (type) token of
// one of its OR alternatives ("Creature" in "Creature.Other+YouCtrl").
bool filter_names_token(const std::string &spec, const std::string &token);
bool filter_has_head(const std::string &spec, const std::string &head);

// A filter spec for cards outside the battlefield and the stack, where a card's controller is its
// owner (CR 108.4a): every YouCtrl / OppCtrl token becomes YouOwn / OppOwn.
std::string owner_relative_filter(const std::string &spec);

// ── Forge-style comma-OR filter matching (one convention, one place) ────────
// A Forge `Valid$`/`ValidTgts$` spec lists its OR alternatives separated by ',' (e.g.
// Mox Amber's "Creature.Legendary+YouCtrl,Planeswalker.Legendary+YouCtrl"), whereas the
// single-clause matchers above use ';' as their internal OR delimiter. These wrappers split
// the comma-joined spec into its alternatives and OR the per-clause matcher over them, so a
// caller hands a Forge comma-OR spec straight through and gets correct OR semantics WITHOUT
// hand-normalizing ',' → ';' (or hand-splitting) at the call site. ',' is not meaningful
// inside a single clause (the clause grammar joins qualifiers with '.'/'+'), so splitting on
// ',' is unambiguous. Empty alternatives are skipped; an all-empty/empty spec matches nothing.
bool permanent_matches_any(Entity e, const std::string &comma_or_spec,
                           const MatchCtx &ctx = MatchCtx{});
bool card_matches_any(Entity e, const std::string &comma_or_spec,
                      const MatchCtx &ctx = MatchCtx{});

// Pull a STATIC numeric mana-value qualifier out of a filter spec into a MatchCtx
// (e.g. "Card.Colorless+cmcGE7" → ctx.cmc_op = "GE", ctx.cmc_bound = 7). The qualifier
// evaluator defers `cmcLE3`/`cmcGE7`/… to ctx.cmc_bound (returning true for the bare token),
// so a caller that matches a static cmc filter through card_matches_filter must seed the bound
// here or the comparator is silently ignored. Non-numeric forms (cmcLEX / cmcEQX, keyed off X
// paid) are left to the inline evaluator and not extracted. Mirrors the per-token extraction in
// svar_eval.cpp; shared so static-cmc filter sites (ReduceCost, …) cannot drift on the parsing.
void extract_static_cmc_bound(const std::string &spec, MatchCtx &ctx);

// Enforce a positive color target restriction (e.g. ValidTgts$ Permanent.Blue on Red Elemental
// Blast: "target blue permanent", CR 115.1) against an already-resolved color set. Sharing the
// color set (rather than re-reading printed colors) is what lets battlefield/last-known callers
// honour effective color uniformly.
bool color_set_passes(const std::string &vt, const std::set<Colors> &colors);

// Enforce a "non<Color>" target restriction (e.g. ValidTgts$ Creature.nonBlack on Snuff Out)
// against an already-resolved color set: reject when the candidate is one of the excluded
// colors. Counterpart to color_set_passes; routed through effective_colors for the same reason.
bool color_set_passes_noncolor(const std::string &vt, const std::set<Colors> &colors);

// Enforce any "non<CardType>" main-type negation in a filter/target spec (e.g.
// ValidTgts$ Permanent.nonLand+cmcLE3 on Abrupt Decay, or "nonCreature"/"nonArtifact"
// on other cards) against a permanent's live type list. Returns false when the type
// list carries a card type the spec excludes (CR 115.1: target restrictions are checked
// against the candidate's characteristics). General over the permanent card types
// (CR 110.4a) plus the spell-only types, so it is not special-cased to any one card.
// `types` is the permanent's (or card's) full Type list; only kind == TYPE entries are
// considered for the negation. The substring scan keys on "non" + the type name, matched
// ASCII case-insensitively (Forge scripts vary the casing — "nonland" vs "nonLand"),
// so it is safe alongside other '.'/'+' qualifiers in the same spec string.
bool type_set_passes_nontype(const std::string &spec, const std::set<Type> &types);

// An Aura whose Enchant restriction names "inZoneGraveyard" (K:Enchant:Creature.inZoneGraveyard,
// Animate Dead) enchants a creature CARD sitting in a graveyard rather than a battlefield
// permanent (CR 303.4). The aura's cast-time target search must therefore look in graveyards, not
// on the battlefield: setting Ability::target_in_graveyard routes both build_valid_targets and
// is_legal_target through their graveyard branches. General over any "enchant a card in a
// graveyard" aura, keyed on the filter token, not on a specific card.
inline bool enchant_targets_graveyard(const std::string &enchant_filter) {
    return enchant_filter.find("inZoneGraveyard") != std::string::npos;
}

#endif /* QUERIES_FILTERS_H */
