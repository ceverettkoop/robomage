#ifndef QUERIES_FILTERS_H
#define QUERIES_FILTERS_H

#include <climits>
#include <set>
#include <string>
#include <vector>
#include "../classes/colors.h"
#include "../components/filter_spec.h"
#include "../components/types.h"
#include "../components/zone.h"
#include "../ecs/entity.h"

struct CardData;
struct ResolveCtx;

// ── Unified filter matcher (CR 109/110/115 characteristic matching) ─────────
// One grammar, one evaluator, two entry points. A Forge filter spec is a ';'-delimited
// list of OR alternatives; each alternative is a head type/subtype name (or a "Card" /
// "Permanent" / "Spell" wildcard) followed by '.'/'+'-joined qualifiers that are ANDed:
// colors (White/nonGreen/Colorless), supertypes (Basic/nonBasic), main-type negations
// (nonLand/nonCreature…), subtypes (Goblin, Plains), power/toughness comparators
// (toughnessLE2), mana-value bounds (cmcLEX / the dynamic ctx bound), control (YouCtrl/
// OppCtrl), token state (token/nonToken), combat/tap state (attacking/blocking/tapped/
// untapped), timing (ThisTurnEntered) and identity (IsRemembered/Other/nonChosenCard).
// Unrecognized qualifiers fail closed, negated ones too, and warn once.
//
// Every qualifier is one row of the table in filters.cpp: a name or prefix, a parser for what a
// prefix carries, and an evaluator of one of three shapes. Object-local reads the object alone;
// set-relative reads it against the context (the "you" and "this" of MatchCtx, or the row's
// ObjectSpec resolved against the ResolveCtx); population-relative is decided over a candidate set,
// through select_matching. A spec is compiled through the table once (FilterSpec), --parse-report
// reads the same rows, and nothing here re-reads a spec's text to match an object.
//
// `card_matches_filter` reads PRINTED characteristics (a card in hand/library/graveyard
// or a spell on the stack), but for the color a layer-5 static sets in any zone (Mycosynth
// Lattice: apply_color_statics). `card_matches_printed` reads them with no such exception, for
// that static's own Affected$ filter. `permanent_matches_filter` reads a battlefield permanent's
// LIVE characteristics — types and P/T from its components, but color and mana value from
// the card (CR 105/112.7: the engine has no live color/MV layer; effective_colors is the
// single seam). Both share one evaluator, so qualifier coverage can never drift again.
struct MatchCtx {
    Zone::Ownership controller = Zone::UNKNOWN;  // the "you" reference for YouCtrl/OppCtrl
    Entity source = 0;                           // self-exclusion source for .Other
    // Value of the SVar a cmc<OP><SVar> qualifier names (Aether Vial's cmcEQX), which only the
    // caller can resolve; <0 = none. The qualifier compares it by its own operator, and passes with
    // none. A cmc<OP><n> qualifier needs none: the matcher compares its literal.
    int cmc_bound = -1;
    // Dynamic value of SVar X for a power/toughness-vs-X qualifier (Ensnaring Bridge's
    // Creature.powerGTX): the caller resolves X (e.g. the controller's hand size) and supplies
    // it here. INT_MIN = "no X provided", in which case a powerGTX/-style qualifier fails closed.
    int x_bound = INT_MIN;
    // What a set-relative qualifier resolves its ObjectSpec against when the set needs the ability
    // being read (its chosen targets: `targetedBy`, Cloak and Dagger, Entwined's "the chosen
    // creature"; `ControlledBy ParentTarget`, its DBPump's "target creature they control", where
    // "they" is the opponent TrigRevealHand targeted). Null: only `controller` and `source`, and
    // such a qualifier fails closed.
    const ResolveCtx *resolve = nullptr;
    // The zone a spell is being cast from, for a cost static's wasCastFromYourHand: the spell isn't
    // cast yet, so only the caller knows.
    Zone::ZoneValue cast_from = Zone::HAND;
};

bool card_matches_filter(Entity e, const FilterSpec &spec, const MatchCtx &ctx = MatchCtx{});
bool card_matches_filter(const CardData &cd, const FilterSpec &spec, const MatchCtx &ctx = MatchCtx{});
// `e` where it is, with `face`'s characteristics: the spell a card would be as it is cast (CR 601.3e).
bool card_as_face_matches_filter(Entity e, const CardData &face, const FilterSpec &spec, const MatchCtx &ctx = MatchCtx{});
bool card_matches_printed(Entity e, const FilterSpec &spec, const MatchCtx &ctx = MatchCtx{});
bool permanent_matches_filter(Entity e, const FilterSpec &spec, const MatchCtx &ctx = MatchCtx{});
// `e` in whatever zone it is in: a battlefield permanent by its live characteristics
// (permanent_matches_filter), any other object by its card's (card_matches_filter).
bool object_matches_filter(Entity e, const FilterSpec &spec, const MatchCtx &ctx = MatchCtx{});

// The `candidates` matching `spec`, in order, each read as object_matches_filter reads it. The call
// shape of a population-relative qualifier, which compares a candidate with the others: an
// alternative's other qualifiers narrow the candidates first, then its population qualifiers narrow
// what is left, and the alternatives are ORed. A population qualifier has no reading for one object
// alone, so the single-object matchers above fail it closed.
std::vector<Entity> select_matching(const std::vector<Entity> &candidates, const FilterSpec &spec,
                                    const MatchCtx &ctx = MatchCtx{});

// The object a zone-change event names, matched as CR 603.6a / 603.10a direct: an object that
// left the battlefield (origin == BATTLEFIELD) is matched as it last existed there (its
// last-known information: live types, controller, colors and P/T as it left, CR 608.2h) — this
// also covers a token that has since ceased to exist; an object that entered the battlefield is
// matched by its live characteristics as a permanent (so an animated land counts as a creature,
// a Clue token doesn't, and an opponent's creature you reanimated is YouCtrl), or by its
// last-known information if it has already left again; any other move matches the card's
// characteristics in its new zone. The one matcher behind trigger ValidCard$ filters.
bool zone_change_object_matches(Entity e, Zone::ZoneValue origin, Zone::ZoneValue destination,
                                const FilterSpec &spec, const MatchCtx &ctx = MatchCtx{});

// The tokens of filter spec `spec` (its ','/';' alternatives) the matcher has no reading for: a
// head that is no card type, subtype or supertype (nor Card / Permanent / Spell / Self / Other), a
// lowercase qualifier with no row in the table (it fails closed and warns at runtime), a PascalCase
// qualifier that is no type name (read as a type-line name no object has), unless its row says the
// code the filter reaches reads it. Reads no game state (--parse-report, parse_report.h).
std::vector<std::string> filter_unknown_qualifiers(const std::string &spec);

// A "cmc<OP><bound>" qualifier of a filter (read at parse time, not for matching an object).
struct FilterCmcBound {
    std::string token;        // as the script spells it
    std::string op;           // its two-letter comparator
    std::string bound;        // its literal, or the name of the SVar it reads
    bool names_svar = false;  // no literal: its reader supplies the value (MatchCtx::cmc_bound)
};

// Every cmc qualifier of `spec` with a bound, in order.
std::vector<FilterCmcBound> filter_cmc_bounds(const FilterSpec &spec);

// Filter-spec structure (read at parse time, not for matching an object): true when `token`
// is one whole '.'/'+'/','/';'-delimited token of the spec — "nonCreature" is a token of
// "Card.nonCreature" but "Creature" is not — or, for filter_has_head, the head (type) token of
// one of its OR alternatives ("Creature" in "Creature.Other+YouCtrl").
bool filter_names_token(const std::string &spec, const std::string &token);
bool filter_has_head(const std::string &spec, const std::string &head);
// Whether each of `spec`'s OR alternatives has `token` among its tokens.
bool filter_every_alternative_names(const std::string &spec, const std::string &token);

// A filter spec for cards outside the battlefield and the stack, where a card's controller is its
// owner (CR 108.4a): every YouCtrl / OppCtrl token becomes YouOwn / OppOwn.
FilterSpec owner_relative_filter(const FilterSpec &spec);

// A ValidTgts$ spec in the matcher's grammar: "YouDontCtrl" is Forge's spelling of OppCtrl (a
// permanent you don't control), and "Any" as a permanent target is "any creature or planeswalker"
// (CR 115.4 / 306.7).
FilterSpec target_filter(const FilterSpec &valid_tgts);

// Enforce a positive color target restriction (e.g. ValidTgts$ Permanent.Blue on Red Elemental
// Blast: "target blue permanent", CR 115.1) against an already-resolved color set. Sharing the
// color set (rather than re-reading printed colors) is what lets battlefield/last-known callers
// honour effective color uniformly.
bool color_set_passes(const std::string &vt, const std::set<Colors> &colors);

// An Aura whose Enchant restriction names "inZoneGraveyard" (K:Enchant:Creature.inZoneGraveyard,
// Animate Dead) enchants a creature CARD sitting in a graveyard rather than a battlefield
// permanent (CR 303.4). The aura's cast-time target search must therefore look in graveyards, not
// on the battlefield: a graveyard target zone routes both build_valid_targets and
// is_legal_target through their graveyard branches. General over any "enchant a card in a
// graveyard" aura, keyed on the filter token, not on a specific card.
inline bool enchant_targets_graveyard(const std::string &enchant_filter) {
    return enchant_filter.find("inZoneGraveyard") != std::string::npos;
}

#endif /* QUERIES_FILTERS_H */
