#ifndef GAME_QUERIES_H
#define GAME_QUERIES_H

#include <cctype>
#include <climits>
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
#include "str_util.h"

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
    // Hybrid pips carry color too (CR 105.2/202.3f): {W/U} makes the card both white and blue,
    // {2/W} makes it white. hybrid_mana is kept out of mana_cost, so fold its colors in here.
    for (const auto &pip : cd.hybrid_mana)
        for (Colors c : pip.colors)
            if (c != COLORLESS && c != GENERIC) result.insert(c);
    // Phyrexian pips carry color too (CR 202.2d): {B/P} makes the card black no matter how the
    // cost is actually paid (mana or life). phyrexian_mana is kept out of mana_cost like hybrid,
    // so fold its colors in here — otherwise a Phyrexian-only cost (Surgical Extraction, {B/P}
    // {B/P}) would read as colorless.
    for (Colors c : cd.phyrexian_mana)
        if (c != COLORLESS && c != GENERIC) result.insert(c);
    return result;
}

// Mana value (converted mana cost, CR 202.3) of a card: one per non-hybrid pip in mana_cost
// plus each hybrid pip's contribution (1 for a color hybrid, N for an {N/color} twobrid —
// CR 202.3f: a hybrid symbol's MV is the greatest of its component symbols' MVs). X counts 0
// outside the stack/cast (CR 202.3b). Each Phyrexian pip ({B/P}) counts 1 toward mana value
// (CR 202.3f: a Phyrexian symbol is worth its color component, MV 1) regardless of whether the
// cost is actually paid with mana or life; phyrexian_mana is kept out of mana_cost (mirroring
// hybrid_mana), so fold it in here. Single source for "card CMC".
inline int card_mana_value(const CardData &cd) {
    int mv = static_cast<int>(cd.mana_cost.size());
    for (const auto &pip : cd.hybrid_mana) mv += pip.mana_value;
    mv += static_cast<int>(cd.phyrexian_mana.size());
    return mv;
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
inline bool type_set_passes_nontype(const std::string &spec, const std::set<Type> &types) {
    static const char *kCardTypes[] = {
        "Land", "Creature", "Artifact", "Enchantment", "Planeswalker",
        "Battle", "Instant", "Sorcery", "Tribal"};
    const std::string spec_lc = ascii_lower(spec);
    for (const char *ct : kCardTypes) {
        std::string tok = ascii_lower(std::string("non") + ct);
        if (spec_lc.find(tok) == std::string::npos) continue;
        for (const auto &t : types)
            if (t.kind == TYPE && t.name == ct) return false;
    }
    return true;
}

// An Aura whose Enchant restriction names "inZoneGraveyard" (K:Enchant:Creature.inZoneGraveyard,
// Animate Dead) enchants a creature CARD sitting in a graveyard rather than a battlefield
// permanent (CR 303.4). The aura's cast-time target search must therefore look in graveyards, not
// on the battlefield: setting Ability::target_in_graveyard routes both build_valid_targets and
// is_legal_target through their graveyard branches. General over any "enchant a card in a
// graveyard" aura, keyed on the filter token, not on a specific card.
inline bool enchant_targets_graveyard(const std::string &enchant_filter) {
    return enchant_filter.find("inZoneGraveyard") != std::string::npos;
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

// Human-readable name for an entity, used in game_log output and action labels. The single
// shared resolver (do not open-code a CardData/Permanent name ternary at a log site):
// a battlefield object's Permanent name (a token is tagged " token"), then the card's
// printed CardData name, then a lingering Token component (a token that already left the
// battlefield keeps only Token), then a standalone ability entity described via its source
// card ("Sylvan Library's ability") or its effect category. "<unknown>" only when the
// entity carries no name-bearing component at all. Defined in game_queries.cpp.
std::string entity_name(Entity e);

// Strip the battlefield-state components (Permanent/Creature/Damage) from a card that is no
// longer on the battlefield — clearing its equipment/aura attachment links first so no dangling
// reference survives (CR 704.5n). Shared by the state-based off-battlefield strip
// (apply_permanent_components) and by add_to_zone's battlefield-entry reset: a card that left
// and returned within a single resolution (same-resolution flicker, Ajani's exile-and-return
// transform) re-enters before the state-based pass could strip it, and per CR 400.7 the
// returning card is a NEW object that must not keep its stale tapped/summoning-sickness/
// counter/attachment state. Defined in game_queries.cpp.
void strip_permanent_components(Entity entity);

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
};

bool card_matches_filter(Entity e, const std::string &spec, const MatchCtx &ctx = MatchCtx{});
bool card_matches_filter(const CardData &cd, const std::string &spec, const MatchCtx &ctx = MatchCtx{});
bool permanent_matches_filter(Entity e, const std::string &spec, const MatchCtx &ctx = MatchCtx{});

// Count the battlefield permanents matching a Forge `Count$Valid <filter>` spec — the single
// shared implementation behind both the spell/ability dynamic-amount path (evaluate_dynamic_amount)
// and the static-buff svar path (evaluate_sa_svar), so the `controller` ("you") reference, the
// `source` reference (for source-relative qualifiers like +Other / sameName), and the
// battlefield/phasing guard are identical on both. `filter_spec` is the bare filter (the text after
// "Count$Valid "); control/type/etc. qualifiers in it are enforced by permanent_matches_filter.
int count_battlefield_matching(const std::string &filter_spec, Zone::Ownership controller,
                               Entity source);

// ── Forge-style comma-OR filter matching (one convention, one place) ────────
// A Forge `Valid$`/`ValidTgts$` spec lists its OR alternatives separated by ',' (e.g.
// Mox Amber's "Creature.Legendary+YouCtrl,Planeswalker.Legendary+YouCtrl"), whereas the
// single-clause matchers above use ';' as their internal OR delimiter. These wrappers split
// the comma-joined spec into its alternatives and OR the per-clause matcher over them, so a
// caller hands a Forge comma-OR spec straight through and gets correct OR semantics WITHOUT
// hand-normalizing ',' → ';' (or hand-splitting) at the call site. ',' is not meaningful
// inside a single clause (the clause grammar joins qualifiers with '.'/'+'), so splitting on
// ',' is unambiguous. Empty alternatives are skipped; an all-empty/empty spec matches nothing.
inline bool permanent_matches_any(Entity e, const std::string &comma_or_spec,
                                  const MatchCtx &ctx = MatchCtx{}) {
    for (const auto &alt : split(comma_or_spec, ',', /*skip_empty=*/true))
        if (permanent_matches_filter(e, alt, ctx)) return true;
    return false;
}
inline bool card_matches_any(Entity e, const std::string &comma_or_spec,
                             const MatchCtx &ctx = MatchCtx{}) {
    for (const auto &alt : split(comma_or_spec, ',', /*skip_empty=*/true))
        if (card_matches_filter(e, alt, ctx)) return true;
    return false;
}

// Pull a STATIC numeric mana-value qualifier out of a filter spec into a MatchCtx
// (e.g. "Card.Colorless+cmcGE7" → ctx.cmc_op = "GE", ctx.cmc_bound = 7). The qualifier
// evaluator defers `cmcLE3`/`cmcGE7`/… to ctx.cmc_bound (returning true for the bare token),
// so a caller that matches a static cmc filter through card_matches_filter must seed the bound
// here or the comparator is silently ignored. Non-numeric forms (cmcLEX / cmcEQX, keyed off X
// paid) are left to the inline evaluator and not extracted. Mirrors the per-token extraction in
// svar_eval.cpp; shared so static-cmc filter sites (ReduceCost, …) cannot drift on the parsing.
inline void extract_static_cmc_bound(const std::string &spec, MatchCtx &ctx) {
    size_t p = 0;
    while (p < spec.size()) {
        size_t pos = spec.find("cmc", p);
        if (pos == std::string::npos) return;
        // A numeric cmc qualifier is "cmc" + 2-letter op + at least one digit (cmcGE7).
        if (pos + 5 < spec.size() &&
            std::isdigit(static_cast<unsigned char>(spec[pos + 5]))) {
            size_t e = pos + 5;
            while (e < spec.size() && std::isdigit(static_cast<unsigned char>(spec[e]))) ++e;
            ctx.cmc_op = spec.substr(pos + 3, 2);
            ctx.cmc_bound = std::stoi(spec.substr(pos + 5, e - (pos + 5)));
            return;
        }
        p = pos + 3;
    }
}

// ── Defined$ player resolution (CR 109.5 / 608.2g) ──────────────────────────
// Who controls an ability's SOURCE object: its live Permanent.controller while on the
// battlefield, else the owner of its current zone. The "source controller" idiom that
// token/amass/mobilize/delayed-trigger/deal-damage/etc. otherwise repeat inline.
Zone::Ownership source_controller(Entity source);

// CR 702.16: is `player_entity` currently under a "protection from everything" grant
// (cur_game.player_protection_from_everything)? Protection from everything is protection from ALL
// sources — including the protected player's OWN sources — so this returns true whenever the grant
// is active for that player, regardless of who controls `source`. True means damage from `source`
// to that player is prevented. Shared by the effect-damage chokepoint (deal_damage_to_player) and
// the combat-damage path so the prevention rule lives in one place.
bool player_protected_from_source(Entity player_entity, Entity source);

// CR 702.16d (damage facet): is `perm_target` a permanent with "protection from colored spells"
// (Emrakul) being dealt damage by a SOURCE that is a colored spell? True means that damage is
// prevented. Tightly gated: the source must be a spell object (entity_has_component<Spell>) — so
// combat damage (creature source) and ability damage are never prevented — and one or more colors
// (a colorless spell's damage is not prevented). Reuses has_protection_from_colored_spells and
// effective_colors so the targeting block and the damage block share one definition of the rule.
bool permanent_protected_from_colored_spell_source(Entity perm_target, Entity source);

// Last-known controller of an object for a "that permanent's controller" effect
// (CR 608.2g/h): its Zone.controller while it still records one, else the live
// Permanent.controller (mid-resolution before the SBA strips it), else the controller
// captured in its last-known info as it left the battlefield. UNKNOWN if never controlled.
Zone::Ownership last_known_controller(Entity e);

// Resolve the Defined$ player an ability designates, or UNKNOWN when it names none (the
// caller then falls back to its own chosen target):
//   Defined$ You                -> the source's controller
//   Defined$ Player.Opponent    -> that controller's single opponent (2-player; CR 109.5)
//   Defined$ TargetedController -> the last-known controller of ab.target
//   Defined$ TriggeredActivator -> the player bound when the trigger fired
Zone::Ownership resolve_defined_player(const Ability &ab);

// True if a turn-long "can't gain life" prohibition (CR 119.x) currently applies to this player
// (Roiling Vortex's {R} ability, cur_game.cant_gain_life_this_turn). Defined out-of-line in
// game_queries.cpp (needs cur_game). Consulted by player_gain_life so every life-gain site obeys
// the prohibition.
bool player_cant_gain_life(Entity player_entity);

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
    // A hybrid pip ({W/U}, {2/W}) carries color, so a card with one is not colorless (unless a
    // Colors: override said so, handled above). Only consulted when there is no explicit override.
    if (cd.explicit_colors.empty())
        for (const auto &pip : cd.hybrid_mana)
            for (Colors c : pip.colors)
                if (c != COLORLESS && c != GENERIC) return false;
    // A Phyrexian pip carries color the same way (CR 202.2d): {B/P} makes the card black even
    // when paid with life, so it is never colorless (absent a Colors: override, handled above).
    if (cd.explicit_colors.empty())
        for (Colors c : cd.phyrexian_mana)
            if (c != COLORLESS && c != GENERIC) return false;
    return true;
}

// True if the entity is colorless (CR 105.2c), handling both real cards (CardData) and
// tokens (Token, which have no mana cost — their color is the token's color indicator).
inline bool is_colorless_entity(Entity e) {
    // A global SetColor$ override (Mycosynth Lattice) decides colorlessness first (CR 613.1e).
    std::set<Colors> override_colors;
    if (setcolor_override_for(e, override_colors)) return override_colors.empty();
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

// True if a counter type names a keyword ability, so a counter of that type grants the
// keyword to its permanent (CR 122.1d: keyword counters). The names match the keyword
// strings the engine checks via creature_has_keyword (e.g. "Flying", "Trample"). Single
// source so the keyword-counter rebuild and any future "remove keyword counter" share it.
inline bool is_keyword_counter_type(const std::string &type) {
    static const std::set<std::string> kKeywordCounters = {
        "Flying", "First Strike", "Double Strike", "Deathtouch", "Haste", "Hexproof",
        "Indestructible", "Lifelink", "Menace", "Reach", "Trample", "Vigilance",
        "Shadow", "Skulk", "Fear", "Intimidate", "Horsemanship", "Infect", "Wither",
        "Toxic", "Defender", "Flash", "Persist", "Undying", "Decayed"};
    return kKeywordCounters.count(type) != 0;
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
// is the phasing subsystem itself (the untap-step phase-in/skip in game.cpp), the
// rare loop that must still process phased-out permanents (e.g. resetting their cached
// P/T before skipping them when gathering static abilities), and the ML serialization
// (machine_io.cpp's populate_gamestate), which deliberately INCLUDES phased-out
// permanents in the observation with an is_phased_out flag instead of hiding them.
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

// CR 702.131b: Ascend on a permanent is a static ability — "ANY TIME you control ten or
// more permanents and you don't have the city's blessing, you get the city's blessing for
// the rest of the game" (a one-way latch, never lost once gained, 702.131c). Because it
// applies "any time", the grant must be visible IMMEDIATELY when the tenth permanent
// arrives — in particular mid-resolution, between a Token sub-ability creating the 10th
// permanent and a later Condition$ Blessing gate reading the flag (Ocelot Pride's copy
// clause) — not only at the next state-based pass. Re-evaluates and latches the blessing
// for both players. The SBA preamble runs it every pass; any code that READS the blessing
// flag after possibly changing the permanent count should call it first. Pass the
// iterating system's mEntities (or orderer->mEntities). Defined in game_queries.cpp
// (needs cur_game's player entities and game_log).
void refresh_city_blessing(const std::set<Entity> &entities);

// The most recently exiled card linked to `host` (Permanent::exiled_with) that STILL has a live
// "return it from exile" path scheduled in cur_game.delayed_triggers, or 0 if none. This
// distinguishes an exile-with-return host (a Static Prison holding a real Murktide) from a
// permanent exiler with no return (Skyclave Apparition) or one holding a token. Two return
// shapes exist (see effect_change_zone.cpp / effect_delayed_trigger.cpp):
//   (A) "exile until host leaves" (Static Prison, Sheltered by Ghosts): a fire_on_leave_battlefield
//       trigger watching `host`, whose ChangeZone fire ability moves the card EXILE -> its origin,
//       with the card in the fire ability's restore_remembered_exiled_with.
//   (B) end-of-turn blink (Flickerwisp / Phelia): a phase-based trigger whose ChangeZone fire
//       ability moves the card EXILE -> battlefield, with the card in the trigger's
//       remembered_objects (also mirrored into restore_remembered_exiled_with).
// Skyclave Apparition / Keen-Eyed Curator populate exiled_with but register NO such trigger, so
// they return 0. Conservative: any ChangeZone delayed trigger that would move the card OUT of exile
// (origin EXILE, destination != EXILE) and references that card counts as a return path. Defined in
// game_queries.cpp (needs cur_game.delayed_triggers).
Entity returnable_exiled_card(Entity host);

// The card `source` exiled and still tracks via Permanent::exiled_with — the association a Saga
// records at chapter I so its later chapters can act on "the card exiled with this" (Defined$
// ExiledWith / ExiledWith$CardManaCost, The Creation of Avacyn). Returns the most-recently exiled
// entry that still has a live Zone (skipping any that have since left the game), or 0 if none.
inline Entity exiled_with_card(Entity source) {
    if (source == 0 || !global_coordinator.entity_has_component<Permanent>(source)) return 0;
    const auto &ew = global_coordinator.GetComponent<Permanent>(source).exiled_with;
    for (auto it = ew.rbegin(); it != ew.rend(); ++it)
        if (global_coordinator.entity_has_component<Zone>(*it)) return *it;
    return 0;
}

// Unblocked attackers controlled by `ctrl` (CR 509.1h): battlefield creatures that are
// attacking and were not blocked at declare-blockers. Used to gate and pay Ninjutsu
// (CR 702.49e) — the offer requires one, and activating returns one to hand.
inline std::vector<Entity> unblocked_attackers(
    const std::set<Entity> &entities, Zone::Ownership ctrl) {
    std::vector<Entity> out;
    for (auto e : entities) {
        if (!is_battlefield_permanent(e, ctrl)) continue;
        if (!global_coordinator.entity_has_component<Creature>(e)) continue;
        auto &cr = global_coordinator.GetComponent<Creature>(e);
        if (cr.is_attacking && !cr.is_blocked) out.push_back(e);
    }
    return out;
}

// True if the creature carries the given keyword string (exact match).
inline bool creature_has_keyword(const Creature &cr, const char *kw) {
    for (const auto &k : cr.keywords)
        if (k == kw) return true;
    return false;
}

// True if keyword `kw` is currently SUPPRESSED on permanent `e` by an until-end-of-turn
// "loses <keyword>" effect (AB$ AnimateAll | RemoveKeywords$, Shadowspear — CR 613 layer 6).
// While suppressed the keyword is treated as absent regardless of how it was granted (printed,
// counter, continuous). Single gate consulted by every effective-keyword accessor below so the
// removal applies uniformly to creatures and noncreature permanents. Cleared at cleanup (514.2).
inline bool keyword_removed_eot(Entity e, const char *kw) {
    if (!global_coordinator.entity_has_component<Permanent>(e)) return false;
    const auto &removed = global_coordinator.GetComponent<Permanent>(e).removed_keywords_eot;
    return removed.find(kw) != removed.end();
}

// True if the permanent `e` is indestructible (CR 702.12b: it can't be destroyed —
// "destroy" effects don't destroy it, and it ignores the lethal-damage / deathtouch
// state-based actions, CR 704.5g/h). Reads the keyword from the object's effective
// keyword list: for a creature that is `Creature::keywords` (rebuilt each static pass
// from the printed list plus any granted keywords), otherwise the printed keywords on
// the CardData (or Token) — so a non-creature permanent like an artifact land with
// `K:Indestructible` is covered too. Indestructible does NOT prevent sacrifice, exile,
// "put into graveyard", or the 0-toughness SBA (CR 704.5f); those callers do not consult
// this. Single source shared by the Destroy effects and the lethal-damage SBA.
inline bool is_indestructible(Entity e) {
    if (keyword_removed_eot(e, "Indestructible")) return false;
    if (global_coordinator.entity_has_component<Creature>(e)) {
        return creature_has_keyword(global_coordinator.GetComponent<Creature>(e), "Indestructible");
    }
    if (global_coordinator.entity_has_component<CardData>(e)) {
        for (const auto &k : global_coordinator.GetComponent<CardData>(e).keywords)
            if (k == "Indestructible") return true;
        return false;
    }
    if (global_coordinator.entity_has_component<Token>(e)) {
        for (const auto &k : global_coordinator.GetComponent<Token>(e).keywords)
            if (k == "Indestructible") return true;
    }
    return false;
}

// True if the permanent `e` currently has the keyword `kw`, reading its EFFECTIVE
// keyword list the same way is_indestructible does: a creature's `Creature::keywords`
// (rebuilt each static pass from the printed list plus any granted keywords — Pump
// grants, continuous effects, keyword counters), otherwise the printed CardData (or
// Token) keywords. Single source for "does this permanent currently have keyword K";
// targeting (Shroud/Hexproof, CR 702.18/702.11) and any future keyword query share it
// so they cannot drift on how a granted keyword is stored.
inline bool permanent_has_keyword(Entity e, const char *kw) {
    if (keyword_removed_eot(e, kw)) return false;
    if (global_coordinator.entity_has_component<Creature>(e))
        return creature_has_keyword(global_coordinator.GetComponent<Creature>(e), kw);
    if (global_coordinator.entity_has_component<CardData>(e)) {
        for (const auto &k : global_coordinator.GetComponent<CardData>(e).keywords)
            if (k == kw) return true;
        return false;
    }
    if (global_coordinator.entity_has_component<Token>(e)) {
        for (const auto &k : global_coordinator.GetComponent<Token>(e).keywords)
            if (k == kw) return true;
    }
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

// True if the spell `spell` (an entity on the stack) can't be countered because some live
// battlefield permanent has a continuous "spells … can't be countered" replacement that covers
// it (CR 614.13/CantHappen) — e.g. Hexing Squelcher's "Spells you control can't be countered."
// Scans battlefield permanents for a battlefield-scoped CANT_BE_COUNTERED replacement and tests
// the spell against its ValidSA$ filter. The controller scope (YouCtrl/OppCtrl) is read from the
// spell's caster relative to the replacement source's controller, because a stack spell has no
// Permanent controller for the generic filter matcher to read (its YouCtrl token is a no-op off
// the battlefield). Consulted at counter-resolution time; reusable by any future can't-be-countered
// permanent. `entities` is the iterating system's mEntities (e.g. orderer->mEntities).
inline bool spell_uncounterable_by_static(Entity spell, const std::set<Entity> &entities) {
    if (!global_coordinator.entity_has_component<CardData>(spell)) return false;
    Zone::Ownership spell_ctrl = global_coordinator.entity_has_component<Spell>(spell)
                                     ? global_coordinator.GetComponent<Spell>(spell).caster
                                     : Zone::UNKNOWN;
    for (auto e : battlefield_permanents(entities)) {
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        Zone::Ownership perm_ctrl = global_coordinator.GetComponent<Permanent>(e).controller;
        for (const auto &r : global_coordinator.GetComponent<CardData>(e).replacement_effects) {
            if (r.kind != Effect::Replacement::CANT_BE_COUNTERED || !r.from_battlefield) continue;
            // Controller scope is read from the spell's caster (the matcher can't read it for a
            // stack object). YouCtrl → caster is the source's controller; OppCtrl → it isn't.
            if (r.valid_sa_filter.find("YouCtrl") != std::string::npos) {
                if (spell_ctrl != perm_ctrl) continue;
            } else if (r.valid_sa_filter.find("OppCtrl") != std::string::npos) {
                if (spell_ctrl == perm_ctrl || spell_ctrl == Zone::UNKNOWN) continue;
            }
            // Any remaining type/characteristic qualifiers on the filter (head "Spell", color,
            // type) are checked against the spell's printed characteristics.
            MatchCtx ctx;
            ctx.controller = perm_ctrl;
            if (card_matches_filter(spell, r.valid_sa_filter, ctx)) return true;
        }
    }
    return false;
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

// Battlefield permanents controlled by `player` matching the ';'-delimited `spec` used by
// Sacrifice-a-<type> / Return-a-<type> activation costs (e.g. "Forest;Plains", "Creature",
// "Creature.Other", "Creature.Green"). Drives both Sacrifice-a-<type> and Return-a-<type>
// activation costs: non-empty == the cost is payable (legality), and the list itself is the
// player's choice menu (payment). `exclude_entity` is the cost's source, honoured by a
// `.Other` qualifier in the spec (e.g. Wight of the Reliquary's "Sacrifice another creature").
// Matching runs through the shared permanent_matches_filter so the full qualifier grammar
// (colors, P/T, subtypes, …) is available here too.
inline std::vector<Entity> controlled_permanents_matching(
    Zone::Ownership player, const std::string &spec, const std::set<Entity> &entities,
    Entity exclude_entity = 0) {
    std::vector<Entity> out;
    MatchCtx ctx;
    ctx.controller = player;
    ctx.source = exclude_entity;
    for (auto e : entities) {
        if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location != Zone::BATTLEFIELD) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.controller != player) continue;
        if (permanent_matches_filter(e, spec, ctx)) out.push_back(e);
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

// True if the spell's SPELL ability carries a VARIABLE life cost (Cost$ ... PayLife<X>): the
// amount of life paid IS the spell's X (Count$xPaid), chosen as an additional cost while casting
// (Toxic Deluge). Distinct from a fixed PayLife<N>, which is paid as a flat life cost. Reading it
// off the SPELL ability keeps the parser's real Cost$ tag authoritative (no retag); the cast path
// prompts for X, sets cur_game.x_paid, and pays that much life.
inline bool spell_has_variable_life_cost(const CardData &cd) {
    for (const auto &ab : cd.abilities)
        if (ab.ability_type == Ability::SPELL && ab.life_cost_is_x)
            return true;
    return false;
}

// Single source for "a player gains life": raises their life total and accumulates
// life_gained_this_turn (118.9 / the "if you gained life this turn" check on cards like
// Ocelot Pride). Every life-gain site (lifelink, GainLife effects, combat lifelink) routes
// through this so the per-turn counter cannot drift from life_total. Pass the player's
// Player-component entity. No-op for amount <= 0.
inline void player_gain_life(Entity player_entity, int32_t amount) {
    if (amount <= 0 || !global_coordinator.entity_has_component<Player>(player_entity)) return;
    // CR 119.x life-gain prohibition (Roiling Vortex): if this player can't gain life this turn,
    // the gain is replaced with nothing — no life added and no life_gained_this_turn accrual.
    if (player_cant_gain_life(player_entity)) return;
    auto &pl = global_coordinator.GetComponent<Player>(player_entity);
    pl.life_total += amount;
    pl.life_gained_this_turn += amount;
}

// Single source for "a player loses life": lowers their life total and accumulates
// life_lost_this_turn (the mirror of player_gain_life). Read by Spectacle (CR 702.107a,
// "you may cast this spell for its spectacle cost … if an opponent lost life this turn").
// Damage dealt to a player IS a loss of life (CR 120.3), so the damage-to-player sites
// (combat + noncombat DealDamage) and the explicit "lose life" effects route through here
// so the per-turn counter cannot drift from life_total. Life PAID as a cost is ALSO a loss
// of life (CR 119.4, "If a player pays life, the player loses that much life"), so the
// cost-payment sites (fetch/painland PayLife, Phyrexian pips, activation/alt/deferred life
// costs, "unless you pay N life", Sylvan Library, enters-tapped-unless-pay) increment
// life_lost_this_turn alongside their subtraction — inline rather than through this helper,
// since most hold only a Player& and not the entity. That keeps Spectacle (Skewer the
// Critics / Light Up the Stage) correct when an opponent pays life on your turn. Pass the
// player's Player-component entity. No-op for amount <= 0.
inline void player_lose_life(Entity player_entity, int32_t amount) {
    if (amount <= 0 || !global_coordinator.entity_has_component<Player>(player_entity)) return;
    auto &pl = global_coordinator.GetComponent<Player>(player_entity);
    pl.life_total -= amount;
    pl.life_lost_this_turn += amount;
}

// ── Player energy ({E}, CR 122.1c) ──────────────────────────────────────────
// Energy is stored as an "ENERGY" counter in Player::counters. These are the single
// read/spend path so every energy producer/consumer (Guide of Souls, Wrath of the Skies,
// Amped Raptor) agrees on the key and the "can't pay if insufficient" rule.

// Amount of energy ({E}) the player currently has.
inline int player_energy(const Player &pl) { return pl.counter_count("ENERGY"); }

// Pay `n` energy from `pl` (CR 122.1c / 118.x — paying {E} is a cost). Returns false and
// leaves the pool untouched if the player has fewer than `n`; otherwise deducts and returns
// true. `n <= 0` is a trivially-payable no-op (returns true). Reusable by every "pay {E}"
// cost / optional payment.
inline bool pay_energy(Player &pl, int n) {
    if (n <= 0) return true;
    if (player_energy(pl) < n) return false;
    pl.add_counters("ENERGY", -n);
    return true;
}

// ── Activation conditions (CR 602.5 "activate only if …") ───────────────────
// Named gates that make an activated ability illegal to activate unless the named
// condition holds for its controller. Parsed from Activation$ <name>; evaluated at
// activation-legality time (mana-source enumeration, non-mana activated enumeration,
// and the action processor guard). Keep general: add a named condition + a case in
// activation_condition_met() rather than special-casing one card.

// Metalcraft (CR 702.46): the player controls three or more artifacts. The activating
// permanent (e.g. Mox Opal itself, an artifact) is counted. Reusable by any Metalcraft card.
inline bool controller_has_metalcraft(Zone::Ownership controller, const std::set<Entity> &entities) {
    int artifacts = 0;
    for (auto e : battlefield_permanents(entities, controller))
        if (type_set_has(global_coordinator.GetComponent<Permanent>(e).types, "Artifact"))
            if (++artifacts >= 3) return true;
    return false;
}

// True if `ab`'s Activation$ gate (if any) is satisfied for `controller`. An ability with
// no activation_condition is always allowed (returns true). `source` is the gated ability's
// source permanent (needed by per-permanent gates like NotMonstrous; pass 0 if unknown).
// Unknown condition names fail closed (return false) so a misparsed gate never silently
// permits activation.
inline bool activation_condition_met(const Ability &ab, Zone::Ownership controller,
                                     const std::set<Entity> &entities, Entity source = 0) {
    if (ab.activation_condition.empty()) return true;
    if (ab.activation_condition == "Metalcraft")
        return controller_has_metalcraft(controller, entities);
    // NotMonstrous (CR 701.37a): a monstrosity ability is legal only while its source isn't
    // already monstrous. Keyed on the source permanent's is_monstrous designation. Installed
    // automatically by parse_put_counter for any Monstrosity$ ability.
    if (ab.activation_condition == "NotMonstrous")
        return source != 0 &&
               global_coordinator.entity_has_component<Permanent>(source) &&
               !global_coordinator.GetComponent<Permanent>(source).is_monstrous;
    return false;
}

// Distinct card types (CR 205.2) among cards in `owner`'s graveyard, excluding `except`
// (pass 0 to count every card). Single source for delirium / Escape's ExileFromGrave
// group-type constraint / any "card types in your graveyard" count over a live entity set.
inline int graveyard_card_types(Zone::Ownership owner, const std::set<Entity> &entities,
                                Entity except = 0) {
    std::set<std::string> type_names;
    for (auto entity : entities) {
        if (entity == except) continue;
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(entity);
        if (z.location != Zone::GRAVEYARD || z.owner != owner) continue;
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;
        for (auto &t : global_coordinator.GetComponent<CardData>(entity).types)
            if (t.kind == TYPE) type_names.insert(t.name);
    }
    return static_cast<int>(type_names.size());
}

// Returns true when the given player has 4+ card types among cards in their graveyard.
inline bool check_delirium(Zone::Ownership owner, const std::set<Entity> &entities) {
    return graveyard_card_types(owner, entities) >= 4;
}

// Number of cards in `owner`'s graveyard, excluding `except` (pass 0 to count every card).
// Single source for the literal-count Escape ExileFromGrave cost (Uro: "exile five other
// cards") legality check and payment loop.
inline int graveyard_card_count(Zone::Ownership owner, const std::set<Entity> &entities,
                                Entity except = 0) {
    int n = 0;
    for (auto entity : entities) {
        if (entity == except) continue;
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(entity);
        if (z.location != Zone::GRAVEYARD || z.owner != owner) continue;
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;
        n++;
    }
    return n;
}

#endif /* GAME_QUERIES_H */
