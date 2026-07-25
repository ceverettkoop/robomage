#include "state_manager.h"
#include "state_manager_internal.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "../action_processor.h"
#include "../card_vocab.h"
#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/color_identity.h"
#include "../components/creature.h"
#include "../components/static_ability.h"
#include "../components/damage.h"
#include "../components/effect.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/token.h"
#include "../components/types.h"
#include "../transform.h"
#include "../day_night.h"
#include "../type_constants.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../cli_output.h"
#include "../game_driver.h"
#include "../game_queries.h"
#include "../input_logger.h"
#include "../mana_system.h"
#include "../name_card_choices.h"
#include "../parse.h"
#include "../pending_query.h"
#include "../saga.h"
#include "../svar_eval.h"
#include "../systems/stack_manager.h"
#include "replacement_effects.h"
#include "continuous_effects.h"
#include "../effects/effects.h"
#include "orderer.h"

// Forward declaration (defined below): is this Affected$ a general permanent filter
// (anthem-style class) vs the single-target EquippedBy / Self / no-filter forms? Shared by
// affected_permanents_for_static and the layer-6/7 appliers so they all agree which statics
// fan out across the affected set.
static bool affected_is_general_filter(const std::string &aff);
static void mark_unearthed_permanent(Entity entity, Permanent &perm);
static void mark_warp_permanent(Entity entity, Permanent &perm);
static void apply_global_addtype_statics(const std::set<Entity> &entities);
static bool etb_ability_removal_applies(Entity entity, const std::set<Entity> &entities);
static void remerge_animate_granted_abilities(Entity entity);
static const char *zone_display_name(Zone::ZoneValue loc);

// Unearth (CR 702.84): finalize an Unearth-returned permanent the instant its Permanent is
// created. The permanent gains haste (it enters with summoning sickness cleared and the keyword
// granted so the layer pass keeps it through end of turn — mirrors the earthbend-haste path), is
// flagged `unearthed` (the leaves-the-battlefield → exile redirect in Orderer::add_to_zone reads
// it), and registers a one-shot delayed triggered ability (CR 603.7b) that exiles it at the
// beginning of the next end step. The delayed trigger fires on the controller's end step (the next
// one, since unearth is sorcery-speed on the controller's turn); its ChangeZone moves this exact
// permanent from the battlefield to exile.
static void mark_unearthed_permanent(Entity entity, Permanent &perm) {
    perm.unearthed = true;
    perm.has_summoning_sickness = false;  // gains haste
    bool has_haste = false;
    for (const auto &kw : perm.animate_added_keywords)
        if (kw == "Haste") { has_haste = true; break; }
    if (!has_haste) perm.animate_added_keywords.push_back("Haste");

    Ability fire_ab;
    fire_ab.ability_type = Ability::TRIGGERED;
    fire_ab.category = "ChangeZone";
    fire_ab.defined_remembered = true;
    fire_ab.restore_remembered_exiled_with = {entity};  // resolve() seeds the remembered set to this card
    fire_ab.source = entity;
    fire_ab.origin = Zone::BATTLEFIELD;
    fire_ab.destination = Zone::EXILE;

    DelayedTrigger dt;
    dt.ability = fire_ab;
    dt.fire_on = Events::END_STEP_BEGAN;
    dt.owner_entity = get_player_entity(perm.controller);
    dt.fire_on_turn = cur_game.turn;
    cur_game.delayed_triggers.push_back(dt);
    game_log("%s is unearthed (haste; exiled at the next end step).\n", perm.name.c_str());
}

// Warp (a 2025 keyword; not in the checked-in CR snapshot): finalize a warp-cast permanent the
// instant its Permanent is created. Registers a one-shot delayed triggered ability (CR 603.7b,
// like Unearth) that fires at the beginning of the next end step and — via the WarpExile effect —
// exiles this exact permanent and grants its owner a lasting cast-from-exile permission (recast for
// its normal cost while it remains exiled). Unlike Unearth, warp grants NO haste and installs NO
// leaves-the-battlefield → exile redirect: a warp-cast permanent that leaves before the end step
// (dies, is bounced) simply follows normal rules. The delayed trigger fires on the next end step
// whoever is active (warp is sorcery-speed on the caster's own turn, so that is this turn's end
// step). General over any warp card.
static void mark_warp_permanent(Entity entity, Permanent &perm) {
    Ability fire_ab;
    fire_ab.ability_type = Ability::TRIGGERED;
    fire_ab.category = "WarpExile";
    fire_ab.source = entity;

    DelayedTrigger dt;
    dt.ability = fire_ab;
    dt.fire_on = Events::END_STEP_BEGAN;
    dt.owner_entity = get_player_entity(perm.controller);
    dt.fire_on_turn = cur_game.turn;
    cur_game.delayed_triggers.push_back(dt);
    game_log("%s was cast with warp (exiled at the next end step; castable from exile later).\n",
             perm.name.c_str());
}

// Evaluate a continuous static's IsPresent$/PresentZone$/PresentCompare$ gate (CR 604.3 /
// 611): count the cards/permanents matching `filter` in `zone` and compare against
// `compare`. General over any zone, any compare op, and any filter — Elvish Reclaimer's
// "+2/+2 as long as there are 3+ land cards in your graveyard" is the lands-in-graveyard
// case. Ownership qualifiers (YouOwn/YouCtrl/OppOwn/OppCtrl) are honoured against the
// containing zone's owner (a card off the battlefield has no controller, so .YouOwn is
// scoped via Zone::owner here and stripped before the type/characteristic match), mirroring
// evaluate_present_condition's owner handling. `controller` is the static source's controller
// (the "You" reference per CR 109.5). An empty filter means "no gate" → always met.
static bool static_present_condition_met(const std::string &filter, const std::string &zone_str,
                                         const std::string &compare_in, Zone::Ownership controller,
                                         Entity source, const std::set<Entity> &entities) {
    if (filter.empty()) return true;
    std::string compare = compare_in.empty() ? "GE1" : compare_in;

    // Default zone is the battlefield (Forge convention when PresentZone$ is omitted).
    Zone::ZoneValue zone = Zone::BATTLEFIELD;
    if      (zone_str == "Graveyard") zone = Zone::GRAVEYARD;
    else if (zone_str == "Hand")      zone = Zone::HAND;
    else if (zone_str == "Exile")     zone = Zone::EXILE;
    else if (zone_str == "Library")   zone = Zone::LIBRARY;
    else if (zone_str == "Stack")     zone = Zone::STACK;
    // empty / "Battlefield" → BATTLEFIELD

    MatchCtx ctx;
    ctx.controller = controller;
    ctx.source = source;   // .Self / .Other self-reference (Trinisphere: Card.Self+untapped)

    int count = 0;
    if (zone == Zone::BATTLEFIELD) {
        // permanent_matches_filter understands the full YouCtrl/YouOwn/OppCtrl/OppOwn grammar
        // against a permanent's live controller/owner, so the original filter is passed through
        // unchanged — a controller qualifier (YouCtrl) must test the controller, not Zone::owner
        // (which would mishandle a permanent you control but don't own, e.g. a stolen creature).
        for (auto e : entities) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            const auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location != zone) continue;
            if (!is_battlefield_permanent(e)) continue;
            if (!permanent_matches_filter(e, filter, ctx)) continue;
            count++;
        }
    } else {
        // Off-battlefield zones have no controller, so a YouCtrl/YouOwn (or OppCtrl/OppOwn)
        // qualifier collapses to ownership (a card in your graveyard/hand/exile is "yours").
        // card_matches_filter has no owner/controller token, so split those off and check them
        // against Zone::owner here, passing only the characteristic portion to the matcher.
        bool you = false, opp = false;
        std::string type_filter;
        {
            bool first = true;
            size_t i = 0;
            while (i <= filter.size()) {
                size_t nx = filter.find_first_of(".+", i);
                size_t end = (nx == std::string::npos) ? filter.size() : nx;
                std::string tok = filter.substr(i, end - i);
                char sep = (i == 0) ? '\0' : filter[i - 1];
                if (first) { type_filter = tok; first = false; }
                else if (tok == "YouOwn" || tok == "YouCtrl") you = true;
                else if (tok == "OppOwn" || tok == "OppCtrl") opp = true;
                else { type_filter += sep; type_filter += tok; }  // keep other qualifiers
                if (nx == std::string::npos) break;
                i = nx + 1;
            }
        }
        Zone::Ownership owner_req = you ? controller
                                 : opp ? (controller == Zone::PLAYER_A ? Zone::PLAYER_B : Zone::PLAYER_A)
                                       : Zone::UNKNOWN;
        for (auto e : entities) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            const auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location != zone) continue;
            if (owner_req != Zone::UNKNOWN && z.owner != owner_req) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            if (!card_matches_filter(e, type_filter, ctx)) continue;
            count++;
        }
    }
    return compare_svar(count, compare);
}

int active_raise_cost_for(const CardData &card_data, Zone::Ownership caster) {
    bool is_creature = is_creature_card(card_data);
    // Caster's spells already cast this turn — the "other spells" the relative surcharge counts
    // (Damping Sphere). Evaluated once; 0 when the caster is unknown or has no Player component.
    int spells_cast_this_turn = 0;
    if (caster != Zone::UNKNOWN) {
        Entity pe = get_player_entity(caster);
        if (global_coordinator.entity_has_component<Player>(pe))
            spells_cast_this_turn =
                static_cast<int>(global_coordinator.GetComponent<Player>(pe).spells_cast_this_turn);
    }
    int total = 0;
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.sa->category != "RaiseCost") continue;
        if (as.sa->raise_cost_filter == "nonCreature" && is_creature) continue;
        if (as.sa->match_named_card) {
            if (!global_coordinator.entity_has_component<Permanent>(as.entity)) continue;
            auto &src = global_coordinator.GetComponent<Permanent>(as.entity);
            if (src.chosen_name.empty() || src.chosen_name != card_data.name) continue;
        }
        total += as.sa->raise_cost;
        // Relative per-spell surcharge (Damping Sphere): {1} more for each other spell the
        // caster has cast this turn (CR 601.2f). Applies to every spell that player casts.
        if (as.sa->raise_cost_per_spell_cast) total += spells_cast_this_turn;
    }
    return total;
}

// Number of artifacts `caster` controls on the battlefield (Affinity count, CR 702.41).
static int artifacts_controlled_by(Zone::Ownership caster) {
    int count = 0;
    Entity max_e = global_coordinator.GetMaxIssuedEntity();
    for (Entity e = 0; e < max_e; ++e) {
        if (!is_battlefield_permanent(e, caster)) continue;
        if (permanent_has_type(global_coordinator.GetComponent<Permanent>(e), "Artifact")) count++;
    }
    return count;
}

// Total generic mana that active ReduceCost statics remove from the cost of casting
// `card_data` for player `caster` (the mirror of active_raise_cost_for). Each ReduceCost
// static reduces matching spells' generic cost by its Amount$ (CR 118.7 / 601.2f); an
// Activator$ You static only reduces spells cast by its own controller. The caller clamps
// the generic portion at zero and never reduces colored pips.
int active_reduce_cost_for(const CardData &card_data, Zone::Ownership caster) {
    int total = 0;
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.sa->category != "ReduceCost") continue;
        if (as.sa->reduce_cost_you_only && as.controller != caster) continue;
        // An empty filter means the reduction applies to every spell (no characteristic gate).
        if (!as.sa->reduce_cost_filter.empty()) {
            // Seed any static mana-value qualifier (It That Heralds the End's cmcGE7) into the
            // MatchCtx — the evaluator defers cmc comparators to ctx.cmc_bound, so without this
            // the bound is silently ignored and every colorless spell would be reduced.
            MatchCtx ctx;
            extract_static_cmc_bound(as.sa->reduce_cost_filter, ctx);
            if (!card_matches_filter(card_data, as.sa->reduce_cost_filter, ctx)) continue;
        }
        total += as.sa->reduce_cost;
    }
    return total;
}

int active_cost_floor_for(const CardData &card_data) {
    int floor = 0;
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;        // 613.1f: source lost all abilities (Humility)
        if (!as.condition_met) continue;    // IsPresent$ gate (Trinisphere: source untapped)
        if (as.sa->category != "SetCost" || !as.sa->set_cost_raise_to) continue;
        // An empty filter means the floor applies to every spell (Trinisphere: ValidCard$ Card).
        if (!as.sa->set_cost_filter.empty()) {
            MatchCtx ctx;
            extract_static_cmc_bound(as.sa->set_cost_filter, ctx);
            if (!card_matches_filter(card_data, as.sa->set_cost_filter, ctx)) continue;
        }
        floor = std::max(floor, as.sa->set_cost_min);
    }
    return floor;
}

// Resolve the battlefield permanents a continuous static's Affected$ filter designates
// (declared in state_manager.h). The EquippedBy / Self forms target a single creature that
// the layer appliers resolve directly (equipped_to / the source), so this returns empty for
// them and for an empty filter; every other Affected$ value (e.g. "Creature.Colorless+
// Other+YouCtrl") is a permanent filter matched through permanent_matches_filter with the
// static's controller as the YouCtrl reference and the source permanent as the .Other
// self-exclusion seam. A static numeric mana-value qualifier (cmcGE7) in the filter is seeded
// into the MatchCtx so the comparator is honoured rather than silently passing.
std::vector<Entity> affected_permanents_for_static(const ActiveStatic &as,
                                                   const std::set<Entity> &entities) {
    std::vector<Entity> out;
    const std::string &aff = as.sa->affected;
    // EquippedBy / Self / no-filter forms are single-target (the layer appliers resolve them
    // directly); only a general permanent filter fans out here. Shared predicate so this and
    // the layer-6/7 appliers agree on which statics are general.
    if (!affected_is_general_filter(aff)) return out;
    MatchCtx ctx;
    ctx.controller = as.controller;   // YouCtrl/OppCtrl reference (CR 109.5)
    ctx.source = as.entity;           // .Other self-exclusion
    extract_static_cmc_bound(aff, ctx);
    for (auto e : entities)
        if (permanent_matches_filter(e, aff, ctx)) out.push_back(e);
    return out;
}

// Parse a SetColor$ spec into a color set (CR 105.2/202.2). "Colorless" — or an empty/unrecognized
// token — contributes no color (CR 105.2c → empty set). Color words are space- or comma-delimited
// so an explicit multi-color set (e.g. "White Blue") is handled the same way.
static std::set<Colors> parse_setcolor_spec(const std::string &spec) {
    std::set<Colors> out;
    size_t p = 0;
    while (p < spec.size()) {
        size_t e = spec.find_first_of(" ,", p);
        if (e == std::string::npos) e = spec.size();
        std::string tok = spec.substr(p, e - p);
        if      (tok == "White") out.insert(WHITE);
        else if (tok == "Blue")  out.insert(BLUE);
        else if (tok == "Black") out.insert(BLACK);
        else if (tok == "Red")   out.insert(RED);
        else if (tok == "Green") out.insert(GREEN);
        // "Colorless" / unrecognized → no color contributed
        p = e + 1;
    }
    return out;
}

// Does a SetColor$ static's AffectedZone$ reach a card currently in `zone`? "All" = every zone;
// empty = the Forge default (battlefield only); otherwise a space-/comma-delimited zone list.
static bool setcolor_zone_matches(const std::string &az, Zone::ZoneValue zone, bool has_zone) {
    if (az == "All") return true;
    if (az.empty()) return has_zone && zone == Zone::BATTLEFIELD;  // Forge default for a continuous static
    if (!has_zone) return false;
    size_t p = 0;
    while (p < az.size()) {
        size_t e = az.find_first_of(" ,", p);
        if (e == std::string::npos) e = az.size();
        std::string tok = az.substr(p, e - p);
        if ((tok == "Battlefield" && zone == Zone::BATTLEFIELD) ||
            (tok == "Graveyard"   && zone == Zone::GRAVEYARD)   ||
            (tok == "Hand"        && zone == Zone::HAND)        ||
            (tok == "Library"     && zone == Zone::LIBRARY)     ||
            (tok == "Exile"       && zone == Zone::EXILE)       ||
            (tok == "Stack"       && zone == Zone::STACK))
            return true;
        p = e + 1;
    }
    return false;
}

// Non-recursive test of a SetColor$ static's Affected$ filter against a battlefield TOKEN
// permanent (CR 111.1: a token is not a card, so it has no CardData and the card_matches_filter
// path can't read it). permanent_matches_filter WOULD read it, but it reads effective_colors,
// which is exactly the caller of setcolor_override_for — so using it here would recurse
// (effective_colors → setcolor_override_for → permanent_matches_filter → effective_colors). So a
// token is matched WITHOUT consulting its color:
//   • head "Card" / "Permanent" matches any battlefield permanent; a type name matches if the
//     token carries that type/subtype (Permanent::types).
//   • YouCtrl / OppCtrl are honoured against the token's live controller; token is honoured
//     (this is a token); nonToken can't be satisfied by a token.
//   • Any other type/subtype qualifier matches if the token's type line carries it.
// A COLOR qualifier (White/Blue/…/Colorless or non<Color>) can't be evaluated without reading the
// token's color (which would recurse), so a filter carrying one is treated as NOT matching the
// token — documented limitation. The only in-vocab global SetColor static, Mycosynth Lattice, uses
// a bare "Card" filter, which matches every battlefield permanent (tokens included), so this
// limitation is not currently reachable.
static bool setcolor_filter_matches_token(const std::string &aff, const Permanent &perm,
                                          Zone::Ownership static_controller) {
    if (aff.empty()) return false;
    Zone::Ownership opp =
        (static_controller == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
    bool first = true;
    size_t i = 0;
    while (i <= aff.size()) {
        size_t nx = aff.find_first_of(".+", i);
        size_t end = (nx == std::string::npos) ? aff.size() : nx;
        std::string tok = aff.substr(i, end - i);
        if (first) {
            // Head: a permanent wildcard or a type the token carries.
            if (tok != "Card" && tok != "Permanent" && !permanent_has_type(perm, tok))
                return false;
            first = false;
        } else if (tok == "YouCtrl") {
            if (perm.controller != static_controller) return false;
        } else if (tok == "OppCtrl") {
            if (perm.controller != opp) return false;
        } else if (tok == "token" || tok == "Token") {
            // a token satisfies this
        } else if (tok == "nonToken") {
            return false;  // this IS a token
        } else if (tok == "White" || tok == "Blue" || tok == "Black" || tok == "Red" ||
                   tok == "Green" || tok == "Colorless" || tok.rfind("non", 0) == 0) {
            // A color (or non-color) qualifier would require reading the token's color — skip
            // to stay recursion-safe (documented limitation; not reachable in current vocab).
            return false;
        } else if (!permanent_has_type(perm, tok)) {
            return false;  // any remaining type/subtype qualifier must be on the token's type line
        }
        if (nx == std::string::npos) break;
        i = nx + 1;
    }
    return true;
}

// Layer-5 (CR 613.1e / 612) global color-changing override — see game_queries.h. Scans the active-
// statics cache for a SetColor$ continuous static whose AffectedZone$ + Affected$ filter designate
// `e`; the FIRST match wins (no current vocab stacks two global SetColor statics). For a real card
// the match is done against the card's PRINTED characteristics (card_matches_filter) so this never
// recurses back into effective_colors; a battlefield TOKEN permanent (no CardData) is matched by
// setcolor_filter_matches_token, a deliberately color-free type/controller test, for the same
// recursion-safety reason. A static whose zone/filter excludes `e` yields no override.
bool setcolor_override_for(Entity e, std::set<Colors> &out) {
    if (g_active_statics.empty()) return false;
    bool has_card = global_coordinator.entity_has_component<CardData>(e);
    bool is_token_perm = !has_card &&
                         global_coordinator.entity_has_component<Token>(e) &&
                         is_battlefield_permanent(e);
    if (!has_card && !is_token_perm) return false;
    Zone::ZoneValue zone = Zone::LIBRARY;
    bool has_zone = global_coordinator.entity_has_component<Zone>(e);
    if (has_zone) zone = global_coordinator.GetComponent<Zone>(e).location;
    for (const auto &as : g_active_statics) {
        if (as.suppressed || !as.condition_met) continue;
        if (as.sa->category != "Continuous" || as.sa->set_color.empty()) continue;
        if (!setcolor_zone_matches(as.sa->affected_zone, zone, has_zone)) continue;
        if (has_card) {
            MatchCtx ctx;
            ctx.controller = as.controller;
            ctx.source = as.entity;
            extract_static_cmc_bound(as.sa->affected, ctx);
            if (!card_matches_filter(e, as.sa->affected, ctx)) continue;
        } else {
            // Battlefield token permanent: recursion-safe type/controller filter match.
            const auto &perm = global_coordinator.GetComponent<Permanent>(e);
            if (!setcolor_filter_matches_token(as.sa->affected, perm, as.controller)) continue;
        }
        out = parse_setcolor_spec(as.sa->set_color);
        return true;
    }
    return false;
}

bool any_mana_as_any_color_active() {
    for (const auto &as : g_active_statics) {
        if (as.suppressed || !as.condition_met) continue;
        if (as.sa->category != "ManaConvert") continue;
        if (as.sa->mana_conversion == "AnyType->AnyColor") return true;
    }
    return false;
}

ManaValue effective_base_cost(const CardData &card_data, Zone::Ownership caster) {
    ManaValue cost = card_data.mana_cost;
    int raise_total = active_raise_cost_for(card_data, caster);
    for (int ri = 0; ri < raise_total; ri++) cost.insert(GENERIC);
    // ReduceCost statics (Eye of Ugin): reduce the generic portion by Amount$ per matching
    // static. Applied after additions (CR 601.2f); only generic pips removed (never a colored
    // pip — CR 118.7), never below 0. Caster-gated for Activator$ You statics.
    if (caster != Zone::UNKNOWN) {
        int reduce_total = active_reduce_cost_for(card_data, caster);
        while (reduce_total-- > 0) {
            auto it = cost.find(GENERIC);
            if (it == cost.end()) break;
            cost.erase(it);
        }
    }
    // Affinity for artifacts (CR 702.41): reduce the generic portion by {1} per artifact
    // the caster controls. Cost reductions are applied after additions (601.2f) and only
    // the generic pips can be removed (a colored pip is never reduced); never go below 0.
    if (card_data.affinity_artifact && caster != Zone::UNKNOWN) {
        int reduce = artifacts_controlled_by(caster);
        while (reduce-- > 0) {
            auto it = cost.find(GENERIC);
            if (it == cost.end()) break;
            cost.erase(it);
        }
    }
    // SetCost cost floor (Trinisphere): applied LAST — after every other increase/reduction
    // (CR 601.2f) — so the floor is checked against the spell's already-adjusted cost. If the
    // total mana value is below the floor, add generic pips up to the floor; a cost already at
    // or above it (or a spell no floor applies to) is untouched. Colored pips are never removed.
    int floor = active_cost_floor_for(card_data);
    while (static_cast<int>(cost.size()) < floor) cost.insert(GENERIC);
    return cost;
}

ManaValue floored_alt_mana_cost(const CardData &card_data, const ManaValue &alt_mana,
                                Zone::Ownership caster) {
    // The alternative cost SUBSTITUTES for the printed mana cost (CR 601.2b); cost increases
    // and the SetCost floor then apply ON TOP (CR 601.2f) — the same ordering as
    // effective_base_cost, just against the substituted mana instead of the printed cost.
    ManaValue cost = alt_mana;
    // Cost-increase statics (Thalia, Guardian of Thraben: noncreature spells cost {1} more) add
    // an additive generic surcharge that applies to the total cost regardless of whether a normal
    // or alternative cost is being paid. Folded in here so a pitch/alternative cast (Daze, Force
    // of Will) is taxed exactly like its normal-cost cast. NamedCard-aware / nonCreature-filtered
    // via the shared active_raise_cost_for; the per-spell surcharge needs the casting player.
    int raise_total = active_raise_cost_for(card_data, caster);
    for (int ri = 0; ri < raise_total; ri++) cost.insert(GENERIC);
    // SetCost floor (Trinisphere): applied LAST — after the increase (601.2f). Colored pips stay;
    // only generic is added to raise a sub-floor total.
    int floor = active_cost_floor_for(card_data);
    while (static_cast<int>(cost.size()) < floor) cost.insert(GENERIC);
    return cost;
}

static void add_keywords_from_spec(Creature &cr, const std::string &spec);
static bool removal_affects(const ActiveStatic &r, Entity entity);

// Is this Affected$ filter a *general* permanent filter (an anthem-style class like
// "Creature.Colorless+Other+YouCtrl"), as opposed to the single-target EquippedBy / Self /
// no-filter forms? The general forms are resolved through affected_permanents_for_static
// (which buffs/grants to every matching permanent); the single-target forms keep the source/
// equipped-creature path. Shared by the layer-6 keyword-grant pass and the layer-7 P/T pass so
// both agree on which statics fan out across the affected set (CR 613 layers 6 / 7c).
// An attached-permanent static buffs the single object it is attached to: equipment uses the
// "EquippedBy" filter and Auras use "EnchantedBy" (CR 303). Both resolve to the source
// permanent's equipped_to link (auras reuse the same attachment field), so the layer appliers
// treat them identically — single-target, not a general fan-out filter.
static bool affected_is_attached_target(const std::string &aff) {
    return aff.find("EquippedBy") != std::string::npos ||
           aff.find("EnchantedBy") != std::string::npos;
}

static bool affected_is_general_filter(const std::string &aff) {
    return !aff.empty() &&
           !affected_is_attached_target(aff) &&
           aff.find("Self") == std::string::npos;
}

// Lands whose subtype was set to a basic land type this SBE pass (Blood Moon / Magus of
// the Moon). Rule 305.7: such a land loses all abilities generated from its rules text,
// keeping only the regenerated intrinsic (subtype-derived) mana ability. Populated in
// apply_type_changing_effects (layer 4), consumed by recompute_abilities (after layer 7);
// cleared each pass in gather_active_statics.
static std::vector<Entity> g_type_set_lands;

// Apply a " & "-delimited keyword spec (e.g. "Flying & Trample") to a creature. Removal
// of a granted keyword needs no counterpart helper: gather_active_statics rebuilds each
// creature's keyword set from its printed base every SBE pass (rule 611.3a), so a grant
// that stops applying simply isn't re-added.
static void add_keywords_from_spec(Creature &cr, const std::string &spec) {
    size_t p = 0;
    while (p < spec.size()) {
        size_t sep = spec.find(" & ", p);
        if (sep == std::string::npos) sep = spec.size();
        std::string kw = spec.substr(p, sep - p);
        if (!kw.empty()) cr.keywords.push_back(kw);
        p = (sep < spec.size()) ? sep + 3 : sep;
    }
}

// entity_name moved to game_queries.cpp — the single shared display-name resolver.

std::string target_display_name(const Game &game, Entity tgt) {
    if (global_coordinator.entity_has_component<Player>(tgt))
        return player_name((tgt == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B);
    return entity_name(tgt);
}

static Colors mana_color_for_subtype(const std::string &subtype) {
    if (subtype == "Mountain") return RED;
    if (subtype == "Forest") return GREEN;
    if (subtype == "Plains") return WHITE;
    if (subtype == "Island") return BLUE;
    if (subtype == "Swamp") return BLACK;
    if (subtype == "Wastes") return COLORLESS;
    return NO_COLOR;
}

// Permanents on battlefield set to have appropriate components
// if they are in a different zone these are removed as no longer applicable
void StateManager::apply_permanent_components(Game &game, std::shared_ptr<Orderer> orderer) {
    // Collect token entities that have left the battlefield for destruction after iteration.
    std::vector<Entity> tokens_to_destroy;

    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;

        // Handle token entities (no CardData)
        if (!global_coordinator.entity_has_component<CardData>(entity)) {
            if (!global_coordinator.entity_has_component<Token>(entity)) continue;
            auto &zone = global_coordinator.GetComponent<Zone>(entity);
            auto &token = global_coordinator.GetComponent<Token>(entity);
            if (zone.location == Zone::BATTLEFIELD) {
                bootstrap_token_components(entity, token, zone.controller, game.timestamp);
                apply_keyword_abilities(entity);
                remerge_animate_granted_abilities(entity);
            } else {
                // Token has left the battlefield — strip components (shared helper also clears
                // attachment links, 704.5n) and schedule for destruction
                strip_permanent_components(entity);
                tokens_to_destroy.push_back(entity);
            }
            continue;
        }

        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location == Zone::BATTLEFIELD) {  // on battlefield, check to add components
            // check types
            auto &card_data = global_coordinator.GetComponent<CardData>(entity);
            // A transformed DFC shows its back face: determine creature/land-ness from the
            // active face so a later SBA pass doesn't re-add a front-face Creature component
            // to a permanent that flipped to a non-creature back face (Ajani -> planeswalker).
            const CardData *face = &card_data;
            if (global_coordinator.entity_has_component<Permanent>(entity) &&
                global_coordinator.GetComponent<Permanent>(entity).transformed && card_data.backside)
                face = card_data.backside.get();
            bool is_creature = is_creature_card(*face);  // can be creature and land
            bool is_land = is_land_card(*face);
            // Reconfigure (CR 702.151b): while a reconfigure equipment is attached, it is an
            // Equipment and is NOT a creature. Suppress its creature-ness (and strip its
            // Creature/Damage components below) for as long as equipped_to is set; unattaching
            // restores them on the next state-based pass.
            bool reconfigured_attached =
                card_data.is_reconfigure &&
                global_coordinator.entity_has_component<Permanent>(entity) &&
                global_coordinator.GetComponent<Permanent>(entity).equipped_to != 0;
            if (reconfigured_attached) is_creature = false;
            int etb_p1p1 = 0;  // counters this permanent enters with (614.1c), applied once the Permanent exists
            std::string etb_counter_type = "P1P1";  // kind of "enters with" counter (P1P1, CHARGE, ...)
            // providing permanent component if doesn't have
            if (!global_coordinator.entity_has_component<Permanent>(entity)) {
                Permanent perm;
                perm.name = card_data.name;
                perm.types = card_data.types;
                perm.controller = zone.controller;
                perm.has_summoning_sickness = is_creature;
                perm.is_tapped = false;
                // Replacement effects at the point the permanent enters (rule 614): "enters
                // tapped" (a self-replacement, or a fetch's "onto the battlefield tapped") and
                // "enters with +1/+1 counters". The counters are applied once the Creature
                // component exists (below).
                {
                    // Containment Priest's "exile instead of entering" (614.1a) is handled
                    // earlier, in the MOVE_TO_ZONE dispatch inside add_to_zone, so a redirected
                    // creature never reaches the battlefield and is never seen by this scan.
                    // Here we only read self-replacements that shape how the permanent enters:
                    // "enters tapped" (614.1d) and "enters with counters" (614.1c).
                    ReplacementEvent rev;
                    rev.type = ReplacementEvent::ENTERS_BATTLEFIELD;
                    rev.entity = entity;
                    rev.affected_player = zone.controller;  // 616.1: the permanent's controller chooses
                    replacement::dispatch(rev);
                    // The tapped-unless-life y/n parked a pending decision
                    // (SBE_LATCHED, same contract as the choose-type site
                    // below): suspend mid-apply BEFORE the Permanent is
                    // created — nothing about this entity has been mutated,
                    // so the resumed pass re-reaches it, re-dispatches, and
                    // consumes the latch at the same ask.
                    if (cur_game.pending_query.active && !cur_game.pending_query.answered)
                        return;
                    perm.is_tapped = rev.enters_tapped;
                    etb_p1p1 = rev.etb_p1p1;
                    etb_counter_type = rev.etb_counter_type;
                }
                // It was cast and is now becoming a real permanent — consume the one-shot
                // "was cast" marker so a later non-cast re-entry isn't treated as a cast, and
                // record it on the permanent (The One Ring's Card.wasCastByYou ETB gate).
                if (game.cast_to_battlefield.erase(entity)) perm.entered_by_cast = true;
                // Likewise consume the "cast from your hand by you" marker and record it on
                // the permanent (Amped Raptor's Card.wasCastFromYourHandByYou gate). Only a
                // spell the controller cast from their own hand sets this; any other entry
                // (reanimation, tokens, ChangeZone, impulse cast from exile) leaves it false.
                if (game.cast_from_hand.erase(entity)) perm.cast_from_hand_by_controller = true;
                // Daybound (CR 702.145b): "If it is night and this permanent is represented by a
                // double-faced card, it enters transformed." Mark a daybound DFC entering while
                // it's night to flip to its back (night) face once its components exist — reusing
                // the same pending_enters_transformed path Ajani-style transformed entries use.
                if (card_has_daybound(card_data) && game.day_night == Game::DN_NIGHT)
                    game.pending_enters_transformed.insert(entity);
                if (perm.is_tapped) game_log("%s enters tapped.\n", perm.name.c_str());
                // Spell was cast for its evoke cost — mark the permanent so its evoke
                // self-sacrifice ETB trigger fires (consumed one-shot here).
                if (game.pending_evoked.erase(entity)) perm.evoked = true;
                // Spell was cast from the graveyard for its Escape cost — mark the permanent as
                // having "escaped" so an "if it escaped" clause (Uro's sacrifice-unless) reads it.
                if (game.pending_escaped.erase(entity)) perm.cast_with_escape = true;
                // Spell was cast with its Offspring additional cost — mark the permanent so
                // its offspring token-copy ETB trigger fires (consumed one-shot here).
                if (game.pending_offspring.erase(entity)) perm.entered_with_offspring = true;
                // Spell was cast for its Impending alternate cost (CR 702.175d): the permanent
                // enters with N time counters. Set them on the Permanent being built (before it
                // is added below) so this same pass's creature-suppression check already sees
                // them and strips the Creature component — it enters as a noncreature.
                if (game.pending_impending.erase(entity) && card_data.alt_cost.impending_count > 0) {
                    perm.counters["TIME"] = card_data.alt_cost.impending_count;
                    perm.entered_via_impending = true;  // marks these TIME counters as impending
                    game_log("%s enters with %d time counter(s) (impending).\n",
                             card_data.name.c_str(), card_data.alt_cost.impending_count);
                }
                // Unearth (CR 702.84): returned to the battlefield by its unearth ability — gains
                // haste, gets the delayed end-step exile, and the leaves→exile redirect.
                if (game.pending_unearthed.erase(entity)) mark_unearthed_permanent(entity, perm);
                // Warp: cast for its warp alternate cost — register the delayed end-step exile that
                // then grants the recast-from-exile permission (no haste, no leaves→exile redirect).
                if (game.pending_warp.erase(entity)) mark_warp_permanent(entity, perm);
                // Planeswalkers enter with loyalty counters equal to printed loyalty (306.5b).
                if (is_planeswalker_card(card_data)) perm.counters["LOYALTY"] = card_data.starting_loyalty;
                perm.timestamp_entered_battlefield = game.timestamp++;
                perm.entered_on_turn = game.turn;
                // A DB$ Attach resolved onto this creature before its Permanent existed
                // (reanimate-then-attach, Pre-War Formalwear): finalize the equip link now
                // that the Permanent is being created. The equipment kept its own Permanent.
                {
                    auto pa = game.pending_attach.find(entity);
                    if (pa != game.pending_attach.end()) {
                        Entity equip = pa->second;
                        if (global_coordinator.entity_has_component<Permanent>(equip)) {
                            auto &eq_perm = global_coordinator.GetComponent<Permanent>(equip);
                            if (eq_perm.equipped_to != 0 &&
                                global_coordinator.entity_has_component<Permanent>(eq_perm.equipped_to))
                                global_coordinator.GetComponent<Permanent>(eq_perm.equipped_to).equipped_by = 0;
                            eq_perm.equipped_to = entity;
                            perm.equipped_by = equip;
                            game_log("Equipment attached.\n");
                        }
                        game.pending_attach.erase(pa);
                    }
                }
                // An Aura that resolved onto the battlefield (CR 303.4f) attaches to the object it
                // was cast targeting. The aura is the entity whose Permanent is being created;
                // its enchanted object was recorded at cast (pending_aura_target). Reuse the
                // equipped_to/equipped_by attachment link so the aura's static buffs (Affected$
                // Creature.EnchantedBy) and the aura state-based check find the enchanted object.
                {
                    auto pat = game.pending_aura_target.find(entity);
                    if (pat != game.pending_aura_target.end()) {
                        Entity enchanted = pat->second;
                        if (enchanted != 0 && global_coordinator.entity_has_component<Permanent>(enchanted)) {
                            perm.equipped_to = enchanted;
                            game_log("%s is attached to %s.\n", perm.name.c_str(),
                                     entity_name(enchanted).c_str());
                            game.pending_aura_target.erase(pat);
                        }
                        // Animate Dead-style aura (K:Enchant:Creature.inZoneGraveyard, CR 303.4):
                        // its enchant target is a creature card still in a graveyard, so it has no
                        // Permanent to attach to yet — the aura enters UNATTACHED and its ETB
                        // trigger reanimates the card and attaches. Leave the pending_aura_target
                        // entry in place so (a) the trigger's Defined$ Enchanted can find the card
                        // and (b) the "unattached aura" state-based action skips this aura until the
                        // reanimation resolves (see state_manager.cpp). The entry is cleared by the
                        // reanimating ChangeZone (Defined$ Enchanted) once the card is returned.
                    }
                }
                global_coordinator.AddComponent(entity, perm);
                // Non-P1P1 "enters with" counters (614.1c) attach to any permanent, not just
                // creatures — Chalice of the Void enters with X CHARGE counters. P1P1 counters
                // are applied in the creature block below so its P/T can be logged.
                // P/T counters (P1P1, M1M1) are applied in the creature block below, after the
                // Creature component exists, so add_counters can resync the cached P/T and log it.
                if (etb_p1p1 > 0 && etb_counter_type != "P1P1" && etb_counter_type != "M1M1") {
                    add_counters(entity, etb_counter_type, etb_p1p1);
                    game_log("%s enters with %d %s counter(s).\n",
                        card_data.name.c_str(), etb_p1p1, etb_counter_type.c_str());
                }
                // CR 714.3a/714.2b: a Saga enters with one lore counter; reaching chapter 1 fires
                // its first chapter ability. Done once here in the newly-entered block (the
                // Permanent now exists). Saga lifecycle lives in src/saga.cpp. This runs BEFORE
                // this pass's layer run can flag the fresh Permanent, so consult the removal
                // statics directly: per CR 613.1 an ability-removal effect applies immediately,
                // and a Saga entering under Magus of the Moon enters as a Mountain — no lore
                // counter, no chapter I (the flag-based gates in saga.cpp cover later turns).
                if (card_is_saga(card_data) && !etb_ability_removal_applies(entity, mEntities))
                    saga_add_lore_counters(entity, 1);
            }
            // copy activated abilities from card_data to permanent; incl mana abilities although mana abilities innate to basic land types
            // added elsewhere
            for (const auto &ab : card_data.abilities) {
                if (ab.ability_type != Ability::ACTIVATED) continue;
                auto &perm_abilities = global_coordinator.GetComponent<Permanent>(entity).abilities;
                bool already_present = false;
                for (auto &existing : perm_abilities) {
                    if (existing.identical_activated_ability(ab)) {
                        already_present = true;
                        break;
                    }
                }
                if (already_present) continue;
                // Copy only the ability actually added (in steady state this copies nothing —
                // the already-present check short-circuits every pass after the first), instead
                // of copying every card_data ability by value just to discard it each pass.
                perm_abilities.push_back(ab);
                perm_abilities.back().source = entity;
            }
            // Re-merge rest-of-game ability grants (DB$ Animate | Abilities$ | Duration$
            // Permanent — Urza's Saga chapters) that a layer-6 ability removal (Magus of the
            // Moon) stripped from perm.abilities; a no-op in steady state (dedup) and while
            // the remover is still present (recompute_abilities strips again after layers).
            remerge_animate_granted_abilities(entity);

            // copy static abilities from card_data to permanent (applied = false by default)
            if (global_coordinator.GetComponent<Permanent>(entity).static_abilities.empty() &&
                !card_data.static_abilities.empty()) {
                auto &perm_sa = global_coordinator.GetComponent<Permanent>(entity).static_abilities;
                for (auto &sa : card_data.static_abilities)
                    perm_sa.push_back(sa);
            }

            // providing creature related components if applicable
            if (is_creature && !global_coordinator.entity_has_component<Creature>(entity)) {
                Creature creature;
                creature.base_power = static_cast<int>(card_data.power);
                creature.base_toughness = static_cast<int>(card_data.toughness);
                creature.keywords = card_data.keywords;
                recompute_pt(creature);
                global_coordinator.AddComponent(entity, creature);
                // damage component
                Damage damage;
                damage.damage_counters = 0;
                global_coordinator.AddComponent(entity, damage);

                // Apply the "enters with" P/T counters chosen by the ETB replacement
                // dispatch above (rule 614.1c), now that the Creature component exists so
                // add_counters can resync the cached P/T. Honors the declared counter kind:
                // +1/+1 (Hangarback) or -1/-1 (Moonshadow enters with six -1/-1 counters).
                if (etb_p1p1 > 0) {
                    std::string pt_type = (etb_counter_type == "M1M1") ? "M1M1" : "P1P1";
                    add_counters(entity, pt_type, etb_p1p1);
                    auto &cr = global_coordinator.GetComponent<Creature>(entity);
                    game_log("%s enters with %d %s counter(s) (%u/%u).\n",
                        card_data.name.c_str(), etb_p1p1,
                        pt_type == "M1M1" ? "-1/-1" : "+1/+1", cr.power, cr.toughness);
                }
            }
            // Impending (CR 702.175d-e): a permanent that entered for its impending cost ISN'T a
            // creature while it still has time counters (it is on the battlefield only as its other
            // types). The same Creature/Damage strip the reconfigure suppression uses applies here,
            // gated on a remaining TIME counter; the creature block above re-adds the components on
            // the pass after the last time counter sheds at the controller's end step (game.cpp),
            // so it "becomes a creature" via the normal state-based recompute.
            bool impending_suppressed =
                global_coordinator.entity_has_component<Permanent>(entity) &&
                global_coordinator.GetComponent<Permanent>(entity).entered_via_impending &&
                get_counters(entity, "TIME") > 0;
            // Reconfigure (CR 702.151b): strip the Creature/Damage components while attached so the
            // permanent isn't a creature for combat, targeting, and state-based actions. They are
            // re-added by the block above on the next pass once it unattaches (equipped_to == 0).
            if (reconfigured_attached || impending_suppressed) {
                if (global_coordinator.entity_has_component<Creature>(entity))
                    global_coordinator.RemoveComponent<Creature>(entity);
                if (global_coordinator.entity_has_component<Damage>(entity))
                    global_coordinator.RemoveComponent<Damage>(entity);
            }
            if (is_land) {
                apply_land_abilities(entity);
            }
            // Earthbended land (DB$/AB$ Animate make_creature): an animated land is a creature
            // even though is_creature_card(card) is false, so re-bootstrap its Creature/Damage
            // components here if they were lost. Idempotent; honors the 0/0 base + Haste grant.
            {
                const auto &perm_anim = global_coordinator.GetComponent<Permanent>(entity);
                if (perm_anim.animate_make_creature || perm_anim.animate_make_creature_eot)
                    effects::apply_animate_creature_bootstrap(entity);
            }
            apply_keyword_abilities(entity);

            // A card moved here "transformed" (Ajani's exile-and-return) enters showing
            // its DFC back face. The front-face components exist now, so flip to the back
            // face before triggers are checked — this also suppresses the front-face ETB
            // triggers (a permanent entering as its back face fires only that face's ETBs).
            if (game.pending_enters_transformed.erase(entity)) set_permanent_face(entity, true);

            // Daybound (CR 702.145d): any time a player controls a front-face-up permanent with
            // daybound and it's neither day nor night, it becomes day. This continuous check fires
            // the first time such a permanent is on the battlefield while it's still neither; once
            // it becomes day the guard stops it re-firing. (If it was night, it entered transformed
            // above; if already day, nothing happens.)
            if (game.day_night == Game::DN_NEITHER && card_has_daybound(card_data) &&
                global_coordinator.entity_has_component<Permanent>(entity) &&
                !global_coordinator.GetComponent<Permanent>(entity).transformed)
                become_day();

            // Ninjutsu (CR 702.49e): a card put onto the battlefield "tapped and attacking" by a
            // ninjutsu ability. A normal creature ninja has its Creature component now, so mark it
            // attacking the defender the returned attacker had been attacking (it already entered
            // tapped via pending_enters_tapped). A ninja that becomes a creature only via a
            // conditional static — Kaito, a planeswalker that's a 3/4 Ninja creature during your
            // turn — has no Creature component yet (it is bootstrapped later this pass in layer 4),
            // so LEAVE the mark pending; the layer-4 self-animate bootstrap consumes it and sets the
            // attacking state once the Creature exists.
            {
                auto pea = game.pending_enters_attacking.find(entity);
                if (pea != game.pending_enters_attacking.end() &&
                    global_coordinator.entity_has_component<Creature>(entity)) {
                    auto &ncr = global_coordinator.GetComponent<Creature>(entity);
                    ncr.is_attacking = true;
                    ncr.attack_target = pea->second;
                    ncr.is_blocked = false;
                    game_log("%s is attacking (ninjutsu).\n", entity_name(entity).c_str());
                    game.pending_enters_attacking.erase(pea);
                }
            }

            // ETBReplacement: choose creature type (Cavern of Souls)
            if (card_data.has_etb_choose_creature_type) {
                auto &perm_ref = global_coordinator.GetComponent<Permanent>(entity);
                if (perm_ref.chosen_type.empty()) {
                Entity player_entity = get_player_entity(perm_ref.controller);
                auto &player = global_coordinator.GetComponent<Player>(player_entity);

                if (!player.creature_subtypes.empty()) {
                    // Build subtype name list from all_subtypes index
                    std::vector<std::string> subtype_names;
                    for (auto &pair : player.creature_subtypes) {
                        auto it = all_subtypes.begin();
                        std::advance(it, pair.second);
                        subtype_names.push_back(*it);
                    }

                    // Count frequency of each subtype among owned creatures for sorting
                    std::vector<int> freq(subtype_names.size(), 0);
                    for (auto e2 : mEntities) {
                        if (!global_coordinator.entity_has_component<CardData>(e2)) continue;
                        auto &z2 = global_coordinator.GetComponent<Zone>(e2);
                        if (z2.owner != perm_ref.controller) continue;
                        auto &cd2 = global_coordinator.GetComponent<CardData>(e2);
                        if (!is_creature_card(cd2)) continue;
                        for (auto &t : cd2.types) {
                            if (t.kind != SUBTYPE) continue;
                            for (size_t i = 0; i < subtype_names.size(); i++) {
                                if (t.name == subtype_names[i]) freq[i]++;
                            }
                        }
                    }

                    // Sort by frequency descending (most prominent first)
                    std::vector<size_t> order(subtype_names.size());
                    for (size_t i = 0; i < order.size(); i++) order[i] = i;
                    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                        return freq[a] > freq[b];
                    });

                    std::vector<LegalAction> type_choices;
                    for (size_t idx : order) {
                        LegalAction la(PASS_PRIORITY, entity, "Choose creature type: " + subtype_names[idx]);
                        la.category = ActionCategory::CHOOSE_TYPE;
                        // Menu position distinguishes the same-entity options (all carry the
                        // naming permanent). Bounded by the normalizer; the list is small.
                        la.option_ordinal = static_cast<int>(type_choices.size());
                        type_choices.push_back(la);
                    }

                    // Latched-answer site (tag SBE_LATCHED): the type choice is a
                    // loop-top pending decision. Re-finding is deterministic — the
                    // suspension freezes the state mid-apply, and on resume the
                    // re-run of apply_permanent_components reaches this same entity
                    // (mEntities order is stable, earlier permanents are idempotent
                    // no-ops) with chosen_type still empty, rebuilding the identical
                    // menu (subtypes/freq derive from the frozen ECS).
                    uint64_t key = pq_key(SbeSite::ETB_CHOOSE_TYPE,
                                          perm_ref.controller == Zone::PLAYER_A,
                                          entity, type_choices.size());
                    int choice = -1;
                    if (!pq_take_latched(key, &choice)) {
                        game_log("Choose a creature type for %s:\n", perm_ref.name.c_str());
                        if (in_main_loop()) {
                            // Park the choice (priority persisted at the chooser) and
                            // suspend mid-apply: the caller chain (this function, then
                            // state_based_effects) early-returns cooperatively.
                            pq_arm_sbe(key, std::move(type_choices), perm_ref.controller,
                                       /*decision_source=*/0);
                            return;
                        }
                        // Blocking fallback for an SBE call outside the main loop.
                        // Since the pregame gate (Batch 13) even the preplaced-preset
                        // SBE pass runs inside the loop (a preplaced Cavern's choice
                        // rides SBE_LATCHED through the gate), so this is defensive
                        // only.
                        bool prev_priority = cur_game.player_a_has_priority;
                        cur_game.player_a_has_priority = (perm_ref.controller == Zone::PLAYER_A);
                        choice = InputLogger::instance().get_input(type_choices);
                        cur_game.player_a_has_priority = prev_priority;
                    }
                    perm_ref.chosen_type = subtype_names[order[static_cast<size_t>(choice)]];
                    game_log("%s chose creature type: %s\n",
                             player_name(perm_ref.controller).c_str(), perm_ref.chosen_type.c_str());
                }
                }
            }

            // ETBReplacement: choose a card name (Disruptor Flute). Candidates are the
            // distinct vocab cards in the opponent's deck; the chosen name keys this
            // permanent's Card.NamedCard RaiseCost / CantBeActivated statics.
            if (card_data.has_etb_name_card) {
                auto &perm_ref = global_coordinator.GetComponent<Permanent>(entity);
                if (perm_ref.chosen_name.empty()) {
                    Zone::Ownership opp = (perm_ref.controller == Zone::PLAYER_A)
                        ? Zone::PLAYER_B : Zone::PLAYER_A;
                    // Distinct opponent-owned vocab cards (whole deck; empty ValidCards$
                    // filter = no restriction, so lands are included), built by the shared
                    // helper also used by Cabal Therapy's SP$ NameCard.
                    std::vector<std::string> names;
                    std::vector<LegalAction> name_choices =
                        build_name_card_choices(mEntities, opp, /*valid_filter=*/"", names);
                    if (!name_choices.empty()) {
                        // Latched-answer site (tag SBE_LATCHED), same re-derivation
                        // contract as the choose-type site above: the resumed
                        // apply_permanent_components re-reaches this entity with
                        // chosen_name still empty and rebuilds the identical menu
                        // (the opponent's deck contents are frozen).
                        uint64_t key = pq_key(SbeSite::ETB_NAME_CARD,
                                              perm_ref.controller == Zone::PLAYER_A,
                                              entity, name_choices.size());
                        int choice = -1;
                        if (!pq_take_latched(key, &choice)) {
                            game_log("Choose a card name for %s:\n", perm_ref.name.c_str());
                            if (in_main_loop()) {
                                pq_arm_sbe(key, std::move(name_choices), perm_ref.controller,
                                           /*decision_source=*/0);
                                return;
                            }
                            // Blocking fallback for an SBE call outside the main
                            // loop (defensive only since the Batch 13 pregame gate).
                            bool prev_priority = cur_game.player_a_has_priority;
                            cur_game.player_a_has_priority = (perm_ref.controller == Zone::PLAYER_A);
                            choice = InputLogger::instance().get_input(name_choices);
                            cur_game.player_a_has_priority = prev_priority;
                        }
                        perm_ref.chosen_name = names[static_cast<size_t>(choice)];
                        game_log("%s names card: %s\n",
                                 player_name(perm_ref.controller).c_str(), perm_ref.chosen_name.c_str());
                    }
                }
            }

        } else {  // off battlefield, check to remove
            // Shared strip (game_queries.cpp): clears equipment attachment links (704.5n) before
            // removing Permanent/Creature/Damage. Same helper add_to_zone uses for the
            // same-resolution-return entry reset (CR 400.7).
            strip_permanent_components(entity);
        }
    }

    // A token that left the battlefield is first put into its actual destination zone and
    // then ceases to exist as a state-based action (CR 111.7 / 704.5d). The move to that zone
    // already happened (firing the appropriate CARD_CHANGED_ZONE at the mover), so this cleanup
    // only reports where the token went and removes the now-empty entity. Moving to a
    // non-graveyard zone (hand/library/exile) is NOT destruction — CR 701.7 destruction is a
    // battlefield->graveyard move only — so it must never be logged or treated as "destroyed"
    // (that would wrongly imply a dies event). A token that genuinely went to the graveyard did
    // die (CR 704.5d); that death was already logged and its dies-triggers queued by the mover
    // that put it there, so here we only note that it ceases to exist. Done after iteration to
    // avoid invalidating iterators.
    for (auto e : tokens_to_destroy) {
        const char *tok_name = global_coordinator.entity_has_component<Token>(e)
            ? global_coordinator.GetComponent<Token>(e).name.c_str() : "Token";
        Zone::ZoneValue dest = global_coordinator.GetComponent<Zone>(e).location;
        game_log("%s (token) is put into %s and ceases to exist.\n",
                 tok_name, zone_display_name(dest));
        global_coordinator.DestroyEntity(e);
    }

}

// Human-readable name of a zone, for narrative logging.
static const char *zone_display_name(Zone::ZoneValue loc) {
    switch (loc) {
        case Zone::LIBRARY:     return "the library";
        case Zone::BATTLEFIELD: return "the battlefield";
        case Zone::HAND:        return "its owner's hand";
        case Zone::STACK:       return "the stack";
        case Zone::GRAVEYARD:   return "the graveyard";
        case Zone::EXILE:       return "exile";
        case Zone::SIDEBOARD:   return "the sideboard";
    }
    return "another zone";
}

// 702.131b: Ascend on a permanent — any time its controller controls ten or more
// permanents and doesn't yet have the city's blessing, they get the city's blessing
// for the rest of the game (a one-way latch; never lost once gained, 702.131c). Run as
// part of the continuous-effects preamble each SBA pass. The keyword lives on the source
// CardData, so this also covers non-creature permanents that have Ascend.
void StateManager::update_city_blessing(Game &game) {
    bool ascend_a = false, ascend_b = false;
    int perms_a = 0, perms_b = 0;
    for (auto entity : mEntities) {
        if (!is_battlefield_permanent(entity)) continue;
        Zone::Ownership ctrl = global_coordinator.GetComponent<Permanent>(entity).controller;
        if (ctrl == Zone::PLAYER_A) ++perms_a;
        else if (ctrl == Zone::PLAYER_B) ++perms_b;
        if (global_coordinator.entity_has_component<CardData>(entity)) {
            auto &cd = global_coordinator.GetComponent<CardData>(entity);
            for (const auto &kw : cd.keywords)
                if (kw == "Ascend") {
                    if (ctrl == Zone::PLAYER_A) ascend_a = true;
                    else if (ctrl == Zone::PLAYER_B) ascend_b = true;
                }
        }
    }
    auto grant = [](Entity pe, int perms, const char *who) {
        auto &pl = global_coordinator.GetComponent<Player>(pe);
        if (!pl.has_city_blessing && perms >= 10) {
            pl.has_city_blessing = true;
            game_log("%s gets the city's blessing.\n", who);
        }
    };
    if (ascend_a) grant(game.player_a_entity, perms_a, "Player A");
    if (ascend_b) grant(game.player_b_entity, perms_b, "Player B");
}

// Applies mana abilities to lands based on the land subtypes in perm.types.
// Type-changing effects (Blood Moon, etc.) modify perm.types before this runs.
void StateManager::apply_land_abilities(Entity entity) {
    // assumes called with entity that has permanent component and is on battlefield and is land
    auto &perm = global_coordinator.GetComponent<Permanent>(entity);
    std::vector<std::string> land_subtypes;
    for (auto &type : perm.types) {
        if (type.kind == SUBTYPE && is_basic_land_subtype(type.name)) {
            land_subtypes.push_back(type.name);
        }
    }
    if (land_subtypes.empty()) return;
    // find mana abilities for corresponding land subtype
    for (auto subtype : land_subtypes) {
        Colors required_color = mana_color_for_subtype(subtype);
        if (required_color == NO_COLOR) continue;

        // Skip only if this exact color ability already exists
        auto &perm_abilities = perm.abilities;
        bool already_present = false;
        for (const auto &ab : perm_abilities) {
            if (ab.category == "AddMana" && ab.color == required_color && ab.amount == 1) {
                already_present = true;
                break;
            }
        }
        if (already_present) continue;

        Ability mana_ability;
        mana_ability.ability_type = Ability::ACTIVATED;
        mana_ability.category = "AddMana";
        mana_ability.color = required_color;
        mana_ability.amount = 1;
        mana_ability.tap_cost = true;
        mana_ability.subtype_derived = true;

        mana_ability.source = entity;
        perm_abilities.push_back(mana_ability);
    }
}

// Re-merge the rest-of-game activated-ability grants recorded on
// Permanent::animate_granted_abilities (DB$ Animate | Abilities$ ... | Duration$ Permanent —
// Urza's Saga chapters I & II) into perm.abilities. The layer-6 ability-removal strip
// (recompute_abilities, Magus of the Moon / Humility) erases perm.abilities each pass and the
// other persistent sources (CardData copies, subtype mana, keywords) are re-derived by this
// SBE pass — this is the matching re-derive for resolved "gains [activated ability]" grants,
// so they come back once the remover leaves. Deduped, so it's a no-op in steady state.
static void remerge_animate_granted_abilities(Entity entity) {
    auto &perm = global_coordinator.GetComponent<Permanent>(entity);
    for (const auto &granted : perm.animate_granted_abilities) {
        bool present = false;
        for (auto &existing : perm.abilities)
            if (existing.identical_activated_ability(granted)) { present = true; break; }
        if (!present) perm.abilities.push_back(granted);
    }
}

static Ability keyword_triggered_ability(const std::string &keyword);

void StateManager::apply_keyword_abilities(Entity entity) {
    if (!global_coordinator.entity_has_component<Creature>(entity)) return;
    auto &cr = global_coordinator.GetComponent<Creature>(entity);
    auto &perm_abilities = global_coordinator.GetComponent<Permanent>(entity).abilities;

    for (const auto &kw : cr.keywords) {
        Ability ab = keyword_triggered_ability(kw);
        if (ab.trigger_on == 0) continue;

        bool already_present = false;
        for (const auto &existing : perm_abilities) {
            if (existing.ability_type == Ability::TRIGGERED &&
                existing.category == ab.category &&
                existing.trigger_on == ab.trigger_on) {
                already_present = true;
                break;
            }
        }
        if (already_present) continue;

        ab.source = entity;
        perm_abilities.push_back(ab);
    }
}

// Maps keywords to their corresponding triggered abilities.
// Returns an ability with trigger_on == 0 if the keyword has no triggered ability.
static Ability keyword_triggered_ability(const std::string &keyword) {
    Ability ab;
    if (keyword == "Prowess") {
        ab.ability_type = Ability::TRIGGERED;
        ab.trigger_on = Events::NONCREATURE_SPELL_CAST;
        ab.trigger_valid_player_is_controller = true;
        ab.category = "ProwessBonus";
        ab.amount = 1;
    } else if (keyword == "Exalted") {
        ab.ability_type = Ability::TRIGGERED;
        ab.trigger_on = Events::CREATURE_ATTACKED_ALONE;
        ab.trigger_valid_player_is_controller = true;
        ab.category = "ExaltedBonus";
        ab.amount = 1;
    } else if (keyword.rfind("Mobilize:", 0) == 0) {
        // Mobilize N (702.176): whenever this creature attacks, create N tapped and
        // attacking 1/1 red Warrior creature tokens; sacrifice them at the beginning
        // of the next end step. trigger_only_self restricts it to this creature attacking.
        ab.ability_type = Ability::TRIGGERED;
        ab.trigger_on = Events::CREATURE_ATTACKED;
        ab.trigger_only_self = true;
        ab.category = "Mobilize";
        ab.amount = static_cast<size_t>(std::stoi(keyword.substr(9)));
    }
    return ab;
}

// Rule 613.1d — Layer 4: type-changing continuous effects.
// Resets land types to CardData originals, then applies type-changing effects
// sorted by timestamp (rule 613.7: later timestamp wins within the same layer).
// Regenerates subtype-derived mana abilities after types are finalized.
// Parse a " & "-delimited AddType$ card-type list (Kaito: "Creature & Ninja") into Type objects,
// classifying each token as TYPE / SUBTYPE / SUPERTYPE the same way the printed Types: line is
// (all_types / all_subtypes / all_supertypes). Used by the self card-type animate applier.
static std::vector<Type> parse_self_added_types(const std::string &spec) {
    std::vector<Type> out;
    size_t p = 0;
    while (p < spec.size()) {
        size_t sep = spec.find(" & ", p);
        if (sep == std::string::npos) sep = spec.size();
        std::string name = spec.substr(p, sep - p);
        if (!name.empty()) {
            Type t;
            t.name = name;
            if      (all_subtypes.find(name) != all_subtypes.end())   t.kind = SUBTYPE;
            else if (all_types.find(name) != all_types.end())         t.kind = TYPE;
            else if (all_supertypes.find(name) != all_supertypes.end()) t.kind = SUPERTYPE;
            else t.kind = SUBTYPE;
            out.push_back(t);
        }
        p = (sep < spec.size()) ? sep + 3 : sep;
    }
    return out;
}

// Layer 4 (CR 613.1d) — conditionally-active self card-type animate statics (Kaito, Bane of
// Nightmares). For each active "Continuous" static whose AddType$ names CARD types and whose
// Affected$ references the source itself, add those types to the source while the static's
// condition holds and remove them when it lapses. If the added types include "Creature", the
// source becomes a real creature: a Creature/Damage component is bootstrapped (so layers 6/7 set
// its P/T and grant its keyword this same pass) when active, and stripped when inactive — unless
// the source is a creature by a permanent means (printed face or an Animate). RemoveCardTypes$
// drops the source's printed CARD types other than Planeswalker ("that's still a planeswalker").
static void apply_self_animate_statics() {
    for (auto &a : g_active_statics) {
        if (a.suppressed) continue;
        if (a.sa->category != "Continuous") continue;
        if (a.sa->add_type.empty() || a.sa->add_type == "AllNonBasicLandType") continue;
        if (a.sa->affected.find("Self") == std::string::npos) continue;
        if (!global_coordinator.entity_has_component<Permanent>(a.entity)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(a.entity);

        std::vector<Type> added = parse_self_added_types(a.sa->add_type);
        bool adds_creature = false;
        for (const auto &t : added)
            if (t.kind == TYPE && t.name == "Creature") { adds_creature = true; break; }

        // Strip this static's added types first so the grant is idempotent across passes and lapses
        // cleanly when the condition turns off (re-added below only while the gate holds).
        for (const auto &t : added) perm.types.erase(t);

        if (a.condition_met) {
            // RemoveCardTypes$ True: drop the printed CARD types other than Planeswalker before
            // adding the new ones (a planeswalker becoming a creature stays a planeswalker, CR 306).
            if (a.sa->remove_card_types) {
                for (auto it = perm.types.begin(); it != perm.types.end();) {
                    if (it->kind == TYPE && it->name != "Planeswalker")
                        it = perm.types.erase(it);
                    else
                        ++it;
                }
            }
            for (const auto &t : added) perm.types.insert(t);

            if (adds_creature && !global_coordinator.entity_has_component<Creature>(a.entity)) {
                Creature cr;
                cr.base_power = 0;      // layer 7b's SetPower$/SetToughness$ override this to 3/4
                cr.base_toughness = 0;
                if (global_coordinator.entity_has_component<CardData>(a.entity))
                    cr.keywords = global_coordinator.GetComponent<CardData>(a.entity).keywords;
                global_coordinator.AddComponent(a.entity, cr);
                if (!global_coordinator.entity_has_component<Damage>(a.entity)) {
                    Damage dmg;
                    dmg.damage_counters = 0;
                    global_coordinator.AddComponent(a.entity, dmg);
                }
                // Summoning sickness (CR 302.6 / 508.1a): a permanent that becomes a creature can
                // attack only if its controller has controlled it continuously since their most
                // recent turn began. Approximate with "entered this turn" — sick the turn it enters,
                // able to attack thereafter (untap clears it for permanents that keep their Creature).
                perm.has_summoning_sickness = (perm.entered_on_turn == cur_game.turn);

                // Ninjutsu (CR 702.49e): a planeswalker put onto the battlefield "tapped and
                // attacking" had its enters-attacking mark left pending by apply_permanent_components
                // (it had no Creature yet). Consume it now that the Creature exists.
                auto pea = cur_game.pending_enters_attacking.find(a.entity);
                if (pea != cur_game.pending_enters_attacking.end()) {
                    auto &ncr = global_coordinator.GetComponent<Creature>(a.entity);
                    ncr.is_attacking = true;
                    ncr.attack_target = pea->second;
                    ncr.is_blocked = false;
                    game_log("%s is attacking (ninjutsu).\n", entity_name(a.entity).c_str());
                    cur_game.pending_enters_attacking.erase(pea);
                }
            }
        } else if (adds_creature) {
            // Gate off: strip a Creature/Damage that exists ONLY because of this static (the source
            // is not a creature by its printed face and isn't animated by another effect).
            bool printed_creature =
                global_coordinator.entity_has_component<CardData>(a.entity) &&
                is_creature_card(global_coordinator.GetComponent<CardData>(a.entity));
            bool animated = perm.animate_make_creature || perm.animate_make_creature_eot;
            if (!printed_creature && !animated) {
                if (global_coordinator.entity_has_component<Creature>(a.entity))
                    global_coordinator.RemoveComponent<Creature>(a.entity);
                if (global_coordinator.entity_has_component<Damage>(a.entity))
                    global_coordinator.RemoveComponent<Damage>(a.entity);
            }
        }
    }
}

// Layer 4 (CR 613.1d) — general additive card-type statics over an Affected$ permanent filter
// (Mycosynth Lattice: "All permanents are artifacts in addition to their other types."). For each
// active "Continuous" static whose AddType$ names types and whose Affected$ is a general permanent
// filter — NOT the Self self-animate (apply_self_animate_statics), NOT the Land.nonBasic land-
// subtype setter (Blood Moon, which strips/sets land subtypes via RemoveLandTypes$), NOT the
// AllNonBasicLandType self-CDA — add those types to EVERY matching battlefield permanent in
// addition to its existing types. Self-cleaning: each pass first strips the types this subsystem
// recorded on each permanent (static_added_types), then re-adds for the currently-active statics,
// so the grant lapses the instant the source leaves the battlefield. Only genuinely-new types are
// recorded (a type the permanent already has is left untracked and never erased).
static void apply_global_addtype_statics(const std::set<Entity> &entities) {
    // Strip last pass's globally-added types from every permanent before re-deriving.
    for (auto entity : entities) {
        if (!is_battlefield_permanent(entity)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(entity);
        if (perm.static_added_types.empty()) continue;
        for (const auto &t : perm.static_added_types) perm.types.erase(t);
        perm.static_added_types.clear();
    }
    for (auto &a : g_active_statics) {
        if (a.suppressed) continue;
        if (a.sa->category != "Continuous") continue;
        if (a.sa->add_type.empty() || a.sa->add_type == "AllNonBasicLandType") continue;
        if (a.sa->remove_land_types) continue;                  // Blood Moon land-subtype setter
        if (a.sa->affected == "Land.nonBasic") continue;        // pure land type-changer form
        if (!affected_is_general_filter(a.sa->affected)) continue;  // excludes Self/EquippedBy/empty
        std::vector<Type> added = parse_self_added_types(a.sa->add_type);
        MatchCtx ctx;
        ctx.controller = a.controller;   // YouCtrl/OppCtrl reference (CR 109.5)
        ctx.source = a.entity;           // .Other self-exclusion
        extract_static_cmc_bound(a.sa->affected, ctx);
        for (auto entity : entities) {
            if (!is_battlefield_permanent(entity)) continue;
            if (!permanent_matches_filter(entity, a.sa->affected, ctx)) continue;
            auto &perm = global_coordinator.GetComponent<Permanent>(entity);
            for (const auto &t : added)
                if (perm.types.insert(t).second) perm.static_added_types.insert(t);
        }
    }
}

void StateManager::apply_type_changing_effects() {
    // Layer 4 reapply of DB$ Animate "becomes ..." type grants (Guide of Souls). These are
    // baked onto each permanent (animate_added_types) by the Animate effect rather than sourced
    // from a battlefield static, so reassert them here every pass — survives any same-pass type
    // reset (e.g. the land type-changer below) and the rest of the game.
    for (auto entity : mEntities) {
        if (!is_battlefield_permanent(entity)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(entity);
        for (const auto &t : perm.animate_added_types) perm.types.insert(t);
        // "Until end of turn" Animate type grants (CR 514.2) — reasserted each pass while live;
        // the CLEANUP step erases the bucket so they lapse at end of turn.
        for (const auto &t : perm.animate_added_types_eot) perm.types.insert(t);
    }

    // Self card-type animate statics (Kaito's "During your turn ... he's a 3/4 Ninja creature with
    // hexproof that's still a planeswalker"). A "Continuous" static whose AddType$ names CARD types
    // and whose Affected$ references the source itself (Permanent.Self) conditionally adds those
    // types to the source (CR 613 layer 4). When the gate (condition_met) holds the types are added
    // and, if "Creature" is among them, a Creature/Damage component is bootstrapped so the source is
    // a real creature this pass (its P/T set in 7b, AddKeyword$ granted in 6); when the gate lapses
    // the added types and the bootstrapped Creature are stripped. Runs before the land type-changer
    // (a distinct Land.nonBasic Affected$ form) and its early-out below.
    apply_self_animate_statics();

    // Self-CDA "is every nonbasic land type" (Planar Nexus: AddType$ AllNonBasicLandType,
    // CharacteristicDefining$ True, Affected$ Card.Self). Add the full set of nonbasic land
    // subtypes to the source permanent itself. Idempotent (a set), reasserted every pass; a
    // later Land.nonBasic type-setter (Blood Moon) still wins because its reset/insert runs
    // below. General for any AllNonBasicLandType source.
    static const char *kNonBasicLandTypes[] = {"Desert", "Gate",        "Lair",  "Locus",
                                               "Mine",   "Power-Plant", "Sphere", "Tower",
                                               "Urza's", "Cave"};
    for (auto &a : g_active_statics) {
        if (a.suppressed) continue;
        if (a.sa->add_type != "AllNonBasicLandType") continue;
        if (!global_coordinator.entity_has_component<Permanent>(a.entity)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(a.entity);
        for (const char *t : kNonBasicLandTypes) perm.types.insert({SUBTYPE, t});
    }

    // Collect type-changing statics from the already-populated g_active_statics.
    struct TypeChanger {
        ActiveStatic *as;
        size_t timestamp;  // source permanent's ETB timestamp
    };
    std::vector<TypeChanger> changers;
    for (auto &a : g_active_statics) {
        if (a.suppressed) continue;
        if (a.sa->add_type.empty()) continue;
        // AllNonBasicLandType is the self-CDA handled above, not a Land.nonBasic affector.
        if (a.sa->add_type == "AllNonBasicLandType") continue;
        if (!global_coordinator.entity_has_component<Permanent>(a.entity)) continue;
        auto &src_perm = global_coordinator.GetComponent<Permanent>(a.entity);
        changers.push_back({&a, src_perm.timestamp_entered_battlefield});
    }

    if (changers.empty()) {
        // No land-subtype changers, but a general AddType static (Mycosynth Lattice) may still
        // apply card types to every permanent. Run it and return.
        apply_global_addtype_statics(mEntities);
        return;
    }

    // Sort by timestamp ascending — later entries override earlier ones on the same permanent.
    std::sort(changers.begin(), changers.end(),
              [](const TypeChanger &a, const TypeChanger &b) { return a.timestamp < b.timestamp; });

    for (auto entity : mEntities) {
        if (!is_battlefield_permanent(entity)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(entity);

        bool is_land = false;
        bool is_basic = false;
        for (auto &t : perm.types) {
            if (t.kind == TYPE && t.name == "Land") is_land = true;
            if (t.kind == SUPERTYPE && t.name == "Basic") is_basic = true;
        }
        if (!is_land || is_basic) continue;

        // Find the winning (latest-timestamp) type-changing effect that affects this land.
        // Because changers is sorted ascending, the last match wins.
        const TypeChanger *winner = nullptr;
        for (auto &tc : changers) {
            if (tc.as->sa->affected == "Land.nonBasic") {
                winner = &tc;  // later entry overwrites
            }
        }
        if (!winner) continue;

        // Reset land subtypes to CardData originals
        if (global_coordinator.entity_has_component<CardData>(entity)) {
            auto &card_data = global_coordinator.GetComponent<CardData>(entity);
            // Restore subtypes from CardData
            std::set<Type> new_types;
            for (auto &t : card_data.types) {
                if (t.kind != SUBTYPE) new_types.insert(t);
            }
            // Non-subtype types come from CardData; subtypes are replaced below
            perm.types = new_types;
        } else {
            // Token land — strip existing land subtypes
            std::set<Type> new_types;
            for (auto &t : perm.types) {
                if (t.kind == SUBTYPE && is_basic_land_subtype(t.name)) continue;
                new_types.insert(t);
            }
            perm.types = new_types;
        }

        // Apply the winning type
        if (winner->as->sa->remove_land_types) {
            // Already stripped above; add the new subtype
            perm.types.insert({SUBTYPE, winner->as->sa->add_type});
            // 305.7: setting a land's subtype to a basic land type makes it lose all
            // abilities generated from its rules text (printed activated/triggered/static
            // and any scripted mana ability). Suppress its statics now so layers 6/7 skip
            // them, and record it so recompute_abilities (after layer 7) erases the rest;
            // the subtype-derived mana ability regenerated just below is kept.
            g_type_set_lands.push_back(entity);
            for (auto &a : g_active_statics)
                if (a.entity == entity) a.suppressed = true;
        }

        // Strip subtype-derived mana abilities and regenerate from new types
        auto &abilities = perm.abilities;
        abilities.erase(std::remove_if(abilities.begin(), abilities.end(),
                                       [](const Ability &ab) { return ab.subtype_derived; }),
                        abilities.end());
        apply_land_abilities(entity);
    }

    // General additive AddType statics run AFTER the land-subtype changer so a permanent whose
    // land subtypes were just reset (Blood Moon) still gains its global type (Mycosynth's Artifact).
    apply_global_addtype_statics(mEntities);
}

// Preamble for the continuous-effects engine: rebuild the g_active_statics cache from
// every battlefield permanent and evaluate each gathered static's condition exactly
// once (stored in ActiveStatic::condition_met) so every layer applier reads the same
// result. Resetting each creature's static_*_bonus here means continuous buffs are
// rebuilt from scratch each pass and need no per-ability revert.
void StateManager::gather_active_statics(Game &game) {
    (void)game;
    g_active_statics.clear();
    g_type_set_lands.clear();

    for (auto entity : mEntities) {
        // Phased-out permanents are treated as nonexistent (702.26e): skip them entirely.
        // Their cached P/T and keywords are rebuilt from base on a later pass once they
        // phase back in, so there is nothing to reset here.
        if (!is_battlefield_permanent(entity)) continue;

        auto &perm = global_coordinator.GetComponent<Permanent>(entity);
        // Layer-6 removal is recomputed from scratch each pass: recompute_abilities (after
        // layer 7) re-flags every permanent still affected, so clearing here makes the flag
        // lapse the moment the removal source leaves the battlefield.
        perm.abilities_removed = false;
        if (global_coordinator.entity_has_component<Creature>(entity)) {
            auto &cr = global_coordinator.GetComponent<Creature>(entity);
            cr.static_power_bonus = 0;
            cr.static_toughness_bonus = 0;
            // Rebuild the layer-7b "set P/T" slot from scratch each pass (Humility); the 7b
            // applier re-sets it when a setter applies, recompute_pt reads it.
            cr.has_set_pt = false;
            cr.set_power = 0;
            cr.set_toughness = 0;
            // Rebuild keywords from the printed base each pass (rule 611.3a) so layer-6
            // grants (apply_layer6_ability_effects) and removals (recompute_abilities,
            // Humility) are not sticky — mirrors the static_*_bonus rebuild above. A
            // transformed DFC keeps its back-face keywords (set at transform; it is not
            // gathered for statics below), so skip the reset for it.
            if (!perm.transformed) {
                if (global_coordinator.entity_has_component<CardData>(entity))
                    cr.keywords = global_coordinator.GetComponent<CardData>(entity).keywords;
                else if (global_coordinator.entity_has_component<Token>(entity))
                    cr.keywords = global_coordinator.GetComponent<Token>(entity).keywords;
            }
            // Re-merge "until end of turn" keyword grants (e.g. Haste from Eldrazi
            // Linebreaker) onto the freshly-rebuilt base keyword list. These persist
            // across the per-pass rebuild and are cleared at cleanup (514.2).
            for (const auto &kw : cr.eot_keywords) {
                if (std::find(cr.keywords.begin(), cr.keywords.end(), kw) == cr.keywords.end())
                    cr.keywords.push_back(kw);
            }
            // Re-merge permanent keyword grants baked on by DB$ Animate (Duration$ Permanent),
            // which also survive the per-pass base rebuild (these are NOT cleared at cleanup).
            for (const auto &kw : perm.animate_added_keywords) {
                if (std::find(cr.keywords.begin(), cr.keywords.end(), kw) == cr.keywords.end())
                    cr.keywords.push_back(kw);
            }
            // Keyword counters (CR 122.1d / 702.x): a counter whose type names a keyword ability
            // grants that keyword to the permanent for as long as the counter is present (Guide
            // of Souls' flying counter ⇒ Flying). Reapplied each pass like other granted keywords.
            for (const auto &kc : perm.counters) {
                if (!is_keyword_counter_type(kc.first) || kc.second <= 0) continue;
                if (std::find(cr.keywords.begin(), cr.keywords.end(), kc.first) == cr.keywords.end())
                    cr.keywords.push_back(kc.first);
            }
            // Layer-6 keyword REMOVAL (CR 613, AB$ AnimateAll | RemoveKeywords$ — Shadowspear).
            // A keyword suppressed until end of turn by a "loses <keyword>" effect is stripped
            // from the rebuilt list after every grant is merged, so a direct reader of cr.keywords
            // agrees with the effective-keyword accessors (which also gate on removed_keywords_eot).
            if (!perm.removed_keywords_eot.empty()) {
                cr.keywords.erase(
                    std::remove_if(cr.keywords.begin(), cr.keywords.end(),
                                   [&](const std::string &k) {
                                       return perm.removed_keywords_eot.count(k) != 0;
                                   }),
                    cr.keywords.end());
            }
        }
        if (perm.transformed) {
            for (auto &sa : perm.static_abilities) sa.applied = false;
            continue;
        }
        for (auto &sa : perm.static_abilities)
            g_active_statics.push_back({entity, &sa, perm.controller, false});
    }

    // Emblems (CR 114): zoneless, unremovable continuous-effect sources owned by a player. Their
    // statics are gathered with entity 0 (no Permanent) and the emblem owner as controller, so the
    // layer appliers fan them out (e.g. Kaito's "Ninjas you control get +1/+1.") through the same
    // path as a battlefield anthem. Pointers into game.emblems[*].statics are stable for this pass.
    for (auto &emb : game.emblems)
        for (auto &sa : emb.statics)
            g_active_statics.push_back({0, &sa, emb.controller, false});

    // Evaluate only the conditions actually referenced; compute each at most once per
    // player rather than once per permanent.
    bool need_delirium_a = false, need_delirium_b = false;
    for (auto &a : g_active_statics) {
        if (a.sa->condition == "Delirium") {
            if (a.controller == Zone::PLAYER_A) need_delirium_a = true;
            else                                need_delirium_b = true;
        }
    }
    bool delirium_a = need_delirium_a ? check_delirium(Zone::PLAYER_A, mEntities) : false;
    bool delirium_b = need_delirium_b ? check_delirium(Zone::PLAYER_B, mEntities) : false;

    for (auto &a : g_active_statics) {
        if (a.sa->condition.empty() && a.sa->check_svar_expr.empty()) {
            // No Condition$/CheckSVar$ — provisionally true (a present-only IsPresent$ static
            // lands here and is gated by the present-count AND below); an unconditional static
            // stays true.
            a.condition_met = true;
        } else if (a.sa->condition == "Delirium") {
            a.condition_met = (a.controller == Zone::PLAYER_A) ? delirium_a : delirium_b;
        } else if (a.sa->condition == "PlayerTurn") {
            // Active during the source controller's own turn (Voice of Victory).
            bool a_turn = cur_game.player_a_turn;
            a.condition_met = (a.controller == Zone::PLAYER_A) ? a_turn : !a_turn;
        } else if (!a.sa->check_svar_expr.empty()) {
            // SVar-based condition (e.g. Keen-Eyed Curator: GE4 distinct card types
            // among exiled_with). a.entity is the source permanent the SVar belongs to.
            int svar_val = evaluate_sa_svar(a.sa->check_svar_expr, a.controller, a.entity);
            a.condition_met = compare_svar(svar_val, a.sa->svar_compare);
        } else {
            a.condition_met = false;  // unrecognised condition — treat as unmet
        }

        // AND in the present-count gate (IsPresent$/PresentZone$/PresentCompare$), if any.
        // It composes with any Condition$/CheckSVar$ above: the static is active only when
        // every gate it declares is satisfied (CR 604.3 — a continuous static functions only
        // while its conditions hold). Re-evaluated each SBA pass so it turns on/off live.
        if (a.condition_met && !a.sa->present_filter.empty()) {
            a.condition_met = static_present_condition_met(
                a.sa->present_filter, a.sa->present_zone, a.sa->present_compare,
                a.controller, a.entity, mEntities);
        }

        // AND in the per-source counter gate (Kaito: Affected$ ...+counters_GE1_LOYALTY — the static
        // is active only while the SOURCE has the required number of counters of the named type).
        // Re-evaluated each pass so Kaito stops being a creature the instant his loyalty hits 0.
        if (a.condition_met && !a.sa->self_counter_type.empty()) {
            int ct = 0;
            if (global_coordinator.entity_has_component<Permanent>(a.entity)) {
                auto &perm = global_coordinator.GetComponent<Permanent>(a.entity);
                auto it = perm.counters.find(a.sa->self_counter_type);
                if (it != perm.counters.end()) ct = it->second;
            }
            a.condition_met = compare_svar(ct, a.sa->self_counter_compare);
        }
    }
}

// Layer 6 (rule 613.1f): ability-adding/removing continuous effects. Today this is
// keyword grants from "Continuous" statics, granted/removed on condition transitions
// (tracked by StaticAbility::applied). P/T is handled entirely in layer 7, so this
// pass never touches static_*_bonus — only keywords and the applied/last_applied state.
void StateManager::apply_layer6_ability_effects() {
    for (auto &a : g_active_statics) {
        if (a.suppressed) continue;  // source lost all abilities (Humility) — grant is gone
        if (a.sa->category != "Continuous") continue;
        // Characteristic-defining P/T statics are pure layer 7a (no keyword); skip them
        // here exactly as the original combined loop did before the keyword branch.
        if (a.sa->characteristic_defining &&
            (!a.sa->set_power_svar.empty() || !a.sa->set_toughness_svar.empty()))
            continue;

        // A general Affected$ filter (an anthem class like "Creature.Colorless+Other+YouCtrl")
        // fans out across every matching permanent — never the single source. Its P/T delta is
        // applied entirely in layer 7's per-target loop; here we apply its keyword grant (if any)
        // to that same affected set (CR 613 layers 6 / 7c — "other creatures you control get
        // +1/+1 and have trample" must give BOTH to all matching creatures). A general-filter
        // static with no keyword has nothing to do in layer 6, so skip it (layer 7 buffs P/T).
        if (affected_is_general_filter(a.sa->affected)) {
            if (a.sa->add_keyword.empty()) continue;
            // Keywords are rebuilt from each creature's printed base every SBE pass
            // (gather_active_statics, rule 611.3a), so re-granting here each pass cannot stack.
            std::vector<Entity> targets =
                a.condition_met ? affected_permanents_for_static(a, mEntities)
                                : std::vector<Entity>{};
            bool granted_any = false;
            for (Entity e : targets) {
                if (!global_coordinator.entity_has_component<Creature>(e)) continue;
                add_keywords_from_spec(global_coordinator.GetComponent<Creature>(e),
                                       a.sa->add_keyword);
                granted_any = true;
            }
            if (a.condition_met && granted_any) {
                if (!a.sa->applied) {
                    a.sa->applied = true;
                    game_log("Creatures gain %s(%s)\n", (a.sa->add_keyword + " ").c_str(),
                             a.sa->condition.empty() ? "always" : a.sa->condition.c_str());
                }
            } else if (a.sa->applied) {
                // No affected creature this pass (or condition unmet): keywords already dropped
                // by the per-pass base rebuild; just clear state and log.
                a.sa->applied = false;
                game_log("Creatures lose %s bonus\n",
                         a.sa->condition.empty() ? "static" : a.sa->condition.c_str());
            }
            continue;
        }

        // Determine which entity receives the grant (source or attached creature).
        // Affected$ is stored verbatim (e.g. "Creature.EquippedBy" / "Creature.EnchantedBy"),
        // so match by substring; both attachment forms resolve to equipped_to.
        Entity target_entity = a.entity;
        if (affected_is_attached_target(a.sa->affected)) {
            if (!global_coordinator.entity_has_component<Permanent>(a.entity)) continue;
            target_entity = global_coordinator.GetComponent<Permanent>(a.entity).equipped_to;
        }

        // Keywords are rebuilt from base every pass (gather_active_statics), so a grant
        // that moved to a different creature (equipment re-attached) needs no manual
        // revert on the old one — just reset the applied/log state so it re-logs.
        if (a.sa->applied && a.sa->last_applied_entity != target_entity)
            a.sa->applied = false;

        if (target_entity == 0 || !global_coordinator.entity_has_component<Creature>(target_entity)) {
            // No valid target; mark unapplied so keywords re-grant when one appears.
            if (a.sa->applied) a.sa->applied = false;
            continue;
        }

        auto &cr = global_coordinator.GetComponent<Creature>(target_entity);
        const std::string name_for_log = entity_name(target_entity);

        if (a.condition_met) {
            // Re-grant onto the base-reset keyword set every pass (keywords are no longer
            // sticky); log only on the condition/target transition.
            if (!a.sa->add_keyword.empty()) add_keywords_from_spec(cr, a.sa->add_keyword);
            if (!a.sa->applied) {
                a.sa->applied = true;
                a.sa->last_applied_entity = static_cast<uint32_t>(target_entity);
                game_log("%s gains %s%s(%s)\n", name_for_log.c_str(),
                         a.sa->add_power != 0 ? (std::to_string(a.sa->add_power) + "/" +
                                                  std::to_string(a.sa->add_toughness) + " ").c_str() : "",
                         !a.sa->add_keyword.empty() ? (a.sa->add_keyword + " ").c_str() : "",
                         a.sa->condition.empty() ? "always" : a.sa->condition.c_str());
            }
        } else if (a.sa->applied) {
            // Keywords already dropped by the base rebuild; just clear state and log.
            a.sa->applied = false;
            game_log("%s loses %s bonus\n", name_for_log.c_str(),
                     a.sa->condition.empty() ? "static" : a.sa->condition.c_str());
        }
    }
}

// Strip the ".NamedCard" qualifier from an Affected$ filter, returning the remaining filter
// (e.g. "Land.NamedCard" -> "Land", "Land.NamedCard+YouCtrl" -> "Land.YouCtrl"). The NamedCard
// match is handled out-of-band against the source permanent's chosen_name (mirroring
// active_raise_cost_for), because the shared qualifier evaluator has no NamedCard token.
static std::string strip_named_card_qualifier(const std::string &filter) {
    // The head (before the first '.'/'+') is always kept; each subsequent '.'/'+' token is
    // kept with its preceding separator unless it is the NamedCard qualifier being dropped.
    size_t head_end = filter.find_first_of(".+");
    std::string out = filter.substr(0, head_end);
    size_t p = head_end;
    while (p != std::string::npos && p < filter.size()) {
        char sep = filter[p];                       // the '.' or '+' separator
        size_t nx = filter.find_first_of(".+", p + 1);
        size_t tok_end = (nx == std::string::npos) ? filter.size() : nx;
        std::string tok = filter.substr(p + 1, tok_end - (p + 1));
        if (tok != "NamedCard") out += sep + tok;
        p = nx;
    }
    return out;
}

// Resolve the permanents an AddAbility$ static affects. Reuses affected_permanents_for_static
// for the general Affected$ filter grammar, then — if the filter carried a NamedCard qualifier
// — keeps only permanents whose name equals the source permanent's chosen_name (the same
// per-source named-card state that match_named_card statics read). A source with no chosen
// name yet (ETB name choice not made) grants nothing.
static std::vector<Entity> ability_grant_targets(const ActiveStatic &as, const std::set<Entity> &entities) {
    bool named = as.sa->affected.find("NamedCard") != std::string::npos;
    std::string base_filter = named ? strip_named_card_qualifier(as.sa->affected) : as.sa->affected;
    std::string chosen;
    if (named) {
        if (!global_coordinator.entity_has_component<Permanent>(as.entity)) return {};
        chosen = global_coordinator.GetComponent<Permanent>(as.entity).chosen_name;
        if (chosen.empty()) return {};
    }
    // Build a temporary ActiveStatic view with the NamedCard-stripped filter so the shared
    // resolver evaluates the rest of the grammar (subtypes, YouCtrl, …) normally.
    ActiveStatic view = as;
    StaticAbility tmp = *as.sa;
    tmp.affected = base_filter;
    view.sa = &tmp;
    std::vector<Entity> candidates = affected_permanents_for_static(view, entities);
    if (!named) return candidates;
    std::vector<Entity> out;
    for (Entity e : candidates) {
        if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
        if (global_coordinator.GetComponent<Permanent>(e).name == chosen) out.push_back(e);
    }
    return out;
}

// Layer 6 (rule 613.1f): AddAbility$ continuous ability grants. A "Continuous" static with a
// non-empty add_ability body (Petrified Hamlet's "Lands with the chosen name have '{T}: Add
// {C}.'") attaches that activated ability to every permanent its Affected$ filter designates,
// so the recipient's controller can use it through the normal Permanent-abilities activation
// path. The grant is rebuilt from scratch every SBA pass: first strip every ability marked
// granted_by_static from all battlefield permanents, then re-add for each active grant whose
// condition is met. This makes the granted ability appear/disappear as named lands and the
// granting source enter/leave, and de-dupes per source static (one copy per recipient). It
// runs unconditionally (even with no statics) so a grant left by a source that just left the
// battlefield is cleaned up.
// Per-pass runtime state of one static-granted ability, snapshotted before the grant is
// stripped so it can be carried onto the freshly re-parsed copy in the same pass. Keyed by
// (recipient permanent, granting static, ability identity) — mirroring restore_mana_state's
// (entity, ability-identity) keying for normal mana abilities — so the ActivationLimit$
// counter (Ability::activations_this_turn) survives the rebuild instead of resetting to 0
// every continuous-effects pass. Without this a granted limited-activation ability could be
// activated more than its ActivationLimit$ allows (CR 602.5). The ability is held by value so
// identical_activated_ability can match it against the rebuilt copy regardless of vector index.
struct GrantedAbilityRuntime {
    Entity recipient = 0;
    Entity granted_by_static = 0;
    Ability ability;  // the pre-strip granted instance (carries activations_this_turn)
};

void StateManager::apply_layer6_ability_grants() {
    // Phase 1: remove all previously static-granted abilities from every battlefield permanent,
    // snapshotting their per-pass runtime counters first so Phase 2's rebuild doesn't clobber them.
    std::vector<GrantedAbilityRuntime> saved_runtime;
    for (auto entity : mEntities) {
        if (!is_battlefield_permanent(entity)) continue;
        auto &abilities = global_coordinator.GetComponent<Permanent>(entity).abilities;
        for (auto &ab : abilities) {
            if (ab.granted_by_static == 0) continue;
            if (ab.activations_this_turn == 0) continue;  // nothing to preserve
            saved_runtime.push_back({entity, ab.granted_by_static, ab});
        }
        abilities.erase(std::remove_if(abilities.begin(), abilities.end(),
                                       [](const Ability &ab) { return ab.granted_by_static != 0; }),
                        abilities.end());
    }

    // Phase 2: re-grant from each active AddAbility$ static whose condition is met.
    for (auto &a : g_active_statics) {
        if (a.suppressed) continue;  // source lost all abilities (Humility)
        if (a.sa->category != "Continuous") continue;
        if (a.sa->add_ability.empty()) continue;
        if (!a.condition_met) continue;

        std::vector<Entity> targets = ability_grant_targets(a, mEntities);
        if (targets.empty()) continue;

        // Materialize the granted ability once from the stored body (full Forge ability
        // grammar via the parser), then attach a per-recipient copy tagged with the source.
        Ability granted = parse_ability_body(a.sa->add_ability);
        if (granted.category.empty()) continue;  // unparsable body — grant nothing
        granted.granted_by_static = a.entity;

        for (Entity e : targets) {
            auto &abilities = global_coordinator.GetComponent<Permanent>(e).abilities;
            Ability copy = granted;
            copy.source = e;
            // Carry over this pass's runtime counter from the pre-strip instance of the same
            // grant on the same recipient, so ActivationLimit$ is enforced across the whole turn
            // even though continuous effects rebuild the ability every pass (CR 602.5).
            for (auto &saved : saved_runtime) {
                if (saved.recipient != e) continue;
                if (saved.granted_by_static != a.entity) continue;
                if (!saved.ability.identical_activated_ability(copy)) continue;
                copy.activations_this_turn = saved.ability.activations_this_turn;
                break;
            }
            // De-dupe: skip if an identical ability (granted or intrinsic) already exists on
            // the recipient, so a land that already has "{T}: Add {C}" isn't given a duplicate.
            bool dup = false;
            for (auto &existing : abilities) {
                if (existing.identical_activated_ability(copy)) { dup = true; break; }
            }
            if (dup) continue;
            abilities.push_back(copy);
        }
    }

    // Phase 3: re-grant from each active AddTrigger$ static whose condition is met (The Tabernacle
    // at Pendrell Vale: "All creatures have 'At the beginning of your upkeep, sacrifice this
    // creature unless you pay {1}.'"). CR 613.1f (layer 6) / 603: the whole TRIGGERED ability is
    // attached to every affected permanent. The granted trigger is placed in the recipient's
    // ability list (which the trigger scan reads); it is controller-relative — ValidPlayer$ You
    // gates it to fire only at the RECIPIENT's controller's upkeep, and Defined$ Self makes the
    // destroy act on the recipient. Like Phase 2 it is rebuilt from scratch every pass (Phase 1
    // stripped it), so the grant appears/disappears as creatures and the source enter/leave.
    for (auto &a : g_active_statics) {
        if (a.suppressed) continue;  // source lost all abilities (Humility)
        if (a.sa->category != "Continuous") continue;
        if (a.sa->add_trigger.empty()) continue;
        if (!a.condition_met) continue;

        std::vector<Entity> targets = ability_grant_targets(a, mEntities);
        if (targets.empty()) continue;

        // Materialize the granted trigger once (full trigger grammar + its Execute$ SVar), then
        // attach a per-recipient copy tagged with the source static.
        Ability granted = parse_granted_trigger(a.sa->add_trigger, a.sa->add_trigger_svar_name,
                                                a.sa->add_trigger_svar);
        if (granted.trigger_on == 0 && !granted.trigger_state_condition) continue;  // unparsable
        granted.ability_type = Ability::TRIGGERED;
        granted.granted_by_static = a.entity;

        for (Entity e : targets) {
            auto &abilities = global_coordinator.GetComponent<Permanent>(e).abilities;
            // De-dupe: skip if this same static's trigger (same category + trigger_on) already
            // sits on the recipient this pass.
            bool dup = false;
            for (auto &existing : abilities) {
                if (existing.ability_type == Ability::TRIGGERED &&
                    existing.granted_by_static == a.entity &&
                    existing.category == granted.category &&
                    existing.trigger_on == granted.trigger_on) {
                    dup = true;
                    break;
                }
            }
            if (dup) continue;
            Ability copy = granted;
            copy.source = e;
            abilities.push_back(copy);
        }
    }
}

// Layer 6 (rule 613.1f) ability REMOVAL — the counterpart to the keyword grants above.
// An effect that says an object "loses all abilities" (Humility) is applied in two phases
// so it interacts correctly with the rest of the layer engine:
//   * suppress_removed_statics() runs right after gather, before any layer applier, and
//     marks every gathered static whose SOURCE loses its abilities as suppressed. A static
//     cannot be erased from perm.static_abilities — g_active_statics holds raw pointers
//     into that vector — so the layer appliers skip suppressed entries instead. This makes
//     the affected object's CDAs, P/T pumps and keyword grants vanish in layers 4/6/7.
//   * recompute_abilities() runs after layer 7 and erases the affected object's activated/
//     triggered abilities and clears its (already base-rebuilt) keywords. They are
//     re-derived next pass by apply_permanent_components / apply_land_abilities / the
//     keyword rebuild in gather, so the removal is reversible once the effect leaves.
//
// LIMITATION: removal currently wins over same-layer grants unconditionally (it runs after
// the grant pass). Timestamp ordering between a grant and a removal within layer 6 (the
// Humility + anthem interaction, rule 613.7) is deferred to the dependency work (§5).

// Does ability-removal static `r` apply to `entity`? Honours the Affected$ filter through the
// shared permanent matcher, so "Creature" (Humility), "Land" (Toxicrene), and any other filter
// grammar all work. An unfiltered remover applies to all permanents.
static bool removal_affects(const ActiveStatic &r, Entity entity) {
    const std::string &aff = r.sa->affected;
    if (aff.empty()) return true;
    MatchCtx ctx;
    ctx.controller = r.controller;
    ctx.source = r.entity;
    return permanent_matches_filter(entity, aff, ctx);
}

// Does an ability-removal continuous effect from an ALREADY-PRESENT battlefield source apply to
// `entity` as it enters the battlefield? Needed by the "as it enters" Saga lore counter
// (CR 714.3a), which is awarded inside apply_permanent_components BEFORE this pass's layer run
// can set Permanent::abilities_removed on the fresh Permanent — but per CR 613.1 the removal
// applies immediately, so a Saga entering under Magus of the Moon enters as a Mountain with no
// lore counter and never fires chapter I. Scans battlefield statics directly (g_active_statics
// may predate this entity). Covers both removal shapes recompute_abilities handles: a
// remove_all_abilities remover (Humility/Toxicrene) matched through its Affected$ filter, and a
// remove_land_types basic-type setter (Magus/Blood Moon — hardcoded Land.nonBasic, mirroring
// apply_type_changing_effects). Simplification: only unconditional statics are considered —
// every current-vocab remover is unconditional; a conditional one would need this pass's
// condition evaluation, which hasn't run yet for the entering permanent.
static bool etb_ability_removal_applies(Entity entity, const std::set<Entity> &entities) {
    bool is_land = false, is_basic = false;
    if (global_coordinator.entity_has_component<CardData>(entity)) {
        const auto &cd = global_coordinator.GetComponent<CardData>(entity);
        for (const auto &t : cd.types) {
            if (t.kind == TYPE && t.name == "Land") is_land = true;
            if (t.kind == SUPERTYPE && t.name == "Basic") is_basic = true;
        }
    }
    for (auto src : entities) {
        if (src == entity) continue;
        if (!is_battlefield_permanent(src)) continue;
        const auto &sperm = global_coordinator.GetComponent<Permanent>(src);
        for (const auto &sa : sperm.static_abilities) {
            if (sa.category != "Continuous") continue;
            if (!sa.condition.empty() || !sa.check_svar_expr.empty() || !sa.present_filter.empty())
                continue;  // conditional — see simplification note above
            if (sa.remove_land_types) {
                if (is_land && !is_basic && sa.affected == "Land.nonBasic") return true;
                continue;
            }
            if (!sa.remove_all_abilities) continue;
            if (sa.affected.empty()) return true;
            MatchCtx ctx;
            ctx.controller = sperm.controller;
            ctx.source = src;
            if (permanent_matches_filter(entity, sa.affected, ctx)) return true;
        }
    }
    return false;
}

// Gather every active static that removes all abilities and whose condition is met
// (Humility and friends). Shared by suppress_removed_statics / recompute_abilities.
static std::vector<const ActiveStatic *> collect_ability_removers() {
    std::vector<const ActiveStatic *> removers;
    for (auto &a : g_active_statics)
        if (a.sa->remove_all_abilities && a.condition_met) removers.push_back(&a);
    return removers;
}

void StateManager::suppress_removed_statics(Game &game) {
    (void)game;
    std::vector<const ActiveStatic *> removers = collect_ability_removers();
    if (removers.empty()) return;

    for (auto &a : g_active_statics) {
        if (a.sa->remove_all_abilities) continue;  // a remover keeps its own ability
        for (auto *r : removers)
            if (removal_affects(*r, a.entity)) { a.suppressed = true; break; }
    }
}

void StateManager::recompute_abilities(Game &game) {
    (void)game;
    std::vector<const ActiveStatic *> removers = collect_ability_removers();
    if (removers.empty() && g_type_set_lands.empty()) return;

    for (auto entity : mEntities) {
        if (!is_battlefield_permanent(entity)) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(entity);

        // (a) "Loses all abilities" (Humility) — a full clear, intrinsic mana included.
        // Collect which removers affect this entity so a grant from a remover that ALSO grants
        // an ability (Toxicrene: "all lands have '{T}: Add any color' and lose all OTHER
        // abilities" — one static) survives its own removal. A pure remover (Humility) grants
        // nothing, so nothing is preserved and the clear is total, as before.
        std::vector<Entity> affecting_removers;
        for (auto *r : removers)
            if (removal_affects(*r, entity)) affecting_removers.push_back(r->entity);
        bool full_removal = !affecting_removers.empty();

        // (b) 305.7 land set to a basic type — loses its rules-text abilities but keeps
        //     the regenerated intrinsic (subtype-derived) mana ability.
        bool type_set = std::find(g_type_set_lands.begin(), g_type_set_lands.end(), entity)
                        != g_type_set_lands.end();

        if (!full_removal && !type_set) continue;

        // Mark the removal for the CardData/Token-reading paths (trigger collection, the Saga
        // chapter machinery) — erasing perm.abilities below only strips the ACTIVATED copies;
        // innate TRIGGERED abilities are matched straight off CardData at trigger time and must
        // consult this flag (the pre-fix bug: Urza's Saga chapters kept firing under a Magus of
        // the Moon because trigger collection never saw the layer-6 removal).
        perm.abilities_removed = true;

        // These are re-derived next pass by apply_permanent_components / apply_land_abilities
        // / the keyword rebuild in gather, so removal is reversible once the effect leaves.
        auto &abilities = perm.abilities;
        if (full_removal) {
            // Drop every ability except one granted by a remover that affects this entity
            // (the remover's own "and gains ..." clause, e.g. Toxicrene's any-color mana).
            abilities.erase(std::remove_if(abilities.begin(), abilities.end(),
                                           [&](const Ability &ab) {
                                               return std::find(affecting_removers.begin(),
                                                                affecting_removers.end(),
                                                                ab.granted_by_static) ==
                                                      affecting_removers.end();
                                           }),
                            abilities.end());
        } else {  // type_set only — keep subtype-derived mana, drop printed rules-text abilities
            abilities.erase(std::remove_if(abilities.begin(), abilities.end(),
                                           [](const Ability &ab) { return !ab.subtype_derived; }),
                            abilities.end());
        }
        if (global_coordinator.entity_has_component<Creature>(entity))
            global_coordinator.GetComponent<Creature>(entity).keywords.clear();
    }
}

// Layer 7 (rule 613.4): power/toughness continuous effects, in sublayer order.
//   7a — characteristic-defining abilities set base P/T (rule 604.3), applied first
//        and unconditionally; recompute_pt then layers counters/pumps/etc. on top.
//   7c — additive +N/+N statics, collected as timestamp-tagged ContinuousEffects,
//        ordered per 613.7/613.8, then accumulated into static_*_bonus. Accumulation
//        is commutative so the result matches the old fixed-order sum; the ordering
//        machinery is wired in so future non-commutative layer-7 effects order
//        correctly. (Sublayers 7b "set" and 7d "switch" have storage slots on Creature
//        but no current-vocab source — see recompute_pt.)
void StateManager::apply_layer7_pt_effects() {
    // 7a — characteristic-defining base P/T.
    for (auto &a : g_active_statics) {
        if (a.suppressed) continue;
        if (a.sa->category != "Continuous") continue;
        if (!(a.sa->characteristic_defining &&
              (!a.sa->set_power_svar.empty() || !a.sa->set_toughness_svar.empty())))
            continue;
        if (!global_coordinator.entity_has_component<Creature>(a.entity)) continue;
        auto &cr = global_coordinator.GetComponent<Creature>(a.entity);
        cr.base_power = !a.sa->set_power_svar.empty()
            ? evaluate_sa_svar(a.sa->set_power_svar, a.controller, a.entity) : 0;
        cr.base_toughness = !a.sa->set_toughness_svar.empty()
            ? evaluate_sa_svar(a.sa->set_toughness_svar, a.controller, a.entity) : 0;
        a.sa->applied = true;
    }

    // 7b — non-CDA "set power/toughness to N" statics (Humility). Unlike 7a/7c (which apply
    // to the source or its equipped creature), a 7b setter may affect a whole class of
    // objects via Affected$ (e.g. Affected$ Creature = every creature). Collect the active
    // setters, then for each battlefield creature apply the latest-timestamp setter that
    // matches it (rule 613.7; recompute_pt reads has_set_pt/set_power/set_toughness).
    {
        struct SetPT { ActiveStatic *a; size_t timestamp; };
        std::vector<SetPT> setters;
        for (auto &a : g_active_statics) {
            if (a.suppressed) continue;
            if (a.sa->category != "Continuous") continue;
            if (a.sa->characteristic_defining) continue;  // CDA setters are 7a
            if (a.sa->set_power_svar.empty() && a.sa->set_toughness_svar.empty()) continue;
            if (!a.condition_met) continue;
            size_t ts = global_coordinator.entity_has_component<Permanent>(a.entity)
                ? global_coordinator.GetComponent<Permanent>(a.entity).timestamp_entered_battlefield
                : 0;
            setters.push_back({&a, ts});
        }
        if (!setters.empty()) {
            std::stable_sort(setters.begin(), setters.end(),
                             [](const SetPT &x, const SetPT &y) { return x.timestamp < y.timestamp; });
            for (auto entity : mEntities) {
                if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
                if (!is_battlefield_permanent(entity)) continue;
                const ActiveStatic *winner = nullptr;
                for (auto &s : setters) {
                    const std::string &aff = s.a->sa->affected;
                    bool match;
                    if (affected_is_attached_target(aff)) {
                        match = global_coordinator.entity_has_component<Permanent>(s.a->entity) &&
                                global_coordinator.GetComponent<Permanent>(s.a->entity).equipped_to == entity;
                    } else if (aff.find("Self") != std::string::npos) {
                        match = (s.a->entity == entity);
                    } else if (aff.find("Creature") != std::string::npos) {
                        match = true;  // Affected$ Creature — every creature
                    } else {
                        match = (s.a->entity == entity);
                    }
                    if (match) winner = s.a;  // later timestamp overwrites
                }
                if (!winner) continue;
                auto &cr = global_coordinator.GetComponent<Creature>(entity);
                cr.has_set_pt = true;
                cr.set_power = !winner->sa->set_power_svar.empty()
                    ? evaluate_sa_svar(winner->sa->set_power_svar, winner->controller, winner->entity) : 0;
                cr.set_toughness = !winner->sa->set_toughness_svar.empty()
                    ? evaluate_sa_svar(winner->sa->set_toughness_svar, winner->controller, winner->entity) : 0;
            }
        }
    }

    // 7c — additive modifications, gathered then ordered before accumulation.
    std::vector<ContinuousEffect> mods;
    for (auto &a : g_active_statics) {
        if (a.suppressed) continue;
        if (a.sa->category != "Continuous") continue;
        // Setters (CDA or not) are handled in 7a / 7b, never as additive modifiers.
        if (!a.sa->set_power_svar.empty() || !a.sa->set_toughness_svar.empty())
            continue;
        if (!a.condition_met) continue;

        // Pass the static's SOURCE entity so a source-scoped count resolves correctly — e.g.
        // Lion Sash's "+1/+1 for each +1/+1 counter on CARDNAME" (AddPower$ X, X =
        // Count$CardCounters.P1P1) must read the counters on Lion Sash itself, not return 0.
        int dp = a.sa->add_power_svar.empty()
                     ? a.sa->add_power
                     : evaluate_sa_svar(a.sa->add_power_svar, a.controller, a.entity);
        int dt = a.sa->add_toughness_svar.empty()
                     ? a.sa->add_toughness
                     : evaluate_sa_svar(a.sa->add_toughness_svar, a.controller, a.entity);
        // 613.7a — a static ability's effect has its source object's timestamp.
        size_t ts = global_coordinator.entity_has_component<Permanent>(a.entity)
            ? global_coordinator.GetComponent<Permanent>(a.entity).timestamp_entered_battlefield
            : 0;

        // Resolve the affected creature set. A general Affected$ filter (anthem:
        // "Creature.Colorless+Other+YouCtrl" on It That Heralds the End) buffs every matching
        // battlefield permanent — go through the shared resolver, which honours +Other (skip
        // self) and YouCtrl (controller scope). The EquippedBy / Self / no-Affected$ forms keep
        // the original single-target behaviour (source, or the equipped creature).
        const std::string &aff = a.sa->affected;
        bool general_filter = affected_is_general_filter(aff);
        std::vector<Entity> targets;
        if (general_filter) {
            targets = affected_permanents_for_static(a, mEntities);
        } else {
            Entity target_entity = a.entity;
            if (affected_is_attached_target(aff)) {
                if (!global_coordinator.entity_has_component<Permanent>(a.entity)) continue;
                target_entity = global_coordinator.GetComponent<Permanent>(a.entity).equipped_to;
            }
            if (target_entity != 0) targets.push_back(target_entity);
        }

        for (Entity target_entity : targets) {
            if (target_entity == 0 || !global_coordinator.entity_has_component<Creature>(target_entity))
                continue;
            ContinuousEffect e;
            e.layer    = Layer::PT;
            e.sublayer = PTSublayer::MODIFY;
            e.source   = a.entity;
            e.affected = target_entity;
            e.delta_power = dp;
            e.delta_toughness = dt;
            e.timestamp = ts;
            mods.push_back(e);
        }
    }

    order_continuous_effects(mods);
    for (const auto &e : mods) {
        if (!global_coordinator.entity_has_component<Creature>(e.affected)) continue;
        auto &cr = global_coordinator.GetComponent<Creature>(e.affected);
        cr.static_power_bonus     += e.delta_power;
        cr.static_toughness_bonus += e.delta_toughness;
    }
}

// Rule 613.11: continuous effects that modify the rules rather than characteristics,
// applied after all other continuous effects. Today: MustAttack.
void StateManager::apply_rules_modifying_effects() {
    for (auto &a : g_active_statics) {
        if (a.sa->category != "MustAttack") continue;
        if (!global_coordinator.entity_has_component<Creature>(a.entity)) continue;
        auto &cr = global_coordinator.GetComponent<Creature>(a.entity);
        cr.must_attack = a.condition_met;
        a.sa->applied = a.condition_met;
    }
}

// Recompute cached effective P/T from contributions for every battlefield creature.
void StateManager::recompute_battlefield_pt() {
    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        if (global_coordinator.GetComponent<Zone>(entity).location != Zone::BATTLEFIELD) continue;
        recompute_pt(global_coordinator.GetComponent<Creature>(entity));
    }
}

