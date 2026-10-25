#include "rules_modifying.h"

#include <algorithm>
#include <climits>

#include "../classes/game.h"  // cur_game, CastWithFlashPermission
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/static_ability.h"
#include "../components/types.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../mana_system.h"  // get_player_entity
#include "../queries/battlefield.h"
#include "../queries/characteristics.h"
#include "../queries/filters.h"
#include "../queries/players.h"
#include "../queries/types.h"
#include "../svar_eval.h"   // evaluate_svar (CantAttack dynamic-X hand count)
#include "state_manager.h"  // g_active_statics, ActiveStatic

namespace rules_mod {

static bool cant_activate_filter_matches(Entity permanent_entity, const std::string &filter,
                                         Zone::Ownership source_controller);
static std::string activation_source_name(Entity source);
static int mv_land_bound(const ActiveStatic &as, Zone::Ownership caster, const CardData &card);
static bool may_play_player_covers(
    StaticAbility::MayPlayPlayer who, Zone::Ownership controller, Zone::Ownership caster);

// Does `permanent_entity` satisfy a CantBeActivated ValidCard$ filter (a comma-OR list of
// Forge clauses, e.g. "Artifact" for Null Rod, "Artifact,Creature,Planeswalker" for Clarion
// Conqueror, or "Artifact.OppCtrl" for Karn, the Great Creator)? Routed through the shared
// permanent_matches_any so type qualifiers AND controller qualifiers (.YouCtrl/.OppCtrl) are
// honored — the controller reference is the static's own controller, so Karn's ".OppCtrl"
// correctly restricts the lock to the opponent's artifacts and never the controller's own.
static bool cant_activate_filter_matches(Entity permanent_entity, const std::string &filter,
                                         Zone::Ownership source_controller) {
    if (filter.empty()) return false;
    MatchCtx ctx;
    ctx.controller = source_controller;  // the "you" for YouCtrl/OppCtrl in the spec
    return permanent_matches_any(permanent_entity, filter, ctx);
}

// The name a NamedCard CantBeActivated static (Pithing Needle, Disruptor Flute) compares against
// its chosen name: a permanent's current name, or the name of a card in another zone (a channel
// land in hand, CR 201.2). Empty for an object with neither.
static std::string activation_source_name(Entity source) {
    if (global_coordinator.entity_has_component<Permanent>(source))
        return global_coordinator.GetComponent<Permanent>(source).name;
    if (global_coordinator.entity_has_component<CardData>(source))
        return global_coordinator.GetComponent<CardData>(source).name;
    return "";
}

// Lavinia, Azorius Renegade (CantBeCast Caster$ Opponent | cmcGT$ Land): the largest mana value
// `caster` may cast `card` with under static `as` — the number of lands the caster controls — or
// -1 when this static puts no mana-value bound on this card (not that form, not live, the caster
// is its controller, or the card is outside its ValidCard$ filter).
static int mv_land_bound(const ActiveStatic &as, Zone::Ownership caster, const CardData &card) {
    if (as.suppressed || as.sa()->category != "CantBeCast") return -1;
    if (!as.sa()->cant_cast_by_opponent || !as.sa()->cant_cast_cmc_gt_land) return -1;
    if (as.sa()->only_sorcery_speed || !as.condition_met || caster == as.controller) return -1;
    MatchCtx ctx;
    ctx.controller = caster;  // "you" reference for the ValidCard$ filter
    if (!as.sa()->cant_cast_filter.empty() && !card_matches_filter(card, as.sa()->cant_cast_filter, ctx))
        return -1;  // creature / land spells are unaffected
    return count_battlefield_matching("Land.YouCtrl", caster, 0);
}

bool mana_activation_prohibited(Entity permanent_entity) {
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.sa()->category != "CantBeActivated" || as.sa()->cant_activate_card_filter.empty()) continue;
        if (cant_activate_filter_matches(permanent_entity, as.sa()->cant_activate_card_filter, as.controller))
            return true;
    }
    return false;
}

bool activation_prohibited(Entity source) {
    bool is_permanent = global_coordinator.entity_has_component<Permanent>(source);
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.sa()->category != "CantBeActivated") continue;
        if (as.sa()->match_named_card) {
            // NamedCard (Disruptor Flute): suppress sources whose name matches the chosen name,
            // in whatever zone the ability is activated from
            if (!global_coordinator.entity_has_component<Permanent>(as.entity)) continue;
            auto &src = global_coordinator.GetComponent<Permanent>(as.entity);
            if (!src.chosen_name.empty() && src.chosen_name == activation_source_name(source))
                return true;
        } else if (is_permanent &&
                   cant_activate_filter_matches(source, as.sa()->cant_activate_card_filter,
                                                as.controller)) {
            // A type filter ("Artifact", "Artifact.OppCtrl") names permanents (CR 109.2), so it
            // never reaches a card activated from the hand or graveyard.
            return true;
        }
    }
    return false;
}

bool cast_prohibited(Zone::Ownership caster, const CardData &card, Zone::ZoneValue cast_from) {
    bool card_is_creature = is_creature_card(card);
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.sa()->category != "CantBeCast") continue;
        // Creatures are unaffected by a nonCreature restriction.
        if (as.sa()->cant_cast_filter.find("nonCreature") != std::string::npos && card_is_creature)
            continue;
        // Origin$ Graveyard,Library (Grafdigger's Cage): only spells cast from those zones are
        // prohibited. A spell cast from any other zone (e.g. the hand) is unaffected by this
        // static; when neither origin flag is set the restriction is zone-agnostic.
        if (as.sa()->cant_cast_from_graveyard || as.sa()->cant_cast_from_library) {
            bool origin_restricted =
                (cast_from == Zone::GRAVEYARD && as.sa()->cant_cast_from_graveyard) ||
                (cast_from == Zone::LIBRARY && as.sa()->cant_cast_from_library);
            if (!origin_restricted) continue;
            return true;
        }
        // Caster$ Opponent (Voice of Victory): the controller's opponents can't cast spells.
        // condition_met (e.g. Condition$ PlayerTurn) gates when the static is live; an
        // unconditional opponent-lock has condition_met == true already.
        if (as.sa()->cant_cast_by_opponent) {
            // OnlySorcerySpeed$ (Teferi, Time Raveler) is a TIMING restriction, NOT a blanket
            // "can't cast at all" — it does not prohibit the cast here; the cast-speed gate
            // enforces it via opponent_sorcery_speed_locked. Skip it in this prohibition scan.
            if (as.sa()->only_sorcery_speed) continue;
            if (!as.condition_met) continue;
            if (caster != as.controller) {  // caster is an opponent of the source
                // Lavinia, Azorius Renegade: cmcGT$ Land is a DYNAMIC bound — the opponent can't
                // cast a spell matching ValidCard$ (noncreature nonland) whose mana value exceeds
                // the number of lands THEY control. An {X} counts as 0 here (CR 107.3g, 601.3a:
                // X = 0 may make the cast legal); the announced X is bounded by max_castable_x.
                if (as.sa()->cant_cast_cmc_gt_land) {
                    int bound = mv_land_bound(as, caster, card);
                    if (bound >= 0 && card_mana_value(card) > bound) return true;
                    continue;
                }
                return true;  // blanket opponent lock (Voice of Victory)
            }
            continue;
        }
        if (as.sa()->cant_cast_limit_per_turn > 0) {
            auto &pp = global_coordinator.GetComponent<Player>(get_player_entity(caster));
            if (static_cast<int>(pp.noncreature_spells_cast_this_turn) >=
                as.sa()->cant_cast_limit_per_turn)
                return true;
            continue;
        }
        // Characteristic-based prohibition (Gaddock Teeg: "Noncreature spells with mana value 4 or
        // greater can't be cast", "…with {X} in their mana costs can't be cast"). The static's
        // ValidCard$ filter is matched against the spell's PRINTED characteristics — including the
        // numeric mana-value bound (cmcGE4) seeded via extract_static_cmc_bound and the {X}-cost
        // qualifier (hasXCost). Reached only by statics that aren't one of the special-cased forms
        // above, so it never double-applies to an origin/opponent/per-turn static. Applies to all
        // casters (Gaddock Teeg's lock is symmetric, CR 614/611 continuous prohibition).
        if (!as.sa()->cant_cast_filter.empty()) {
            MatchCtx ctx;
            ctx.controller = caster;
            extract_static_cmc_bound(as.sa()->cant_cast_filter, ctx);
            if (card_matches_filter(card, as.sa()->cant_cast_filter, ctx)) return true;
        }
    }
    return false;
}

int max_castable_x(Zone::Ownership caster, const CardData &card) {
    int cap = INT_MAX;
    if (card.x_pip_count <= 0) return cap;
    for (const auto &as : g_active_statics) {
        int bound = mv_land_bound(as, caster, card);
        if (bound < 0) continue;
        int room = bound - card_mana_value(card);
        cap = std::min(cap, room < 0 ? 0 : room / card.x_pip_count);
    }
    return cap;
}

bool opponent_sorcery_speed_locked(Zone::Ownership caster) {
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.sa()->category != "CantBeCast" || !as.sa()->only_sorcery_speed) continue;
        if (!as.condition_met) continue;
        // Teferi, Time Raveler: the lock applies to the source controller's opponents. Reuses the
        // same Caster$ Opponent CantBeCast path a cmc/land-count opponent restriction (Lavinia)
        // would key on.
        if (as.sa()->cant_cast_by_opponent && caster != as.controller) return true;
    }
    return false;
}

bool cast_with_flash_active(Zone::Ownership caster, const CardData &card) {
    for (const auto &perm : cur_game.resolved_effects.cast_with_flash_permissions) {
        if (perm.controller != caster) continue;
        // Empty filter = every spell; otherwise the spell's printed characteristics must match
        // (Teferi's grant: ValidCard$ Sorcery). MatchCtx.controller is the caster for any
        // YouCtrl/OppCtrl qualifier in the filter.
        if (perm.filter.empty()) return true;
        MatchCtx ctx;
        ctx.controller = caster;
        if (card_matches_filter(card, perm.filter, ctx)) return true;
    }
    return false;
}

// Does a MayPlayPlayer$ scope, read from the static's `controller`, include `caster`?
static bool may_play_player_covers(
    StaticAbility::MayPlayPlayer who, Zone::Ownership controller, Zone::Ownership caster) {
    switch (who) {
        case StaticAbility::MayPlayPlayer::ANY:
            return true;
        case StaticAbility::MayPlayPlayer::OPPONENT:
            return caster == opponent_of(controller);
        case StaticAbility::MayPlayPlayer::YOU:
            break;
    }
    return caster == controller;
}

FreeCastGrant free_cast_grant(Entity card, const CardData &face, Zone::Ownership caster) {
    FreeCastGrant grant;
    if (!global_coordinator.entity_has_component<Zone>(card) ||
        !global_coordinator.entity_has_component<CardData>(card))
        return grant;
    const Zone::ZoneValue zone = global_coordinator.GetComponent<Zone>(card).location;
    // The card's own face is matched as the object (its owner for YouOwn); a back face or second
    // half by that face's characteristics, in its owner's hand.
    const bool own_face = &face == &global_coordinator.GetComponent<CardData>(card);
    for (const auto &as : g_active_statics) {
        if (as.suppressed || !as.condition_met) continue;  // 613.1f: source lost all abilities (Humility)
        const StaticAbility &sa = *as.sa();
        if (sa.category != "Continuous" || !sa.may_play_without_mana_cost) continue;
        if (!may_play_player_covers(sa.may_play_player, as.controller, caster)) continue;
        if (!affected_zone_matches(sa.affected_zone, zone, /*has_zone=*/true)) continue;
        MatchCtx ctx;
        ctx.controller = as.controller;  // "you" of YouOwn: the static's controller
        ctx.source = as.entity;
        extract_static_cmc_bound(sa.affected, ctx);  // Aluren's cmcLE3
        if (!(own_face ? card_matches_filter(card, sa.affected, ctx) : card_matches_filter(face, sa.affected, ctx)))
            continue;
        grant.granted = true;
        grant.with_flash = grant.with_flash || sa.may_play_with_flash;
    }
    return grant;
}

bool attack_prohibited(Entity creature_entity) {
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.sa()->category != "CantAttack") continue;
        if (as.sa()->cant_attack_filter.empty()) continue;  // targeted / unhandled "can't attack you" form
        if (!as.condition_met) continue;                  // gated statics (IsPresent$, etc.)
        MatchCtx ctx;
        ctx.controller = as.controller;  // "you" reference for any YouCtrl/OppCtrl in the filter
        ctx.source = as.entity;
        // Resolve a dynamic X (Ensnaring Bridge: hand size) against the static's controller, so
        // "your hand" is the source controller's hand — the threshold is the same for every
        // creature, matching the card (it affects all creatures vs. its controller's hand).
        if (!as.sa()->cant_attack_x_svar.empty())
            ctx.x_bound = evaluate_svar(as.sa()->cant_attack_x_svar, as.controller, as.entity);
        if (permanent_matches_filter(creature_entity, as.sa()->cant_attack_filter, ctx)) return true;
    }
    return false;
}

int land_play_bonus(Zone::Ownership player) {
    int bonus = 0;
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.controller != player) continue;
        if (as.sa()->adjust_land_plays > 0) bonus += as.sa()->adjust_land_plays;
    }
    return bonus;
}

int land_play_limit(Zone::Ownership player) {
    return 1 + land_play_bonus(player);
}

int land_drops_remaining(Zone::Ownership player) {
    Entity player_entity = get_player_entity(player);
    if (!global_coordinator.entity_has_component<Player>(player_entity)) return 0;
    int played = static_cast<int>(
        global_coordinator.GetComponent<Player>(player_entity).lands_played_this_turn);
    int remaining = land_play_limit(player) - played;
    return remaining > 0 ? remaining : 0;
}

bool may_play_lands_from_graveyard(Zone::Ownership player) {
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.controller != player) continue;
        if (as.sa()->may_play_from_graveyard) return true;
    }
    return false;
}

bool etb_triggers_suppressed(Entity entering) {
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.sa()->category != "DisableTriggers") continue;
        if (entering != 0 && global_coordinator.entity_has_component<CardData>(entering)) {
            auto &ecd = global_coordinator.GetComponent<CardData>(entering);
            for (auto &t : ecd.types)
                if (as.sa()->disable_triggers_cause.find(t.name) != std::string::npos) return true;
        }
    }
    return false;
}

}  // namespace rules_mod
