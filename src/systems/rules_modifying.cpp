#include "rules_modifying.h"

#include "state_manager.h"  // g_active_statics, ActiveStatic
#include "../ecs/coordinator.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/types.h"
#include "../game_queries.h"  // card_matches_filter, extract_static_cmc_bound, is_creature_card
#include "../mana_system.h"  // get_player_entity

namespace rules_mod {

// Does `permanent` have any of the card types named in `filter` (a comma-separated
// CantBeActivated ValidCard$ list, e.g. "Artifact" or "Artifact,Creature,Planeswalker")?
static bool permanent_matches_type_filter(const Permanent &permanent, const std::string &filter) {
    if (filter.empty()) return false;
    size_t pos = 0;
    while (pos < filter.size()) {
        size_t comma = filter.find(',', pos);
        std::string type_name = filter.substr(pos, comma == std::string::npos ? std::string::npos
                                                                              : comma - pos);
        for (auto &t : permanent.types)
            if (t.kind == TYPE && t.name == type_name) return true;
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return false;
}

bool mana_activation_prohibited(Entity permanent_entity) {
    auto &permanent = global_coordinator.GetComponent<Permanent>(permanent_entity);
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.sa->category != "CantBeActivated" || as.sa->cant_activate_card_filter.empty()) continue;
        if (permanent_matches_type_filter(permanent, as.sa->cant_activate_card_filter)) return true;
    }
    return false;
}

bool activation_prohibited(Entity permanent_entity) {
    auto &permanent = global_coordinator.GetComponent<Permanent>(permanent_entity);
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.sa->category != "CantBeActivated") continue;
        if (as.sa->match_named_card) {
            // NamedCard (Disruptor Flute): suppress sources whose name matches the chosen name
            if (!global_coordinator.entity_has_component<Permanent>(as.entity)) continue;
            auto &src = global_coordinator.GetComponent<Permanent>(as.entity);
            if (!src.chosen_name.empty() && src.chosen_name == permanent.name) return true;
        } else if (permanent_matches_type_filter(permanent, as.sa->cant_activate_card_filter)) {
            return true;
        }
    }
    return false;
}

bool cast_prohibited(Zone::Ownership caster, const CardData &card, Zone::ZoneValue cast_from) {
    bool card_is_creature = is_creature_card(card);
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.sa->category != "CantBeCast") continue;
        // Creatures are unaffected by a nonCreature restriction.
        if (as.sa->cant_cast_filter.find("nonCreature") != std::string::npos && card_is_creature)
            continue;
        // Origin$ Graveyard,Library (Grafdigger's Cage): only spells cast from those zones are
        // prohibited. A spell cast from any other zone (e.g. the hand) is unaffected by this
        // static; when neither origin flag is set the restriction is zone-agnostic.
        if (as.sa->cant_cast_from_graveyard || as.sa->cant_cast_from_library) {
            bool origin_restricted =
                (cast_from == Zone::GRAVEYARD && as.sa->cant_cast_from_graveyard) ||
                (cast_from == Zone::LIBRARY && as.sa->cant_cast_from_library);
            if (!origin_restricted) continue;
            return true;
        }
        // Caster$ Opponent (Voice of Victory): the controller's opponents can't cast spells.
        // condition_met (e.g. Condition$ PlayerTurn) gates when the static is live; an
        // unconditional opponent-lock has condition_met == true already.
        if (as.sa->cant_cast_by_opponent) {
            if (!as.condition_met) continue;
            if (caster != as.controller) return true;  // caster is an opponent of the source
            continue;
        }
        if (as.sa->cant_cast_limit_per_turn > 0) {
            auto &pp = global_coordinator.GetComponent<Player>(get_player_entity(caster));
            if (static_cast<int>(pp.noncreature_spells_cast_this_turn) >=
                as.sa->cant_cast_limit_per_turn)
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
        if (!as.sa->cant_cast_filter.empty()) {
            MatchCtx ctx;
            ctx.controller = caster;
            extract_static_cmc_bound(as.sa->cant_cast_filter, ctx);
            if (card_matches_filter(card, as.sa->cant_cast_filter, ctx)) return true;
        }
    }
    return false;
}

int land_play_bonus(Zone::Ownership player) {
    int bonus = 0;
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.controller != player) continue;
        if (as.sa->adjust_land_plays > 0) bonus += as.sa->adjust_land_plays;
    }
    return bonus;
}

bool may_play_lands_from_graveyard(Zone::Ownership player) {
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.controller != player) continue;
        if (as.sa->may_play_from_graveyard) return true;
    }
    return false;
}

bool etb_triggers_suppressed(Entity entering) {
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.sa->category != "DisableTriggers") continue;
        if (entering != 0 && global_coordinator.entity_has_component<CardData>(entering)) {
            auto &ecd = global_coordinator.GetComponent<CardData>(entering);
            for (auto &t : ecd.types)
                if (as.sa->disable_triggers_cause.find(t.name) != std::string::npos) return true;
        }
    }
    return false;
}

}  // namespace rules_mod
