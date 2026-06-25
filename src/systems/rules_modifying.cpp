#include "rules_modifying.h"

#include "state_manager.h"  // g_active_statics, ActiveStatic
#include "../ecs/coordinator.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/types.h"
#include "../mana_system.h"  // get_player_entity

namespace rules_mod {

bool mana_activation_prohibited(Entity permanent_entity) {
    auto &permanent = global_coordinator.GetComponent<Permanent>(permanent_entity);
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.sa->category != "CantBeActivated" || as.sa->cant_activate_card_filter.empty()) continue;
        if (as.sa->cant_activate_card_filter == "Artifact") {
            for (auto &t : permanent.types)
                if (t.kind == TYPE && t.name == "Artifact") return true;
        }
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
        } else if (as.sa->cant_activate_card_filter == "Artifact") {
            for (auto &t : permanent.types)
                if (t.kind == TYPE && t.name == "Artifact") return true;
        }
    }
    return false;
}

bool cast_prohibited(Zone::Ownership caster, bool card_is_creature) {
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.sa->category != "CantBeCast") continue;
        // Creatures are unaffected by a nonCreature restriction.
        if (as.sa->cant_cast_filter.find("nonCreature") != std::string::npos && card_is_creature)
            continue;
        auto &pp = global_coordinator.GetComponent<Player>(get_player_entity(caster));
        if (as.sa->cant_cast_limit_per_turn > 0 &&
            static_cast<int>(pp.noncreature_spells_cast_this_turn) >= as.sa->cant_cast_limit_per_turn)
            return true;
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

bool untap_prevented(Entity permanent_entity) {
    if (!global_coordinator.entity_has_component<Permanent>(permanent_entity)) return false;
    auto &permanent = global_coordinator.GetComponent<Permanent>(permanent_entity);
    for (const auto &as : g_active_statics) {
        if (as.suppressed) continue;  // 613.1f: source lost all abilities (Humility)
        if (as.sa->hidden_keyword.empty() ||
            as.sa->hidden_keyword.find("doesn't untap") == std::string::npos ||
            as.sa->affected_subtype.empty())
            continue;
        for (auto &t : permanent.types)
            if (t.name == as.sa->affected_subtype) return true;
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
