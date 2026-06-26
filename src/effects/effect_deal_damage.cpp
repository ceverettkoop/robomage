#include "effects.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <string>

#include "../cli_output.h"
#include "../components/damage.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../error.h"
#include "../game_queries.h"
#include "../mana_system.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

bool deal_damage(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Delirium-conditional damage (Unholy Heat)
    size_t dmg = ab.amount;
    // Dynamic damage (e.g. Ajani's "damage equal to the number of creatures you control",
    // NumDmg$ X with X = Count$Valid Creature.YouCtrl). Mirrors lose_life/gain_life, which
    // evaluate their dynamic amount at resolution.
    if (!ab.dynamic_amount_expr.empty()) {
        Zone::Ownership caster = global_coordinator.entity_has_component<Permanent>(ab.source)
                                     ? global_coordinator.GetComponent<Permanent>(ab.source).controller
                                     : global_coordinator.GetComponent<Zone>(ab.source).owner;
        dmg = evaluate_dynamic_amount(ab.dynamic_amount_expr, caster, orderer, ab.target);
    }
    const DamageParams *dp = std::get_if<DamageParams>(&ab.params);
    if (dp && dp->is_delirium_scale) {
        Zone::Ownership caster = global_coordinator.entity_has_component<Permanent>(ab.source)
                                     ? global_coordinator.GetComponent<Permanent>(ab.source).controller
                                     : global_coordinator.GetComponent<Zone>(ab.source).owner;
        if (check_delirium(caster, orderer->mEntities)) dmg = dp->delirium_amount;
    }
    // Defined$ Player.Opponent — "deals N damage to each opponent" (no chosen target).
    // CR 109.5: in a two-player game, "each opponent" is the source controller's single
    // opponent. Resolve the opponent from the source's controller at resolution time.
    if (ab.defined_each_opponent) {
        Zone::Ownership caster = global_coordinator.entity_has_component<Permanent>(ab.source)
                                     ? global_coordinator.GetComponent<Permanent>(ab.source).controller
                                     : global_coordinator.GetComponent<Zone>(ab.source).owner;
        Zone::Ownership opp = (caster == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
        Entity opp_entity = get_player_entity(opp);
        deal_damage_to_player(ab.source, opp_entity, dmg);
        auto &player = global_coordinator.GetComponent<Player>(opp_entity);
        game_log("Dealt %zu damage to player (now at %d life)\n", dmg, player.life_total);
        return true;
    }
    if (global_coordinator.entity_has_component<Player>(ab.target)) {
        deal_damage_to_player(ab.source, ab.target, dmg);
        auto &player = global_coordinator.GetComponent<Player>(ab.target);
        game_log("Dealt %zu damage to player (now at %d life)\n", dmg, player.life_total);
    } else if (is_planeswalker_permanent(ab.target)) {
        // Damage to a planeswalker removes that many loyalty counters (306.8).
        damage_planeswalker(ab.target, dmg);
        auto &pw = global_coordinator.GetComponent<Permanent>(ab.target);
        game_log("Dealt %zu damage to %s (loyalty now %d)\n", dmg, pw.name.c_str(), get_counters(ab.target, "LOYALTY"));
    } else {
        if (::deal_damage(ab.source, ab.target, dmg)) {
            game_log("Dealt %zu damage to creature\n", dmg);
        } else {
#ifndef NDEBUG
            fprintf(stderr, "SOURCE:");
            dump_entity(ab.source);
            fprintf(stderr, "TARGET:");
            dump_entity(ab.target);
#endif
            non_fatal_error("Damage should have fizzled prior to this");
        }
    }
    return true;
}

bool parse_deal_damage(Ability &ab, const std::string &key, const std::string &value) {
    if (key != "NumDmg") return false;
    // Check if value is numeric; if not, store as SVar key for resolution later
    if (!value.empty() && (std::isdigit(static_cast<unsigned char>(value[0])) ||
                           (value[0] == '-' && value.size() > 1 &&
                            std::isdigit(static_cast<unsigned char>(value[1]))))) {
        ab.amount = static_cast<size_t>(std::stoi(value));
    } else {
        ab.amount_svar = value;
    }
    return true;
}

}  // namespace effects
