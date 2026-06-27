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
        dmg = evaluate_dynamic_amount(ab.dynamic_amount_expr, source_controller(ab.source), orderer, ab.target);
    }
    const DamageParams *dp = std::get_if<DamageParams>(&ab.params);
    if (dp && dp->is_delirium_scale) {
        if (check_delirium(source_controller(ab.source), orderer->mEntities)) dmg = dp->delirium_amount;
    }
    // Defined$ player routes — "deals N damage to <that player>": You (the source's
    // controller, e.g. Ancient Tomb's pain), Player.Opponent (each opponent — the single
    // opponent in a two-player game, CR 109.5), or TargetedController (the target permanent's
    // controller, e.g. Smash to Smithereens). For TargetedController the Destroy sub-ability
    // may have already moved the permanent this same resolution, so the controller is read via
    // last-known information (CR 608.2g/h) — Zone.controller, else the live Permanent.controller,
    // else the controller captured as it left the battlefield. All three are centralized in
    // resolve_defined_player; each resolves to one player we damage.
    if (ab.defined_you || ab.defined_each_opponent || ab.defined_targeted_controller) {
        Zone::Ownership who = resolve_defined_player(ab);
        if (who != Zone::UNKNOWN) {
            Entity pe = get_player_entity(who);
            deal_damage_to_player(ab.source, pe, dmg);
            auto &player = global_coordinator.GetComponent<Player>(pe);
            game_log("Dealt %zu damage to player (now at %d life)\n", dmg, player.life_total);
        }
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
