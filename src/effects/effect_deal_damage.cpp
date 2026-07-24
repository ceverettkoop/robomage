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

static void deal_damage_to_target(Ability &ab, Entity tgt, size_t dmg);

HandlerResult deal_damage(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    // Delirium-conditional damage (Unholy Heat)
    size_t dmg = ab.amount;
    // Dynamic damage (e.g. Ajani's "damage equal to the number of creatures you control",
    // NumDmg$ X with X = Count$Valid Creature.YouCtrl). Mirrors lose_life/gain_life, which
    // evaluate their dynamic amount at resolution.
    if (!ab.dynamic_amount_expr.empty()) {
        // Thread the source through so a source-relative count (Summon: Bahamut's Mega Flare,
        // X = Count$Valid Permanent.YouCtrl+Other$CardManaCost — total MV of OTHER permanents you
        // control) can exclude the source itself via the +Other qualifier.
        dmg = evaluate_dynamic_amount(ab.dynamic_amount_expr, source_controller(ab.source), orderer,
                                      ab.target, ab.source);
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
    // TriggeredActivator (Eidolon of the Great Revel: "deals 2 damage to that player" — the
    // player who cast the spell) also routes through resolve_defined_player, which binds the
    // player captured at trigger-fire time. Without this it would fall to the targeted path
    // with an unset target and trip the "should have fizzled" guard.
    if (ab.defined_you || ab.defined_each_opponent || ab.defined_targeted_controller ||
        ab.defined_triggered_activator || ab.defined_triggered_player) {
        Zone::Ownership who = resolve_defined_player(ab);
        if (who != Zone::UNKNOWN) {
            Entity pe = get_player_entity(who);
            deal_damage_to_player(ab.source, pe, dmg);
            auto &player = global_coordinator.GetComponent<Player>(pe);
            game_log("Dealt %zu damage to player (now at %d life)\n", dmg, player.life_total);
        }
        return HandlerResult::DONE_RUN_SUBS;
    }
    // Multi-target DealDamage (Prismari Charm: "deals 1 damage to each of one or two targets"):
    // action_processor stores every chosen target in ab.targets (target_max > 1). Deal `dmg` to
    // EACH of them, reusing the single per-target dispatch. A single-target ability leaves
    // ab.targets empty and uses ab.target. Each target is checked independently at resolution, so
    // one target having become illegal (gone from the battlefield, protection) skips only that
    // target without aborting the rest (CR 608.2c).
    if (!ab.targets.empty()) {
        for (Entity tgt : ab.targets) deal_damage_to_target(ab, tgt, dmg);
    } else {
        deal_damage_to_target(ab, ab.target, dmg);
    }
    return HandlerResult::DONE_RUN_SUBS;
}

// Deal `dmg` to a single already-chosen target — a player, a planeswalker, or a creature — the
// shared per-target dispatch used by both the single- and multi-target DealDamage paths.
static void deal_damage_to_target(Ability &ab, Entity tgt, size_t dmg) {
    if (global_coordinator.entity_has_component<Player>(tgt)) {
        deal_damage_to_player(ab.source, tgt, dmg);
        auto &player = global_coordinator.GetComponent<Player>(tgt);
        game_log("Dealt %zu damage to player (now at %d life)\n", dmg, player.life_total);
    } else if (is_planeswalker_permanent(tgt)) {
        // Damage to a planeswalker removes that many loyalty counters (306.8).
        damage_planeswalker(tgt, dmg);
        auto &pw = global_coordinator.GetComponent<Permanent>(tgt);
        game_log("Dealt %zu damage to %s (loyalty now %d)\n", dmg, pw.name.c_str(), get_counters(tgt, "LOYALTY"));
    } else if (permanent_protected_from_colored_spell_source(tgt, ab.source)) {
        // Protection from colored spells, damage facet (Emrakul, CR 702.16d): a colored spell
        // source can't deal damage to this permanent. Prevented here (mirroring the player
        // protection-from-everything check above) so the targeted path skips the damage cleanly
        // instead of tripping the "should have fizzled" guard below.
        game_log("Permanent has protection from colored spells — %zu damage prevented\n", dmg);
    } else {
        if (::deal_damage(ab.source, tgt, dmg)) {
            game_log("Dealt %zu damage to creature\n", dmg);
        } else {
#ifndef NDEBUG
            fprintf(stderr, "SOURCE:");
            dump_entity(ab.source);
            fprintf(stderr, "TARGET:");
            dump_entity(tgt);
#endif
            non_fatal_error("Damage should have fizzled prior to this");
        }
    }
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
