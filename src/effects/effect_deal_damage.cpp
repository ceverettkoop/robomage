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
#include "../mana_system.h"
#include "../queries/players.h"
#include "../queries/zones.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

static void deal_damage_to_target(Ability &ab, Entity tgt, size_t dmg);

HandlerResult deal_damage(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    // Delirium-conditional damage (Unholy Heat)
    size_t dmg = ab.def->amount;
    // Dynamic damage (e.g. Ajani's "damage equal to the number of creatures you control",
    // NumDmg$ X with X = Count$Valid Creature.YouCtrl). Mirrors lose_life/gain_life, which
    // evaluate their dynamic amount at resolution.
    if (!ab.def->dynamic_amount_expr.empty()) {
        // Thread the source through so a source-relative count (Summon: Bahamut's Mega Flare,
        // X = Count$Valid Permanent.YouCtrl+Other$CardManaCost — total MV of OTHER permanents you
        // control) can exclude the source itself via the +Other qualifier.
        dmg = evaluate_dynamic_amount(ab.def->dynamic_amount_expr, ab.controller, orderer, ab.target.get(),
                                      ab.source.lki_entity());
    }
    const DamageParams *dp = std::get_if<DamageParams>(&ab.def->params);
    if (dp && dp->is_delirium_scale) {
        if (check_delirium(ab.controller, orderer->mEntities)) dmg = dp->delirium_amount;
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
    if (ab.def->defined_you || ab.def->defined_each_opponent || ab.def->defined_targeted_controller ||
        ab.def->defined_triggered_activator || ab.def->defined_triggered_player ||
        ab.def->defined_triggered_card_controller) {
        Zone::Ownership who = resolve_defined_player(ab);
        if (who != Zone::UNKNOWN) ::deal_damage(ab.source.lki_entity(), get_player_entity(who), dmg, false);
        return HandlerResult::DONE_RUN_SUBS;
    }
    // Multi-target DealDamage (Prismari Charm: "deals 1 damage to each of one or two targets"):
    // action_processor stores every chosen target in ab.targets (target_max > 1). Deal `dmg` to
    // EACH of them, reusing the single per-target dispatch. A single-target ability leaves
    // ab.targets empty and uses ab.target. Each target is checked independently at resolution, so
    // one target having become illegal (gone from the battlefield, protection) skips only that
    // target without aborting the rest (CR 608.2c).
    if (!ab.targets.empty()) {
        for (Entity tgt : live_entities(ab.targets)) deal_damage_to_target(ab, tgt, dmg);
    } else {
        deal_damage_to_target(ab, ab.target.get(), dmg);
    }
    return HandlerResult::DONE_RUN_SUBS;
}

// Deal `dmg` to a single already-chosen target — a player, a planeswalker, or a creature — the
// shared per-target dispatch used by both the single- and multi-target DealDamage paths. The
// target was re-checked at resolution (CR 608.2b), so one that can't be dealt damage should have
// fizzled.
static void deal_damage_to_target(Ability &ab, Entity tgt, size_t dmg) {
    if (tgt == 0) return;  // the object is gone (CR 400.7): nothing is dealt damage
    if (can_be_dealt_damage(tgt)) {
        ::deal_damage(ab.source.lki_entity(), tgt, dmg, false);
        return;
    }
#ifndef NDEBUG
    fprintf(stderr, "SOURCE:");
    dump_entity(ab.source.lki_entity());
    fprintf(stderr, "TARGET:");
    dump_entity(tgt);
#endif
    non_fatal_error("Damage should have fizzled prior to this");
}

bool parse_deal_damage(AbilityDef &ab, const std::string &key, const std::string &value) {
    if (key == "ValidPlayers" && ab.category == "DamageAll") {
        // DamageAll's players (Pyroclasm-style "each creature and each player").
        effect_params<DamageParams>(ab).valid_players = value;
        return true;
    }
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
