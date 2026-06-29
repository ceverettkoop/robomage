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
#include "../type_constants.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../cli_output.h"
#include "../game_queries.h"
#include "../input_logger.h"
#include "../mana_system.h"
#include "../svar_eval.h"
#include "../systems/stack_manager.h"
#include "orderer.h"

// Should this creature deal damage during this combat damage step?
bool should_deal_damage(const Creature &cr, bool first_strike_only) {
    bool has_fs = creature_has_keyword(cr, "First Strike");
    bool has_ds = creature_has_keyword(cr, "Double Strike");
    if (first_strike_only) return creature_deals_first_strike_damage(cr);
    // Regular damage step: skip first-strikers (they already dealt), but double strikers hit again
    if (has_fs && !has_ds) return false;
    return true;
}

// Accumulate combat damage and lifelink gain in a single pass so that rule 119.3
// (simultaneous damage/life-gain) applies: a controller only dies if the NET
// life change leaves them at 0 or less after both effects resolve together.
static void apply_lifelink_if_any(Entity source, size_t amount,
                                  int &life_delta_a, int &life_delta_b) {
    if (amount == 0) return;
    if (!global_coordinator.entity_has_component<Creature>(source)) return;
    auto &cr = global_coordinator.GetComponent<Creature>(source);
    bool has_lifelink = false;
    for (const auto &k : cr.keywords)
        if (k == "Lifelink") { has_lifelink = true; break; }
    if (!has_lifelink) return;
    if (!global_coordinator.entity_has_component<Permanent>(source)) return;
    Zone::Ownership ctrl = global_coordinator.GetComponent<Permanent>(source).controller;
    if (ctrl == Zone::PLAYER_A)      life_delta_a += static_cast<int>(amount);
    else if (ctrl == Zone::PLAYER_B) life_delta_b += static_cast<int>(amount);
}

// Combat damage is a turn-based action (rule 510.2), not a state-based action
void StateManager::deal_combat_damage(Game &game, bool first_strike_only) {
    game_log("\n--- %sCombat Damage ---\n", first_strike_only ? "First Strike " : "");

    // Accumulate lifelink life gains; apply at end so they are simultaneous with damage.
    int life_delta_a = 0;
    int life_delta_b = 0;

    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
        auto &cr = global_coordinator.GetComponent<Creature>(entity);
        if (!cr.is_attacking) continue;
        if (!should_deal_damage(cr, first_strike_only)) continue;

        std::string attacker_name = entity_name(entity);

        // Collect blockers for this attacker
        std::vector<Entity> blockers;
        for (auto b : mEntities) {
            if (!global_coordinator.entity_has_component<Creature>(b)) continue;
            auto &bcr = global_coordinator.GetComponent<Creature>(b);
            if (bcr.is_blocking && bcr.blocking_target == entity) {
                blockers.push_back(b);
            }
        }

        if (!cr.is_blocked) {
            // Unblocked — deal damage to attack target
            uint32_t dmg = cr.power;
            if (dmg > 0) {
                deal_damage(entity, cr.attack_target, dmg);
                // CR 702.16d: a player with protection from everything (The One Ring) isn't dealt
                // combat damage by an opponent's attacker — prevent the life loss, the
                // damage-to-player trigger, and the attacker's lifelink (prevented damage isn't
                // dealt, so no life is gained).
                bool prevented = global_coordinator.entity_has_component<Player>(cr.attack_target) &&
                                 player_protected_from_source(cr.attack_target, entity);
                if (prevented) {
                    game_log("  %s has protection from everything — %u combat damage prevented\n",
                             target_display_name(game, cr.attack_target).c_str(), dmg);
                } else if (global_coordinator.entity_has_component<Player>(cr.attack_target)) {
                    auto &target_player = global_coordinator.GetComponent<Player>(cr.attack_target);
                    target_player.life_total -= static_cast<int>(dmg);
                    std::string tname = target_display_name(game, cr.attack_target);
                    game_log("  %s deals %u damage to %s\n", attacker_name.c_str(), dmg, tname.c_str());

                    Event ev(Events::COMBAT_DAMAGE_TO_PLAYER);
                    ev.SetParam(Params::ENTITY, entity);
                    ev.SetParam(Params::PLAYER, cr.attack_target);
                    ev.SetParam(Params::AMOUNT, dmg);
                    global_coordinator.SendEvent(ev);
                } else if (is_planeswalker_permanent(cr.attack_target) && on_battlefield(cr.attack_target)) {
                    // Combat damage to a planeswalker removes that many loyalty counters (306.8).
                    damage_planeswalker(cr.attack_target, dmg);
                    game_log("  %s deals %u damage to %s (loyalty now %d)\n", attacker_name.c_str(), dmg,
                             entity_name(cr.attack_target).c_str(),
                             get_counters(cr.attack_target, "LOYALTY"));
                }
                if (!prevented) apply_lifelink_if_any(entity, dmg, life_delta_a, life_delta_b);
            }
        } else {
            // Blocked — assign damage to blockers, blockers deal damage back.
            // T3.10: if the controller was prompted to divide damage (it couldn't kill every
            // blocker), apply their stored per-blocker assignment; otherwise auto-assign lethal
            // in order. Either way lethal accounts for damage already marked on the blocker
            // (the T3.11 fix) and treats deathtouch as lethal-1 (702.2c) — see lethal_needed_for_blocker.
            auto assign_it = game.combat_damage_assignment.find(entity);
            bool have_assignment = (assign_it != game.combat_damage_assignment.end());
            bool has_trample = creature_has_keyword(cr, "Trample");
            uint32_t remaining = cr.power;
            for (auto blocker : blockers) {
                auto &bcr = global_coordinator.GetComponent<Creature>(blocker);
                std::string blocker_name = entity_name(blocker);

                // Blocker deals damage to attacker (only if the blocker qualifies for this step)
                if (bcr.power > 0 && should_deal_damage(bcr, first_strike_only)) {
                    deal_damage(blocker, entity, bcr.power);
                    game_log("  %s deals %u damage to %s\n", blocker_name.c_str(), bcr.power, attacker_name.c_str());
                    apply_lifelink_if_any(blocker, bcr.power, life_delta_a, life_delta_b);
                }

                // Attacker deals damage to this blocker.
                uint32_t assigned = 0;
                if (have_assignment) {
                    auto bit = assign_it->second.find(blocker);
                    assigned = (bit != assign_it->second.end()) ? bit->second : 0u;
                    if (assigned > remaining) assigned = remaining;
                } else if (remaining > 0) {
                    // CR 510.1c: a blocked creature must assign ALL its combat damage. Auto-assign
                    // lethal to each blocker in order; the LAST blocker absorbs any leftover
                    // (harmless overkill) unless the attacker has trample, in which case the excess
                    // tramples over to the attack target below. So a single blocker receives the
                    // attacker's full power (Solitude [3/2] blocked by a [1/1] deals 3, lifelinking
                    // 3) — not just the blocker's lethal amount.
                    uint32_t needed = lethal_needed_for_blocker(entity, blocker);
                    bool is_last_blocker = (blocker == blockers.back());
                    if (is_last_blocker && !has_trample)
                        assigned = remaining;
                    else
                        assigned = (remaining >= needed) ? needed : remaining;
                }
                if (assigned > 0) {
                    deal_damage(entity, blocker, assigned);
                    game_log("  %s deals %u damage to %s\n", attacker_name.c_str(), assigned, blocker_name.c_str());
                    remaining -= assigned;
                    apply_lifelink_if_any(entity, assigned, life_delta_a, life_delta_b);
                }
            }
            // Trample: excess damage goes to attack target
            if (remaining > 0) {
                if (has_trample && global_coordinator.entity_has_component<Player>(cr.attack_target) &&
                    player_protected_from_source(cr.attack_target, entity)) {
                    // Protection from everything (The One Ring) also prevents trampled-over combat
                    // damage to the protected player (CR 702.16d) — no life loss, trigger, or
                    // lifelink.
                    game_log("  %s has protection from everything — %u trample damage prevented\n",
                             target_display_name(game, cr.attack_target).c_str(), remaining);
                } else if (has_trample && global_coordinator.entity_has_component<Player>(cr.attack_target)) {
                    deal_damage(entity, cr.attack_target, remaining);
                    auto &target_player = global_coordinator.GetComponent<Player>(cr.attack_target);
                    target_player.life_total -= static_cast<int>(remaining);
                    std::string tname = target_display_name(game, cr.attack_target);
                    game_log("  %s tramples %u damage to %s\n", attacker_name.c_str(), remaining, tname.c_str());

                    Event ev(Events::COMBAT_DAMAGE_TO_PLAYER);
                    ev.SetParam(Params::ENTITY, entity);
                    ev.SetParam(Params::PLAYER, cr.attack_target);
                    ev.SetParam(Params::AMOUNT, remaining);
                    global_coordinator.SendEvent(ev);
                    apply_lifelink_if_any(entity, remaining, life_delta_a, life_delta_b);
                } else if (has_trample && is_planeswalker_permanent(cr.attack_target) &&
                           on_battlefield(cr.attack_target)) {
                    // Trample excess over the blockers goes to the attacked planeswalker (306.8).
                    damage_planeswalker(cr.attack_target, remaining);
                    game_log("  %s tramples %u damage to %s (loyalty now %d)\n", attacker_name.c_str(),
                             remaining, entity_name(cr.attack_target).c_str(),
                             get_counters(cr.attack_target, "LOYALTY"));
                    apply_lifelink_if_any(entity, remaining, life_delta_a, life_delta_b);
                }
            }
        }
    }

    // Apply lifelink life gains simultaneously with damage taken (rule 119.3).
    if (life_delta_a != 0) {
        player_gain_life(game.player_a_entity, life_delta_a);
        game_log("  Player A gains %d life (lifelink)\n", life_delta_a);
    }
    if (life_delta_b != 0) {
        player_gain_life(game.player_b_entity, life_delta_b);
        game_log("  Player B gains %d life (lifelink)\n", life_delta_b);
    }

    game.combat_damage_dealt = true;
    game_log("--- End %sCombat Damage ---\n\n", first_strike_only ? "First Strike " : "");
}

