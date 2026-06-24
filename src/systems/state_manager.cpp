#include "state_manager.h"
#include "state_manager_internal.h"

#include <algorithm>
#include <cstddef>
#include <map>
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

std::vector<ActiveStatic> g_active_statics;

void StateManager::init() {
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>());
    global_coordinator.SetSystemSignature<StateManager>(signature);
}

// Turn-based actions happen at the start of specific steps (rules 508, 509, 510, 514)
void StateManager::process_turn_based_actions(Game &game, std::shared_ptr<Orderer> orderer) {
    game.pending_choice = NONE;

    // First strike combat damage (rule 510.1)
    if (game.cur_step == FIRST_STRIKE_DAMAGE && !game.combat_damage_dealt) {
        deal_combat_damage(game, true);
    }
    // Regular combat damage (rule 510.2)
    if (game.cur_step == COMBAT_DAMAGE && !game.combat_damage_dealt) {
        deal_combat_damage(game, false);
    }

    // Declare attackers (rule 508.1)
    if (game.cur_step == DECLARE_ATTACKERS && !game.attackers_declared) {
        game.pending_choice = DECLARE_ATTACKERS_CHOICE;
        return;
    }
    // Declare blockers (rule 509.1)
    if (game.cur_step == DECLARE_BLOCKERS && !game.blockers_declared) {
        game.pending_choice = DECLARE_BLOCKERS_CHOICE;
        return;
    }
    // Cleanup discard (rule 514.1)
    if (game.cur_step == CLEANUP) {
        Zone::Ownership active_player = game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;
        size_t hand_size = 0;
        for (auto entity : mEntities) {
            if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
            auto &zone = global_coordinator.GetComponent<Zone>(entity);
            if (zone.location == Zone::HAND && zone.owner == active_player) hand_size++;
        }
        if (hand_size > 7) {
            game.pending_choice = CLEANUP_DISCARD;
            return;
        }
    }
}

// State-based actions are checked simultaneously and loop until stable (rule 704.3)
void StateManager::state_based_effects(Game &game, std::shared_ptr<Orderer> orderer) {
    for (;;) {
        // Continuous effects define the game state that SBAs evaluate
        apply_permanent_components(game);
        apply_static_ability_effects();

        bool any_applied = false;

        // 704.5a - player with 0 or less life loses
        auto &player_a = global_coordinator.GetComponent<Player>(game.player_a_entity);
        auto &player_b = global_coordinator.GetComponent<Player>(game.player_b_entity);
        if (player_a.life_total <= 0) {
            printf("\nPlayer A has %d life - Player B wins!\n", player_a.life_total);
            game.ended = true;
            game.winner = Zone::PLAYER_B;
            return;
        }
        if (player_b.life_total <= 0) {
            printf("\nPlayer B has %d life - Player A wins!\n", player_b.life_total);
            game.ended = true;
            game.winner = Zone::PLAYER_A;
            return;
        }

        // 704.5d - tokens in zones other than battlefield cease to exist
        // (handled by apply_permanent_components above)

        // 704.5f - creature with toughness 0 or less goes to graveyard
        // 704.5g - creature with lethal damage is destroyed
        std::vector<Entity> creatures_to_destroy;
        for (auto entity : mEntities) {
            if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
            if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
            auto &zone = global_coordinator.GetComponent<Zone>(entity);
            if (zone.location != Zone::BATTLEFIELD) continue;
            if (global_coordinator.entity_has_component<Permanent>(entity) &&
                global_coordinator.GetComponent<Permanent>(entity).is_phased_out) continue;

            auto &creature = global_coordinator.GetComponent<Creature>(entity);
            if (creature.toughness == 0) {
                creatures_to_destroy.push_back(entity);
            } else if (global_coordinator.entity_has_component<Damage>(entity)) {
                auto &damage = global_coordinator.GetComponent<Damage>(entity);
                // 702.2b: any nonzero damage from a deathtouch source is lethal.
                bool deathtouched = damage.has_deathtouch_damage && damage.damage_counters > 0;
                if (deathtouched || damage.damage_counters >= creature.toughness) {
                    creatures_to_destroy.push_back(entity);
                }
            }
        }

        for (auto entity : creatures_to_destroy) {
            std::string name = entity_name(entity);
            auto &creature = global_coordinator.GetComponent<Creature>(entity);
            if (creature.toughness == 0)
                game_log("%s dies (zero toughness)\n", name.c_str());
            else
                game_log("%s is destroyed (lethal damage)\n", name.c_str());
            orderer->add_to_zone(false, entity, Zone::GRAVEYARD);
            any_applied = true;
        }

        // 704.5i - a planeswalker with 0 (or less) loyalty is put into its owner's graveyard
        for (auto entity : mEntities) {
            if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
            if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
            if (global_coordinator.GetComponent<Zone>(entity).location != Zone::BATTLEFIELD) continue;
            auto &perm = global_coordinator.GetComponent<Permanent>(entity);
            if (perm.is_phased_out || !is_planeswalker(perm.types)) continue;
            if (perm.loyalty <= 0) {
                game_log("%s dies (0 loyalty)\n", entity_name(entity).c_str());
                orderer->add_to_zone(false, entity, Zone::GRAVEYARD);
                any_applied = true;
            }
        }

        // 704.5j - legend rule: a player who controls two or more legendary permanents with
        // the same name chooses one to keep; the rest go to their owners' graveyards. Affected
        // players choose in APNAP order (active player first); one conflict is resolved per pass,
        // then the SBA loop re-evaluates.
        {
            Zone::Ownership legend_order[2] = {
                game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B,
                game.player_a_turn ? Zone::PLAYER_B : Zone::PLAYER_A};
            bool legend_applied = false;
            for (Zone::Ownership owner : legend_order) {
                if (legend_applied) break;
                std::map<std::string, std::vector<Entity>> by_name;
                for (auto entity : mEntities) {
                    if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
                    if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
                    if (global_coordinator.GetComponent<Zone>(entity).location != Zone::BATTLEFIELD) continue;
                    auto &perm = global_coordinator.GetComponent<Permanent>(entity);
                    if (perm.controller != owner || perm.is_phased_out) continue;
                    if (!has_legendary_supertype(perm.types)) continue;
                    by_name[perm.name].push_back(entity);
                }
                for (auto &grp : by_name) {
                    if (grp.second.size() < 2) continue;
                    std::vector<LegalAction> choices;
                    for (auto e : grp.second) {
                        LegalAction la(PASS_PRIORITY, e, "Keep " + entity_name(e));
                        la.category = ActionCategory::OTHER_CHOICE;
                        choices.push_back(la);
                    }
                    game_log("Legend rule: %s controls %zu copies of %s; choose one to keep.\n",
                             player_name(owner).c_str(), grp.second.size(), grp.first.c_str());
                    int keep = InputLogger::instance().get_input(choices);
                    Entity kept = grp.second[static_cast<size_t>(keep)];
                    for (auto e : grp.second) {
                        if (e == kept) continue;
                        game_log("%s is put into the graveyard (legend rule)\n", entity_name(e).c_str());
                        orderer->add_to_zone(false, e, Zone::GRAVEYARD);
                    }
                    any_applied = true;
                    legend_applied = true;
                    break;
                }
            }
        }

        if (!any_applied) break;
    }

    // SBA loop settled; triggered abilities go on the stack (rule 704.3)
    check_triggered_abilities(game, orderer);
}

