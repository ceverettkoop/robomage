#include "action_processor.h"

#include <cstdio>

#include "components/ability.h"
#include "components/carddata.h"
#include "components/creature.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/zone.h"
#include "debug.h"
#include "ecs/coordinator.h"
#include "input_logger.h"
#include "mana_system.h"
#include "systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

void process_action(const LegalAction& action, Game& game, std::shared_ptr<Orderer> orderer) {

    switch (action.type) {
        case PASS_PRIORITY:
            game.pass_priority();
            break;

        case SPECIAL_ACTION: {
            // Play land
            Entity land_entity = action.source_entity;
            auto& zone = global_coordinator.GetComponent<Zone>(land_entity);
            auto& card_data = global_coordinator.GetComponent<CardData>(land_entity);

            // Move to battlefield
            zone.location = Zone::BATTLEFIELD;
            zone.controller = zone.owner;

            // Add Permanent component
            Permanent permanent;
            permanent.controller = zone.owner;
            permanent.has_summoning_sickness = false;  // Lands don't have summoning sickness
            permanent.is_tapped = false;
            permanent.timestamp_entered_battlefield = game.timestamp++;
            global_coordinator.AddComponent(land_entity, permanent);

            // Update player's lands played counter
            Entity player_entity = get_player_entity(zone.owner);
            auto& player = global_coordinator.GetComponent<Player>(player_entity);
            player.lands_played_this_turn++;

            printf("%s played %s\n", player_name(zone.owner).c_str(), card_data.name.c_str());

            // Playing a land uses take_action() (resets pass tracking)
            game.take_action();
            break;
        }

        case ACTIVATE_ABILITY: {
            Entity permanent_entity = action.source_entity;
            Entity ability_entity = action.target_entity;

            auto& permanent = global_coordinator.GetComponent<Permanent>(permanent_entity);
            auto& ability = global_coordinator.GetComponent<Ability>(ability_entity);
            auto& card_data = global_coordinator.GetComponent<CardData>(permanent_entity);

            permanent.is_tapped = true;

            Zone::Ownership controller = permanent.controller;
            Colors mana_color = static_cast<Colors>(ability.amount);
            add_mana(controller, mana_color);

            printf("%s tapped %s for mana\n", player_name(controller).c_str(), card_data.name.c_str());

            //PRIORITY DOES NOT PASS
            //TAKE ACTION IS NOT CALLED
            break;
        }

        case CAST_SPELL: {
            Entity spell_entity = action.source_entity;
            auto& zone = global_coordinator.GetComponent<Zone>(spell_entity);
            auto& card_data = global_coordinator.GetComponent<CardData>(spell_entity);

            Zone::Ownership caster = zone.owner;

            // Pay mana cost
            spend_mana(caster, card_data.mana_cost);

            // Check if spell requires targets
            bool requires_target = false;
            for (auto ability_entity : card_data.abilities) {
                auto& ability = global_coordinator.GetComponent<Ability>(ability_entity);
                if (ability.ability_type == Ability::SPELL) {
                    // Check category for targeting requirements
                    if (ability.category == "DealDamage") {
                        requires_target = true;

                        // Prompt for target
                        printf("Choose target:\n");
                        std::vector<Entity> valid_targets;

                        // Players are always valid for "Any" targeting
                        valid_targets.push_back(cur_game.player_a_entity);
                        valid_targets.push_back(cur_game.player_b_entity);

                        // Creatures on battlefield are valid
                        for (auto entity : orderer->mEntities) {
                            if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
                            auto& target_zone = global_coordinator.GetComponent<Zone>(entity);
                            if (target_zone.location == Zone::BATTLEFIELD) {
                                // Check if it's a creature
                                if (global_coordinator.entity_has_component<CardData>(entity)) {
                                    auto& target_card = global_coordinator.GetComponent<CardData>(entity);
                                    bool is_creature = false;
                                    for (auto& type : target_card.types) {
                                        if (type.kind == TYPE && type.name == "Creature") {
                                            is_creature = true;
                                            break;
                                        }
                                    }
                                    if (is_creature) {
                                        valid_targets.push_back(entity);
                                    }
                                }
                            }
                        }

                        // Display targets
                        for (size_t i = 0; i < valid_targets.size(); i++) {
                            Entity target = valid_targets[i];
                            if (global_coordinator.entity_has_component<Player>(target)) {
                                auto& player = global_coordinator.GetComponent<Player>(target);
                                std::string name = (target == cur_game.player_a_entity) ? "Player A" : "Player B";
                                printf("  %zu: %s (%d life)\n", i, name.c_str(), player.life_total);
                            } else {
                                auto& target_card = global_coordinator.GetComponent<CardData>(target);
                                printf("  %zu: %s\n", i, target_card.name.c_str());
                            }
                        }

                        int target_choice = InputLogger::instance().get_logged_input(cur_game.turn);
                        if (target_choice >= 0 && target_choice < static_cast<int>(valid_targets.size())) {
                            ability.target = valid_targets[static_cast<size_t>(target_choice)];
                            printf("Targeting choice %d\n", target_choice);
                        } else {
                            printf("Invalid target, spell fizzles\n");
                            // Move card to graveyard instead
                            zone.location = Zone::GRAVEYARD;
                            game.take_action();
                            return;
                        }
                    }
                }
            }
            printf("%s casts %s\n", player_name(caster).c_str(), card_data.name.c_str());

            // Move to stack
            zone.location = Zone::STACK;
            orderer->add_to_zone(false, spell_entity, Zone::STACK);  // Top of stack

            game.take_action();
            break;
        }
    }
}

static void declare_attackers(Game& game, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership active_player = game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;
    Entity defending_entity = game.player_a_turn ? game.player_b_entity : game.player_a_entity;

    // Collect eligible attackers with stable indices
    std::vector<Entity> eligible;
    for (auto entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
        auto& permanent = global_coordinator.GetComponent<Permanent>(entity);
        if (permanent.controller != active_player) continue;
        if (permanent.is_tapped) continue;
        if (permanent.has_summoning_sickness) continue;
        eligible.push_back(entity);
    }

    if (eligible.empty()) {
        printf("No creatures eligible to attack.\n");
        game.attackers_declared = true;
        game.pending_choice = NONE;
        return;
    }

    // Build targets: defending player first, then planeswalkers (TODO)
    std::vector<Entity> targets;
    targets.push_back(defending_entity);

    // Selection loop — creature indices stay stable throughout
    while (true) {
        printf("\n--- Declare Attackers (%s) ---\n", player_name(active_player).c_str());
        for (size_t i = 0; i < eligible.size(); i++) {
            auto& cd = global_coordinator.GetComponent<CardData>(eligible[i]);
            auto& cr = global_coordinator.GetComponent<Creature>(eligible[i]);
            printf("  %zu: %s [%d/%d]", i, cd.name.c_str(), cr.power, cr.toughness);
            if (cr.is_attacking) {
                Zone::Ownership t = (cr.attack_target == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
                printf(" -> %s", player_name(t).c_str());
            }
            printf("\n");
        }
        printf("  -1: Confirm\n");

        int creature_choice = InputLogger::instance().get_logged_input(cur_game.turn);

        if (creature_choice == -1) break;

        if (creature_choice < 0 || creature_choice >= static_cast<int>(eligible.size())) {
            printf("Invalid selection.\n");
            continue;
        }

        auto& cr = global_coordinator.GetComponent<Creature>(eligible[static_cast<size_t>(creature_choice)]);
        auto& cd = global_coordinator.GetComponent<CardData>(eligible[static_cast<size_t>(creature_choice)]);

        if (cr.is_attacking) {
            cr.is_attacking = false;
            cr.attack_target = 0;
            printf("%s removed from attackers.\n", cd.name.c_str());
        } else {
            printf("Select target for %s:\n", cd.name.c_str());
            for (size_t i = 0; i < targets.size(); i++) {
                auto& player = global_coordinator.GetComponent<Player>(targets[i]);
                Zone::Ownership t = (targets[i] == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
                printf("  %zu: %s (%d life)\n", i, player_name(t).c_str(), player.life_total);
                // TODO: planeswalker entries here
            }

            int target_choice = InputLogger::instance().get_logged_input(cur_game.turn);

            if (target_choice >= 0 && target_choice < static_cast<int>(targets.size())) {
                cr.is_attacking = true;
                cr.attack_target = targets[static_cast<size_t>(target_choice)];
                Zone::Ownership t = (cr.attack_target == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
                printf("%s attacking %s.\n", cd.name.c_str(), player_name(t).c_str());
            } else {
                printf("Invalid target.\n");
            }
        }
    }

    printf("\nAttackers declared:\n");
    bool any = false;
    for (auto entity : eligible) {
        auto& cr = global_coordinator.GetComponent<Creature>(entity);
        if (cr.is_attacking) {
            any = true;
            auto& cd = global_coordinator.GetComponent<CardData>(entity);
            Zone::Ownership t = (cr.attack_target == game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
            printf("  %s -> %s\n", cd.name.c_str(), player_name(t).c_str());
        }
    }
    if (!any) printf("  (none)\n");

    game.attackers_declared = true;
    game.pending_choice = NONE;
}

void proc_mandatory_choice(Game& game, std::shared_ptr<Orderer> orderer) {
    switch (game.pending_choice) {
        case DECLARE_ATTACKERS_CHOICE:
            declare_attackers(game, orderer);
            break;
        case DECLARE_BLOCKERS_CHOICE:
            printf("TODO: Declare blockers\n");
            game.blockers_declared = true;
            game.pending_choice = NONE;
            break;
        case CLEANUP_DISCARD:
            printf("TODO: Cleanup discard\n");
            game.pending_choice = NONE;
            break;
        case CHOOSE_ENTITY:
            printf("TODO: Choose entity\n");
            game.pending_choice = NONE;
            break;
        case NONE:
            break;
    }
}
