#include "action_processor.h"

#include <cstdio>

#include "components/ability.h"
#include "components/carddata.h"
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
            // Tap for mana (mana ability doesn't use stack)
            Entity permanent_entity = action.source_entity;
            Entity ability_entity = action.target_entity;

            auto& permanent = global_coordinator.GetComponent<Permanent>(permanent_entity);
            auto& ability = global_coordinator.GetComponent<Ability>(ability_entity);
            auto& card_data = global_coordinator.GetComponent<CardData>(permanent_entity);

            // Tap the permanent
            permanent.is_tapped = true;

            // Add mana to player's pool
            Zone::Ownership controller = permanent.controller;
            Colors mana_color = static_cast<Colors>(ability.amount);
            add_mana(controller, mana_color);

            printf("%s tapped %s for mana\n", player_name(controller).c_str(), card_data.name.c_str());

            // Mana abilities use take_action()
            game.take_action();
            break;
        }

        case CAST_SPELL: {
            Entity spell_entity = action.source_entity;
            auto& zone = global_coordinator.GetComponent<Zone>(spell_entity);
            auto& card_data = global_coordinator.GetComponent<CardData>(spell_entity);

            Zone::Ownership caster = zone.owner;

            // Pay mana cost
            spend_mana(caster, card_data.mana_cost);

            printf("%s casts %s\n", player_name(caster).c_str(), card_data.name.c_str());

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

                        int target_choice = InputLogger::instance().get_logged_input();
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

            // Move to stack
            zone.location = Zone::STACK;
            orderer->add_to_zone(false, spell_entity, Zone::STACK);  // Top of stack

            game.take_action();
            break;
        }
    }
}
