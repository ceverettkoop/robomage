#include "stack_manager.h"

#include <cstddef>
#include <string>
#include <vector>

#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/damage.h"
#include "../components/spell.h"
#include "../components/zone.h"
#include "../debug.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "orderer.h"

extern Game cur_game;

void StackManager::init() {
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>());
    global_coordinator.SetSystemSignature<StackManager>(signature);
}

bool StackManager::is_empty() {
    for (auto &&entity : mEntities) {
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location == Zone::STACK) {
            return false;
        }
    }
    return true;
}

void StackManager::resolve_top(std::shared_ptr<Orderer> orderer) {
    Entity top_entity = 0;
    size_t min_distance = SIZE_MAX;
    bool found = false;

    // Find the entity on top of the stack (closest to top, distance_from_top == 0)
    for (auto &&entity : mEntities) {
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location == Zone::STACK && zone.distance_from_top < min_distance) {
            top_entity = entity;
            min_distance = zone.distance_from_top;
            found = true;
        }
    }

    if (!found) return;

    // Check if it's a spell card (not just an ability)
    if (global_coordinator.entity_has_component<CardData>(top_entity)) {
        auto &card_data = global_coordinator.GetComponent<CardData>(top_entity);

        // Check if it's a permanent type (Creature, Artifact, Enchantment, Planeswalker)
        bool is_permanent = false;
        bool is_creature = false;
        for (auto &type : card_data.types) {
            if (type.kind == TYPE) {
                if (type.name == "Creature" || type.name == "Artifact" || type.name == "Enchantment" ||
                    type.name == "Planeswalker") {
                    is_permanent = true;
                }
                if (type.name == "Creature") {
                    is_creature = true;
                }
            }
        }

        if (is_permanent) {
            // Move to battlefield; Permanent component added by apply_permanent_components on next SBA pass
            orderer->add_to_zone(false, top_entity, Zone::BATTLEFIELD);
            auto &top_zone = global_coordinator.GetComponent<Zone>(top_entity);
            top_zone.controller = top_zone.owner;
            //damage component added by state based effects
            //TODO event here
            printf("%s enters the battlefield\n", card_data.name.c_str());
        } else {
            // Instant/Sorcery - resolve the Ability component added at cast time, then go to graveyard
            if (global_coordinator.entity_has_component<Ability>(top_entity)) {
                global_coordinator.GetComponent<Ability>(top_entity).resolve();
                global_coordinator.RemoveComponent<Ability>(top_entity);
            }
            global_coordinator.RemoveComponent<Spell>(top_entity);
            orderer->add_to_zone(false, top_entity, Zone::GRAVEYARD);
        }
    } else if (global_coordinator.entity_has_component<Ability>(top_entity)) {
        // It's a standalone ability (not attached to a card on the stack)
        auto &stack_zone = global_coordinator.GetComponent<Zone>(top_entity);
        auto &ability = global_coordinator.GetComponent<Ability>(top_entity);
        // TODO CHECK THIS NONSENSE
        if (ability.category == "ChangeZone") {
            Zone::Ownership owner = stack_zone.owner;

            // Parse comma-separated ChangeType$ subtypes
            std::vector<std::string> subtypes;
            std::string ct = ability.change_type;
            size_t p = 0;
            while (true) {
                size_t comma = ct.find(',', p);
                if (comma == std::string::npos) {
                    subtypes.push_back(ct.substr(p));
                    break;
                }
                subtypes.push_back(ct.substr(p, comma - p));
                p = comma + 1;
            }

            // Collect matching lands from library
            std::vector<Entity> choices;
            for (auto lib_entity : orderer->get_library_contents(owner)) {
                auto &lcd = global_coordinator.GetComponent<CardData>(lib_entity);
                bool matches = false;
                for (auto &t : lcd.types) {
                    for (auto &st : subtypes) {
                        if (t.name == st) {
                            matches = true;
                            break;
                        }
                    }
                    if (matches) break;
                }
                if (matches) choices.push_back(lib_entity);
            }

            // choices[0] = "Fail to find"; choices[1..n] = matching lands
            printf("Search your library:\n");
            printf("  0: Fail to find\n");
            for (size_t i = 0; i < choices.size(); i++) {
                auto &lcd = global_coordinator.GetComponent<CardData>(choices[i]);
                printf("  %zu: %s\n", i + 1, lcd.name.c_str());
            }
            std::vector<ActionCategory> cats(choices.size() + 1, ActionCategory::OTHER_CHOICE);
            int choice = InputLogger::instance().get_logged_input(cur_game.turn, cats);
            if (choice >= 1 && choice <= static_cast<int>(choices.size())) {
                Entity chosen = choices[static_cast<size_t>(choice - 1)];
                auto &chosen_cd = global_coordinator.GetComponent<CardData>(chosen);
                auto &chosen_zone = global_coordinator.GetComponent<Zone>(chosen);
                orderer->add_to_zone(false, chosen, Zone::BATTLEFIELD);
                chosen_zone.controller = owner;
                printf("%s searches and puts %s onto the battlefield\n", player_name(owner).c_str(),
                    chosen_cd.name.c_str());
            } else {
                printf("%s fails to find\n", player_name(owner).c_str());
            }

            orderer->shuffle_library(owner);
            printf("%s shuffles their library\n", player_name(owner).c_str());
        } else {
            ability.resolve();
        }

        // Destroy the standalone ability entity — it has no card zone to return to
        global_coordinator.DestroyEntity(top_entity);
    }
}

std::vector<Entity> StackManager::get_stack_contents() {
    std::vector<Entity> stack_entities;

    for (auto &&entity : mEntities) {
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location == Zone::STACK) {
            stack_entities.push_back(entity);
        }
    }

    return stack_entities;
}
