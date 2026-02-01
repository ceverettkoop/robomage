#include "stack_manager.h"

#include <cstddef>

#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/damage.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"

extern Game cur_game;

void StackManager::init() {
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>());
    signature.set(global_coordinator.GetComponentType<Ability>());
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

void StackManager::resolve_top() {
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

    auto& zone = global_coordinator.GetComponent<Zone>(top_entity);

    // Check if it's a spell card (not just an ability)
    if (global_coordinator.entity_has_component<CardData>(top_entity)) {
        auto& card_data = global_coordinator.GetComponent<CardData>(top_entity);

        // Check if it's a permanent type (Creature, Artifact, Enchantment, Planeswalker)
        bool is_permanent = false;
        bool is_creature = false;
        for (auto& type : card_data.types) {
            if (type.kind == TYPE) {
                if (type.name == "Creature" || type.name == "Artifact" ||
                    type.name == "Enchantment" || type.name == "Planeswalker") {
                    is_permanent = true;
                }
                if (type.name == "Creature") {
                    is_creature = true;
                }
            }
        }

        if (is_permanent) {
            // Move to battlefield
            zone.location = Zone::BATTLEFIELD;

            // Add Permanent component
            Permanent permanent;
            permanent.controller = zone.owner;  // Controller is the owner for now
            permanent.has_summoning_sickness = is_creature;  // Only creatures have summoning sickness
            permanent.is_tapped = false;
            permanent.timestamp_entered_battlefield = cur_game.timestamp++;
            global_coordinator.AddComponent(top_entity, permanent);

            // Add Damage component for creatures
            if (is_creature) {
                Damage damage;
                damage.damage_counters = 0;
                global_coordinator.AddComponent(top_entity, damage);
            }

            printf("%s enters the battlefield\n", card_data.name.c_str());
        } else {
            // Instant/Sorcery - resolve abilities then go to graveyard
            for (auto ability_entity : card_data.abilities) {
                auto& ability = global_coordinator.GetComponent<Ability>(ability_entity);
                if (ability.ability_type == Ability::SPELL) {
                    ability.resolve();
                }
            }
            zone.location = Zone::GRAVEYARD;
        }
    } else if (global_coordinator.entity_has_component<Ability>(top_entity)) {
        // It's a standalone ability (not attached to a card on the stack)
        auto &ability = global_coordinator.GetComponent<Ability>(top_entity);
        ability.resolve();
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
