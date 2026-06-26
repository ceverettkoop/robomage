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
#include "../cli_output.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
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
        for (auto &type : card_data.types) {
            if (type.kind == TYPE) {
                if (type.name == "Creature" || type.name == "Artifact" || type.name == "Enchantment" ||
                    type.name == "Planeswalker") {
                    is_permanent = true;
                }
            }
        }
        if (is_permanent) {
            // Move to battlefield; Permanent component added by apply_permanent_components on next SBA pass
            // Capture evoke status before the Spell component (which carries it) is removed;
            // apply_permanent_components consumes pending_evoked to set Permanent::evoked.
            if (global_coordinator.entity_has_component<Spell>(top_entity) &&
                global_coordinator.GetComponent<Spell>(top_entity).cast_with_evoke)
                cur_game.pending_evoked.insert(top_entity);
            if (global_coordinator.entity_has_component<Spell>(top_entity) &&
                global_coordinator.GetComponent<Spell>(top_entity).cast_with_offspring)
                cur_game.pending_offspring.insert(top_entity);
            // Carry the X paid for an X-cost permanent into the ETB so an "enters with X
            // counters" replacement can read it (Chalice of the Void).
            if (global_coordinator.entity_has_component<Spell>(top_entity) &&
                global_coordinator.GetComponent<Spell>(top_entity).x_paid > 0)
                cur_game.pending_etb_xpaid[top_entity] =
                    global_coordinator.GetComponent<Spell>(top_entity).x_paid;
            if (global_coordinator.entity_has_component<Spell>(top_entity))
                global_coordinator.RemoveComponent<Spell>(top_entity);
            if (global_coordinator.entity_has_component<Ability>(top_entity))
                global_coordinator.RemoveComponent<Ability>(top_entity);
            // This permanent is entering the battlefield because it was cast (CR 614.12):
            // mark it so an ETB replacement that cares about "wasn't cast" (Containment Priest)
            // lets it through. Consumed when its Permanent component is created.
            cur_game.cast_to_battlefield.insert(top_entity);
            orderer->add_to_zone(false, top_entity, Zone::BATTLEFIELD);
            auto &top_zone = global_coordinator.GetComponent<Zone>(top_entity);
            top_zone.controller = top_zone.owner;
            // TODO ETB event here
            game_log("%s enters the battlefield\n", card_data.name.c_str());
        } else {
            // Instant/Sorcery - resolve the Ability component added at cast time, then go to graveyard
            bool was_flashback = spell_cast_with_flashback(top_entity);
            if (global_coordinator.entity_has_component<Ability>(top_entity)) {
                auto &ab = global_coordinator.GetComponent<Ability>(top_entity);
                bool prev_priority = cur_game.player_a_has_priority;
                cur_game.player_a_has_priority = (ab.controller == Zone::PLAYER_A);
                ab.resolve(orderer);
                cur_game.player_a_has_priority = prev_priority;
                global_coordinator.RemoveComponent<Ability>(top_entity);
            }
            global_coordinator.RemoveComponent<Spell>(top_entity);
            // Shuffle into library instead of graveyard (e.g. Green Sun's Zenith)
            if (card_data.shuffle_into_library) {
                orderer->add_to_zone(false, top_entity, Zone::LIBRARY);
                orderer->shuffle_library(global_coordinator.GetComponent<Zone>(top_entity).owner);
                game_log("%s is shuffled into its owner's library\n", card_data.name.c_str());
            } else if (was_flashback) {
                orderer->add_to_zone(false, top_entity, Zone::EXILE);
                game_log("%s is exiled (flashback)\n", card_data.name.c_str());
            } else {
                orderer->add_to_zone(false, top_entity, Zone::GRAVEYARD);
            }
        }
    }
    // CASE FOR ABILITY ON STACK; not spell
    else if (global_coordinator.entity_has_component<Ability>(top_entity)) {
        auto &ability = global_coordinator.GetComponent<Ability>(top_entity);
        // Track resolution count for Count$ResolvedThisTurn (Scythecat Cub)
        if (ability.ability_type == Ability::TRIGGERED)
            cur_game.ability_resolution_counts[ability.source]++;
        bool prev_priority = cur_game.player_a_has_priority;
        cur_game.player_a_has_priority = (ability.controller == Zone::PLAYER_A);
        ability.resolve(orderer);
        cur_game.player_a_has_priority = prev_priority;

        // Destroy the standalone ability entity — it has no card zone to return to
        global_coordinator.DestroyEntity(top_entity);
    }
}
