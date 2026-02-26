#include "mana_system.h"

#include "classes/game.h"
#include "components/player.h"
#include "ecs/coordinator.h"

extern Coordinator global_coordinator;
extern Game cur_game;

Entity get_player_entity(Zone::Ownership player) {
    return (player == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
}

bool can_afford_pool(const std::multiset<Colors>& pool, const std::multiset<Colors>& cost) {
    auto remaining = pool;

    // Pay specific colors first
    for (auto color : cost) {
        if (color == GENERIC) continue;
        auto it = remaining.find(color);
        if (it != remaining.end()) {
            remaining.erase(it);
        } else {
            return false;
        }
    }

    size_t generic_needed = cost.count(GENERIC);
    return remaining.size() >= generic_needed;
}

bool can_afford(Zone::Ownership player_owner, const std::multiset<Colors>& cost) {
    Entity player_entity = get_player_entity(player_owner);
    if (!global_coordinator.entity_has_component<Player>(player_entity)) {
        return false;
    }
    auto& player = global_coordinator.GetComponent<Player>(player_entity);
    return can_afford_pool(player.mana, cost);
}

void spend_mana(Zone::Ownership player_owner, const std::multiset<Colors>& cost) {
    Entity player_entity = get_player_entity(player_owner);
    auto& player = global_coordinator.GetComponent<Player>(player_entity);

    // Pay specific colors first
    for (auto color : cost) {
        if (color == GENERIC) continue;
        auto it = player.mana.find(color);
        if (it != player.mana.end()) {
            player.mana.erase(it);
        }
    }

    // Pay generic with any remaining mana
    size_t generic_needed = cost.count(GENERIC);
    for (size_t i = 0; i < generic_needed; i++) {
        if (!player.mana.empty()) {
            auto it = player.mana.begin();
            player.mana.erase(it);
        }
    }
}

void add_mana(Zone::Ownership player_owner, Colors mana_color) {
    Entity player_entity = get_player_entity(player_owner);
    auto& player = global_coordinator.GetComponent<Player>(player_entity);
    player.mana.insert(mana_color);
}

void empty_mana_pool(Zone::Ownership player_owner) {
    Entity player_entity = get_player_entity(player_owner);
    auto& player = global_coordinator.GetComponent<Player>(player_entity);
    player.mana.clear();
}
