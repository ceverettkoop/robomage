#include "mana_system.h"

#include <cstddef>
#include <cstdio>

#include "classes/game.h"
#include "components/player.h"
#include "ecs/coordinator.h"
#include "error.h"
#include "input_logger.h"

extern Coordinator global_coordinator;
extern Game cur_game;

Entity get_player_entity(Zone::Ownership player) {
    return (player == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
}

bool can_afford_pool(const std::multiset<Colors> &pool, const std::multiset<Colors> &cost) {
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

//TODO incorporate alternate costs incl. tapping
bool can_afford(Zone::Ownership player_owner, const std::multiset<Colors> &cost) {
    Entity player_entity = get_player_entity(player_owner);
    if (!global_coordinator.entity_has_component<Player>(player_entity)) {
        return false;
    }
    auto &player = global_coordinator.GetComponent<Player>(player_entity);
    return can_afford_pool(player.mana, cost);
}

void spend_mana(Zone::Ownership player_owner, const std::multiset<Colors> &cost) {
    Entity player_entity = get_player_entity(player_owner);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);

    if (!can_afford_pool(player.mana, cost)) {
        non_fatal_error("spend_mana called with insufficient mana in pool");
    }

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

void add_mana(Zone::Ownership player_owner, Colors mana_color, size_t amount) {
    Entity player_entity = get_player_entity(player_owner);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);
    for (size_t i = 0; i < amount; i++) {
        player.mana.insert(mana_color);
    }
}

ManaValue pay_partial(Zone::Ownership player_owner, const ManaValue &cost) {
    Entity player_entity = get_player_entity(player_owner);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);
    ManaValue remaining;

    // Pay specific colors first
    for (auto color : cost) {
        if (color == GENERIC) continue;
        auto it = player.mana.find(color);
        if (it != player.mana.end()) {
            player.mana.erase(it);
        } else {
            remaining.insert(color);
        }
    }

    // Pay generic with any remaining mana
    size_t generic_needed = cost.count(GENERIC);
    for (size_t i = 0; i < generic_needed; i++) {
        if (!player.mana.empty()) {
            auto it = player.mana.begin();
            player.mana.erase(it);
        } else {
            remaining.insert(GENERIC);
        }
    }

    return remaining;
}

void empty_mana_pool(Zone::Ownership player_owner) {
    Entity player_entity = get_player_entity(player_owner);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);
    if (!player.mana.empty() && InputLogger::instance().is_machine_mode()) {
        // Signal the Python env to apply a shaping penalty to the model if this
        // player's side is the one being trained.
        const char* side = (player_owner == Zone::PLAYER_A) ? "A" : "B";
        printf("MANA_WASTED: %s\n", side);
        fflush(stdout);
    }
    player.mana.clear();
}
