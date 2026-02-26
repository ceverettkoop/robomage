#include "machine_io.h"

#include <algorithm>
#include <vector>

#include "classes/game.h"
#include "components/creature.h"
#include "components/damage.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "systems/stack_manager.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static void push_player_state(std::vector<float>& out, Entity player_entity, Zone::Ownership owner) {
    auto& player = global_coordinator.GetComponent<Player>(player_entity);

    out.push_back(player.life_total / 20.0f);
    out.push_back(player.lands_played_this_turn / 10.0f);  // reuse slot as hand-size proxy
    out.push_back(player.poison_counters / 10.0f);

    // Count mana by color
    int mana_counts[6] = {0, 0, 0, 0, 0, 0};  // W U B R G C
    for (Colors c : player.mana) {
        int idx = static_cast<int>(c);
        if (idx >= 0 && idx < 6) mana_counts[idx]++;
    }
    for (int i = 0; i < 6; i++) out.push_back(mana_counts[i] / 10.0f);

    // Hand size: count cards in hand owned by this player
    int hand_size = 0;
    for (Entity e = 0; e < MAX_ENTITIES; ++e) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto& zone = global_coordinator.GetComponent<Zone>(e);
        if (zone.location == Zone::HAND && zone.owner == owner) hand_size++;
    }
    // Overwrite the hand-size slot (index 1 of this player block = out[out.size()-8])
    out[out.size() - 8] = hand_size / 10.0f;
}

std::vector<float> serialize_state() {
    std::vector<float> state;
    state.reserve(static_cast<size_t>(STATE_SIZE));

    // Player states (9 floats each)
    push_player_state(state, cur_game.player_a_entity, Zone::PLAYER_A);
    push_player_state(state, cur_game.player_b_entity, Zone::PLAYER_B);

    // Step one-hot (12 floats)
    for (int i = 0; i < 12; i++) {
        state.push_back((cur_game.cur_step == static_cast<Step>(i)) ? 1.0f : 0.0f);
    }

    // Active player, priority player, stack size (3 floats)
    state.push_back(cur_game.player_a_turn ? 1.0f : 0.0f);
    state.push_back(cur_game.player_a_has_priority ? 1.0f : 0.0f);

    // Stack size: count entities on stack
    int stack_size = 0;
    for (Entity e = 0; e < MAX_ENTITIES; ++e) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        if (global_coordinator.GetComponent<Zone>(e).location == Zone::STACK) stack_size++;
    }
    state.push_back(stack_size / 10.0f);

    // Battlefield slots: collect A's and B's creatures separately
    std::vector<Entity> a_creatures, b_creatures;
    for (Entity e = 0; e < MAX_ENTITIES; ++e) {
        if (!global_coordinator.entity_has_component<Creature>(e)) continue;
        if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto& zone = global_coordinator.GetComponent<Zone>(e);
        if (zone.location != Zone::BATTLEFIELD) continue;
        auto& perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.controller == Zone::PLAYER_A)
            a_creatures.push_back(e);
        else
            b_creatures.push_back(e);
    }
    std::sort(a_creatures.begin(), a_creatures.end());
    std::sort(b_creatures.begin(), b_creatures.end());

    auto push_creature_slot = [&](Entity e) {
        auto& cr = global_coordinator.GetComponent<Creature>(e);
        auto& perm = global_coordinator.GetComponent<Permanent>(e);
        float dmg = 0.0f;
        if (global_coordinator.entity_has_component<Damage>(e)) {
            dmg = global_coordinator.GetComponent<Damage>(e).damage_counters / 10.0f;
        }
        state.push_back(cr.power / 10.0f);
        state.push_back(cr.toughness / 10.0f);
        state.push_back(perm.is_tapped ? 1.0f : 0.0f);
        state.push_back(cr.is_attacking ? 1.0f : 0.0f);
        state.push_back(cr.is_blocking ? 1.0f : 0.0f);
        state.push_back(perm.has_summoning_sickness ? 1.0f : 0.0f);
        state.push_back(dmg);
        state.push_back((perm.controller == Zone::PLAYER_A) ? 1.0f : 0.0f);
    };

    auto push_empty_slot = [&]() {
        for (int i = 0; i < 8; i++) state.push_back(0.0f);
    };

    // 10 slots for A, 10 for B
    for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++) {
        if (i < static_cast<int>(a_creatures.size()))
            push_creature_slot(a_creatures[static_cast<size_t>(i)]);
        else
            push_empty_slot();
    }
    for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++) {
        if (i < static_cast<int>(b_creatures.size()))
            push_creature_slot(b_creatures[static_cast<size_t>(i)]);
        else
            push_empty_slot();
    }

    return state;
}
