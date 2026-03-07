#include "machine_io.h"

#include <algorithm>
#include <cassert>
#include <vector>

#include "card_vocab.h"
#include "classes/game.h"
#include "components/ability.h"
#include "components/carddata.h"
#include "components/creature.h"
#include "components/damage.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/zone.h"
#include "ecs/coordinator.h"

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

    // ── Perspective locals ────────────────────────────────────────────────────
    // State is always emitted from the priority player's perspective ("self").
    Zone::Ownership priority_owner = cur_game.player_a_has_priority
        ? Zone::PLAYER_A : Zone::PLAYER_B;
    Zone::Ownership opponent_owner = cur_game.player_a_has_priority
        ? Zone::PLAYER_B : Zone::PLAYER_A;
    Entity priority_entity = cur_game.player_a_has_priority
        ? cur_game.player_a_entity : cur_game.player_b_entity;
    Entity opponent_entity_id = cur_game.player_a_has_priority
        ? cur_game.player_b_entity : cur_game.player_a_entity;

    // ── Header: player states, step, game flags ───────────────────────────────

    // Self first, opponent second (9 floats each)
    push_player_state(state, priority_entity, priority_owner);
    push_player_state(state, opponent_entity_id, opponent_owner);

    // Step one-hot (12 floats)
    for (int i = 0; i < 12; i++) {
        state.push_back((cur_game.cur_step == static_cast<Step>(i)) ? 1.0f : 0.0f);
    }

    // [30]: 1.0 if priority player is the active player (self's turn)
    state.push_back((cur_game.player_a_turn == cur_game.player_a_has_priority) ? 1.0f : 0.0f);
    // [31]: 1.0 if self is Player A
    state.push_back(cur_game.player_a_has_priority ? 1.0f : 0.0f);

    // Stack size (1 float)
    int stack_size = 0;
    for (Entity e = 0; e < MAX_ENTITIES; ++e) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        if (global_coordinator.GetComponent<Zone>(e).location == Zone::STACK) stack_size++;
    }
    state.push_back(stack_size / 10.0f);

    // ── Shared helpers ────────────────────────────────────────────────────────

    // Push N_CARD_TYPES one-hot floats for a card entity (all zeros if unregistered)
    auto push_card_id = [&](Entity e) {
        int idx = -1;
        if (global_coordinator.entity_has_component<CardData>(e)) {
            idx = card_name_to_index(global_coordinator.GetComponent<CardData>(e).name);
        }
        for (int i = 0; i < N_CARD_TYPES; i++) {
            state.push_back(i == idx ? 1.0f : 0.0f);
        }
    };

    // Push an empty 40-float permanent slot (all zeros)
    auto push_empty_perm_slot = [&]() {
        for (int i = 0; i < PERM_SLOT_SIZE; i++) state.push_back(0.0f);
    };

    // ── Creature slots [33-1952] ──────────────────────────────────────────────

    std::vector<Entity> self_creatures, opp_creatures;
    for (Entity e = 0; e < MAX_ENTITIES; ++e) {
        if (!global_coordinator.entity_has_component<Creature>(e)) continue;
        if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto& zone = global_coordinator.GetComponent<Zone>(e);
        if (zone.location != Zone::BATTLEFIELD) continue;
        auto& perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.controller == priority_owner)
            self_creatures.push_back(e);
        else
            opp_creatures.push_back(e);
    }
    std::sort(self_creatures.begin(), self_creatures.end());
    std::sort(opp_creatures.begin(), opp_creatures.end());

    auto push_creature_slot = [&](Entity e) {
        auto& cr   = global_coordinator.GetComponent<Creature>(e);
        auto& perm = global_coordinator.GetComponent<Permanent>(e);
        float dmg  = 0.0f;
        if (global_coordinator.entity_has_component<Damage>(e))
            dmg = global_coordinator.GetComponent<Damage>(e).damage_counters / 10.0f;
        state.push_back(cr.power / 10.0f);
        state.push_back(cr.toughness / 10.0f);
        state.push_back(perm.is_tapped ? 1.0f : 0.0f);
        state.push_back(cr.is_attacking ? 1.0f : 0.0f);
        state.push_back(cr.is_blocking ? 1.0f : 0.0f);
        state.push_back(perm.has_summoning_sickness ? 1.0f : 0.0f);
        state.push_back(dmg);
        state.push_back((perm.controller == priority_owner) ? 1.0f : 0.0f);
        push_card_id(e);
    };

    for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++) {
        if (i < static_cast<int>(self_creatures.size()))
            push_creature_slot(self_creatures[static_cast<size_t>(i)]);
        else
            push_empty_perm_slot();
    }
    for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++) {
        if (i < static_cast<int>(opp_creatures.size()))
            push_creature_slot(opp_creatures[static_cast<size_t>(i)]);
        else
            push_empty_perm_slot();
    }

    // ── Land slots [1953-3872] ────────────────────────────────────────────────
    // Same 40-float format as creature slots; power/toughness/combat/damage = 0.

    std::vector<Entity> self_lands, opp_lands;
    for (Entity e = 0; e < MAX_ENTITIES; ++e) {
        if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        if (global_coordinator.entity_has_component<Creature>(e)) continue;  // already in creature section
        auto& zone = global_coordinator.GetComponent<Zone>(e);
        if (zone.location != Zone::BATTLEFIELD) continue;
        auto& perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.controller == priority_owner)
            self_lands.push_back(e);
        else
            opp_lands.push_back(e);
    }
    std::sort(self_lands.begin(), self_lands.end());
    std::sort(opp_lands.begin(), opp_lands.end());

    auto push_land_slot = [&](Entity e) {
        auto& perm = global_coordinator.GetComponent<Permanent>(e);
        state.push_back(0.0f);  // power (unused for lands)
        state.push_back(0.0f);  // toughness (unused)
        state.push_back(perm.is_tapped ? 1.0f : 0.0f);
        state.push_back(0.0f);  // is_attacking (unused)
        state.push_back(0.0f);  // is_blocking (unused)
        state.push_back(0.0f);  // has_summoning_sickness (unused)
        state.push_back(0.0f);  // damage (unused)
        state.push_back((perm.controller == priority_owner) ? 1.0f : 0.0f);
        push_card_id(e);
    };

    for (int i = 0; i < MAX_LAND_SLOTS; i++) {
        if (i < static_cast<int>(self_lands.size()))
            push_land_slot(self_lands[static_cast<size_t>(i)]);
        else
            push_empty_perm_slot();
    }
    for (int i = 0; i < MAX_LAND_SLOTS; i++) {
        if (i < static_cast<int>(opp_lands.size()))
            push_land_slot(opp_lands[static_cast<size_t>(i)]);
        else
            push_empty_perm_slot();
    }

    // ── Stack slots [3873-4268] ───────────────────────────────────────────────
    // 12 slots × 33 floats: controller_is_self(1) + card_id one-hot(32).
    // Ordered by descending entity ID (higher ID = more recently added = nearer top).
    // Standalone activated-ability entities (no CardData) use their source permanent's
    // card identity.

    std::vector<Entity> stack_ents;
    for (Entity e = 0; e < MAX_ENTITIES; ++e) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        if (global_coordinator.GetComponent<Zone>(e).location == Zone::STACK)
            stack_ents.push_back(e);
    }
    std::sort(stack_ents.begin(), stack_ents.end(), std::greater<Entity>());

    for (int i = 0; i < MAX_STACK_DISPLAY; i++) {
        if (i < static_cast<int>(stack_ents.size())) {
            Entity e = stack_ents[static_cast<size_t>(i)];
            auto& z  = global_coordinator.GetComponent<Zone>(e);
            state.push_back((z.owner == priority_owner) ? 1.0f : 0.0f);

            // Activated ability entities have no CardData; resolve via Ability::source
            Entity card_e = e;
            if (!global_coordinator.entity_has_component<CardData>(e) &&
                global_coordinator.entity_has_component<Ability>(e)) {
                card_e = global_coordinator.GetComponent<Ability>(e).source;
            }
            int idx = -1;
            if (global_coordinator.entity_has_component<CardData>(card_e))
                idx = card_name_to_index(global_coordinator.GetComponent<CardData>(card_e).name);
            for (int j = 0; j < N_CARD_TYPES; j++)
                state.push_back(j == idx ? 1.0f : 0.0f);
        } else {
            // Empty stack slot: controller + card_id
            for (int j = 0; j < STACK_SLOT_SIZE; j++) state.push_back(0.0f);
        }
    }

    // ── Graveyard slots [4269-8364] ───────────────────────────────────────────
    // 64 slots per player × 32 floats (card_id one-hot only).
    // Sorted by entity ID for stability; most recently added card has highest ID.

    std::vector<Entity> self_gy, opp_gy;
    for (Entity e = 0; e < MAX_ENTITIES; ++e) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto& z = global_coordinator.GetComponent<Zone>(e);
        if (z.location != Zone::GRAVEYARD) continue;
        if (z.owner == priority_owner)      self_gy.push_back(e);
        else if (z.owner == opponent_owner) opp_gy.push_back(e);
    }
    std::sort(self_gy.begin(), self_gy.end());
    std::sort(opp_gy.begin(), opp_gy.end());

    for (int i = 0; i < MAX_GY_SLOTS; i++) {
        if (i < static_cast<int>(self_gy.size())) push_card_id(self_gy[static_cast<size_t>(i)]);
        else for (int j = 0; j < GY_SLOT_SIZE; j++) state.push_back(0.0f);
    }
    for (int i = 0; i < MAX_GY_SLOTS; i++) {
        if (i < static_cast<int>(opp_gy.size())) push_card_id(opp_gy[static_cast<size_t>(i)]);
        else for (int j = 0; j < GY_SLOT_SIZE; j++) state.push_back(0.0f);
    }

    // ── Priority player's hand [8365-8684] ────────────────────────────────────
    std::vector<Entity> hand_cards;
    for (Entity e = 0; e < MAX_ENTITIES; ++e) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto& zone = global_coordinator.GetComponent<Zone>(e);
        if (zone.location == Zone::HAND && zone.owner == priority_owner)
            hand_cards.push_back(e);
    }
    std::sort(hand_cards.begin(), hand_cards.end());

    for (int i = 0; i < MAX_HAND_SLOTS; i++) {
        if (i < static_cast<int>(hand_cards.size()))
            push_card_id(hand_cards[static_cast<size_t>(i)]);
        else
            for (int j = 0; j < N_CARD_TYPES; j++) state.push_back(0.0f);
    }

    assert(static_cast<int>(state.size()) == STATE_SIZE);
    return state;
}
