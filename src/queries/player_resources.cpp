#include "player_resources.h"

#include "../classes/game.h"
#include "../ecs/coordinator.h"

void player_gain_life(Entity player_entity, int32_t amount) {
    if (amount <= 0 || !global_coordinator.entity_has_component<Player>(player_entity)) return;
    // CR 119.x life-gain prohibition (Roiling Vortex): if this player can't gain life this turn,
    // the gain is replaced with nothing — no life added and no life_gained_this_turn accrual.
    if (player_cant_gain_life(player_entity)) return;
    auto &pl = global_coordinator.GetComponent<Player>(player_entity);
    pl.life_total += amount;
    pl.life_gained_this_turn += amount;
}

void player_lose_life(Entity player_entity, int32_t amount) {
    if (amount <= 0 || !global_coordinator.entity_has_component<Player>(player_entity)) return;
    auto &pl = global_coordinator.GetComponent<Player>(player_entity);
    pl.life_total -= amount;
    pl.life_lost_this_turn += amount;
}

bool pay_life(Player &pl, int n) {
    if (n <= 0) return true;
    if (!can_pay_life(pl, n)) return false;
    pl.life_total -= n;
    pl.life_lost_this_turn += n;
    return true;
}

bool pay_energy(Player &pl, int n) {
    if (n <= 0) return true;
    if (player_energy(pl) < n) return false;
    pl.add_counters("ENERGY", -n);
    return true;
}

bool player_cant_gain_life(Entity player_entity) {
    if (cur_game.resolved_effects.cant_gain_life_this_turn.empty()) return false;
    Zone::Ownership who = (player_entity == cur_game.player_a_entity) ? Zone::PLAYER_A
                        : (player_entity == cur_game.player_b_entity) ? Zone::PLAYER_B
                                                                      : Zone::UNKNOWN;
    return who != Zone::UNKNOWN && cur_game.resolved_effects.cant_gain_life_this_turn.count(who) > 0;
}
