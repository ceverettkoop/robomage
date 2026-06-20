#include "effects.h"

#include <cstdint>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

bool lose_life(Ability &ab, std::shared_ptr<Orderer> orderer) {
    Zone::Ownership lose_controller = ab.controller;
    size_t lose_amount = ab.amount;
    if (!ab.dynamic_amount_expr.empty())
        lose_amount = evaluate_dynamic_amount(ab.dynamic_amount_expr, lose_controller, orderer, ab.target);
    Entity ctrl_entity = (lose_controller == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
    auto &player = global_coordinator.GetComponent<Player>(ctrl_entity);
    player.life_total -= static_cast<int32_t>(lose_amount);
    game_log(
        "%s loses %zu life (now at %d)\n", player_name(lose_controller).c_str(), lose_amount, player.life_total);
    return true;
}

}  // namespace effects
