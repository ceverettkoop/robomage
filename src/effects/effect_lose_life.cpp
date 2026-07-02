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
    // Defined$ TriggeredActivator — the player who caused the trigger (the caster of the
    // triggering spell) loses the life, not the source's controller. Bound at trigger-fire
    // time (CR 603.x). Mai, Scornful Striker: "Whenever a player casts a noncreature spell,
    // they lose 2 life."
    if (ab.defined_triggered_activator && ab.triggered_activator != Zone::UNKNOWN)
        lose_controller = ab.triggered_activator;
    size_t lose_amount = ab.amount;
    if (!ab.dynamic_amount_expr.empty())
        lose_amount = evaluate_dynamic_amount(ab.dynamic_amount_expr, lose_controller, orderer, ab.target, ab.source);
    // "Target player/opponent loses N life" (Witherbloom Command): the chosen target
    // player is the one who loses the life. The dynamic-amount reference above stays the
    // controller's "you"; only the loser is redirected to the targeted player. Redirect
    // ONLY when this ability itself declared the target — its own ValidTgts$, or an
    // explicit Defined$ naming the parent's target. A LoseLife with no Defined$ means
    // Forge's default of "You" (the ability's controller) even though sub-ability
    // chaining copies parent.target into ab.target — Thoughtseize's DBLoseLife ("You
    // lose 2 life") must hit the caster, not the discard target.
    Zone::Ownership loser = lose_controller;
    bool targets_player = (ab.valid_tgts != "N_A") || ab.defined == "Targeted" ||
                          ab.defined == "ParentTarget" || ab.defined == "Parent";
    if (targets_player && ab.target != 0 && global_coordinator.entity_has_component<Player>(ab.target))
        loser = (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    Entity ctrl_entity = (loser == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
    auto &player = global_coordinator.GetComponent<Player>(ctrl_entity);
    player.life_total -= static_cast<int32_t>(lose_amount);
    game_log(
        "%s loses %zu life (now at %d)\n", player_name(loser).c_str(), lose_amount, player.life_total);
    return true;
}

}  // namespace effects
