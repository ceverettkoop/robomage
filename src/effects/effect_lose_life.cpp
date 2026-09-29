#include "effects.h"

#include <cstdint>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../queries/player_resources.h"
#include "../queries/players.h"
#include "../svar_eval.h"
#include "../queries/affected.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

HandlerResult lose_life(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    Zone::Ownership lose_controller = ab.controller;
    // Defined$ TriggeredActivator — the player who caused the trigger (the caster of the
    // triggering spell) loses the life, not the source's controller. Bound at trigger-fire
    // time (CR 603.x). Mai, Scornful Striker: "Whenever a player casts a noncreature spell,
    // they lose 2 life."
    if (ab.def->defined_triggered_activator && ab.triggered_activator != Zone::UNKNOWN)
        lose_controller = ab.triggered_activator;
    size_t lose_amount = ab.def->amount;
    if (!ab.def->dynamic_amount_expr.empty())
        lose_amount = evaluate_amount(ab.def->dynamic_amount_expr, lose_controller, ab.source.lki_entity(),
                                      ab.target.get());
    // "Target player/opponent loses N life" (Witherbloom Command): the chosen target
    // player is the one who loses the life. The dynamic-amount reference above stays the
    // controller's "you"; only the loser is redirected to the targeted player. A LoseLife with
    // no Defined$ means Forge's default of "You" (the ability's controller) even though
    // sub-ability chaining copies parent.target into ab.target — Thoughtseize's DBLoseLife ("You
    // lose 2 life") must hit the caster, not the discard target (affected_player).
    Zone::Ownership loser = affected_player(ab);
    Entity ctrl_entity = get_player_entity(loser);
    auto &player = global_coordinator.GetComponent<Player>(ctrl_entity);
    // Route through the shared helper so Spectacle's life_lost_this_turn tracker stays in sync.
    player_lose_life(ctrl_entity, static_cast<int32_t>(lose_amount));
    game_log(
        "%s loses %zu life (now at %d)\n", player_name(loser).c_str(), lose_amount, player.life_total);
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
