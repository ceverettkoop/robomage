#include "effects.h"

#include <cstdio>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../game_queries.h"

extern Game cur_game;

namespace effects {

HandlerResult wins_game(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    // First game-ending event wins (CR 104.1: the game ends immediately). If the game is already
    // decided — a deck-out loss processed by the SBA check, an earlier win effect — this
    // resolution must not overwrite the winner.
    if (cur_game.ended) return HandlerResult::DONE_NO_SUBS;
    // Alternative win condition (Thassa's Oracle, Jace Wielder of Mysteries' -8 sub-ability).
    // condition_passed is checked in resolve()'s prologue; reaching here means the player wins.
    cur_game.end_game(ab.controller, entity_name(ab.source));
    return HandlerResult::DONE_NO_SUBS;  // original returned early — skip the standard subability loop
}

}  // namespace effects
