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
    Zone::Ownership winner = ab.controller;
    std::string src = entity_name(ab.source);
    printf("\n%s wins the game! (%s)\n", player_name(winner).c_str(), src.c_str());
    game_log("%s wins the game!\n", player_name(winner).c_str());
    cur_game.ended = true;
    cur_game.winner = static_cast<int>(winner);
    return HandlerResult::DONE_NO_SUBS;  // original returned early — skip the standard subability loop
}

}  // namespace effects
