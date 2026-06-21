#include "effects.h"

#include <cstdio>

#include "../classes/game.h"
#include "../cli_output.h"

extern Game cur_game;

namespace effects {

bool wins_game(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    // Alternative win condition (Thassa's Oracle). condition_passed is checked in
    // resolve()'s prologue; reaching here means the player wins.
    Zone::Ownership winner = ab.controller;
    if (winner == Zone::PLAYER_A) {
        printf("\nPlayer A wins the game! (Thassa's Oracle)\n");
    } else {
        printf("\nPlayer B wins the game! (Thassa's Oracle)\n");
    }
    game_log("%s wins the game!\n", player_name(winner).c_str());
    cur_game.ended = true;
    cur_game.winner = static_cast<int>(winner);
    return false;  // original returned early — skip the standard subability loop
}

}  // namespace effects
