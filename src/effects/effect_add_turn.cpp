#include "effects.h"

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/zone.h"

extern Game cur_game;

namespace effects {

// DB$ AddTurn (Emrakul, the Aeons Torn's self-cast trigger: "take an extra turn after this
// one"). The Defined$ player takes NumTurns$ extra turns after the current turn (CR 500.7 /
// 720). Defined$ You = the ability's controller (the caster). Each extra turn is pushed onto
// cur_game.extra_turns, the LIFO queue consulted at turn hand-off (Game::advance_step): when
// the current turn ends, the player on top takes the next turn instead of the active player
// flipping to the opponent. General over any "take an extra turn" effect; NumTurns$ N queues N.
bool add_turn(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    // Defined$ You is the only player this effect targets in the current vocab; the player who
    // takes the extra turn(s) is the ability's controller (CR 109.5). amount holds NumTurns$.
    Zone::Ownership taker = ab.controller;
    size_t num_turns = (ab.amount > 0) ? ab.amount : 1;
    for (size_t i = 0; i < num_turns; ++i) cur_game.extra_turns.push_back(taker);
    game_log("%s takes %zu extra turn%s.\n", player_name(taker).c_str(), num_turns,
             num_turns == 1 ? "" : "s");
    return true;
}

}  // namespace effects
