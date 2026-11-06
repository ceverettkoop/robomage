#include "effects.h"

#include "../cli_output.h"
#include "../queries/affected.h"
#include "../queries/players.h"
#include "../systems/orderer.h"

namespace effects {

// DB$ Shuffle (Gamble's "then shuffle"): the player it names shuffles their library (CR 701.24a).
HandlerResult shuffle(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)ctx;
    const Zone::Ownership owner = affected_player(ab);
    orderer->shuffle_library(owner);
    game_log("%s shuffles their library\n", player_name(owner).c_str());
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
