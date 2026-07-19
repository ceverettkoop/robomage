#include "effects.h"

#include "../classes/game.h"

extern Game cur_game;

namespace effects {

HandlerResult cleanup(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    if (ab.clear_remembered) cur_game.remembered_entities.clear();
    if (ab.clear_chosen) cur_game.chosen_cards.clear();
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
