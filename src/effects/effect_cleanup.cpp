#include "effects.h"

#include "../classes/game.h"

extern Game cur_game;

namespace effects {

HandlerResult cleanup(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    if (ab.def->clear_remembered) cur_game.resolution.memory.remembered.clear();
    if (ab.def->clear_chosen) cur_game.resolution.memory.chosen_cards.clear();
    if (ab.def->clear_imprinted) cur_game.resolution.memory.imprinted.clear();
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
