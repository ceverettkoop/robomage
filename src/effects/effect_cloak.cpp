#include "effects.h"

#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/cast_option.h"
#include "../components/carddata.h"
#include "../queries/affected.h"
#include "../queries/face_down.h"
#include "../resolution_memory.h"
#include "../systems/orderer.h"

extern Game cur_game;

namespace effects {

// CR 701.58a: each card goes onto the battlefield face down as a 2/2 with ward {2}, one at a time
// (CR 701.58e). A card that stays where it was isn't cloaked (CR 701.58f), so what follows has
// nothing to name.
HandlerResult cloak(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &) {
    for (Entity card : affected_objects(ab)) {
        if (!orderer->put_onto_battlefield_face_down(card, SpellOverlay::CLOAKED)) continue;
        game_log("%s cloaks %s.\n", player_name(ab.controller).c_str(), face_down_card(card)->name.c_str());
        report_object(ab, EffectObjects::MOVED, card);
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
