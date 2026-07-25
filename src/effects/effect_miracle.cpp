#include "effects.h"

#include "../classes/game.h"

extern Game cur_game;

namespace effects {

// Miracle (CR 702.94a), the linked triggered ability: "When you reveal this card this way, you may
// cast it by paying [cost] rather than its mana cost." It is synthesized and put on the stack when
// the card's owner chooses to reveal a freshly-drawn first-of-turn miracle card (see
// proc_mandatory_choice's miracle-reveal branch). Having sat on the stack — so the opponent had a
// window to respond to the revealed card — it resolves here by arming the owner's single immediate
// "cast it for its miracle cost / do not cast" decision (Game::miracle_cast_pending). That decision
// is presented right away by proc_mandatory_choice's miracle-cast branch — the cast, if taken,
// happens then and there rather than lingering as a castable option through the turn. General over
// any miracle card. ab.source is the card in hand; ab.controller is its owner.
HandlerResult miracle_cast(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    (void)ctx;
    cur_game.miracle_cast_pending = ab.source;
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
