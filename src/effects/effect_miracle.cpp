#include "effects.h"

#include <string>

#include "../action_processor.h"
#include "../classes/action.h"
#include "../classes/game.h"
#include "../game_queries.h"
#include "../resolution_frame.h"
#include "../systems/state_manager.h"

extern Game cur_game;

namespace effects {

// Miracle (CR 702.94a), the linked triggered ability: "When you reveal this card this way, you may
// cast it by paying [cost] rather than its mana cost." It is synthesized and put on the stack when
// the card's owner chooses to reveal a freshly-drawn first-of-turn miracle card (see
// proc_mandatory_choice's miracle-reveal branch), so the opponent had a window to respond to the
// revealed card. The cast is part of this resolution (CR 608.2g, cast_during_resolution): the owner
// is offered it while the card is still in their hand and castable for its miracle cost (timing
// ignored; targets, prohibitions and the cost checked by miracle_castable), and the spell goes on
// the stack above this ability. If it isn't cast then, the opportunity is gone. General over any
// miracle card. ab.source is the card in hand; ab.controller is its owner.
HandlerResult miracle_cast(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    ResolutionCastRt local_rt;
    ResolutionCastRt &rt = ctx.can_suspend() ? ctx.rt<ResolutionCastRt>() : local_rt;
    Entity card = ab.source;
    bool castable =
        rt.stage == ResolutionCastRt::OFFER && miracle_castable(card, ab.controller, orderer);
    const std::string nm = entity_name(card);
    LegalAction cast(CAST_SPELL, card, "Cast " + nm + " (miracle)");
    cast.category = ActionCategory::CAST_SPELL;
    cast.use_alt_cost = true;
    cast.option_ordinal = 1;
    if (cast_during_resolution(cast, ab.controller, castable, "Cast " + nm + " for its miracle cost",
                               rt, ctx, orderer) == ResolutionCastStatus::SUSPENDED)
        return HandlerResult::SUSPENDED;
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
