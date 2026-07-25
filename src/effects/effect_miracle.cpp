#include "effects.h"

#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../ecs/coordinator.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Miracle (CR 702.94a), the linked triggered ability: "When you reveal this card this way, you may
// cast it by paying [cost] rather than its mana cost." It is synthesized and put on the stack when
// the card's owner chooses to reveal a freshly-drawn first-of-turn miracle card (see
// proc_mandatory_choice's miracle-reveal branch). Having sat on the stack — so the opponent had a
// window to respond to the revealed card — it resolves here by opening the miracle-cast window for
// the source card: can_afford_alt then offers the "Cast <card> (miracle)" alternate-cost line while
// the card sits in that window, so the owner makes the actual cast decision at their following
// priority. The window is cleared each cleanup (the opportunity lapses at end of turn). General over
// any miracle card. ab.source is the card in hand; ab.controller is its owner.
HandlerResult miracle_cast(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    (void)ctx;
    Entity card = ab.source;
    cur_game.miracle_window.insert(card);
    std::string nm = global_coordinator.entity_has_component<CardData>(card)
                         ? global_coordinator.GetComponent<CardData>(card).name
                         : "the card";
    game_log("%s may cast %s for its miracle cost.\n", player_name(ab.controller).c_str(),
             nm.c_str());
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
