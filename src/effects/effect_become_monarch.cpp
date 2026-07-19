#include "effects.h"

#include "../classes/game.h"
#include "../components/zone.h"
#include "../mana_system.h"

extern Game cur_game;

namespace effects {

// DB$ BecomeMonarch (CR 725, Forth Eorlingas!): the ability's controller becomes the monarch.
// The previous monarch (if any) ceases to be the monarch (725.3); the sourceless inherent
// monarch triggered abilities (end-step draw, steal-on-combat-damage) are driven by the trigger
// scan against cur_game.monarch_entity. General over any "you become the monarch" effect.
HandlerResult become_monarch(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    cur_game.set_monarch(get_player_entity(ab.controller));
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
