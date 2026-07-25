#include "effects.h"

#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Warp end-step exile (a 2025 keyword; not in the checked-in CR snapshot). This is the fire
// ability of the one-shot delayed triggered ability registered by mark_warp_permanent when a
// warp-cast permanent enters: "at the beginning of the next end step, exile it." On resolution it
// (1) exiles ab.source if it is still on the battlefield as the same object, and (2) grants its
// owner a lasting cast-from-exile permission — the card may be cast on a later turn for its NORMAL
// cost for as long as it remains exiled this way (a warp ImpulseCastPermission).
//
// A warp-cast permanent that has already left the battlefield (died, was bounced) when the end
// step arrives simply follows normal rules — the exile does nothing and no permission is granted
// (the object it would grant a recast of is gone). General over any warp card.
HandlerResult warp_exile(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)ctx;
    Entity card = ab.source;
    // Only exile it if it is still the same object on the battlefield (CR 400.7): a re-entered
    // object is new and is not the warp-cast one.
    if (!is_battlefield_permanent(card)) return HandlerResult::DONE_RUN_SUBS;

    Zone::Ownership owner = global_coordinator.GetComponent<Zone>(card).owner;
    std::string nm = global_coordinator.entity_has_component<CardData>(card)
                         ? global_coordinator.GetComponent<CardData>(card).name
                         : "It";

    orderer->add_to_zone(false, card, Zone::EXILE);

    // Grant the recast-from-exile permission: NORMAL cost (the card's own mana cost), persisting
    // while the card remains exiled (the warp flag exempts it from per-turn cleanup). The casting
    // path offers it from exile at the card's normal timing (sorcery speed for a creature).
    Game::ImpulseCastPermission perm;
    perm.resource = Game::ImpulseCastPermission::NORMAL;
    perm.amount = 0;
    perm.caster = owner;
    perm.warp = true;
    cur_game.impulse_cast_permission[card] = perm;

    game_log("%s is exiled with warp; %s may cast it from exile for its normal cost.\n",
             nm.c_str(), player_name(owner).c_str());
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
