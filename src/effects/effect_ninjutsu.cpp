#include "effects.h"

#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/entry_info.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Ninjutsu (CR 702.49a): "Put this card onto the battlefield from your hand tapped and attacking."
// The creature returned as the cost was recorded with the ability, and the ninja attacks the same
// player or planeswalker (CR 702.49c). It was never declared as an attacker, so no "whenever ...
// attacks" ability triggers (CR 508.4). A card that left its owner's hand while the ability was on
// the stack stays where it is.
HandlerResult ninjutsu(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)ctx;
    Entity card = ab.source.get();
    if (card == 0 || !global_coordinator.entity_has_component<Zone>(card)) return HandlerResult::DONE_RUN_SUBS;
    const Zone &zone = global_coordinator.GetComponent<Zone>(card);
    std::string name = entity_name(card);
    if (zone.location != Zone::HAND || zone.owner != ab.controller) {
        game_log("%s is no longer in hand; ninjutsu does nothing.\n", name.c_str());
        return HandlerResult::DONE_RUN_SUBS;
    }
    EntryInfo &entry = entry_info(card);
    entry.enters_tapped = true;
    if (ab.ninjutsu_attack_target.get() != 0) entry.enters_attacking = ab.ninjutsu_attack_target;
    orderer->add_to_zone(false, card, Zone::BATTLEFIELD);
    game_log("%s puts %s onto the battlefield tapped and attacking (ninjutsu)\n",
             player_name(ab.controller).c_str(), name.c_str());
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
