#include "effects.h"

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Suspend upkeep tick (CR 702.62a). Combines the second and third suspend triggered abilities for
// a single suspended card: remove one time counter, and — if that was the last one — let its owner
// cast it without paying its mana cost.
//
// The suspended card lives in the EXILE zone and is not a permanent, so its time counters can't be
// stored in Permanent::counters; they are tracked in cur_game.suspend_time_counters keyed by the
// card entity (ab.source). This handler decrements that count. When it reaches 0 the card stops
// being suspended (702.62b) and its owner is granted a FREE from_suspend impulse-cast permission
// (cur_game.impulse_cast_permission) — the same free-cast-from-exile machinery Ugin's -11 uses —
// so the casting path offers "Cast <card> (from exile, no cost)" and the spell is put on the stack
// with its targets chosen then (CR 702.62d / 601.2b). If the owner never casts it (the permission
// lapses at cleanup), the card simply remains exiled, matching "if you don't, it remains exiled."
// General over any Suspend card.
HandlerResult suspend_tick(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    (void)ctx;
    Entity card = ab.source;
    auto it = cur_game.suspend_time_counters.find(card);
    if (it == cur_game.suspend_time_counters.end()) return HandlerResult::DONE_RUN_SUBS;
    // The card must still be a suspended card (in exile). If it left exile some other way, drop the
    // stale tracking and do nothing.
    if (!global_coordinator.entity_has_component<Zone>(card) ||
        global_coordinator.GetComponent<Zone>(card).location != Zone::EXILE) {
        cur_game.suspend_time_counters.erase(it);
        return HandlerResult::DONE_RUN_SUBS;
    }

    const char *nm = global_coordinator.entity_has_component<CardData>(card)
                         ? global_coordinator.GetComponent<CardData>(card).name.c_str()
                         : "card";
    it->second -= 1;
    game_log("Removed a time counter from %s (%d remaining).\n", nm, it->second);

    if (it->second <= 0) {
        // Last time counter removed (CR 702.62a third ability): the owner may cast it without
        // paying its mana cost. Grant a FREE from_suspend impulse-cast permission to the card's
        // owner (the suspending player) and stop tracking it as suspended.
        Zone::Ownership owner = global_coordinator.GetComponent<Zone>(card).owner;
        Game::ImpulseCastPermission perm;
        perm.resource = Game::ImpulseCastPermission::FREE;
        perm.amount = 0;
        perm.caster = owner;
        perm.from_suspend = true;
        cur_game.impulse_cast_permission[card] = perm;
        cur_game.suspend_time_counters.erase(it);
        game_log("%s may cast %s without paying its mana cost (suspend).\n",
                 player_name(owner).c_str(), nm);
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
