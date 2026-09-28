#include "effects.h"

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/zone.h"
#include "../action_processor.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../resolution_frame.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static bool is_exiled(Entity card);

namespace effects {

// Suspend upkeep tick (CR 702.62a, second ability): "At the beginning of your upkeep, if this card
// is suspended, remove a time counter from it."
//
// The suspended card lives in the EXILE zone and is not a permanent, so its time counters can't be
// stored in Permanent::counters; they are tracked in cur_game.suspend_time_counters keyed by the
// card entity (ab.source). This handler decrements that count. When it reaches 0 the card stops
// being suspended (702.62b), and removing the last counter triggers the third ability ("When the
// last time counter is removed from this card, if it's exiled, you may play it without paying its
// mana cost if able"), queued as its own triggered ability so players get priority before it
// resolves (suspend_cast). General over any Suspend card.
HandlerResult suspend_tick(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    (void)ctx;
    // The card must still be the suspended object in exile (one that left exile is a new object
    // with no time counters, CR 400.7 / 122.2).
    Entity card = ab.source.get();
    int *counters = cur_game.suspend_time_counters.find(card);
    if (counters == nullptr || !is_exiled(card)) return HandlerResult::DONE_RUN_SUBS;
    *counters -= 1;
    game_log("Removed a time counter from %s (%d remaining).\n", entity_name(card).c_str(),
             *counters);
    if (*counters > 0) return HandlerResult::DONE_RUN_SUBS;
    cur_game.suspend_time_counters.erase(card);
    Ability cast_trigger;
    cast_trigger.ability_type = Ability::TRIGGERED;
    cast_trigger.category = "SuspendCast";
    cast_trigger.source = ObjectRef::of(card);
    cast_trigger.controller = global_coordinator.GetComponent<Zone>(card).owner;
    cur_game.queue_trigger(cast_trigger, entity_name(card) +
                                             " triggers: the last time counter was removed (suspend).");
    return HandlerResult::DONE_RUN_SUBS;
}

// Suspend's third ability (CR 702.62a): "When the last time counter is removed from this card, if
// it's exiled, you may play it without paying its mana cost if able." Its owner may cast it right
// then, as part of this resolution (CR 608.2g, through cast_during_resolution): the spell goes on
// the stack above this ability with its targets chosen now. If it isn't cast, it remains exiled —
// the cast can't be held for later in the turn. General over any Suspend card.
HandlerResult suspend_cast(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    ResolutionCastRt local_rt;
    ResolutionCastRt &rt = ctx.can_suspend() ? ctx.rt<ResolutionCastRt>() : local_rt;
    Entity card = ab.source.get();
    if (rt.stage == ResolutionCastRt::OFFER && (card == 0 || !is_exiled(card))) return HandlerResult::DONE_RUN_SUBS;
    Game::ImpulseCastPermission grant;
    grant.resource = Game::ImpulseCastPermission::FREE;
    if (cast_during_resolution(card, ab.controller, grant, rt, ctx, orderer) ==
        ResolutionCastStatus::SUSPENDED)
        return HandlerResult::SUSPENDED;
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects

// True if `card` is in the exile zone.
static bool is_exiled(Entity card) {
    return global_coordinator.entity_has_component<Zone>(card) &&
           global_coordinator.GetComponent<Zone>(card).location == Zone::EXILE;
}
