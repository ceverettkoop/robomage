#include "effects.h"

#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../game_queries.h"
#include "../mana_system.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// DB$/AB$ Earthbend N (Badgermole Cub: Num$ 1; Ba Sing Se: Num$ 2). CR-style keyword action:
//   "Target land you control becomes a 0/0 creature with haste that's still a land. Put N
//    +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped
//    under its owner's control."
//
// The land-animation reuses the Animate extension points on Permanent (animate_make_creature,
// animate_set_pt + 0/0 base, animate_added_keywords = Haste, Duration Permanent). The added
// types/keywords persist on the permanent so the layer system reapplies them each SBA pass; the
// Creature/Damage components are bootstrapped here (and re-bootstrapped each pass if lost) via
// apply_animate_creature_bootstrap. The N +1/+1 counters are added immediately so the
// permanent is N/N (not the transient 0/0) before any state-based action runs. The
// return-tapped clause is a delayed trigger (CR 603.6e) watching this specific permanent's
// departure from the battlefield TO the graveyard or exile ("when it dies or is exiled");
// it fires once and returns the card to the battlefield tapped. A bounce to hand or a shuffle
// into the library does NOT fire it — the trigger expires unfired (the object is gone).
HandlerResult earthbend(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    Entity tgt = ab.target;
    // Target must still be a land on the battlefield at resolution (CR 608.2b re-check).
    if (tgt == 0 || !is_battlefield_permanent(tgt)) return HandlerResult::DONE_RUN_SUBS;
    auto &perm = global_coordinator.GetComponent<Permanent>(tgt);
    bool is_land = false;
    for (const auto &t : perm.types)
        if (t.kind == TYPE && t.name == "Land") { is_land = true; break; }
    if (!is_land) return HandlerResult::DONE_RUN_SUBS;

    int n = static_cast<int>(ab.amount);
    if (n <= 0) n = 1;

    // "becomes a 0/0 creature with haste that's still a land" — bake the land-animation onto the
    // permanent so the layer system reapplies it for the rest of the game (Duration Permanent).
    perm.animate_make_creature = true;
    perm.animate_set_pt = true;
    perm.animate_power = 0;
    perm.animate_toughness = 0;
    bool has_haste = false;
    for (const auto &kw : perm.animate_added_keywords)
        if (kw == "Haste") { has_haste = true; break; }
    if (!has_haste) perm.animate_added_keywords.push_back("Haste");
    // It is no longer summoning sick relative to attacking/tapping (it has haste), but clear the
    // flag too so the haste check isn't even needed.
    perm.has_summoning_sickness = false;

    // Turn the land into a 0/0 creature now (Creature + Damage components), then layer the N
    // +1/+1 counters on top so it is N/N before the toughness-0 state-based action runs.
    apply_animate_creature_bootstrap(tgt);
    add_counters(tgt, "P1P1", n);

    game_log("%s becomes a 0/0 creature with haste that's still a land; put %d +1/+1 counter(s) on it.\n",
             perm.name.c_str(), n);

    // Register the delayed "when it dies or is exiled, return it tapped under its owner's
    // control" trigger (CR 603.6e). The fire ability is a generic ChangeZone of the watched card
    // itself (Defined$ Self) from wherever it went back onto the battlefield, entering tapped.
    // The destination filter restricts firing to graveyard/exile departures per the oracle text.
    Ability fire_ab;
    fire_ab.ability_type = Ability::TRIGGERED;
    fire_ab.category = "ChangeZone";
    fire_ab.defined_self = true;
    fire_ab.source = tgt;                 // the card to return (its Zone.owner names its owner)
    fire_ab.origin = Zone::GRAVEYARD;     // unused for a Defined$ Self move; informational
    fire_ab.destination = Zone::BATTLEFIELD;
    fire_ab.enters_tapped = true;

    DelayedTrigger dt;
    dt.ability = fire_ab;
    dt.fire_on = Events::CARD_CHANGED_ZONE;
    dt.owner_entity = get_player_entity(perm.controller);
    dt.fire_on_turn = cur_game.turn;
    dt.watch_entity = tgt;
    dt.fire_on_leave_battlefield = true;
    dt.fire_dest_zones = {Zone::GRAVEYARD, Zone::EXILE};
    cur_game.delayed_triggers.push_back(dt);

    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
