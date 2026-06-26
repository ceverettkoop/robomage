#include "effects.h"

#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// AB$ Effect granting "you may cast that card this turn" (Emry, Lurker of the Loch).
//
// Forge models this as a transient continuous Effect object whose static ability
// (MayPlay$ True, AffectedZone$ Graveyard) lets the remembered card be cast from the
// graveyard until end of turn. Rather than instantiate a stack/effect object, we record
// the targeted card in cur_game.may_cast_this_turn — a per-turn cast-permission set
// (CR 601.3e) consumed by the casting path in determine_legal_actions and cleared each
// cleanup. The target's legality (an artifact card in the controller's own graveyard) is
// already enforced when the ability is put on the stack and re-verified at resolution;
// here we only grant the permission for a target that is still in a graveyard.
bool grant_cast(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;

    // DB$ Effect | StaticAbilities$ Unblockable | RememberObjects$ Self — a transient
    // continuous effect that makes the source unblockable until end of turn (Kappa
    // Cannoneer, CR 509.1b / 702.x). Modeled as a per-turn "can't be blocked" mark on the
    // remembered creature (the source), set here and cleared at the cleanup step (514.2),
    // rather than instantiating a continuous-effect object. The mark is read by the combat
    // blocker-legality check so the creature is removed from every blocker's legal list.
    if (ab.effect_static_ability == "Unblockable") {
        Entity who = ab.effect_remember_self ? ab.source : ab.target;
        if (who != 0 && global_coordinator.entity_has_component<Creature>(who) &&
            is_battlefield_permanent(who)) {
            global_coordinator.GetComponent<Creature>(who).cant_be_blocked_this_turn = true;
            const char *nm = global_coordinator.entity_has_component<Permanent>(who)
                                 ? global_coordinator.GetComponent<Permanent>(who).name.c_str()
                                 : "Creature";
            game_log("%s can't be blocked this turn.\n", nm);
        }
        return true;
    }

    Entity tgt = ab.target;
    if (tgt == 0 || !global_coordinator.entity_has_component<Zone>(tgt)) return true;
    if (global_coordinator.GetComponent<Zone>(tgt).location != Zone::GRAVEYARD) return true;

    cur_game.may_cast_this_turn.insert(tgt);

    std::string tname = global_coordinator.entity_has_component<CardData>(tgt)
        ? global_coordinator.GetComponent<CardData>(tgt).name : "card";
    game_log("%s may be cast from the graveyard this turn\n", tname.c_str());
    return true;
}

}  // namespace effects
