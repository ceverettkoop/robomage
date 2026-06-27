#include "effects.h"

#include "../action_processor.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// ImmediateTrigger ("when you do ...") is a reflexive trigger created mid-resolution of
// its parent ability (Ajani's [0]: create a token, then if you control a red permanent
// other than Ajani, deal damage). We resolve it inline: gate on the ConditionPresent$
// filter, then run the Execute$ sub-ability (selecting a target if it needs one). Any
// Cleanup sub-ability runs regardless so remembered objects are cleared. Returns false so
// resolve() does not also chain the sub-abilities (we ran them here).
bool immediate_trigger(Ability &ab, std::shared_ptr<Orderer> orderer) {
    bool fire = ab.condition_present.empty();
    if (!fire) {
        for (auto e : orderer->mEntities) {
            if (permanent_matches_filter(e, ab.condition_present, MatchCtx{ab.controller, ab.source})) {
                fire = true;
                break;
            }
        }
    }

    for (auto sub : ab.subabilities) {
        sub.source = ab.source;
        sub.controller = ab.controller;
        if (sub.category == "Cleanup") {
            sub.resolve(orderer);
            continue;
        }
        if (!fire) continue;  // condition not met: skip the reflexive effect
        if (sub.valid_tgts != "N_A" && has_legal_targets(sub, orderer))
            select_target(sub, orderer, ab.controller);
        sub.resolve(orderer);
    }
    return false;
}

}  // namespace effects
