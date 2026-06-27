#include "effects.h"

#include <string>
#include <vector>

#include "../classes/colors.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// PutCounterAll (Ajani's +2): put CounterNum counters of CounterType on every permanent
// matching the ValidCards$ filter (e.g. a +1/+1 counter on each Cat you control).
bool put_counter_all(Ability &ab, std::shared_ptr<Orderer> orderer) {
    const CounterParams *cp = std::get_if<CounterParams>(&ab.params);
    std::string ctype = (cp && !cp->type.empty()) ? cp->type : "P1P1";
    int n = cp ? cp->count : 1;
    const std::string ctype2 = cp ? cp->type2 : "";
    const int n2 = cp ? cp->count2 : 0;
    if (n <= 0 && ctype2.empty()) return true;

    std::vector<Entity> targets;
    // ValidCards$ Creature.targetedBy — "each creature targeted by [the parent ability]".
    // The parent (a DB$ Pump) chose the target; PutCounterAll inherited it as ab.target.
    // CR 109/115: this picks out exactly that targeted permanent (Guide of Souls puts the
    // +1/+1s and flying counter on the attacking creature the trigger targeted).
    if (ab.valid_cards_filter.find("targetedBy") != std::string::npos) {
        if (ab.target != 0 && is_battlefield_permanent(ab.target)) targets.push_back(ab.target);
    } else {
        for (auto e : orderer->mEntities)
            if (permanent_matches_filter(e, ab.valid_cards_filter, MatchCtx{ab.controller, ab.source}))
                targets.push_back(e);
    }

    for (auto e : targets) {
        const char *nm = global_coordinator.GetComponent<Permanent>(e).name.c_str();
        if (n > 0) {
            add_counters(e, ctype, n);
            game_log("Put %d %s counter(s) on %s.\n", n, ctype.c_str(), nm);
        }
        // Optional second counter kind (Guide of Souls: a flying counter alongside the +1/+1s).
        if (!ctype2.empty() && n2 > 0) {
            add_counters(e, ctype2, n2);
            game_log("Put %d %s counter(s) on %s.\n", n2, ctype2.c_str(), nm);
        }
    }
    return true;
}

}  // namespace effects
