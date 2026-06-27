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
    if (n <= 0) return true;

    std::vector<Entity> targets;
    for (auto e : orderer->mEntities)
        if (permanent_matches_filter(e, ab.valid_cards_filter, MatchCtx{ab.controller, ab.source}))
            targets.push_back(e);

    for (auto e : targets) {
        add_counters(e, ctype, n);
        game_log("Put %d %s counter(s) on %s.\n", n, ctype.c_str(),
                 global_coordinator.GetComponent<Permanent>(e).name.c_str());
    }
    return true;
}

}  // namespace effects
