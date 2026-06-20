#include "effects.h"

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/creature.h"
#include "../ecs/coordinator.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

bool put_counter(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    // Use target if set (e.g. from a Pump parent), otherwise put counters on source
    Entity counter_tgt =
        (ab.target != 0 && global_coordinator.entity_has_component<Creature>(ab.target)) ? ab.target : ab.source;
    if (!global_coordinator.entity_has_component<Creature>(counter_tgt)) return true;
    auto &cr = global_coordinator.GetComponent<Creature>(counter_tgt);
    if (ab.counter_type == "P1P1") {
        int n = ab.counter_count;
        if (ab.counter_count_from_delve) {
            n = static_cast<int>(cur_game.delve_exiled.size());
            cur_game.delve_exiled.clear();
        }
        if (n <= 0) return true;
        cr.plus_one_counters += n;
        recompute_pt(cr);
        game_log("Put %d +1/+1 counter(s) on creature (now %u/%u).\n", n, cr.power, cr.toughness);
    }
    return true;
}

}  // namespace effects
