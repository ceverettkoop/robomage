#include "effects.h"

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"

extern Coordinator global_coordinator;

namespace effects {

bool put_counter(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    // Use target if set (e.g. from a Pump parent), otherwise put counters on source
    // (Defined$ Self — e.g. Aether Vial's upkeep "put a charge counter on it"). Counters
    // can go on any permanent, not just creatures (CR 122.1), so gate on Permanent: a
    // creature gets +1/+1-style P/T resync via add_counters, a non-creature (Aether Vial,
    // an artifact) just accrues the typed counter in its counter map.
    Entity counter_tgt =
        (ab.target != 0 && global_coordinator.entity_has_component<Permanent>(ab.target)) ? ab.target : ab.source;
    if (!global_coordinator.entity_has_component<Permanent>(counter_tgt)) return true;
    const CounterParams *cp = std::get_if<CounterParams>(&ab.params);
    if (cp && !cp->type.empty()) {
        int n = cp->count;
        if (n <= 0) return true;
        int total = add_counters(counter_tgt, cp->type, n);
        if (global_coordinator.entity_has_component<Creature>(counter_tgt)) {
            auto &cr = global_coordinator.GetComponent<Creature>(counter_tgt);
            if (cp->type == "P1P1")
                game_log("Put %d +1/+1 counter(s) on creature (now %u/%u).\n", n, cr.power, cr.toughness);
            else
                game_log("Put %d %s counter(s) on creature (now %u/%u).\n", n, cp->type.c_str(), cr.power, cr.toughness);
        } else {
            const char *nm = global_coordinator.entity_has_component<CardData>(counter_tgt)
                                 ? global_coordinator.GetComponent<CardData>(counter_tgt).name.c_str()
                                 : "permanent";
            game_log("Put %d %s counter(s) on %s (now %d).\n", n, cp->type.c_str(), nm, total);
        }
    }
    return true;
}

bool parse_put_counter(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "CounterType") {
        auto &cp = effect_params<CounterParams>(ab);
        cp.type = value;
        // Forge defaults CounterNum$ to 1 when omitted (e.g. Kappa Cannoneer's
        // "put a +1/+1 counter"). A later explicit CounterNum$ overrides this.
        if (cp.count == 0) cp.count = 1;
        return true;
    }
    if (key == "CounterNum")  { effect_params<CounterParams>(ab).count = std::stoi(value); return true; }
    return false;
}

}  // namespace effects
