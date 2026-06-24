#include "effects.h"

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/creature.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"

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
    const CounterParams *cp = std::get_if<CounterParams>(&ab.params);
    if (cp && !cp->type.empty()) {
        int n = cp->count;
        if (cp->count_from_delve) {
            n = static_cast<int>(cur_game.delve_exiled.size());
            cur_game.delve_exiled.clear();
        }
        if (n <= 0) return true;
        add_counters(counter_tgt, cp->type, n);
        if (cp->type == "P1P1")
            game_log("Put %d +1/+1 counter(s) on creature (now %u/%u).\n", n, cr.power, cr.toughness);
        else
            game_log("Put %d %s counter(s) on creature (now %u/%u).\n", n, cp->type.c_str(), cr.power, cr.toughness);
    }
    return true;
}

bool parse_put_counter(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "CounterType") { effect_params<CounterParams>(ab).type = value; return true; }
    if (key == "CounterNum")  { effect_params<CounterParams>(ab).count = std::stoi(value); return true; }
    return false;
}

}  // namespace effects
