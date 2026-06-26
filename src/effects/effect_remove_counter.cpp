#include "effects.h"

#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"

extern Coordinator global_coordinator;

namespace effects {

// DB$ RemoveCounter | Defined$ Self | CounterType$ <T> | CounterNum$ <N> — remove up
// to N counters of a given type from a permanent (CR 122.5). The counter type/count are
// parsed into CounterParams by parse_put_counter (shared with PutCounter). The target is
// the spell/ability's chosen target when one was selected, otherwise the source itself
// (Defined$ Self — Moonshadow removes a -1/-1 counter from itself). Removing more counters
// than are present just removes all of them (122.5 — you can't go below zero). +1/+1 and
// -1/-1 changes resync the creature's P/T via add_counters.
bool remove_counter(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    Entity tgt =
        (ab.target != 0 && global_coordinator.entity_has_component<Permanent>(ab.target)) ? ab.target
                                                                                          : ab.source;
    if (!global_coordinator.entity_has_component<Permanent>(tgt)) return true;
    const CounterParams *cp = std::get_if<CounterParams>(&ab.params);
    if (!cp || cp->type.empty()) return true;
    int have = get_counters(tgt, cp->type);
    if (have <= 0) return true;
    int n = (cp->count > 0) ? std::min(cp->count, have) : have;
    add_counters(tgt, cp->type, -n);
    if (global_coordinator.entity_has_component<Creature>(tgt)) {
        auto &cr = global_coordinator.GetComponent<Creature>(tgt);
        game_log("Removed %d %s counter(s) (now %u/%u).\n", n, cp->type.c_str(), cr.power, cr.toughness);
    } else {
        const char *nm = global_coordinator.entity_has_component<CardData>(tgt)
                             ? global_coordinator.GetComponent<CardData>(tgt).name.c_str()
                             : "permanent";
        game_log("Removed %d %s counter(s) from %s.\n", n, cp->type.c_str(), nm);
    }
    return true;
}

}  // namespace effects
