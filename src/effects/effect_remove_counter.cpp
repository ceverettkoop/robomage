#include "effects.h"

#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../queries/counters.h"
#include "../queries/affected.h"

extern Coordinator global_coordinator;

namespace effects {

static void remove_counters(const Ability &ab, const CounterParams &cp, Entity tgt,
                            std::shared_ptr<Orderer> orderer);

// DB$ RemoveCounter | Defined$ Self | CounterType$ <T> | CounterNum$ <N> — remove up
// to N counters of a given type from a permanent (CR 122.5). The counter type/count are
// parsed into CounterParams by parse_put_counter (shared with PutCounter). The counters come off
// the affected objects: the spell/ability's chosen target, or the source itself (Defined$ Self, or
// no Defined$ — Moonshadow removes a -1/-1 counter from itself; a target whose object is gone,
// CR 400.7, loses nothing). Removing more counters than are present just removes all of them
// (122.5 — you can't go below zero). +1/+1 and -1/-1 changes resync the creature's P/T via
// add_counters.
HandlerResult remove_counter(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    const CounterParams *cp = std::get_if<CounterParams>(&ab.def->params);
    if (!cp || cp->type.empty()) return HandlerResult::DONE_RUN_SUBS;
    for (Entity tgt : affected_objects(ab))
        if (global_coordinator.entity_has_component<Permanent>(tgt)) remove_counters(ab, *cp, tgt, orderer);
    return HandlerResult::DONE_RUN_SUBS;
}

// Remove CounterNum$ (All: every) counters of `cp`'s type from permanent `tgt`.
static void remove_counters(const Ability &ab, const CounterParams &cp, Entity tgt,
                            std::shared_ptr<Orderer> orderer) {
    int have = get_counters(tgt, cp.type);
    if (have <= 0) return;
    // CounterNum$ All removes every counter of the type; otherwise the (possibly dynamic) count,
    // capped at what is there (CR 122.5).
    int want = (cp.count_expr == "All") ? have : resolve_counter_num(ab, cp, orderer);
    int n = std::min(want, have);
    if (n <= 0) return;
    add_counters(tgt, cp.type, -n);
    if (global_coordinator.entity_has_component<Creature>(tgt)) {
        auto &cr = global_coordinator.GetComponent<Creature>(tgt);
        game_log("Removed %d %s counter(s) (now %u/%u).\n", n, cp.type.c_str(), cr.power, cr.toughness);
    } else {
        const char *nm = global_coordinator.entity_has_component<CardData>(tgt)
                             ? global_coordinator.GetComponent<CardData>(tgt).name.c_str()
                             : "permanent";
        game_log("Removed %d %s counter(s) from %s.\n", n, cp.type.c_str(), nm);
    }
}

}  // namespace effects
