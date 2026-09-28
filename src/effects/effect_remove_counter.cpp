#include "effects.h"

#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../queries/counters.h"

extern Coordinator global_coordinator;

namespace effects {

// DB$ RemoveCounter | Defined$ Self | CounterType$ <T> | CounterNum$ <N> — remove up
// to N counters of a given type from a permanent (CR 122.5). The counter type/count are
// parsed into CounterParams by parse_put_counter (shared with PutCounter). The target is
// the spell/ability's chosen target when one was selected, otherwise the source itself
// (Defined$ Self — Moonshadow removes a -1/-1 counter from itself). Removing more counters
// than are present just removes all of them (122.5 — you can't go below zero). +1/+1 and
// -1/-1 changes resync the creature's P/T via add_counters.
HandlerResult remove_counter(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    // A target whose object is gone (CR 400.7) loses nothing, and the source isn't used instead.
    if (!ab.target.empty() && ab.target.get() == 0) return HandlerResult::DONE_RUN_SUBS;
    const Entity t = ab.target.get();
    Entity tgt = (t != 0 && global_coordinator.entity_has_component<Permanent>(t)) ? t : ab.source.get();
    if (!global_coordinator.entity_has_component<Permanent>(tgt)) return HandlerResult::DONE_RUN_SUBS;
    const CounterParams *cp = std::get_if<CounterParams>(&ab.params);
    if (!cp || cp->type.empty()) return HandlerResult::DONE_RUN_SUBS;
    int have = get_counters(tgt, cp->type);
    if (have <= 0) return HandlerResult::DONE_RUN_SUBS;
    // CounterNum$ All removes every counter of the type; otherwise the (possibly dynamic) count,
    // capped at what is there (CR 122.5).
    int want = (cp->count_expr == "All") ? have : resolve_counter_num(ab, *cp, orderer);
    int n = std::min(want, have);
    if (n <= 0) return HandlerResult::DONE_RUN_SUBS;
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
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
