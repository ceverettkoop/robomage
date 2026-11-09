#include "effects.h"

#include <climits>

#include "../queries/affected.h"
#include "../resolution.h"
#include "../svar_eval.h"

namespace effects {

static bool repeat_check_passes(const Ability &ab);

// RepeatCheckSVar$ / RepeatSVarCompare$: does the process go round again?
static bool repeat_check_passes(const Ability &ab) {
    return compare_svar(evaluate_svar(ab.def->repeat_check_svar, ResolveCtx::of(ab)), ab.def->repeat_svar_compare);
}

// DB$ Repeat (Grindstone: "If two cards that share a color were milled this way, repeat this
// process"): resolve the RepeatSubAbility, then again for as long as RepeatCheckSVar$ passes, at
// most MaxRepeat$ times, read once as it starts. The trailing SubAbility$ links chain after it.
// Suspendable: the iteration and the body's index persist in RepeatLoopRt, so a resume re-derives
// the same child, and each child resolves as a persisted REPEAT_SUB level.
HandlerResult repeat(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    RepeatLoopRt local_rt;
    RepeatLoopRt &rt = ctx.can_suspend() ? ctx.rt<RepeatLoopRt>() : local_rt;
    if (!rt.init) {
        rt.max_iterations = ab.def->dynamic_amount_expr.empty()
                                ? INT_MAX
                                : static_cast<int>(evaluate_amount(ab.def->dynamic_amount_expr, ResolveCtx::of(ab)));
        rt.init = true;
    }
    const int body_count = static_cast<int>(ab.def->repeat_sub_count);
    while (rt.iteration < rt.max_iterations) {
        for (; rt.sub_idx < body_count; ++rt.sub_idx) {
            const Ability &body = ab.subabilities[static_cast<size_t>(rt.sub_idx)];
            if (ctx.can_suspend()) {
                Ability *parent = &ab;
                auto bind = [parent](Ability &sub) { bind_chained_sub(*parent, sub); };
                if (ctx.resolve_child(body, FrameLevel::REPEAT_SUB, rt.sub_idx, rt.iteration, bind, orderer) ==
                    ResolveStatus::SUSPENDED)
                    return HandlerResult::SUSPENDED;
            } else {
                Ability sub = body;
                bind_chained_sub(ab, sub);
                resolve_ability(sub, orderer);
            }
        }
        rt.sub_idx = 0;
        ++rt.iteration;
        if (!repeat_check_passes(ab)) break;
    }
    return HandlerResult::DONE_RUN_SUBS;
}

// MaxRepeat$ names the SVar of the amount, RepeatCheckSVar$ an SVar and RepeatSVarCompare$ its
// comparison, each read as the ability resolves.
bool parse_repeat(AbilityDef &ab, const std::string &key, const std::string &value) {
    if (ab.category != "Repeat") return false;
    if (key == "MaxRepeat") {
        ab.amount_svar = value;
        return true;
    }
    if (key == "RepeatCheckSVar") {
        ab.repeat_check_svar = value;
        return true;
    }
    if (key == "RepeatSVarCompare") {
        ab.repeat_svar_compare = value;
        return true;
    }
    return false;
}

}  // namespace effects
