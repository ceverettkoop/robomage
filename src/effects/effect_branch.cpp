#include "effects.h"

#include "../queries/affected.h"
#include "../resolution.h"
#include "../svar_eval.h"

namespace effects {

static bool branch_condition_holds(const Ability &ab);

// BranchConditionSVar$ / BranchConditionSVarCompare$: the sum of the condition's terms against its
// comparison.
static bool branch_condition_holds(const Ability &ab) {
    int total = 0;
    for (const std::string &term : ab.def->branch_terms) total += evaluate_svar(term, ResolveCtx::of(ab));
    return compare_svar(total, ab.def->repeat_svar_compare);
}

// DB$ Branch (Goblin Welder: "If both targets are still legal"): resolve TrueSubAbility$ or
// FalseSubAbility$, whichever its condition picks as it runs. The side persists in BranchRt, read
// once, so a resume re-derives the same child whatever the child has done to the condition; the
// child resolves as a persisted SUB level.
HandlerResult branch(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    BranchRt local_rt;
    BranchRt &rt = ctx.can_suspend() ? ctx.rt<BranchRt>() : local_rt;
    if (!rt.init) {
        rt.side = branch_condition_holds(ab) ? ab.def->branch_true : ab.def->branch_false;
        rt.init = true;
    }
    if (rt.side < 0) return HandlerResult::DONE_RUN_SUBS;
    const Ability &side = ab.subabilities[static_cast<size_t>(rt.side)];
    if (ctx.can_suspend()) {
        Ability *parent = &ab;
        auto bind = [parent](Ability &sub) { bind_chained_sub(*parent, sub); };
        if (ctx.resolve_child(side, FrameLevel::SUB, rt.side, -1, bind, orderer) == ResolveStatus::SUSPENDED)
            return HandlerResult::SUSPENDED;
    } else {
        Ability sub = side;
        bind_chained_sub(ab, sub);
        resolve_ability(sub, orderer);
    }
    return HandlerResult::DONE_RUN_SUBS;
}

// BranchConditionSVar$ names the SVar of the condition, BranchConditionSVarCompare$ its comparison.
// The two sides are composite params (parse.cpp).
bool parse_branch(AbilityDef &ab, const std::string &key, const std::string &value) {
    if (ab.category != "Branch") return false;
    if (key == "BranchConditionSVar") {
        ab.repeat_check_svar = value;
        return true;
    }
    if (key == "BranchConditionSVarCompare") {
        ab.repeat_svar_compare = value;
        return true;
    }
    return false;
}

}  // namespace effects
