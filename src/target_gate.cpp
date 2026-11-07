#include "target_gate.h"

#include <algorithm>
#include <utility>

#include "coverage.h"
#include "queries/affected.h"
#include "queries/players.h"
#include "targeting.h"

// A constraint a target candidate must meet beyond its own legality. `applies` says whether the
// row reads this pick at all. A row that isn't `required` gives way when it refuses every
// candidate: all of them are offered.
struct TargetGateRow {
    const char *name;  // the --coverage key, hit when the row refuses a candidate
    bool required;
    bool (*applies)(const TargetPick &pick);
    bool (*allows)(const TargetPick &pick, Entity candidate);
};

static bool holds_a_target(const TargetPick &pick);
static bool not_chosen_yet(const TargetPick &pick, Entity candidate);
static bool one_controller_for_all(const TargetPick &pick);
static bool shares_the_controller(const TargetPick &pick, Entity candidate);
static bool answers_to_a_defined_controller(const TargetPick &pick);
static bool controlled_as_defined(const TargetPick &pick, Entity candidate);
static const Ability *instruction_answering_to_this_target(const Ability &ab);
static bool an_instruction_answers_to_this_target(const TargetPick &pick);
static bool leaves_that_instruction_a_target(const TargetPick &pick, Entity candidate);
static bool a_target_is_changing(const TargetPick &pick);
static bool is_another_target(const TargetPick &pick, Entity candidate);
static bool cost_is_gated(const TargetPick &pick);
static bool leaves_cost_payable(const TargetPick &pick, Entity candidate);

static const TargetGateRow kTargetGateRows[] = {
    // CR 115.3: the same target can't be chosen twice for one instance of the word "target".
    {"target_gate:distinct", true, holds_a_target, not_chosen_yet},
    // TargetsWithSameController$: one graveyard for all of them (Unlicensed Hearse).
    {"target_gate:same_controller", true, one_controller_for_all, shares_the_controller},
    // TargetsWithDefinedController$: a target among what the player the name resolves to controls.
    {"target_gate:defined_controller", true, answers_to_a_defined_controller, controlled_as_defined},
    // CR 601.2c: a target an instruction chained to this one has no target to go with isn't offered
    // (Goblin Welder's pair), so a first pick can't leave the second's menu empty.
    {"target_gate:leaves_a_target", true, an_instruction_answers_to_this_target, leaves_that_instruction_a_target},
    // CR 115.7a: a changed target becomes another legal target (CR 115.5: not the object itself).
    {"target_gate:another_target", true, a_target_is_changing, is_another_target},
    // CR 601.2f: a cost reduced only for some targets (Mystical Dispute).
    {"target_gate:cost_payable", false, cost_is_gated, leaves_cost_payable},
};

static bool holds_a_target(const TargetPick &pick) { return pick.picked > 0; }

static bool not_chosen_yet(const TargetPick &pick, Entity candidate) {
    for (int i = 0; i < pick.picked; i++)
        if (pick.ability.targets[static_cast<size_t>(i)].get() == candidate) return false;
    return true;
}

static bool one_controller_for_all(const TargetPick &pick) {
    return pick.picked > 0 && pick.ability.def->targets_same_controller;
}

static bool shares_the_controller(const TargetPick &pick, Entity candidate) {
    return source_controller(candidate) == source_controller(pick.ability.targets[0].get());
}

static bool answers_to_a_defined_controller(const TargetPick &pick) {
    return !pick.ability.def->targets_defined_controller.is(ObjectSpec::DEFAULT) && !pick.ability.parent_target.empty();
}

static bool controlled_as_defined(const TargetPick &pick, Entity candidate) {
    const std::vector<Zone::Ownership> players =
        resolve_players(pick.ability.def->targets_defined_controller, ResolveCtx::of(pick.ability));
    return std::find(players.begin(), players.end(), source_controller(candidate)) != players.end();
}

// The first required instruction chained to `ab` that reads the target before it, if one does.
static const Ability *instruction_answering_to_this_target(const Ability &ab) {
    for (const Ability &sub : ab.subabilities) {
        if (sub.def->from_delayed_execute) continue;
        if (!sub.def->targets_defined_controller.is(ObjectSpec::DEFAULT) && sub.target_min > 0) return &sub;
        if (const Ability *deeper = instruction_answering_to_this_target(sub)) return deeper;
    }
    return nullptr;
}

static bool an_instruction_answers_to_this_target(const TargetPick &pick) {
    return instruction_answering_to_this_target(pick.ability) != nullptr;
}

static bool leaves_that_instruction_a_target(const TargetPick &pick, Entity candidate) {
    Ability chained = *instruction_answering_to_this_target(pick.ability);
    chained.source = pick.ability.source;
    chained.controller = pick.chooser;
    chained.parent_target = ObjectRef::of(candidate);
    return !target_candidates(chained, pick.orderer, pick.chooser).empty();
}

static bool a_target_is_changing(const TargetPick &pick) { return pick.gate && pick.gate->changing_from != 0; }

static bool is_another_target(const TargetPick &pick, Entity candidate) {
    return candidate != pick.gate->changing_from && candidate != pick.gate->changing_object;
}

static bool cost_is_gated(const TargetPick &pick) { return pick.gate && pick.gate->cost_payable; }

static bool leaves_cost_payable(const TargetPick &pick, Entity candidate) { return pick.gate->cost_payable(candidate); }

void apply_target_gate(std::vector<Entity> &pool, const TargetPick &pick) {
    for (const TargetGateRow &row : kTargetGateRows) {
        if (!row.applies(pick)) continue;
        std::vector<Entity> kept;
        for (Entity e : pool)
            if (row.allows(pick, e)) kept.push_back(e);
        if (kept.size() == pool.size()) continue;
        coverage_row(row.name);
        if (row.required || !kept.empty()) pool = std::move(kept);
    }
}
