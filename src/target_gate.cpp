#include "target_gate.h"

#include <utility>

#include "coverage.h"

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
static bool cost_is_gated(const TargetPick &pick);
static bool leaves_cost_payable(const TargetPick &pick, Entity candidate);

static const TargetGateRow kTargetGateRows[] = {
    // CR 115.3: the same target can't be chosen twice for one instance of the word "target".
    {"target_gate:distinct", true, holds_a_target, not_chosen_yet},
    // CR 601.2f: a cost reduced only for some targets (Mystical Dispute).
    {"target_gate:cost_payable", false, cost_is_gated, leaves_cost_payable},
};

static bool holds_a_target(const TargetPick &pick) { return pick.picked > 0; }

static bool not_chosen_yet(const TargetPick &pick, Entity candidate) {
    for (int i = 0; i < pick.picked; i++)
        if (pick.ability.targets[static_cast<size_t>(i)].get() == candidate) return false;
    return true;
}

static bool cost_is_gated(const TargetPick &pick) { return pick.gate != nullptr; }

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
