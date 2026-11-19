#include "table_combat.h"

#include <algorithm>

static int seat_of(uint64_t seat);
static uint64_t pending_source(const agents::DecisionView &v);
static bool in_combat(const agents::DecisionView &v, uint64_t attacker, uint64_t blocker);
static rmwork::PlanStep select_step(const rmwork::CombatPlan &plan, const agents::DecisionView &v, int confirm);
static rmwork::PlanStep target_step(const rmwork::CombatPlan &plan, const agents::DecisionView &v);
static rmwork::PlanStep ended_step(const rmwork::CombatPlan &plan, const agents::DecisionView &v);

// The `seat` enum (1 A, 2 B) as 0 A, 1 B; -1 for none.
static int seat_of(uint64_t seat) {
    return seat == 1 ? 0 : seat == 2 ? 1 : -1;
}

static uint64_t pending_source(const agents::DecisionView &v) {
    const agents::Table p = v.table("pending");
    return p.rows() ? p.u(0, p.col("source")) : 0;
}

// The frame's combat table has `attacker` attacking (blocked by `blocker`, when not 0).
static bool in_combat(const agents::DecisionView &v, uint64_t attacker, uint64_t blocker) {
    const agents::Table c = v.table("combat");
    for (size_t r = 0; r < c.rows(); ++r)
        if (c.u(r, c.col("attacker")) == attacker && (blocker == 0 || c.u(r, c.col("blocker")) == blocker)) return true;
    return false;
}

// At the declaration's menu: the first planned creature still offered, else the confirm row.
static rmwork::PlanStep select_step(const rmwork::CombatPlan &plan, const agents::DecisionView &v, int confirm) {
    rmwork::PlanStep out;
    const agents::Table menu = v.table("menu");
    const int c_src = menu.col("source");
    const char *select = plan.key().blockers ? "SELECT_BLOCKER" : "SELECT_ATTACKER";
    std::vector<uint64_t> creatures;
    for (const rmwork::PlannedAttack &a : plan.attacks()) creatures.push_back(a.attacker);
    for (const rmwork::PlannedBlock &b : plan.blocks()) creatures.push_back(b.blocker);
    for (uint64_t c : creatures)
        for (size_t i = 0; i < v.n(); ++i)
            if (v.category(i) == select && menu.u(i, c_src) == c) {
                out.row = static_cast<int>(i);
                out.creature = c;
                return out;
            }
    out.row = confirm;
    out.finished = true;
    return out;
}

// ATTACK_TARGET / BLOCK_TARGET: the row naming the planned target of the creature being declared.
static rmwork::PlanStep target_step(const rmwork::CombatPlan &plan, const agents::DecisionView &v) {
    rmwork::PlanStep out;
    out.creature = pending_source(v);
    const agents::Table menu = v.table("menu");
    const int c_src = menu.col("source"), c_sp = menu.col("source_player");
    const rmwork::PlannedAttack *a = plan.key().blockers ? nullptr : plan.attack_of(out.creature);
    const rmwork::PlannedBlock *b = plan.key().blockers ? plan.block_of(out.creature) : nullptr;
    if (!a && !b) {
        out.stop = rmwork::PlanStop::UNPLANNED;
        return out;
    }
    for (size_t i = 0; i < v.n(); ++i) {
        const uint64_t src = menu.u(i, c_src);
        const bool names = b           ? src == b->attacker
                           : a->target ? src == a->target
                                       : src == 0 && seat_of(menu.u(i, c_sp)) == a->target_seat;
        if (names) {
            out.row = static_cast<int>(i);
            return out;
        }
    }
    out.stop = b ? rmwork::PlanStop::CANNOT_BLOCK : rmwork::PlanStop::NO_TARGET;
    return out;
}

// Past the declaration: finished; ENDED when a planned creature was neither sent nor declared.
static rmwork::PlanStep ended_step(const rmwork::CombatPlan &plan, const agents::DecisionView &v) {
    rmwork::PlanStep out;
    out.finished = true;
    for (const rmwork::PlannedAttack &a : plan.attacks())
        if (!plan.sent(a.attacker) && !in_combat(v, a.attacker, 0)) {
            out.stop = rmwork::PlanStop::ENDED;
            out.creature = a.attacker;
        }
    for (const rmwork::PlannedBlock &b : plan.blocks())
        if (!plan.sent(b.blocker) && !in_combat(v, b.attacker, b.blocker)) {
            out.stop = rmwork::PlanStop::ENDED;
            out.creature = b.blocker;
        }
    return out;
}

namespace rmwork {

void CombatPlan::follow(const PlanKey &key) {
    if (key == key_) return;
    key_ = key;
    clear();
}

void CombatPlan::clear() {
    attacks_.clear();
    blocks_.clear();
    sent_.clear();
    armed_ = 0;
    sending_ = false;
    answered_ = -1;
    changed();
}

const PlannedAttack *CombatPlan::attack_of(uint64_t attacker) const {
    for (const PlannedAttack &a : attacks_)
        if (a.attacker == attacker) return &a;
    return nullptr;
}

const PlannedBlock *CombatPlan::block_of(uint64_t blocker) const {
    for (const PlannedBlock &b : blocks_)
        if (b.blocker == blocker) return &b;
    return nullptr;
}

void CombatPlan::plan_attack(const PlannedAttack &a) {
    for (PlannedAttack &p : attacks_)
        if (p.attacker == a.attacker) {
            p = a;
            changed();
            return;
        }
    attacks_.push_back(a);
    changed();
}

void CombatPlan::plan_block(const PlannedBlock &b) {
    for (PlannedBlock &p : blocks_)
        if (p.blocker == b.blocker) {
            p = b;
            changed();
            return;
        }
    blocks_.push_back(b);
    changed();
}

void CombatPlan::drop(uint64_t creature) {
    attacks_.erase(std::remove_if(attacks_.begin(), attacks_.end(),
                       [creature](const PlannedAttack &a) { return a.attacker == creature; }),
        attacks_.end());
    blocks_.erase(std::remove_if(blocks_.begin(), blocks_.end(),
                      [creature](const PlannedBlock &b) { return b.blocker == creature; }),
        blocks_.end());
    if (armed_ == creature) armed_ = 0;
    changed();
}

void CombatPlan::arm(uint64_t creature) {
    if (armed_ == creature) return;
    armed_ = creature;
    changed();
}

void CombatPlan::start_sending() {
    sending_ = true;
    armed_ = 0;
    changed();
}

void CombatPlan::stop_sending() {
    sending_ = false;
    changed();
}

void CombatPlan::record(const PlanStep &step, long index) {
    answered_ = index;
    if (step.creature && !sent(step.creature)) sent_.push_back(step.creature);
}

bool CombatPlan::sent(uint64_t creature) const {
    return std::find(sent_.begin(), sent_.end(), creature) != sent_.end();
}

PlanStep next_plan_step(const CombatPlan &plan, const agents::DecisionView &v) {
    const bool blockers = plan.key().blockers;
    const std::string select = blockers ? "SELECT_BLOCKER" : "SELECT_ATTACKER";
    const std::string confirm = blockers ? "CONFIRM_BLOCKERS" : "CONFIRM_ATTACKERS";
    const std::string target = blockers ? "BLOCK_TARGET" : "ATTACK_TARGET";
    int confirm_row = -1;
    bool selecting = false;
    for (size_t i = 0; i < v.n(); ++i) {
        const std::string &c = v.category(i);
        if (c == select || c == confirm) selecting = true;
        if (c == confirm) confirm_row = static_cast<int>(i);
    }
    if (selecting) return select_step(plan, v, confirm_row);
    if (v.n() && v.category(0) == target) return target_step(plan, v);
    return ended_step(plan, v);
}

}  // namespace rmwork
