#ifndef RMWORK_TABLE_COMBAT_H
#define RMWORK_TABLE_COMBAT_H

#include <cstdint>
#include <string>
#include <vector>

#include "../../src/agents/decision_view.h"

// ── A combat declaration planned on the Table, then sent to the engine ──────────────────────────
//
// The engine declares attackers (CR 508.1) and blockers (509.1) one creature at a time: a
// SELECT_ATTACKER row declares its creature at once (an ATTACK_TARGET decision follows when the
// defending player has a planeswalker), a SELECT_BLOCKER row is followed by a BLOCK_TARGET decision
// naming the attacker it blocks, and no row takes a declared creature back. So on the Table the human
// plans the declaration (docs/workbench_plan.md §4.2, item 7): clicks add and remove planned attackers,
// each with what it attacks, or planned blocks (a blocker and the attacker it blocks); nothing is sent
// until the prompt line's confirm button, which sends the plan one reply per decision (next_plan_step:
// each planned creature's row, its target's row, the confirm row last). A creature already declared
// (answered in Choices) is the frame's and stays declared. The engine takes a menu left with one row
// itself, unasked, so the plan's last confirm (or a lone attacker's row) may never be asked.
//
// A plan belongs to one declaration (PlanKey: the game, the turn, attackers or blockers): the Table
// starts an empty one when it shows another. Planned blocks are checked against the frame's evasion only
// (TableDecision::may_block); when the engine's BLOCK_TARGET rows (the blocker's legal attackers) do not
// hold the planned attacker, or its ATTACK_TARGET rows the planned target, sending stops at that
// decision, the rest of the plan kept, and the human chooses there.

namespace rmwork {

struct PlannedAttack {
        uint64_t attacker = 0;
        uint64_t target = 0;   // a planeswalker's handle, 0 for the defending player
        int target_seat = -1;  // the defending player's seat (0 A, 1 B) when target is 0
};

struct PlannedBlock {
        uint64_t blocker = 0, attacker = 0;
};

struct PlanKey {
        int game = -1;
        long turn = -1;
        bool blockers = false;
        bool operator==(const PlanKey &o) const { return game == o.game && turn == o.turn && blockers == o.blockers; }
        bool operator!=(const PlanKey &o) const { return !(*this == o); }
};

// Why sending a plan stopped before its end.
enum class PlanStop {
    NONE,
    NO_TARGET,     // ATTACK_TARGET: no row names the planned attacker's target
    CANNOT_BLOCK,  // BLOCK_TARGET: no row names the planned blocker's attacker
    UNPLANNED,     // the engine asks about a creature the plan does not hold
    ENDED          // the declaration ended with planned creatures never sent
};

struct PlanStep {
        int row = -1;           // the reply to send, -1 none
        uint64_t creature = 0;  // the planned creature the row (or the stop) is about
        bool finished = false;  // the plan is sent (the row, if any, is its last reply)
        PlanStop stop = PlanStop::NONE;
};

class CombatPlan {
    public:
        const PlanKey &key() const { return key_; }
        // Keeps the plan when `key` is its own, else starts an empty one for it.
        void follow(const PlanKey &key);
        // Nothing planned or armed, not sending (the key kept).
        void clear();
        // Moves at every change, for a view that caches what it drew.
        unsigned generation() const { return generation_; }
        bool empty() const { return attacks_.empty() && blocks_.empty(); }

        const std::vector<PlannedAttack> &attacks() const { return attacks_; }
        const std::vector<PlannedBlock> &blocks() const { return blocks_; }
        const PlannedAttack *attack_of(uint64_t attacker) const;
        const PlannedBlock *block_of(uint64_t blocker) const;
        // Adds the planned attack (block), or replaces the creature's.
        void plan_attack(const PlannedAttack &a);
        void plan_block(const PlannedBlock &b);
        // Drops the creature's planned attack or block (and disarms it).
        void drop(uint64_t creature);

        // The creature a second click completes: a blocker waiting for its attacker, a planned attacker
        // for a planeswalker to attack instead of the player; 0 none.
        uint64_t armed() const { return armed_; }
        void arm(uint64_t creature);

        // Sending: started by the confirm button, stepped at each of the human's decisions.
        bool sending() const { return sending_; }
        void start_sending();
        void stop_sending();
        // The decision the last reply answered (-1 none): a decision is answered once.
        long answered() const { return answered_; }
        // Records the reply of `step` at decision `index` (its creature counted sent).
        void record(const PlanStep &step, long index);
        bool sent(uint64_t creature) const;

    private:
        void changed() { ++generation_; }

        PlanKey key_;
        std::vector<PlannedAttack> attacks_;
        std::vector<PlannedBlock> blocks_;
        std::vector<uint64_t> sent_;
        uint64_t armed_ = 0;
        bool sending_ = false;
        long answered_ = -1;
        unsigned generation_ = 0;
};

// The reply that carries `plan` one step further at `v`, the planning seat's decision: at the
// declaration's menu the row of the first planned creature still offered, else the confirm row (the
// plan finished); at ATTACK_TARGET / BLOCK_TARGET the row naming the planned target or attacker of the
// creature being declared (the pending source); at any other decision the plan is finished (the
// engine took the last rows itself), stopped ENDED if a planned creature was never sent.
PlanStep next_plan_step(const CombatPlan &plan, const agents::DecisionView &v);

}  // namespace rmwork

#endif /* RMWORK_TABLE_COMBAT_H */
