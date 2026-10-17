#ifndef KNOWLEDGE_SOLVER_H
#define KNOWLEDGE_SOLVER_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "bounds.h"
#include "network.h"

// ── The exact solver (docs/knowledge_store.md §4.3) ────────────────────────────────────────────
//
// For groups of live nodes of one player's network (the k_pool rows: the hand, a library run, a
// block of runs, ...), the least and most copies of each card the group holds over every world:
// every assignment of a card to every unit that satisfies every pin, root and junction and every
// node's size. Sound always; exact when it finishes within its step budget.
//
// A world is an integer point x[node][class] ≥ 0 with Σ_class x = size per node, Σ in + fixed =
// Σ out + fixed per junction and class, pins and EXACT roots fixing a node, LOWER roots bounding it
// from below. The solver:
//   1. compacts a copy of the network keeping only the groups' nodes (compact.h: exact for their
//      projections), then builds that system over it, seeding every node's interval from a
//      BoundStore (sound bounds; only for a node compaction left unchanged) and its own unary
//      constraints; a node whose classes are all determined is a constant. A LOWER root R is its
//      lower bounds (constants) plus a free node of the remaining size, so no variable keeps a
//      constraint of its own;
//   2. splits the variables into components (variables joined by a junction); the worlds are the
//      product of the components' worlds, so a group's least and most copies are sums over the
//      components it meets;
//   3. per component, searches interval states. A state is a box of per-(node, class) intervals
//      that bound propagation (each equation narrowed by the others' intervals, to a fixpoint)
//      keeps sound, so a box it empties holds no world. A search branches by halving one interval
//      of a variable that blocks the closed form below, so the two children partition the state's
//      worlds. A state is a leaf when every junction with a free variable is a transportation
//      problem: its free variables, all untouched by branching and in no other such junction, are
//      on one side of it, or one side's free variables are each in no group and no other junction
//      (a slack: only their sum matters), every other member being fixed. A transportation
//      problem's projections are exact closed forms: with T the fixed cards the free side holds
//      besides the slack and N its size, a subset of size s holds between max(0, s + T_c − N) and
//      min(s, T_c) copies of c (with a slack, T_c becomes max(0, T_c) and the upper end grows by
//      the slack's spare units). A leaf's interval per group is the sum of these over its
//      junctions plus its fixed members, and every value at its ends is reached by a world of the
//      leaf (the leaf's worlds are its junctions' worlds combined freely);
//   4. a group's least (most) copies of a card is the least (most) over all leaves. The root's
//      propagated interval bounds it, tightened by the card's flow relaxation: per card, every
//      junction conserves copies and every variable leaves one junction and enters one (the outside
//      world counting as one more), so the card's cells form a network flow whose optimum (a
//      min-cost flow, the node sizes coupling the cards dropped) bounds every world's. After a
//      first search for any leaf (none: no world), each answer not yet reached by a leaf is
//      searched for with the root narrowed to its proven bound, a subtree being cut when the
//      card's flow bound below it falls short, and the half of the card's own interval toward the
//      answer tried first. The first leaf found that reaches it settles it (every leaf updates
//      every answer); when the whole narrowed search finds none, no world reaches it, so the proven
//      bound moves one step and the search repeats. When every answer is settled the results are
//      exact. When the budget runs out first, a settled answer is exact and the rest report their
//      proven bounds: sound, and `exact` is false.
//
// Steps: one per constraint evaluation, search node, augmenting path and Bellman-Ford pass. Results
// are memoised per component (its full description, including its seed intervals and the budget it
// was given), so an unchanged component costs no search on the next frame; a memoised answer
// reports the steps its search took, so the step counts (and every output) do not depend on the
// memo.

namespace knowledge {

struct GroupBounds {
        std::vector<uint8_t> lo, hi;  // per class of the universe (OTHER last)
};

struct Solution {
        std::vector<GroupBounds> groups;
        bool exact = false;
        bool contradiction = false;  // no world: the model is wrong
        uint64_t steps = 0;
};

class Solver {
    public:
        // The bounds of each of `groups` (live nodes of `net`) over the worlds of `net` (roots
        // constraining), seeded with `store`'s bounds, within `budget` steps.
        Solution solve(const Network &net, const BoundStore &store, const std::vector<std::vector<NodeId>> &groups,
            uint64_t budget) const;

    private:
        struct Memo {
                bool finished = false;
                bool infeasible = false;
                uint64_t steps = 0;
                uint64_t budget = 0;          // the budget it ran under, when not finished
                std::vector<int16_t> lo, hi;  // per local group × class
                std::vector<uint8_t> settled;
        };
        mutable std::map<std::string, Memo> memo_;
};

}  // namespace knowledge

#endif /* KNOWLEDGE_SOLVER_H */
