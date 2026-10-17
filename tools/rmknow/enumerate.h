#ifndef RMKNOW_ENUMERATE_H
#define RMKNOW_ENUMERATE_H

#include <cstdint>
#include <vector>

#include "../../src/knowledge/bounds.h"
#include "../../src/knowledge/network.h"

// ── A naive enumerator of a constraint system's worlds (docs/knowledge_store.md §7, exactness) ──
//
// Written independently of the solver (src/knowledge/solver.*): it shares no code with it and does
// no interval reasoning. Its input is a plain description of one player's constraints over K card
// classes: nodes (a size, and a class when pinned), roots (exact contents, lower bounds or free) and
// junctions (the in nodes' cards plus fixed cards equal the out nodes' cards plus fixed cards, per
// class; every node is made by at most one junction and consumed by at most one). A world gives
// every node a count per class summing to its size and satisfying every pin, root and junction.
//
// The worlds are swept forward through the junctions in dependency order. A state is the value of
// every node on the frontier (made and not yet consumed) that something still reads: a later
// junction or a group. The sweep starts from every combination of the roots' allowed values; each
// junction then takes every state, adds its in nodes' values and fixed cards, subtracts its out
// side's fixed cards (no class may go negative) and divides the total among its out nodes in every
// way their sizes and pins allow. Two frontier nodes of the same size, consumer and groups are
// interchangeable (swapping their values maps every continuation to one with the same group
// projections), so a state lists their values sorted and equal states are kept once. At the end
// each state gives every group its counts; the result is their least and most per class.

namespace rmknow_test {

struct EnumSpec {
        struct Node {
                int size = 0;
                int pin = -1;  // the class of a pinned (one-unit) node
        };
        struct Root {
                int node = 0;
                int kind = 0;  // 0 exact, 1 lower bounds, 2 free
                std::vector<int> counts;
        };
        struct Junction {
                std::vector<int> in, out;
                std::vector<int> in_fixed, out_fixed;  // per class
        };
        int K = 0;
        std::vector<Node> nodes;
        std::vector<Root> roots;
        std::vector<Junction> junctions;
};

struct EnumResult {
        bool complete = false;                 // false: more than `cap` states at some point (or a cycle)
        uint64_t worlds = 0;                   // final states: 0 when no world exists
        std::vector<std::vector<int>> lo, hi;  // per group × class
};

// The belief layer's likelihood (src/knowledge/belief.h) of `spec`, independently: the same forward
// sweep, every state carrying the probability of reaching it when each junction divides its in side
// (its in nodes' cards plus its fixed cards) uniformly at random among its out nodes and fixed cards,
// Π_c y_c! / Π_o Π_c x_oc! × Π_o s_o! / n! per division (every out node enumerated, sinks included;
// no closed form, no interchangeable-node merging). Every root must be exact. Returns the
// probability that every pin and fixed card holds; `complete` false past `cap` states (or a root
// that is not exact, or a cycle).
double weighted_world_probability(const EnumSpec &spec, uint64_t cap, bool *complete);
// The description of `net` over `uni`'s classes (cards, then OTHER).
EnumSpec spec_of(const knowledge::Network &net, const knowledge::Universe &uni);
// The projections of `spec`'s worlds onto `groups` (at most `cap` states at a time).
EnumResult enumerate_worlds(const EnumSpec &spec, const std::vector<std::vector<int>> &groups, uint64_t cap);

}  // namespace rmknow_test

#endif /* RMKNOW_ENUMERATE_H */
