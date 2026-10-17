#ifndef KNOWLEDGE_BOUNDS_H
#define KNOWLEDGE_BOUNDS_H

#include <cstdint>
#include <string>
#include <vector>

#include "network.h"

// ── Bound propagation over one player's network (docs/knowledge_store.md §4.3, step 1) ─────────
//
// For every node and every card of the universe, the least and most copies of that card the node
// can hold in any world. Each bound is sound (never excludes a possible world), not necessarily
// exact: the per-card intervals cannot express joint constraints ("the top two are one Force of
// Will and one Daze") or disjunctions, which the exact search (T10) settles.
//
// Constraints propagated: a node's size (its cards sum to it), pins, roots, every junction (the in
// side sums to the out side, per card), and the current cut (the roots sum to the live nodes).
// Each constraint narrows each member's interval from the others' (interval arithmetic); a change
// re-queues the constraints the node is in, to a fixpoint. Constraints only accumulate within a
// game, so the store keeps its bounds between calls and only processes what is new: nodes,
// junctions and pins added since the last call, and the cut.
//
// The universe is the card list the store tracks, plus one OTHER class for every card not in it
// (an unconstrained root's unseen cards). A card entering the universe resets the store.

namespace knowledge {

struct Universe {
        std::vector<int> cards;  // ascending card ids; class index cards.size() = OTHER
        size_t classes() const { return cards.size() + 1; }
        // The class of `card`: its index, else OTHER.
        size_t index_of(int card) const;
        bool contains(int card) const;
        // Adds `card`; true when it was not there.
        bool add(int card);
};

class BoundStore {
    public:
        // `use_roots` false: roots are free (only their sizes count); the observation-only store
        // that measures what the viewer has seen.
        explicit BoundStore(bool use_roots = true) : use_roots_(use_roots) {}

        Universe universe;
        void reset();
        // Brings the bounds up to date with `net` and the cut over `live`. Returns the steps used
        // (constraint evaluations); stops early, still sound, past `budget`.
        uint64_t propagate(const Network &net, const std::vector<NodeId> &live, uint64_t budget);

        // Bounds of node `n` for class `k`; [0, size] for a node not propagated yet.
        int lo(const Network &net, NodeId n, size_t k) const;
        int hi(const Network &net, NodeId n, size_t k) const;
        bool knows(NodeId n) const { return n < nodes_done_; }
        // The first contradiction found (an empty interval), empty if none: the model is wrong.
        const std::string &contradiction() const { return contradiction_; }
        bool budget_hit() const { return budget_hit_; }

        void serialize(std::string &out) const;
        bool deserialize(const std::string &in, size_t *at);

    private:
        void init_node(const Network &net, NodeId n);
        void apply_pin(const Network &net, NodeId n);
        bool tighten_size(const Network &net, NodeId n);
        bool narrow(NodeId n, size_t k, int lo, int hi);
        void node_changed(const Network &net, NodeId n);
        void eval_relation(const Network &net, const std::vector<NodeId> &in, const std::vector<NodeId> &out);
        void enqueue(uint32_t j);
        void fail(const std::string &what);

        bool use_roots_;
        size_t classes_ = 0;
        std::vector<uint8_t> lo_, hi_;  // node × class
        size_t nodes_done_ = 0;
        size_t junctions_done_ = 0;
        size_t pins_done_ = 0;
        std::string contradiction_;
        bool budget_hit_ = false;
        // Worklist (transient within propagate): junction ids, CUT for the cut.
        std::vector<uint32_t> queue_;
        std::vector<uint8_t> queued_;
        bool cut_queued_ = false;
        std::vector<NodeId> changed_;
        std::vector<uint8_t> is_changed_;
        const std::vector<NodeId> *live_ = nullptr;
        std::vector<NodeId> roots_;
};

}  // namespace knowledge

#endif /* KNOWLEDGE_BOUNDS_H */
