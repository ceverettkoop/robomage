#ifndef KNOWLEDGE_NETWORK_H
#define KNOWLEDGE_NETWORK_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// ── The constraint record: one player's cards as a flow network over time ─────────────────────
// (docs/knowledge_store.md §3.2)
//
// A node is a set of that player's physical cards (units) the viewer cannot tell apart, as it was
// at one moment: an immutable multiset of card identities whose size is known. Nodes never change;
// an event that moves units makes new nodes and records how they relate:
//   - a junction says the units of its `in` nodes are exactly the units of its `out` nodes (a
//     split: one in, several out; a merge: several in, one out);
//   - a pin says a one-unit node is a known card;
//   - a root is a node no junction produced: a registered list (its exact contents), a list with
//     only lower bounds (a closed decklist after cards were seen), or unconstrained.
// A world assigns a card to every unit consistently with every junction, pin and root; knowledge
// is what holds in every world (state_format §1). Every node is produced by at most one junction
// and consumed by at most one; a node no junction consumed is live: it is somewhere now (Loc).
// Every unit of the player is in exactly one live node, so the live nodes sum to the roots (the
// current cut, which the bound propagation uses as one more junction).
//
// Pins may land on a consumed node (a reveal that names the card a past draw brought in): they
// constrain the past, and through the junctions the present.

namespace knowledge {

using NodeId = uint32_t;
constexpr NodeId NO_NODE = 0xFFFFFFFFu;
constexpr uint32_t NO_JUNCTION = 0xFFFFFFFFu;

// Where a live node is. GONE = consumed by a junction (history).
enum class Loc : uint8_t {
    GONE,
    LIBRARY,    // a run of consecutive library positions (PlayerModel::library)
    HAND,       // part of the hand (PlayerModel::hand)
    SIDEBOARD,  // part of the sideboard
    SET_ASIDE,  // registered cards not yet in a zone at game start (test-harness presets, a companion)
    FACE_DOWN,  // one face-down card in a public zone, keyed by its handle
    PUBLIC,     // one card in a public zone (battlefield, stack, graveyard, face-up exile)
};

struct Node {
        uint16_t size = 0;
        int32_t pin = -1;  // the known card of a one-unit node, else -1
        uint32_t in = NO_JUNCTION;
        uint32_t out = NO_JUNCTION;
        int32_t root = -1;  // index into Network::roots
        Loc loc = Loc::GONE;
        int16_t drawn_turn = -1;  // in a hand: the turn its units were drawn (non-opening draws), else -1
        uint32_t handle = 0;      // the viewer's handle for its one unit while the viewer holds one
        uint32_t draw_seq = 0;    // a one-unit node a draw brought in: that draw's seq
};

struct Junction {
        std::vector<NodeId> in;
        std::vector<NodeId> out;
};

enum class RootKind : uint8_t {
    EXACT,  // contents = counts
    LOWER,  // contents ≥ counts (per card), otherwise free
    FREE,   // any cards
};

struct Root {
        NodeId node = NO_NODE;
        RootKind kind = RootKind::FREE;
        std::vector<std::pair<int, int>> counts;  // (card, copies), ascending card
};

class Network {
    public:
        // A root of `size` units at `loc` (EXACT / LOWER take `counts`).
        NodeId add_root(uint16_t size, Loc loc, RootKind kind, const std::vector<std::pair<int, int>> &counts);
        // Splits live `parent` into nodes of `sizes` (each > 0, summing to its size), all at the
        // parent's location; returns them in order. A parent of a single part is returned as is.
        std::vector<NodeId> split(NodeId parent, const std::vector<uint16_t> &sizes);
        // Merges live `parts` (at least one) into one node at `loc`; a single part is returned as is.
        NodeId merge(const std::vector<NodeId> &parts, Loc loc);
        // Pins one-unit node `n` to `card`. False when it is pinned to another card already.
        bool pin(NodeId n, int card);

        Node &node(NodeId n) { return nodes_[n]; }
        const Node &node(NodeId n) const { return nodes_[n]; }
        size_t node_count() const { return nodes_.size(); }
        const Junction &junction(uint32_t j) const { return junctions_[j]; }
        size_t junction_count() const { return junctions_.size(); }
        const std::vector<Root> &roots() const { return roots_; }
        // Nodes pinned after they were made, in pin order (BoundStore applies them).
        const std::vector<NodeId> &pin_log() const { return pin_log_; }

        void serialize(std::string &out) const;
        bool deserialize(const std::string &in, size_t *at);

    private:
        NodeId add_node(uint16_t size, Loc loc);
        std::vector<Node> nodes_;
        std::vector<Junction> junctions_;
        std::vector<Root> roots_;
        std::vector<NodeId> pin_log_;
};

}  // namespace knowledge

#endif /* KNOWLEDGE_NETWORK_H */
