#ifndef KNOWLEDGE_COMPACT_H
#define KNOWLEDGE_COMPACT_H

#include <cstdint>
#include <vector>

#include "network.h"

// ── Network compaction (docs/knowledge_store.md §4.3, §9.2) ────────────────────────────────────
//
// compact_network rebuilds a network without the history that no longer constrains the kept nodes
// (every live node, and the nodes a later event may still name: this turn's draws). It is exactly
// knowledge-preserving: the worlds of the new network, projected onto the kept nodes (and the
// roots it keeps), are the worlds of the old one projected the same way. It applies these
// eliminations to a fixpoint, each exact on its own:
//   - a pinned node that is not kept (and is no EXACT or LOWER root) becomes its card: a fixed card
//     on the out side of the junction that made it and on the in side of the one that consumed it;
//   - a node that is neither kept, pinned nor a root, between junctions P (made it) and C (consumed
//     it), is removed and P and C become one junction when it is the only member of P's out side
//     or of C's in side, fixed cards included (it is then the sum of the other side, so its own
//     non-negativity is implied); when P and C are the same junction it is on both sides and drops;
//   - such nodes with the same P and the same C are one node (only their sum appears anywhere, and
//     any sum splits into parts of the given sizes);
//   - a FREE root that is the only member of its consumer's in side makes that junction vacuous:
//     the junction drops and the nodes it made become FREE roots;
//   - a card fixed on both sides of one junction cancels; a junction left with no node and its
//     fixed cards balanced drops.
// No rule touches a kept node, an EXACT or LOWER root, or a pinned root other than a FREE one, so
// the roots' constraints and the observation store (roots free) are preserved alike.

namespace knowledge {

// The compacted copy of `net`. `keep` has one flag per node of `net`; kept nodes keep every field
// (size, pin, location, handle, drawn-this-turn turn, draw seq). `remap` receives, per old node, its
// id in the new network or NO_NODE (an eliminated node). The new network's pin log is empty: every
// pin is on its node.
Network compact_network(const Network &net, const std::vector<uint8_t> &keep, std::vector<NodeId> *remap);

}  // namespace knowledge

#endif /* KNOWLEDGE_COMPACT_H */
