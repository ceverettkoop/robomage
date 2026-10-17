#ifndef KNOWLEDGE_PLAYER_MODEL_H
#define KNOWLEDGE_PLAYER_MODEL_H

#include <cstdint>
#include <string>
#include <vector>

#include "bounds.h"
#include "frame_view.h"
#include "network.h"

// ── One player's cards as the viewer knows them: the network plus where its live nodes are ────
//
// Every live node of the player's network sits in exactly one of the lists below:
//   - library: the library as a sequence of runs, top first. A run is one node whose units fill
//     that many consecutive positions in an order the viewer does not know (a one-unit run is an
//     exact position);
//   - hand, sideboard, set_aside: the zone as a set of nodes (sub-pools the viewer can still tell
//     apart: a known card, a card drawn this turn, the rest);
//   - face_down: one-unit nodes of face-down public objects, each keyed by its handle;
//   - publics: one-unit nodes of the player's cards in public zones, each with its handle.
// The library operations split and merge runs through the network, so every position fact is a
// junction: removing a unit from a run splits it off the run, inserting into the middle of a run
// cuts the run at that position, a range of positions the viewer cannot tell apart merges into one
// run.

namespace knowledge {

// Whether a live node at `loc` can be the card an event moves out of `zone` (a set-aside unit
// leaves from zone none).
bool loc_matches(Loc loc, Zn zone);

class PlayerModel {
    public:
        Network net;
        BoundStore full{true};   // the deductions (roots constrain)
        BoundStore seen{false};  // what was observed (roots free): copies seen
        std::vector<NodeId> library;
        std::vector<NodeId> hand;
        std::vector<NodeId> sideboard;
        std::vector<NodeId> set_aside;
        std::vector<NodeId> face_down;
        std::vector<NodeId> publics;
        // The turn in which the hand's drawn-this-turn units were merged with other cards (so which
        // hand units were drawn this turn is no longer known); -1 none.
        int dtt_unknown_turn = -1;

        size_t library_size() const;
        // The run holding top index `index` and the index's offset in it.
        size_t locate(size_t index, size_t *offset) const;
        // The library top index where run `seg` starts.
        size_t run_start(size_t seg) const;
        // Removes the unit at top index `index` and returns it as a one-unit node (split off its run).
        NodeId take_library(size_t index);
        // Removes run `seg` from the library (its node stays as it is).
        void erase_library_run(size_t seg);
        // Inserts node `n` so that its first unit lands at top index `index`.
        void insert_library(size_t index, NodeId n);
        // Makes the unit at top index `index` a run of its own; returns it.
        NodeId isolate_library(size_t index);
        // Merges the positions [a, b) into one run; returns it (NO_NODE for an empty range).
        NodeId merge_library_range(size_t a, size_t b);
        // Merges the whole library into one run (a shuffle); returns it (NO_NODE when empty).
        NodeId merge_library();
        // Removes one unit from the node at `at` in `zone` (a set zone) and returns it; the rest of
        // the node stays in `zone`.
        NodeId split_unit(std::vector<NodeId> &zone, size_t at);
        // Every live node: the current cut.
        std::vector<NodeId> live() const;
        // The list a live node of location `loc` is in (nullptr for LIBRARY and GONE).
        std::vector<NodeId> *zone_list(Loc loc);
        const std::vector<NodeId> *zone_list(Loc loc) const;

        void serialize(std::string &out) const;
        bool deserialize(const std::string &in, size_t *at);

    private:
        size_t cut_before(size_t index);
        void rejoin(size_t seg);
};

}  // namespace knowledge

#endif /* KNOWLEDGE_PLAYER_MODEL_H */
