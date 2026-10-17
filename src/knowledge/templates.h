#ifndef KNOWLEDGE_TEMPLATES_H
#define KNOWLEDGE_TEMPLATES_H

#include <string>
#include <utility>
#include <vector>

#include "bounds.h"
#include "frame_view.h"
#include "network.h"

// ── Archetype inference: reference lists as alternative roots (docs/knowledge_store.md §10) ───
//
// The METAGAME frame names reference lists, each with an archetype label. For a player whose
// registered list the viewer does not see (the opponent under --closed-decklist), a reference list L
// is consistent when some world of the player's network exists with L as the player's root: game 1
// the main deck exactly L's main deck and the sideboard exactly L's sideboard, games 2-3 the 75
// exactly L's 75 (split between deck and sideboard in any way). That is the solver's feasibility
// question on the hypothesis network below.
//
// The roots it replaces are the list roots (Root::part MAIN / SIDE / ALL): under a closed decklist
// lower-bound roots of the library and sideboard sizes the game started with (game 1 one each for
// the main deck and the sideboard, games 2-3 one for both). A list root of s units standing for
// contents T (|T| ≥ s, else no world) becomes the output of a junction from a new EXACT root T,
// beside a set-aside node of |T| - s units: L's cards outside the library and sideboard at the
// start (a companion, a test-harness preset). The player's other units, the cards created from zone
// none (FREE roots, and the pinned ones compaction turned into fixed cards: the network's net fixed
// cards, its in-side fixed cards minus its out-side ones), come out of those set-aside nodes: one
// junction from them to the FREE roots, the net fixed cards and a remainder of the set-aside units
// not seen yet (≥ 0, else no world). Nothing else changes, so every pin, junction and size of the
// network constrains the hypothesis as it constrains the player's own root.

namespace knowledge {

// `player`'s registered counts in `lists`, (card, copies) ascending with copies > 0.
void registered_counts(
    const Decklists &lists, int player, std::vector<std::pair<int, int>> *main, std::vector<std::pair<int, int>> *side);
// Whether reference list `l` has exactly these counts.
bool same_list(
    const MetaList &l, const std::vector<std::pair<int, int>> &main, const std::vector<std::pair<int, int>> &side);
// The ids of the reference lists with exactly these counts, ascending.
std::vector<int> matching_lists(
    const Metagame &m, const std::vector<std::pair<int, int>> &main, const std::vector<std::pair<int, int>> &side);
// The classification of a registered list named `name` with these counts: the reference list of
// that name if there is one, else the one with the most cards in common (the sum over cards of the
// lesser main count plus the lesser side count; ties to the lower id), else (nothing in common, or
// no reference list) -1.
int classify_list(const Metagame &m, const std::string &name, const std::vector<std::pair<int, int>> &main,
    const std::vector<std::pair<int, int>> &side);

// The hypothesis network of `net` with reference counts (main, side) as the player's list (header
// comment). False when it has no world by sizes alone (a list root larger than the list part it
// stands for, or more cards created from zone none than the list leaves set aside).
bool hypothesis_network(const Network &net, const std::vector<std::pair<int, int>> &main,
    const std::vector<std::pair<int, int>> &side, Network *out);
// The classes a feasibility check of `net` needs: every card a node is pinned to or a junction
// fixes (the cards nothing names are interchangeable, so they share the OTHER class exactly).
Universe network_universe(const Network &net);

}  // namespace knowledge

#endif /* KNOWLEDGE_TEMPLATES_H */
