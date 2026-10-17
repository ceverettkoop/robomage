#ifndef KNOWLEDGE_BELIEF_H
#define KNOWLEDGE_BELIEF_H

#include <cstdint>
#include <map>
#include <vector>

#include "frame_view.h"
#include "network.h"

// ── The belief layer: a posterior over the reference lists (docs/knowledge_store.md §10.6) ────
//
// Not a deduction: the b_* tables (level expert+belief) are probabilities under a stated model,
// kept apart from the k_* facts. The model of one game of a player whose list L is hidden:
//   - the prior over the METAGAME frame's reference lists is uniform;
//   - every junction of L's hypothesis network (templates.h) distributes the units it takes in
//     uniformly at random among its out nodes (and its out-side fixed cards): a shuffled library's
//     order, a draw from the top of a run, and the player's own choices among cards the viewer cannot
//     tell apart (which card of the hand is cast, discarded or put back) all count as uniform. The
//     evidence is every pin and fixed card of the network (each identity the viewer saw), and the
//     likelihood P(evidence | L) is the probability that all of them hold;
//   - a junction J with in-side composition y (copies per card, its fixed cards included, n units in
//     all) gives its out nodes compositions x_o (sizes s_o) with probability
//         Π_c y_c! / Π_o Π_c x_oc!  ×  Π_o s_o! / n!          (multivariate hypergeometric),
//     which counts the arrangements of the physical cards (distinguishable copies), not the
//     multisets: a one-card draw that turns out to be Lightning Bolt has probability 4/60 under a
//     list with four Bolts and 1/60 under one with one, where a count of consistent multiset worlds
//     (1 each) would not tell them apart;
//   - so P(evidence | L) = Σ over the network's consistent worlds of the product of its junctions'
//     probabilities: the weighted count of the worlds the sound layer asks the existence of.
// Games are independent shuffles, so a match's likelihood is the product over its games.
//
// network_likelihood computes that sum exactly, in double arithmetic in a fixed evaluation order
// (so a function of the network alone), by sweeping the junctions backwards. A junction's uniform
// distribution is a uniformly random bijection from its out slots to its in slots, so a pinned slot
// holds the card of the in slot it maps to: a known one (a pinned in node, an in-side fixed card),
// a slot of an EXACT root (whose slots are a shuffle of its contents: probability Π_c (T_c)_d_c over
// the slots' count) or a slot of an unpinned in node, which passes the demand for that card on to the
// junction that made the node. The state is the demand on every node whose consumer has been swept
// and whose producer not yet (a count per named card), so its size is bounded by the pins, not by the
// nodes' sizes; the slots of a node are exchangeable, so the demand's multiset is all that matters.
// Per junction, with D the demand on its out side (n in slots):
//     P = Π_c D_c! / (a_c! Π_t d_tc!)  ×  Π_c (K_c)_a_c  ×  Π_t (s_t)_|d_t|  /  (n)_|D|
// summed over the ways a_c of each card's demand land on its K_c known slots and d_t on each other
// in node t of s_t slots ((a)_k the falling factorial). Past its step budget it reports unfinished and
// the caller uses seen_log_likelihood instead.

namespace knowledge {

// The fixed-point unit of a b_* probability: probabilities are integers of PROB_SCALE (10^-9).
constexpr uint32_t PROB_SCALE = 1000000000u;

struct Likelihood {
        bool finished = false;  // the weighted count completed within its budget
        bool possible = false;  // finished and some world has positive probability
        double log_p = 0;       // finished and possible: log P(every pin and fixed card holds)
        uint64_t steps = 0;
};

// P(evidence) for `net` under the model above; every root of `net` must be EXACT (a hypothesis
// network), else the result is unfinished.
Likelihood network_likelihood(const Network &net, uint64_t budget);

// The fallback when the weighted count runs past its budget: the cards `seen` (the fewest copies of
// each the observations imply, k_seen.game) taken as a uniformly random ordered sample of `list`'s
// 75: Σ_c log (T_c)_k_c − log (N)_m, (a)_k the falling factorial. Finite whenever the list holds at
// least the copies seen.
double seen_log_likelihood(const MetaList &list, const std::map<int, uint8_t> &seen);

// The posterior of lists with log weights `log_w` (prior included) over `support` (1 = possible) in
// PROB_SCALE units: zero outside the support, at least one unit inside it, summing to PROB_SCALE
// exactly (largest remainder, ties to the lower index) unless the support is empty (all zero).
std::vector<uint32_t> fixed_point_posterior(const std::vector<double> &log_w, const std::vector<uint8_t> &support);

// The posterior as doubles (the same weights, unrounded; all zero for an empty support).
std::vector<double> posterior(const std::vector<double> &log_w, const std::vector<uint8_t> &support);

// A log-likelihood as the b_list column: milli-nats, rounded to nearest, clamped to the i32 range
// above its minimum (INT32_MIN marks an excluded list).
int32_t loglik_column(double log_p);

}  // namespace knowledge

#endif /* KNOWLEDGE_BELIEF_H */
