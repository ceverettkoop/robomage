#ifndef AGENTS_SB_EXPLORE_H
#define AGENTS_SB_EXPLORE_H

#include <map>
#include <string>
#include <vector>

// ── Structured sideboard exploration: the proposal q of an exploring phase ────────────────────
//
// docs/rnad_training_plan.md §6.2 part 2, docs/actor.md §2 (the schedule's `sb_explore`). Standard
// library and the agents' frame reader only: q is a function of the deciding seat's frame (its menu
// rows' categories and cards, the MATCH table's sb_swaps) and the seat's configuration.
//
// With s the phase's completed swaps (MATCH.sb_swaps: an IN paired with the OUT that balances it)
// and P the count weights normalised (k = 0 .. 10 swaps):
//
//   a balanced deck (SIDEBOARD_DONE offered): q(Done) = P(k = s) / P(k >= s), 1 when P(k >= s) is 0
//     or no SIDEBOARD_IN row is offered; the rest, 1 - q(Done), over the SIDEBOARD_IN rows in
//     proportion to their card's in-weight;
//   a deck with an IN outstanding (no Done, SIDEBOARD_OUT rows): the OUT rows in proportion to
//     their card's out-weight (IN rows likewise, should a menu offer only those);
//   every other row 0; a menu with none of the three kinds: uniform.
//
// A card's weight is its entry in the seat's in / out map, matched by sb_card_key (ASCII letters
// and digits, lower case: "Grafdigger's Cage" and "grafdiggers cage" are one card), else the
// default weight. Weights are positive, so every offered IN / OUT row has mass.

namespace agents {

class DecisionView;

// The number of count weights: k = 0 .. SB_MAX_COUNT swaps.
constexpr int SB_MAX_COUNT = 10;

// One deck's per-card weights, keyed by sb_card_key.
struct SbCardWeights {
        std::map<std::string, double> in, out;
};

// One seat's exploration in a match: the phase coin's probability, the count distribution's
// weights (SB_MAX_COUNT + 1 non-negative entries, not all 0) and its deck's card weights.
struct SbExplore {
        double p = 0;
        std::vector<double> count_weights = std::vector<double>(SB_MAX_COUNT + 1, 1.0);
        SbCardWeights weights;
        double default_weight = 1.0;
};

// A card name as the weights are keyed: ASCII letters and digits, lower-cased (other bytes dropped).
std::string sb_card_key(const std::string &name);

// The proposal q over the decision's menu rows (above). False with `err` when the frame has no
// MATCH row or a sideboarding row has no card.
bool sb_proposal(const DecisionView &view, const SbExplore &cfg, std::vector<double> *q, std::string *err);

}  // namespace agents

#endif /* AGENTS_SB_EXPLORE_H */
