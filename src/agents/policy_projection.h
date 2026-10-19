#ifndef AGENTS_POLICY_PROJECTION_H
#define AGENTS_POLICY_PROJECTION_H

#include <vector>

// ── DeepNash's play-time policy projection: thresholding, then discretisation ────────────────
//
// rnad_training_plan.md T6; OpenSpiel's FineTuning (open_spiel/python/algorithms/rnad/rnad.py,
// _threshold and _discretize_single), in double precision over one decision's menu rows. Std only:
// the nn: agent (src/actor/nn_agent.h) projects the distribution it samples from, and the gate
// links it to recompute the agent's behaviour distributions.
//
// threshold_policy(mu, p), p > 0: the rows whose probability is below p are set to 0 and the rest
// renormalised (each divided by their sum). When every row is below p the distribution is kept
// as it is, so at least the most probable row always survives. p <= 0 leaves mu unchanged.
//
// discretize_policy(mu, n), n > 0: each row gets an integer count, the counts sum to n, and the
// row's probability becomes count / n. The rows are visited in descending probability, ties in
// row order (a stable sort); each takes ceil(mu[row] * n), capped at what is left of n; whatever is
// left after the last row (only when the rows' probabilities sum to less than 1) goes to the first
// row visited. A count within 1e-9 above an integer rounds down to it, so a probability that is
// k / n in exact arithmetic is not pushed to k + 1 by float noise (OpenSpiel takes the plain
// ceiling); a projected distribution is therefore a fixed point. The most probable row (the first
// in row order among equals) keeps the largest count, a row of probability 0 gets 0, and the
// result depends only on mu and n. n <= 0 leaves mu unchanged. For n a power of two the
// probabilities sum to exactly 1 in floating point; otherwise the counts do.

namespace agents {

void threshold_policy(std::vector<double> *mu, double threshold);
void discretize_policy(std::vector<double> *mu, int n);

}  // namespace agents

#endif /* AGENTS_POLICY_PROJECTION_H */
