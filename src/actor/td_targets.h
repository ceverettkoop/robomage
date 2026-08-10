#ifndef ACTOR_TD_TARGETS_H
#define ACTOR_TD_TARGETS_H

// n-step TD value targets for recorded self-play samples — the C++ twin of
// train/az_selfplay.py::compute_td_targets. The two MUST agree exactly: both
// backends write the same shard schema, and the trainer mixes the td_q column
// into the value target without knowing which backend produced it.
//
// The rule, for sample t inside one CHAIN (a contiguous run of samples the
// bootstrap may run inside — never crossing a game boundary or the bo3
// sideboard boundary) whose last sample is L, with E the first sample after t
// in the same chain whose played action was EXPLORATORY (non-argmax; +inf when
// there is none) and j_max = min(E - 1, L):
//
//   * t + n <= j_max  -> bootstrap:  td_q[t] = s * q[t + n]
//   * j_max == L      -> the window reaches the end of the game, so the true
//                        outcome is in reach: td_q[t] = z[t]
//   * otherwise       -> SHORTENED horizon at the exploratory move:
//                        td_q[t] = s * q[j_max]  (z[t] when j_max == t)
//
// s is the mandatory PERSPECTIVE FLIP: q and z are each stored from their own
// sample's MOVER's point of view and the seat alternates freely between
// samples, so s = +1 when the bootstrap sample's mover is the same seat as t's
// and -1 otherwise. Sideboard-root samples form their own chain and always keep
// td_q = z (a run shorter than the horizon is terminal-in-window by the rule).
//
// The Python twin additionally treats a NaN q as "no root value recorded" and
// falls back to z; that case is unreachable here because the actor stores a real
// root value at every sample (it has no expert/behavior-cloning path).

#include <cstdint>
#include <vector>

// One sample's inputs to the rule, in play order within a GAME (see
// compute_td_targets).
struct TdSample {
    float z = 0.0f;             // per-game outcome, mover perspective
    float q = 0.0f;             // search root value, mover perspective
    bool explored = false;      // the played action != the search's argmax
    bool is_sideboard = false;  // a bo3 sideboard root (its own chain)
    bool mover_is_a = false;    // the deciding seat
};

// Fill `out` (resized to samples.size()) with each sample's n-step TD target.
// `samples` is ONE game's stored samples in play order — including any leading
// bo3 sideboard-root samples, which this function isolates into their own chain.
void compute_td_targets(const std::vector<TdSample>& samples, int td_n,
                        std::vector<float>& out);

#endif /* ACTOR_TD_TARGETS_H */
