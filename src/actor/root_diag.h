#ifndef ACTOR_ROOT_DIAG_H
#define ACTOR_ROOT_DIAG_H

// Per-searched-root diagnostics carried alongside a recorded sample, written
// by ShardAccumulator into the shard's same-stem `.diag` sidecar — the C++
// twin of train/shard_record.py's diag dicts (DIAG_KEYS / pack_diag), so the
// browsers' decision table, search line and F7 tree rebuild
// (train/tree_rebuild.py) read an actor recording exactly like a GUI one.
// Vectors are menu-length (num_choices) or per-world; the writer pads them to
// the fixed sidecar widths.

#include <cstdint>
#include <vector>

// Row kinds (mirror shard_record.DIAG_KIND_*).
constexpr int8_t DIAG_KIND_NONE = 0;      // no search ran
constexpr int8_t DIAG_KIND_SEARCH = 1;    // in-game PUCT search
constexpr int8_t DIAG_KIND_PLAN = 3;      // sideboard plan search

struct RootDiag {
    int8_t kind = DIAG_KIND_NONE;
    int32_t num_choices = 0;
    std::vector<int32_t> visits;          // summed root visits per action
    std::vector<float> priors;            // the evaluator's RAW root priors
    //                                       (pre-noise, pre-merge)
    std::vector<float> q_act;             // ΣW/ΣN per action (0 unvisited)
    std::vector<float> w_sum;             // ΣW per action
    float root_value = 0.0f;
    int32_t sims_run = 0;
    int32_t sim_steps = 0;
    int32_t memo_hits = 0;
    std::vector<int64_t> world_seeds;     // per-world determinize seeds
    std::vector<int32_t> world_visits;    // per-world root N.sum()
    std::vector<float> world_values;      // per-world root ΣW/ΣN
    // Index of this decision in the match's real-action list (the `.rmplay`
    // sidecar's row_decision_idx: how many actions to replay to land here).
    int32_t decision_idx = 0;
};

#endif /* ACTOR_ROOT_DIAG_H */
