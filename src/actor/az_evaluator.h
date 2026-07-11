#ifndef ACTOR_AZ_EVALUATOR_H
#define ACTOR_AZ_EVALUATOR_H

// Thin wrapper around a TorchScript-exported AZNet (train/az_net.py). Loads the
// serialized module (`<ckpt>.ts.pt`), verifies the sibling `.ts.meta.json`
// layout handshake against the engine's compile-time constants, and evaluates
// the net for a single decision. EVERY libtorch call is wrapped in try/catch and
// on any error prints to stderr + exit(1) — a C++ exception must never propagate
// into the -fno-exceptions engine frames that call back into this via the input
// provider hook.
//
// This is the only actor TU that includes libtorch; keep torch out of the engine
// build (see the Makefile filter of src/actor/%).

#include <memory>
#include <string>
#include <vector>

// torch is confined to az_evaluator.cpp via the pimpl (Impl) below — this header
// stays torch-free so the rest of the actor (obs_builder, main) never sees it.

struct AZEvalResult {
    std::vector<float> priors;  // softmax over the first num_choices logits
    float value;                // tanh value in [-1, 1], current-mover view
};

// Double-precision evaluation result for the PUCT tree math. Mirrors
// train/az_net.py::AZEvaluator.evaluate EXACTLY: the float32 softmax over the
// first num_choices logits is cast to float64 and renormalized by its float64
// sum (uniform fallback on a non-finite/zero total), and the tanh value is the
// float32 head output widened to double. Used by the MCTS so the tree
// arithmetic is bit-identical to the Python reference.
struct AZEvalResultD {
    std::vector<double> priors;  // normalized float64 priors, length num_choices
    double value;                // tanh value in [-1, 1], current-mover view
};

class AZEvaluator {
public:
    AZEvaluator();
    ~AZEvaluator();

    // Load `<path>.ts.pt` and verify `<path>.ts.meta.json`. Fatal (stderr+exit)
    // on any load/verify failure.
    void load(const std::string& path);

    // Forward the net for one decision. `obs` points at ACTOR_OBS_SIZE floats.
    AZEvalResult evaluate(const float* obs, int num_choices);

    // Double-precision evaluation for the PUCT tree (see AZEvalResultD).
    AZEvalResultD evaluate_double(const float* obs, int num_choices);

    // Batched double-precision evaluation: `k` decisions laid out as
    // `obs` = k contiguous ACTOR_OBS_SIZE-float rows, `num_choices[i]` the legal
    // count of row i. Returns k results in order. One forward over a [k, OBS]
    // batch — used by the MCTS batched-leaf mode (--batch K>1). Each row's priors
    // are computed exactly as evaluate_double (per-row float32 softmax over the
    // first num_choices logits, widened to float64 and renormalized).
    std::vector<AZEvalResultD> evaluate_double_batch(const float* obs,
                                                     const std::vector<int>& num_choices);

    // Greedy action: argmax of the masked logits over the first num_choices.
    int argmax_action(const float* obs, int num_choices);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif /* ACTOR_AZ_EVALUATOR_H */
