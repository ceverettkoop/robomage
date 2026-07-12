#ifndef ACTOR_AZ_MCTS_H
#define ACTOR_AZ_MCTS_H

// In-process determinized PUCT MCTS for the AlphaZero actor (Phase D, M6).
//
// This is the C++ twin of train/mcts.py::run_search, reproducing it EXACTLY so
// whole-game visit-count parity holds at batch=1 (the M6 gate). Where the Python
// reference "pulls" (restore -> determinize -> descend via sim_step), the engine
// "pushes": each decision invokes the input-provider hook, and returning an
// action really advances the engine. This class is that inversion as an explicit
// state machine:
//
//   * IDLE           — between searches. A provider call at a real decision either
//                      begins a search (loop-safe + num_choices>1) or returns the
//                      raw-policy argmax (fallback: not searchable / the only
//                      action).
//   * DESCENDING     — inside one simulation's descent. Each provider call either
//                      selects the next PUCT action (returning it as a real engine
//                      step into the simulation) or, at a leaf / depth cap,
//                      evaluates + backs up and latches a restore (returning 0,
//                      which the cooperative unwind discards).
//   * AWAITING_ROOT  — a restore was latched; the next provider call lands back at
//                      the (restored) root. Advance the sim/world counters and
//                      either start the next simulation (determinize + descend) or,
//                      when all worlds are done, release the snapshot and return
//                      the chosen real action (argmax visits, temperature 0).
//
// A simulated line that reaches game over fires the game-end hook (on_game_end),
// which backs up the terminal value and latches a restore — terminals never go
// through the provider. See az_mcts.cpp for the exact mapping to mcts.py.

#include <cstdint>
#include <memory>
#include <vector>

#include "classes/action.h"  // LegalAction

class AZEvaluator;

struct MCTSConfig {
    int sims = 128;
    int worlds = 4;
    double c_puct = 1.5;
    int max_depth = 60;
    int batch = 1;                  // 1 = exact mcts.py parity; K>1 = virtual-loss batching
    uint32_t world_seed_base = 42;  // world seed(root r, world w) = base + 100003*r + w

    // ── self-play (--selfplay) ──────────────────────────────────────────────
    // When `selfplay` is set, each SEARCHED root stores a training sample and the
    // first `temp_moves` real moves sample the real action from the visit
    // distribution (rng) instead of argmax. Root Dirichlet noise (eps/alpha) is
    // mixed into the base priors per world in begin_world. Defaults keep the
    // parity paths (--search without --selfplay) noise-free (eps=0) and argmax.
    bool selfplay = false;
    double noise_eps = 0.0;         // 0 disables root noise (parity default)
    double noise_alpha = 1.0;       // Dirichlet concentration
    int temp_moves = 20;            // # of leading real moves that sample-from-visits
    uint32_t selfplay_rng_seed = 0; // seeds the per-run noise+sampling RNG
};

// One stored self-play training sample (z is backfilled at real game end).
struct SelfPlaySample {
    std::vector<float> obs;     // ACTOR_OBS_SIZE — clean root obs (before determinize)
    std::vector<float> pi;      // MAX_ACTIONS — normalized root visits in [:nc], else 0
    std::vector<uint8_t> mask;  // MAX_ACTIONS — 1 in [:nc], else 0
    bool mover_is_a;            // root mover seat (obs[SELF_IS_A]>0.5)
};

// One searched root's outcome (recorded for --dump-visits and stats).
struct SearchRootResult {
    int root_index;                // 0-based counter over searched roots this game
    int num_choices;               // root menu width
    std::vector<int64_t> visits;   // summed root visit counts across worlds
    double root_value;             // visit-weighted root Q (root mover perspective)
    int sims_run;
    long sim_steps;                // total engine sim decisions consumed (cost metric)
};

class AZMcts {
public:
    // `evaluator` may be null for a uniform (torch-free) evaluator.
    AZMcts(const MCTSConfig& cfg, AZEvaluator* evaluator);
    ~AZMcts();

    // Input-provider hook body: called for every machine decision. Returns the
    // engine action index to play (a real move at IDLE/finalization, or a
    // simulation step during a descent).
    int on_decision(const std::vector<LegalAction>& actions);

    // Game-end hook body: called when a simulated line reaches game over.
    // `winner` is cur_game.winner (0=draw/none, 1=Player A, 2=Player B). Returns
    // true to keep the main loop parked (a restore is latched) so the search
    // continues; false to let the real game end (never returned while searching).
    bool on_game_end(int winner);

    // Per-searched-root results, in order.
    const std::vector<SearchRootResult>& results() const;

    // ── self-play ───────────────────────────────────────────────────────────
    // Reset per-game state (real-move counter + this game's stored samples).
    // Call at the start of each self-play game (the RNG streams across games).
    void begin_game();
    // Samples stored during the current game (valid until the next begin_game()).
    const std::vector<SelfPlaySample>& game_samples() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif /* ACTOR_AZ_MCTS_H */
