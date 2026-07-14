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

    // ── bo3 sideboard-root budget (mirrors az_selfplay.py's sb_sims/sb_worlds/
    // sb_max_depth) ─────────────────────────────────────────────────────────
    // A between-game sideboard prompt is a valid MCTS root (Stage 1-3), but each
    // rollout there re-crosses init_ecs() + deck load + shuffle on RESTORE and its
    // horizon spans the whole next game, so it gets its own (deeper, fewer-sim)
    // budget. -1 = INHERIT the in-game sims/worlds/max_depth (so the parity paths
    // and existing callers are unchanged by default). Selected per-search from the
    // `sideboard_phase` engine global at root setup, then stored per-search so the
    // whole descent uses the ROOT's budget.
    int sb_sims = -1;
    int sb_worlds = -1;
    int sb_max_depth = -1;

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
    // Full reset of per-match sample-buffer state (real-move counter + stored
    // samples). Call ONCE before a bo3 match (the RNG streams across games), and
    // per-game in the bo1 loop.
    void begin_match();
    // End-of-game reset: clears the buffered samples and resets the tau/move
    // counter. Call AFTER a game's samples have been priced+flushed (from the
    // actor's backfill hook), so any sideboard samples recorded before the NEXT
    // game start stay buffered and are priced by that next game's result — and the
    // tau counter resets at the game boundary (before the sideboard prompts),
    // matching Python's per-game game_move.
    void end_game();
    // Samples stored since the last begin_match()/end_game() (valid until the next
    // one). In bo3, between a game's backfill and the next game's start this holds
    // the sideboard-root samples awaiting the next game's z.
    const std::vector<SelfPlaySample>& game_samples() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif /* ACTOR_AZ_MCTS_H */
