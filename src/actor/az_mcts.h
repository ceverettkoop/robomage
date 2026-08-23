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
#include <functional>
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
    // Cross-world batched leaf evaluation (Stage 0 of
    // docs/gpu_selfplay_inference_plan.md): run the per-world sims round-robin
    // and defer each freshly expanded (or depth-capped) leaf into a PendingLeaf,
    // flushed in one batched forward (K <= worlds) before any world's OWN next
    // descent starts — so NO virtual loss is needed and every per-world tree is
    // arithmetically identical to the sequential batch=1 search. Only the
    // batched GEMM's last-ulp logits can differ; under a uniform evaluator the
    // visit counts are bit-identical (the test_mcts_parity cross-world gate).
    // Mutually exclusive with batch > 1. Inert (sequential path, unbatched) for
    // a search whose budget has leaf rollouts on — sb roots by default — since
    // a deferred leaf cannot drive a playout.
    bool cross_world = false;
    uint32_t world_seed_base = 42;  // world seed(root r, world w) = base + 100003*r + w
    // Merge interchangeable duplicate menu actions into one search edge
    // (src/actor/menu_merge.h, mirroring mcts.py's run_search merge_dupes
    // kwarg — the two sides must agree or visit parity breaks). Must stay
    // constant for the process (never flip across a persisted sideboard
    // boundary's trees).
    bool merge_dupes = true;

    // ── bo3 sideboard PLAN search (mirrors mcts.py::run_plan_search) ────────
    // A between-game sideboard prompt is NOT searched with PUCT: the root is a
    // flat set of PLANS — complete pick sequences through the mover's Done. A
    // coverage pass builds one argmax-greedy completion per legal root action,
    // then `sb_branches` deterministic alternate completions of the best-Q
    // first picks (variant v takes the SECOND-best prior at the v-th
    // completion decision — no rng, no cross-language rng parity needed).
    // Every plan is priced on every `sb_worlds` determinized world (replay its
    // picks, then a leaf rollout to end of player-turn `sb_rollout_turns` of
    // the next game); plan value = mean over worlds, Q per first pick = its
    // best plan's value, and the training/pi target is softmax(Q / kSbPiTau).
    // Plan values are memoized per (world seed, sorted pick multiset) in a
    // boundary-shared table (multisets holding one card in both directions
    // excluded — see memo_eligible in az_mcts.cpp), which
    // both dedups converging coverage plans and re-prices consistent plans for
    // free at the boundary's later picks — this REPLACES tree-based boundary
    // persistence. -1 = the compiled default (kDefaultSbBranches /
    // kDefaultSbWorlds / kDefaultSbRolloutTurns in az_mcts.cpp, mirroring
    // cli_spec's DEFAULT_SB_*; production callers pass the flags explicitly).
    int sb_branches = -1;
    int sb_worlds = -1;
    int sb_rollout_turns = -1;

    // ── leaf rollouts (mirrors mcts.py's rollout_turns; in-game roots) ──────
    // When the budget in force has rollout_turns > 0, a freshly expanded leaf is
    // not evaluated in place: the raw policy (argmax of the evaluator's priors,
    // both seats, no rng) plays the determinized world forward to the end of
    // player-turn `anchor + rollout_turns` (anchor = the ROOT's turn) and THAT
    // state's net value (or a true terminal ±1) is backed up. Rolled-out states
    // are never added to the tree; the max_depth cap still bounds tree descent
    // only (a depth-cap leaf does NOT roll out, matching mcts.py). NOTE: when
    // the root's budget has rollouts on, leaf evaluation always takes the
    // immediate (batch=1) path even under batch>1 — deferred PendingLeaf
    // evaluation cannot drive a playout.
    int rollout_turns = 0;

    // ── self-play (--selfplay) ──────────────────────────────────────────────
    // When `selfplay` is set, each SEARCHED root stores a training sample and the
    // first `temp_moves` real moves sample the real action from the visit
    // distribution (rng) instead of argmax. Root Dirichlet noise (eps/alpha) is
    // mixed into the base priors per world in begin_world. Defaults keep the
    // parity paths (--search without --selfplay) noise-free (eps=0) and argmax.
    bool selfplay = false;
    // Record searched-root samples WITHOUT the self-play exploration knobs
    // (no root noise, always argmax): eval/gate matches (--search --record)
    // then write trainer-schema shards of what the two nets actually played,
    // so a gate's decisions are browsable/probeable like self-play data. Implied
    // by `selfplay`; on its own it never changes a played action.
    bool record = false;
    double noise_eps = 0.0;         // 0 disables root noise (parity default)
    double noise_alpha = 1.0;       // Dirichlet concentration
    int temp_moves = 20;            // # of leading real moves that sample-from-visits
    uint32_t selfplay_rng_seed = 0; // seeds the per-run noise+sampling RNG

    // ── vs-scripted seat (mirrors az_selfplay._play_match's agent/net_is_a) ──
    // 0 = none (pure self-play); 1 = Player A, 2 = Player B is piloted by the
    // scripted oracle (set_scripted_provider). That seat's REAL decisions are
    // answered by the provider — no search, no sample, no searched/fallback
    // counters — but they DO advance the per-game tau counter (Python's
    // game_move counts every decision) and latch into a live sideboard
    // boundary. Search simulations never consult the provider: tree play is
    // net-both-seats, exactly like the Python reference.
    int scripted_seat = 0;
};

// One stored self-play training sample (z + td_q are backfilled at real game end).
struct SelfPlaySample {
    std::vector<float> obs;     // ACTOR_OBS_SIZE — clean root obs (before determinize)
    std::vector<float> pi;      // MAX_ACTIONS — normalized root visits in [:nc], else 0
    std::vector<uint8_t> mask;  // MAX_ACTIONS — 1 in [:nc], else 0
    bool mover_is_a;            // root mover seat (obs[SELF_IS_A]>0.5)
    // ── n-step TD target inputs (mirror az_selfplay.py's per-sample q/explored;
    // see td_targets.h for the rule they feed) ──────────────────────────────
    float q = 0.0f;             // this search's ROOT VALUE, root-mover perspective
    bool explored = false;      // the played action != the visit argmax (tau branch)
    bool is_sideboard = false;  // the root was a bo3 sideboard prompt (own chain)
};

// One searched root's outcome (recorded for --dump-visits and stats).
struct SearchRootResult {
    int root_index;                // 0-based counter over searched roots this game
    int num_choices;               // root menu width
    std::vector<int64_t> visits;   // summed root visit counts across worlds; at
    //                                a PLAN root, llround(pi * 1e6) fixed-point
    double root_value;             // visit-weighted root Q (root mover perspective);
    //                                pi-weighted mean Q at a plan root
    int sims_run;                  // NEW sims this search; evaluator calls at a
    //                                plan root (its pacing unit)
    long sim_steps;                // total engine sim decisions consumed (cost metric)
    int memo_hits = 0;             // plan-value memo hits this search
    // Plan (sideboard) roots only — empty for in-game tree roots. The float64
    // per-action Q / pi the parity gate compares bit-exactly against
    // mcts.run_plan_search.
    bool plan_root = false;
    std::vector<double> q;
    std::vector<double> pi;
};

class AZMcts {
public:
    // `evaluator` may be null for a uniform (torch-free) evaluator.
    // `evaluator_b` (gate/eval matches, az_actor --model-b) is Player B's OWN
    // evaluator: when non-null, each REAL decision selects the net by the seat
    // to move — Player A's decisions (searches, fallbacks) use `evaluator`,
    // Player B's use `evaluator_b` — and the selected net evaluates EVERY
    // node of that decision's search (either seat's simulated positions),
    // mirroring the Python gate where each seat's controller owns its
    // evaluator. Null (the default) keeps the single-model behavior exactly.
    AZMcts(const MCTSConfig& cfg, AZEvaluator* evaluator,
           AZEvaluator* evaluator_b = nullptr);
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

    // vs-scripted: install the scripted seat's decision source (the oracle
    // client). Required when cfg.scripted_seat != 0 — a scripted-seat decision
    // with no provider is fatal. fn(obs, num_choices) -> action index.
    void set_scripted_provider(std::function<int(const float*, int)> fn);

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
