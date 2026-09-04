#include "az_mcts.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "actor/sb_rules.h"   // sb_dead_mask (dead-sideboard-card pruning)
#include "az_evaluator.h"
#include "classes/action.h"   // LegalAction::description (divergence diagnostic)
#include "classes/game.h"     // cur_game.turn (rollout horizon check)
#include "cli_output.h"       // step_to_string (divergence diagnostic)
#include "components/zone.h"  // Zone::PLAYER_A / PLAYER_B
#include "error.h"
#include "menu_merge.h"       // menu_merge_reps (duplicate-edge merging)
#include "obs_builder.h"      // build_obs, ACTOR_OBS_SIZE, ACTOR_SELF_IS_A_IDX
#include "search_server.h"    // search_loop_safe, search_request_restore, determinize_hidden_state
#include "snapshot.h"         // snapshot_save, snapshot_release_all

// True while a between-game bo3 sideboard prompt is live (the same engine global
// the obs is_sideboard_phase flag serializes from). Selects the sideboard search
// budget at root setup — do NOT use match_game_number, which reflects the just-
// ended game at a g1->g2 sideboard root.
extern bool sideboard_phase;

// Index of the game currently in progress (0-based); the obs serializes
// match_game_number + 1 during a sideboard phase (the UPCOMING game), which is
// the boundary-identity component mcts.py's sb_root_key reads back out.
extern int match_game_number;

// Observation float "self is Player A" (mirrors train/env.py::_SELF_IS_A_IDX).
// Determines which seat moves at a node.
static constexpr int SELF_IS_A_IDX = ACTOR_SELF_IS_A_IDX;

// The single snapshot slot the search reserves (matches mcts.py's snapshot_slot=0).
static constexpr int SEARCH_SLOT = 0;

// Leaf-rollout step cap per horizon turn. MIRRORS mcts.py's
// ROLLOUT_STEPS_PER_TURN — a cap-triggered cutoff must fire at the same step on
// both sides or visit parity breaks.
static constexpr long kRolloutStepsPerTurn = 40;

// ── sideboard plan-search constants (mirror mcts.py / cli_spec) ──────────────
// Temperature of the pi = softmax(Q/tau) sideboard target (mcts.SB_PI_TAU).
static constexpr double kSbPiTau = 0.25;
// Compiled fallbacks for MCTSConfig's -1 sentinels (cli_spec DEFAULT_SB_*).
static constexpr int kDefaultSbBranches = 8;
static constexpr int kDefaultSbWorlds = 4;
static constexpr int kDefaultSbRolloutTurns = 6;
// Safety bound on one plan's own pick sequence (mcts._PLAN_PICK_CAP), and an
// EXACT one: the engine's menu is IN-FIRST, so every decision is either the IN
// half of a swap, the OUT half that closes it, or Done. SIDEBOARD_SWAP_CAP (15)
// completed swaps therefore cost at most 15 * 2 + 1 (Done) = 31 decisions, and
// the stranded path is no longer: 14 swaps + the stranding IN + its forced OUT +
// the Done-only menu is 31 too. The check is a strict `>` applied BEFORE the pick
// is appended, so a legal line (whose longest prefix at check time is 30) never
// trips it and 31 is not off by one. Anything past it is a broken menu loop.
static constexpr int kPlanPickCap = 31;
// Bound on extras attempts per root: branches * this (mcts.run_plan_search's
// max_attempts multiplier).
static constexpr int kExtrasAttemptFactor = 4;

// Hash coin — the C++ twin of az_selfplay._hash_coin: Bernoulli(prob) from a
// murmur3 finalizer of (seed, idx), 24-bit threshold. The two MUST stay in
// exact lockstep so a knob means the same coin on either backend. prob >= 1 /
// <= 0 short-circuit without consulting the hash.
static bool hash_coin(uint32_t seed, uint32_t idx, double prob) {
    if (prob >= 1.0) return true;
    if (prob <= 0.0) return false;
    uint32_t x = seed ^ (0x9E3779B9u * idx);
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x ^= x >> 16;
    return (x >> 8) < static_cast<uint32_t>(prob * static_cast<double>(1u << 24));
}

// Playout-cap coin (az_selfplay.playout_cap_full): does the root_idx-th
// searched in-game root of this match get the full sims budget?
static bool playout_cap_full(uint32_t cap_seed, uint32_t root_idx,
                             double full_search_frac) {
    return hash_coin(cap_seed, root_idx, full_search_frac);
}

// Exploration clock (az_selfplay.explore_prob; knobs documented on
// MCTSConfig): P(a learner searched root at game `turn` samples its real
// action from the visit distribution). Same double arithmetic as the Python
// twin so the coin threshold matches bit-for-bit.
static double explore_prob(int turn, const MCTSConfig& cfg) {
    const double floor = std::min(std::max(cfg.explore_floor, 0.0), 1.0);
    if (turn <= cfg.explore_full_turns) return 1.0;
    if (cfg.explore_decay_turns > 0 &&
        turn <= cfg.explore_full_turns + cfg.explore_decay_turns) {
        const double remaining = static_cast<double>(
            cfg.explore_full_turns + cfg.explore_decay_turns - turn);
        return floor + (1.0 - floor) *
                           (remaining / static_cast<double>(cfg.explore_decay_turns));
    }
    return floor;
}

// The game turn an exploration-clock root belongs to (az_selfplay.obs_turn):
// 0 at a sideboard root (the obs turn slot still holds the ENDED game's turn),
// else the root obs's current-turn float decoded.
static int obs_turn(const float* o, bool sideboard_root) {
    if (sideboard_root) return 0;
    return static_cast<int>(
        std::lround(static_cast<double>(o[CUR_TURN_IDX]) * TURN_NORMALIZER));
}

// Salt XORed into the match seed for the exploration coin so it never
// correlates with the playout-cap coin at the same root index
// (az_selfplay._EXPLORE_COIN_SALT).
static constexpr uint32_t kExploreCoinSalt = 0x5BD1E995u;

// Exploration coin (az_selfplay.explore_coin): hash_coin under `prob` on the
// salted match seed and the playout cap's searched-root index.
static bool explore_coin(uint32_t cap_seed, uint32_t root_idx, double prob) {
    return hash_coin(cap_seed ^ kExploreCoinSalt, root_idx, prob);
}

namespace {

struct PathEntry {
    struct Node* node;
    int action;
};

// One tree node, mirroring mcts.py::_Node. Q is stored in this node's OWN mover
// perspective (self_is_a), so plain argmax selects correctly at both seats.
struct Node {
    int num_choices;
    bool self_is_a;
    std::vector<double> P;        // priors (float64, from the evaluator)
    std::vector<int64_t> N;       // visit counts
    std::vector<double> W;        // value sums (this node's perspective)
    std::unordered_map<int, Node*> children;
    // Duplicate-edge merge partition (menu_merge.h; mirrors _Node.rep):
    // rep[i] is action i's representative index. Empty = merging disabled.
    // Filled (and P folded onto the representatives) by Impl::init_merge
    // after every P assignment; select() skips non-representatives, so
    // N/W/children only ever live at representative indices.
    std::vector<int16_t> rep;
#ifndef NDEBUG
    // Debug builds only: the menu's per-action metadata (cat / card id / ctrl /
    // zone / slot ref / ordinal, from the obs blocks) recorded at node creation,
    // so a world-consistency mismatch can report WHAT changed in the menu
    // instead of only the count. Compiled out under -DNDEBUG (BUILD=RELEASE).
    std::vector<float> dbg_menu;   // 6 floats per action
    std::vector<float> dbg_state;  // state header (36) + self/opp library cts

#endif

    Node(int nc, std::vector<double> priors, bool is_a)
        : num_choices(nc), self_is_a(is_a), P(std::move(priors)),
          N(static_cast<size_t>(nc), 0), W(static_cast<size_t>(nc), 0.0) {}

    // argmax(Q + c*P*sqrt(1+ΣN)/(1+N)) with Q = W/N (0 unvisited). All float64,
    // first-max tie-break (numpy argmax). Bit-identical to _Node.select.
    int select(double c_puct) const {
        int64_t sumN = 0;
        for (int64_t n : N) sumN += n;
        double sqrt_sum = std::sqrt(1.0 + static_cast<double>(sumN));
        int best = 0;
        double best_val = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < num_choices; i++) {
            // Merged-away duplicates are never selected (Python: sel_mask
            // -inf). rep[0] == 0, so the best=0 init stays a representative.
            if (!rep.empty() && rep[static_cast<size_t>(i)] != i) continue;
            double ni = static_cast<double>(N[static_cast<size_t>(i)]);
            double q = N[static_cast<size_t>(i)] > 0
                           ? W[static_cast<size_t>(i)] / ni
                           : 0.0;
            double u = c_puct * P[static_cast<size_t>(i)] * (sqrt_sum / (1.0 + ni));
            double val = q + u;
            if (val > best_val) {
                best_val = val;
                best = i;
            }
        }
        return best;
    }
};

// A leaf whose net evaluation is deferred to the next batched forward (--batch K>1).
struct PendingLeaf {
    std::vector<float> obs;        // ACTOR_OBS_SIZE floats (leaf state)
    int num_choices;
    std::vector<PathEntry> path;   // full descent path (nodes are pool-stable)
    Node* parent;
    int action;
};

}  // namespace

struct AZMcts::Impl {
    // Tree-search phases (in-game roots) plus the sideboard PLAN-search phases
    // (mirroring mcts.run_plan_search under the engine's push model):
    //   PLAN_GEN     — generating the plan in flight's pick sequence on world 0
    //                  (each provider call inside the mover's sideboard run
    //                  returns the greedy — or variant-deviated — pick).
    //   PLAN_REPLAY  — force-replaying the recorded picks on world w > 0.
    //   PLAN_ROLLOUT — pricing the plan on the current world: the greedy
    //                  playout from the handoff state to the horizon.
    //   AWAITING_ROOT is shared: with plan_active it advances plan/world
    //   scheduling instead of sim/world counters.
    enum Phase { IDLE, DESCENDING, ROLLOUT, AWAITING_ROOT,
                 PLAN_GEN, PLAN_REPLAY, PLAN_ROLLOUT };

    MCTSConfig cfg;
    // ACTIVE evaluator: every eval_one/eval_priors/flush_pending forward of the
    // decision in flight goes through this pointer. Single-model runs point it
    // at eval_a once and never touch it; two-model runs (eval_b set — gate/eval
    // matches) re-select it at every REAL decision by the seat to move (see
    // begin_or_fallback). Null → uniform evaluator (torch-free).
    AZEvaluator* eval;
    AZEvaluator* eval_a;  // Player A's net (== the single net when eval_b null)
    AZEvaluator* eval_b;  // Player B's net (null → single-model: A's for both)
    // vs-scripted: the scripted seat's decision source (the oracle client).
    std::function<int(const float*, int)> scripted_provider;
    Phase phase = IDLE;
    int root_counter = 0;

    // Per-search state (valid IDLE→...→finalize).
    int this_root = 0;
    int root_n = 0;
    bool root_is_a = false;
    bool root_is_sb = false;  // this root is a bo3 sideboard prompt (sample tag)
    std::vector<double> root_priors;
    // Human-readable descriptions of the CURRENT search root's live menu, captured
    // at search start (release-safe, unlike the NDEBUG-only dbg_menu float capture).
    // Used only by dump_divergence() to compare the root menu against the live menu
    // when a returned action index lands outside the engine's current menu.
    std::vector<std::string> root_menu_desc;
    std::vector<int64_t> visit_totals;  // summed root.N across worlds
    std::vector<double> w_totals;       // summed root.W across worlds (per action;
    //                                     w_totals[a]/visit_totals[a] = the Q the
    //                                     self-play sample's q records for the
    //                                     played action, mirroring mcts.py's
    //                                     SearchResult.q)
    double value_acc = 0.0;
    int sims_run = 0;
    long sim_steps = 0;
    int sims_per_world = 1;
    // The budget in force for THIS search (chosen at root setup: the sideboard
    // budget when the root is a sideboard prompt and the sb_* config is >= 0, else
    // the in-game budget). Stored per-search so the world/sim loop bound and the
    // descent depth cap all use the ROOT's budget throughout.
    int cur_sims = 0;
    int cur_worlds = 0;
    int cur_max_depth = 0;
    // Leaf-rollout budget in force for THIS search (latched at root setup like
    // the sims/worlds/depth budget above; mirrors run_search's rollout_turns /
    // _rollout_anchor). rollout_steps counts the actions of the rollout
    // currently in flight (mirrors mcts.py::_rollout's `steps`).
    int cur_rollout_turns = 0;
    int cur_rollout_anchor = 0;
    long cur_rollout_cap = 0;
    long rollout_steps = 0;
    int cur_world = 0;
    int cur_sim = 0;  // index of the simulation currently running within cur_world
    uint32_t cur_world_seed = 0;
    Node* cur_root = nullptr;

    // ── cross-world batching (cfg.cross_world; see az_mcts.h) ──────────────
    // Latched per search at root setup: cross scheduling applies only when the
    // budget in force has rollouts OFF (a deferred leaf cannot drive a playout,
    // same restriction as batch>1). world_seeds/world_pending are indexed by
    // world; a set pending flag means that world's previous sim deferred a
    // leaf that has not been flushed yet.
    bool cross_active = false;
    std::vector<uint32_t> world_seeds;
    std::vector<uint8_t> world_pending;

    std::vector<Node*> world_roots;    // this search's per-world roots
    std::vector<int> cur_budgets;      // per-world sim budgets

    // ── sideboard plan search (mirrors mcts.run_plan_search) ────────────────
    // One plan = a complete pick sequence through the mover's Done, priced on
    // every world; the boundary-shared memo keys plan values by (world seed,
    // sorted pick multiset). Boundary identity is (seat, upcoming game): the
    // memo + the latched real picks carry across the boundary's consecutive
    // roots; the world seeds stay pinned to the boundary's FIRST searched
    // root.
    struct Plan {
        int first_action = 0;
        int variant = 0;                              // 0 = coverage / greedy
        std::vector<int> pick_actions;                // full pick sequence
        std::vector<std::array<int16_t, 3>> picks;    // descriptors (no Done)
        std::vector<std::array<int16_t, 3>> multiset; // sorted(sb_picks+picks)
        std::vector<double> values;                   // per world, ROOT persp
    };
    bool plan_active = false;          // this root runs the plan search
    bool sb_active = false;            // a boundary is latched
    bool sb_seat_is_a = false;         // boundary identity: seat...
    int sb_game = 0;                   // ...and upcoming game number
    int sb_seed_root = 0;              // first searched root's index (pins seeds)
    // Real picks since the BOUNDARY root, as (cat, card id, seat) descriptors —
    // the memo key's base every plan's multiset extends.
    std::vector<std::array<int16_t, 3>> sb_picks;
    using PlanKey = std::pair<uint32_t, std::vector<std::array<int16_t, 3>>>;
    std::map<PlanKey, double> plan_memo;      // cleared at boundary end
    std::set<std::vector<std::array<int16_t, 3>>> plan_evaluated;
    std::vector<Plan> plans;           // this root's evaluated plans, in order
    Plan cur_plan;                     // the plan in flight
    bool cur_plan_memoable = false;
    int plan_world = 0;                // world being priced
    size_t plan_replay_pos = 0;        // replay cursor into cur_plan.pick_actions
    int plan_dev = 0;                  // completion-decision counter (variant)
    int coverage_cursor = 0;           // next coverage first pick
    std::vector<int> plan_ranked;      // extras ranking (set after coverage)
    std::vector<bool> plan_dead;       // root dead-IN mask (sb_rules.h)
    std::vector<bool> gen_dead;        // per-completion-step scratch mask
    int extra_attempt = 0;
    int novel_extras = 0;
    int memo_hits = 0;

    std::vector<std::unique_ptr<Node>> pool;  // owns every node this search
    std::vector<PathEntry> path;              // current descent
    std::vector<PendingLeaf> pending;         // batched-leaf collection (batch>1)

    std::vector<SearchRootResult> results;

    // ── self-play state ─────────────────────────────────────────────────────
    std::mt19937 rng;                        // noise + real-action sampling RNG
    std::vector<float> root_obs;             // clean root obs (captured before determinize)
    std::vector<SelfPlaySample> game_samples; // samples stored this game
    // ── playout cap + exploration coin (see az_mcts.h) ─────────────────────
    // cap_seed_ = the match's engine seed; cap_root_counter counts searched
    // in-game roots WITHIN the match (never reset at a game boundary,
    // mirroring the Python per-match counter); cur_root_idx is the in-flight
    // in-game root's index (the key both coins share); cur_full is the
    // playout-cap coin's verdict for the search in flight (latched at root
    // setup like cur_sims).
    uint32_t cap_seed_ = 0;
    long cap_root_counter = 0;
    uint32_t cur_root_idx = 0;
    bool cur_full = true;

    explicit Impl(const MCTSConfig& c, AZEvaluator* e, AZEvaluator* eb)
        : cfg(c), eval(e), eval_a(e), eval_b(eb), rng(c.selfplay_rng_seed) {}

    void begin_match(uint32_t cap_seed) {
        game_samples.clear();
        cap_seed_ = cap_seed;
        cap_root_counter = 0;
    }

    void end_game() {
        // Called after a game's samples are priced+flushed. Any sideboard samples
        // recorded before the next game starts accumulate into the now-empty
        // buffer and get priced by the next game's z.
        game_samples.clear();
    }

    // Exploration-clock verdict for the real decision being finalized: sample
    // the real action from the visit distribution (true) or play argmax.
    // Self-play only; `sideboard_root` roots are turn 0 (eps = 1, coin
    // short-circuits, root index unused).
    bool explore_this_root(bool sideboard_root) const {
        if (!cfg.selfplay) return false;
        const double eps = explore_prob(obs_turn(root_obs.data(), sideboard_root), cfg);
        return explore_coin(cap_seed_, sideboard_root ? 0u : cur_root_idx, eps);
    }

    // ── evaluator wrappers (uniform-safe) ──────────────────────────────────
    AZEvalResultD eval_one(const float* o, int nc) {
        if (eval) return eval->evaluate_double(o, nc);
        AZEvalResultD r;
        r.priors.assign(static_cast<size_t>(nc), 1.0 / static_cast<double>(nc));
        r.value = 0.0;
        return r;
    }

    std::vector<double> eval_priors(const float* o, int nc) {
        return eval_one(o, nc).priors;
    }

    // ── tree bookkeeping ───────────────────────────────────────────────────
    Node* make_node(int nc, std::vector<double> priors, bool is_a) {
        pool.push_back(std::make_unique<Node>(nc, std::move(priors), is_a));
        return pool.back().get();
    }

    // Fill n->rep from the node's creation obs and fold P onto the
    // representatives (ascending member index, float64) — mirror of
    // _Node.set_rep. MUST run exactly once after EVERY assignment of raw
    // priors to n->P (fresh creation, boundary-root adoption, batched
    // flush_pending), never twice: adoption always overwrites P with fresh
    // raw priors first, so double-folding is structurally impossible.
    void init_merge(Node* n, const float* o) {
        if (!cfg.merge_dupes) {
            n->rep.clear();
            return;
        }
        menu_merge_reps(o, n->num_choices, n->rep);
        for (int i = 1; i < n->num_choices; i++) {
            const int r = n->rep[static_cast<size_t>(i)];
            if (r != i) {
                n->P[static_cast<size_t>(r)] += n->P[static_cast<size_t>(i)];
                n->P[static_cast<size_t>(i)] = 0.0;
            }
        }
    }

#ifndef NDEBUG
    // Debug builds only (see Node::dbg_menu): record the per-action metadata
    // blocks of `o`'s menu into `out`.
    static void capture_menu(const float* o, int nc, std::vector<float>& out) {
        out.resize(static_cast<size_t>(nc) * 6);
        for (int i = 0; i < nc; i++)
            for (int b = 0; b < 6; b++)
                out[static_cast<size_t>(i) * 6 + static_cast<size_t>(b)] =
                    o[STATE_SIZE + b * MAX_ACTIONS + i];
    }

    // Debug builds only: the compact state fingerprint printed by the
    // world-consistency dump — the 36-float header plus both library counts.
    static void capture_state(const float* o, std::vector<float>& out) {
        out.assign(o, o + STATE_HEADER_SIZE);
        out.push_back(o[LIBRARY_CTX_START]);
        out.push_back(o[LIBRARY_CTX_START + 1]);
    }

    static std::string fmt_state(const std::vector<float>& s) {
        if (s.size() < STATE_HEADER_SIZE + 2u) return "  (no state capture)\n";
        auto r = [&](int i, double scale) {
            return std::to_string(static_cast<int>(std::lround(
                static_cast<double>(s[static_cast<size_t>(i)]) * scale)));
        };
        int step = 0;
        for (int i = 0; i < STEP_ONEHOT_SIZE; i++)
            if (s[static_cast<size_t>(2 * PLAYER_BLOCK_SIZE + i)] > 0.5f) step = i;
        std::string mana;
        for (int i = 0; i < 6; i++) mana += r(3 + i, 10) + (i < 5 ? "/" : "");
        return "  self: life=" + r(0, 20) + " hand=" + r(1, 10) +
               " mana(WUBRGC)=" + mana + " lib=" + r(STATE_HEADER_SIZE, 60) +
               " | opp: life=" + r(PLAYER_BLOCK_SIZE, 20) +
               " hand=" + r(PLAYER_BLOCK_SIZE + 1, 10) +
               " lib=" + r(STATE_HEADER_SIZE + 1, 60) +
               " | step=" + std::to_string(step) +
               " is_active=" + r(2 * PLAYER_BLOCK_SIZE + STEP_ONEHOT_SIZE, 1) +
               " self_is_a=" + r(2 * PLAYER_BLOCK_SIZE + STEP_ONEHOT_SIZE + 1, 1) +
               " stack=" + std::to_string(
                   s[static_cast<size_t>(2 * PLAYER_BLOCK_SIZE + STEP_ONEHOT_SIZE + 2)]) +
               "\n";
    }

    // Undo the env.py normalizations (see obs_builder.cpp's per-block contract)
    // so the dump prints raw enum/int values.
    static std::string fmt_menu(const std::vector<float>& m) {
        auto denorm = [](float v, double scale, int bias) {
            return static_cast<int>(std::lround(static_cast<double>(v) * scale)) - bias;
        };
        std::string s;
        for (size_t i = 0; i * 6 < m.size(); i++) {
            s += "  [" + std::to_string(i) + "] cat=" +
                 std::to_string(denorm(m[i * 6 + 0], ACTION_CATEGORY_MAX, 0)) +
                 " id=" + std::to_string(denorm(m[i * 6 + 1], N_CARD_TYPES, 0)) +
                 " ctrl=" + std::to_string(static_cast<int>(m[i * 6 + 2])) +
                 " zone=" + std::to_string(denorm(m[i * 6 + 3], static_cast<int>(REF_PLAYER_OPP), 0)) +
                 " ref=" + std::to_string(denorm(m[i * 6 + 4], N_ENTITY_REF_SLOTS, 1)) +
                 " ord=" + std::to_string(denorm(m[i * 6 + 5], OPTION_ORDINAL_MAX + 1, 1)) +
                 "\n";
        }
        return s;
    }
#endif  // !NDEBUG

    void backup(const std::vector<PathEntry>& p, double leaf_value, bool leaf_seat_is_a) {
        for (const PathEntry& e : p) {
            e.node->N[static_cast<size_t>(e.action)] += 1;
            e.node->W[static_cast<size_t>(e.action)] +=
                (e.node->self_is_a == leaf_seat_is_a) ? leaf_value : -leaf_value;
        }
    }

    // Virtual loss (batch>1): make each action along the path look like a loss
    // for that node's OWN mover (W += -1 in the node's perspective, N += 1) so
    // concurrent collection diversifies; undone in flush before the real backup.
    void apply_virtual_loss(const std::vector<PathEntry>& p) {
        for (const PathEntry& e : p) {
            e.node->N[static_cast<size_t>(e.action)] += 1;
            e.node->W[static_cast<size_t>(e.action)] += -1.0;
        }
    }
    void remove_virtual_loss(const std::vector<PathEntry>& p) {
        for (const PathEntry& e : p) {
            e.node->N[static_cast<size_t>(e.action)] -= 1;
            e.node->W[static_cast<size_t>(e.action)] -= -1.0;
        }
    }

    // ── world / sim lifecycle (in-game tree search) ────────────────────────
    // Is the current searched ROOT the learner's seat? Always true outside
    // the opponent-pool mode (cfg.net_seat == 0); an opponent root searches
    // noise-free, plays argmax, and records no sample.
    bool learner_root() const {
        return cfg.net_seat == 0 || (root_is_a == (cfg.net_seat == 1));
    }

    void begin_world(int w) {
        cur_world_seed =
            cfg.world_seed_base +
            100003u * static_cast<uint32_t>(this_root) +
            static_cast<uint32_t>(w);
        // Root Dirichlet noise, redrawn per world over the shared base priors —
        // mcts.py: priors = (1-eps)*root_priors + eps*dirichlet(alpha*[1]*nc).
        // eps=0 (parity default) reuses the base priors verbatim. A standard
        // Dirichlet is gamma(alpha) per component normalized by the sum, drawn
        // from the per-run RNG (no cross-language parity required). A playout-
        // cap FAST root mixes no noise (it plays the game; its pi is never a
        // policy target), mirroring the Python fast branch's eps=0. An
        // opponent-pool root mixes none either (_opp_net_action's eps=0).
        if (cfg.noise_eps > 0.0 && cur_full && learner_root()) {
            std::gamma_distribution<double> gamma(cfg.noise_alpha, 1.0);
            std::vector<double> noise(static_cast<size_t>(root_n));
            double sum = 0.0;
            for (int i = 0; i < root_n; i++) {
                double g = gamma(rng);
                noise[static_cast<size_t>(i)] = g;
                sum += g;
            }
            std::vector<double> priors(static_cast<size_t>(root_n));
            for (int i = 0; i < root_n; i++) {
                double d = sum > 0.0 ? noise[static_cast<size_t>(i)] / sum
                                     : 1.0 / static_cast<double>(root_n);
                priors[static_cast<size_t>(i)] =
                    (1.0 - cfg.noise_eps) * root_priors[static_cast<size_t>(i)] +
                    cfg.noise_eps * d;
            }
            cur_root = make_node(root_n, priors, root_is_a);
        } else {
            cur_root = make_node(root_n, root_priors, root_is_a);
        }
        // The branch above just assigned raw (clean or noise-mixed) priors to
        // cur_root->P, so the merge fold applies exactly once.
        init_merge(cur_root, root_obs.data());
        world_roots[static_cast<size_t>(w)] = cur_root;
#ifndef NDEBUG
        capture_menu(root_obs.data(), root_n, cur_root->dbg_menu);
        capture_state(root_obs.data(), cur_root->dbg_state);
#endif
    }

    int start_descent() {
        path.clear();
        int a = cur_root->select(cfg.c_puct);
        path.push_back({cur_root, a});
        sim_steps += 1;
        return a;
    }

    // Enter worlds from cur_world onward: allocate/adopt each world's root,
    // start its first sim if it has top-up budget, else fold its (inherited)
    // visits into the totals and move on. Returns the first sim's action, or
    // -1 when no remaining world has budget (caller finalizes). Mirrors
    // run_search's `for _ in range(budgets[w])` loop, where a fully-inherited
    // world runs zero sims but still reports cumulative visits.
    int start_next_world_sim() {
        while (cur_world < cur_worlds) {
            begin_world(cur_world);
            cur_sim = 0;
            if (cur_budgets[static_cast<size_t>(cur_world)] > 0) {
                determinize_hidden_state(cur_world_seed);
#ifndef NDEBUG
                std::fprintf(stderr, "[sim] root=%d world=%d sim=%d\n", this_root,
                             cur_world, cur_sim);
#endif
                int a = start_descent();
                phase = DESCENDING;
                return a;
            }
            accumulate_world();
            cur_world += 1;
        }
        return -1;
    }

    void finish_sim() {
        sims_run += 1;
        search_request_restore(SEARCH_SLOT);
        phase = AWAITING_ROOT;
    }

    void accumulate_world() {
        for (int i = 0; i < root_n; i++) {
            visit_totals[static_cast<size_t>(i)] += cur_root->N[static_cast<size_t>(i)];
            w_totals[static_cast<size_t>(i)] += cur_root->W[static_cast<size_t>(i)];
        }
        double wsum = 0.0;
        for (double w : cur_root->W) wsum += w;
        value_acc += wsum;
    }

    // ── cross-world scheduler (cross_active searches only) ─────────────────
    // Build ALL world roots up front (begin_world in ascending world order —
    // the same order the sequential loop draws each world's noise in, so the
    // rng stream is unchanged; nothing during a sim draws from it), record
    // each world's determinize seed, then start the first budgeted world's
    // sim. Returns the first descent action, or -1 when no world has top-up
    // budget (caller finalizes; inherited worlds are accumulated there).
    int start_cross_search() {
        world_seeds.assign(static_cast<size_t>(cur_worlds), 0u);
        world_pending.assign(static_cast<size_t>(cur_worlds), 0);
        for (int w = 0; w < cur_worlds; w++) {
            begin_world(w);
            world_seeds[static_cast<size_t>(w)] = cur_world_seed;
        }
        cur_world = -1;
        return schedule_next_cross();
    }

    // Round-robin: the next world after cur_world (wrapping, cur_world itself
    // included last) with remaining budget runs one sim. Budgets count down at
    // sim START (a deferred leaf's backup is bookkeeping, not budget). The
    // flush-before-reentry rule is what makes deferral loss-free: a world with
    // an in-flight PendingLeaf is flushed before its own next descent selects,
    // so every tree sees exactly the backups the sequential loop would have
    // applied. Returns the descent's first action, or -1 when every budget is
    // spent (caller finalizes; the final flush happens there).
    int schedule_next_cross() {
        for (int k = 1; k <= cur_worlds; k++) {
            int w = (cur_world + k + cur_worlds) % cur_worlds;
            if (cur_budgets[static_cast<size_t>(w)] <= 0) continue;
            if (world_pending[static_cast<size_t>(w)]) flush_pending();
            cur_budgets[static_cast<size_t>(w)] -= 1;
            cur_world = w;
            cur_sim = sims_per_world - cur_budgets[static_cast<size_t>(w)] - 1;
            cur_root = world_roots[static_cast<size_t>(w)];
            cur_world_seed = world_seeds[static_cast<size_t>(w)];
            determinize_hidden_state(cur_world_seed);
#ifndef NDEBUG
            std::fprintf(stderr, "[sim] root=%d world=%d sim=%d (cross)\n",
                         this_root, cur_world, cur_sim);
#endif
            int a = start_descent();
            phase = DESCENDING;
            return a;
        }
        return -1;
    }

    void flush_pending() {
        if (pending.empty()) return;
        const int k = static_cast<int>(pending.size());
        std::vector<AZEvalResultD> res;
        if (eval) {
            std::vector<float> big(static_cast<size_t>(k) * static_cast<size_t>(ACTOR_OBS_SIZE));
            std::vector<int> ncs(static_cast<size_t>(k));
            for (int i = 0; i < k; i++) {
                std::copy(pending[static_cast<size_t>(i)].obs.begin(),
                          pending[static_cast<size_t>(i)].obs.end(),
                          big.begin() + static_cast<long>(i) * ACTOR_OBS_SIZE);
                ncs[static_cast<size_t>(i)] = pending[static_cast<size_t>(i)].num_choices;
            }
            res = eval->evaluate_double_batch(big.data(), ncs);
        } else {
            res.reserve(static_cast<size_t>(k));
            for (int i = 0; i < k; i++) {
                PendingLeaf& pl = pending[static_cast<size_t>(i)];
                res.push_back(eval_one(pl.obs.data(), pl.num_choices));
            }
        }
        for (int i = 0; i < k; i++) {
            PendingLeaf& pl = pending[static_cast<size_t>(i)];
            // Cross-world deferral applies no virtual loss (one leaf per world,
            // flushed before that world's own next descent), so there is none
            // to remove.
            if (cfg.batch > 1) remove_virtual_loss(pl.path);
            bool is_a = pl.obs[SELF_IS_A_IDX] > 0.5f;
            // The find() guard serves two cases: under batch>1 virtual loss,
            // two same-world descents can defer the same unexpanded leaf (the
            // second result only backs up); under cross-world, a DEPTH-CAP
            // deferral targets an already-existing child (value-only backup,
            // mirrors the sequential depth-cap eval).
            if (pl.parent->children.find(pl.action) == pl.parent->children.end()) {
                pl.parent->children[pl.action] =
                    make_node(pl.num_choices, res[static_cast<size_t>(i)].priors, is_a);
                init_merge(pl.parent->children[pl.action], pl.obs.data());
#ifndef NDEBUG
                capture_menu(pl.obs.data(), pl.num_choices,
                             pl.parent->children[pl.action]->dbg_menu);
                capture_state(pl.obs.data(), pl.parent->children[pl.action]->dbg_state);
#endif
            }
            backup(pl.path, res[static_cast<size_t>(i)].value, is_a);
        }
        pending.clear();
        std::fill(world_pending.begin(), world_pending.end(), 0);
    }

    // ── sideboard plan-search helpers (mirror mcts.py's shared helpers) ────
    // mcts.py::sb_pick_descriptor read straight from an obs: appended to `out`
    // when the action is a sideboard pick (IN/OUT; Done carries none).
    static void note_pick_from_obs(const float* o, int a,
                                   std::vector<std::array<int16_t, 3>>& out) {
        int cat = static_cast<int>(std::lround(
            static_cast<double>(o[STATE_SIZE + 0 * MAX_ACTIONS + a]) *
            ACTION_CATEGORY_MAX));
        if (cat != static_cast<int>(ActionCategory::SIDEBOARD_IN) &&
            cat != static_cast<int>(ActionCategory::SIDEBOARD_OUT))
            return;
        int16_t cid = static_cast<int16_t>(std::lround(
            static_cast<double>(o[STATE_SIZE + 1 * MAX_ACTIONS + a]) * N_CARD_TYPES));
        out.push_back({static_cast<int16_t>(cat), cid,
                       static_cast<int16_t>(o[SELF_IS_A_IDX] > 0.5f ? 1 : 0)});
    }

    // mcts.py::rollout_memo_eligible — a (card, seat) appearing in BOTH pick
    // directions (only reachable via the forced OUT of a stranded IN, which may
    // cut a name that was just sided in) diverges the one-shot direction locks
    // without changing the deck, so such paths are never memoized.
    static bool memo_eligible(const std::vector<std::array<int16_t, 3>>& picks) {
        for (size_t i = 0; i < picks.size(); i++)
            for (size_t j = 0; j < picks.size(); j++)
                if (picks[i][0] != picks[j][0] && picks[i][1] == picks[j][1] &&
                    picks[i][2] == picks[j][2])
                    return false;
        return true;
    }

    // mcts.py's _plan_softmax: float64 softmax with max-subtraction — the pi
    // target arithmetic, bit-mirrored.
    static std::vector<double> plan_softmax(const std::vector<double>& q) {
        double mx = -std::numeric_limits<double>::infinity();
        for (double v : q) mx = std::max(mx, v);
        std::vector<double> e(q.size());
        double sum = 0.0;
        for (size_t i = 0; i < q.size(); i++) {
            e[i] = std::exp((q[i] - mx) / kSbPiTau);
            sum += e[i];
        }
        for (double& v : e) v /= sum;
        return e;
    }

    // mcts.py's _second_best: first-max argmax excluding the first-max argmax.
    static int second_best(const std::vector<double>& priors, int nc) {
        int best = 0;
        for (int i = 1; i < nc; i++)
            if (priors[static_cast<size_t>(i)] > priors[static_cast<size_t>(best)])
                best = i;
        int second = -1;
        for (int i = 0; i < nc; i++) {
            if (i == best) continue;
            if (second < 0 ||
                priors[static_cast<size_t>(i)] > priors[static_cast<size_t>(second)])
                second = i;
        }
        return second < 0 ? best : second;
    }

    // ── leaf-rollout helpers ───────────────────────────────────────────────
    // First-max argmax over float64 priors (numpy argmax tie-break) — the raw-
    // policy action, shared by the fallback path and the rollout playout.
    static int argmax_priors(const std::vector<double>& priors, int nc) {
        int best = 0;
        for (int i = 1; i < nc; i++)
            if (priors[static_cast<size_t>(i)] > priors[static_cast<size_t>(best)]) best = i;
        return best;
    }

    // True when the CURRENT engine state has reached the rollout horizon
    // (mirrors mcts.py::_rollout_stop, which reads the same fields from the
    // obs: is_sideboard_phase serializes `sideboard_phase`, turn serializes
    // `cur_game.turn`). The sideboard gate exempts a sideboard-root sim's
    // sideboard/mulligan prefix, where the turn still belongs to the ENDED game.
    bool rollout_stop_here() const {
        return !sideboard_phase &&
               static_cast<int>(cur_game.turn) >= cur_rollout_anchor + cur_rollout_turns;
    }

    // One ROLLOUT-phase provider call (mirrors one mcts.py::_rollout loop
    // iteration): the engine sits at the state reached by the previous rollout
    // action. Evaluate it once; at the horizon / step cap back its value up
    // (the state's own mover perspective), else play its argmax action.
    // Terminals never reach here — they fire on_game_end. The tree `path` is
    // NOT extended by rollout steps.
    int rollout_step(const float* o, int nc) {
        rollout_steps += 1;
        AZEvalResultD r = eval_one(o, nc);
        if (rollout_steps >= cur_rollout_cap || rollout_stop_here()) {
            bool seat = o[SELF_IS_A_IDX] > 0.5f;
            backup(path, r.value, seat);
            finish_sim();
            return 0;
        }
        sim_steps += 1;
        return argmax_priors(r.priors, nc);
    }

    // ── plan-search lifecycle (sideboard roots; mirrors run_plan_search) ───
    PlanKey plan_key(int w, const Plan& p) const {
        return {cfg.world_seed_base +
                    100003u * static_cast<uint32_t>(sb_seed_root) +
                    static_cast<uint32_t>(w),
                p.multiset};
    }

    uint32_t plan_world_seed(int w) const {
        return cfg.world_seed_base +
               100003u * static_cast<uint32_t>(sb_seed_root) +
               static_cast<uint32_t>(w);
    }

    // Begin generating one plan. The engine sits at the (restored) root with
    // the snapshot live; determinize world 0 and return the first pick as a
    // sim step. Mirrors _run_plan's generation entry (restore+determinize
    // happened via the cooperative unwind / snapshot save).
    int begin_plan(int first_action, int variant) {
        cur_plan = Plan{};
        cur_plan.first_action = first_action;
        cur_plan.variant = variant;
        cur_plan.pick_actions.push_back(first_action);
        note_pick_from_obs(root_obs.data(), first_action, cur_plan.picks);
        cur_plan_memoable = false;
        plan_world = 0;
        plan_dev = 0;
        determinize_hidden_state(plan_world_seed(0));
        phase = PLAN_GEN;
        sim_steps += 1;
        return first_action;
    }

    // One PLAN_GEN provider call: the state after the previous pick. While the
    // mover's sideboard run continues, choose (and record) the next pick;
    // when it ends, this state is the HANDOFF — complete the plan's identity
    // and start pricing world 0 (mirrors _generate_plan's loop condition).
    int plan_gen_step(const float* o, int nc) {
        bool seat = o[SELF_IS_A_IDX] > 0.5f;
        if (sideboard_phase && seat == root_is_a) {
            if (static_cast<int>(cur_plan.pick_actions.size()) > kPlanPickCap)
                fatal_error("az_mcts: plan-search pick sequence exceeded "
                            "kPlanPickCap — sideboard menu loop did not "
                            "terminate");
            AZEvalResultD r = eval_one(o, nc);
            plan_dev += 1;
            // Dead-card rules: a completion never sides in a table-dead card
            // (Done/OUT are never masked, so an argmax target remains).
            // MIRRORED in mcts.py _generate_plan.
            sb_dead_mask(o, nc, gen_dead);
            int n_live = nc;
            for (int i = 0; i < nc; i++)
                if (gen_dead[static_cast<size_t>(i)]) {
                    r.priors[static_cast<size_t>(i)] = 0.0;
                    n_live -= 1;
                }
            int a;
            if (cur_plan.variant > 0 && plan_dev == cur_plan.variant &&
                nc >= 2 && n_live >= 2)
                a = second_best(r.priors, nc);
            else
                a = argmax_priors(r.priors, nc);
            note_pick_from_obs(o, a, cur_plan.picks);
            cur_plan.pick_actions.push_back(a);
            sim_steps += 1;
            return a;
        }
        // Handoff: the plan's picks are complete — fix its identity, then
        // price world 0 from this very state (memo first).
        cur_plan.multiset = sb_picks;
        cur_plan.multiset.insert(cur_plan.multiset.end(), cur_plan.picks.begin(),
                                 cur_plan.picks.end());
        std::sort(cur_plan.multiset.begin(), cur_plan.multiset.end());
        cur_plan_memoable = memo_eligible(cur_plan.multiset);
        cur_plan.values.assign(static_cast<size_t>(cur_worlds), 0.0);
        if (cur_plan_memoable) {
            auto hit = plan_memo.find(plan_key(0, cur_plan));
            if (hit != plan_memo.end()) {
                memo_hits += 1;
                return finish_plan_world(hit->second, /*from_memo=*/true);
            }
        }
        rollout_steps = 0;
        phase = PLAN_ROLLOUT;
        return plan_rollout_step(o, nc);
    }

    // One PLAN_REPLAY provider call: force the recorded pick on world w; after
    // the last pick the state in hand is the handoff — start the rollout
    // (this world's memo was checked before the replay began).
    int plan_replay_step(const float* o, int nc) {
        if (plan_replay_pos < cur_plan.pick_actions.size()) {
            int a = cur_plan.pick_actions[plan_replay_pos];
            if (a >= nc)
                fatal_error("az_mcts: plan-search world-consistency violation: "
                            "pick " + std::to_string(a) + " not replayable (" +
                            std::to_string(nc) + " choices, world " +
                            std::to_string(plan_world) + ")");
            plan_replay_pos += 1;
            sim_steps += 1;
            return a;
        }
        rollout_steps = 0;
        phase = PLAN_ROLLOUT;
        return plan_rollout_step(o, nc);
    }

    // One PLAN_ROLLOUT provider call (mirrors _eval_plan_world / _rollout):
    // evaluate the state; at the horizon / step cap record its value as this
    // world's plan value (ROOT perspective), else play its argmax action.
    // Terminals fire on_game_end instead.
    int plan_rollout_step(const float* o, int nc) {
        rollout_steps += 1;
        AZEvalResultD r = eval_one(o, nc);
        if (rollout_steps >= cur_rollout_cap || rollout_stop_here()) {
            bool seat = o[SELF_IS_A_IDX] > 0.5f;
            double v_root = (seat == root_is_a) ? r.value : -r.value;
            return finish_plan_world(v_root, /*from_memo=*/false);
        }
        sim_steps += 1;
        return argmax_priors(r.priors, nc);
    }

    // Record the plan's value on plan_world, store it in the memo, and latch a
    // restore; AWAITING_ROOT (plan_active) advances to the next world / plan.
    int finish_plan_world(double v_root, bool from_memo) {
        cur_plan.values[static_cast<size_t>(plan_world)] = v_root;
        if (cur_plan_memoable && !from_memo)
            plan_memo[plan_key(plan_world, cur_plan)] = v_root;
        search_request_restore(SEARCH_SLOT);
        phase = AWAITING_ROOT;
        return 0;
    }

    // AWAITING_ROOT (plan_active): the engine is back at the restored root.
    // Price the plan's remaining worlds (memo hits cost nothing), commit the
    // finished plan, and schedule the next one — or finalize.
    int advance_plan() {
        plan_world += 1;
        while (plan_world < cur_worlds) {
            if (cur_plan_memoable) {
                auto hit = plan_memo.find(plan_key(plan_world, cur_plan));
                if (hit != plan_memo.end()) {
                    memo_hits += 1;
                    cur_plan.values[static_cast<size_t>(plan_world)] = hit->second;
                    plan_world += 1;
                    continue;
                }
            }
            // Replay this plan's picks under this world's deal.
            determinize_hidden_state(plan_world_seed(plan_world));
            plan_replay_pos = 0;
            phase = PLAN_REPLAY;
            // The root query in hand after the restore+determinize IS the
            // state the first pick applies to.
            return plan_replay_step_root();
        }
        commit_plan();
        return next_plan_or_finalize();
    }

    // First replay action on a fresh world: no obs was built for the restored
    // root query (AWAITING_ROOT skips it), but the first pick needs no menu
    // read — the root-width tripwire already ran.
    int plan_replay_step_root() {
        int a = cur_plan.pick_actions[0];
        plan_replay_pos = 1;
        sim_steps += 1;
        return a;
    }

    void commit_plan() {
        bool novel = !(cur_plan_memoable &&
                       plan_evaluated.count(cur_plan.multiset) > 0);
        if (cur_plan_memoable) plan_evaluated.insert(cur_plan.multiset);
        if (cur_plan.variant > 0 && novel) novel_extras += 1;
        plans.push_back(cur_plan);
    }

    // Coverage ascending, then the deterministic extras schedule (mirrors
    // run_plan_search's candidate order). Returns the next plan's first
    // action, or the finalized real action.
    int next_plan_or_finalize() {
        // Rule-dead first picks are skipped: they keep q = -inf, so the
        // softmax gives them exactly zero policy mass (mirrors mcts.py).
        while (coverage_cursor < root_n) {
            int a = coverage_cursor++;
            if (!plan_dead[static_cast<size_t>(a)]) return begin_plan(a, 0);
        }
        if (plan_ranked.empty()) {
            // Rank the LIVE first picks by coverage Q, descending, stable
            // (numpy argsort(-q[live], kind="stable") over live indices).
            std::vector<double> cov_q(static_cast<size_t>(root_n), 0.0);
            for (const Plan& p : plans)
                if (p.variant == 0) {
                    double v = 0.0;
                    for (double x : p.values) v += x;
                    cov_q[static_cast<size_t>(p.first_action)] =
                        v / static_cast<double>(cur_worlds);
                }
            for (int i = 0; i < root_n; i++)
                if (!plan_dead[static_cast<size_t>(i)])
                    plan_ranked.push_back(i);
            std::stable_sort(plan_ranked.begin(), plan_ranked.end(),
                             [&](int a, int b) {
                                 return cov_q[static_cast<size_t>(a)] >
                                        cov_q[static_cast<size_t>(b)];
                             });
        }
        const int branches = cfg.sb_branches >= 0 ? cfg.sb_branches
                                                  : kDefaultSbBranches;
        const int max_attempts = std::max(0, branches) * kExtrasAttemptFactor;
        // Extras cycle the live actions only; branch/variant arithmetic is
        // over n_live (n_live >= 1: Done is never dead). MIRRORED in mcts.py.
        const int n_live = static_cast<int>(plan_ranked.size());
        if (novel_extras < branches && extra_attempt < max_attempts) {
            int branch = plan_ranked[static_cast<size_t>(extra_attempt % n_live)];
            int variant = 1 + extra_attempt / n_live;
            extra_attempt += 1;
            return begin_plan(branch, variant);
        }
        return finalize_plan_search();
    }

    // Prior-rollout sideboard decision (cfg.sb_prior_mode): no plan search —
    // one net eval, the dead-card mask, the behavior policy
    //   b = (1-eps) * softmax(log p / temp) + eps * uniform(live),
    // a sampled pick (argmax on the opponent seat / without --selfplay), and
    // a ONE-HOT training row priced by the realized next-game z. The p/b
    // arithmetic MIRRORS az_selfplay.sb_prior_policy in float64 (only the
    // sample draw is backend-native). The dump payload is (q=p, pi=b), both
    // deterministic, via the ordinary plan_root path.
    int sb_prior_pick(const float* o, int nc) {
        sim_steps = 0;
        memo_hits = 0;
        AZEvalResultD r = eval_one(o, nc);
        sb_dead_mask(o, nc, gen_dead);
        std::vector<double> p(static_cast<size_t>(nc), 0.0);
        double total = 0.0;
        int n_live = 0;
        for (int i = 0; i < nc; i++) {
            if (!gen_dead[static_cast<size_t>(i)]) {
                p[static_cast<size_t>(i)] = r.priors[static_cast<size_t>(i)];
                total += p[static_cast<size_t>(i)];
                n_live += 1;
            }
        }
        if (total <= 0.0) {
            // Degenerate masked prior (all mass on dead actions): uniform live.
            total = 0.0;
            for (int i = 0; i < nc; i++) {
                p[static_cast<size_t>(i)] =
                    gen_dead[static_cast<size_t>(i)] ? 0.0 : 1.0;
                total += p[static_cast<size_t>(i)];
            }
        }
        for (double& v : p) v /= total;
        const double t = std::max(cfg.sb_explore_temp, 1e-3);
        double zmax = -std::numeric_limits<double>::infinity();
        std::vector<double> zv(static_cast<size_t>(nc),
                               -std::numeric_limits<double>::infinity());
        for (int i = 0; i < nc; i++)
            if (p[static_cast<size_t>(i)] > 0.0) {
                zv[static_cast<size_t>(i)] =
                    std::log(p[static_cast<size_t>(i)]) / t;
                zmax = std::max(zmax, zv[static_cast<size_t>(i)]);
            }
        std::vector<double> b(static_cast<size_t>(nc), 0.0);
        double bsum = 0.0;
        for (int i = 0; i < nc; i++)
            if (std::isfinite(zv[static_cast<size_t>(i)])) {
                b[static_cast<size_t>(i)] =
                    std::exp(zv[static_cast<size_t>(i)] - zmax);
                bsum += b[static_cast<size_t>(i)];
            }
        for (double& v : b) v /= bsum;
        const double eps =
            std::min(std::max(cfg.sb_explore_eps, 0.0), 1.0);
        if (eps > 0.0) {
            bsum = 0.0;
            for (int i = 0; i < nc; i++) {
                const double u = gen_dead[static_cast<size_t>(i)]
                                     ? 0.0
                                     : 1.0 / static_cast<double>(n_live);
                b[static_cast<size_t>(i)] =
                    (1.0 - eps) * b[static_cast<size_t>(i)] + eps * u;
                bsum += b[static_cast<size_t>(i)];
            }
            for (double& v : b) v /= bsum;
        }
        // greedy = masked-prior first-max argmax (np.where(live, p, -1.0)).
        int greedy = 0;
        double bestp = -2.0;
        for (int i = 0; i < nc; i++) {
            const double v =
                gen_dead[static_cast<size_t>(i)] ? -1.0 : p[static_cast<size_t>(i)];
            if (v > bestp) {
                bestp = v;
                greedy = i;
            }
        }

        SearchRootResult sr;
        sr.root_index = this_root;
        sr.num_choices = nc;
        sr.plan_root = true;
        sr.q = p;
        sr.pi = b;
        sr.visits.resize(static_cast<size_t>(nc));
        for (int i = 0; i < nc; i++)
            sr.visits[static_cast<size_t>(i)] =
                std::llround(b[static_cast<size_t>(i)] * 1e6);
        sr.root_value = r.value;
        sr.sims_run = 1;
        sr.sim_steps = 0;
        sr.memo_hits = 0;
        results.push_back(std::move(sr));

        int chosen = greedy;
        // Opponent-pool seat: argmax masked prior, no sample.
        if ((cfg.selfplay || cfg.record) && learner_root()) {
            if (cfg.selfplay) {
                // Prior mode samples at EVERY learner sb root (mirrors
                // _sb_prior_sample's unconditional rng.choice).
                std::discrete_distribution<int> dist(b.begin(), b.end());
                chosen = dist(rng);
            }
            SelfPlaySample s;
            s.obs = root_obs;
            s.pi.assign(static_cast<size_t>(MAX_ACTIONS), 0.0f);
            s.mask.assign(static_cast<size_t>(MAX_ACTIONS), 0);
            s.mover_is_a = root_is_a;
            s.is_sideboard = true;
            s.pi[static_cast<size_t>(chosen)] = 1.0f;
            for (int i = 0; i < nc; i++) s.mask[static_cast<size_t>(i)] = 1;
            s.q = static_cast<float>(r.value);
            s.explored = chosen != greedy;
            game_samples.push_back(std::move(s));
        }
        phase = IDLE;
        return chosen;
    }

    // Mirrors run_plan_search's tail: Q per first pick, pi = softmax(Q/tau),
    // the SearchRootResult, the self-play sample, and the chosen real action.
    int finalize_plan_search() {
        snapshot_release_all();

        std::vector<double> q(static_cast<size_t>(root_n),
                              -std::numeric_limits<double>::infinity());
        for (const Plan& p : plans) {
            double v = 0.0;
            for (double x : p.values) v += x;
            v /= static_cast<double>(cur_worlds);
            size_t fa = static_cast<size_t>(p.first_action);
            q[fa] = std::max(q[fa], v);
        }
        std::vector<double> pi = plan_softmax(q);
        // Rule-dead (uncovered) first picks: pi is exactly 0; sanitize their
        // -inf q so root_value / the sample's q stay finite (mirrors mcts.py).
        for (double& v : q)
            if (!std::isfinite(v)) v = 0.0;
        const int n_evals = static_cast<int>(plans.size()) * cur_worlds;
        double root_value = 0.0;
        for (int i = 0; i < root_n; i++)
            root_value += pi[static_cast<size_t>(i)] * q[static_cast<size_t>(i)];

        SearchRootResult sr;
        sr.root_index = this_root;
        sr.num_choices = root_n;
        sr.plan_root = true;
        sr.q = q;
        sr.pi = pi;
        // Fixed-point posterior for the shared int64 visits field (parity
        // compares the float q/pi arrays; this keeps summaries readable).
        sr.visits.resize(static_cast<size_t>(root_n));
        for (int i = 0; i < root_n; i++)
            sr.visits[static_cast<size_t>(i)] =
                std::llround(pi[static_cast<size_t>(i)] * 1e6);
        sr.root_value = root_value;
        sr.sims_run = n_evals;
        sr.sim_steps = sim_steps;
        sr.memo_hits = memo_hits;
        results.push_back(std::move(sr));

        int best = 0;
        for (int i = 1; i < root_n; i++)
            if (pi[static_cast<size_t>(i)] > pi[static_cast<size_t>(best)]) best = i;

        int chosen = best;
        // Opponent-pool sb plan root: no sample, no exploration — argmax pick
        // only (mirrors _opp_net_action's plan-search branch).
        if ((cfg.selfplay || cfg.record) && learner_root()) {
            SelfPlaySample s;
            s.obs = root_obs;
            s.pi.assign(static_cast<size_t>(MAX_ACTIONS), 0.0f);
            s.mask.assign(static_cast<size_t>(MAX_ACTIONS), 0);
            s.mover_is_a = root_is_a;
            s.is_sideboard = true;
            for (int i = 0; i < root_n; i++) {
                s.pi[static_cast<size_t>(i)] =
                    static_cast<float>(pi[static_cast<size_t>(i)]);
                s.mask[static_cast<size_t>(i)] = 1;
            }
            // A sideboard root is turn 0 of the upcoming game: the exploration
            // clock always samples here under --selfplay.
            if (explore_this_root(true)) {
                std::discrete_distribution<int> dist(pi.begin(), pi.end());
                chosen = dist(rng);
            }
            // TD bootstrap = the played first pick's plan Q (mirrors
            // az_selfplay.finalize_searched_sample over run_plan_search's q).
            s.q = static_cast<float>(q[static_cast<size_t>(chosen)]);
            s.explored = chosen != best;
            game_samples.push_back(std::move(s));
        }

        phase = IDLE;
        plan_active = false;
        // The boundary stays live past this real pick: latch the chosen
        // action's descriptor for the memo-key base (the memo itself carries).
        note_pick_from_obs(root_obs.data(), chosen, sb_picks);
        plans.clear();
        return chosen;
    }

    // ── phase handlers ─────────────────────────────────────────────────────
    int begin_or_fallback(const float* o, int nc) {
        // Per-seat evaluator selection (two-model gate/eval matches). The seat
        // to move at this REAL decision owns the net for everything the
        // decision spawns — the root priors, every sim leaf/rollout eval and
        // the fallback argmax — matching the Python gate, where each seat's
        // SearchController carries its own evaluator through its whole search.
        // Selected only here (a real decision at IDLE): sim steps and restores
        // re-enter through other phases and must keep the ROOT mover's net.
        if (eval_b) eval = (o[SELF_IS_A_IDX] > 0.5f) ? eval_a : eval_b;
        // Vs-scripted seat (mirrors _play_match's non-net_to_move branch): the
        // scripted seat's real decisions come from the oracle provider — no
        // search, no sample, no searched/fallback counters — but they DO latch
        // into a live sideboard boundary (the walk must replay the true action
        // sequence, whoever played it). Search simulations never reach here
        // (DESCENDING/ROLLOUT phases), so tree play stays net-both-seats
        // exactly like the Python reference.
        if (cfg.scripted_seat != 0 &&
            (o[SELF_IS_A_IDX] > 0.5f) == (cfg.scripted_seat == 1)) {
            int a = 0;
            if (nc > 1) {
                if (!scripted_provider)
                    fatal_error("az_mcts: scripted_seat is set but no scripted "
                                "provider was installed (set_scripted_provider)");
                a = scripted_provider(o, nc);
            }
            if (sb_active) note_pick_from_obs(o, a, sb_picks);
            return a;
        }
        bool searchable = search_loop_safe() && nc > 1;
        if (!searchable) {
            // A trivial / unsafe real decision: not stored, evaluator-argmax
            // fallback.
            int a = 0;
            if (nc > 1) a = argmax_priors(eval_priors(o, nc), nc);
            // A forced/fallback pick inside an active boundary (direction
            // locks can prune a sideboard menu to one entry) still changes the
            // configuration — latch its descriptor so the memo-key base stays
            // exact (mirrors the Python consumers latching stepped picks).
            if (sb_active) note_pick_from_obs(o, a, sb_picks);
            return a;
        }
        // BEGIN SEARCH (mirrors mcts.py::run_search setup). Capture the CLEAN root
        // obs (the state the net saw for base priors) before any determinize —
        // this is what a self-play sample stores.
        root_obs.assign(o, o + ACTOR_OBS_SIZE);
        this_root = root_counter++;
        root_n = nc;
        root_is_a = o[SELF_IS_A_IDX] > 0.5f;
        root_priors = eval_priors(o, nc);
        snapshot_save(SEARCH_SLOT);
        bool sb = sideboard_phase;
        root_is_sb = sb;

        if (sb) {
            // Prior-rollout mode (generation only): no plan search, no
            // boundary state — the REAL next game is the rollout.
            if (cfg.sb_prior_mode) {
                cur_full = true;
                plan_active = false;
                return sb_prior_pick(o, nc);
            }
            // ── sideboard PLAN search (mirrors mcts.run_plan_search) ────────
            // Boundary continue / start (mcts.py sb_root_key identity: seat +
            // upcoming game number). On continue the plan memo and the latched
            // real picks carry over; the world seeds stay pinned to the
            // boundary's FIRST searched root.
            int game = match_game_number + 1;
            if (!(sb_active && root_is_a == sb_seat_is_a && game == sb_game)) {
                plan_memo.clear();
                sb_picks.clear();
                sb_active = true;
                sb_seat_is_a = root_is_a;
                sb_game = game;
                sb_seed_root = this_root;
            }
            cur_full = true;   // sb plan roots are exempt from the playout cap
            cur_worlds = cfg.sb_worlds >= 0 ? cfg.sb_worlds : kDefaultSbWorlds;
            cur_worlds = std::max(1, cur_worlds);
            cur_rollout_turns = cfg.sb_rollout_turns >= 0 ? cfg.sb_rollout_turns
                                                          : kDefaultSbRolloutTurns;
            cur_rollout_anchor = 0;
            cur_rollout_cap = kRolloutStepsPerTurn * cur_rollout_turns;
            sim_steps = 0;
            memo_hits = 0;
            plan_active = true;
            cross_active = false;
            plans.clear();
            plan_evaluated.clear();   // per-root novelty (the memo is per-boundary)
            plan_ranked.clear();
            coverage_cursor = 0;
            extra_attempt = 0;
            novel_extras = 0;
            pool.clear();
            pending.clear();
            path.clear();
            // Dead-card rules mask for this root (sb_rules.h; mirrors
            // mcts.py's sb_dead_mask at the top of run_plan_search). Dead
            // first picks are skipped by coverage; Done is never dead, so at
            // least one live first pick always exists.
            sb_dead_mask(o, nc, plan_dead);
            while (coverage_cursor < root_n &&
                   plan_dead[static_cast<size_t>(coverage_cursor)])
                coverage_cursor++;
            int first = coverage_cursor++;
            return begin_plan(first, 0);
        }

        // ── in-game tree search (unchanged budget + iteration) ──────────────
        // An in-game real decision ends any live sideboard boundary.
        sb_active = false;
        plan_active = false;
        plan_memo.clear();
        sb_picks.clear();
        // Searched-root index — the key both hash coins share; the counter
        // advances only under self-play (the coins are never consulted
        // elsewhere: the driver forces frac=1.0 and sampling is off).
        cur_root_idx = static_cast<uint32_t>(cap_root_counter);
        if (cfg.selfplay) cap_root_counter++;
        // Playout-cap coin: a fast root searches cfg.fast_sims, mixes no
        // root noise, and its sample carries an all-zero pi (see finalize).
        cur_full = !cfg.selfplay ||
                   playout_cap_full(cap_seed_, cur_root_idx, cfg.full_search_frac);
        cur_sims = cur_full ? cfg.sims : cfg.fast_sims;
        cur_worlds = cfg.worlds;
        cur_max_depth = cfg.max_depth;
        cur_rollout_turns = cfg.rollout_turns;
        cur_rollout_anchor = static_cast<int>(cur_game.turn);
        cur_rollout_cap = kRolloutStepsPerTurn * cur_rollout_turns;
        visit_totals.assign(static_cast<size_t>(nc), 0);
        w_totals.assign(static_cast<size_t>(nc), 0.0);
        value_acc = 0.0;
        sims_run = 0;
        sim_steps = 0;
        memo_hits = 0;
        sims_per_world = std::max(1, cur_sims / std::max(1, cur_worlds));
        pool.clear();
        pending.clear();
        path.clear();
        world_roots.assign(static_cast<size_t>(cur_worlds), nullptr);
        cur_budgets.assign(static_cast<size_t>(cur_worlds), sims_per_world);
        cur_world = 0;
        cur_sim = 0;
        // Cross-world scheduling only when this search's budget has rollouts
        // OFF (same restriction as batch>1 — a deferred leaf cannot drive a
        // playout); a rollout-budget search under --cross-world runs the
        // unchanged sequential path below, with per-leaf immediate evals.
        cross_active = cfg.cross_world && cur_rollout_turns == 0;
        if (cross_active) {
            int a = start_cross_search();
            if (a < 0) return finalize();
            return a;
        }
        // Sim 0: the snapshot IS the current (clean) root state, so determinize
        // directly — no intervening restore (mcts.py restores before every sim,
        // but restore-immediately-after-snapshot is a no-op). A fully-inherited
        // search (every world's budget 0) finalizes immediately.
        int a = start_next_world_sim();
        if (a < 0) return finalize();
        return a;
    }

    int descend_step(const float* o, int nc) {
        // This call is the query produced by path.back().action (query_k, with
        // k = path.size()). Process it exactly as one _simulate iteration.
        PathEntry pe = path.back();
        Node* parent = pe.node;
        int paction = pe.action;
        auto it = parent->children.find(paction);
        if (it == parent->children.end()) {
            // New leaf. Rollouts force the immediate (batch=1) eval path —
            // a deferred PendingLeaf cannot drive a playout (cross_active is
            // never set on a rollout-budget search).
            bool leaf_is_a = o[SELF_IS_A_IDX] > 0.5f;
            if ((cfg.batch > 1 || cross_active) && cur_rollout_turns == 0) {
                PendingLeaf pl;
                pl.obs.assign(o, o + ACTOR_OBS_SIZE);
                pl.num_choices = nc;
                pl.path = path;
                pl.parent = parent;
                pl.action = paction;
                if (cross_active)
                    world_pending[static_cast<size_t>(cur_world)] = 1;
                else
                    apply_virtual_loss(path);
                pending.push_back(std::move(pl));
                finish_sim();
                return 0;
            }
            AZEvalResultD r = eval_one(o, nc);
            parent->children[paction] = make_node(nc, r.priors, leaf_is_a);
            init_merge(parent->children[paction], o);
#ifndef NDEBUG
            capture_menu(o, nc, parent->children[paction]->dbg_menu);
            capture_state(o, parent->children[paction]->dbg_state);
#endif
            // Leaf rollout (mirrors mcts.py::_rollout with the leaf as state 0):
            // unless the leaf already meets the horizon, play the raw policy
            // forward instead of backing up the leaf's value. The engine state
            // at this moment IS the leaf state, so check the stop condition on
            // the live globals (identical to Python's read of the leaf obs).
            if (cur_rollout_turns > 0 && !rollout_stop_here()) {
                rollout_steps = 0;
                phase = ROLLOUT;
                sim_steps += 1;
                return argmax_priors(r.priors, nc);
            }
            backup(path, r.value, leaf_is_a);
            finish_sim();
            return 0;
        }
        Node* child = it->second;
#ifndef NDEBUG
        // Debug builds: strengthened world-consistency check — same world + same
        // path must re-derive the same MENU, not merely the same count. Content
        // divergence (a card-list menu with different cards) precedes a count
        // mismatch by several plies, so compare the full metadata capture and
        // dump both menus decoded. Release builds keep only the O(1) count check
        // below (the cheap tripwire).
        std::vector<float> got;
        capture_menu(o, nc, got);
        if (child->num_choices != nc || child->dbg_menu != got) {
            std::string path_s;
            for (const PathEntry& e : path) path_s += std::to_string(e.action) + " ";
            // The stored menus of every node ALONG the path localize where the
            // states diverged invisibly (identical menus can mask different
            // hidden state, e.g. a scry prompt over a different card).
            std::string chain;
            for (size_t k = 0; k < path.size(); k++) {
                chain += "  path node " + std::to_string(k) + " (took action " +
                         std::to_string(path[k].action) + "):\n" +
                         fmt_state(path[k].node->dbg_state) +
                         fmt_menu(path[k].node->dbg_menu);
            }
            std::vector<float> got_state;
            capture_state(o, got_state);
            fatal_error("az_mcts: world-consistency violation: node expected " +
                        std::to_string(child->num_choices) + " choices, engine gave " +
                        std::to_string(nc) +
                        "\n  root#" + std::to_string(this_root) +
                        " cap_root=" + std::to_string(cap_root_counter) +
                        " world=" + std::to_string(cur_world) +
                        " (seed=" + std::to_string(cur_world_seed) + ")" +
                        " sim=" + std::to_string(cur_sim) +
                        " depth=" + std::to_string(path.size()) +
                        " sideboard=" + std::to_string(sideboard_phase ? 1 : 0) +
                        "\n  path actions: " + path_s +
                        "\n" + chain +
                        "  node menu (at creation):\n" + fmt_state(child->dbg_state) +
                        fmt_menu(child->dbg_menu) +
                        "  engine menu (now):\n" + fmt_state(got_state) + fmt_menu(got));
        }
#else
        if (child->num_choices != nc) {
            fatal_error("az_mcts: world-consistency violation: node expected " +
                        std::to_string(child->num_choices) + " choices, engine gave " +
                        std::to_string(nc));
        }
#endif
        // Descend into the existing child (node = child). Depth cap: value-only.
        if (static_cast<int>(path.size()) >= cur_max_depth) {
            // Cross-world defers the depth-cap eval too: the child exists, so
            // flush's find() guard skips node creation and only backs the
            // batched value up (obs seat == child->self_is_a by the
            // world-consistency check above). batch>1 keeps its documented
            // immediate eval here.
            if (cross_active) {
                PendingLeaf pl;
                pl.obs.assign(o, o + ACTOR_OBS_SIZE);
                pl.num_choices = nc;
                pl.path = path;
                pl.parent = parent;
                pl.action = paction;
                world_pending[static_cast<size_t>(cur_world)] = 1;
                pending.push_back(std::move(pl));
                finish_sim();
                return 0;
            }
            AZEvalResultD r = eval_one(o, nc);
            backup(path, r.value, child->self_is_a);
            finish_sim();
            return 0;
        }
        int a = child->select(cfg.c_puct);
        path.push_back({child, a});
        sim_steps += 1;
        return a;
    }

    int advance_after_restore() {
        // Plan search: the finished (or memo-priced) world's restore landed us
        // back at the root — advance the plan/world schedule.
        if (plan_active) return advance_plan();
        // Cross-world scheduling: the finished sim's budget was decremented at
        // its start, so advancing is just picking the next budgeted world
        // round-robin (finalize when none remains — it runs the final flush
        // and accumulates every world).
        if (cross_active) {
            int a = schedule_next_cross();
            if (a < 0) return finalize();
            return a;
        }
        // We are back at the restored true root. The just-finished sim was
        // (cur_world, cur_sim); advance to the next sim / world / finalize.
        cur_sim += 1;
        if (cur_sim >= cur_budgets[static_cast<size_t>(cur_world)]) {
            if (cfg.batch > 1) flush_pending();
            accumulate_world();
            cur_world += 1;
            int a = start_next_world_sim();  // skips fully-inherited worlds
            if (a < 0) return finalize();
            return a;
        }
        if (cfg.batch > 1 && static_cast<int>(pending.size()) >= cfg.batch) flush_pending();
        determinize_hidden_state(cur_world_seed);
#ifndef NDEBUG
        // Sim delimiter for reading interleaved sim narratives in debug logs.
        std::fprintf(stderr, "[sim] root=%d world=%d sim=%d\n", this_root, cur_world, cur_sim);
#endif
        int a = start_descent();
        phase = DESCENDING;
        return a;
    }

    int finalize() {
        // Back at the true root with all sims done. Drop snapshots BEFORE the
        // real step (a real game-end with a live snapshot would park in the
        // intercept). mcts.py: env.restore(0); env.release() — the restore already
        // happened via the last sim's cooperative unwind, so only release remains.
        if (cfg.batch > 1 || cross_active) flush_pending();
        if (cross_active) {
            // The sequential loop accumulates each world as its budget runs
            // out; cross mode defers ALL accumulation to here (after the final
            // flush), in ascending world order — the same summation order, so
            // value_acc's float result is unchanged. Zero-budget (fully
            // inherited) worlds fold in their cumulative visits here too.
            for (int w = 0; w < cur_worlds; w++) {
                cur_root = world_roots[static_cast<size_t>(w)];
                accumulate_world();
            }
        }
        snapshot_release_all();

        SearchRootResult sr;
        sr.root_index = this_root;
        sr.num_choices = root_n;
        sr.visits = visit_totals;
        int64_t total = 0;
        for (int64_t v : visit_totals) total += v;
        sr.root_value = total > 0 ? value_acc / static_cast<double>(total) : 0.0;
        sr.sims_run = sims_run;
        sr.sim_steps = sim_steps;
        sr.memo_hits = memo_hits;
        results.push_back(std::move(sr));

        int best = 0;
        for (int i = 1; i < root_n; i++)
            if (visit_totals[static_cast<size_t>(i)] > visit_totals[static_cast<size_t>(best)])
                best = i;

        // Self-play: store the searched sample and pick the real action per the
        // exploration clock (sample-from-visits when the root's coin says
        // explore, argmax otherwise). Parity/eval mode stores nothing and
        // always plays argmax; --record stores the sample but keeps the
        // eval-mode argmax pick. An opponent-pool root (selfplay with net_seat
        // set, opponent to move) stores nothing and plays argmax.
        int chosen = best;
        if ((cfg.selfplay || cfg.record) && learner_root()) {
            SelfPlaySample s;
            s.obs = root_obs;                                    // clean root obs
            s.pi.assign(static_cast<size_t>(MAX_ACTIONS), 0.0f);
            s.mask.assign(static_cast<size_t>(MAX_ACTIONS), 0);
            s.mover_is_a = root_is_a;
            s.is_sideboard = root_is_sb;
            for (int i = 0; i < root_n; i++) {
                // A playout-cap fast root records NO policy target: pi stays
                // all-zero (the schema's "no pi" marker — trainer and every
                // pi consumer key off pi.sum() > 0); q/explored stay real so
                // the value/TD side records from this row too.
                if (cur_full)
                    s.pi[static_cast<size_t>(i)] =
                        total > 0
                            ? static_cast<float>(
                                  static_cast<double>(
                                      visit_totals[static_cast<size_t>(i)]) /
                                  static_cast<double>(total))
                            : 0.0f;
                s.mask[static_cast<size_t>(i)] = 1;
            }

            if (total > 0 && explore_this_root(false)) {
                std::discrete_distribution<int> dist(
                    visit_totals.begin(), visit_totals.begin() + root_n);
                chosen = dist(rng);
            }
            // n-step TD inputs (mirrors az_selfplay.finalize_searched_sample):
            // the search's Q of the action actually PLAYED (the visit-weighted
            // root_value folds in noise-forced exploratory visits the played
            // line never follows), and whether that action leaves the
            // search's own line.
            s.q = visit_totals[static_cast<size_t>(chosen)] > 0
                      ? static_cast<float>(
                            w_totals[static_cast<size_t>(chosen)] /
                            static_cast<double>(
                                visit_totals[static_cast<size_t>(chosen)]))
                      : 0.0f;
            s.explored = chosen != best;
            game_samples.push_back(std::move(s));
        }

        phase = IDLE;
        pool.clear();
        pending.clear();
        path.clear();
        return chosen;
    }

    // Rich divergence diagnostic (release-safe): the action index we are about to
    // hand the engine falls outside its live menu, so the caller's
    // check_machine_choice is about to fatal. Print the full search context first
    // — which phase produced the index, this search's root width, and the root vs
    // live menus decoded — so a rare self-play divergence (e.g. a miracle prompt
    // whose affordability shifted between the search root and the restored line)
    // leaves enough of a trail to root-cause without a live repro.
    void dump_divergence(const std::vector<LegalAction>& actions, int r) const {
        const char* phname = phase == IDLE          ? "IDLE"
                             : phase == DESCENDING   ? "DESCENDING"
                             : phase == ROLLOUT      ? "ROLLOUT"
                             : phase == PLAN_GEN     ? "PLAN_GEN"
                             : phase == PLAN_REPLAY  ? "PLAN_REPLAY"
                             : phase == PLAN_ROLLOUT ? "PLAN_ROLLOUT"
                                                     : "AWAITING_ROOT";
        std::string live;
        for (size_t i = 0; i < actions.size(); i++)
            live += "\n    [" + std::to_string(i) + "] " + actions[i].description;
        std::string root;
        for (size_t i = 0; i < root_menu_desc.size(); i++)
            root += "\n    [" + std::to_string(i) + "] " + root_menu_desc[i];
        std::fprintf(
            stderr,
            "az_mcts DIVERGENCE: about to return index %d into a %zu-action live menu "
            "(turn=%zu step=%s)\n"
            "  phase=%s this_root=%d root_n=%d cap_root=%ld "
            "world=%d/%d sim=%d/%d sims_run=%d sb_active=%d sideboard_phase=%d\n"
            "  live menu (%zu action(s)):%s\n"
            "  search-root menu (%zu action(s)):%s\n",
            r, actions.size(), cur_game.turn, step_to_string(cur_game.cur_step),
            phname, this_root, root_n, cap_root_counter, cur_world, cur_worlds, cur_sim,
            cur_sims, sims_run, sb_active ? 1 : 0, sideboard_phase ? 1 : 0,
            actions.size(), live.c_str(), root_menu_desc.size(), root.c_str());
        std::fflush(stderr);
    }

    int on_decision(const std::vector<LegalAction>& actions) {
        // populate_query (and therefore build_obs's num_choices, which is what
        // root_n records) clamps an over-wide legal menu to MAX_ACTIONS, so
        // every search roots on the clamped width and answers indices inside
        // it. Compare and capture against the SAME clamped width here — a raw
        // actions.size() would make the restore-contract tripwire below fire
        // spuriously on any decision whose menu exceeds MAX_ACTIONS (root_n is
        // clamped, the re-emitted actions.size() is not), and would also over-
        // read the MAX_ACTIONS-wide per-action obs blocks in the debug capture.
        const int menu_n = std::min(static_cast<int>(actions.size()), MAX_ACTIONS);
        int r;
        // AWAITING_ROOT only advances sim/world bookkeeping and never reads the
        // observation, so skip the full obs rebuild there — it fires once per
        // simulation, right after the cooperative restore.
        if (phase == AWAITING_ROOT) {
            // Restore contract tripwire: the loop-top re-derivation after a
            // snapshot restore must re-emit the SAME decision this search
            // rooted on — the action returned below is answered blindly, so a
            // different query here (state outside the snapshot leaking into
            // the re-derivation, e.g. the resolution_just_completed miracle
            // window) would be silently consumed and corrupt the whole line,
            // surfacing plies later as a world-consistency violation at best.
            // Fail at the source instead. The width compare is one integer
            // against the vector already in hand; the debug build additionally
            // compares the full per-action menu content against the root's.
            if (menu_n != root_n) {
                fatal_error("az_mcts: restored root re-derived a different decision: "
                            "search rooted on " + std::to_string(root_n) +
                            " choices, engine re-emitted " + std::to_string(menu_n) +
                            "\n  root#" + std::to_string(this_root) +
                            " world=" + std::to_string(cur_world) +
                            " sim=" + std::to_string(cur_sim) +
                            " sideboard=" + std::to_string(sideboard_phase ? 1 : 0));
            }
#ifndef NDEBUG
            {
                ActorObs ob = build_obs(actions);
                std::vector<float> want, got;
                capture_menu(root_obs.data(), root_n, want);
                capture_menu(ob.obs.data(), menu_n, got);
                if (want != got) {
                    std::vector<float> got_state;
                    capture_state(ob.obs.data(), got_state);
                    std::vector<float> want_state;
                    capture_state(root_obs.data(), want_state);
                    fatal_error("az_mcts: restored root re-derived a same-width but "
                                "different decision (root#" + std::to_string(this_root) +
                                " world=" + std::to_string(cur_world) +
                                " sim=" + std::to_string(cur_sim) + ")" +
                                "\n  root menu (at search begin):\n" + fmt_state(want_state) +
                                fmt_menu(want) +
                                "  engine menu (re-emitted):\n" + fmt_state(got_state) +
                                fmt_menu(got));
                }
            }
#endif
            r = advance_after_restore();
        } else {
            ActorObs ob = build_obs(actions);
            const float* o = ob.obs.data();
            int nc = ob.num_choices;
            switch (phase) {
                case IDLE: {
                    // A searchable IDLE decision opens a new search root; record its
                    // live menu descriptions for the divergence diagnostic (this is
                    // the ONLY place root_menu_desc is refreshed, so it always names
                    // the root of the search that produced the returned index).
                    bool searching = search_loop_safe() && nc > 1;
                    r = begin_or_fallback(o, nc);
                    if (searching) {
                        root_menu_desc.clear();
                        for (const auto& a : actions)
                            root_menu_desc.push_back(a.description);
                    }
                    break;
                }
                case DESCENDING:
                    r = descend_step(o, nc);
                    break;
                case ROLLOUT:
                    r = rollout_step(o, nc);
                    break;
                case PLAN_GEN:
                    r = plan_gen_step(o, nc);
                    break;
                case PLAN_REPLAY:
                    r = plan_replay_step(o, nc);
                    break;
                case PLAN_ROLLOUT:
                    r = plan_rollout_step(o, nc);
                    break;
                default:
                    fatal_error("az_mcts: unreachable phase");
                    r = 0;
                    break;
            }
        }
        // The returned index is fed straight to the engine's input; if it is out of
        // range the caller's check_machine_choice will fatal with only the menu.
        // Emit the search-side context first so the two together fully localize the
        // divergence. (No recovery here — a wrong index must still fail loudly.)
        if (r < 0 || r >= menu_n) dump_divergence(actions, r);
        return r;
    }

    bool on_game_end(int winner) {
        // A simulated line reached game over (fires with a live snapshot and no
        // pending restore). Mirror mcts.py's terminal convention: value ±1 vs
        // the ROOT seat, DRAW 0.
        double leaf_value;
        if (winner == static_cast<int>(Zone::PLAYER_A) ||
            winner == static_cast<int>(Zone::PLAYER_B)) {
            bool a_won = winner == static_cast<int>(Zone::PLAYER_A);
            leaf_value = (a_won == root_is_a) ? 1.0 : -1.0;
        } else {
            leaf_value = 0.0;
        }
        if (phase == PLAN_ROLLOUT) {
            // The plan's playout finished the sampled game: the terminal value
            // IS this world's plan value (already root-relative — mirrors
            // _rollout's terminal return through _eval_plan_world).
            finish_plan_world(leaf_value, /*from_memo=*/false);
            return true;
        }
        if (phase == PLAN_GEN || phase == PLAN_REPLAY) {
            // A game cannot legally end during the sideboard phase; a terminal
            // here means the forced line diverged from the recorded plan.
            fatal_error("az_mcts: plan-search world-consistency violation: "
                        "game ended during plan " +
                        std::string(phase == PLAN_GEN ? "generation" : "replay"));
        }
        backup(path, leaf_value, root_is_a);
        finish_sim();
        return true;
    }
};

AZMcts::AZMcts(const MCTSConfig& cfg, AZEvaluator* evaluator,
               AZEvaluator* evaluator_b)
    : impl_(std::make_unique<Impl>(cfg, evaluator, evaluator_b)) {}
AZMcts::~AZMcts() = default;

int AZMcts::on_decision(const std::vector<LegalAction>& actions) {
    return impl_->on_decision(actions);
}
bool AZMcts::on_game_end(int winner) { return impl_->on_game_end(winner); }
void AZMcts::set_scripted_provider(std::function<int(const float*, int)> fn) {
    impl_->scripted_provider = std::move(fn);
}
const std::vector<SearchRootResult>& AZMcts::results() const { return impl_->results; }
void AZMcts::begin_match(uint32_t cap_seed) { impl_->begin_match(cap_seed); }
void AZMcts::end_game() { impl_->end_game(); }
const std::vector<SelfPlaySample>& AZMcts::game_samples() const {
    return impl_->game_samples;
}
