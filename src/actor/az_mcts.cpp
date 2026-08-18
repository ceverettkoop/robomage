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
#include <string>
#include <unordered_map>
#include <vector>

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
    // Per-action (category, card id) captured from this node's obs at CREATION
    // (mirrors mcts.py _Node.pick_meta) — the descent has no obs in hand when
    // re-selecting from interior nodes (AWAITING_ROOT never builds one), so
    // the rollout memo's pick descriptors come from these. Filled only when a
    // persistent sideboard-boundary search is active; empty otherwise.
    std::vector<int16_t> menu_cat;
    std::vector<int16_t> menu_id;
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
    enum Phase { IDLE, DESCENDING, ROLLOUT, AWAITING_ROOT };

    MCTSConfig cfg;
    AZEvaluator* eval;  // null → uniform evaluator (torch-free)
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

    // ── sideboard-boundary persistence (mirrors mcts.py's boundary consumers:
    // sb_root_key / walk_reuse_root / rollout_memo_*) ──────────────────────
    std::vector<Node*> world_roots;    // this search's per-world roots
    std::vector<Node*> walked_roots;   // boundary walk results (nullptr = fresh)
    std::vector<int> cur_budgets;      // per-world NEW-sim budgets (top-up)
    long reused_visits = 0;            // inherited root visits at search entry
    bool memo_active = false;          // this search persists + memoizes
    bool sb_active = false;            // a boundary is latched
    bool sb_seat_is_a = false;         // boundary identity: seat...
    int sb_game = 0;                   // ...and upcoming game number
    int sb_seed_root = 0;              // first searched root's index (pins seeds)
    std::vector<Node*> sb_roots;       // previous searched root's world trees
    std::vector<int> sb_played;        // actions since the previous searched root
    // Real picks since the BOUNDARY root, as (cat, card id, seat) descriptors —
    // the memo key's base; sim_picks extends it with the sim path's picks.
    std::vector<std::array<int16_t, 3>> sb_picks;
    std::vector<std::array<int16_t, 3>> sim_picks;
    struct MemoKey {
        uint32_t seed;
        bool seat;
        std::vector<std::array<int16_t, 3>> picks;  // sorted
        bool operator<(const MemoKey& o) const {
            if (seed != o.seed) return seed < o.seed;
            if (seat != o.seat) return seat < o.seat;
            return picks < o.picks;  // lexicographic == Python sorted-tuple order
        }
    };
    struct MemoVal {
        bool terminal;   // true: `winner` holds the engine winner (0 draw)
        double value;    // horizon net value (mover perspective) when !terminal
        bool seat_is_a;  // the horizon state's mover when !terminal
        int winner;
    };
    std::map<MemoKey, MemoVal> rollout_memo;  // cleared at boundary end
    bool memo_pending = false;                // a missed rollout is in flight
    MemoKey memo_pending_key;
    int memo_hits = 0;

    std::vector<std::unique_ptr<Node>> pool;  // owns every node this search
    std::vector<PathEntry> path;              // current descent
    std::vector<PendingLeaf> pending;         // batched-leaf collection (batch>1)

    std::vector<SearchRootResult> results;

    // ── self-play state ─────────────────────────────────────────────────────
    std::mt19937 rng;                        // noise + real-action sampling RNG
    long move_counter = 0;                   // per-game real-decision counter
    std::vector<float> root_obs;             // clean root obs (captured before determinize)
    std::vector<SelfPlaySample> game_samples; // samples stored this game

    explicit Impl(const MCTSConfig& c, AZEvaluator* e)
        : cfg(c), eval(e), rng(c.selfplay_rng_seed) {}

    void begin_match() {
        move_counter = 0;
        game_samples.clear();
    }

    void end_game() {
        // Called after a game's samples are priced+flushed. Any sideboard samples
        // recorded before the next game starts accumulate into the now-empty
        // buffer and get priced by the next game's z; the tau counter restarts at
        // the game boundary (before the sideboard prompts), matching Python.
        move_counter = 0;
        game_samples.clear();
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

    // ── world / sim lifecycle ──────────────────────────────────────────────
    void begin_world(int w) {
        // Under an active boundary the seeds are pinned to the boundary's
        // FIRST searched root (mcts.py: the caller passes the latched
        // world_seeds back in) — at a sideboard root the seed IS the sampled
        // next-game deal, so a re-rooted tree is only coherent under the seed
        // it was built with.
        cur_world_seed =
            cfg.world_seed_base +
            100003u * static_cast<uint32_t>(sb_active ? sb_seed_root : this_root) +
            static_cast<uint32_t>(w);
        // Root Dirichlet noise, redrawn per world over the shared base priors —
        // mcts.py: priors = (1-eps)*root_priors + eps*dirichlet(alpha*[1]*nc).
        // eps=0 (parity default) reuses the base priors verbatim. A standard
        // Dirichlet is gamma(alpha) per component normalized by the sum, drawn
        // from the per-run RNG (no cross-language parity required). A REUSED
        // root (boundary walk survivor) keeps its N/W/children but its P is
        // overwritten with the fresh clean/noised priors, exactly like
        // run_search's reuse_roots adoption.
        Node* reused = (w < static_cast<int>(walked_roots.size()))
                           ? walked_roots[static_cast<size_t>(w)]
                           : nullptr;
        if (cfg.noise_eps > 0.0) {
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
            if (reused != nullptr) {
                reused->P = std::move(priors);
                cur_root = reused;
            } else {
                cur_root = make_node(root_n, priors, root_is_a);
            }
        } else if (reused != nullptr) {
            reused->P = root_priors;
            cur_root = reused;
        } else {
            cur_root = make_node(root_n, root_priors, root_is_a);
        }
        // Every branch above just assigned raw (clean or noise-mixed) priors
        // to cur_root->P, so the merge fold applies exactly once — including
        // re-adopted boundary roots, whose partition is refilled from THIS
        // search's root obs (same menu by world consistency).
        init_merge(cur_root, root_obs.data());
        if (memo_active && cur_root->menu_cat.empty())
            fill_pick_meta(cur_root, root_obs.data(), root_n);
        world_roots[static_cast<size_t>(w)] = cur_root;
#ifndef NDEBUG
        capture_menu(root_obs.data(), root_n, cur_root->dbg_menu);
        capture_state(root_obs.data(), cur_root->dbg_state);
#endif
    }

    int start_descent() {
        path.clear();
        if (memo_active) sim_picks = sb_picks;
        int a = cur_root->select(cfg.c_puct);
        if (memo_active) note_pick(cur_root, a, sim_picks);
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
        for (int i = 0; i < root_n; i++)
            visit_totals[static_cast<size_t>(i)] += cur_root->N[static_cast<size_t>(i)];
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
                fill_pick_meta(pl.parent->children[pl.action], pl.obs.data(),
                               pl.num_choices);
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

    // ── sideboard-boundary helpers (mirror mcts.py's shared helpers) ───────
    static long nsum(const Node* n) {
        long s = 0;
        for (int64_t v : n->N) s += v;
        return s;
    }

    // mcts.py::pick_meta_for — (cat, id) per action from the obs blocks,
    // captured at node creation. Only meaningful when memo_active.
    void fill_pick_meta(Node* n, const float* o, int nc) {
        if (!memo_active) return;
        n->menu_cat.resize(static_cast<size_t>(nc));
        n->menu_id.resize(static_cast<size_t>(nc));
        for (int i = 0; i < nc; i++) {
            n->menu_cat[static_cast<size_t>(i)] = static_cast<int16_t>(
                std::lround(static_cast<double>(o[STATE_SIZE + 0 * MAX_ACTIONS + i]) *
                            ACTION_CATEGORY_MAX));
            n->menu_id[static_cast<size_t>(i)] = static_cast<int16_t>(
                std::lround(static_cast<double>(o[STATE_SIZE + 1 * MAX_ACTIONS + i]) *
                            N_CARD_TYPES));
        }
    }

    // mcts.py::sb_pick_descriptor for the edge (n, a): appended to `out` when
    // the action is a sideboard pick (IN/OUT; Done and in-game actions carry
    // no descriptor).
    static void note_pick(const Node* n, int a,
                          std::vector<std::array<int16_t, 3>>& out) {
        if (n->menu_cat.empty()) return;
        int16_t cat = n->menu_cat[static_cast<size_t>(a)];
        if (cat != static_cast<int16_t>(ActionCategory::SIDEBOARD_IN) &&
            cat != static_cast<int16_t>(ActionCategory::SIDEBOARD_OUT))
            return;
        out.push_back({cat, n->menu_id[static_cast<size_t>(a)],
                       static_cast<int16_t>(n->self_is_a ? 1 : 0)});
    }

    // The same descriptor read straight from an obs (for latching REAL picks:
    // the finalize-chosen action and fallback/forced actions).
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

    // mcts.py::rollout_memo_eligible — a takeback ((card, seat) in BOTH
    // directions) diverges the one-shot direction locks, so such paths are
    // never memoized.
    static bool memo_eligible(const std::vector<std::array<int16_t, 3>>& picks) {
        for (size_t i = 0; i < picks.size(); i++)
            for (size_t j = 0; j < picks.size(); j++)
                if (picks[i][0] != picks[j][0] && picks[i][1] == picks[j][1] &&
                    picks[i][2] == picks[j][2])
                    return false;
        return true;
    }

    MemoKey make_memo_key(bool leaf_seat_is_a) const {
        MemoKey k;
        k.seed = cur_world_seed;
        k.seat = leaf_seat_is_a;
        k.picks = sim_picks;
        std::sort(k.picks.begin(), k.picks.end());
        return k;
    }

    // mcts.py::walk_reuse_root — child-by-played-action; the survivor must
    // match the new root's menu size and seat.
    static Node* walk_node(Node* root, const std::vector<int>& actions,
                           int num_choices, bool seat_is_a) {
        Node* n = root;
        for (int a : actions) {
            // Real played actions may be merged-away duplicates; children are
            // keyed by representative indices only.
            if (!n->rep.empty() && a >= 0 && a < n->num_choices)
                a = n->rep[static_cast<size_t>(a)];
            auto it = n->children.find(a);
            if (it == n->children.end()) return nullptr;
            n = it->second;
        }
        if (n->num_choices != num_choices || n->self_is_a != seat_is_a)
            return nullptr;
        return n;
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
            if (memo_pending) {
                rollout_memo[memo_pending_key] = {false, r.value, seat, 0};
                memo_pending = false;
            }
            backup(path, r.value, seat);
            finish_sim();
            return 0;
        }
        sim_steps += 1;
        return argmax_priors(r.priors, nc);
    }

    // ── phase handlers ─────────────────────────────────────────────────────
    int begin_or_fallback(const float* o, int nc) {
        // Vs-scripted seat (mirrors _play_match's non-net_to_move branch): the
        // scripted seat's real decisions come from the oracle provider — no
        // search, no sample, no searched/fallback counters — but they DO
        // advance the per-game tau counter (Python's game_move counts every
        // decision, both seats) and latch into a live sideboard boundary (the
        // walk must replay the true action sequence, whoever played it).
        // Search simulations never reach here (DESCENDING/ROLLOUT phases), so
        // tree play stays net-both-seats exactly like the Python reference.
        if (cfg.scripted_seat != 0 &&
            (o[SELF_IS_A_IDX] > 0.5f) == (cfg.scripted_seat == 1)) {
            move_counter += 1;
            int a = 0;
            if (nc > 1) {
                if (!scripted_provider)
                    fatal_error("az_mcts: scripted_seat is set but no scripted "
                                "provider was installed (set_scripted_provider)");
                a = scripted_provider(o, nc);
            }
            if (sb_active) {
                sb_played.push_back(a);
                note_pick_from_obs(o, a, sb_picks);
            }
            return a;
        }
        bool searchable = search_loop_safe() && nc > 1;
        if (!searchable) {
            // A trivial / unsafe real decision: not stored, evaluator-argmax
            // fallback. It still counts as one real move for the tau schedule.
            move_counter += 1;
            int a = 0;
            if (nc > 1) a = argmax_priors(eval_priors(o, nc), nc);
            // A forced/fallback action inside an active boundary (direction
            // locks can prune a sideboard menu to one entry) still advances
            // the real line — latch it so the next search's walk stays exact
            // (mirrors the Python consumers latching every stepped action).
            if (sb_active) {
                sb_played.push_back(a);
                note_pick_from_obs(o, a, sb_picks);
            }
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
        // Select this root's budget: the sideboard budget when the root is a
        // sideboard prompt (and the sb_* config is set), else the in-game budget.
        bool sb = sideboard_phase;
        root_is_sb = sb;
        cur_sims = (sb && cfg.sb_sims >= 0) ? cfg.sb_sims : cfg.sims;
        cur_worlds = (sb && cfg.sb_worlds >= 0) ? cfg.sb_worlds : cfg.worlds;
        cur_max_depth = (sb && cfg.sb_max_depth >= 0) ? cfg.sb_max_depth : cfg.max_depth;
        // Leaf-rollout budget + anchor (mirrors run_search's rollout_turns and
        // _rollout_anchor: 0 at a sideboard root — the horizon is "through
        // player-turn N of the sampled NEXT game" — else the root's own turn).
        cur_rollout_turns =
            (sb && cfg.sb_rollout_turns >= 0) ? cfg.sb_rollout_turns : cfg.rollout_turns;
        cur_rollout_anchor = sb ? 0 : static_cast<int>(cur_game.turn);
        cur_rollout_cap = kRolloutStepsPerTurn * cur_rollout_turns;
        visit_totals.assign(static_cast<size_t>(nc), 0);
        value_acc = 0.0;
        sims_run = 0;
        sim_steps = 0;
        memo_hits = 0;
        memo_pending = false;
        reused_visits = 0;
        sims_per_world = std::max(1, cur_sims / std::max(1, cur_worlds));

        // Sideboard-boundary continue / start / end (mirrors mcts.py's
        // sb_root_key identity — seat + upcoming game number — plus the walk
        // of every action played since the previous searched root). On
        // continue the node pool SURVIVES (the walked subtrees live in it);
        // everywhere else it is cleared as before.
        bool cont = false;
        walked_roots.assign(static_cast<size_t>(cur_worlds), nullptr);
        if (cfg.sb_tree_persist && sb) {
            memo_active = true;
            int game = match_game_number + 1;
            if (sb_active && root_is_a == sb_seat_is_a && game == sb_game &&
                static_cast<int>(sb_roots.size()) == cur_worlds) {
                cont = true;
                for (int w = 0; w < cur_worlds; w++) {
                    Node* n = walk_node(sb_roots[static_cast<size_t>(w)],
                                        sb_played, nc, root_is_a);
                    walked_roots[static_cast<size_t>(w)] = n;
                    if (n != nullptr) reused_visits += nsum(n);
                }
            } else {
                pool.clear();
                sb_roots.clear();
                rollout_memo.clear();
                sb_picks.clear();
                sb_active = true;
                sb_seat_is_a = root_is_a;
                sb_game = game;
                sb_seed_root = this_root;
            }
        } else {
            pool.clear();
            sb_roots.clear();
            rollout_memo.clear();
            sb_picks.clear();
            sb_active = false;
            memo_active = false;
        }
        pending.clear();
        path.clear();
        world_roots.assign(static_cast<size_t>(cur_worlds), nullptr);
        cur_budgets.assign(static_cast<size_t>(cur_worlds), sims_per_world);
        if (cont) {
            for (int w = 0; w < cur_worlds; w++) {
                Node* n = walked_roots[static_cast<size_t>(w)];
                if (n != nullptr)
                    cur_budgets[static_cast<size_t>(w)] = std::max(
                        0, sims_per_world - static_cast<int>(nsum(n)));
            }
        }
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
            fill_pick_meta(parent->children[paction], o, nc);
#ifndef NDEBUG
            capture_menu(o, nc, parent->children[paction]->dbg_menu);
            capture_state(o, parent->children[paction]->dbg_state);
#endif
            // Leaf rollout (mirrors mcts.py::_rollout with the leaf as state 0):
            // unless the leaf already meets the horizon, play the raw policy
            // forward instead of backing up the leaf's value. The engine state
            // at this moment IS the leaf state, so check the stop condition on
            // the live globals (identical to Python's read of the leaf obs).
            if (cur_rollout_turns > 0) {
                // Rollout memo (mirrors _simulate's lookup-before-rollout):
                // eligible when the leaf is still inside the sideboard phase
                // and the accumulated pick multiset contains no takeback.
                memo_pending = false;
                if (memo_active && sideboard_phase && memo_eligible(sim_picks)) {
                    MemoKey key = make_memo_key(leaf_is_a);
                    auto hit = rollout_memo.find(key);
                    if (hit != rollout_memo.end()) {
                        memo_hits += 1;
                        const MemoVal& v = hit->second;
                        if (v.terminal) {
                            double lv = 0.0;
                            if (v.winner == static_cast<int>(Zone::PLAYER_A) ||
                                v.winner == static_cast<int>(Zone::PLAYER_B)) {
                                bool a_won =
                                    v.winner == static_cast<int>(Zone::PLAYER_A);
                                lv = (a_won == root_is_a) ? 1.0 : -1.0;
                            }
                            backup(path, lv, root_is_a);
                        } else {
                            backup(path, v.value, v.seat_is_a);
                        }
                        finish_sim();
                        return 0;
                    }
                    memo_pending = true;
                    memo_pending_key = std::move(key);
                }
                if (!rollout_stop_here()) {
                    rollout_steps = 0;
                    phase = ROLLOUT;
                    sim_steps += 1;
                    return argmax_priors(r.priors, nc);
                }
                // Leaf already at the horizon: its own value IS the rollout
                // result (mcts.py stores this zero-step case too).
                if (memo_pending) {
                    rollout_memo[memo_pending_key] = {false, r.value, leaf_is_a, 0};
                    memo_pending = false;
                }
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
                        " move=" + std::to_string(move_counter) +
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
        if (memo_active) note_pick(child, a, sim_picks);
        path.push_back({child, a});
        sim_steps += 1;
        return a;
    }

    int advance_after_restore() {
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
        sr.reused_visits = reused_visits;
        sr.memo_hits = memo_hits;
        results.push_back(std::move(sr));

        int best = 0;
        for (int i = 1; i < root_n; i++)
            if (visit_totals[static_cast<size_t>(i)] > visit_totals[static_cast<size_t>(best)])
                best = i;

        // Self-play: store the searched sample and pick the real action per the
        // tau schedule (sample-from-visits for the first temp_moves real moves,
        // argmax after). Parity/eval mode stores nothing and always plays argmax.
        int chosen = best;
        if (cfg.selfplay) {
            SelfPlaySample s;
            s.obs = root_obs;                                    // clean root obs
            s.pi.assign(static_cast<size_t>(MAX_ACTIONS), 0.0f);
            s.mask.assign(static_cast<size_t>(MAX_ACTIONS), 0);
            s.mover_is_a = root_is_a;
            // n-step TD inputs (mirrors az_selfplay.py): this root's value, and
            // whether the action actually played leaves the search's own line.
            s.q = static_cast<float>(results.back().root_value);
            s.is_sideboard = root_is_sb;
            for (int i = 0; i < root_n; i++) {
                s.pi[static_cast<size_t>(i)] =
                    total > 0 ? static_cast<float>(
                                    static_cast<double>(visit_totals[static_cast<size_t>(i)]) /
                                    static_cast<double>(total))
                              : 0.0f;
                s.mask[static_cast<size_t>(i)] = 1;
            }

            if (move_counter < cfg.temp_moves && total > 0) {
                std::discrete_distribution<int> dist(
                    visit_totals.begin(), visit_totals.begin() + root_n);
                chosen = dist(rng);
            }
            s.explored = chosen != best;
            game_samples.push_back(std::move(s));
            move_counter += 1;
        }

        phase = IDLE;
        if (sb_active) {
            // Boundary continues past this real pick: keep the pool (the next
            // search re-roots into it) and latch the chosen action for the
            // walk + its descriptor for the memo key base (mirrors the Python
            // consumers latching each stepped action at choose time).
            sb_roots = world_roots;
            sb_played.clear();
            sb_played.push_back(chosen);
            note_pick_from_obs(root_obs.data(), chosen, sb_picks);
        } else {
            pool.clear();
        }
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
            "  phase=%s this_root=%d root_n=%d move_counter=%ld "
            "world=%d/%d sim=%d/%d sims_run=%d sb_active=%d sideboard_phase=%d\n"
            "  live menu (%zu action(s)):%s\n"
            "  search-root menu (%zu action(s)):%s\n",
            r, actions.size(), cur_game.turn, step_to_string(cur_game.cur_step),
            phname, this_root, root_n, move_counter, cur_world, cur_worlds, cur_sim,
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
        // A simulated line reached game over (fires during DESCENDING or ROLLOUT
        // with a live snapshot and no pending restore). Mirror mcts.py's terminal
        // convention: value ±1 vs the ROOT seat, DRAW 0, leaf_seat = root seat —
        // identical for a descent terminal and a rollout terminal. A rollout
        // terminal with a memo entry pending stores the raw winner (converted
        // vs the HITTING search's root seat at lookup, matching Python's
        // ("t", winner) tag); descent terminals are never memoized.
        if (phase == ROLLOUT && memo_pending) {
            rollout_memo[memo_pending_key] = {true, 0.0, false, winner};
            memo_pending = false;
        }
        double leaf_value;
        if (winner == static_cast<int>(Zone::PLAYER_A) ||
            winner == static_cast<int>(Zone::PLAYER_B)) {
            bool a_won = winner == static_cast<int>(Zone::PLAYER_A);
            leaf_value = (a_won == root_is_a) ? 1.0 : -1.0;
        } else {
            leaf_value = 0.0;
        }
        backup(path, leaf_value, root_is_a);
        finish_sim();
        return true;
    }
};

AZMcts::AZMcts(const MCTSConfig& cfg, AZEvaluator* evaluator)
    : impl_(std::make_unique<Impl>(cfg, evaluator)) {}
AZMcts::~AZMcts() = default;

int AZMcts::on_decision(const std::vector<LegalAction>& actions) {
    return impl_->on_decision(actions);
}
bool AZMcts::on_game_end(int winner) { return impl_->on_game_end(winner); }
void AZMcts::set_scripted_provider(std::function<int(const float*, int)> fn) {
    impl_->scripted_provider = std::move(fn);
}
const std::vector<SearchRootResult>& AZMcts::results() const { return impl_->results; }
void AZMcts::begin_match() { impl_->begin_match(); }
void AZMcts::end_game() { impl_->end_game(); }
const std::vector<SelfPlaySample>& AZMcts::game_samples() const {
    return impl_->game_samples;
}
