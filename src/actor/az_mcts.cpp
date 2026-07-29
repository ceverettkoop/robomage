#include "az_mcts.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "az_evaluator.h"
#include "components/zone.h"  // Zone::PLAYER_A / PLAYER_B
#include "error.h"
#include "obs_builder.h"      // build_obs, ACTOR_OBS_SIZE, ACTOR_SELF_IS_A_IDX
#include "search_server.h"    // search_loop_safe, search_request_restore, determinize_hidden_state
#include "snapshot.h"         // snapshot_save, snapshot_release_all

// True while a between-game bo3 sideboard prompt is live (the same engine global
// the obs is_sideboard_phase flag serializes from). Selects the sideboard search
// budget at root setup — do NOT use match_game_number, which reflects the just-
// ended game at a g1->g2 sideboard root.
extern bool sideboard_phase;

// Observation float "self is Player A" (mirrors train/env.py::_SELF_IS_A_IDX).
// Determines which seat moves at a node.
static constexpr int SELF_IS_A_IDX = ACTOR_SELF_IS_A_IDX;

// The single snapshot slot the search reserves (matches mcts.py's snapshot_slot=0).
static constexpr int SEARCH_SLOT = 0;

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
    enum Phase { IDLE, DESCENDING, AWAITING_ROOT };

    MCTSConfig cfg;
    AZEvaluator* eval;  // null → uniform evaluator (torch-free)
    Phase phase = IDLE;
    int root_counter = 0;

    // Per-search state (valid IDLE→...→finalize).
    int this_root = 0;
    int root_n = 0;
    bool root_is_a = false;
    std::vector<double> root_priors;
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
    int cur_world = 0;
    int cur_sim = 0;  // index of the simulation currently running within cur_world
    uint32_t cur_world_seed = 0;
    Node* cur_root = nullptr;

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
        cur_world_seed = cfg.world_seed_base +
                         100003u * static_cast<uint32_t>(this_root) +
                         static_cast<uint32_t>(w);
        // Root Dirichlet noise, redrawn per world over the shared base priors —
        // mcts.py: priors = (1-eps)*root_priors + eps*dirichlet(alpha*[1]*nc).
        // eps=0 (parity default) reuses the base priors verbatim. A standard
        // Dirichlet is gamma(alpha) per component normalized by the sum, drawn
        // from the per-run RNG (no cross-language parity required).
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
            cur_root = make_node(root_n, priors, root_is_a);
        } else {
            cur_root = make_node(root_n, root_priors, root_is_a);
        }
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
            remove_virtual_loss(pl.path);
            bool is_a = pl.obs[SELF_IS_A_IDX] > 0.5f;
            if (pl.parent->children.find(pl.action) == pl.parent->children.end()) {
                pl.parent->children[pl.action] =
                    make_node(pl.num_choices, res[static_cast<size_t>(i)].priors, is_a);
#ifndef NDEBUG
                capture_menu(pl.obs.data(), pl.num_choices,
                             pl.parent->children[pl.action]->dbg_menu);
                capture_state(pl.obs.data(), pl.parent->children[pl.action]->dbg_state);
#endif
            }
            backup(pl.path, res[static_cast<size_t>(i)].value, is_a);
        }
        pending.clear();
    }

    // ── phase handlers ─────────────────────────────────────────────────────
    int begin_or_fallback(const float* o, int nc) {
        bool searchable = search_loop_safe() && nc > 1;
        if (!searchable) {
            // A trivial / unsafe real decision: not stored, evaluator-argmax
            // fallback. It still counts as one real move for the tau schedule.
            move_counter += 1;
            if (nc <= 1) return 0;
            std::vector<double> priors = eval_priors(o, nc);
            int best = 0;
            for (int i = 1; i < nc; i++)
                if (priors[static_cast<size_t>(i)] > priors[static_cast<size_t>(best)]) best = i;
            return best;
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
        cur_sims = (sb && cfg.sb_sims >= 0) ? cfg.sb_sims : cfg.sims;
        cur_worlds = (sb && cfg.sb_worlds >= 0) ? cfg.sb_worlds : cfg.worlds;
        cur_max_depth = (sb && cfg.sb_max_depth >= 0) ? cfg.sb_max_depth : cfg.max_depth;
        visit_totals.assign(static_cast<size_t>(nc), 0);
        value_acc = 0.0;
        sims_run = 0;
        sim_steps = 0;
        sims_per_world = std::max(1, cur_sims / std::max(1, cur_worlds));
        pool.clear();
        pending.clear();
        path.clear();
        cur_world = 0;
        cur_sim = 0;
        begin_world(0);
        // Sim 0 of world 0: the snapshot IS the current (clean) root state, so
        // determinize directly — no intervening restore (mcts.py restores before
        // every sim, but restore-immediately-after-snapshot is a no-op).
        determinize_hidden_state(cur_world_seed);
        int a = start_descent();
        phase = DESCENDING;
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
            // New leaf.
            bool leaf_is_a = o[SELF_IS_A_IDX] > 0.5f;
            if (cfg.batch > 1) {
                PendingLeaf pl;
                pl.obs.assign(o, o + ACTOR_OBS_SIZE);
                pl.num_choices = nc;
                pl.path = path;
                pl.parent = parent;
                pl.action = paction;
                apply_virtual_loss(path);
                pending.push_back(std::move(pl));
                finish_sim();
                return 0;
            }
            AZEvalResultD r = eval_one(o, nc);
            parent->children[paction] = make_node(nc, r.priors, leaf_is_a);
#ifndef NDEBUG
            capture_menu(o, nc, parent->children[paction]->dbg_menu);
            capture_state(o, parent->children[paction]->dbg_state);
#endif
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
        // We are back at the restored true root. The just-finished sim was
        // (cur_world, cur_sim); advance to the next sim / world / finalize.
        cur_sim += 1;
        if (cur_sim >= sims_per_world) {
            if (cfg.batch > 1) flush_pending();
            accumulate_world();
            cur_world += 1;
            if (cur_world >= cur_worlds) return finalize();
            begin_world(cur_world);
            cur_sim = 0;
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
        if (cfg.batch > 1) flush_pending();
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
            for (int i = 0; i < root_n; i++) {
                s.pi[static_cast<size_t>(i)] =
                    total > 0 ? static_cast<float>(
                                    static_cast<double>(visit_totals[static_cast<size_t>(i)]) /
                                    static_cast<double>(total))
                              : 0.0f;
                s.mask[static_cast<size_t>(i)] = 1;
            }
            game_samples.push_back(std::move(s));

            if (move_counter < cfg.temp_moves && total > 0) {
                std::discrete_distribution<int> dist(
                    visit_totals.begin(), visit_totals.begin() + root_n);
                chosen = dist(rng);
            }
            move_counter += 1;
        }

        phase = IDLE;
        pool.clear();
        pending.clear();
        path.clear();
        return chosen;
    }

    int on_decision(const std::vector<LegalAction>& actions) {
        // AWAITING_ROOT only advances sim/world bookkeeping and never reads the
        // observation, so skip the full obs rebuild there — it fires once per
        // simulation, right after the cooperative restore.
        if (phase == AWAITING_ROOT) return advance_after_restore();
        ActorObs ob = build_obs(actions);
        const float* o = ob.obs.data();
        int nc = ob.num_choices;
        switch (phase) {
            case IDLE:
                return begin_or_fallback(o, nc);
            case DESCENDING:
                return descend_step(o, nc);
            case AWAITING_ROOT:
                break;  // handled above
        }
        fatal_error("az_mcts: unreachable phase");
        return 0;
    }

    bool on_game_end(int winner) {
        // A simulated line reached game over (only fires during DESCENDING with a
        // live snapshot and no pending restore). Mirror mcts.py's terminal
        // convention: value ±1 vs the ROOT seat, DRAW 0, leaf_seat = root seat.
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
const std::vector<SearchRootResult>& AZMcts::results() const { return impl_->results; }
void AZMcts::begin_match() { impl_->begin_match(); }
void AZMcts::end_game() { impl_->end_game(); }
const std::vector<SelfPlaySample>& AZMcts::game_samples() const {
    return impl_->game_samples;
}
